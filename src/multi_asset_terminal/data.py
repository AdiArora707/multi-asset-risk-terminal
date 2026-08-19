"""Resilient market and macroeconomic data collection utilities."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import TerminalConfig

LOGGER = logging.getLogger(__name__)
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class DataDownloadError(RuntimeError):
    """Raised when a remote data source cannot provide usable observations."""


def _cache_path(cache_dir: str | Path, namespace: str, payload: dict[str, object]) -> Path:
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{namespace}_{key}.csv"


def _read_cached_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    return frame.sort_index()


def _normalise_yfinance_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract adjusted close data across current yfinance column layouts."""

    if raw.empty:
        raise DataDownloadError("Yahoo Finance returned an empty response.")

    if isinstance(raw.columns, pd.MultiIndex):
        level_zero = raw.columns.get_level_values(0)
        level_one = raw.columns.get_level_values(1)
        if "Close" in level_zero:
            close = raw.xs("Close", axis=1, level=0, drop_level=True)
        elif "Adj Close" in level_zero:
            close = raw.xs("Adj Close", axis=1, level=0, drop_level=True)
        elif "Close" in level_one:
            close = raw.xs("Close", axis=1, level=1, drop_level=True)
        elif "Adj Close" in level_one:
            close = raw.xs("Adj Close", axis=1, level=1, drop_level=True)
        else:
            raise DataDownloadError("Yahoo Finance response contains no close-price field.")
    else:
        field = "Close" if "Close" in raw.columns else "Adj Close"
        if field not in raw.columns:
            raise DataDownloadError("Yahoo Finance response contains no close-price field.")
        close = raw[[field]].copy()
        close.columns = [tickers[0]]

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close.columns = [str(column).upper() for column in close.columns]
    close = close.apply(pd.to_numeric, errors="coerce")
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    close = close.loc[~close.index.duplicated(keep="last")].sort_index().dropna(how="all")
    return close


def download_adjusted_prices(config: TerminalConfig, refresh: bool = False) -> pd.DataFrame:
    """Download split- and distribution-adjusted ETF prices from Yahoo Finance.

    The response is cached as CSV for reproducibility and to reduce pressure on
    an unofficial endpoint. ``auto_adjust=True`` makes yfinance's ``Close``
    field the total-return-compatible adjusted close series.
    """

    payload = {"tickers": config.all_tickers, "start": config.start, "end": config.end}
    cache_file = _cache_path(config.cache_dir, "yahoo_prices", payload)
    if cache_file.exists() and not refresh:
        LOGGER.info("Loading market prices from %s", cache_file)
        return _read_cached_frame(cache_file)

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise DataDownloadError("yfinance is required to download ETF prices.") from exc

    # yfinance treats `end` as exclusive. Add one day so the configured date is inclusive.
    end = None
    if config.end:
        end = (pd.Timestamp(config.end) + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        raw = yf.download(
            tickers=config.all_tickers,
            start=config.start,
            end=end,
            auto_adjust=True,
            actions=False,
            progress=False,
            group_by="column",
            threads=True,
            timeout=30,
        )
    except Exception as exc:  # yfinance exposes several transport exception types
        raise DataDownloadError(f"Yahoo Finance download failed: {exc}") from exc

    prices = _normalise_yfinance_close(raw, config.all_tickers)
    missing = [ticker for ticker in config.all_tickers if ticker not in prices.columns]
    if missing:
        raise DataDownloadError(f"No usable adjusted prices for: {', '.join(missing)}")
    prices = prices.loc[:, config.all_tickers]
    if len(prices) < config.periods_per_year:
        raise DataDownloadError(
            f"Only {len(prices)} price rows were returned; at least "
            f"{config.periods_per_year} are required."
        )
    prices.to_csv(cache_file, index_label="Date")
    return prices


def align_prices(
    prices: pd.DataFrame,
    required_columns: Iterable[str],
    mode: str = "intersection",
) -> pd.DataFrame:
    """Clean prices and apply an explicit common-calendar policy.

    No forward-filling is performed: synthesizing unchanged prices on missing
    trading days would create false zero returns and understate volatility.
    """

    columns = list(required_columns)
    missing = [column for column in columns if column not in prices.columns]
    if missing:
        raise ValueError(f"Required price columns are absent: {missing}")
    clean = prices.loc[:, columns].copy()
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()
    clean = clean.where(clean > 0)
    clean = clean.dropna(how="any" if mode == "intersection" else "all")
    if clean.empty:
        raise ValueError("Calendar alignment removed every observation.")
    return clean


def _requests_session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "multi-asset-risk-terminal/1.0"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_fred_series(
    series_id: str,
    start: str,
    end: str | None = None,
    api_key: str | None = None,
    cache_dir: str | Path = "data/cache",
    refresh: bool = False,
) -> pd.Series:
    """Download one FRED series through the API or official CSV fallback."""

    series_id = series_id.upper().strip()
    payload = {"series_id": series_id, "start": start, "end": end}
    cache_file = _cache_path(cache_dir, f"fred_{series_id.lower()}", payload)
    if cache_file.exists() and not refresh:
        cached = _read_cached_frame(cache_file)
        return cached.iloc[:, 0].rename(series_id)

    api_key = api_key.strip() if api_key else None
    if api_key and not re.fullmatch(r"[a-z0-9]{32}", api_key):
        raise DataDownloadError(
            "FRED_API_KEY must be exactly 32 lowercase alphanumeric characters."
        )

    session = _requests_session()
    series: pd.Series | None = None
    failures: list[str] = []
    try:
        if api_key:
            try:
                params = {
                    "series_id": series_id,
                    "api_key": api_key,
                    "file_type": "json",
                    "observation_start": start,
                }
                if end:
                    params["observation_end"] = end
                response = session.get(FRED_OBSERVATIONS_URL, params=params, timeout=(10, 30))
                response.raise_for_status()
                observations = response.json().get("observations", [])
                frame = pd.DataFrame(observations)
                if frame.empty or not {"date", "value"}.issubset(frame.columns):
                    raise DataDownloadError(
                        f"Authenticated FRED API returned no observations for {series_id}."
                    )
                series = pd.Series(
                    pd.to_numeric(frame["value"].replace(".", np.nan), errors="coerce").to_numpy(),
                    index=pd.to_datetime(frame["date"]),
                    name=series_id,
                )
            except (requests.RequestException, ValueError, KeyError, DataDownloadError) as exc:
                # Never include the raw exception: requests may embed the secret query parameter.
                failures.append(f"authenticated API ({type(exc).__name__})")

        if series is None:
            # The public graph CSV is a secondary path. The analytics pipeline has
            # a clearly disclosed constant-rate fallback if both FRED routes fail.
            params = {"id": series_id, "cosd": start}
            if end:
                params["coed"] = end
            try:
                response = session.get(FRED_CSV_URL, params=params, timeout=(10, 20))
                response.raise_for_status()
                frame = pd.read_csv(io.StringIO(response.text))
                if len(frame.columns) < 2:
                    raise DataDownloadError(f"FRED CSV response is malformed for {series_id}.")
                series = pd.Series(
                    pd.to_numeric(
                        frame.iloc[:, 1].replace(".", np.nan), errors="coerce"
                    ).to_numpy(),
                    index=pd.to_datetime(frame.iloc[:, 0]),
                    name=series_id,
                )
            except (requests.RequestException, ValueError, KeyError, DataDownloadError) as exc:
                failures.append(f"public CSV ({type(exc).__name__})")
    finally:
        session.close()

    if series is None:
        attempted = ", ".join(failures) or "no data route"
        raise DataDownloadError(f"FRED download failed for {series_id}; attempted {attempted}.")
    series = series[~series.index.duplicated(keep="last")].sort_index().dropna()
    if series.empty:
        raise DataDownloadError(f"FRED returned no numeric observations for {series_id}.")
    series.to_frame().to_csv(cache_file, index_label="Date")
    return series


def align_macro_features(
    index: pd.DatetimeIndex,
    annual_yield_percent: pd.Series,
    cpi_level: pd.Series,
    periods_per_year: int = 252,
    fallback_annual_rate: float = 0.02,
    inflation_release_lag_days: int = 15,
) -> pd.DataFrame:
    """Create daily risk-free and publication-lagged trailing inflation features."""

    index = pd.DatetimeIndex(index).tz_localize(None).sort_values()

    def asof_align(series: pd.Series) -> pd.Series:
        series = series.copy()
        series.index = pd.DatetimeIndex(series.index).tz_localize(None)
        expanded = series.reindex(series.index.union(index)).sort_index().ffill()
        return expanded.reindex(index)

    annual_rate = asof_align(annual_yield_percent).div(100.0)
    annual_rate = annual_rate.fillna(float(fallback_annual_rate)).clip(lower=-0.99)
    daily_rate = np.expm1(np.log1p(annual_rate) / periods_per_year)

    cpi = cpi_level.sort_index().astype(float)
    inflation_yoy = cpi.pct_change(12, fill_method=None)
    # FRED dates monthly CPI to the observation month, before that month's value
    # was knowable. Shift to mid-next-month as a conservative publication proxy.
    inflation_yoy.index = (
        inflation_yoy.index + pd.offsets.MonthEnd(1) + pd.Timedelta(days=inflation_release_lag_days)
    )
    inflation_daily = asof_align(inflation_yoy)
    return pd.DataFrame(
        {
            "annual_risk_free_rate": annual_rate,
            "daily_risk_free_rate": daily_rate,
            "inflation_yoy": inflation_daily,
        },
        index=index,
    )
