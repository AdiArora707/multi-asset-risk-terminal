"""Configuration objects and validation for the analytics terminal."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

DEFAULT_ASSETS = {
    "US Equity": "VTI",
    "International Equity": "VXUS",
    "US Bonds": "BND",
    "Gold": "GLD",
    "Real Estate": "VNQ",
    "Cash ETF": "BIL",
}

DEFAULT_WEIGHTS = {
    "US Equity": 0.30,
    "International Equity": 0.15,
    "US Bonds": 0.25,
    "Gold": 0.10,
    "Real Estate": 0.10,
    "Cash ETF": 0.10,
}


@dataclass(frozen=True)
class TerminalConfig:
    """Validated run configuration.

    Weights intentionally do not have to sum to one. The difference between
    one and the net risky-asset weight is treated as a cash (or borrowing)
    position accruing at the FRED risk-free rate. This lets the engine diagnose
    both unlevered and leveraged portfolios without hiding financing effects.
    """

    assets: Mapping[str, str] = field(default_factory=lambda: DEFAULT_ASSETS.copy())
    weights: Mapping[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    benchmark: str = "SPY"
    start: str = "2015-01-01"
    end: str | None = None
    rebalance_frequency: str | None = "M"
    calendar_mode: str = "intersection"
    risk_free_series: str = "DGS3MO"
    inflation_series: str = "CPIAUCSL"
    inflation_release_lag_days: int = 15
    periods_per_year: int = 252
    rolling_window: int = 252
    var_confidence: float = 0.95
    fallback_annual_risk_free_rate: float = 0.02
    cache_dir: str = "data/cache"
    output_dir: str = "outputs"

    def __post_init__(self) -> None:
        assets = {str(k): str(v).upper().strip() for k, v in self.assets.items()}
        weights = {str(k): float(v) for k, v in self.weights.items()}
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "benchmark", self.benchmark.upper().strip())
        self.validate()

    def validate(self) -> None:
        """Raise a clear error for configurations that would corrupt results."""

        if not self.assets:
            raise ValueError("At least one asset must be configured.")
        if set(self.assets) != set(self.weights):
            mismatch = set(self.assets).symmetric_difference(self.weights)
            raise ValueError(f"Assets and weights must use identical labels; mismatch: {mismatch}")
        if any(not ticker for ticker in self.assets.values()):
            raise ValueError("Asset tickers cannot be empty.")
        if sum(abs(weight) for weight in self.weights.values()) == 0:
            raise ValueError("At least one portfolio weight must be non-zero.")
        if self.calendar_mode not in {"intersection", "union"}:
            raise ValueError("calendar_mode must be 'intersection' or 'union'.")
        if self.periods_per_year <= 1 or self.rolling_window <= 1:
            raise ValueError("Annualization and rolling-window values must exceed one.")
        if self.inflation_release_lag_days < 0:
            raise ValueError("inflation_release_lag_days cannot be negative.")
        if not 0.50 < self.var_confidence < 1.0:
            raise ValueError("var_confidence must be between 0.50 and 1.0.")
        start = date.fromisoformat(self.start)
        if self.end is not None and date.fromisoformat(self.end) <= start:
            raise ValueError("end must be later than start.")

    @property
    def all_tickers(self) -> list[str]:
        """Unique asset and benchmark tickers, retaining configured order."""

        return list(dict.fromkeys([*self.assets.values(), self.benchmark]))

    @property
    def ticker_to_label(self) -> dict[str, str]:
        mapping = {ticker: label for label, ticker in self.assets.items()}
        mapping[self.benchmark] = "Benchmark"
        return mapping

    @property
    def gross_exposure(self) -> float:
        return float(sum(abs(weight) for weight in self.weights.values()))

    @property
    def net_exposure(self) -> float:
        return float(sum(self.weights.values()))

    @classmethod
    def from_json(cls, path: str | Path) -> TerminalConfig:
        with Path(path).open("r", encoding="utf-8") as file:
            payload: dict[str, Any] = json.load(file)
        return cls(**payload)

    def to_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as file:
            json.dump(asdict(self), file, indent=2)
        return destination
