"""Command-line entry point for repeatable batch analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import TerminalConfig
from .pipeline import run_analysis
from .reporting import export_analysis_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Multi-Asset Risk Terminal.")
    parser.add_argument("--config", type=Path, default=Path("config/default_config.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--refresh", action="store_true", help="Ignore cached source data.")
    parser.add_argument(
        "--fred-api-key", default=None, help="Prefer FRED_API_KEY environment variable."
    )
    parser.add_argument("--skip-quantstats", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    config = TerminalConfig.from_json(args.config)
    results = run_analysis(config, refresh=args.refresh, fred_api_key=args.fred_api_key)
    paths = export_analysis_artifacts(
        results,
        output_dir=args.output_dir,
        include_quantstats=not args.skip_quantstats,
    )
    print(f"Analysis complete. Tear sheet: {paths['tear_sheet'].resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
