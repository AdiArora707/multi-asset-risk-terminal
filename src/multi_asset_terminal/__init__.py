"""Multi-Asset Performance and Risk Terminal public interface."""

from .config import TerminalConfig
from .pipeline import AnalysisResults, run_analysis
from .reporting import export_analysis_artifacts

__all__ = [
    "AnalysisResults",
    "TerminalConfig",
    "export_analysis_artifacts",
    "run_analysis",
]

__version__ = "1.0.0"
