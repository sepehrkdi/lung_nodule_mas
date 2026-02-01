"""
API Module for Lung Nodule Multi-Agent System
==============================================

Provides FastAPI REST endpoints for:
- Nodule data retrieval
- Multi-agent analysis with real-time status updates
- Batch processing
- Evaluation metrics
"""

from .main import app
from .analysis_state import AnalysisStateManager, analysis_manager

__all__ = ["app", "AnalysisStateManager", "analysis_manager"]
