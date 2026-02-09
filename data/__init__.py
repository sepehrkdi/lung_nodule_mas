"""
Data Module
===========

Dataset loading and preprocessing for medical imaging data.

SUPPORTED DATASET:
- NLMCXR (NLM Chest X-Ray, Open-I Indiana University)
  - Paired chest X-rays with radiology reports
  - ~3,956 cases with ~7,472 images
  - Supports PA, Lateral, and other views
  - Ground truth derived from report text via NLP

EDUCATIONAL PURPOSE:
- Medical Imaging: X-ray processing
- Paired Data: Image + text for CV + NLP agents
- Multi-view Analysis: Combining predictions from multiple views
"""

from .base_loader import BaseNoduleLoader, LoaderFactory
from .nlmcxr_loader import NLMCXRLoader
from .nlmcxr_parser import NLMCXRCase, parse_all_nlmcxr_cases

__all__ = [
    'BaseNoduleLoader',
    'LoaderFactory',
    'NLMCXRLoader',
    'NLMCXRCase',
    'parse_all_nlmcxr_cases',
]
