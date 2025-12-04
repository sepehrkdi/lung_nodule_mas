"""
Data Module
===========

Dataset loading and preprocessing for medical imaging data.

SUPPORTED DATASETS:
1. Open-I Indiana University Chest X-ray Collection (recommended)
   - Paired images with radiology reports
   - ~7,500 chest X-rays
   - https://openi.nlm.nih.gov/

2. LIDC-IDRI (optional, requires pylidc)
   - CT scans with radiologist annotations
   - Malignancy scores 1-5

EDUCATIONAL PURPOSE:
- Medical Imaging: X-ray/CT processing
- Paired Data: Image + text for CV + NLP agents
- Report Generation: Structured to natural language conversion
"""

from .openi_loader import OpenILoader, LIDCLoader  # LIDCLoader is alias
from .report_generator import ReportGenerator

__all__ = ['OpenILoader', 'LIDCLoader', 'ReportGenerator']
