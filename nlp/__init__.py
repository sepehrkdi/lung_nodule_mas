"""
NLP Module
==========

Natural Language Processing components for pathology report analysis.

EDUCATIONAL PURPOSE - NLP CONCEPTS:
- Tokenization: Breaking text into meaningful units
- Named Entity Recognition (NER): Identifying medical entities
- Pattern Matching: Regex-based extraction
- Dependency Parsing: Grammatical structure analysis
- Section Parsing: Splitting reports into FINDINGS/IMPRESSION
- Negation Detection: NegEx-style trigger/scope analysis
- Uncertainty Detection: Hedging language identification
"""

from .extractor import MedicalNLPExtractor, ExtractionResult, ExtractedEntity, extract_from_report
from .report_parser import ReportParser, ParsedReport, SectionData, split_sections
from .negation_detector import NegExDetector, Certainty, EntityCertainty, is_negated, is_uncertain, get_certainty

__all__ = [
    # Core extractor
    'MedicalNLPExtractor',
    'ExtractionResult',
    'ExtractedEntity',
    'extract_from_report',
    # Section parsing
    'ReportParser',
    'ParsedReport',
    'SectionData',
    'split_sections',
    # Negation/Uncertainty detection
    'NegExDetector',
    'Certainty',
    'EntityCertainty',
    'is_negated',
    'is_uncertain',
    'get_certainty',
]

