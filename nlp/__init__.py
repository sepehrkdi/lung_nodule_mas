"""
NLP Module
==========

Natural Language Processing components for pathology report analysis.

EDUCATIONAL PURPOSE - NLP CONCEPTS:
- Tokenization: Breaking text into meaningful units
- Named Entity Recognition (NER): Identifying medical entities
- Pattern Matching: Regex-based extraction
- Dependency Parsing: Grammatical structure analysis
"""

from .extractor import MedicalNLPExtractor, ExtractionResult, ExtractedEntity, extract_from_report

__all__ = [
    'MedicalNLPExtractor',
    'ExtractionResult',
    'ExtractedEntity',
    'extract_from_report',
]

