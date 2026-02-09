"""
Knowledge Base Module
=====================

Contains Prolog knowledge bases for symbolic reasoning.

EDUCATIONAL PURPOSE - SYMBOLIC AI CONCEPTS:
- First-Order Logic (FOL): Facts and rules
- Prolog: Unification, backtracking, inference
- Knowledge Representation: Ontologies, taxonomies

STRICT MODE - NO FALLBACKS:
This module requires SWI-Prolog and PySwip to be installed.
The system will fail fast if Prolog is unavailable.
"""

from knowledge.prolog_engine import (
    PrologEngine,
    PrologUnavailableError,
    PrologQueryError,
    LungRADSKnowledgeBase,
    validate_prolog_installation,
)

__all__ = [
    "PrologEngine",
    "PrologUnavailableError",
    "PrologQueryError",
    "LungRADSKnowledgeBase",
    "validate_prolog_installation",
]
