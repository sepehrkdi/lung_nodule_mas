"""
Central Configuration for Lung Nodule MAS
==========================================

This module provides a single source of truth for all configuration
parameters, including random seeds for reproducibility.
"""

import os
from pathlib import Path
from typing import Dict, Any

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

RANDOM_SEED = 42  # Fixed seed for reproducibility

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
ABLATION_RESULTS_DIR = RESULTS_DIR / "ablation_results"
CV_RESULTS_DIR = RESULTS_DIR / "cv_results"

# =============================================================================
# EVALUATION PARAMETERS
# =============================================================================

DEFAULT_CV_FOLDS = 5
DEFAULT_BOOTSTRAP_SAMPLES = 1000
SIGNIFICANCE_LEVEL = 0.05  # Alpha for statistical tests

# =============================================================================
# CLASS INFORMATION  
# =============================================================================

CLASS_NAMES = {0: "benign", 1: "malignant"}
POSITIVE_CLASS = 1  # Malignant is the positive class

# =============================================================================
# AGENT REGISTRY
# =============================================================================

RADIOLOGIST_AGENTS = ["radiologist_densenet", "radiologist_resnet", "radiologist_rulebased"]
PATHOLOGIST_AGENTS = ["pathologist_regex", "pathologist_spacy", "pathologist_context"]
ALL_AGENTS = RADIOLOGIST_AGENTS + PATHOLOGIST_AGENTS

# Short names for display
AGENT_SHORT_NAMES = {
    "radiologist_densenet": "R1",
    "radiologist_resnet": "R2",
    "radiologist_rulebased": "R3",
    "pathologist_regex": "P1",
    "pathologist_spacy": "P2",
    "pathologist_context": "P3",
}

# Inverse mapping
AGENT_FULL_NAMES = {v: k for k, v in AGENT_SHORT_NAMES.items()}

# =============================================================================
# WEIGHTING MODES
# =============================================================================

class WeightingMode:
    DYNAMIC = "dynamic"      # Per-case richness-based scaling
    STATIC = "static"        # Fixed BASE_WEIGHTS
    EQUAL = "equal"          # All weights = 1.0

# =============================================================================
# CONSENSUS MODES
# =============================================================================

class ConsensusMode:
    PROLOG = "prolog"        # Prolog-based consensus
    PYTHON = "python"        # Pure Python consensus
    NONE = "none"            # Direct weighted average

# =============================================================================
# NLP MODES
# =============================================================================

class NLPMode:
    FULL = "full"            # Full pipeline (spaCy, dependency parsing, NegEx)
    NEGEX_ONLY = "negex_only"  # Regex + NegEx
    REGEX_ONLY = "regex_only"  # Regex patterns only

# =============================================================================
# ABLATION CONFIGURATION
# =============================================================================

# Default ablation settings
ABLATION_DEFAULTS = {
    "weighting_mode": WeightingMode.DYNAMIC,
    "consensus_mode": ConsensusMode.PROLOG,
    "nlp_mode": NLPMode.FULL,
    "use_dependency_parsing": True,
    "use_negex": True,
    "use_lung_rads": True,
    "use_reliability_adjustment": True,
    "use_size_penalty": True,
    "excluded_agents": [],
}


def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [RESULTS_DIR, ABLATION_RESULTS_DIR, CV_RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary."""
    return {
        "random_seed": RANDOM_SEED,
        "cv_folds": DEFAULT_CV_FOLDS,
        "bootstrap_samples": DEFAULT_BOOTSTRAP_SAMPLES,
        "significance_level": SIGNIFICANCE_LEVEL,
        "class_names": CLASS_NAMES,
        "positive_class": POSITIVE_CLASS,
        "agents": ALL_AGENTS,
    }
