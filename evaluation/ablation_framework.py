"""
Ablation Study Framework
========================

This module provides a comprehensive framework for running ablation studies
to validate architectural decisions in the lung nodule classification system.

Ablation Categories:
1. Agent-Level Ablations: Remove individual/groups of agents
2. Weighting Mechanism Ablations: Static vs dynamic weights
3. Symbolic Layer Ablations: Prolog vs Python consensus
4. NLP Component Ablations: With/without NegEx, dependency parsing

Each ablation answers: "Does this component contribute measurably?"
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from itertools import combinations
import copy

from config import (
    ALL_AGENTS, RADIOLOGIST_AGENTS, PATHOLOGIST_AGENTS,
    AGENT_SHORT_NAMES, ABLATION_RESULTS_DIR, ensure_directories,
    WeightingMode, ConsensusMode, NLPMode, ABLATION_DEFAULTS
)

logger = logging.getLogger(__name__)


# =============================================================================
# ABLATION CONFIGURATION
# =============================================================================

@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    description: str
    
    # Agent configuration
    excluded_agents: List[str] = field(default_factory=list)
    
    # Weighting configuration
    weighting_mode: str = WeightingMode.DYNAMIC
    use_reliability_adjustment: bool = True
    use_size_penalty: bool = True
    
    # Consensus configuration
    consensus_mode: str = ConsensusMode.PROLOG
    use_disagreement_resolution: bool = True
    use_lung_rads: bool = True
    
    # NLP configuration
    nlp_mode: str = NLPMode.FULL
    use_negex: bool = True
    use_dependency_parsing: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AblationConfig":
        return cls(**d)
    
    def active_agents(self) -> List[str]:
        """Get list of active (non-excluded) agents."""
        return [a for a in ALL_AGENTS if a not in self.excluded_agents]


# =============================================================================
# PREDEFINED ABLATION CONFIGURATIONS
# =============================================================================

class AblationCategory(Enum):
    """Categories of ablation studies."""
    BASELINE = "baseline"
    AGENT = "agent"
    WEIGHTING = "weighting"
    SYMBOLIC = "symbolic"
    NLP = "nlp"


def create_baseline_ablations() -> Dict[str, AblationConfig]:
    """Create baseline configuration (full system)."""
    return {
        "full_system": AblationConfig(
            name="full_system",
            description="Full system with all components enabled"
        ),
    }


def create_agent_ablations() -> Dict[str, AblationConfig]:
    """
    Create agent-level ablation configurations.
    
    Tests whether "multi-agent" structure matters:
    - Single agent removals (6→5)
    - Modality removals (CNN-only, NLP-only)
    - Systematic agent reduction
    """
    ablations = {}
    
    # Single agent removals
    for agent in ALL_AGENTS:
        short_name = AGENT_SHORT_NAMES.get(agent, agent)
        ablations[f"remove_{short_name}"] = AblationConfig(
            name=f"remove_{short_name}",
            description=f"Remove {agent} from the ensemble",
            excluded_agents=[agent]
        )
    
    # Modality-level ablations
    ablations["cnn_only"] = AblationConfig(
        name="cnn_only",
        description="CNN radiologists only (remove all NLP pathologists)",
        excluded_agents=PATHOLOGIST_AGENTS.copy()
    )
    
    ablations["nlp_only"] = AblationConfig(
        name="nlp_only",
        description="NLP pathologists only (remove all CNN radiologists)",
        excluded_agents=RADIOLOGIST_AGENTS.copy()
    )
    
    # Two-agent removals (6→4)
    for combo in combinations(ALL_AGENTS, 2):
        names = [AGENT_SHORT_NAMES.get(a, a) for a in combo]
        ablations[f"remove_{names[0]}_{names[1]}"] = AblationConfig(
            name=f"remove_{names[0]}_{names[1]}",
            description=f"Remove {combo[0]} and {combo[1]}",
            excluded_agents=list(combo)
        )
    
    # Three-agent removals (6→3) - keeping one from each modality
    for rad_agents in combinations(RADIOLOGIST_AGENTS, 2):
        for path_agents in combinations(PATHOLOGIST_AGENTS, 2):
            excluded = list(rad_agents) + list(path_agents)
            kept_rad = [a for a in RADIOLOGIST_AGENTS if a not in rad_agents][0]
            kept_path = [a for a in PATHOLOGIST_AGENTS if a not in path_agents][0]
            kept_names = [AGENT_SHORT_NAMES.get(kept_rad), AGENT_SHORT_NAMES.get(kept_path)]
            ablations[f"only_{kept_names[0]}_{kept_names[1]}"] = AblationConfig(
                name=f"only_{kept_names[0]}_{kept_names[1]}",
                description=f"Keep only {kept_rad} and {kept_path}",
                excluded_agents=excluded
            )
    
    return ablations


def create_weighting_ablations() -> Dict[str, AblationConfig]:
    """
    Create weighting mechanism ablation configurations.
    
    Tests whether weighting contributes meaningfully:
    - Dynamic vs static weighting
    - Static vs equal weights
    - Component removal (reliability, size penalty)
    """
    ablations = {}
    
    # Static weighting (no dynamic adjustment)
    ablations["static_weights"] = AblationConfig(
        name="static_weights",
        description="Static base weights (no richness-based adjustment)",
        weighting_mode=WeightingMode.STATIC
    )
    
    # Equal weights (all agents weight = 1.0)
    ablations["equal_weights"] = AblationConfig(
        name="equal_weights",
        description="Equal weights for all agents (no prior knowledge)",
        weighting_mode=WeightingMode.EQUAL
    )
    
    # Remove reliability adjustment
    ablations["no_reliability_adjust"] = AblationConfig(
        name="no_reliability_adjust",
        description="Dynamic weights without continual learning updates",
        use_reliability_adjustment=False
    )
    
    # Remove size penalty
    ablations["no_size_penalty"] = AblationConfig(
        name="no_size_penalty",
        description="Dynamic weights without size-source penalty",
        use_size_penalty=False
    )
    
    # Remove both adjustments
    ablations["no_weight_adjustments"] = AblationConfig(
        name="no_weight_adjustments",
        description="Base dynamic weights only (no penalties or learning)",
        use_reliability_adjustment=False,
        use_size_penalty=False
    )
    
    return ablations


def create_symbolic_ablations() -> Dict[str, AblationConfig]:
    """
    Create symbolic layer ablation configurations.
    
    Tests whether Prolog and neuro-symbolic framing are necessary:
    - Prolog vs Python consensus
    - Remove Prolog entirely
    - Remove disagreement resolution
    - Remove Lung-RADS categorization
    """
    ablations = {}
    
    # Python consensus (no Prolog)
    ablations["python_consensus"] = AblationConfig(
        name="python_consensus",
        description="Python-only consensus (no Prolog)",
        consensus_mode=ConsensusMode.PYTHON
    )
    
    # Pure ensemble (no consensus logic)
    ablations["pure_ensemble"] = AblationConfig(
        name="pure_ensemble",
        description="Direct weighted average (no consensus/disagreement resolution)",
        consensus_mode=ConsensusMode.NONE
    )
    
    # No disagreement resolution
    ablations["no_disagreement_resolution"] = AblationConfig(
        name="no_disagreement_resolution",
        description="Prolog consensus without disagreement resolution",
        use_disagreement_resolution=False
    )
    
    # No Lung-RADS
    ablations["no_lung_rads"] = AblationConfig(
        name="no_lung_rads",
        description="Skip Lung-RADS categorization step",
        use_lung_rads=False
    )
    
    # Python + no disagreement (pure weighted average in Python)
    ablations["python_pure_average"] = AblationConfig(
        name="python_pure_average",
        description="Python weighted average without disagreement resolution",
        consensus_mode=ConsensusMode.PYTHON,
        use_disagreement_resolution=False
    )
    
    return ablations


def create_nlp_ablations() -> Dict[str, AblationConfig]:
    """
    Create NLP component ablation configurations.
    
    Tests whether "advanced NLP" components help:
    - With/without NegEx
    - With/without dependency parsing
    - Regex-only extraction
    """
    ablations = {}
    
    # No NegEx
    ablations["no_negex"] = AblationConfig(
        name="no_negex",
        description="NLP pipeline without negation detection (NegEx)",
        use_negex=False
    )
    
    # No dependency parsing
    ablations["no_dependency_parsing"] = AblationConfig(
        name="no_dependency_parsing",
        description="NLP pipeline without dependency frame extraction",
        use_dependency_parsing=False
    )
    
    # Regex only (no spaCy, no NegEx, no dependency parsing)
    ablations["regex_only"] = AblationConfig(
        name="regex_only",
        description="Regex-only NLP extraction (no spaCy/NegEx/dependencies)",
        nlp_mode=NLPMode.REGEX_ONLY,
        use_negex=False,
        use_dependency_parsing=False
    )
    
    # NegEx only (regex + negation, no dependency frames)
    ablations["negex_only"] = AblationConfig(
        name="negex_only",
        description="Regex extraction with NegEx (no dependency parsing)",
        nlp_mode=NLPMode.NEGEX_ONLY,
        use_dependency_parsing=False
    )
    
    # No NLP pathologists (agent removal + NLP ablation combined)
    ablations["no_nlp_agents"] = AblationConfig(
        name="no_nlp_agents",
        description="Remove all pathologists that use NLP",
        excluded_agents=["pathologist_spacy", "pathologist_context"]
    )
    
    return ablations


def get_all_ablation_configs() -> Dict[str, AblationConfig]:
    """Get all predefined ablation configurations."""
    all_configs = {}
    all_configs.update(create_baseline_ablations())
    all_configs.update(create_agent_ablations())
    all_configs.update(create_weighting_ablations())
    all_configs.update(create_symbolic_ablations())
    all_configs.update(create_nlp_ablations())
    return all_configs


def get_ablation_configs_by_category(category: AblationCategory) -> Dict[str, AblationConfig]:
    """Get ablation configurations for a specific category."""
    category_map = {
        AblationCategory.BASELINE: create_baseline_ablations,
        AblationCategory.AGENT: create_agent_ablations,
        AblationCategory.WEIGHTING: create_weighting_ablations,
        AblationCategory.SYMBOLIC: create_symbolic_ablations,
        AblationCategory.NLP: create_nlp_ablations,
    }
    return category_map.get(category, lambda: {})()


# =============================================================================
# ABLATION RESULTS
# =============================================================================

@dataclass
class AblationResult:
    """Result of a single ablation experiment."""
    config: AblationConfig
    metrics: Dict[str, float]
    cv_metrics: Optional[Dict[str, Tuple[float, float]]] = None  # mean, std
    predictions: List[int] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)
    runtime_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "cv_metrics": {k: list(v) for k, v in (self.cv_metrics or {}).items()},
            "runtime_seconds": self.runtime_seconds
        }


@dataclass
class AblationStudyResults:
    """Results of a complete ablation study."""
    study_name: str
    baseline_result: AblationResult
    ablation_results: Dict[str, AblationResult]
    comparisons: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_name": self.study_name,
            "baseline": self.baseline_result.to_dict(),
            "ablations": {k: v.to_dict() for k, v in self.ablation_results.items()},
            "comparisons": self.comparisons
        }
    
    def save(self, output_dir: Path = None) -> Path:
        """Save study results to JSON."""
        ensure_directories()
        output_dir = output_dir or ABLATION_RESULTS_DIR
        output_path = output_dir / f"{self.study_name}_results.json"
        
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Ablation study results saved to {output_path}")
        return output_path


# =============================================================================
# ABLATION RUNNER
# =============================================================================

class AblationRunner:
    """
    Runner for systematic ablation studies.
    
    Orchestrates the execution of ablation experiments and
    collects results for analysis.
    """
    
    def __init__(
        self,
        predict_fn: Callable[[AblationConfig, List[str]], Tuple[List[int], List[float]]],
        metrics_fn: Callable[[List[int], List[int], List[float]], Dict[str, float]],
        case_ids: List[str],
        y_true: List[int],
        use_cv: bool = False,
        n_cv_folds: int = 5
    ):
        """
        Initialize ablation runner.
        
        Args:
            predict_fn: Function(config, case_ids) -> (predictions, probabilities)
            metrics_fn: Function(y_true, y_pred, y_prob) -> metrics_dict
            case_ids: List of case identifiers
            y_true: Ground truth labels
            use_cv: Whether to use cross-validation
            n_cv_folds: Number of CV folds
        """
        self.predict_fn = predict_fn
        self.metrics_fn = metrics_fn
        self.case_ids = case_ids
        self.y_true = y_true
        self.use_cv = use_cv
        self.n_cv_folds = n_cv_folds
    
    def run_single_ablation(self, config: AblationConfig) -> AblationResult:
        """Run a single ablation experiment."""
        import time
        
        logger.info(f"Running ablation: {config.name}")
        logger.info(f"  Description: {config.description}")
        logger.info(f"  Active agents: {config.active_agents()}")
        
        start_time = time.time()
        
        try:
            predictions, probabilities = self.predict_fn(config, self.case_ids)
            metrics = self.metrics_fn(self.y_true, predictions, probabilities)
        except Exception as e:
            logger.error(f"Ablation {config.name} failed: {e}")
            metrics = {"error": 1.0}
            predictions, probabilities = [], []
        
        runtime = time.time() - start_time
        
        result = AblationResult(
            config=config,
            metrics=metrics,
            predictions=predictions,
            probabilities=probabilities,
            runtime_seconds=runtime
        )
        
        logger.info(f"  Metrics: {metrics}")
        logger.info(f"  Runtime: {runtime:.2f}s")
        
        return result
    
    def run_study(
        self,
        study_name: str,
        configs: Dict[str, AblationConfig] = None,
        category: AblationCategory = None
    ) -> AblationStudyResults:
        """
        Run a complete ablation study.
        
        Args:
            study_name: Name for this study
            configs: Dict of ablation configs to run (if None, uses category)
            category: Ablation category to run (if configs is None)
            
        Returns:
            AblationStudyResults with all results
        """
        # Get configs
        if configs is None:
            if category is not None:
                configs = get_ablation_configs_by_category(category)
            else:
                configs = get_all_ablation_configs()
        
        # Ensure baseline is included
        baseline_configs = create_baseline_ablations()
        if "full_system" not in configs:
            configs = {**baseline_configs, **configs}
        
        logger.info(f"Starting ablation study: {study_name}")
        logger.info(f"Running {len(configs)} configurations")
        
        # Run baseline first
        baseline_result = self.run_single_ablation(configs["full_system"])
        
        # Run ablations
        ablation_results = {}
        for name, config in configs.items():
            if name == "full_system":
                continue
            ablation_results[name] = self.run_single_ablation(config)
        
        # Compute comparisons
        from evaluation.statistical_tests import compare_models
        
        comparisons = {}
        baseline_preds = baseline_result.predictions
        
        for name, result in ablation_results.items():
            if result.predictions and baseline_preds:
                comparison = compare_models(
                    self.y_true,
                    baseline_preds,
                    result.predictions,
                    model_a_name="full_system",
                    model_b_name=name
                )
                comparisons[name] = {
                    "difference": comparison.difference,
                    "is_better": comparison.is_b_better(),
                    "tests": [t.to_dict() for t in comparison.test_results]
                }
        
        study_results = AblationStudyResults(
            study_name=study_name,
            baseline_result=baseline_result,
            ablation_results=ablation_results,
            comparisons=comparisons
        )
        
        # Save results
        study_results.save()
        
        return study_results
    
    def run_all_studies(self) -> Dict[str, AblationStudyResults]:
        """Run ablation studies for all categories."""
        all_results = {}
        
        for category in AblationCategory:
            study_name = f"ablation_{category.value}"
            all_results[category.value] = self.run_study(
                study_name=study_name,
                category=category
            )
        
        return all_results


# =============================================================================
# ABLATION MATRIX GENERATION
# =============================================================================

def generate_ablation_matrix(
    study_results: AblationStudyResults,
    metric_name: str = "accuracy"
) -> str:
    """
    Generate a matrix comparing all ablations.
    
    Returns:
        Formatted string with ablation comparison matrix
    """
    lines = ["=" * 70]
    lines.append(f"Ablation Study: {study_results.study_name}")
    lines.append("=" * 70)
    
    baseline_metric = study_results.baseline_result.metrics.get(metric_name, 0)
    lines.append(f"\nBaseline ({metric_name}): {baseline_metric:.4f}")
    lines.append("-" * 70)
    lines.append(f"{'Ablation':<35} {metric_name:>10} {'Δ':>10} {'Sig':>8}")
    lines.append("-" * 70)
    
    for name, result in sorted(study_results.ablation_results.items()):
        ablation_metric = result.metrics.get(metric_name, 0)
        delta = ablation_metric - baseline_metric
        
        # Get significance
        comparison = study_results.comparisons.get(name, {})
        sig_marker = ""
        if comparison:
            for test in comparison.get("tests", []):
                p = test.get("p_value", 1)
                if p < 0.001:
                    sig_marker = "***"
                elif p < 0.01:
                    sig_marker = "**"
                elif p < 0.05:
                    sig_marker = "*"
        
        lines.append(f"{name:<35} {ablation_metric:>10.4f} {delta:>+10.4f} {sig_marker:>8}")
    
    lines.append("-" * 70)
    lines.append("\nSignificance: * p<0.05, ** p<0.01, *** p<0.001")
    lines.append("Positive Δ means ablation performs BETTER than baseline")
    lines.append("Negative Δ means component HELPS performance")
    
    return "\n".join(lines)


# =============================================================================
# HYPERPARAMETER SENSITIVITY SWEEP
# =============================================================================

def run_sensitivity_sweep(
    parameter_name: str,
    parameter_values: List[Any],
    base_config: AblationConfig,
    predict_fn: Callable,
    metrics_fn: Callable,
    case_ids: List[str],
    y_true: List[int]
) -> Dict[Any, Dict[str, float]]:
    """
    Run sensitivity sweep over a hyperparameter.
    
    Args:
        parameter_name: Name of parameter to sweep
        parameter_values: Values to test
        base_config: Base configuration to modify
        predict_fn, metrics_fn, case_ids, y_true: As in AblationRunner
        
    Returns:
        Dict mapping parameter values to metrics
    """
    results = {}
    
    for value in parameter_values:
        # Create modified config
        config_dict = base_config.to_dict()
        config_dict[parameter_name] = value
        config_dict["name"] = f"sweep_{parameter_name}_{value}"
        config = AblationConfig.from_dict(config_dict)
        
        # Run prediction
        try:
            predictions, probabilities = predict_fn(config, case_ids)
            metrics = metrics_fn(y_true, predictions, probabilities)
            results[value] = metrics
        except Exception as e:
            logger.error(f"Sweep failed for {parameter_name}={value}: {e}")
            results[value] = {"error": 1.0}
    
    return results
