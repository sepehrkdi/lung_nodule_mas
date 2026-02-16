"""
Evaluation Module
=================

Metrics and visualization for model and system evaluation.

Provides:
- Binary classification metrics (benign vs malignant)
- Agent interaction analysis
- Cross-validation framework
- Statistical tests for model comparison
- Ablation study framework
- Claim verification

EDUCATIONAL PURPOSE:
- Classification Metrics: Accuracy, Precision, Recall, F1
- ROC/AUC Analysis: Probability-based evaluation
- Confusion Matrix: Binary performance visualization
- Statistical Significance: McNemar's test, bootstrap CI
"""

from .metrics import (
    EvaluationMetrics,
    ClassificationMetrics,
    evaluate_results,
)

from .baselines import (
    BaselinePredictor,
    MajorityClassBaseline,
    RandomBaseline,
    SingleAgentBaseline,
    UnweightedMajorityVote,
    StaticWeightedAverage,
    SklearnVotingEquivalent,
    PurePythonWeightedAverage,
    get_all_baselines,
    get_baseline,
    evaluate_baselines,
)

from .cross_validation import (
    StratifiedKFoldSplitter,
    CrossValidationEvaluator,
    FoldResult,
    CVResult,
    compute_cv_metrics_fn,
    run_simple_cv,
)

from .statistical_tests import (
    TestResult,
    ComparisonResult,
    mcnemar_test,
    paired_ttest,
    bootstrap_confidence_interval,
    bootstrap_metric_comparison,
    cohens_d,
    interpret_effect_size,
    bonferroni_correction,
    fdr_correction,
    compare_models,
    run_all_pairwise_comparisons,
    generate_comparison_report,
)

from .ablation_framework import (
    AblationConfig,
    AblationCategory,
    AblationResult,
    AblationStudyResults,
    AblationRunner,
    get_all_ablation_configs,
    get_ablation_configs_by_category,
    generate_ablation_matrix,
    create_agent_ablations,
    create_weighting_ablations,
    create_symbolic_ablations,
    create_nlp_ablations,
)

from .claim_verification import (
    ClaimStatus,
    VerifiedClaim,
    ClaimVerifier,
    VerificationReport,
    generate_verification_report,
    ARCHITECTURAL_CLAIMS,
)

from .results_generator import (
    TableConfig,
    ResultsGenerator,
    generate_summary_statistics,
)

__all__ = [
    # Metrics
    'EvaluationMetrics',
    'ClassificationMetrics',
    'evaluate_results',
    
    # Baselines
    'BaselinePredictor',
    'MajorityClassBaseline',
    'RandomBaseline',
    'SingleAgentBaseline',
    'UnweightedMajorityVote',
    'StaticWeightedAverage',
    'SklearnVotingEquivalent',
    'PurePythonWeightedAverage',
    'get_all_baselines',
    'get_baseline',
    'evaluate_baselines',
    
    # Cross-validation
    'StratifiedKFoldSplitter',
    'CrossValidationEvaluator',
    'FoldResult',
    'CVResult',
    'compute_cv_metrics_fn',
    'run_simple_cv',
    
    # Statistical tests
    'TestResult',
    'ComparisonResult',
    'mcnemar_test',
    'paired_ttest',
    'bootstrap_confidence_interval',
    'bootstrap_metric_comparison',
    'cohens_d',
    'interpret_effect_size',
    'bonferroni_correction',
    'fdr_correction',
    'compare_models',
    'run_all_pairwise_comparisons',
    'generate_comparison_report',
    
    # Ablation framework
    'AblationConfig',
    'AblationCategory',
    'AblationResult',
    'AblationStudyResults',
    'AblationRunner',
    'get_all_ablation_configs',
    'get_ablation_configs_by_category',
    'generate_ablation_matrix',
    'create_agent_ablations',
    'create_weighting_ablations',
    'create_symbolic_ablations',
    'create_nlp_ablations',
    
    # Claim verification
    'ClaimStatus',
    'VerifiedClaim',
    'ClaimVerifier',
    'VerificationReport',
    'generate_verification_report',
    'ARCHITECTURAL_CLAIMS',
    
    # Results generator
    'TableConfig',
    'ResultsGenerator',
    'generate_summary_statistics',
]

