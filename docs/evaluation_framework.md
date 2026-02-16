# Evaluation & Ablation Framework

This document provides detailed technical documentation for the evaluation and ablation study framework implemented in the lung nodule multi-agent classification system.

## Overview

The framework addresses a critical question: **Does the system's complexity earn its place?**

By systematically disabling components and comparing against baselines, we determine whether:
- 6 agents are better than 3 (or 1)
- Dynamic weighting outperforms static/equal weighting
- Prolog consensus adds value over pure Python
- NLP components (NegEx, dependency parsing) contribute meaningfully

## Module Reference

### 1. Baselines (`evaluation/baselines.py`)

Provides mandatory baseline predictors for fair comparison.

#### Classes

| Class | Description |
|-------|-------------|
| `MajorityClassBaseline` | Always predicts most frequent class |
| `RandomBaseline` | Random with class prior distribution |
| `SingleAgentBaseline` | Wraps individual agent predictions |
| `UnweightedMajorityVote` | Simple majority (no weights) |
| `StaticWeightedAverage` | Fixed expert-assigned weights |
| `SklearnVotingEquivalent` | Standard sklearn VotingClassifier |
| `PurePythonWeightedAverage` | Python-only weighted average |

#### Usage

```python
from evaluation.baselines import get_all_baselines, evaluate_baselines

# Get all baseline predictors
baselines = get_all_baselines()

# Evaluate on ground truth
y_true = np.array([0, 1, 0, 1, 0])
results = evaluate_baselines(y_true)
```

### 2. Cross-Validation (`evaluation/cross_validation.py`)

Stratified K-fold cross-validation with proper metric aggregation.

#### Classes

| Class | Description |
|-------|-------------|
| `StratifiedKFoldSplitter` | Maintains class distribution across folds |
| `CrossValidationEvaluator` | Runs CV and aggregates metrics |
| `FoldResult` | Results from a single fold |
| `CVResult` | Aggregated CV results with CI |

#### Usage

```python
from evaluation.cross_validation import CrossValidationEvaluator

evaluator = CrossValidationEvaluator(
    n_folds=5,
    stratified=True,
    shuffle=True,
    random_state=42
)

# Run cross-validation
cv_results = run_simple_cv(X, y, predict_fn, n_folds=5)
print(f"Accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
```

### 3. Statistical Tests (`evaluation/statistical_tests.py`)

Statistical significance testing for model comparison.

#### Functions

| Function | Purpose |
|----------|---------|
| `mcnemar_test(y1, y2, y_true)` | Compare two classifiers on same data |
| `bootstrap_confidence_interval(metric_fn, y_pred, y_true)` | Bootstrap CI for any metric |
| `cohens_d(group1, group2)` | Effect size measurement |
| `bonferroni_correction(p_values)` | Multiple testing correction |
| `fdr_correction(p_values)` | Benjamini-Hochberg FDR |

#### Usage

```python
from evaluation.statistical_tests import mcnemar_test, bootstrap_confidence_interval

# McNemar's test
result = mcnemar_test(model1_preds, model2_preds, y_true)
print(f"p-value: {result.p_value:.4f}, significant: {result.significant}")

# Bootstrap CI for accuracy
ci = bootstrap_confidence_interval(
    metric_fn=lambda y, yp: accuracy_score(y, yp),
    y_pred=predictions,
    y_true=labels,
    n_bootstrap=1000,
    confidence=0.95
)
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

### 4. Ablation Framework (`evaluation/ablation_framework.py`)

Systematic ablation study execution.

#### Classes

| Class | Description |
|-------|-------------|
| `AblationConfig` | Configuration for single ablation experiment |
| `AblationCategory` | Enum: BASELINE, AGENT, WEIGHTING, SYMBOLIC, NLP |
| `AblationRunner` | Executes ablation experiments |
| `AblationStudyResults` | Aggregated results with comparisons |

#### Config Generators

```python
from evaluation.ablation_framework import (
    create_agent_ablations,
    create_weighting_ablations,
    create_symbolic_ablations,
    create_nlp_ablations
)

# Generate all ablation configurations
agent_configs = create_agent_ablations()  # Remove agents
weight_configs = create_weighting_ablations()  # Dynamic/static/equal
symbolic_configs = create_symbolic_ablations()  # Prolog vs Python
nlp_configs = create_nlp_ablations()  # NegEx, dependency parsing
```

#### Usage

```python
from evaluation.ablation_framework import AblationRunner, AblationConfig

runner = AblationRunner(output_dir="results/ablations")

# Run single ablation
config = AblationConfig(
    name="no_negex",
    category=AblationCategory.NLP,
    enabled_agents=["R1", "R2", "R3", "P1", "P2", "P3"],
    use_negex=False,
    use_dependency_parsing=True
)

result = runner.run_single(config, X, y, predict_fn)
```

### 5. Claim Verification (`evaluation/claim_verification.py`)

Automated verification of architectural claims.

#### Claims Tested

| Claim ID | Description | Verification Method |
|----------|-------------|---------------------|
| `multi_agent_vs_single` | Ensemble > best single agent | Accuracy comparison |
| `dynamic_vs_static_weights` | Dynamic weighting helps | Accuracy delta |
| `dynamic_vs_equal_weights` | Dynamic > equal weights | Accuracy delta |
| `prolog_vs_python` | Prolog ≈ Python consensus | Equivalence test |
| `negex_contribution` | NegEx improves F1 | F1 delta |
| `dependency_parsing_contribution` | Parsing improves F1 | F1 delta |
| `ensemble_improves_recall` | Ensemble recall > single | Recall comparison |
| `beats_majority_baseline` | System > naive baseline | Accuracy delta |

#### Usage

```python
from evaluation.claim_verification import ClaimVerifier

verifier = ClaimVerifier()

# Verify all claims
report = verifier.verify_all(ablation_results)

# Generate markdown report
md_report = verifier.generate_markdown_report()
```

### 6. Results Generator (`evaluation/results_generator.py`)

Generates formatted output in multiple formats.

#### Outputs

| Format | File | Description |
|--------|------|-------------|
| Markdown | `baseline_comparison.md` | Tables for documentation |
| LaTeX | `baseline_comparison.tex` | Tables for papers |
| JSON | `all_results.json` | Programmatic access |
| Markdown | `ablation_results.md` | Ablation matrix |
| Markdown | `cv_summary.md` | Cross-validation summary |

#### Usage

```python
from evaluation.results_generator import ResultsGenerator

generator = ResultsGenerator(output_dir="results/")

# Generate all formats
paths = generator.generate_all(
    ablation_results=ablation_dict,
    baseline_results=baseline_dict,
    cv_results=cv_dict
)
```

## Configuration (`config.py`)

Central configuration with ablation-related enums.

### Enums

```python
from config import WeightingMode, ConsensusMode, NLPMode

# Weighting modes
WeightingMode.DYNAMIC  # Learned from data richness
WeightingMode.STATIC   # Fixed expert weights
WeightingMode.EQUAL    # Uniform 1/n

# Consensus modes
ConsensusMode.PROLOG   # SWI-Prolog backend
ConsensusMode.PYTHON   # Pure Python (ablation)

# NLP modes
NLPMode.FULL           # All components
NLPMode.NO_NEGEX       # Without negation detection
NLPMode.NO_DEPENDENCY  # Without dep parsing
NLPMode.MINIMAL        # Regex only
```

### Default Configuration

```python
ABLATION_DEFAULTS = {
    "weighting_mode": WeightingMode.DYNAMIC,
    "consensus_mode": ConsensusMode.PROLOG,
    "nlp_mode": NLPMode.FULL,
    "use_negex": True,
    "use_dependency_parsing": True,
    "enabled_agents": ["R1", "R2", "R3", "P1", "P2", "P3"],
}
```

## Data Loader Extensions (`data/nlmcxr_loader.py`)

### New Methods

| Method | Purpose |
|--------|---------|
| `get_all_labeled_case_ids()` | Get cases WITHOUT NLP filtering |
| `get_nlp_rich_case_ids(no_filter=True)` | Bypass richness filter |
| `get_filtering_comparison()` | Compare filtered vs unfiltered |

### Filtering Transparency

```python
from data.nlmcxr_loader import NLMCXRLoader

loader = NLMCXRLoader()

# Compare what gets filtered
comparison = loader.get_filtering_comparison(min_score=3.0, limit=100)
print(f"Excluded: {comparison['excluded_count']}")
print(f"Exclusion rate: {comparison['exclusion_rate']:.1%}")
print(f"Bias warning: {comparison['bias_warning']}")
```

## Metrics Extensions (`evaluation/metrics.py`)

### Added: PR-AUC

Precision-Recall AUC is more informative than ROC-AUC for imbalanced data.

```python
from evaluation.metrics import EvaluationMetrics

evaluator = EvaluationMetrics()
metrics = evaluator.probability_evaluation(y_true, y_proba)

print(f"ROC-AUC: {metrics['auc_roc']:.3f}")
print(f"PR-AUC: {metrics['pr_auc']:.3f}")  # NEW
```

## Best Practices

### 1. Always Run Baselines First
```bash
python main_extended.py --run-baselines
```
This establishes the performance floor.

### 2. Use Cross-Validation
```bash
python main_extended.py --evaluate --cv-folds 5
```
Avoids overfitting to a single train/test split.

### 3. Report Both Filtered and Unfiltered
```bash
# With NLP filter (default)
python main_extended.py --evaluate

# Without filter
python main_extended.py --evaluate --no-filter
```
Transparency about case selection.

### 4. Check Statistical Significance
```python
from evaluation.statistical_tests import compare_models

result = compare_models(
    y_pred_baseline, y_pred_system, y_true,
    model1_name="Baseline",
    model2_name="Full System"
)
print(result.summary())
```

### 5. Document Effect Sizes
```python
from evaluation.statistical_tests import cohens_d, interpret_effect_size

d = cohens_d(system_accuracies, baseline_accuracies)
print(f"Cohen's d = {d:.2f} ({interpret_effect_size(d)})")
```

## Example: Complete Evaluation Pipeline

```python
import asyncio
from evaluation import (
    evaluate_baselines,
    CrossValidationEvaluator,
    AblationRunner,
    ClaimVerifier,
    ResultsGenerator
)

async def run_full_evaluation():
    # 1. Baselines
    baseline_results = evaluate_baselines(y_true)
    
    # 2. Cross-validation
    cv = CrossValidationEvaluator(n_folds=5)
    cv_results = await run_cv_evaluation(cv, cases, predict_fn)
    
    # 3. Ablations
    runner = AblationRunner()
    ablation_results = await runner.run_all(cases, y_true)
    
    # 4. Claim verification
    verifier = ClaimVerifier()
    claims = verifier.verify_all(ablation_results)
    
    # 5. Generate reports
    generator = ResultsGenerator()
    generator.generate_all(ablation_results, baseline_results, cv_results)
    generator.generate_claim_verification_report(claims)
    
    return claims

asyncio.run(run_full_evaluation())
```

## Interpretation Guide

### When System Complexity is Justified

✓ Multi-agent accuracy > best single agent by ≥ 2%  
✓ Dynamic weights > equal weights (statistically significant)  
✓ McNemar p-value < 0.05 for key comparisons  
✓ Cohen's d > 0.2 (small effect) for improvements  

### When to Simplify

✗ Single agent achieves same accuracy as ensemble  
✗ Equal weights perform as well as dynamic  
✗ Prolog consensus adds latency without accuracy gain  
✗ NLP components don't improve F1  

---

*Last updated: February 2026*
