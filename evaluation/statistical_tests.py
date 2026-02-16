"""
Statistical Tests for Model Comparison
======================================

This module implements statistical tests for rigorous evaluation:

1. McNemar's Test: Paired classifier comparison on binary outcomes
2. Paired t-test: Comparison across CV folds
3. Bootstrap Confidence Intervals: Non-parametric uncertainty estimation
4. Effect Size (Cohen's d): Magnitude of differences
5. Multiple Testing Correction: Bonferroni and FDR

These tests answer: "Are improvements statistically real or noise?"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from scipy import stats
import logging

from config import RANDOM_SEED, SIGNIFICANCE_LEVEL, DEFAULT_BOOTSTRAP_SAMPLES

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool  # At SIGNIFICANCE_LEVEL
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "effect_size": self.effect_size,
            "confidence_interval": list(self.confidence_interval) if self.confidence_interval else None,
            "interpretation": self.interpretation
        }
    
    def significance_marker(self) -> str:
        """Return significance marker (* p<0.05, ** p<0.01, *** p<0.001)."""
        if self.p_value < 0.001:
            return "***"
        elif self.p_value < 0.01:
            return "**"
        elif self.p_value < 0.05:
            return "*"
        else:
            return ""


@dataclass
class ComparisonResult:
    """Result of comparing two models/methods."""
    model_a_name: str
    model_b_name: str
    metric_name: str
    model_a_value: float
    model_b_value: float
    difference: float  # model_b - model_a
    test_results: List[TestResult]
    
    def is_b_better(self) -> bool:
        """Check if model B is significantly better than model A."""
        return self.difference > 0 and any(t.significant for t in self.test_results)
    
    def summary(self) -> str:
        """Generate summary string."""
        sig_markers = "".join(t.significance_marker() for t in self.test_results if t.significant)
        return (f"{self.model_b_name} vs {self.model_a_name} ({self.metric_name}): "
                f"Δ={self.difference:+.4f} {sig_markers}")


# =============================================================================
# MCNEMAR'S TEST
# =============================================================================

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    correction: bool = True
) -> TestResult:
    """
    McNemar's test for comparing two classifiers on paired samples.
    
    Tests whether the disagreements between two classifiers are symmetric.
    
    Args:
        y_true: Ground truth labels
        y_pred_a: Predictions from classifier A
        y_pred_b: Predictions from classifier B
        correction: Whether to apply continuity correction
        
    Returns:
        TestResult with statistic, p-value, and significance
    """
    y_true = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)
    
    # Compute contingency table of disagreements
    # correct_a_incorrect_b: A correct, B incorrect
    # incorrect_a_correct_b: A incorrect, B correct
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    
    # c01: A correct, B incorrect
    c01 = np.sum(correct_a & ~correct_b)
    # c10: A incorrect, B correct  
    c10 = np.sum(~correct_a & correct_b)
    
    # McNemar statistic
    if correction:
        # With continuity correction
        if c01 + c10 == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            statistic = (abs(c01 - c10) - 1) ** 2 / (c01 + c10)
            p_value = stats.chi2.sf(statistic, df=1)
    else:
        # Without correction
        if c01 + c10 == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            statistic = (c01 - c10) ** 2 / (c01 + c10)
            p_value = stats.chi2.sf(statistic, df=1)
    
    # Interpretation
    if c10 > c01:
        interp = f"Model B corrected {c10-c01} more errors than A"
    elif c01 > c10:
        interp = f"Model A corrected {c01-c10} more errors than B"
    else:
        interp = "Models have equal error correction rates"
    
    return TestResult(
        test_name="McNemar's test",
        statistic=statistic,
        p_value=p_value,
        significant=p_value < SIGNIFICANCE_LEVEL,
        effect_size=(c10 - c01) / max(c01 + c10, 1),  # Normalized difference
        interpretation=interp
    )


# =============================================================================
# PAIRED T-TEST
# =============================================================================

def paired_ttest(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alternative: str = "two-sided"
) -> TestResult:
    """
    Paired t-test for comparing metrics across CV folds.
    
    Args:
        values_a: Metric values from method A (e.g., per-fold accuracies)
        values_b: Metric values from method B
        alternative: "two-sided", "less", or "greater"
        
    Returns:
        TestResult with statistic, p-value, and significance
    """
    values_a = np.array(values_a)
    values_b = np.array(values_b)
    
    if len(values_a) != len(values_b):
        raise ValueError("Arrays must have same length")
    
    if len(values_a) < 2:
        return TestResult(
            test_name="Paired t-test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            interpretation="Insufficient samples for t-test"
        )
    
    # Compute differences
    diff = values_b - values_a
    mean_diff = np.mean(diff)
    
    # Perform t-test
    statistic, p_value = stats.ttest_rel(values_a, values_b, alternative=alternative)
    
    # Effect size (Cohen's d for paired samples)
    std_diff = np.std(diff, ddof=1)
    effect_size = mean_diff / std_diff if std_diff > 0 else 0.0
    
    # Interpretation
    if mean_diff > 0:
        interp = f"Method B is higher by {mean_diff:.4f} on average"
    elif mean_diff < 0:
        interp = f"Method A is higher by {-mean_diff:.4f} on average"
    else:
        interp = "Methods have equal means"
    
    return TestResult(
        test_name="Paired t-test",
        statistic=statistic,
        p_value=p_value,
        significant=p_value < SIGNIFICANCE_LEVEL,
        effect_size=effect_size,
        interpretation=interp
    )


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    ci_level: float = 0.95,
    random_state: int = RANDOM_SEED
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predictions (or probabilities)
        metric_fn: Function(y_true, y_pred) -> float
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (default 0.95 for 95% CI)
        random_state: Random seed
        
    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_samples = len(y_true)
    
    rng = np.random.RandomState(random_state)
    
    # Point estimate
    point_estimate = metric_fn(y_true, y_pred)
    
    # Bootstrap samples
    bootstrap_values = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_y_true = y_true[indices]
        boot_y_pred = y_pred[indices]
        
        try:
            boot_value = metric_fn(boot_y_true, boot_y_pred)
            bootstrap_values.append(boot_value)
        except Exception:
            continue
    
    if not bootstrap_values:
        return point_estimate, point_estimate, point_estimate
    
    bootstrap_values = np.array(bootstrap_values)
    
    # Compute percentile CI
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def bootstrap_metric_comparison(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    random_state: int = RANDOM_SEED
) -> TestResult:
    """
    Bootstrap test for comparing two methods on a metric.
    
    Tests whether the difference in metric values is significantly
    different from zero using bootstrap sampling.
    
    Args:
        y_true: Ground truth labels
        y_pred_a: Predictions from method A
        y_pred_b: Predictions from method B
        metric_fn: Function to compute metric
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed
        
    Returns:
        TestResult with significance of difference
    """
    y_true = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)
    n_samples = len(y_true)
    
    rng = np.random.RandomState(random_state)
    
    # Point estimates
    metric_a = metric_fn(y_true, y_pred_a)
    metric_b = metric_fn(y_true, y_pred_b)
    observed_diff = metric_b - metric_a
    
    # Bootstrap the difference
    diff_values = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_y_true = y_true[indices]
        boot_y_pred_a = y_pred_a[indices]
        boot_y_pred_b = y_pred_b[indices]
        
        try:
            boot_metric_a = metric_fn(boot_y_true, boot_y_pred_a)
            boot_metric_b = metric_fn(boot_y_true, boot_y_pred_b)
            diff_values.append(boot_metric_b - boot_metric_a)
        except Exception:
            continue
    
    if not diff_values:
        return TestResult(
            test_name="Bootstrap comparison",
            statistic=observed_diff,
            p_value=1.0,
            significant=False,
            interpretation="Bootstrap sampling failed"
        )
    
    diff_values = np.array(diff_values)
    
    # Two-sided p-value: proportion of bootstrap diffs on opposite side of 0
    if observed_diff >= 0:
        p_value = np.mean(diff_values <= 0) * 2
    else:
        p_value = np.mean(diff_values >= 0) * 2
    p_value = min(p_value, 1.0)
    
    # Confidence interval on difference
    ci_lower = np.percentile(diff_values, 2.5)
    ci_upper = np.percentile(diff_values, 97.5)
    
    return TestResult(
        test_name="Bootstrap comparison",
        statistic=observed_diff,
        p_value=p_value,
        significant=p_value < SIGNIFICANCE_LEVEL,
        confidence_interval=(ci_lower, ci_upper),
        effect_size=observed_diff / (np.std(diff_values) + 1e-10),
        interpretation=f"Difference: {observed_diff:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
    )


# =============================================================================
# EFFECT SIZE
# =============================================================================

def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        group_a: Values from group A
        group_b: Values from group B
        
    Returns:
        Cohen's d (positive if B > A)
    """
    group_a = np.array(group_a)
    group_b = np.array(group_b)
    
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    
    # Pooled standard deviation
    n_a = len(group_a)
    n_b = len(group_b)
    
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (mean_b - mean_a) / pooled_std


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Follows standard conventions:
    - |d| < 0.2: negligible
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium
    - |d| ≥ 0.8: large
    """
    abs_d = abs(d)
    direction = "higher" if d > 0 else "lower"
    
    if abs_d < 0.2:
        return f"Negligible effect ({direction})"
    elif abs_d < 0.5:
        return f"Small effect ({direction})"
    elif abs_d < 0.8:
        return f"Medium effect ({direction})"
    else:
        return f"Large effect ({direction})"


# =============================================================================
# MULTIPLE TESTING CORRECTION
# =============================================================================

def bonferroni_correction(p_values: List[float], alpha: float = SIGNIFICANCE_LEVEL) -> Tuple[List[float], List[bool]]:
    """
    Apply Bonferroni correction for multiple testing.
    
    Args:
        p_values: List of p-values
        alpha: Significance level
        
    Returns:
        Tuple of (adjusted_p_values, is_significant)
    """
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    
    adjusted_p = [min(p * n_tests, 1.0) for p in p_values]
    is_significant = [p < adjusted_alpha for p in p_values]
    
    return adjusted_p, is_significant


def fdr_correction(p_values: List[float], alpha: float = SIGNIFICANCE_LEVEL) -> Tuple[List[float], List[bool]]:
    """
    Apply Benjamini-Hochberg FDR correction.
    
    Args:
        p_values: List of p-values
        alpha: Target FDR
        
    Returns:
        Tuple of (adjusted_p_values, is_significant)
    """
    n_tests = len(p_values)
    
    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # Compute adjusted p-values
    adjusted_p = np.zeros(n_tests)
    for i, idx in enumerate(sorted_indices):
        rank = i + 1
        adjusted_p[idx] = sorted_p[i] * n_tests / rank
    
    # Make monotonic (cumulative minimum from right)
    adjusted_p = np.minimum.accumulate(adjusted_p[::-1])[::-1]
    adjusted_p = np.minimum(adjusted_p, 1.0)
    
    is_significant = (adjusted_p < alpha).tolist()
    
    return adjusted_p.tolist(), is_significant


# =============================================================================
# COMPARISON RUNNERS
# =============================================================================

def compare_models(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_prob_a: np.ndarray = None,
    y_prob_b: np.ndarray = None,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    cv_values_a: np.ndarray = None,
    cv_values_b: np.ndarray = None
) -> ComparisonResult:
    """
    Comprehensive comparison of two models.
    
    Runs McNemar's test, bootstrap comparison, and optional paired t-test.
    
    Args:
        y_true: Ground truth labels
        y_pred_a: Predictions from model A
        y_pred_b: Predictions from model B
        y_prob_a: Probabilities from model A (optional)
        y_prob_b: Probabilities from model B (optional)
        model_a_name: Name of model A
        model_b_name: Name of model B
        cv_values_a: Per-fold accuracies for model A (for t-test)
        cv_values_b: Per-fold accuracies for model B (for t-test)
        
    Returns:
        ComparisonResult with all test results
    """
    from sklearn.metrics import accuracy_score
    
    y_true = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)
    
    # Compute accuracies
    acc_a = accuracy_score(y_true, y_pred_a)
    acc_b = accuracy_score(y_true, y_pred_b)
    
    test_results = []
    
    # McNemar's test
    mcnemar_result = mcnemar_test(y_true, y_pred_a, y_pred_b)
    test_results.append(mcnemar_result)
    
    # Bootstrap comparison on accuracy
    def accuracy_fn(yt, yp):
        return accuracy_score(yt, yp)
    
    bootstrap_result = bootstrap_metric_comparison(
        y_true, y_pred_a, y_pred_b, accuracy_fn
    )
    test_results.append(bootstrap_result)
    
    # Paired t-test if CV values provided
    if cv_values_a is not None and cv_values_b is not None:
        ttest_result = paired_ttest(cv_values_a, cv_values_b)
        test_results.append(ttest_result)
    
    return ComparisonResult(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        metric_name="accuracy",
        model_a_value=acc_a,
        model_b_value=acc_b,
        difference=acc_b - acc_a,
        test_results=test_results
    )


def run_all_pairwise_comparisons(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    baseline_name: str = None
) -> Dict[str, ComparisonResult]:
    """
    Run pairwise comparisons between all models or against a baseline.
    
    Args:
        y_true: Ground truth labels
        predictions_dict: Dict mapping model names to predictions
        baseline_name: If provided, compare all models against this baseline
        
    Returns:
        Dict mapping comparison names to ComparisonResults
    """
    results = {}
    model_names = list(predictions_dict.keys())
    
    if baseline_name:
        # Compare all against baseline
        if baseline_name not in predictions_dict:
            raise ValueError(f"Baseline {baseline_name} not in predictions")
        
        baseline_pred = predictions_dict[baseline_name]
        for name, pred in predictions_dict.items():
            if name == baseline_name:
                continue
            
            comparison = compare_models(
                y_true, baseline_pred, pred,
                model_a_name=baseline_name,
                model_b_name=name
            )
            results[f"{name}_vs_{baseline_name}"] = comparison
    else:
        # All pairwise comparisons
        for i, name_a in enumerate(model_names):
            for name_b in model_names[i+1:]:
                comparison = compare_models(
                    y_true,
                    predictions_dict[name_a],
                    predictions_dict[name_b],
                    model_a_name=name_a,
                    model_b_name=name_b
                )
                results[f"{name_b}_vs_{name_a}"] = comparison
    
    return results


# =============================================================================
# SUMMARY REPORT GENERATION
# =============================================================================

def generate_comparison_report(
    comparisons: Dict[str, ComparisonResult],
    apply_correction: str = "bonferroni"
) -> str:
    """
    Generate a formatted report of model comparisons.
    
    Args:
        comparisons: Dict of comparison results
        apply_correction: "bonferroni", "fdr", or None
        
    Returns:
        Formatted string report
    """
    lines = ["=" * 60]
    lines.append("Model Comparison Report")
    lines.append("=" * 60)
    
    # Collect all p-values for correction
    all_p_values = []
    comparison_names = []
    for name, comp in comparisons.items():
        for test in comp.test_results:
            all_p_values.append(test.p_value)
            comparison_names.append(f"{name}_{test.test_name}")
    
    # Apply correction
    if apply_correction == "bonferroni" and all_p_values:
        adjusted_p, _ = bonferroni_correction(all_p_values)
        lines.append(f"\nMultiple testing correction: Bonferroni (n={len(all_p_values)})")
    elif apply_correction == "fdr" and all_p_values:
        adjusted_p, _ = fdr_correction(all_p_values)
        lines.append(f"\nMultiple testing correction: FDR (n={len(all_p_values)})")
    else:
        adjusted_p = all_p_values
        lines.append("\nNo multiple testing correction applied")
    
    lines.append("-" * 60)
    
    p_idx = 0
    for name, comp in comparisons.items():
        lines.append(f"\n{comp.model_b_name} vs {comp.model_a_name}:")
        lines.append(f"  {comp.metric_name}: {comp.model_a_value:.4f} → {comp.model_b_value:.4f} "
                    f"(Δ = {comp.difference:+.4f})")
        
        for test in comp.test_results:
            adj_p = adjusted_p[p_idx] if p_idx < len(adjusted_p) else test.p_value
            sig = "✓" if adj_p < SIGNIFICANCE_LEVEL else "✗"
            marker = test.significance_marker()
            lines.append(f"    {test.test_name}: p={test.p_value:.4f} "
                        f"(adj={adj_p:.4f}) {marker} [{sig}]")
            if test.effect_size is not None:
                lines.append(f"      Effect size: {test.effect_size:.3f} "
                            f"({interpret_effect_size(test.effect_size)})")
            p_idx += 1
    
    lines.append("\n" + "=" * 60)
    lines.append(f"Significance level: α = {SIGNIFICANCE_LEVEL}")
    lines.append("Markers: * p<0.05, ** p<0.01, *** p<0.001")
    
    return "\n".join(lines)
