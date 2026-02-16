"""
Cross-Validation Framework
==========================

This module implements stratified k-fold cross-validation for
rigorous evaluation of the lung nodule classification system.

Features:
1. Stratified splits maintaining class balance
2. Per-fold metric collection
3. Mean ± standard deviation reporting
4. Reproducible splits using fixed random seed
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable, Iterator
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path

from config import RANDOM_SEED, DEFAULT_CV_FOLDS, CV_RESULTS_DIR, ensure_directories

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FoldResult:
    """Results from a single CV fold."""
    fold_idx: int
    train_indices: List[int]
    test_indices: List[int]
    metrics: Dict[str, float]
    predictions: List[int] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)
    ground_truth: List[int] = field(default_factory=list)


@dataclass
class CVResult:
    """Aggregated cross-validation results."""
    n_folds: int
    fold_results: List[FoldResult]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    ci_metrics: Dict[str, Tuple[float, float]]  # 95% CI
    all_predictions: List[int] = field(default_factory=list)
    all_probabilities: List[float] = field(default_factory=list)
    all_ground_truth: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_folds": self.n_folds,
            "mean_metrics": self.mean_metrics,
            "std_metrics": self.std_metrics,
            "ci_metrics": {k: list(v) for k, v in self.ci_metrics.items()},
            "per_fold_metrics": [
                {"fold": fr.fold_idx, "metrics": fr.metrics}
                for fr in self.fold_results
            ]
        }
    
    def summary_string(self) -> str:
        """Generate summary string with mean ± std."""
        lines = [f"Cross-Validation Results ({self.n_folds}-fold):", "=" * 50]
        for metric, mean_val in self.mean_metrics.items():
            std_val = self.std_metrics.get(metric, 0)
            ci = self.ci_metrics.get(metric, (mean_val, mean_val))
            lines.append(
                f"  {metric:20s}: {mean_val:.4f} ± {std_val:.4f} "
                f"(95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])"
            )
        return "\n".join(lines)


# =============================================================================
# STRATIFIED K-FOLD SPLITTER
# =============================================================================

class StratifiedKFoldSplitter:
    """
    Stratified K-Fold cross-validator.
    
    Provides train/test indices for K-fold cross-validation,
    ensuring each fold maintains the original class distribution.
    """
    
    def __init__(self, n_splits: int = DEFAULT_CV_FOLDS, shuffle: bool = True, 
                 random_state: int = RANDOM_SEED):
        """
        Initialize splitter.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle before splitting
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X: np.ndarray, y: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into train and test sets.
        
        Args:
            X: Features (or any array of same length as y)
            y: Target labels
            
        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        y = np.array(y)
        n_samples = len(y)
        indices = np.arange(n_samples)
        
        # Shuffle if requested
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)
        
        # Separate indices by class
        classes = np.unique(y)
        class_indices = {c: indices[y[indices] == c] for c in classes}
        
        # Create folds with stratified sampling
        folds = [[] for _ in range(self.n_splits)]
        
        for c, c_indices in class_indices.items():
            # Split this class's indices across folds
            fold_sizes = np.full(self.n_splits, len(c_indices) // self.n_splits, dtype=int)
            fold_sizes[:len(c_indices) % self.n_splits] += 1
            
            current = 0
            for fold_idx, fold_size in enumerate(fold_sizes):
                folds[fold_idx].extend(c_indices[current:current + fold_size])
                current += fold_size
        
        # Yield train/test splits
        for fold_idx in range(self.n_splits):
            test_indices = np.array(folds[fold_idx])
            train_indices = np.concatenate([
                np.array(folds[i]) for i in range(self.n_splits) if i != fold_idx
            ])
            yield train_indices, test_indices
    
    def get_n_splits(self) -> int:
        """Return the number of splits."""
        return self.n_splits


# =============================================================================
# CROSS-VALIDATION EVALUATOR
# =============================================================================

class CrossValidationEvaluator:
    """
    Cross-validation evaluator for the lung nodule classification system.
    
    Supports:
    - Stratified K-fold splitting
    - Per-fold metric collection
    - Aggregated statistics (mean, std, CI)
    - Result persistence
    """
    
    def __init__(
        self,
        n_folds: int = DEFAULT_CV_FOLDS,
        random_state: int = RANDOM_SEED,
        save_results: bool = True
    ):
        """
        Initialize evaluator.
        
        Args:
            n_folds: Number of CV folds
            random_state: Random seed for reproducibility
            save_results: Whether to save results to disk
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.save_results = save_results
        self.splitter = StratifiedKFoldSplitter(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state
        )
    
    def evaluate(
        self,
        case_ids: List[str],
        y_true: np.ndarray,
        predict_fn: Callable[[List[str]], Tuple[np.ndarray, np.ndarray]],
        metrics_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], Dict[str, float]],
        experiment_name: str = "cv_experiment"
    ) -> CVResult:
        """
        Run cross-validation evaluation.
        
        Args:
            case_ids: List of case identifiers
            y_true: Ground truth labels
            predict_fn: Function(case_ids) -> (predictions, probabilities)
            metrics_fn: Function(y_true, y_pred, y_prob) -> metrics_dict
            experiment_name: Name for saving results
            
        Returns:
            CVResult with aggregated statistics
        """
        case_ids = np.array(case_ids)
        y_true = np.array(y_true)
        
        fold_results = []
        all_predictions = []
        all_probabilities = []
        all_ground_truth = []
        all_test_indices = []
        
        logger.info(f"Starting {self.n_folds}-fold cross-validation...")
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.splitter.split(case_ids, y_true)):
            logger.info(f"  Fold {fold_idx + 1}/{self.n_folds}: "
                       f"train={len(train_idx)}, test={len(test_idx)}")
            
            # Get test cases
            test_case_ids = case_ids[test_idx].tolist()
            test_y_true = y_true[test_idx]
            
            # Run prediction on test set
            try:
                predictions, probabilities = predict_fn(test_case_ids)
                predictions = np.array(predictions)
                probabilities = np.array(probabilities)
            except Exception as e:
                logger.error(f"Prediction failed for fold {fold_idx}: {e}")
                continue
            
            # Compute metrics
            metrics = metrics_fn(test_y_true, predictions, probabilities)
            
            # Store fold results
            fold_result = FoldResult(
                fold_idx=fold_idx,
                train_indices=train_idx.tolist(),
                test_indices=test_idx.tolist(),
                metrics=metrics,
                predictions=predictions.tolist(),
                probabilities=probabilities.tolist(),
                ground_truth=test_y_true.tolist()
            )
            fold_results.append(fold_result)
            
            # Collect for aggregate metrics
            all_predictions.extend(predictions.tolist())
            all_probabilities.extend(probabilities.tolist())
            all_ground_truth.extend(test_y_true.tolist())
            all_test_indices.extend(test_idx.tolist())
            
            logger.info(f"    Fold {fold_idx + 1} metrics: " + 
                       ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        
        if not fold_results:
            raise RuntimeError("All CV folds failed")
        
        # Compute aggregated statistics
        mean_metrics, std_metrics, ci_metrics = self._aggregate_metrics(fold_results)
        
        result = CVResult(
            n_folds=self.n_folds,
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            ci_metrics=ci_metrics,
            all_predictions=all_predictions,
            all_probabilities=all_probabilities,
            all_ground_truth=all_ground_truth
        )
        
        # Save results
        if self.save_results:
            self._save_results(result, experiment_name)
        
        logger.info(f"Cross-validation complete:\n{result.summary_string()}")
        
        return result
    
    def _aggregate_metrics(
        self, 
        fold_results: List[FoldResult]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Tuple[float, float]]]:
        """Compute mean, std, and 95% CI across folds."""
        # Collect all metric names
        all_metric_names = set()
        for fr in fold_results:
            all_metric_names.update(fr.metrics.keys())
        
        mean_metrics = {}
        std_metrics = {}
        ci_metrics = {}
        
        for metric_name in all_metric_names:
            values = [fr.metrics.get(metric_name, np.nan) for fr in fold_results]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                
                # 95% CI using t-distribution approximation
                n = len(values)
                if n > 1:
                    from scipy import stats
                    t_val = stats.t.ppf(0.975, n - 1)
                    margin = t_val * std_val / np.sqrt(n)
                    ci = (mean_val - margin, mean_val + margin)
                else:
                    ci = (mean_val, mean_val)
                
                mean_metrics[metric_name] = mean_val
                std_metrics[metric_name] = std_val
                ci_metrics[metric_name] = ci
        
        return mean_metrics, std_metrics, ci_metrics
    
    def _save_results(self, result: CVResult, experiment_name: str) -> None:
        """Save CV results to disk."""
        ensure_directories()
        
        output_path = CV_RESULTS_DIR / f"{experiment_name}.json"
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        summary_path = CV_RESULTS_DIR / f"{experiment_name}_summary.txt"
        with open(summary_path, "w") as f:
            f.write(result.summary_string())
        
        logger.info(f"CV results saved to {output_path}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_cv_metrics_fn(y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: np.ndarray) -> Dict[str, float]:
    """
    Default metrics function for CV evaluation.
    
    Returns comprehensive metrics including per-class measures.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, brier_score_loss
    )
    
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    metrics["precision_malignant"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics["recall_malignant"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics["f1_malignant"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Specificity (recall of benign class)
    metrics["specificity"] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    # Probability-based metrics (if probabilities are valid)
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.5
        
        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["pr_auc"] = 0.5
        
        metrics["brier_score"] = brier_score_loss(y_true, y_prob)
    
    return metrics


def run_simple_cv(
    y_true: np.ndarray,
    y_pred_fn: Callable[[np.ndarray], np.ndarray],
    y_prob_fn: Callable[[np.ndarray], np.ndarray] = None,
    n_folds: int = DEFAULT_CV_FOLDS,
    random_state: int = RANDOM_SEED
) -> CVResult:
    """
    Simplified CV runner for array-based predictions.
    
    Args:
        y_true: Ground truth labels
        y_pred_fn: Function to get predictions given indices
        y_prob_fn: Optional function to get probabilities given indices
        n_folds: Number of folds
        random_state: Random seed
        
    Returns:
        CVResult with aggregated statistics
    """
    y_true = np.array(y_true)
    n_samples = len(y_true)
    X_dummy = np.arange(n_samples)  # Just indices
    
    splitter = StratifiedKFoldSplitter(n_folds, shuffle=True, random_state=random_state)
    
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_dummy, y_true)):
        y_test = y_true[test_idx]
        y_pred = y_pred_fn(test_idx)
        y_prob = y_prob_fn(test_idx) if y_prob_fn else np.full(len(test_idx), 0.5)
        
        metrics = compute_cv_metrics_fn(y_test, y_pred, y_prob)
        
        fold_results.append(FoldResult(
            fold_idx=fold_idx,
            train_indices=train_idx.tolist(),
            test_indices=test_idx.tolist(),
            metrics=metrics,
            predictions=y_pred.tolist(),
            probabilities=y_prob.tolist(),
            ground_truth=y_test.tolist()
        ))
    
    # Aggregate
    mean_metrics = {}
    std_metrics = {}
    ci_metrics = {}
    
    all_metric_names = set()
    for fr in fold_results:
        all_metric_names.update(fr.metrics.keys())
    
    for metric_name in all_metric_names:
        values = [fr.metrics[metric_name] for fr in fold_results if metric_name in fr.metrics]
        if values:
            mean_metrics[metric_name] = np.mean(values)
            std_metrics[metric_name] = np.std(values, ddof=1) if len(values) > 1 else 0.0
            ci_metrics[metric_name] = (
                np.percentile(values, 2.5) if len(values) > 1 else values[0],
                np.percentile(values, 97.5) if len(values) > 1 else values[0]
            )
    
    return CVResult(
        n_folds=n_folds,
        fold_results=fold_results,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        ci_metrics=ci_metrics
    )
