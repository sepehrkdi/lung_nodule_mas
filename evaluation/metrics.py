"""
Evaluation Metrics for Lung Nodule Classification
==================================================

EDUCATIONAL PURPOSE:

This module demonstrates evaluation metrics for medical AI systems:

1. CLASSIFICATION METRICS:
   - Accuracy: Overall correctness
   - Precision: True positives / predicted positives
   - Recall (Sensitivity): True positives / actual positives
   - F1 Score: Harmonic mean of precision and recall
   - Specificity: True negatives / actual negatives

2. MULTI-CLASS EVALUATION:
   - 5-class accuracy (malignancy 1-5)
   - Confusion matrix
   - Per-class metrics

3. BINARY EVALUATION:
   - Benign (1-2) vs Malignant (4-5)
   - Indeterminate (3) excluded
   - Clinical relevance: detecting cancer

4. ROC ANALYSIS:
   - Receiver Operating Characteristic curve
   - Area Under Curve (AUC)
   - Optimal threshold selection

5. CALIBRATION:
   - Are probabilities well-calibrated?
   - Reliability diagram

WHY THESE METRICS MATTER FOR MEDICAL AI:

In medical diagnosis:
- False Negatives (missed cancers) can be fatal
- False Positives cause unnecessary procedures
- Balance depends on clinical context

Lung-RADS aims for:
- High Sensitivity (catch cancers)
- Acceptable Specificity (not too many false alarms)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "specificity": self.specificity
        }


class EvaluationMetrics:
    """
    Evaluation metrics for lung nodule classification.
    
    EDUCATIONAL PURPOSE:
    
    This class computes comprehensive evaluation metrics:
    
    1. 5-Class Evaluation:
       - Malignancy levels 1-5
       - Confusion matrix
       - Per-class accuracy
    
    2. Binary Evaluation:
       - Benign (1-2) vs Malignant (4-5)
       - Excludes indeterminate (3)
       - Clinical decision boundary
    
    3. Ranking Metrics:
       - ROC curve
       - AUC score
       - Probability calibration
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.epsilon = 1e-10  # Avoid division by zero
    
    def evaluate(
        self,
        ground_truth: List[int],
        predictions: List[int],
        probabilities: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.
        
        Args:
            ground_truth: List of true malignancy labels (1-5)
            predictions: List of predicted labels (1-5)
            probabilities: Optional list of malignancy probabilities
            
        Returns:
            Dictionary with all metrics
        """
        gt = np.array(ground_truth)
        pred = np.array(predictions)
        
        results = {}
        
        # 5-class evaluation
        results["five_class"] = self.five_class_evaluation(gt, pred)
        
        # Binary evaluation
        results["binary"] = self.binary_evaluation(gt, pred)
        
        # Probability-based metrics
        if probabilities:
            probs = np.array(probabilities)
            results["probability_metrics"] = self.probability_evaluation(gt, probs)
        
        # Confusion matrix
        results["confusion_matrix"] = self.confusion_matrix(gt, pred).tolist()
        
        return results
    
    def five_class_evaluation(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate 5-class malignancy prediction.
        
        EDUCATIONAL NOTE:
        Medical imaging often uses multi-level ratings.
        LIDC-IDRI uses 5 levels:
        1 = Highly Unlikely malignant
        2 = Moderately Unlikely
        3 = Indeterminate
        4 = Moderately Suspicious
        5 = Highly Suspicious
        """
        # Overall accuracy
        accuracy = np.mean(ground_truth == predictions)
        
        # Per-class metrics
        classes = [1, 2, 3, 4, 5]
        per_class = {}
        
        for c in classes:
            mask_true = ground_truth == c
            mask_pred = predictions == c
            
            # True positives, false positives, false negatives
            tp = np.sum(mask_true & mask_pred)
            fp = np.sum(~mask_true & mask_pred)
            fn = np.sum(mask_true & ~mask_pred)
            tn = np.sum(~mask_true & ~mask_pred)
            
            # Metrics
            precision = tp / (tp + fp + self.epsilon)
            recall = tp / (tp + fn + self.epsilon)
            f1 = 2 * precision * recall / (precision + recall + self.epsilon)
            
            per_class[c] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "support": int(np.sum(mask_true))
            }
        
        # Weighted averages
        weights = [per_class[c]["support"] for c in classes]
        total = sum(weights)
        
        weighted_precision = sum(
            per_class[c]["precision"] * per_class[c]["support"] 
            for c in classes
        ) / (total + self.epsilon)
        
        weighted_recall = sum(
            per_class[c]["recall"] * per_class[c]["support"] 
            for c in classes
        ) / (total + self.epsilon)
        
        weighted_f1 = sum(
            per_class[c]["f1_score"] * per_class[c]["support"] 
            for c in classes
        ) / (total + self.epsilon)
        
        # Adjacent accuracy (within 1 class)
        adjacent_correct = np.sum(np.abs(ground_truth - predictions) <= 1)
        adjacent_accuracy = adjacent_correct / len(ground_truth)
        
        return {
            "accuracy": float(accuracy),
            "adjacent_accuracy": float(adjacent_accuracy),
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "weighted_f1": float(weighted_f1),
            "per_class": per_class
        }
    
    def binary_evaluation(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate binary (benign vs malignant) prediction.
        
        EDUCATIONAL NOTE:
        For clinical decisions, we often care about:
        - Can we detect malignancy? (Sensitivity/Recall)
        - Are we creating false alarms? (Specificity)
        
        LIDC-IDRI mapping:
        - Benign: malignancy 1-2
        - Malignant: malignancy 4-5
        - Indeterminate (3): excluded
        """
        # Create binary masks (exclude class 3)
        mask = (ground_truth != 3) & (predictions != 3)
        
        if not np.any(mask):
            return {
                "error": "No valid samples for binary evaluation",
                "n_excluded": len(ground_truth)
            }
        
        gt_binary = ground_truth[mask]
        pred_binary = predictions[mask]
        
        # Convert to binary: benign (1-2) = 0, malignant (4-5) = 1
        gt_bin = (gt_binary >= 4).astype(int)
        pred_bin = (pred_binary >= 4).astype(int)
        
        # Compute metrics
        tp = np.sum((gt_bin == 1) & (pred_bin == 1))
        tn = np.sum((gt_bin == 0) & (pred_bin == 0))
        fp = np.sum((gt_bin == 0) & (pred_bin == 1))
        fn = np.sum((gt_bin == 1) & (pred_bin == 0))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn + self.epsilon)
        sensitivity = tp / (tp + fn + self.epsilon)  # Recall
        specificity = tn / (tn + fp + self.epsilon)
        precision = tp / (tp + fp + self.epsilon)
        npv = tn / (tn + fn + self.epsilon)  # Negative predictive value
        f1 = 2 * precision * sensitivity / (precision + sensitivity + self.epsilon)
        
        return {
            "accuracy": float(accuracy),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "npv": float(npv),
            "f1_score": float(f1),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "n_evaluated": int(np.sum(mask)),
            "n_excluded": int(np.sum(~mask))
        }
    
    def probability_evaluation(
        self,
        ground_truth: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate probability predictions.
        
        EDUCATIONAL NOTE:
        Beyond hard predictions, we want to know:
        - How well do probabilities rank cases? (AUC)
        - Are probabilities calibrated? (Brier score)
        """
        # Binary ground truth for ROC
        mask = ground_truth != 3
        gt_bin = (ground_truth[mask] >= 4).astype(int)
        probs = probabilities[mask]
        
        if len(gt_bin) < 2:
            return {"error": "Insufficient samples for probability evaluation"}
        
        # AUC calculation (manual implementation)
        auc = self._calculate_auc(gt_bin, probs)
        
        # Brier score (for probability calibration)
        brier = np.mean((probs - gt_bin) ** 2)
        
        # Log loss
        probs_clipped = np.clip(probs, 0.01, 0.99)
        log_loss = -np.mean(
            gt_bin * np.log(probs_clipped) + 
            (1 - gt_bin) * np.log(1 - probs_clipped)
        )
        
        # Find optimal threshold
        optimal_threshold = self._find_optimal_threshold(gt_bin, probs)
        
        return {
            "auc": float(auc),
            "brier_score": float(brier),
            "log_loss": float(log_loss),
            "optimal_threshold": float(optimal_threshold)
        }
    
    def _calculate_auc(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> float:
        """
        Calculate Area Under ROC Curve.
        
        EDUCATIONAL NOTE:
        AUC measures the probability that a randomly chosen
        positive example is ranked higher than a randomly
        chosen negative example.
        
        AUC = 0.5: Random classifier
        AUC = 1.0: Perfect classifier
        """
        # Sort by predicted probability
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Count positives and negatives
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        # Calculate AUC using trapezoidal rule
        tpr = np.cumsum(y_true_sorted) / n_pos
        fpr = np.cumsum(1 - y_true_sorted) / n_neg
        
        # Add origin
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])
        
        # Trapezoidal integration
        auc = np.trapz(tpr, fpr)
        
        return auc
    
    def _find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> float:
        """
        Find optimal classification threshold.
        
        Uses Youden's J statistic: J = Sensitivity + Specificity - 1
        """
        thresholds = np.linspace(0, 1, 100)
        best_j = -1
        best_threshold = 0.5
        
        for t in thresholds:
            pred = (y_scores >= t).astype(int)
            
            tp = np.sum((y_true == 1) & (pred == 1))
            tn = np.sum((y_true == 0) & (pred == 0))
            fp = np.sum((y_true == 0) & (pred == 1))
            fn = np.sum((y_true == 1) & (pred == 0))
            
            sens = tp / (tp + fn + self.epsilon)
            spec = tn / (tn + fp + self.epsilon)
            
            j = sens + spec - 1
            
            if j > best_j:
                best_j = j
                best_threshold = t
        
        return best_threshold
    
    def confusion_matrix(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        n_classes: int = 5
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        EDUCATIONAL NOTE:
        Confusion matrix shows the distribution of predictions
        vs ground truth. Diagonal elements are correct predictions.
        
        For 5-class:
        Row = True class (1-5)
        Column = Predicted class (1-5)
        """
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        for gt, pred in zip(ground_truth, predictions):
            if 1 <= gt <= n_classes and 1 <= pred <= n_classes:
                cm[gt - 1, pred - 1] += 1
        
        return cm
    
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate human-readable evaluation report.
        
        Args:
            metrics: Dictionary from evaluate()
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("LUNG NODULE CLASSIFICATION EVALUATION REPORT")
        lines.append("=" * 60)
        
        # 5-Class Results
        five = metrics.get("five_class", {})
        lines.append("\n5-CLASS EVALUATION (Malignancy 1-5):")
        lines.append("-" * 40)
        lines.append(f"Accuracy:          {five.get('accuracy', 0):.1%}")
        lines.append(f"Adjacent Accuracy: {five.get('adjacent_accuracy', 0):.1%}")
        lines.append(f"Weighted F1:       {five.get('weighted_f1', 0):.3f}")
        
        lines.append("\nPer-Class Performance:")
        for c in range(1, 6):
            pc = five.get("per_class", {}).get(c, {})
            lines.append(
                f"  Class {c}: P={pc.get('precision', 0):.2f} "
                f"R={pc.get('recall', 0):.2f} "
                f"F1={pc.get('f1_score', 0):.2f} "
                f"(n={pc.get('support', 0)})"
            )
        
        # Binary Results
        binary = metrics.get("binary", {})
        lines.append("\nBINARY EVALUATION (Benign vs Malignant):")
        lines.append("-" * 40)
        lines.append(f"Accuracy:    {binary.get('accuracy', 0):.1%}")
        lines.append(f"Sensitivity: {binary.get('sensitivity', 0):.1%} (Recall)")
        lines.append(f"Specificity: {binary.get('specificity', 0):.1%}")
        lines.append(f"Precision:   {binary.get('precision', 0):.1%}")
        lines.append(f"F1 Score:    {binary.get('f1_score', 0):.3f}")
        lines.append(f"Excluded (indeterminate): {binary.get('n_excluded', 0)}")
        
        # Probability Metrics
        prob = metrics.get("probability_metrics", {})
        if prob and "error" not in prob:
            lines.append("\nPROBABILITY METRICS:")
            lines.append("-" * 40)
            lines.append(f"AUC:               {prob.get('auc', 0):.3f}")
            lines.append(f"Brier Score:       {prob.get('brier_score', 0):.4f}")
            lines.append(f"Log Loss:          {prob.get('log_loss', 0):.4f}")
            lines.append(f"Optimal Threshold: {prob.get('optimal_threshold', 0.5):.2f}")
        
        # Confusion Matrix
        cm = metrics.get("confusion_matrix", [])
        if cm:
            lines.append("\nCONFUSION MATRIX:")
            lines.append("-" * 40)
            lines.append("      Predicted")
            lines.append("True   1    2    3    4    5")
            for i, row in enumerate(cm):
                row_str = f"  {i+1}  " + "  ".join(f"{v:3d}" for v in row)
                lines.append(row_str)
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)


def evaluate_results(results: List[Any]) -> Dict[str, Any]:
    """
    Evaluate a list of ProcessingResult objects.
    
    Args:
        results: List of ProcessingResult from main.py
        
    Returns:
        Evaluation metrics dictionary
    """
    evaluator = EvaluationMetrics()
    
    ground_truth = [r.ground_truth for r in results]
    predictions = [r.predicted_class for r in results]
    probabilities = [r.malignancy_probability for r in results]
    
    metrics = evaluator.evaluate(ground_truth, predictions, probabilities)
    
    # Print report
    report = evaluator.generate_report(metrics)
    print(report)
    
    return metrics


if __name__ == "__main__":
    # Demo with synthetic data
    print("=== Evaluation Metrics Demo ===\n")
    
    # Synthetic ground truth and predictions
    np.random.seed(42)
    n_samples = 100
    
    # Generate ground truth with class imbalance
    ground_truth = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.15, 0.2, 0.3, 0.2, 0.15])
    
    # Generate predictions with some noise
    predictions = ground_truth.copy()
    noise_indices = np.random.choice(n_samples, size=20, replace=False)
    predictions[noise_indices] = np.random.choice([1, 2, 3, 4, 5], size=20)
    
    # Generate probabilities
    probabilities = (predictions - 1) / 4 + np.random.uniform(-0.2, 0.2, n_samples)
    probabilities = np.clip(probabilities, 0.01, 0.99)
    
    # Evaluate
    evaluator = EvaluationMetrics()
    metrics = evaluator.evaluate(
        ground_truth.tolist(),
        predictions.tolist(),
        probabilities.tolist()
    )
    
    # Print report
    report = evaluator.generate_report(metrics)
    print(report)
