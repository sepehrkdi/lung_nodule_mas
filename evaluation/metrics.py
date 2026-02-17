"""
Evaluation metrics for lung nodule classification.

Supports accuracy, precision, recall, F1, specificity, ROC/AUC analysis.
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
    """Evaluation metrics for lung nodule classification."""
    
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
            ground_truth: List of true labels (0=benign, 1=malignant)
            predictions: List of predicted labels (0=benign, 1=malignant)
            probabilities: Optional list of malignancy probabilities
            
        Returns:
            Dictionary with all metrics
        """
        gt = np.array(ground_truth)
        pred = np.array(predictions)
        
        results = {}
        
        # Binary evaluation (primary)
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
        Evaluate multi-class prediction (legacy, kept for compatibility).
        
        NOTE: The primary classification is now binary (benign vs malignant).
        This method is retained for cases where Lung-RADS category
        information is still available.
        """
        # Overall accuracy
        accuracy = np.mean(ground_truth == predictions)
        
        # Per-class metrics — use classes present in data
        classes = sorted(set(ground_truth.tolist()) | set(predictions.tolist()))
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
        For clinical decisions, we care about:
        - Can we detect malignancy? (Sensitivity/Recall)
        - Are we creating false alarms? (Specificity)
        
        Ground truth mapping:
        - Benign: 0
        - Malignant: 1
        """
        # Create binary masks (exclude indeterminate = -1)
        mask = (ground_truth != -1) & (predictions != -1)
        
        if not np.any(mask):
            return {
                "error": "No valid samples for binary evaluation",
                "n_excluded": len(ground_truth)
            }
        
        gt_bin = ground_truth[mask]
        pred_bin = predictions[mask]
        
        # Compute metrics (1 = abnormal/positive, 0 = normal/negative)
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
        gt_bin = (ground_truth >= 1).astype(int)
        probs = probabilities
        
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
        
        # PR-AUC (better for imbalanced data)
        pr_auc = self._calculate_pr_auc(gt_bin, probs)
        
        return {
            "auc": float(auc),
            "pr_auc": float(pr_auc),
            "brier_score": float(brier),
            "log_loss": float(log_loss),
            "optimal_threshold": float(optimal_threshold)
        }
    
    def _calculate_pr_auc(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> float:
        """
        Calculate Area Under Precision-Recall Curve.
        
        EDUCATIONAL NOTE:
        PR-AUC is more informative than ROC-AUC for imbalanced datasets
        because it focuses on the positive class performance without
        being affected by the large number of true negatives.
        """
        # Sort by predicted probability (descending)
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        n_pos = np.sum(y_true == 1)
        if n_pos == 0:
            return 0.0
        
        # Calculate precision and recall at each threshold
        tp_cumsum = np.cumsum(y_true_sorted)
        fp_cumsum = np.cumsum(1 - y_true_sorted)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + self.epsilon)
        recall = tp_cumsum / n_pos
        
        # Add starting point
        precision = np.concatenate([[1], precision])
        recall = np.concatenate([[0], recall])
        
        # Trapezoidal integration
        pr_auc = np.trapz(precision, recall)
        
        return pr_auc
    
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
        n_classes: int = 2
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        EDUCATIONAL NOTE:
        Confusion matrix shows the distribution of predictions
        vs ground truth. Diagonal elements are correct predictions.
        
        For binary classification:
        Row = True class (0=benign, 1=malignant)
        Column = Predicted class (0=benign, 1=malignant)
        """
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        for gt, pred in zip(ground_truth, predictions):
            if 0 <= gt < n_classes and 0 <= pred < n_classes:
                cm[gt, pred] += 1
        
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
            lines.append("           Predicted")
            lines.append("True   Benign  Malignant")
            row_labels = ["Benign", "Malignant"]
            for i, row in enumerate(cm):
                label = row_labels[i] if i < len(row_labels) else str(i)
                row_str = f"  {label:10}  " + "  ".join(f"{v:3d}" for v in row)
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
    
    return metrics


# =============================================================================
# NLP-SPECIFIC EVALUATION METRICS
# =============================================================================

@dataclass
class EntityMatch:
    """Represents a match between predicted and gold entity."""
    predicted_text: str
    predicted_start: int
    predicted_end: int
    gold_text: str
    gold_start: int
    gold_end: int
    match_type: str  # "exact", "partial", "no_match"
    
    
class NLPMetrics:
    """Evaluation metrics for NLP entity extraction."""
    
    def __init__(self, partial_credit: float = 0.5):
        """
        Initialize NLP metrics.
        
        Args:
            partial_credit: Credit for partial span matches (0-1)
        """
        self.partial_credit = partial_credit
    
    def entity_evaluation(
        self,
        predicted_entities: List[Tuple[str, int, int]],
        gold_entities: List[Tuple[str, int, int]],
        allow_partial: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate entity extraction at entity level.
        
        Args:
            predicted_entities: List of (text, start, end) tuples
            gold_entities: List of (text, start, end) tuples
            allow_partial: If True, give partial credit for overlapping spans
            
        Returns:
            Dict with precision, recall, f1
        """
        if not gold_entities:
            return {"precision": 1.0 if not predicted_entities else 0.0,
                    "recall": 1.0, "f1": 1.0 if not predicted_entities else 0.0}
        
        if not predicted_entities:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Match predictions to gold
        matched_gold = set()
        true_positives = 0.0
        
        for pred_text, pred_start, pred_end in predicted_entities:
            best_match_score = 0.0
            best_gold_idx = None
            
            for i, (gold_text, gold_start, gold_end) in enumerate(gold_entities):
                if i in matched_gold:
                    continue
                
                # Check for overlap
                overlap_start = max(pred_start, gold_start)
                overlap_end = min(pred_end, gold_end)
                overlap_len = max(0, overlap_end - overlap_start)
                
                if overlap_len > 0:
                    # Exact match
                    if pred_start == gold_start and pred_end == gold_end:
                        match_score = 1.0
                    # Partial match
                    elif allow_partial:
                        union_len = max(pred_end, gold_end) - min(pred_start, gold_start)
                        match_score = (overlap_len / union_len) * self.partial_credit
                    else:
                        match_score = 0.0
                    
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_gold_idx = i
            
            if best_gold_idx is not None:
                matched_gold.add(best_gold_idx)
                true_positives += best_match_score
        
        precision = true_positives / len(predicted_entities)
        recall = true_positives / len(gold_entities)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": len(predicted_entities) - true_positives,
            "false_negatives": len(gold_entities) - len(matched_gold)
        }
    
    def cohens_kappa(
        self,
        labels1: List[str],
        labels2: List[str]
    ) -> float:
        """
        Calculate Cohen's Kappa for inter-annotator agreement.
        
        EDUCATIONAL NOTE:
        Cohen's Kappa measures agreement beyond chance:
        - κ = 1: Perfect agreement
        - κ = 0: Agreement expected by chance
        - κ < 0: Less agreement than chance
        
        Interpretation:
        - κ > 0.8: Excellent agreement
        - 0.6 < κ < 0.8: Substantial
        - 0.4 < κ < 0.6: Moderate
        - κ < 0.4: Fair to poor
        
        Args:
            labels1: Labels from annotator 1
            labels2: Labels from annotator 2
            
        Returns:
            Cohen's Kappa coefficient
        """
        if len(labels1) != len(labels2):
            raise ValueError("Label lists must have same length")
        
        if len(labels1) == 0:
            return 1.0  # Perfect agreement on nothing
        
        # Get all unique labels
        all_labels = list(set(labels1) | set(labels2))
        n = len(labels1)
        
        # Build confusion matrix
        matrix = {}
        for l in all_labels:
            matrix[l] = {l2: 0 for l2 in all_labels}
        
        for l1, l2 in zip(labels1, labels2):
            matrix[l1][l2] += 1
        
        # Calculate observed agreement (P_o)
        p_observed = sum(matrix[l][l] for l in all_labels) / n
        
        # Calculate expected agreement (P_e)
        p_expected = 0.0
        for l in all_labels:
            # Proportion labeled l by annotator 1
            p1 = sum(matrix[l].values()) / n
            # Proportion labeled l by annotator 2
            p2 = sum(matrix[l2][l] for l2 in all_labels) / n
            p_expected += p1 * p2
        
        # Kappa = (P_o - P_e) / (1 - P_e)
        if p_expected == 1.0:
            return 1.0
        
        kappa = (p_observed - p_expected) / (1.0 - p_expected)
        return kappa
    
    def certainty_evaluation(
        self,
        predicted_certainties: List[str],
        gold_certainties: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate certainty label assignment (affirmed/negated/uncertain).
        
        Args:
            predicted_certainties: Predicted certainty labels
            gold_certainties: Gold standard certainty labels
            
        Returns:
            Dict with accuracy, per-class metrics, and Cohen's Kappa
        """
        if len(predicted_certainties) != len(gold_certainties):
            raise ValueError("Lists must have same length")
        
        if len(predicted_certainties) == 0:
            return {"accuracy": 0.0, "kappa": 0.0, "per_class": {}}
        
        # Overall accuracy
        correct = sum(1 for p, g in zip(predicted_certainties, gold_certainties) if p == g)
        accuracy = correct / len(predicted_certainties)
        
        # Cohen's Kappa
        kappa = self.cohens_kappa(predicted_certainties, gold_certainties)
        
        # Per-class metrics
        classes = ["affirmed", "negated", "uncertain"]
        per_class = {}
        
        for cls in classes:
            tp = sum(1 for p, g in zip(predicted_certainties, gold_certainties) 
                    if p == cls and g == cls)
            fp = sum(1 for p, g in zip(predicted_certainties, gold_certainties) 
                    if p == cls and g != cls)
            fn = sum(1 for p, g in zip(predicted_certainties, gold_certainties) 
                    if p != cls and g == cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class[cls] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": sum(1 for g in gold_certainties if g == cls)
            }
        
        return {
            "accuracy": accuracy,
            "kappa": kappa,
            "kappa_interpretation": self._interpret_kappa(kappa),
            "per_class": per_class
        }
    
    def _interpret_kappa(self, kappa: float) -> str:
        """Interpret Kappa value."""
        if kappa >= 0.8:
            return "excellent"
        elif kappa >= 0.6:
            return "substantial"
        elif kappa >= 0.4:
            return "moderate"
        elif kappa >= 0.2:
            return "fair"
        else:
            return "poor"
    
    def generate_nlp_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable NLP evaluation report."""
        lines = [
            "=" * 60,
            "NLP EVALUATION REPORT",
            "=" * 60,
            ""
        ]
        
        if "entity_metrics" in results:
            em = results["entity_metrics"]
            lines.extend([
                "ENTITY EXTRACTION:",
                f"  Precision: {em.get('precision', 0):.3f}",
                f"  Recall:    {em.get('recall', 0):.3f}",
                f"  F1 Score:  {em.get('f1', 0):.3f}",
                ""
            ])
        
        if "certainty_metrics" in results:
            cm = results["certainty_metrics"]
            lines.extend([
                "CERTAINTY DETECTION:",
                f"  Accuracy:  {cm.get('accuracy', 0):.3f}",
                f"  Kappa:     {cm.get('kappa', 0):.3f} ({cm.get('kappa_interpretation', 'N/A')})",
                ""
            ])
            
            if "per_class" in cm:
                lines.append("  Per-Class F1:")
                for cls, metrics in cm["per_class"].items():
                    lines.append(f"    {cls:12}: {metrics['f1']:.3f} (n={metrics['support']})")
        
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
    
    # Synthetic ground truth and predictions (binary: 0=benign, 1=malignant)
    np.random.seed(42)
    n_samples = 100
    
    # Generate binary ground truth
    ground_truth = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # Generate predictions with some noise
    predictions = ground_truth.copy()
    noise_indices = np.random.choice(n_samples, size=15, replace=False)
    predictions[noise_indices] = 1 - predictions[noise_indices]  # Flip
    
    # Generate probabilities
    probabilities = predictions.astype(float) + np.random.uniform(-0.3, 0.3, n_samples)
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
