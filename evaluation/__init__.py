"""
Evaluation Module
=================

Metrics and visualization for model and system evaluation.

Provides:
- Binary classification metrics (benign vs malignant)
- Agent interaction analysis

EDUCATIONAL PURPOSE:
- Classification Metrics: Accuracy, Precision, Recall, F1
- ROC/AUC Analysis: Probability-based evaluation
- Confusion Matrix: Binary performance visualization
"""

from .metrics import (
    EvaluationMetrics,
    ClassificationMetrics,
    evaluate_results,
)

__all__ = [
    'EvaluationMetrics',
    'ClassificationMetrics',
    'evaluate_results',
]

