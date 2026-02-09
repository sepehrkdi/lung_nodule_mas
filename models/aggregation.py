"""
Aggregation Strategies for Multi-Image Cases

This module implements multiple strategies for combining predictions
from individual images (e.g., PA and Lateral chest X-ray views) into
a single aggregated prediction.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class AggregationStrategy:
    """Base class for aggregation strategies."""

    @staticmethod
    def aggregate(
        probabilities: List[float],
        classes: List[int],
        image_metadata: List[Dict[str, Any]],
        weights: Optional[List[float]] = None
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Aggregate predictions from multiple images.

        Args:
            probabilities: List of malignancy probabilities (one per image)
            classes: List of predicted classes (one per image)
            image_metadata: List of metadata dicts for each image
            weights: Optional list of weights (for weighted strategies)

        Returns:
            Tuple of (aggregated_probability, aggregated_class, details_dict)
        """
        raise NotImplementedError


def class_from_probability(probability: float) -> int:
    """
    Convert probability to Lung-RADS class (1-5).

    Args:
        probability: Malignancy probability [0, 1]

    Returns:
        Class from 1 (benign) to 5 (highly suspicious)
    """
    if probability < 0.2:
        return 1  # Highly Unlikely
    elif probability < 0.4:
        return 2  # Moderately Unlikely
    elif probability < 0.6:
        return 3  # Indeterminate
    elif probability < 0.8:
        return 4  # Moderately Suspicious
    else:
        return 5  # Highly Suspicious


class WeightedAverageAggregation(AggregationStrategy):
    """
    Weighted average of probabilities with view-based weighting.

    This is the RECOMMENDED strategy as it balances information from
    all views while giving more weight to primary views (e.g., PA).

    View Weights:
    - PA (Posterior-Anterior): 1.0 - Primary frontal view
    - AP (Anterior-Posterior): 0.9 - Alternative frontal view
    - Lateral: 0.8 - Secondary view for depth assessment
    - Frontal: 0.9 - General frontal view
    - Unknown: 0.5 - Cannot classify view type
    """

    VIEW_WEIGHTS = {
        "PA": 1.0,         # Primary view - full weight
        "AP": 0.9,         # Frontal alternative
        "Lateral": 0.8,    # Secondary view (important for depth)
        "Frontal": 0.9,    # General frontal view
        "Unknown": 0.5     # Unknown view - lowest weight
    }

    @staticmethod
    def aggregate(
        probabilities: List[float],
        classes: List[int],
        image_metadata: List[Dict[str, Any]],
        weights: Optional[List[float]] = None
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Compute weighted average of probabilities.

        Args:
            probabilities: List of probabilities from each image
            classes: List of predicted classes (not used, derived from weighted prob)
            image_metadata: Metadata including view_type for weighting
            weights: Optional custom weights (overrides view-based weights)

        Returns:
            (weighted_probability, derived_class, details_dict)
        """
        if not probabilities:
            return 0.5, 3, {"method": "weighted_average", "error": "no probabilities"}

        # Derive weights from view types if not provided
        if weights is None:
            weights = [
                WeightedAverageAggregation.VIEW_WEIGHTS.get(
                    meta.get("view_type", "Unknown"), 0.5
                )
                for meta in image_metadata
            ]

        # Compute weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1.0  # Avoid division by zero

        weighted_prob = sum(p * w for p, w in zip(probabilities, weights)) / total_weight

        # Derive class from weighted probability
        agg_class = class_from_probability(weighted_prob)

        # Details for debugging/analysis
        details = {
            "method": "weighted_average",
            "weights": weights,
            "individual_probs": probabilities,
            "weighted_prob": weighted_prob,
            "view_types": [m.get("view_type", "Unknown") for m in image_metadata]
        }

        return float(weighted_prob), agg_class, details


class MajorityVoteAggregation(AggregationStrategy):
    """
    Majority voting across views.

    Takes the class that appears most frequently across all views.
    The aggregated probability is the average of probabilities from
    images that voted for the majority class.

    This strategy is useful when you want each view to have equal say,
    and you trust the most common prediction.
    """

    @staticmethod
    def aggregate(
        probabilities: List[float],
        classes: List[int],
        image_metadata: List[Dict[str, Any]],
        weights: Optional[List[float]] = None
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Perform majority voting on predicted classes.

        Args:
            probabilities: Probabilities from each image
            classes: Predicted classes from each image
            image_metadata: Metadata for context
            weights: Optional weights for weighted voting

        Returns:
            (averaged_probability, majority_class, details_dict)
        """
        if not classes:
            return 0.5, 3, {"method": "majority_vote", "error": "no classes"}

        # Count class occurrences (optionally weighted)
        if weights is not None:
            # Weighted voting
            class_scores = {}
            for cls, weight in zip(classes, weights):
                class_scores[cls] = class_scores.get(cls, 0) + weight
            majority_class = max(class_scores, key=class_scores.get)
        else:
            # Simple majority
            class_counts = Counter(classes)
            majority_class = class_counts.most_common(1)[0][0]

        # Average probabilities of images that voted for majority class
        majority_probs = [
            prob for prob, cls in zip(probabilities, classes)
            if cls == majority_class
        ]

        if majority_probs:
            agg_prob = np.mean(majority_probs)
        else:
            agg_prob = 0.5  # Fallback

        # Details
        details = {
            "method": "majority_vote",
            "class_counts": dict(Counter(classes)),
            "majority_class": majority_class,
            "majority_support": len(majority_probs),
            "individual_classes": classes
        }

        return float(agg_prob), majority_class, details


class MaxPoolingAggregation(AggregationStrategy):
    """
    Take maximum probability across views (most conservative approach).

    This strategy selects the view with the highest malignancy probability
    and uses both its probability and class as the final prediction.

    MEDICAL RATIONALE:
    In clinical practice, radiologists often follow the "most suspicious finding"
    principle - if any view shows concerning features, that drives the diagnosis.
    This is a conservative approach that minimizes false negatives.
    """

    @staticmethod
    def aggregate(
        probabilities: List[float],
        classes: List[int],
        image_metadata: List[Dict[str, Any]],
        weights: Optional[List[float]] = None
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Select the view with maximum probability.

        Args:
            probabilities: Probabilities from each image
            classes: Predicted classes from each image
            image_metadata: Metadata for identifying the selected view
            weights: Not used (max pooling doesn't weight)

        Returns:
            (max_probability, corresponding_class, details_dict)
        """
        if not probabilities:
            return 0.5, 3, {"method": "max_pooling", "error": "no probabilities"}

        # Find index of maximum probability
        max_idx = int(np.argmax(probabilities))
        max_prob = probabilities[max_idx]
        max_class = classes[max_idx]

        # Identify which view was selected
        selected_view = image_metadata[max_idx].get("view_type", "Unknown") if max_idx < len(image_metadata) else "Unknown"
        selected_image_id = image_metadata[max_idx].get("image_id", "") if max_idx < len(image_metadata) else ""

        # Details
        details = {
            "method": "max_pooling",
            "selected_view": selected_view,
            "selected_image_id": selected_image_id,
            "selected_idx": max_idx,
            "all_probabilities": probabilities
        }

        return float(max_prob), max_class, details


class MinPoolingAggregation(AggregationStrategy):
    """
    Take minimum probability across views (most optimistic approach).

    This strategy selects the view with the LOWEST malignancy probability.

    MEDICAL RATIONALE:
    This is less common clinically, but can be useful in screening scenarios
    where you want to avoid over-treatment. If any view looks reassuring,
    this approach gives it priority.

    USE WITH CAUTION: This strategy can miss malignancies.
    """

    @staticmethod
    def aggregate(
        probabilities: List[float],
        classes: List[int],
        image_metadata: List[Dict[str, Any]],
        weights: Optional[List[float]] = None
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Select the view with minimum probability.

        Args:
            probabilities: Probabilities from each image
            classes: Predicted classes from each image
            image_metadata: Metadata for identifying the selected view
            weights: Not used

        Returns:
            (min_probability, corresponding_class, details_dict)
        """
        if not probabilities:
            return 0.5, 3, {"method": "min_pooling", "error": "no probabilities"}

        # Find index of minimum probability
        min_idx = int(np.argmin(probabilities))
        min_prob = probabilities[min_idx]
        min_class = classes[min_idx]

        # Identify which view was selected
        selected_view = image_metadata[min_idx].get("view_type", "Unknown") if min_idx < len(image_metadata) else "Unknown"
        selected_image_id = image_metadata[min_idx].get("image_id", "") if min_idx < len(image_metadata) else ""

        # Details
        details = {
            "method": "min_pooling",
            "selected_view": selected_view,
            "selected_image_id": selected_image_id,
            "selected_idx": min_idx,
            "all_probabilities": probabilities
        }

        return float(min_prob), min_class, details


# Factory for selecting strategy
AGGREGATION_STRATEGIES = {
    "weighted_average": WeightedAverageAggregation,
    "majority_vote": MajorityVoteAggregation,
    "max_pooling": MaxPoolingAggregation,
    "min_pooling": MinPoolingAggregation,
}


def get_aggregator(strategy: str = "weighted_average") -> AggregationStrategy:
    """
    Get an aggregation strategy by name.

    Args:
        strategy: Name of strategy ("weighted_average", "majority_vote",
                 "max_pooling", or "min_pooling")

    Returns:
        AggregationStrategy class

    Raises:
        ValueError: If strategy name is unknown
    """
    strategy = strategy.lower()
    if strategy not in AGGREGATION_STRATEGIES:
        raise ValueError(
            f"Unknown aggregation strategy: {strategy}. "
            f"Available: {list(AGGREGATION_STRATEGIES.keys())}"
        )

    return AGGREGATION_STRATEGIES[strategy]


def main():
    """Test aggregation strategies."""
    print("=== Aggregation Strategies Test ===\n")

    # Test data: 3 images with different predictions
    probabilities = [0.3, 0.7, 0.5]
    classes = [2, 4, 3]
    metadata = [
        {"view_type": "PA", "image_id": "img1"},
        {"view_type": "Lateral", "image_id": "img2"},
        {"view_type": "Unknown", "image_id": "img3"}
    ]

    print("Input:")
    print(f"  Probabilities: {probabilities}")
    print(f"  Classes: {classes}")
    print(f"  Views: {[m['view_type'] for m in metadata]}\n")

    # Test each strategy
    for strategy_name in AGGREGATION_STRATEGIES.keys():
        print(f"--- {strategy_name.upper().replace('_', ' ')} ---")
        strategy = get_aggregator(strategy_name)
        agg_prob, agg_class, details = strategy.aggregate(
            probabilities, classes, metadata
        )

        print(f"  Aggregated Probability: {agg_prob:.3f}")
        print(f"  Aggregated Class: {agg_class}")

        if "weights" in details:
            print(f"  Weights: {[f'{w:.2f}' for w in details['weights']]}")
        if "majority_class" in details:
            print(f"  Majority Class: {details['majority_class']} ({details['majority_support']} votes)")
        if "selected_view" in details:
            print(f"  Selected View: {details['selected_view']}")

        print()


if __name__ == "__main__":
    main()
