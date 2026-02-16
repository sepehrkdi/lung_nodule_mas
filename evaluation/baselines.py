"""
Baseline Predictors for Ablation Studies
=========================================

This module implements mandatory baselines for rigorous evaluation:

1. MajorityClassBaseline: Always predicts the majority class
2. RandomBaseline: Random predictions weighted by class distribution
3. SingleAgentBaseline: Wrapper to evaluate individual agents
4. UnweightedMajorityVote: Simple majority voting across agents
5. StaticWeightedAverage: Fixed weights without dynamic adjustment
6. SklearnVotingEquivalent: Soft-voting ensemble (sklearn-style)
7. PurePythonWeightedAverage: Full weighted consensus without Prolog

These baselines answer: "Does the system outperform trivial solutions?"
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from collections import Counter
import logging

from config import RANDOM_SEED, ALL_AGENTS, RADIOLOGIST_AGENTS, PATHOLOGIST_AGENTS
from models.dynamic_weights import BASE_WEIGHTS, get_base_weight

logger = logging.getLogger(__name__)


# =============================================================================
# ABSTRACT BASELINE
# =============================================================================

class BaselinePredictor(ABC):
    """Abstract base class for all baseline predictors."""
    
    name: str = "abstract_baseline"
    description: str = "Abstract baseline predictor"
    
    @abstractmethod
    def fit(self, y_train: np.ndarray) -> None:
        """Fit the baseline on training data (if needed)."""
        pass
    
    @abstractmethod
    def predict(self, X: Any = None) -> int:
        """Predict class for a single instance."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: Any = None) -> float:
        """Predict probability of positive class for a single instance."""
        pass
    
    def predict_batch(self, X: List[Any] = None, n: int = None) -> np.ndarray:
        """Predict for multiple instances."""
        if n is None and X is not None:
            n = len(X)
        return np.array([self.predict(x if X else None) for x, _ in zip(X or [None]*n, range(n))])
    
    def predict_proba_batch(self, X: List[Any] = None, n: int = None) -> np.ndarray:
        """Predict probabilities for multiple instances."""
        if n is None and X is not None:
            n = len(X)
        return np.array([self.predict_proba(x if X else None) for x, _ in zip(X or [None]*n, range(n))])


# =============================================================================
# 1. MAJORITY CLASS BASELINE
# =============================================================================

class MajorityClassBaseline(BaselinePredictor):
    """
    Always predicts the majority class from training data.
    
    This is the simplest possible baseline. Any useful model
    must outperform this to be considered non-trivial.
    """
    
    name = "majority_class"
    description = "Always predicts the most frequent class"
    
    def __init__(self):
        self.majority_class: int = 0  # Default to benign
        self.majority_prob: float = 0.5
        
    def fit(self, y_train: np.ndarray) -> None:
        """Learn the majority class from training labels."""
        y = np.array(y_train)
        counts = Counter(y)
        self.majority_class = counts.most_common(1)[0][0]
        self.majority_prob = counts[self.majority_class] / len(y)
        logger.info(f"MajorityClassBaseline: majority={self.majority_class}, prob={self.majority_prob:.3f}")
    
    def predict(self, X: Any = None) -> int:
        """Always return the majority class."""
        return self.majority_class
    
    def predict_proba(self, X: Any = None) -> float:
        """Return probability based on class frequency."""
        # If majority is 1 (malignant), return majority_prob
        # If majority is 0 (benign), return 1 - majority_prob
        if self.majority_class == 1:
            return self.majority_prob
        else:
            return 1 - self.majority_prob


# =============================================================================
# 2. RANDOM BASELINE
# =============================================================================

class RandomBaseline(BaselinePredictor):
    """
    Random predictions weighted by class distribution.
    
    Slightly better than uniform random but still trivial.
    """
    
    name = "random"
    description = "Random predictions weighted by class distribution"
    
    def __init__(self, seed: int = RANDOM_SEED):
        self.rng = np.random.RandomState(seed)
        self.class_probs: Dict[int, float] = {0: 0.5, 1: 0.5}
        
    def fit(self, y_train: np.ndarray) -> None:
        """Learn class distribution from training labels."""
        y = np.array(y_train)
        counts = Counter(y)
        total = len(y)
        self.class_probs = {c: count / total for c, count in counts.items()}
        logger.info(f"RandomBaseline: class_probs={self.class_probs}")
        
    def predict(self, X: Any = None) -> int:
        """Random class prediction based on learned distribution."""
        classes = list(self.class_probs.keys())
        probs = [self.class_probs[c] for c in classes]
        return self.rng.choice(classes, p=probs)
    
    def predict_proba(self, X: Any = None) -> float:
        """Return a random probability sampled around class distribution."""
        # Return probability for class 1 (malignant)
        base_prob = self.class_probs.get(1, 0.5)
        # Add some noise
        noise = self.rng.normal(0, 0.1)
        return np.clip(base_prob + noise, 0, 1)


# =============================================================================
# 3. SINGLE AGENT BASELINE
# =============================================================================

class SingleAgentBaseline(BaselinePredictor):
    """
    Wrapper to evaluate a single agent in isolation.
    
    Requires agent predictions to be pre-computed and passed as input.
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.name = f"single_{agent_name}"
        self.description = f"Single agent: {agent_name}"
        
    def fit(self, y_train: np.ndarray) -> None:
        """No fitting needed for single agent."""
        pass
    
    def predict(self, agent_prediction: Dict[str, Any]) -> int:
        """Extract prediction from agent results."""
        if self.agent_name not in agent_prediction:
            raise ValueError(f"Agent {self.agent_name} not in predictions")
        pred = agent_prediction[self.agent_name]
        if isinstance(pred, dict):
            return pred.get("predicted_class", 1 if pred.get("probability", 0) >= 0.5 else 0)
        return int(pred >= 0.5)
    
    def predict_proba(self, agent_prediction: Dict[str, Any]) -> float:
        """Extract probability from agent results."""
        if self.agent_name not in agent_prediction:
            raise ValueError(f"Agent {self.agent_name} not in predictions")
        pred = agent_prediction[self.agent_name]
        if isinstance(pred, dict):
            return pred.get("probability", 0.5)
        return float(pred)


# =============================================================================
# 4. UNWEIGHTED MAJORITY VOTE
# =============================================================================

class UnweightedMajorityVote(BaselinePredictor):
    """
    Simple majority voting across all agents with equal weights.
    
    Each agent gets one vote, majority wins.
    """
    
    name = "unweighted_majority_vote"
    description = "Simple majority voting across agents (equal weights)"
    
    def __init__(self, agent_names: List[str] = None):
        self.agent_names = agent_names or ALL_AGENTS
        
    def fit(self, y_train: np.ndarray) -> None:
        """No fitting needed."""
        pass
    
    def predict(self, agent_predictions: Dict[str, Any]) -> int:
        """Majority vote across agents."""
        votes = []
        for agent in self.agent_names:
            if agent not in agent_predictions:
                continue
            pred = agent_predictions[agent]
            if isinstance(pred, dict):
                vote = pred.get("predicted_class", 1 if pred.get("probability", 0) >= 0.5 else 0)
            else:
                vote = int(pred >= 0.5)
            votes.append(vote)
        
        if not votes:
            return 0  # Default to benign
        
        # Majority vote
        return int(np.mean(votes) >= 0.5)
    
    def predict_proba(self, agent_predictions: Dict[str, Any]) -> float:
        """Average probability across agents."""
        probs = []
        for agent in self.agent_names:
            if agent not in agent_predictions:
                continue
            pred = agent_predictions[agent]
            if isinstance(pred, dict):
                prob = pred.get("probability", 0.5)
            else:
                prob = float(pred)
            probs.append(prob)
        
        if not probs:
            return 0.5
        return np.mean(probs)


# =============================================================================
# 5. STATIC WEIGHTED AVERAGE
# =============================================================================

class StaticWeightedAverage(BaselinePredictor):
    """
    Weighted average using fixed BASE_WEIGHTS without dynamic adjustment.
    
    Tests whether dynamic weighting adds value over static weights.
    """
    
    name = "static_weighted_average"
    description = "Weighted average with fixed base weights (no dynamic adjustment)"
    
    def __init__(self, weights: Dict[str, float] = None, agent_names: List[str] = None):
        self.weights = weights or BASE_WEIGHTS.copy()
        self.agent_names = agent_names or ALL_AGENTS
        
    def fit(self, y_train: np.ndarray) -> None:
        """No fitting needed."""
        pass
    
    def predict(self, agent_predictions: Dict[str, Any]) -> int:
        """Weighted average prediction."""
        prob = self.predict_proba(agent_predictions)
        return int(prob >= 0.5)
    
    def predict_proba(self, agent_predictions: Dict[str, Any]) -> float:
        """Weighted average probability."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for agent in self.agent_names:
            if agent not in agent_predictions:
                continue
            pred = agent_predictions[agent]
            if isinstance(pred, dict):
                prob = pred.get("probability", 0.5)
            else:
                prob = float(pred)
            
            weight = self.weights.get(agent, 0.5)
            weighted_sum += prob * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        return weighted_sum / total_weight


# =============================================================================
# 6. SKLEARN VOTING CLASSIFIER EQUIVALENT
# =============================================================================

class SklearnVotingEquivalent(BaselinePredictor):
    """
    Soft-voting ensemble matching sklearn's VotingClassifier interface.
    
    Uses equal weights and soft voting (average probabilities).
    """
    
    name = "sklearn_voting"
    description = "Soft-voting ensemble (sklearn VotingClassifier equivalent)"
    
    def __init__(self, agent_names: List[str] = None, voting: str = "soft"):
        self.agent_names = agent_names or ALL_AGENTS
        self.voting = voting  # "soft" or "hard"
        
    def fit(self, y_train: np.ndarray) -> None:
        """No fitting needed."""
        pass
    
    def predict(self, agent_predictions: Dict[str, Any]) -> int:
        """Voting prediction."""
        if self.voting == "hard":
            # Hard voting: majority of class predictions
            votes = []
            for agent in self.agent_names:
                if agent not in agent_predictions:
                    continue
                pred = agent_predictions[agent]
                if isinstance(pred, dict):
                    vote = pred.get("predicted_class", 1 if pred.get("probability", 0) >= 0.5 else 0)
                else:
                    vote = int(pred >= 0.5)
                votes.append(vote)
            return int(np.mean(votes) >= 0.5) if votes else 0
        else:
            # Soft voting: average probabilities
            return int(self.predict_proba(agent_predictions) >= 0.5)
    
    def predict_proba(self, agent_predictions: Dict[str, Any]) -> float:
        """Average probability (soft voting)."""
        probs = []
        for agent in self.agent_names:
            if agent not in agent_predictions:
                continue
            pred = agent_predictions[agent]
            if isinstance(pred, dict):
                prob = pred.get("probability", 0.5)
            else:
                prob = float(pred)
            probs.append(prob)
        
        if not probs:
            return 0.5
        return np.mean(probs)


# =============================================================================
# 7. PURE PYTHON WEIGHTED AVERAGE (NO PROLOG)
# =============================================================================

class PurePythonWeightedAverage(BaselinePredictor):
    """
    Full weighted consensus without Prolog.
    
    Implements the same logic as the Prolog consensus but in pure Python.
    Tests whether Prolog/symbolic reasoning adds measurable value.
    """
    
    name = "python_weighted_average"
    description = "Weighted consensus in pure Python (no Prolog)"
    
    def __init__(self, weights: Dict[str, float] = None, agent_names: List[str] = None):
        self.weights = weights or BASE_WEIGHTS.copy()
        self.agent_names = agent_names or ALL_AGENTS
        
    def fit(self, y_train: np.ndarray) -> None:
        """No fitting needed."""
        pass
    
    def predict(self, agent_predictions: Dict[str, Any], case_metadata: Dict[str, Any] = None) -> int:
        """Weighted prediction with dynamic adjustment if metadata provided."""
        prob = self.predict_proba(agent_predictions, case_metadata)
        return int(prob >= 0.5)
    
    def predict_proba(self, agent_predictions: Dict[str, Any], case_metadata: Dict[str, Any] = None) -> float:
        """Weighted probability with optional dynamic adjustment."""
        # If metadata provided, compute dynamic weights
        if case_metadata:
            weights = self._compute_dynamic_weights(case_metadata)
        else:
            weights = self.weights
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for agent in self.agent_names:
            if agent not in agent_predictions:
                continue
            pred = agent_predictions[agent]
            
            # Extract probability
            if isinstance(pred, dict):
                prob = pred.get("probability", 0.5)
                # Apply size penalty if applicable
                if case_metadata and pred.get("size_source") == "unknown":
                    weight = weights.get(agent, 0.5) * 0.5  # 50% penalty
                else:
                    weight = weights.get(agent, 0.5)
            else:
                prob = float(pred)
                weight = weights.get(agent, 0.5)
            
            weighted_sum += prob * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        return weighted_sum / total_weight
    
    def _compute_dynamic_weights(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Compute dynamic weights based on information richness."""
        # Simplified dynamic weighting (matches models/dynamic_weights.py logic)
        rad_richness = self._compute_radiology_richness(metadata)
        path_richness = self._compute_pathology_richness(metadata)
        
        SCALE_FLOOR = 0.5
        weights = {}
        
        for agent, base_w in self.weights.items():
            if "radiologist" in agent:
                scale = SCALE_FLOOR + (1 - SCALE_FLOOR) * rad_richness
            elif "pathologist" in agent:
                scale = SCALE_FLOOR + (1 - SCALE_FLOOR) * path_richness
            else:
                scale = 1.0
            weights[agent] = base_w * scale
        
        return weights
    
    def _compute_radiology_richness(self, metadata: Dict[str, Any]) -> float:
        """Compute radiology information richness (0-1)."""
        num_images = metadata.get("num_images", 0)
        if num_images == 0:
            return 0.0
        elif num_images == 1:
            return 0.5
        else:
            return min(1.0, 0.5 + 0.25 * num_images)
    
    def _compute_pathology_richness(self, metadata: Dict[str, Any]) -> float:
        """Compute pathology information richness (0-1)."""
        findings = metadata.get("findings", "")
        impression = metadata.get("impression", "")
        combined = f"{findings} {impression}"
        
        # Length score
        length = len(combined)
        if length < 50:
            length_score = 0.2
        elif length < 200:
            length_score = 0.5
        elif length < 500:
            length_score = 0.8
        else:
            length_score = 1.0
        
        return length_score


# =============================================================================
# BASELINE REGISTRY
# =============================================================================

def get_all_baselines() -> Dict[str, BaselinePredictor]:
    """Get dictionary of all available baseline predictors."""
    baselines = {
        "majority_class": MajorityClassBaseline(),
        "random": RandomBaseline(),
        "unweighted_vote": UnweightedMajorityVote(),
        "static_weighted": StaticWeightedAverage(),
        "sklearn_soft_voting": SklearnVotingEquivalent(voting="soft"),
        "sklearn_hard_voting": SklearnVotingEquivalent(voting="hard"),
        "python_weighted": PurePythonWeightedAverage(),
    }
    
    # Add single-agent baselines
    for agent in ALL_AGENTS:
        short_name = agent.replace("radiologist_", "R_").replace("pathologist_", "P_")
        baselines[f"single_{short_name}"] = SingleAgentBaseline(agent)
    
    return baselines


def get_baseline(name: str) -> BaselinePredictor:
    """Get a specific baseline by name."""
    baselines = get_all_baselines()
    if name not in baselines:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(baselines.keys())}")
    return baselines[name]


# =============================================================================
# BASELINE EVALUATION HELPERS
# =============================================================================

def evaluate_baselines(
    y_true: np.ndarray,
    agent_predictions: List[Dict[str, Any]],
    baselines: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple baselines on the same data.
    
    Args:
        y_true: Ground truth labels
        agent_predictions: List of per-case agent prediction dictionaries
        baselines: List of baseline names to evaluate (default: all)
    
    Returns:
        Dictionary mapping baseline name to metrics
    """
    from evaluation.metrics import EvaluationMetrics
    
    if baselines is None:
        baselines = ["majority_class", "random", "unweighted_vote", "static_weighted"]
    
    metrics = EvaluationMetrics()
    results = {}
    
    for baseline_name in baselines:
        baseline = get_baseline(baseline_name)
        baseline.fit(y_true)  # Fit on all data for majority/random baselines
        
        if baseline_name in ["majority_class", "random"]:
            # These don't need agent predictions
            preds = baseline.predict_batch(n=len(y_true))
            probs = baseline.predict_proba_batch(n=len(y_true))
        else:
            # These need agent predictions
            preds = np.array([baseline.predict(ap) for ap in agent_predictions])
            probs = np.array([baseline.predict_proba(ap) for ap in agent_predictions])
        
        eval_result = metrics.evaluate(
            ground_truth=y_true.tolist(),
            predictions=preds.tolist(),
            probabilities=probs.tolist()
        )
        results[baseline_name] = eval_result
    
    return results
