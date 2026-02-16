"""
Pure Python Consensus Engine
============================

This module implements the same consensus logic as the Prolog knowledge base
(knowledge/multi_agent_consensus.pl) in pure Python.

Purpose: Enable ablation study comparing Prolog vs Python consensus to test
whether symbolic reasoning provides measurable benefit.

Implements:
1. Weighted consensus calculation
2. Confidence computation based on agreement/variance
3. Disagreement detection
4. Disagreement resolution strategies
5. Lung-RADS categorization

All logic mirrors the Prolog predicates to ensure fair comparison.
"""

import math
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from config import ALL_AGENTS, RADIOLOGIST_AGENTS, PATHOLOGIST_AGENTS
from models.dynamic_weights import BASE_WEIGHTS, get_base_weight

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AgentFinding:
    """Single agent's finding for a case."""
    agent_name: str
    probability: float
    predicted_class: int
    confidence: float = 1.0
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ConsensusResult:
    """Result of consensus computation."""
    probability: float
    predicted_class: int
    confidence: float
    strategy: str
    radiologist_consensus: float
    pathologist_consensus: float
    has_disagreement: bool
    agent_weights: Dict[str, float] = field(default_factory=dict)
    variance: float = 0.0


class DisagreementStrategy(Enum):
    """Disagreement resolution strategies matching Prolog KB."""
    WEIGHTED_AVERAGE = "weighted_average"
    VISUAL_TEXT_CONFLICT = "visual_text_conflict_recheck"
    TEXT_OVERRIDE = "text_override_missed_visual"
    PATHOLOGIST_OVERRIDE = "pathologist_override"
    CNN_NLP_AGREEMENT = "cnn_nlp_agreement"
    RULE_BASED_TIEBREAKER = "rule_based_tiebreaker"


# =============================================================================
# AGENT TYPE HELPERS
# =============================================================================

def is_radiologist(agent_name: str) -> bool:
    """Check if agent is a radiologist."""
    return "radiologist" in agent_name


def is_pathologist(agent_name: str) -> bool:
    """Check if agent is a pathologist."""
    return "pathologist" in agent_name


def is_cnn_radiologist(agent_name: str) -> bool:
    """Check if agent is a CNN-based radiologist (not rule-based)."""
    return agent_name in ["radiologist_densenet", "radiologist_resnet"]


# =============================================================================
# PYTHON CONSENSUS ENGINE
# =============================================================================

class PythonConsensusEngine:
    """
    Pure Python implementation of multi-agent consensus.
    
    Mirrors the logic in knowledge/multi_agent_consensus.pl for
    ablation comparison between Prolog and Python consensus.
    """
    
    # Disagreement threshold (std dev > 0.08)
    DISAGREEMENT_THRESHOLD = 0.08
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize consensus engine.
        
        Args:
            weights: Agent weights (defaults to BASE_WEIGHTS)
        """
        self.weights = weights or BASE_WEIGHTS.copy()
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update agent weights (for dynamic weighting)."""
        self.weights.update(new_weights)
    
    def compute_consensus(
        self,
        findings: List[AgentFinding],
        resolve_disagreements: bool = True
    ) -> ConsensusResult:
        """
        Compute weighted consensus from agent findings.
        
        Mirrors Prolog predicate: calculate_consensus/3
        
        Args:
            findings: List of agent findings
            resolve_disagreements: Whether to apply disagreement resolution
            
        Returns:
            ConsensusResult with probability, class, confidence, and strategy
        """
        if not findings:
            return ConsensusResult(
                probability=0.5,
                predicted_class=0,
                confidence=0.0,
                strategy="no_findings",
                radiologist_consensus=0.5,
                pathologist_consensus=0.5,
                has_disagreement=False
            )
        
        # Compute weighted average
        weighted_prob, total_weight = self._sum_weighted_probs(findings)
        
        if total_weight == 0:
            mean_prob = 0.5
        else:
            mean_prob = weighted_prob / total_weight
        
        # Compute confidence based on agreement
        confidence = self._calculate_agreement(findings, mean_prob)
        
        # Compute per-type consensus
        rad_consensus = self._compute_type_consensus(findings, is_radiologist)
        path_consensus = self._compute_type_consensus(findings, is_pathologist)
        cnn_consensus = self._compute_type_consensus(findings, is_cnn_radiologist)
        
        # Check for disagreement
        has_disagreement = self._has_disagreement(findings)
        
        # Build weights dict
        agent_weights = {f.agent_name: self.weights.get(f.agent_name, 0.5) 
                        for f in findings}
        
        # Initial result
        result = ConsensusResult(
            probability=mean_prob,
            predicted_class=1 if mean_prob >= 0.5 else 0,
            confidence=confidence,
            strategy=DisagreementStrategy.WEIGHTED_AVERAGE.value,
            radiologist_consensus=rad_consensus,
            pathologist_consensus=path_consensus,
            has_disagreement=has_disagreement,
            agent_weights=agent_weights,
            variance=self._compute_variance(findings, mean_prob)
        )
        
        # Apply disagreement resolution if enabled
        if resolve_disagreements and has_disagreement:
            result = self._resolve_disagreement(findings, result, cnn_consensus, path_consensus)
        
        return result
    
    def _sum_weighted_probs(self, findings: List[AgentFinding]) -> Tuple[float, float]:
        """
        Sum weighted probabilities.
        
        Mirrors Prolog predicate: sum_weighted_probs/3
        """
        total_weighted_prob = 0.0
        total_weight = 0.0
        
        for finding in findings:
            weight = self.weights.get(finding.agent_name, 0.5)
            total_weighted_prob += finding.probability * weight
            total_weight += weight
        
        return total_weighted_prob, total_weight
    
    def _calculate_agreement(self, findings: List[AgentFinding], mean_prob: float) -> float:
        """
        Calculate agreement/confidence based on variance.
        
        Mirrors Prolog predicate: calculate_agreement/3
        Confidence = max(0, 1 - (std_dev * 3))
        """
        if len(findings) <= 1:
            return 0.8  # Default for single agent
        
        variance = self._compute_variance(findings, mean_prob)
        std_dev = math.sqrt(variance)
        
        # Confidence decreases as disagreement increases
        confidence = max(0.0, 1.0 - (std_dev * 3))
        return confidence
    
    def _compute_variance(self, findings: List[AgentFinding], mean_prob: float) -> float:
        """Compute variance of probabilities."""
        if len(findings) <= 1:
            return 0.0
        
        sum_diffs = sum((f.probability - mean_prob) ** 2 for f in findings)
        return sum_diffs / len(findings)
    
    def _compute_type_consensus(
        self, 
        findings: List[AgentFinding],
        type_check_fn
    ) -> float:
        """Compute consensus for a specific agent type."""
        type_findings = [f for f in findings if type_check_fn(f.agent_name)]
        if not type_findings:
            return 0.5
        return sum(f.probability for f in type_findings) / len(type_findings)
    
    def _has_disagreement(self, findings: List[AgentFinding]) -> bool:
        """
        Check if there's significant disagreement (std dev > threshold).
        
        Mirrors Prolog predicate: has_disagreement/1
        """
        if len(findings) < 2:
            return False
        
        probs = [f.probability for f in findings]
        mean_prob = sum(probs) / len(probs)
        variance = sum((p - mean_prob) ** 2 for p in probs) / len(probs)
        std_dev = math.sqrt(variance)
        
        return std_dev > self.DISAGREEMENT_THRESHOLD
    
    def _resolve_disagreement(
        self,
        findings: List[AgentFinding],
        base_result: ConsensusResult,
        cnn_consensus: float,
        path_consensus: float
    ) -> ConsensusResult:
        """
        Apply disagreement resolution strategies.
        
        Mirrors Prolog predicates: resolve_disagreement/4
        
        Strategies (in order of priority):
        1. Visual-Text Conflict: CNN malignant, NLP benign
        2. Text Override: CNN benign, NLP malignant (trust NLP)
        3. Pathologist Override: NLP confident, CNN uncertain
        4. CNN-NLP Agreement: Close probabilities
        5. Rule-Based Tiebreaker: Use rule-based radiologist
        6. Default: Weighted average
        """
        result = ConsensusResult(
            probability=base_result.probability,
            predicted_class=base_result.predicted_class,
            confidence=base_result.confidence,
            strategy=base_result.strategy,
            radiologist_consensus=base_result.radiologist_consensus,
            pathologist_consensus=base_result.pathologist_consensus,
            has_disagreement=base_result.has_disagreement,
            agent_weights=base_result.agent_weights,
            variance=base_result.variance
        )
        
        # Strategy 1: Visual-Text Conflict
        # CNN sees malignancy (>0.65), NLP sees benign (<0.35)
        if cnn_consensus > 0.65 and path_consensus < 0.35:
            result.probability = (cnn_consensus + path_consensus) / 2
            result.confidence = 0.4  # Low confidence due to conflict
            result.strategy = DisagreementStrategy.VISUAL_TEXT_CONFLICT.value
            result.predicted_class = 1 if result.probability >= 0.5 else 0
            logger.info(f"Disagreement resolved: visual_text_conflict (CNN={cnn_consensus:.2f}, NLP={path_consensus:.2f})")
            return result
        
        # Strategy 2: Text Override (Missed Visual)
        # CNN benign (<0.35), NLP malignant (>0.65) - trust NLP
        if cnn_consensus < 0.35 and path_consensus > 0.65:
            result.probability = path_consensus
            result.confidence = 0.8  # High confidence in text
            result.strategy = DisagreementStrategy.TEXT_OVERRIDE.value
            result.predicted_class = 1 if result.probability >= 0.5 else 0
            logger.info(f"Disagreement resolved: text_override (CNN={cnn_consensus:.2f}, NLP={path_consensus:.2f})")
            return result
        
        # Strategy 3: Pathologist Override
        # NLP confident in malignancy (>=0.60), CNN uncertain (0.35-0.65)
        if path_consensus >= 0.60 and 0.35 <= cnn_consensus <= 0.65 and path_consensus > cnn_consensus:
            result.probability = path_consensus
            result.confidence = 0.75
            result.strategy = DisagreementStrategy.PATHOLOGIST_OVERRIDE.value
            result.predicted_class = 1 if result.probability >= 0.5 else 0
            logger.info(f"Disagreement resolved: pathologist_override (CNN={cnn_consensus:.2f}, NLP={path_consensus:.2f})")
            return result
        
        # Strategy 4: CNN-NLP Agreement
        # Probabilities are close (within 0.2)
        if abs(cnn_consensus - path_consensus) < 0.2:
            result.probability = cnn_consensus * 0.6 + path_consensus * 0.4
            result.confidence = min(1.0, base_result.confidence + 0.1)
            result.strategy = DisagreementStrategy.CNN_NLP_AGREEMENT.value
            result.predicted_class = 1 if result.probability >= 0.5 else 0
            logger.info(f"Disagreement resolved: cnn_nlp_agreement (CNN={cnn_consensus:.2f}, NLP={path_consensus:.2f})")
            return result
        
        # Strategy 5: Rule-Based Tiebreaker
        rule_finding = next((f for f in findings if f.agent_name == "radiologist_rulebased"), None)
        if rule_finding:
            result.probability = cnn_consensus * 0.5 + rule_finding.probability * 0.5
            result.strategy = DisagreementStrategy.RULE_BASED_TIEBREAKER.value
            result.predicted_class = 1 if result.probability >= 0.5 else 0
            logger.info(f"Disagreement resolved: rule_based_tiebreaker (CNN={cnn_consensus:.2f}, Rule={rule_finding.probability:.2f})")
            return result
        
        # Default: Keep weighted average
        result.strategy = DisagreementStrategy.WEIGHTED_AVERAGE.value
        return result


# =============================================================================
# LUNG-RADS CLASSIFICATION (PYTHON VERSION)
# =============================================================================

@dataclass
class LungRADSResult:
    """Lung-RADS classification result."""
    category: str
    description: str
    recommendation: str
    urgency: str


def compute_lung_rads(
    size_mm: Optional[float],
    texture: str,
    features: Dict[str, Any] = None
) -> LungRADSResult:
    """
    Compute Lung-RADS category based on nodule characteristics.
    
    Mirrors Prolog predicate: lung_rads/3
    
    Args:
        size_mm: Nodule size in millimeters
        texture: Nodule texture (solid, part_solid, ground_glass)
        features: Additional features (spiculation, lymphadenopathy, etc.)
        
    Returns:
        LungRADSResult with category, description, recommendation, urgency
    """
    features = features or {}
    
    # Category 1: Negative (no nodules)
    if size_mm is None or size_mm <= 0:
        return LungRADSResult(
            category="1",
            description="Negative - no nodules",
            recommendation="Continue annual screening",
            urgency="low"
        )
    
    # Category 2: Benign Appearance
    if texture == "solid" and size_mm < 6:
        return LungRADSResult(
            category="2",
            description="Benign - solid <6mm",
            recommendation="Continue annual screening",
            urgency="low"
        )
    
    if texture == "part_solid" and size_mm < 6:
        return LungRADSResult(
            category="2",
            description="Benign - part-solid <6mm",
            recommendation="Continue annual screening",
            urgency="low"
        )
    
    if texture == "ground_glass" and size_mm < 30:
        return LungRADSResult(
            category="2",
            description="Benign - GGN <30mm",
            recommendation="Continue annual screening",
            urgency="low"
        )
    
    # Category 3: Probably Benign
    if texture == "solid" and 6 <= size_mm < 8:
        return LungRADSResult(
            category="3",
            description="Probably benign - solid 6-8mm",
            recommendation="6-month follow-up CT",
            urgency="low"
        )
    
    if texture == "part_solid" and 6 <= size_mm < 8:
        return LungRADSResult(
            category="3",
            description="Probably benign - part-solid 6-8mm",
            recommendation="6-month follow-up CT",
            urgency="low"
        )
    
    if texture == "ground_glass" and size_mm >= 30:
        return LungRADSResult(
            category="3",
            description="Probably benign - GGN >=30mm",
            recommendation="6-month follow-up CT",
            urgency="low"
        )
    
    # Category 4A: Suspicious
    if texture == "solid" and 8 <= size_mm < 15:
        result = LungRADSResult(
            category="4A",
            description="Suspicious - solid 8-15mm",
            recommendation="3-month follow-up CT or PET/CT",
            urgency="medium"
        )
    elif texture == "part_solid" and size_mm >= 6:
        result = LungRADSResult(
            category="4A",
            description="Suspicious - part-solid >=6mm solid",
            recommendation="3-month follow-up CT or PET/CT",
            urgency="medium"
        )
    # Category 4B: Very Suspicious
    elif texture == "solid" and size_mm >= 15:
        result = LungRADSResult(
            category="4B",
            description="Very suspicious - solid >=15mm",
            recommendation="PET/CT and tissue sampling",
            urgency="high"
        )
    else:
        # Default to 4A for unclear cases with significant size
        result = LungRADSResult(
            category="4A",
            description="Suspicious - indeterminate",
            recommendation="3-month follow-up CT",
            urgency="medium"
        )
    
    # Check for 4X (additional suspicious features)
    suspicious_features = ["spiculation", "lymphadenopathy", "pleural_invasion"]
    has_suspicious = any(features.get(f) for f in suspicious_features)
    
    if has_suspicious and result.category in ["4A", "4B"]:
        return LungRADSResult(
            category="4X",
            description="Very suspicious - additional features",
            recommendation="PET/CT and tissue sampling",
            urgency="high"
        )
    
    return result


# =============================================================================
# T-STAGE CALCULATION (PYTHON VERSION)
# =============================================================================

def compute_t_stage(size_mm: Optional[float]) -> Tuple[str, str]:
    """
    Compute T-stage based on tumor size.
    
    Mirrors Prolog predicate: t_stage/3
    
    Returns:
        Tuple of (stage, description)
    """
    if size_mm is None:
        return ("TX", "Tumor size unknown")
    
    if size_mm <= 10:
        return ("T1a", "Tumor <=1cm")
    elif size_mm <= 20:
        return ("T1b", "Tumor >1-2cm")
    elif size_mm <= 30:
        return ("T1c", "Tumor >2-3cm")
    elif size_mm <= 40:
        return ("T2a", "Tumor >3-4cm")
    elif size_mm <= 50:
        return ("T2b", "Tumor >4-5cm")
    elif size_mm <= 70:
        return ("T3", "Tumor >5-7cm")
    else:
        return ("T4", "Tumor >7cm")


# =============================================================================
# CONSENSUS ENGINE FACTORY
# =============================================================================

def create_consensus_engine(
    weights: Dict[str, float] = None,
    mode: str = "python"
) -> PythonConsensusEngine:
    """
    Factory function to create a consensus engine.
    
    Args:
        weights: Agent weights
        mode: "python" (currently only Python supported here)
        
    Returns:
        PythonConsensusEngine instance
    """
    return PythonConsensusEngine(weights=weights)


# =============================================================================
# HELPER: CONVERT DICT FINDINGS TO AGENT FINDINGS
# =============================================================================

def dict_to_agent_findings(
    agent_predictions: Dict[str, Any]
) -> List[AgentFinding]:
    """
    Convert dictionary of agent predictions to list of AgentFinding objects.
    
    Args:
        agent_predictions: Dict mapping agent_name to prediction dict or probability
        
    Returns:
        List of AgentFinding objects
    """
    findings = []
    
    for agent_name, pred in agent_predictions.items():
        if isinstance(pred, dict):
            probability = pred.get("probability", 0.5)
            predicted_class = pred.get("predicted_class", 1 if probability >= 0.5 else 0)
            confidence = pred.get("confidence", 1.0)
            features = pred.get("features", {})
        else:
            probability = float(pred)
            predicted_class = 1 if probability >= 0.5 else 0
            confidence = 1.0
            features = {}
        
        findings.append(AgentFinding(
            agent_name=agent_name,
            probability=probability,
            predicted_class=predicted_class,
            confidence=confidence,
            features=features
        ))
    
    return findings
