"""
Models Module
=============

Pre-trained deep learning models for image classification,
aggregation, and consensus mechanisms.

EDUCATIONAL PURPOSE:
- Transfer Learning: Using ImageNet pre-trained weights
- CNN Architectures: DenseNet for medical imaging
- Multi-Agent Consensus: Weighted voting and disagreement resolution
"""

from .classifier import NoduleClassifier, classify_nodule
from .aggregation import get_aggregator
from .dynamic_weights import (
    DynamicWeightCalculator,
    WeightingMode,
    BASE_WEIGHTS,
    get_base_weight,
    RichnessScores,
)
from .python_consensus import (
    PythonConsensusEngine,
    AgentFinding,
    ConsensusResult,
    DisagreementStrategy,
    compute_lung_rads,
    compute_t_stage,
    dict_to_agent_findings,
)

__all__ = [
    # Classifier
    'NoduleClassifier', 
    'classify_nodule',
    
    # Aggregation
    'get_aggregator',
    
    # Dynamic weights
    'DynamicWeightCalculator',
    'WeightingMode',
    'BASE_WEIGHTS',
    'get_base_weight',
    'RichnessScores',
    
    # Python consensus
    'PythonConsensusEngine',
    'AgentFinding',
    'ConsensusResult',
    'DisagreementStrategy',
    'compute_lung_rads',
    'compute_t_stage',
    'dict_to_agent_findings',
]

