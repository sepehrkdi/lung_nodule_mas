"""
Multi-Agent System agents module.

Contains BDI agents for lung nodule classification:
- Radiologist agents (DenseNet, ResNet, Rule-based)
- Pathologist agents (Regex, spaCy, Context)
- Oncologist agent (consensus coordination)
"""


# SPADE-BDI implementation (proper AgentSpeak interpreter)
from .spade_base import MedicalAgentBase, Belief as SpadeBelief, get_asl_path
from .spade_radiologist import RadiologistAgent as SpadeRadiologistAgent
from .spade_radiologist import create_spade_radiologist
from .spade_pathologist import PathologistAgent as SpadePathologistAgent
from .spade_pathologist import create_spade_pathologist
from .spade_oncologist import OncologistAgent as SpadeOncologistAgent
from .spade_oncologist import create_spade_oncologist

# Specialized Variant Agents - Extended Architecture
from .radiologist_variants import (
    RadiologistBase,
    RadiologistDenseNet,
    RadiologistResNet,
    RadiologistRules,
    ModelUnavailableError,
    ClassificationError,
    create_radiologist_densenet,
    create_radiologist_resnet,
    create_radiologist_rules,
    create_all_radiologists,
    create_calibrated_radiologists,
)

from .pathologist_variants import (
    PathologistBase,
    PathologistRegex,
    PathologistSpacy,
    PathologistContext,
    create_pathologist_regex,
    create_pathologist_spacy,
    create_pathologist_context,
    create_all_pathologists,
)


__all__ = [
    
    # SPADE-BDI (proper AgentSpeak)
    'MedicalAgentBase',
    'SpadeBelief',
    'get_asl_path',
    'SpadeRadiologistAgent',
    'SpadePathologistAgent',
    'SpadeOncologistAgent',
    'create_spade_radiologist',
    'create_spade_pathologist',
    'create_spade_oncologist',
    
    # Specialized Radiologist Variants
    'RadiologistBase',
    'RadiologistDenseNet',
    'RadiologistResNet',
    'RadiologistRules',
    'ModelUnavailableError',
    'ClassificationError',
    'create_radiologist_densenet',
    'create_radiologist_resnet',
    'create_radiologist_rules',
    'create_all_radiologists',
    'create_calibrated_radiologists',
    
    # Specialized Pathologist Variants
    'PathologistBase',
    'PathologistRegex',
    'PathologistSpacy',
    'PathologistContext',
    'create_pathologist_regex',
    'create_pathologist_spacy',
    'create_pathologist_context',
    'create_all_pathologists',
]


