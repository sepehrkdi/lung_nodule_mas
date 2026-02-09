"""
Multi-Agent System Agents Module
================================

This module contains the BDI (Belief-Desire-Intention) agents for the
lung nodule classification system.

EDUCATIONAL PURPOSE - DISTRIBUTED AI CONCEPTS:
- BDI Architecture: Beliefs, Desires, Intentions model (Bratman)
- Agent Communication: Speech acts (Austin & Searle)
- Multi-Agent Coordination: Parallel task execution with message passing
- SPADE-BDI: Proper AgentSpeak(L) interpreter with XMPP communication

Agents (Original - Custom BDI):
    - RadiologistAgent: Analyzes CT images using pre-trained CNN
    - PathologistAgent: Extracts findings from reports using NLP
    - OncologistAgent: Combines findings using Prolog reasoning

Agents (SPADE-BDI - Proper AgentSpeak Interpreter):
    - SpadeRadiologistAgent: SPADE-BDI agent with DenseNet121
    - SpadePathologistAgent: SPADE-BDI agent with scispaCy NLP
    - SpadeOncologistAgent: SPADE-BDI agent with Prolog reasoning

Specialized Variant Agents (Extended Architecture):
    Radiologists (3 approaches):
    - RadiologistDenseNet: DenseNet121 CNN architecture
    - RadiologistResNet: ResNet50 CNN architecture  
    - RadiologistRuleBased: Size/texture heuristic rules
    
    Pathologists (3 approaches):
    - PathologistRegex: Regular expression pattern matching
    - PathologistSpacy: spaCy NER + semantic rules
    - PathologistContext: Contextual analysis for negation and uncertainty

Note: The SPADE-BDI agents provide proper AgentSpeak(L) plan execution
      as required for academic BDI demonstrations.
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


