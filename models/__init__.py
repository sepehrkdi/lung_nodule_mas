"""
Models Module
=============

Pre-trained deep learning models for image classification.

EDUCATIONAL PURPOSE:
- Transfer Learning: Using ImageNet pre-trained weights
- CNN Architectures: DenseNet for medical imaging
"""

from .classifier import NoduleClassifier, classify_nodule

__all__ = ['NoduleClassifier', 'classify_nodule']

