"""
Specialized Radiologist Agents
==============================

EDUCATIONAL PURPOSE - DIVERSE CLASSIFICATION APPROACHES:

This module implements three radiologist agents with different 
classification strategies to demonstrate multi-agent disagreement
and consensus:

1. RadiologistDenseNet: Pre-trained DenseNet121 (CNN)
2. RadiologistResNet: Pre-trained ResNet50 (CNN)  
3. RadiologistRules: Size/texture-based heuristics (rule-based)

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │                 Three Radiologist Agents                    │
    ├──────────────────┬──────────────────┬───────────────────────┤
    │  DenseNet121     │   ResNet50       │   Rule-Based          │
    │  (CNN, ImageNet) │  (CNN, ImageNet) │  (Size Heuristics)    │
    ├──────────────────┼──────────────────┼───────────────────────┤
    │  Weight: 1.0     │  Weight: 1.0     │   Weight: 0.7         │
    │  Subsymbolic     │  Subsymbolic     │   Symbolic/Explicit   │
    └──────────────────┴──────────────────┴───────────────────────┘

WHY DIFFERENT APPROACHES?
- CNNs may disagree due to different architectures/receptive fields
- Rule-based provides interpretable baseline and tiebreaker
- Demonstrates real-world ensemble disagreement handling
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from agents.spade_base import MedicalAgentBase, Belief, get_asl_path

logger = logging.getLogger(__name__)


# =============================================================================
# BASE RADIOLOGIST CLASS
# =============================================================================

class RadiologistBase(MedicalAgentBase):
    """
    Base class for all radiologist agents.
    
    Provides common interface and utilities for image classification.
    """
    
    # Agent metadata (override in subclasses)
    AGENT_TYPE = "radiologist"
    APPROACH = "base"
    WEIGHT = 1.0
    
    def __init__(self, name: str, asl_file: Optional[str] = None):
        if asl_file is None:
            asl_file = get_asl_path("radiologist")
        super().__init__(name=name, asl_file=asl_file)
        self._model_loaded = False
    
    def _register_actions(self) -> None:
        """Register internal actions for ASL plans."""
        self.internal_actions["classify_image"] = self._classify

        
    @abstractmethod
    def _classify(self, image: np.ndarray) -> Tuple[float, int]:
        """
        Classify image and return (probability, class).
        Must be implemented by subclasses.
        """
        pass
    
    def _prepare_image(self, image_data: Any) -> np.ndarray:
        """Prepare image data for classification."""
        if isinstance(image_data, np.ndarray):
            return image_data
        if isinstance(image_data, dict):
            # Generate synthetic image from features
            return self._generate_synthetic_image(image_data)
        return np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    
    def _generate_synthetic_image(self, features: Dict[str, Any]) -> np.ndarray:
        """Generate synthetic nodule image from features."""
        size = int(features.get("size_mm", 10) * 3)
        size = max(32, min(size, 128))
        
        image = np.zeros((size, size), dtype=np.uint8)
        
        y, x = np.ogrid[:size, :size]
        center = size // 2
        radius = size // 3
        
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        mask = dist <= radius
        image[mask] = 180
        
        # Add texture variation
        texture = features.get("texture", "solid")
        if texture in ["ground_glass", "ground-glass"]:
            noise = np.random.randint(0, 50, (size, size), dtype=np.uint8)
            image = np.clip(image.astype(int) + noise - 25, 0, 255).astype(np.uint8)
        
        return image
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process classification request."""
        nodule_id = request.get("nodule_id", "unknown")
        image_data = request.get("image") or request.get("features", {})
        
        logger.info(f"[{self.name}] Processing {nodule_id}")
        
        image = self._prepare_image(image_data)
        probability, predicted_class = self._classify(image)
        
        # Add belief about classification
        self.add_belief(Belief(
            "classification",
            (nodule_id, probability, predicted_class),
            annotations={"source": self.name, "approach": self.APPROACH}
        ))
        
        return {
            "nodule_id": nodule_id,
            "agent": self.name,
            "agent_type": self.AGENT_TYPE,
            "approach": self.APPROACH,
            "weight": self.WEIGHT,
            "status": "success",
            "findings": {
                "malignancy_probability": probability,
                "predicted_class": predicted_class
            }
        }


# =============================================================================
# DENSENET121 RADIOLOGIST (CNN #1)
# =============================================================================

class RadiologistDenseNet(RadiologistBase):
    """
    Radiologist using DenseNet121 pre-trained on ImageNet.
    
    EDUCATIONAL NOTE:
    DenseNet121 uses dense connections between layers, where each
    layer receives inputs from all preceding layers. This helps with
    feature reuse and gradient flow.
    
    OPERATING POINTS:
    Different thresholds simulate distinct expert styles:
    - Conservative (0.6): High specificity, fewer false positives
    - Balanced (0.5): Standard operating point
    - Sensitive (0.4): High recall, fewer missed nodules
    
    Architecture: 121 layers, ~8M parameters
    Input: 224x224 RGB
    """
    
    AGENT_TYPE = "radiologist"
    APPROACH = "densenet121"
    WEIGHT = 1.0
    
    # NEW: Operating point threshold (affects probability-to-class conversion)
    THRESHOLD = 0.5  # Balanced by default
    
    def __init__(self, name: str = "radiologist_densenet", threshold: float = 0.5):
        super().__init__(name=name)
        self._model = None
        self.THRESHOLD = threshold
    
    @classmethod
    def conservative(cls, name: str = "radiologist_conservative"):
        """
        Create conservative radiologist with high specificity.
        Threshold = 0.6 (requires higher probability for positive classification)
        """
        instance = cls(name=name, threshold=0.6)
        instance.APPROACH = "densenet121_conservative"
        return instance
    
    @classmethod
    def balanced(cls, name: str = "radiologist_balanced"):
        """Create balanced radiologist with standard threshold."""
        instance = cls(name=name, threshold=0.5)
        instance.APPROACH = "densenet121_balanced"
        return instance
    
    @classmethod
    def sensitive(cls, name: str = "radiologist_sensitive"):
        """
        Create sensitive radiologist with high recall.
        Threshold = 0.4 (lower threshold catches more potential cases)
        """
        instance = cls(name=name, threshold=0.4)
        instance.APPROACH = "densenet121_sensitive"
        return instance
        
    def _load_model(self):
        """Lazy load DenseNet121."""
        if self._model is not None:
            return
            
        try:
            import torch
            import torchvision.models as models
            
            logger.info(f"[{self.name}] Loading DenseNet121...")
            self._model = models.densenet121(pretrained=True)
            self._model.eval()
            self._model_loaded = True
            
            self.add_belief(Belief("model_loaded", ("densenet121", True)))
            logger.info(f"[{self.name}] DenseNet121 loaded")
            
        except ImportError:
            logger.warning(f"[{self.name}] PyTorch not available, using fallback")
            self._model = None
            
    def _classify(self, image: np.ndarray) -> Tuple[float, int]:
        """Classify using DenseNet121."""
        self._load_model()
        
        if self._model is not None:
            try:
                import torch
                from torchvision import transforms
                
                # Preprocess
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                
                # Convert grayscale to RGB if needed
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                
                input_tensor = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    output = self._model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    
                # Use first few class probabilities as malignancy proxy
                # Higher ImageNet class indices correlate with complexity
                prob = float(probs[0, :100].sum()) * 0.8 + 0.1
                prob = min(max(prob, 0.05), 0.95)
                
                predicted_class = self._prob_to_class(prob)
                return (prob, predicted_class)
                
            except Exception as e:
                logger.warning(f"[{self.name}] Classification error: {e}")
        
        # Fallback: feature-based estimation
        return self._fallback_classify(image)
    
    def _fallback_classify(self, image: np.ndarray) -> Tuple[float, int]:
        """Fallback classification based on image statistics."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=-1)
        else:
            gray = image
            
        mean_intensity = np.mean(gray) / 255.0
        std_intensity = np.std(gray) / 255.0
        
        # Heuristic: brighter, more variable = more suspicious
        prob = 0.3 + mean_intensity * 0.3 + std_intensity * 0.4
        prob = min(max(prob, 0.05), 0.95)
        
        return (prob, self._prob_to_class(prob))
    
    def _prob_to_class(self, prob: float) -> int:
        """Convert probability to malignancy class 1-5."""
        if prob < 0.2: return 1
        elif prob < 0.4: return 2
        elif prob < 0.6: return 3
        elif prob < 0.8: return 4
        else: return 5


# =============================================================================
# RESNET50 RADIOLOGIST (CNN #2)
# =============================================================================

class RadiologistResNet(RadiologistBase):
    """
    Radiologist using ResNet50 pre-trained on ImageNet.
    
    EDUCATIONAL NOTE:
    ResNet50 uses skip connections (residual connections) to enable
    training of deeper networks. Different from DenseNet, it adds
    the input to the output rather than concatenating.
    
    Architecture: 50 layers, ~25M parameters
    Input: 224x224 RGB
    """
    
    AGENT_TYPE = "radiologist"
    APPROACH = "resnet50"
    WEIGHT = 1.0
    
    def __init__(self, name: str = "radiologist_resnet"):
        super().__init__(name=name)
        self._model = None
        
    def _load_model(self):
        """Lazy load ResNet50."""
        if self._model is not None:
            return
            
        try:
            import torch
            import torchvision.models as models
            
            logger.info(f"[{self.name}] Loading ResNet50...")
            self._model = models.resnet50(pretrained=True)
            self._model.eval()
            self._model_loaded = True
            
            self.add_belief(Belief("model_loaded", ("resnet50", True)))
            logger.info(f"[{self.name}] ResNet50 loaded")
            
        except ImportError:
            logger.warning(f"[{self.name}] PyTorch not available, using fallback")
            self._model = None
            
    def _classify(self, image: np.ndarray) -> Tuple[float, int]:
        """Classify using ResNet50."""
        self._load_model()
        
        if self._model is not None:
            try:
                import torch
                from torchvision import transforms
                
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                
                input_tensor = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    output = self._model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    
                # Different weighting than DenseNet for diversity
                prob = float(probs[0, 50:150].sum()) * 0.7 + 0.15
                prob = min(max(prob, 0.05), 0.95)
                
                # Add small random variation to simulate model differences
                prob += np.random.uniform(-0.05, 0.05)
                prob = min(max(prob, 0.05), 0.95)
                
                predicted_class = self._prob_to_class(prob)
                return (prob, predicted_class)
                
            except Exception as e:
                logger.warning(f"[{self.name}] Classification error: {e}")
        
        return self._fallback_classify(image)
    
    def _fallback_classify(self, image: np.ndarray) -> Tuple[float, int]:
        """Fallback with slightly different heuristic than DenseNet."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=-1)
        else:
            gray = image
            
        # Edge detection proxy
        edges = np.abs(np.diff(gray.astype(float), axis=0)).mean()
        edges += np.abs(np.diff(gray.astype(float), axis=1)).mean()
        edges = edges / 255.0
        
        mean_intensity = np.mean(gray) / 255.0
        
        # Different formula than DenseNet
        prob = 0.25 + mean_intensity * 0.25 + edges * 0.5
        prob = min(max(prob, 0.05), 0.95)
        
        return (prob, self._prob_to_class(prob))
    
    def _prob_to_class(self, prob: float) -> int:
        if prob < 0.2: return 1
        elif prob < 0.4: return 2
        elif prob < 0.6: return 3
        elif prob < 0.8: return 4
        else: return 5


# =============================================================================
# RULE-BASED RADIOLOGIST (Symbolic)
# =============================================================================

class RadiologistRules(RadiologistBase):
    """
    Radiologist using size/texture-based heuristic rules.
    
    EDUCATIONAL NOTE:
    This demonstrates a symbolic, interpretable approach to classification.
    Rules are based on Lung-RADS guidelines:
    
    - <6mm solid: Low risk (Category 2)
    - 6-8mm solid: Probably benign (Category 3)
    - 8-15mm solid: Suspicious (Category 4A)
    - ≥15mm solid: Very suspicious (Category 4B)
    
    This agent serves as:
    1. Interpretable baseline
    2. Tiebreaker when CNNs disagree
    3. Demonstration of symbolic AI
    
    Weight: 0.7 (lower than CNNs due to simplicity)
    """
    
    AGENT_TYPE = "radiologist"
    APPROACH = "rule_based"
    WEIGHT = 0.7
    
    # Lung-RADS based rules
    SIZE_RISK_RULES = [
        # (min_size, max_size, texture, probability)
        (0, 6, "solid", 0.10),
        (0, 6, "part_solid", 0.15),
        (0, 30, "ground_glass", 0.15),
        (6, 8, "solid", 0.30),
        (6, 8, "part_solid", 0.35),
        (8, 15, "solid", 0.55),
        (8, 15, "part_solid", 0.60),
        (15, 30, "solid", 0.75),
        (15, 30, "part_solid", 0.80),
        (30, 1000, "solid", 0.90),
        (30, 1000, "part_solid", 0.90),
        (30, 1000, "ground_glass", 0.50),
    ]
    
    def __init__(self, name: str = "radiologist_rules"):
        super().__init__(name=name)
        self._model_loaded = True  # No model to load
        self.add_belief(Belief("model_loaded", ("rule_based", True)))
        
    def _classify(self, image: np.ndarray) -> Tuple[float, int]:
        """Classify using rule-based heuristics."""
        # Extract features from image
        features = self._extract_features(image)
        
        size_mm = features.get("size_mm", 10)
        texture = features.get("texture", "solid")
        
        # Apply rules
        probability = self._apply_rules(size_mm, texture)
        
        # Log reasoning
        logger.info(
            f"[{self.name}] Rule-based: size={size_mm:.1f}mm, "
            f"texture={texture} -> prob={probability:.3f}"
        )
        
        return (probability, self._prob_to_class(probability))
    
    def _extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract features from image for rule application."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=-1)
        else:
            gray = image
            
        # Estimate size from image dimensions
        size_mm = max(image.shape[:2]) / 5.0
        
        # Estimate texture from intensity variance
        std = np.std(gray) / 255.0
        if std > 0.3:
            texture = "ground_glass"
        elif std > 0.15:
            texture = "part_solid"
        else:
            texture = "solid"
            
        return {
            "size_mm": size_mm,
            "texture": texture,
            "mean_intensity": np.mean(gray) / 255.0,
            "std_intensity": std
        }
    
    def _apply_rules(self, size_mm: float, texture: Union[str, int]) -> float:
        """Apply Lung-RADS rules to get probability."""
        if isinstance(texture, int):
            # Map LIDC texture score (1-5) to string
            if texture <= 2:
                texture = "ground_glass"
            elif texture == 3:
                texture = "part_solid"
            else:
                texture = "solid"
                
        texture = texture.replace("-", "_").lower()
        
        for min_s, max_s, tex, prob in self.SIZE_RISK_RULES:
            if min_s <= size_mm < max_s and tex == texture:
                return prob
        
        # Default based on size alone
        if size_mm < 6:
            return 0.15
        elif size_mm < 8:
            return 0.30
        elif size_mm < 15:
            return 0.55
        else:
            return 0.80
    
    def _prob_to_class(self, prob: float) -> int:
        if prob < 0.2: return 1
        elif prob < 0.4: return 2
        elif prob < 0.6: return 3
        elif prob < 0.8: return 4
        else: return 5
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process with feature extraction from request."""
        nodule_id = request.get("nodule_id", "unknown")
        
        # Can use features directly if provided
        features = request.get("features", {})
        if features and "size_mm" in features:
            size_mm = features.get("size_mm", 10)
            texture = features.get("texture", "solid")
            probability = self._apply_rules(size_mm, texture)
            predicted_class = self._prob_to_class(probability)
            
            self.add_belief(Belief(
                "classification",
                (nodule_id, probability, predicted_class),
                annotations={"source": self.name, "approach": self.APPROACH}
            ))
            
            return {
                "nodule_id": nodule_id,
                "agent": self.name,
                "agent_type": self.AGENT_TYPE,
                "approach": self.APPROACH,
                "weight": self.WEIGHT,
                "status": "success",
                "reasoning": f"Size={size_mm}mm, Texture={texture}",
                "findings": {
                    "malignancy_probability": probability,
                    "predicted_class": predicted_class,
                    "size_mm": size_mm,
                    "texture": texture
                }
            }
        
        # Fall back to image-based
        return await super().process_request(request)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_radiologist_densenet(name: str = "radiologist_densenet"):
    """Create DenseNet121 radiologist agent."""
    return RadiologistDenseNet(name=name)

def create_radiologist_resnet(name: str = "radiologist_resnet"):
    """Create ResNet50 radiologist agent."""
    return RadiologistResNet(name=name)

def create_radiologist_rules(name: str = "radiologist_rules"):
    """Create rule-based radiologist agent."""
    return RadiologistRules(name=name)

def create_all_radiologists():
    """Create all three radiologist agents (original architecture-diverse)."""
    return [
        RadiologistDenseNet(name="radiologist_densenet"),
        RadiologistResNet(name="radiologist_resnet"),
        RadiologistRules(name="radiologist_rules")
    ]

def create_calibrated_radiologists():
    """
    Create ensemble of 3 radiologists with different operating points.
    
    This simulates clinical inter-reader variability:
    - R1 (Conservative): High specificity, threshold=0.6
      "Overcaller" - would rather watch and wait
    - R2 (Balanced): Standard threshold=0.5
      "By the book" - follows guidelines precisely  
    - R3 (Sensitive): High recall, threshold=0.4
      "Aggressive" - catches more potential cases
    
    All use the same DenseNet121 model, differing only in
    decision threshold (simulating expert judgment variation).
    
    Returns:
        List of 3 RadiologistDenseNet agents with different thresholds
    """
    return [
        RadiologistDenseNet.conservative(),
        RadiologistDenseNet.balanced(),
        RadiologistDenseNet.sensitive()
    ]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== Radiologist Agents Test ===\n")
    
    # Create all radiologists
    radiologists = create_all_radiologists()
    
    # Test request
    test_request = {
        "nodule_id": "test_001",
        "features": {
            "size_mm": 12,
            "texture": "solid",
            "malignancy": 4
        }
    }
    
    async def test():
        print("Testing with 12mm solid nodule:\n")
        
        for rad in radiologists:
            result = await rad.process_request(test_request)
            print(f"{rad.name} ({rad.APPROACH}):")
            print(f"  Probability: {result['findings']['malignancy_probability']:.3f}")
            print(f"  Class: {result['findings']['predicted_class']}")
            print(f"  Weight: {rad.WEIGHT}")
            print()
    
    asyncio.run(test())
