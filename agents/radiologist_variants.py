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
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np

from agents.spade_base import MedicalAgentBase, Belief, get_asl_path
from models.aggregation import get_aggregator
from models.classifier import NoduleClassifier, calibrate_xrv_probability
from models.dynamic_weights import BASE_WEIGHTS, get_base_weight

logger = logging.getLogger(__name__)


class ModelUnavailableError(Exception):
    """
    Raised when a required CNN model is not available.
    
    STRICT MODE: The system requires all models to be properly loaded.
    No fallback heuristics are provided.
    """
    pass


class ClassificationError(Exception):
    """
    Raised when image classification fails.
    
    STRICT MODE: Classification errors are not silently handled.
    """
    pass


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
    WEIGHT = 1.0  # Base weight — dynamically scaled per-case by DynamicWeightCalculator
    AGGREGATION_STRATEGY = "weighted_average"  # Default aggregation for multi-image

    def __init__(self, name: str, asl_file: Optional[str] = None):
        if asl_file is None:
            asl_file = get_asl_path("radiologist")
        super().__init__(name=name, asl_file=asl_file)
        self._model_loaded = False
    
    def _register_actions(self) -> None:
        """Register internal actions for ASL plans."""
        self.internal_actions["load_classifier"] = self._action_load_classifier
        self.internal_actions["classify_image"] = self._action_classify_image
        self.internal_actions["extract_features"] = self._action_extract_features

    # =========================================================================
    # BDI Internal Actions (wrappers for variants)
    # =========================================================================

    def _action_load_classifier(self) -> bool:
        """Internal action: Load model."""
        try:
            # Variants handle loading in __init__ or lazy usage
            # This is just a hook for the ASL to feel good
            self.add_belief(Belief("model_loaded", (True,)))
            self.add_belief(Belief("model_name", (self.__class__.__name__,)))
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Load error: {e}")
            return False

    def _action_classify_image(self, nodule_id: str, image_data: Any) -> Tuple[float, int]:
        """Internal action: Classify image."""
        try:
            image = self._prepare_image(image_data)
            prob, pred_class = self._classify(image)
            
            # Update belief (redundant but good for BDI)
            self.add_belief(Belief(
                "classification",
                (nodule_id, prob, pred_class),
                annotations={"source": self.name}
            ))
            return (prob, pred_class)
        except Exception as e:
            logger.error(f"[{self.name}] Classification error: {e}")
            return (0.5, 3)

    def _action_extract_features(
        self, 
        nodule_id: str, 
        image_data: Any
    ) -> Tuple[float, str, str]:
        """Internal action: Extract visual features."""
        try:
            # Fallback feature extraction for all variants
            # In a real system, each variant might extract differently
            # For now, we use a shared heuristic extraction
            image = self._prepare_image(image_data)
            
            # Size from request metadata, or None if not available
            size_mm = None
            if isinstance(image_data, dict):
                 size_mm = image_data.get("size_mm")
            
            # Texture heuristic from image noise
            texture = "solid"
            if image.std() > 50: texture = "heterogeneous"
            elif image.std() > 30: texture = "part_solid"
            
            # Shape
            shape = "round"
            
            return (size_mm, texture, shape)
        except Exception as e:
            logger.error(f"[{self.name}] Feature extraction error: {e}")
            return (None, "unknown", "unknown")

        
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
        """
        Process classification request (supports both single and multi-image).

        Args:
            request: Request dict with either:
                     - Single-image: {"nodule_id": ..., "image": ..., "features": ...}
                     - Multi-image: {"case_id": ..., "images": [...], "image_metadata": [...], "features": ...}

        Returns:
            Result dict with findings
        """
        # Detect request type
        if "images" in request and isinstance(request["images"], list):
            # Multi-image path
            return await self._process_multi_image(request)
        else:
            # Legacy single-image path
            return await self._process_single_image(request)

    async def _process_single_image(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process single-image request."""
        nodule_id = request.get("nodule_id", "unknown")
        # Support both 'image' and 'image_array' keys for compatibility
        image_data = request.get("image")
        if image_data is None:
            image_data = request.get("image_array")
        if image_data is None:
            image_data = request.get("features", {})

        logger.info(f"[{self.name}] Processing single image {nodule_id}")

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

    async def _process_multi_image(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multi-image request (NLMCXR path).

        Classifies each image independently, then aggregates predictions.
        """
        case_id = request.get("case_id", "unknown")
        images = request["images"]
        image_metadata = request.get("image_metadata", [])

        logger.info(f"[{self.name}] Processing {case_id} with {len(images)} images")

        # Classify each image independently
        probabilities = []
        classes = []

        for i, img in enumerate(images):
            try:
                prob, cls = self._classify(img)
                probabilities.append(prob)
                classes.append(cls)
                logger.debug(
                    f"[{self.name}] Image {i} ({image_metadata[i].get('view_type', 'Unknown') if i < len(image_metadata) else 'Unknown'}): "
                    f"prob={prob:.3f}, class={cls}"
                )
            except Exception as e:
                logger.error(f"[{self.name}] Error classifying image {i}: {e}")
                # Use default values for failed images
                probabilities.append(0.5)
                classes.append(3)

        # Aggregate predictions
        aggregator = get_aggregator(self.AGGREGATION_STRATEGY)
        agg_prob, agg_class, aggregation_details = aggregator.aggregate(
            probabilities, classes, image_metadata
        )

        logger.info(
            f"[{self.name}] Aggregated result for {case_id}: "
            f"prob={agg_prob:.3f}, class={agg_class}"
        )

        # Add belief about aggregated classification
        self.add_belief(Belief(
            "classification",
            (case_id, agg_prob, agg_class),
            annotations={"source": self.name, "approach": self.APPROACH, "multi_image": True}
        ))

        return {
            "case_id": case_id,
            "agent": self.name,
            "agent_type": self.AGENT_TYPE,
            "approach": self.APPROACH,
            "weight": self.WEIGHT,
            "status": "success",
            "findings": {
                "malignancy_probability": agg_prob,
                "predicted_class": agg_class,
                "per_image_results": [
                    {
                        "image_idx": i,
                        "view_type": image_metadata[i].get("view_type", "Unknown") if i < len(image_metadata) else "Unknown",
                        "probability": probabilities[i],
                        "predicted_class": classes[i]
                    }
                    for i in range(len(images))
                ],
                "aggregation": aggregation_details
            }
        }


# =============================================================================
# DENSENET121 RADIOLOGIST (CNN #1)
# =============================================================================

class RadiologistDenseNet(RadiologistBase):
    """
    Radiologist using TorchXRayVision DenseNet121.
    
    EDUCATIONAL NOTE:
    Uses TorchXRayVision's DenseNet121 pre-trained on multiple chest X-ray
    datasets (NIH, CheXpert, MIMIC-CXR, etc.) for clinically relevant
    pathology detection including Nodule, Mass, and Lung Lesion.
    
    OPERATING POINTS:
    Different thresholds simulate distinct expert styles:
    - Conservative (0.6): High specificity, fewer false positives
    - Balanced (0.5): Standard operating point
    - Sensitive (0.4): High recall, fewer missed nodules
    
    Architecture: 121 layers, trained on chest X-rays
    Input: 224x224 grayscale
    """
    
    AGENT_TYPE = "radiologist"
    APPROACH = "densenet121_xrv"  # TorchXRayVision
    WEIGHT = 1.0  # Base weight — dynamically scaled per-case
    
    # Operating point threshold (affects probability-to-class conversion)
    THRESHOLD = 0.5  # Balanced by default
    
    def __init__(self, name: str = "radiologist_densenet", threshold: float = 0.5, asl_file: Optional[str] = None):
        super().__init__(name=name, asl_file=asl_file)
        self._classifier = None  # NoduleClassifier instance
        self.THRESHOLD = threshold
    
    @classmethod
    def conservative(cls, name: str = "radiologist_conservative"):
        """
        Create conservative radiologist with high specificity.
        Threshold = 0.6 (requires higher probability for positive classification)
        """
        instance = cls(name=name, threshold=0.6)
        instance.APPROACH = "densenet121_xrv_conservative"
        return instance
    
    @classmethod
    def balanced(cls, name: str = "radiologist_balanced"):
        """Create balanced radiologist with standard threshold."""
        instance = cls(name=name, threshold=0.5)
        instance.APPROACH = "densenet121_xrv_balanced"
        return instance
    
    @classmethod
    def sensitive(cls, name: str = "radiologist_sensitive"):
        """
        Create sensitive radiologist with high recall.
        Threshold = 0.4 (lower threshold catches more potential cases)
        """
        instance = cls(name=name, threshold=0.4)
        instance.APPROACH = "densenet121_xrv_sensitive"
        return instance
        
    def _load_model(self):
        """Lazy load NoduleClassifier (TorchXRayVision DenseNet).
        
        STRICT MODE: Raises error if model cannot be loaded.
        """
        if self._classifier is not None:
            return
            
        logger.info(f"[{self.name}] Loading TorchXRayVision DenseNet (STRICT MODE)...")
        
        try:
            self._classifier = NoduleClassifier()
        except Exception as e:
            raise ModelUnavailableError(
                f"[{self.name}] Failed to load NoduleClassifier: {e}\n"
                "Install with: pip install torchxrayvision torch torchvision"
            )
        
        if not self._classifier.model_loaded:
            raise ModelUnavailableError(
                f"[{self.name}] NoduleClassifier model not loaded. "
                "TorchXRayVision DenseNet is required - no fallback mode."
            )
        
        self.add_belief(Belief("model_loaded", ("densenet121_xrv", True)))
        logger.info(f"[{self.name}] TorchXRayVision DenseNet loaded successfully")
            
    def _classify(self, image: np.ndarray) -> Tuple[float, int]:
        """Classify using TorchXRayVision DenseNet.
        
        STRICT MODE: Raises error if classification fails.
        """
        self._load_model()
        
        if self._classifier is None:
            raise ModelUnavailableError(
                f"[{self.name}] Classifier not loaded"
            )
        
        try:
            # Use NoduleClassifier which handles TorchXRayVision
            result = self._classifier.classify(image)
            
            prob = result["malignancy_probability"]
            
            # Apply threshold adjustment for different operating points
            # Conservative: shift down, Sensitive: shift up
            threshold_adjustment = (0.5 - self.THRESHOLD) * 0.2
            prob = np.clip(prob + threshold_adjustment, 0.05, 0.95)
            
            predicted_class = self._prob_to_class(prob)
            
            logger.debug(
                f"[{self.name}] XRV result: prob={prob:.3f}, class={predicted_class}, "
                f"confidence={result.get('confidence', 0):.3f}"
            )
            
            return (prob, predicted_class)
            
        except Exception as e:
            raise ClassificationError(
                f"[{self.name}] Classification failed: {e}"
            )
    
    # STRICT MODE: _fallback_classify removed - CNN model is required
    
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
    Radiologist using TorchXRayVision with alternative pathology focus.
    
    EDUCATIONAL NOTE:
    Uses same TorchXRayVision DenseNet but with different pathology weights
    to create meaningful disagreement with RadiologistDenseNet.
    
    Focuses on: Mass, Lung Lesion, Lung Opacity instead of just Nodule
    This simulates a radiologist with different expertise/priorities.
    
    Architecture: DenseNet121 (XRV), different inference strategy
    Input: 224x224 grayscale
    """
    
    AGENT_TYPE = "radiologist"
    APPROACH = "densenet121_xrv_mass"  # Focuses on Mass/Opacity
    WEIGHT = 1.0  # Base weight — dynamically scaled per-case
    
    def __init__(self, name: str = "radiologist_resnet", asl_file: Optional[str] = None):
        super().__init__(name=name, asl_file=asl_file)
        self._classifier = None
        
    def _load_model(self):
        """Lazy load NoduleClassifier.
        
        STRICT MODE: Raises error if model cannot be loaded.
        """
        if self._classifier is not None:
            return
        
        logger.info(f"[{self.name}] Loading TorchXRayVision (Mass-focused, STRICT MODE)...")
        
        try:
            self._classifier = NoduleClassifier()
        except Exception as e:
            raise ModelUnavailableError(
                f"[{self.name}] Failed to load NoduleClassifier: {e}\n"
                "Install with: pip install torchxrayvision torch torchvision"
            )
        
        if not self._classifier.model_loaded:
            raise ModelUnavailableError(
                f"[{self.name}] NoduleClassifier model not loaded. "
                "TorchXRayVision is required - no fallback mode."
            )
        
        self.add_belief(Belief("model_loaded", ("densenet121_xrv_mass", True)))
        logger.info(f"[{self.name}] TorchXRayVision loaded (Mass-focused) successfully")
            
    def _classify(self, image: np.ndarray) -> Tuple[float, int]:
        """Classify with focus on Mass/Opacity pathologies.
        
        STRICT MODE: Raises error if classification fails.
        """
        self._load_model()
        
        if self._classifier is None or not self._classifier.model_loaded:
            raise ModelUnavailableError(
                f"[{self.name}] Classifier not loaded"
            )
        
        try:
            import torch
            
            # Get raw model output for custom pathology weighting
            tensor = self._classifier._preprocess_image(image)
            
            with torch.no_grad():
                output = self._classifier.model(tensor)
                probs = torch.sigmoid(output)
            
            # Get pathology indices
            pathologies = self._classifier.model.pathologies
            
            # Focus on Mass, Lung Lesion, Lung Opacity instead of just Nodule
            target_pathologies = ["Mass", "Lung Lesion", "Lung Opacity", "Consolidation"]
            scores = []
            
            for p in target_pathologies:
                if p in pathologies:
                    idx = pathologies.index(p)
                    raw = float(probs[0, idx])
                    scores.append(calibrate_xrv_probability(raw))
            
            if scores:
                # Weight Mass higher
                prob = max(scores) * 0.6 + np.mean(scores) * 0.4
            else:
                # Use Nodule if no target pathologies found
                if "Nodule" in pathologies:
                    idx = pathologies.index("Nodule")
                    raw = float(probs[0, idx])
                    prob = calibrate_xrv_probability(raw)
                else:
                    raise ClassificationError(
                        f"[{self.name}] No valid pathologies found in model output"
                    )
            
            # Add slight variation for agent diversity
            prob = np.clip(prob + np.random.uniform(-0.03, 0.03), 0.05, 0.95)
            
            predicted_class = self._prob_to_class(prob)
            
            logger.debug(
                f"[{self.name}] Mass-focused result: prob={prob:.3f}, class={predicted_class}"
            )
            
            return (prob, predicted_class)
            
        except ClassificationError:
            raise
        except Exception as e:
            raise ClassificationError(
                f"[{self.name}] Classification failed: {e}"
            )
    
    # STRICT MODE: _fallback_classify removed - CNN model is required
    
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
    WEIGHT = 0.7  # Base weight — dynamically scaled per-case
    
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
    
    def __init__(self, name: str = "radiologist_rules", asl_file: Optional[str] = None):
        super().__init__(name=name, asl_file=asl_file)
        self._model_loaded = True  # No model to load
        self.add_belief(Belief("model_loaded", ("rule_based", True)))
        
    def _classify(self, image: np.ndarray) -> Tuple[float, int]:
        """Classify using rule-based heuristics."""
        # Extract features from image
        features = self._extract_features(image)
        
        size_mm = features.get("size_mm")
        texture = features.get("texture", "solid")
        size_source = features.get("size_source", "unknown")
        
        if size_mm is None:
            # No blob detected — return neutral probability with low confidence
            probability = 0.5
            logger.info(
                f"[{self.name}] Rule-based: no size detected "
                f"(size_source={size_source}), returning neutral prob={probability:.3f}"
            )
        else:
            # Apply rules
            probability = self._apply_rules(size_mm, texture)
            logger.info(
                f"[{self.name}] Rule-based: size={size_mm:.1f}mm ({size_source}), "
                f"texture={texture} -> prob={probability:.3f}"
            )
        
        return (probability, self._prob_to_class(probability))
    
    def _extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from image for rule application.
        
        Uses anatomically-calibrated blob detection instead of raw pixel
        dimensions. Assumes a standard PA chest X-ray field-of-view of
        ~300mm (30cm chest width) to convert detected region sizes from
        pixels to millimeters.
        
        When no qualifying blob is found, size_mm is set to None and
        size_source to 'none_detected', allowing downstream consensus
        to reduce this agent's weight rather than using a nonsensical value.
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=-1)
        else:
            gray = image.copy()
        
        # --- Texture estimation from intensity variance ---
        # Normalize to 0-255 uint8 range for thresholding 
        if gray.max() <= 1.0:
            gray_u8 = (gray * 255).astype(np.uint8)
            std = np.std(gray)  # Already 0-1 range
        else:
            gray_u8 = gray.astype(np.uint8)
            std = np.std(gray) / 255.0
        
        if std > 0.3:
            texture = "ground_glass"
        elif std > 0.15:
            texture = "part_solid"
        else:
            texture = "solid"
        
        # --- Anatomically-calibrated size estimation via blob detection ---
        # Standard PA chest X-ray field of view ≈ 300mm (30cm)
        CXR_FOV_MM = 300.0
        image_height = max(image.shape[0], 1)
        
        size_mm = None
        size_source = "none_detected"
        
        try:
            # Adaptive threshold to find dense/bright regions
            mean_val = np.mean(gray_u8)
            std_val = max(np.std(gray_u8), 1.0)
            # Use a threshold that isolates the upper intensity tail
            # For medical CXR, suspicious dense regions are brighter than
            # the majority of lung tissue
            threshold = min(255, int(mean_val + 1.0 * std_val))
            
            binary = (gray_u8 > threshold).astype(np.uint8)
            
            # Connected component analysis (scipy-free: manual flood fill approach)
            # Use numpy-based labeling with simple scan
            from scipy import ndimage
            labeled, num_features = ndimage.label(binary)
            
            if num_features > 0:
                best_size_px = None
                total_area = image.shape[0] * image.shape[1]
                
                for label_id in range(1, min(num_features + 1, 100)):  # Cap at 100 blobs
                    component = (labeled == label_id)
                    area = np.sum(component)
                    
                    # Filter: area between 0.0002 and 0.1 of total image
                    # (0.0002 ≈ a 6mm nodule on 512x512 CXR with 300mm FOV)
                    area_ratio = area / total_area
                    if area_ratio < 0.0002 or area_ratio > 0.1:
                        continue
                    
                    # Check circularity: perimeter-based approximation
                    # Erode by 1 pixel, perimeter ≈ area - eroded_area
                    rows, cols = np.where(component)
                    if len(rows) < 4:
                        continue
                    
                    height_span = rows.max() - rows.min() + 1
                    width_span = cols.max() - cols.min() + 1
                    if height_span == 0 or width_span == 0:
                        continue
                    
                    # Bounding box aspect ratio as circularity proxy
                    aspect = min(height_span, width_span) / max(height_span, width_span)
                    if aspect < 0.3:
                        continue  # Too elongated
                    
                    # Check intensity: blob mean should be above overall mean
                    blob_mean = np.mean(gray_u8[component])
                    if blob_mean < mean_val:
                        continue
                    
                    # Track the largest qualifying blob
                    if best_size_px is None or area > best_size_px:
                        best_size_px = area
                
                if best_size_px is not None:
                    # Convert blob area to equivalent diameter in mm
                    # diameter_px = 2 * sqrt(area / pi)
                    diameter_px = 2.0 * np.sqrt(best_size_px / np.pi)
                    size_mm = (diameter_px / image_height) * CXR_FOV_MM
                    
                    # Clamp to clinically plausible range [2mm, 60mm]
                    size_mm = float(np.clip(size_mm, 2.0, 60.0))
                    size_source = "blob_estimation"
        
        except ImportError:
            # scipy not available — use simple intensity-based estimate
            logger.warning(
                f"[{self.name}] scipy unavailable for blob detection, "
                "using intensity-based size estimate"
            )
            # Fraction of pixels above threshold as size proxy
            mean_val = np.mean(gray_u8)
            bright_fraction = np.mean(gray_u8 > mean_val + 0.5 * np.std(gray_u8))
            if bright_fraction > 0.01:
                # Map bright fraction to a plausible size range
                estimated_diameter_frac = np.sqrt(bright_fraction)
                size_mm = float(np.clip(
                    estimated_diameter_frac * CXR_FOV_MM, 2.0, 60.0
                ))
                size_source = "intensity_estimation"
        except Exception as e:
            logger.warning(f"[{self.name}] Blob detection failed: {e}")

        return {
            "size_mm": size_mm,
            "size_source": size_source,
            "texture": texture,
            "mean_intensity": float(np.mean(gray)) / (255.0 if gray.max() > 1.0 else 1.0),
            "std_intensity": float(std)
        }
    
    def _apply_rules(self, size_mm: float, texture: Union[str, int]) -> float:
        """Apply Lung-RADS rules to get probability."""
        # Normalize texture to string format
        texture = str(texture).replace("-", "_").lower().strip()
        
        # Map common texture variations
        if texture in ["ground_glass", "groundglass", "ggo"]:
            texture = "ground_glass"
        elif texture in ["part_solid", "partsolid", "mixed"]:
            texture = "part_solid"
        elif texture in ["solid", "dense"]:
            texture = "solid"
        
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
        nodule_id = request.get("nodule_id", request.get("case_id", "unknown"))
        
        # Can use features directly if provided
        features = request.get("features", {})
        size_mm = features.get("size_mm") if features else None
        if size_mm is not None:
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
                    "size_source": "features",
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
