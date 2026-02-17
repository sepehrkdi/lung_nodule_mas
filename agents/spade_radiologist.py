"""
SPADE-BDI Radiologist Agent
============================

EDUCATIONAL PURPOSE - BDI AGENT WITH COMPUTER VISION:

This module implements the Radiologist agent using SPADE-BDI.
The agent uses DenseNet121 for image classification, with the
ML code called as internal actions from AgentSpeak plans.

SPADE-BDI INTEGRATION:
- Plans in .asl file define WHAT to do
- Internal actions in Python define HOW to do it
- Beliefs track classification results
- Messages sent via XMPP to other agents
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from agents.spade_base import MedicalAgentBase, Belief, get_asl_path

logger = logging.getLogger(__name__)


class RadiologistAgent(MedicalAgentBase):
    """
    SPADE-BDI Radiologist Agent with DenseNet121 classifier.
    
    This agent:
    1. Receives image analysis requests
    2. Runs DenseNet121 classification (internal action)
    3. Extracts visual features (internal action)
    4. Sends findings to Oncologist
    
    The ML code is encapsulated in internal actions that can
    be called from AgentSpeak plans.
    """
    
    def __init__(self, name: str = "radiologist", asl_file: Optional[str] = None):
        """
        Initialize the Radiologist agent.
        
        Args:
            name: Agent name (used for JID in SPADE)
            asl_file: Path to AgentSpeak plans file
        """
        if asl_file is None:
            asl_file = get_asl_path("radiologist")
        
        super().__init__(name=name, asl_file=asl_file)
        
        # Classifier will be lazy-loaded
        self._classifier = None
        self._model_loaded = False
        
        logger.info(f"[{self.name}] Agent created")
    
    def _register_actions(self) -> None:
        """
        Register internal actions callable from AgentSpeak.
        
        EDUCATIONAL NOTE:
        These functions are called from .asl plans using the
        .action_name(args) syntax. They bridge symbolic planning
        with subsymbolic ML processing.
        """
        self.internal_actions = {
            "load_classifier": self._action_load_classifier,
            "classify_image": self._action_classify_image,
            "extract_features": self._action_extract_features,
        }
    
    # =========================================================================
    # Internal Actions (called from AgentSpeak)
    # =========================================================================
    
    def _action_load_classifier(self) -> bool:
        """
        Internal action: Load the DenseNet121 classifier.
        
        Called from ASL: .load_classifier
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from models.classifier import NoduleClassifier
            
            logger.info(f"[{self.name}] Loading DenseNet121 classifier...")
            self._classifier = NoduleClassifier()
            self._model_loaded = True
            
            # Update beliefs
            self.add_belief(Belief("model_loaded", (True,)))
            self.add_belief(Belief("model_name", ("DenseNet121",)))
            
            logger.info(f"[{self.name}] Classifier loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"[{self.name}] Failed to load classifier: {e}")
            self.add_belief(Belief("model_error", (str(e),)))
            return False
    
    def _action_classify_image(
        self,
        nodule_id: str,
        image_data: Any
    ) -> Tuple[float, int]:
        """
        Internal action: Classify nodule image using DenseNet121.
        
        Called from ASL: .classify_image(NoduleId, ImageData, Probability, PredictedClass)
        
        EDUCATIONAL NOTE:
        This is where the actual ML inference happens.
        The AgentSpeak plan orchestrates WHEN to call this,
        but the Python code defines HOW classification works.
        
        Args:
            nodule_id: Unique identifier for the nodule
            image_data: Image as numpy array or path
            
        Returns:
            Tuple of (probability, predicted_class)
        """
        if not self._model_loaded:
            self._action_load_classifier()
        
        if self._classifier is None:
            logger.warning(f"[{self.name}] Classifier not available")
            return (0.5, 0)  # Default to benign
        
        # Handle different image input types
        image = self._prepare_image(image_data)
        
        # Run classification
        try:
            result = self._classifier.classify(image)
            
            probability = result.get("malignancy_probability", 0.5)
            predicted_class = result.get("predicted_class", 0)
            
            # Add beliefs about classification
            self.add_belief(Belief(
                "classification",
                (nodule_id, probability, predicted_class),
                annotations={"source": "self"}
            ))
            
            logger.info(
                f"[{self.name}] Classified {nodule_id}: "
                f"prob={probability:.3f}, class={predicted_class}"
            )
            
            return (probability, predicted_class)
            
        except Exception as e:
            logger.error(f"[{self.name}] Classification error: {e}")
            return (0.5, 0)
    
    def _action_extract_features(
        self,
        nodule_id: str,
        image_data: Any
    ) -> Tuple[float, str, str]:
        """
        Internal action: Extract visual features from image.
        
        Called from ASL: .extract_features(NoduleId, ImageData, Size, Texture, Shape)
        
        Args:
            nodule_id: Unique identifier for the nodule
            image_data: Image as numpy array or path
            
        Returns:
            Tuple of (size_mm, texture, shape)
        """
        if self._classifier is None:
            self._action_load_classifier()
        
        image = self._prepare_image(image_data)
        
        try:
            if self._classifier:
                result = self._classifier.classify(image)
                stats = result.get("image_stats", {})
            else:
                stats = self._analyze_image_stats(image)
            
            size_mm = stats.get("estimated_size", 10.0)
            
            # Determine texture from image statistics
            std = stats.get("std_intensity", 0.1)
            if std > 0.3:
                texture = "heterogeneous"
            elif std > 0.15:
                texture = "part_solid"
            else:
                texture = "solid"
            
            # Determine shape from edge strength
            edge = stats.get("edge_strength", 0.1)
            if edge > 0.2:
                shape = "irregular"
            elif edge > 0.1:
                shape = "lobulated"
            else:
                shape = "round"
            
            # Add beliefs
            self.add_belief(Belief(
                "visual_features",
                (nodule_id, size_mm, texture, shape),
                annotations={"source": "self"}
            ))
            
            logger.info(
                f"[{self.name}] Features for {nodule_id}: "
                f"size={size_mm:.1f}mm, texture={texture}, shape={shape}"
            )
            
            return (size_mm, texture, shape)
            
        except Exception as e:
            logger.error(f"[{self.name}] Feature extraction error: {e}")
            return (10.0, "unknown", "unknown")
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _prepare_image(self, image_data: Any) -> np.ndarray:
        """Prepare image data for classification."""
        if isinstance(image_data, np.ndarray):
            return image_data
        
        if isinstance(image_data, str):
            # Load from file path
            try:
                from PIL import Image
                img = Image.open(image_data)
                return np.array(img)
            except Exception as e:
                logger.warning(f"Could not load image from {image_data}: {e}")
        
        if isinstance(image_data, dict):
            # Features dict - generate synthetic image
            return self._generate_synthetic_image(image_data)
        
        # Default: generate random image
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
        
        # Add texture
        texture = features.get("texture", "solid")
        if texture in ["ground_glass", "ground-glass"]:
            noise = np.random.randint(0, 50, (size, size), dtype=np.uint8)
            image = np.clip(image.astype(int) + noise - 25, 0, 255).astype(np.uint8)
        elif texture == "part_solid":
            noise = np.random.randint(0, 30, (size, size), dtype=np.uint8)
            image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def _analyze_image_stats(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze basic image statistics."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=-1)
        else:
            gray = image
        
        if gray.max() > 1:
            gray = gray / 255.0
        
        return {
            "mean_intensity": float(np.mean(gray)),
            "std_intensity": float(np.std(gray)),
            "estimated_size": float(np.sqrt(gray.shape[0] * gray.shape[1]) / 5),
            "edge_strength": float(np.mean(np.abs(np.diff(gray))))
        }
    
    # =========================================================================
    # Main Processing Interface
    # =========================================================================
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an analysis request.
        
        This is the main entry point called by the MAS orchestrator
        or when a message is received.
        
        Args:
            request: Dictionary with 'nodule_id' and optional 'image' or 'features'
            
        Returns:
            Analysis results dictionary
        """
        nodule_id = request.get("nodule_id", "unknown")
        image_data = request.get("image") or request.get("features", {})
        
        logger.info(f"[{self.name}] Processing request for {nodule_id}")
        
        # Run classification
        probability, predicted_class = self._action_classify_image(nodule_id, image_data)
        
        # Extract features
        size_mm, texture, shape = self._action_extract_features(nodule_id, image_data)
        
        result = {
            "nodule_id": nodule_id,
            "agent": self.name,
            "status": "success",
            "findings": {
                "malignancy_probability": probability,
                "predicted_class": predicted_class,
                "estimated_size_mm": size_mm,
                "texture": texture,
                "shape": shape
            }
        }
        
        return result


# =============================================================================
# SPADE-BDI Integration
# =============================================================================

def create_spade_radiologist(xmpp_config=None):
    """
    Create a SPADE-BDI Radiologist agent.
    
    Args:
        xmpp_config: XMPP server configuration
        
    Returns:
        SPADE-BDI agent instance or standalone agent
    """
    from agents.spade_base import create_spade_bdi_agent, DEFAULT_XMPP_CONFIG
    
    if xmpp_config is None:
        xmpp_config = DEFAULT_XMPP_CONFIG
    
    asl_file = get_asl_path("radiologist")
    
    return create_spade_bdi_agent(
        agent_class=RadiologistAgent,
        name="radiologist",
        xmpp_config=xmpp_config,
        asl_file=asl_file
    )


# =============================================================================
# Standalone Testing
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== Radiologist Agent Test ===\n")
    
    # Create agent
    agent = RadiologistAgent()
    
    # Test with synthetic features
    test_request = {
        "nodule_id": "test_001",
        "features": {
            "size_mm": 15,
            "texture": "solid",
            "malignancy": 4
        }
    }
    
    # Run async processing
    async def test():
        result = await agent.process_request(test_request)
        print("\nResult:")
        for key, value in result.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        print("\nBeliefs:")
        for belief in agent.beliefs:
            print(f"  {belief}")
    
    asyncio.run(test())
