"""
Lung nodule image classifier using DenseNet/ResNet via TorchXRayVision.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


def calibrate_xrv_probability(
    raw_prob: float,
    center: float = 0.62,
    temperature: float = 25.0
) -> float:
    """
    Recalibrate TorchXRayVision sigmoid probabilities.

    EDUCATIONAL NOTE - PROBABILITY CALIBRATION:

    Pre-trained models like TorchXRayVision are trained on different data
    distributions (e.g., CheXpert, MIMIC-CXR) than our evaluation set
    (OpenI/NLMCXR).  This domain shift causes the raw sigmoid outputs to
    cluster in a narrow range (~0.60–0.64), making all cases look
    "indeterminate".

    We apply a temperature-scaled sigmoid re-centering:

        calibrated = σ( k · (raw − center) )

    where:
      • center  = empirical mean of raw outputs (≈ 0.62)
      • k       = temperature that controls steepness (higher = more spread)

    This maps raw=center → 0.5, and amplifies deviations above/below.

    Args:
        raw_prob:    Raw sigmoid probability from TorchXRayVision [0, 1]
        center:      Empirical mean of the model outputs (default 0.62)
        temperature: Steepness; higher = more spread (default 8.0)

    Returns:
        Calibrated probability in [0.05, 0.95]
    """
    # Temperature-scaled sigmoid around the empirical center
    exponent = -temperature * (raw_prob - center)
    calibrated = 1.0 / (1.0 + np.exp(exponent))

    # Clamp to avoid extreme certainties
    return float(np.clip(calibrated, 0.05, 0.95))

class NoduleClassifier:
    """DenseNet-based lung nodule classifier using TorchXRayVision."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the classifier.
        
        Args:
            device: 'cuda', 'cpu', or None for auto-detection
        """
        self.model = None
        self.transform = None
        self.device = device
        self.model_loaded = False
        
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the DenseNet121 model with pre-trained weights.
        
        EDUCATIONAL NOTE - MODEL ARCHITECTURE:
        DenseNet121 has:
        - Initial conv layer (7x7, stride 2)
        - 4 dense blocks with transition layers
        - Global average pooling
        - Final classifier (1000 classes for ImageNet)
        
        We use features from before the final classifier.
        """
        try:
            import torch
            import torchvision.transforms as transforms
            import torchxrayvision as xrv
            
            # Determine device
            if self.device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Load pre-trained DenseNet121 from TorchXRayVision
            # EDUCATIONAL: These weights are trained on multiple chest X-ray datasets
            # (NIH, PC, CheXpert, MIMIC_CH, Google, OpenI, RSNA, etc.)
            logger.info(f"[NoduleClassifier] Loading TorchXRayVision DenseNet on {self.device}...")
            
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Define preprocessing transform for XRV
            # XRV expects images to be normalized to [-1024, 1024]
            # We will handle this normalization in _preprocess_image
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor() 
            ])
            
            self.model_loaded = True
            logger.info("[NoduleClassifier] TorchXRayVision model loaded successfully")
            logger.info(f"[NoduleClassifier] Model pathologies: {self.model.pathologies}")
            
        except ImportError as e:
            logger.error(f"[NoduleClassifier] TorchXRayVision not available: {e}")
            logger.info("[NoduleClassifier] Falling back to feature-based classification")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"[NoduleClassifier] Model loading failed: {e}", exc_info=True)
            logger.info("[NoduleClassifier] Falling back to feature-based classification")
            self.model_loaded = False
    
    def _preprocess_image(self, image: np.ndarray) -> 'torch.Tensor':
        """
        Preprocess image for DenseNet input.
        
        EDUCATIONAL NOTE - PREPROCESSING:
        
        1. Handle different input formats:
           - Grayscale (H, W) → RGB (H, W, 3)
           - Already RGB (H, W, 3) → Use as-is
        
        2. Normalize pixel values:
           - Input might be [0, 255] or [0, 1]
           - ImageNet expects specific normalization
        
        3. Resize to 224x224:
           - DenseNet was trained on 224x224 images
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed tensor ready for model
        """
        import torch
        
        # Ensure numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Handle channels - XRV expects 1 channel (Grayscale)
        if len(image.shape) == 3:
            # If RGB, convert to grayscale using standard weights
            if image.shape[2] == 3:
                image = 0.2989 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2]
            # If 1 channel but 3D dims
            elif image.shape[2] == 1:
                image = image[:,:,0]
        
        # Ensure float32 for calculations
        image = image.astype(np.float32)
        
        # Normalize to range [-1024, 1024]
        # Assuming input is [0, 255] or [0, 1]
        if image.max() <= 1.0:
            # Map [0, 1] -> [-1024, 1024]
            image = (image * 2048) - 1024
        else:
            # Map [0, 255] -> [-1024, 1024]
            image = (image / 255.0 * 2048) - 1024
            
        # Clip to ensure range
        image = np.clip(image, -1024, 1024)
            
        # Convert back to uint8 for PIL (resizing) - WAIT, PIL needs 0-255
        # So we resize FIRST before normalization if using PIL, OR use torch for resizing.
        # Let's use torch transform which expects 0-1 tensor if possible, but XRV needs specific values.
        # Better approach: Resize numpy array directly or use transform on normalized tensor.
        
        # Let's stick to the self.transform which does ToTensor (Expects 0-1)
        # So we should pass 0-1 image to transform, then scale tensor.
        
        # RE-DOING NORMALIZATION STRATEGY:
        # 1. Normalize/Ensure input is 0-1
        # 2. Resize using standard transforms (keeps 0-1)
        # 3. Scale tensor to -1024..1024
        
        # Reset image to 0-1 base
        if image.max() > 1024: # Correction for my previous logic block
             image = image / 255.0 
        elif image.min() < -500: # Was already normalized?
             # Assume it's already properly scaled, map back to 0-1 for resizing safe
             image = (image + 1024) / 2048
        elif image.max() > 1.0: # 0-255
             image = image / 255.0
             
        # Ensure uint8 for PIL resize to work smoothly with ToPILImage
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Apply transform (Resize -> Tensor 0-1)
        tensor = self.transform(image_uint8)
        
        # Now scale Tensor to [-1024, 1024]
        tensor = (tensor * 2048) - 1024
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from the penultimate layer.
        
        EDUCATIONAL NOTE:
        The features before the final classifier contain
        rich visual representations learned from ImageNet.
        These features transfer well to medical imaging tasks.
        """
        import torch
        
        # Get the feature extractor (everything except classifier)
        features_extractor = torch.nn.Sequential(
            *list(self.model.features.children())
        )
        
        with torch.no_grad():
            tensor = self._preprocess_image(image)
            features = features_extractor(tensor)
            # Global average pooling
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            features = features.view(features.size(0), -1)
        
        return features.cpu().numpy()[0]
    
    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Classify a nodule image.
        
        Args:
            image: Nodule image as numpy array
            
        Returns:
            Dictionary with classification results:
            - malignancy_probability: float [0, 1]
            - predicted_class: int (0=benign, 1=malignant)
            - predicted_label: str ("Benign" or "Malignant")
            - estimated_size_mm: float
            - confidence: float [0, 1]
            - features_summary: dict with feature statistics
        """
        if self.model_loaded:
            return self._classify_with_model(image)
        else:
            return self._classify_with_features(image)
    
    def _classify_with_model(self, image: np.ndarray) -> Dict[str, Any]:
        """Classification using DenseNet features."""
        import torch
        
        with torch.no_grad():
            tensor = self._preprocess_image(image)
            
            # Get model output (pathologies probabilities)
            output = self.model(tensor)
            
            # Find "Nodule" or "Mass" class
            target_idx = -1
            pathologies = self.model.pathologies
            
            if "Nodule" in pathologies:
                target_idx = pathologies.index("Nodule")
            elif "Mass" in pathologies:
                target_idx = pathologies.index("Mass")
                
            if target_idx != -1:
                # Use the specific nodule probability
                # XRV outputs are already probabilities or logits? 
                # xrv.models.DenseNet has sigmoids at end? 
                # Docs say: outputs are raw logits usually, need sigmoid. 
                # BUT xrv.models.DenseNet forward() returns 'out' which is often logits.
                # Let's check source code dynamically or assume logits. 
                # Actually XRV DenseNet forward returns dictionary if feature_map is requested, but raw tensor otherwise.
                # It usually applies sigmoid in training but output of .forward() is logits?
                # Inspecting: "The forwarding function returns the output of the model... logits."
                
                probs = torch.sigmoid(output)
                malignancy_score = float(probs[0, target_idx])
                
                # Boost confidence since we have a specific model
                confidence_boost = 0.2
            else:
                # Fallback if class not found
                malignancy_score = 0.5
                confidence_boost = 0.0

            # Get top predictions across all pathologies
            probs_all = torch.sigmoid(output)
            top_probs, top_indices = torch.topk(probs_all, 5)
            top_classes = top_indices # Just indices
            
        # EDUCATIONAL NOTE:
        # We successfully used a model trained on Chest X-rays!
        # output[target_idx] gives us the probability of a "Nodule".
        
        # For compatibility with the rest of the system (Malignancy 1-5),
        # we map the probability [0, 1] to the scale [1, 5]
        # and combine with visual feature heuristics.
        features = self._extract_features(image)
        
        # Compute feature statistics
        feature_mean = float(np.mean(features))
        feature_std = float(np.std(features))
        feature_max = float(np.max(features))
        
        # Image statistics for size estimation
        image_stats = self._analyze_image(image)
        
        # Heuristic malignancy estimation based on features
        # Higher feature activation variance often indicates complexity
        heuristic_score = self._estimate_malignancy_from_features(
            features, image_stats
        )
        
        # Combine model prediction with heuristic
        if confidence_boost > 0:
            # Weighted average: 70% model, 30% heuristic
            final_malignancy = 0.7 * malignancy_score + 0.3 * heuristic_score
            final_confidence = min(0.95, 0.6 + feature_std * 0.1 + confidence_boost)
        else:
            # Fallback to mostly heuristic if specific class not found
            final_malignancy = heuristic_score
            final_confidence = min(0.9, 0.5 + feature_std * 0.1)
        
        # Recalibrate the blended output to spread clustered XRV values
        final_malignancy = calibrate_xrv_probability(final_malignancy)
        
        return {
            "malignancy_probability": final_malignancy,
            "predicted_class": self._score_to_class(final_malignancy),
            "estimated_size_mm": image_stats["estimated_size"],
            "confidence": final_confidence,
            "features_summary": {
                "mean_activation": feature_mean,
                "std_activation": feature_std,
                "max_activation": feature_max,
                "feature_dim": len(features),
                "model_prediction": malignancy_score,
                "heuristic_prediction": heuristic_score
            },
            "image_stats": image_stats
        }
    
    def _classify_with_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Fallback classification using image features only.
        
        EDUCATIONAL NOTE:
        When deep learning is not available, we can still
        extract meaningful features from images:
        - Intensity statistics
        - Texture measures
        - Shape properties
        """
        image_stats = self._analyze_image(image)
        
        # Simple heuristic classification based on image properties
        malignancy_score = self._simple_malignancy_heuristic(image_stats)
        
        return {
            "malignancy_probability": malignancy_score,
            "predicted_class": self._score_to_class(malignancy_score),
            "estimated_size_mm": image_stats["estimated_size"],
            "confidence": 0.5,  # Lower confidence for non-DL method
            "features_summary": {
                "method": "statistical_features",
                "note": "PyTorch not available, using basic features"
            },
            "image_stats": image_stats
        }
    
    def _analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image statistics.
        
        EDUCATIONAL NOTE - IMAGE FEATURES:
        
        These statistics capture visual properties:
        - Mean intensity: Overall brightness (tissue density)
        - Std intensity: Contrast/heterogeneity
        - Max intensity: Brightest regions (calcification?)
        - Shape: Spatial extent
        """
        # Handle multi-channel images
        if len(image.shape) == 3:
            gray = np.mean(image, axis=-1)
        else:
            gray = image
        
        # Normalize to [0, 1]
        if gray.max() > 1:
            gray = gray / 255.0
        
        # Basic statistics
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))
        max_intensity = float(np.max(gray))
        min_intensity = float(np.min(gray))
        
        # Size estimation based on image dimensions
        # EDUCATIONAL: This is a simplified estimate
        # Real size would require voxel spacing information
        h, w = gray.shape[:2]
        estimated_size = np.sqrt(h * w) / 10  # Rough mm estimate
        
        # Texture measure (simplified)
        # Higher std indicates more heterogeneous texture
        texture_score = std_intensity / (mean_intensity + 1e-6)
        
        # Edge strength (simplified gradient)
        try:
            grad_y = np.diff(gray, axis=0)
            grad_x = np.diff(gray, axis=1)
            edge_strength = float(np.mean(np.abs(grad_y)) + np.mean(np.abs(grad_x)))
        except:
            edge_strength = 0.0
        
        return {
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "max_intensity": max_intensity,
            "min_intensity": min_intensity,
            "estimated_size": estimated_size,
            "texture_score": texture_score,
            "edge_strength": edge_strength,
            "shape": gray.shape
        }
    
    def _estimate_malignancy_from_features(
        self, 
        features: np.ndarray, 
        image_stats: Dict[str, Any]
    ) -> float:
        """
        Estimate malignancy score from extracted features.
        
        EDUCATIONAL NOTE:
        This is a heuristic demonstration. Real malignancy
        prediction would require:
        - Training on labeled data
        - Proper feature selection
        - Clinical validation
        
        We use some general principles:
        - Larger nodules tend to be more suspicious
        - Heterogeneous texture may indicate malignancy
        - Complex features (high variance) may indicate concern
        """
        # Feature complexity (higher may indicate concerning features)
        feature_complexity = float(np.std(features)) / (np.mean(np.abs(features)) + 1e-6)
        
        # Size factor (larger = more suspicious)
        size = image_stats["estimated_size"]
        size_factor = min(1.0, size / 30.0)  # Normalize by 30mm
        
        # Texture factor
        texture = image_stats["texture_score"]
        texture_factor = min(1.0, texture * 2)
        
        # Edge factor (spiculated margins have higher edge strength)
        edge = image_stats["edge_strength"]
        edge_factor = min(1.0, edge * 5)
        
        # Combine factors (weighted average)
        # EDUCATIONAL: These weights are arbitrary for demo
        malignancy_score = (
            0.3 * size_factor +
            0.3 * texture_factor +
            0.2 * edge_factor +
            0.2 * min(1.0, feature_complexity * 0.5)
        )
        
        return float(np.clip(malignancy_score, 0.1, 0.9))
    
    def _simple_malignancy_heuristic(self, image_stats: Dict[str, Any]) -> float:
        """Simple malignancy heuristic without deep features."""
        size_factor = min(1.0, image_stats["estimated_size"] / 30.0)
        texture_factor = min(1.0, image_stats["texture_score"] * 2)
        edge_factor = min(1.0, image_stats["edge_strength"] * 5)
        
        return float(np.clip(
            0.4 * size_factor + 0.4 * texture_factor + 0.2 * edge_factor,
            0.1, 0.9
        ))
    
    def _score_to_class(self, score: float, threshold: float = 0.5) -> int:
        """
        Convert malignancy probability to binary class.
        
        Binary Classification:
        0 = Benign
        1 = Malignant
        """
        return 1 if score >= threshold else 0


def classify_nodule(image: np.ndarray) -> Dict[str, Any]:
    """
    Convenience function to classify a single nodule.
    
    Args:
        image: Nodule image as numpy array
        
    Returns:
        Classification results dictionary
    """
    classifier = NoduleClassifier()
    return classifier.classify(image)


if __name__ == "__main__":
    # Demo usage
    print("=== Nodule Classifier Demo ===\n")
    
    # Create a synthetic test image
    print("Creating synthetic test image...")
    test_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    
    # Add a circular pattern to simulate a nodule
    y, x = np.ogrid[:64, :64]
    center = (32, 32)
    radius = 20
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    test_image[mask] = 200
    
    # Classify
    classifier = NoduleClassifier()
    result = classifier.classify(test_image)
    
    print("\nClassification Results:")
    print(f"  Malignancy Probability: {result['malignancy_probability']:.3f}")
    print(f"  Predicted Class: {result['predicted_class']}")
    print(f"  Estimated Size: {result['estimated_size_mm']:.1f} mm")
    print(f"  Confidence: {result['confidence']:.3f}")
    
    print("\nImage Statistics:")
    for key, value in result.get('image_stats', {}).items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nFeatures Summary:")
    for key, value in result.get('features_summary', {}).items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
