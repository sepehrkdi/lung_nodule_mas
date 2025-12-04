"""
Lung Nodule Image Classifier
=============================

EDUCATIONAL PURPOSE - DEEP LEARNING FOR MEDICAL IMAGING:

This module demonstrates computer vision concepts for medical image analysis:

1. CONVOLUTIONAL NEURAL NETWORKS (CNNs):
   - Learn hierarchical features from images
   - Early layers: edges, textures
   - Deep layers: shapes, structures
   - Final layers: high-level semantic features

2. TRANSFER LEARNING:
   - Use pre-trained weights from ImageNet
   - Fine-tune or freeze for new task
   - Reduces need for large medical datasets

3. DENSENET ARCHITECTURE:
   - Dense connections between layers
   - Feature reuse and gradient flow
   - Efficient parameter usage
   - Well-suited for medical imaging

4. IMAGE PREPROCESSING:
   - Normalization to [-1, 1] or [0, 1]
   - Resizing to model input size
   - Channel handling (grayscale → RGB)

WHY DENSENET FOR MEDICAL IMAGING:
- Strong feature extraction with fewer parameters
- Good gradient flow through dense connections
- Proven performance on medical imaging tasks
- Available pre-trained on ImageNet

LIMITATIONS (EDUCATIONAL):
This is a simplified demonstration. Real medical AI would require:
- Large validated datasets
- Proper train/val/test splits
- Clinical validation studies
- Regulatory approval (FDA/CE marking)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import warnings


class NoduleClassifier:
    """
    DenseNet-based lung nodule classifier.
    
    EDUCATIONAL PURPOSE - TRANSFER LEARNING:
    
    This classifier uses a pre-trained DenseNet121 model from ImageNet.
    We adapt it for nodule classification through:
    
    1. Feature Extraction: Use DenseNet as a fixed feature extractor
    2. The final layer features capture visual patterns
    3. We map these to malignancy likelihood
    
    In this educational demo, we use a simplified approach:
    - Extract visual features from the image
    - Use statistical properties as a proxy for classification
    - In production, you would fine-tune on labeled nodule data
    
    Usage:
        classifier = NoduleClassifier()
        result = classifier.classify(nodule_image)
    """
    
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
            import torchvision.models as models
            import torchvision.transforms as transforms
            
            # Determine device
            if self.device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Load pre-trained DenseNet121
            # EDUCATIONAL: These weights are trained on ImageNet
            print(f"[NoduleClassifier] Loading DenseNet121 on {self.device}...")
            
            # Use weights parameter for newer torchvision versions
            try:
                weights = models.DenseNet121_Weights.IMAGENET1K_V1
                self.model = models.densenet121(weights=weights)
            except AttributeError:
                # Fallback for older torchvision
                self.model = models.densenet121(pretrained=True)
            
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Define preprocessing transform
            # EDUCATIONAL: ImageNet normalization is standard practice
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225]    # ImageNet stds
                )
            ])
            
            self.model_loaded = True
            print("[NoduleClassifier] DenseNet121 loaded successfully")
            
        except ImportError as e:
            print(f"[NoduleClassifier] PyTorch not available: {e}")
            print("[NoduleClassifier] Falling back to feature-based classification")
        except Exception as e:
            print(f"[NoduleClassifier] Model loading failed: {e}")
            print("[NoduleClassifier] Falling back to feature-based classification")
    
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
        
        # Handle grayscale images
        if len(image.shape) == 2:
            # Convert to RGB by stacking
            image = np.stack([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Single channel to RGB
            image = np.concatenate([image, image, image], axis=-1)
        
        # Ensure uint8 for PIL
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Apply transform
        tensor = self.transform(image)
        
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
        
        EDUCATIONAL PURPOSE - CLASSIFICATION PIPELINE:
        
        This demonstrates the classification workflow:
        1. Preprocess input image
        2. Extract visual features
        3. Generate classification scores
        4. Estimate physical properties (size)
        
        In a production system, you would:
        - Train a classifier on labeled nodule data
        - Use proper validation metrics
        - Provide calibrated probabilities
        
        For this demo, we use:
        - Visual feature analysis
        - Statistical properties of the image
        - Heuristic size estimation
        
        Args:
            image: Nodule image as numpy array
            
        Returns:
            Dictionary with classification results:
            - malignancy_probability: float [0, 1]
            - predicted_class: int [1-5]
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
            
            # Get model output
            output = self.model(tensor)
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(output, dim=1)
            
            # Get top predictions
            top_probs, top_classes = torch.topk(probs, 5)
            
        # EDUCATIONAL NOTE:
        # The model outputs ImageNet class probabilities.
        # For a real medical classifier, you would:
        # 1. Replace the final layer with a 5-class output (malignancy 1-5)
        # 2. Fine-tune on labeled nodule data
        # 3. Use proper calibration
        
        # For demo, we extract features and use heuristics
        features = self._extract_features(image)
        
        # Compute feature statistics
        feature_mean = float(np.mean(features))
        feature_std = float(np.std(features))
        feature_max = float(np.max(features))
        
        # Image statistics for size estimation
        image_stats = self._analyze_image(image)
        
        # Heuristic malignancy estimation based on features
        # Higher feature activation variance often indicates complexity
        malignancy_score = self._estimate_malignancy_from_features(
            features, image_stats
        )
        
        return {
            "malignancy_probability": malignancy_score,
            "predicted_class": self._score_to_class(malignancy_score),
            "estimated_size_mm": image_stats["estimated_size"],
            "confidence": min(0.9, 0.5 + feature_std * 0.1),  # Heuristic confidence
            "features_summary": {
                "mean_activation": feature_mean,
                "std_activation": feature_std,
                "max_activation": feature_max,
                "feature_dim": len(features)
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
    
    def _score_to_class(self, score: float) -> int:
        """
        Convert malignancy probability to class (1-5).
        
        LIDC-IDRI Malignancy Scale:
        1 = Highly Unlikely
        2 = Moderately Unlikely
        3 = Indeterminate
        4 = Moderately Suspicious
        5 = Highly Suspicious
        """
        if score < 0.2:
            return 1
        elif score < 0.4:
            return 2
        elif score < 0.6:
            return 3
        elif score < 0.8:
            return 4
        else:
            return 5


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
