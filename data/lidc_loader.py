"""
LIDC-IDRI Data Loader
=====================

Loads lung nodule data from either:
1. Prepared subset (data/subset/) - Created by prepare_dataset.py
2. Fallback dataset (data/fallback/) - Pre-included demo data

EDUCATIONAL PURPOSE:
This module handles medical imaging data from the LIDC-IDRI dataset,
which contains CT scans with radiologist annotations. Each nodule has
multiple annotations that we combine using consensus (median).

LIDC-IDRI ANNOTATIONS:
- Size (diameter in mm)
- Malignancy (1-5): Likelihood of cancer
- Spiculation (1-5): Spiky projections (suspicious feature)
- Margin (1-5): Edge definition (lower = more concerning)
- Texture (1-5): Solid vs ground-glass
- Lobulation (1-5): Scalloped edges
- Calcification (1-6): Type of calcification
"""

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Any
import numpy as np
from PIL import Image


class LIDCLoader:
    """
    Loader for LIDC-IDRI lung nodule dataset.
    
    EDUCATIONAL PURPOSE:
    This class demonstrates medical imaging data handling:
    1. Loading CT image patches
    2. Parsing radiologist annotations
    3. Computing consensus labels from multiple annotators
    
    Usage:
        loader = LIDCLoader()
        for nodule_id, image, features in loader.load_all():
            # Process each nodule
            pass
    """
    
    # Directory paths (relative to project root)
    SUBSET_DIR = "data/subset"
    FALLBACK_DIR = "data/fallback"
    
    # Feature names from LIDC annotations
    FEATURE_NAMES = [
        "diameter_mm", "malignancy", "spiculation", "margin",
        "texture", "lobulation", "calcification", "sphericity",
        "subtlety", "internal_structure"
    ]
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Override data directory (auto-detects if None)
        """
        self.project_root = self._find_project_root()
        
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            # Try subset first, fallback to demo data
            subset_path = self.project_root / self.SUBSET_DIR
            fallback_path = self.project_root / self.FALLBACK_DIR
            
            if subset_path.exists() and any(subset_path.glob("*.json")):
                self.data_dir = subset_path
                print(f"[LIDCLoader] Using subset data from {subset_path}")
            elif fallback_path.exists():
                self.data_dir = fallback_path
                print(f"[LIDCLoader] Using fallback data from {fallback_path}")
            else:
                raise FileNotFoundError(
                    "No data found. Run 'python -m data.prepare_dataset --fallback' first."
                )
    
    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        current = Path(__file__).parent.parent
        # Look for requirements.txt or main.py as indicators
        while current != current.parent:
            if (current / "requirements.txt").exists() or (current / "main.py").exists():
                return current
            current = current.parent
        return Path(__file__).parent.parent
    
    def get_nodule_ids(self) -> List[str]:
        """
        Get list of all available nodule IDs.
        
        Returns:
            List of nodule ID strings
        """
        json_files = list(self.data_dir.glob("nodule_*.json"))
        ids = []
        for f in json_files:
            # Extract ID from filename (nodule_001.json -> 001)
            nodule_id = f.stem.replace("nodule_", "")
            ids.append(nodule_id)
        return sorted(ids)
    
    def load_nodule(self, nodule_id: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Load a single nodule's image and features.
        
        Args:
            nodule_id: The nodule identifier (e.g., "001")
            
        Returns:
            Tuple of (image_array, features_dict)
            - image_array: numpy array of shape (H, W) normalized to [0, 1], or None if no image
            - features_dict: Dictionary with all nodule features
        """
        # Load JSON features
        json_path = self.data_dir / f"nodule_{nodule_id}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Nodule {nodule_id} not found at {json_path}")
            
        with open(json_path, 'r') as f:
            features = json.load(f)
        
        # Try to load image (PNG format)
        image = None
        image_path = self.data_dir / f"nodule_{nodule_id}.png"
        if image_path.exists():
            img = Image.open(image_path).convert('L')  # Grayscale
            image = np.array(img).astype(np.float32) / 255.0
        else:
            # Generate synthetic image if not available
            image = self._generate_synthetic_image(features)
            
        return image, features
    
    def _generate_synthetic_image(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Generate a synthetic nodule image for demo purposes.
        
        EDUCATIONAL NOTE:
        When real images aren't available (fallback mode), we generate
        simple synthetic images that reflect the nodule characteristics.
        This is NOT for clinical use - only for demonstrating the system.
        
        Args:
            features: Nodule feature dictionary
            
        Returns:
            64x64 numpy array with synthetic nodule
        """
        size = 64
        image = np.zeros((size, size), dtype=np.float32)
        
        # Get nodule properties
        diameter = features.get("diameter_mm", 10)
        texture = features.get("texture", 5)
        spiculation = features.get("spiculation", 1)
        margin = features.get("margin", 5)
        
        # Create center coordinates
        cx, cy = size // 2, size // 2
        
        # Nodule radius (scaled to image size)
        radius = int(min(diameter * 1.5, 25))  # Cap at 25 pixels
        
        # Create coordinate grids
        y, x = np.ogrid[:size, :size]
        
        # Distance from center
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Base nodule (ellipse with some variation)
        base_intensity = 0.3 + (texture / 5) * 0.5  # 0.3-0.8 based on texture
        
        # Add nodule with smooth edges based on margin
        edge_sharpness = margin / 5 * 3 + 1  # 1-4 based on margin
        nodule_mask = np.clip(1 - (dist - radius) / edge_sharpness, 0, 1)
        
        image += nodule_mask * base_intensity
        
        # Add spiculation (spiky projections)
        if spiculation >= 3:
            n_spikes = spiculation * 2
            for i in range(n_spikes):
                angle = 2 * np.pi * i / n_spikes
                spike_len = radius + spiculation * 2
                spike_x = int(cx + spike_len * np.cos(angle))
                spike_y = int(cy + spike_len * np.sin(angle))
                if 0 <= spike_x < size and 0 <= spike_y < size:
                    # Draw line from edge to spike tip
                    for t in np.linspace(0, 1, 10):
                        px = int(cx + (radius + t * spiculation * 2) * np.cos(angle))
                        py = int(cy + (radius + t * spiculation * 2) * np.sin(angle))
                        if 0 <= px < size and 0 <= py < size:
                            image[py, px] = max(image[py, px], base_intensity * (1 - t * 0.3))
        
        # Add noise for realism
        noise = np.random.randn(size, size) * 0.05
        image = np.clip(image + noise, 0, 1)
        
        # Add background lung texture (darker regions)
        background = np.random.randn(size, size) * 0.03 + 0.1
        background = np.clip(background, 0, 0.2)
        image = np.maximum(image, background)
        
        return image.astype(np.float32)
    
    def load_all(self) -> Iterator[Tuple[str, np.ndarray, Dict[str, Any]]]:
        """
        Load all available nodules.
        
        Yields:
            Tuple of (nodule_id, image_array, features_dict)
        """
        for nodule_id in self.get_nodule_ids():
            try:
                image, features = self.load_nodule(nodule_id)
                yield nodule_id, image, features
            except Exception as e:
                print(f"[LIDCLoader] Warning: Failed to load nodule {nodule_id}: {e}")
                continue
    
    def get_ground_truth(self, nodule_id: str) -> int:
        """
        Get the consensus malignancy rating (ground truth).
        
        EDUCATIONAL NOTE:
        LIDC-IDRI has 4 radiologists annotating each nodule.
        We use the median malignancy rating as ground truth.
        
        Args:
            nodule_id: The nodule identifier
            
        Returns:
            Malignancy score 1-5
        """
        _, features = self.load_nodule(nodule_id)
        return features.get("malignancy", 3)
    
    def get_binary_label(self, nodule_id: str) -> Optional[int]:
        """
        Get binary label (benign/malignant).
        
        EDUCATIONAL NOTE:
        For binary classification, we exclude indeterminate cases (3):
        - Malignancy 1-2: Benign (0)
        - Malignancy 3: Indeterminate (excluded, returns None)
        - Malignancy 4-5: Malignant (1)
        
        Args:
            nodule_id: The nodule identifier
            
        Returns:
            0 for benign, 1 for malignant, None for indeterminate
        """
        malignancy = self.get_ground_truth(nodule_id)
        if malignancy <= 2:
            return 0  # Benign
        elif malignancy >= 4:
            return 1  # Malignant
        else:
            return None  # Indeterminate (malignancy == 3)
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "total_nodules": 0,
            "malignancy_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "benign_count": 0,
            "malignant_count": 0,
            "indeterminate_count": 0,
            "size_range": {"min": float('inf'), "max": 0, "mean": 0}
        }
        
        sizes = []
        for nodule_id in self.get_nodule_ids():
            _, features = self.load_nodule(nodule_id)
            stats["total_nodules"] += 1
            
            malignancy = features.get("malignancy", 3)
            stats["malignancy_distribution"][malignancy] += 1
            
            if malignancy <= 2:
                stats["benign_count"] += 1
            elif malignancy >= 4:
                stats["malignant_count"] += 1
            else:
                stats["indeterminate_count"] += 1
                
            size = features.get("diameter_mm", 0)
            sizes.append(size)
            stats["size_range"]["min"] = min(stats["size_range"]["min"], size)
            stats["size_range"]["max"] = max(stats["size_range"]["max"], size)
        
        if sizes:
            stats["size_range"]["mean"] = sum(sizes) / len(sizes)
        
        return stats
    
    def __len__(self) -> int:
        """Return the number of nodules in the dataset."""
        return len(self.get_nodule_ids())
    
    def __iter__(self):
        """Iterate over all nodules."""
        return self.load_all()


# Convenience function
def load_dataset(data_dir: Optional[str] = None) -> LIDCLoader:
    """
    Factory function to create a data loader.
    
    Args:
        data_dir: Optional override for data directory
        
    Returns:
        Configured LIDCLoader instance
    """
    return LIDCLoader(data_dir)


if __name__ == "__main__":
    # Demo usage
    print("=== LIDC Data Loader Demo ===\n")
    
    try:
        loader = LIDCLoader()
        stats = loader.get_dataset_statistics()
        
        print(f"Dataset Statistics:")
        print(f"  Total nodules: {stats['total_nodules']}")
        print(f"  Benign: {stats['benign_count']}")
        print(f"  Malignant: {stats['malignant_count']}")
        print(f"  Indeterminate: {stats['indeterminate_count']}")
        print(f"  Size range: {stats['size_range']['min']:.1f} - {stats['size_range']['max']:.1f} mm")
        print(f"\nMalignancy distribution: {stats['malignancy_distribution']}")
        
        # Load first nodule as example
        print("\n--- First Nodule ---")
        nodule_ids = loader.get_nodule_ids()
        if nodule_ids:
            nodule_id = nodule_ids[0]
            image, features = loader.load_nodule(nodule_id)
            print(f"Nodule ID: {nodule_id}")
            print(f"Image shape: {image.shape}")
            print(f"Diameter: {features.get('diameter_mm', 'N/A')} mm")
            print(f"Malignancy: {features.get('malignancy', 'N/A')} ({features.get('malignancy_label', 'N/A')})")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'python -m data.prepare_dataset --fallback' to set up demo data.")
