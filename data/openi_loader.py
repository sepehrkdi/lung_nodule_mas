"""
Open-I Indiana University Chest X-ray Data Loader
==================================================

Loads chest X-ray images with paired radiology reports from the 
Open-I Indiana University collection.

DATASET SOURCE:
    https://openi.nlm.nih.gov/
    Indiana University Chest X-ray Collection
    
EDUCATIONAL PURPOSE:
This dataset is ideal for demonstrating:
1. Paired image + text data (CV + NLP agents)
2. Real radiology reports (authentic medical language)
3. Manageable size for educational project (20-50 samples)

DATASET CHARACTERISTICS:
- ~7,500 chest X-rays with radiology reports
- PNG format images (frontal and lateral views)
- XML metadata with findings, impression, indication
- Free-text radiology reports
- We filter for cases mentioning nodules/masses

DATA STRUCTURE:
    data/openi/
    ├── images/
    │   ├── CXR1_1_IM-0001-1001.png
    │   └── ...
    ├── reports/
    │   ├── CXR1_1_IM-0001-1001.xml
    │   └── ...
    └── manifest.json  (processed subset)

FALLBACK:
If Open-I data not downloaded, uses synthetic fallback data
that mimics the Open-I format.
"""

import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Any
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class OpenILoader:
    """
    Loader for Open-I Indiana University Chest X-ray Collection.
    
    EDUCATIONAL PURPOSE:
    This class demonstrates:
    1. Loading paired image-report medical data
    2. Parsing XML radiology reports
    3. Extracting nodule-relevant cases
    
    Usage:
        loader = OpenILoader()
        for case_id in loader.list_cases():
            data = loader.load_case(case_id)
            image = data["image"]
            report = data["report"]
    """
    
    # Directory paths (relative to project root)
    OPENI_DIR = "data/openi"
    FALLBACK_DIR = "data/fallback"
    
    # Keywords for filtering nodule-relevant cases
    NODULE_KEYWORDS = [
        "nodule", "nodular", "mass", "lesion", "opacity",
        "tumor", "neoplasm", "carcinoma", "malignant", "suspicious",
        "pulmonary nodule", "lung nodule", "solitary nodule"
    ]
    
    # Size patterns for extraction
    SIZE_PATTERN = r"(\d+(?:\.\d+)?)\s*(?:mm|cm|millimeter|centimeter)"
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the Open-I data loader.
        
        Args:
            data_dir: Override data directory (auto-detects if None)
        """
        self.project_root = self._find_project_root()
        
        if data_dir:
            self.data_dir = Path(data_dir)
            self.use_fallback = False
        else:
            # Try Open-I data first, then fallback
            openi_path = self.project_root / self.OPENI_DIR
            fallback_path = self.project_root / self.FALLBACK_DIR
            
            if openi_path.exists() and self._has_openi_data(openi_path):
                self.data_dir = openi_path
                self.use_fallback = False
                logger.info(f"[OpenILoader] Using Open-I data from {openi_path}")
            elif fallback_path.exists():
                self.data_dir = fallback_path
                self.use_fallback = True
                logger.info(f"[OpenILoader] Using fallback data from {fallback_path}")
            else:
                # Create fallback
                self._create_fallback_data(fallback_path)
                self.data_dir = fallback_path
                self.use_fallback = True
        
        # Load manifest if available
        self.manifest = self._load_manifest()
    
    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        current = Path(__file__).parent.parent
        while current != current.parent:
            if (current / "requirements.txt").exists() or (current / "main.py").exists():
                return current
            current = current.parent
        return Path(__file__).parent.parent
    
    def _has_openi_data(self, path: Path) -> bool:
        """Check if Open-I data exists."""
        images_dir = path / "images"
        reports_dir = path / "reports"
        return (
            images_dir.exists() and 
            any(images_dir.glob("*.png")) and
            (reports_dir.exists() or (path / "manifest.json").exists())
        )
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load the dataset manifest."""
        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                return json.load(f)
        return {"cases": []}
    
    def _create_fallback_data(self, fallback_path: Path) -> None:
        """Create fallback data directory if needed."""
        fallback_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"[OpenILoader] Created fallback directory at {fallback_path}")
    
    def list_cases(self) -> List[str]:
        """
        Get list of all available case IDs.
        
        Returns:
            List of case identifiers
        """
        if self.use_fallback:
            return self._list_fallback_cases()
        
        if self.manifest.get("cases"):
            return [c["id"] for c in self.manifest["cases"]]
        
        # Scan for image files
        images_dir = self.data_dir / "images"
        if images_dir.exists():
            return [
                p.stem for p in sorted(images_dir.glob("*.png"))
            ]
        
        return self._list_fallback_cases()
    
    def _list_fallback_cases(self) -> List[str]:
        """List fallback/synthetic case IDs."""
        # Check for JSON files in fallback directory
        json_files = list(self.data_dir.glob("*.json"))
        if json_files:
            return [f.stem for f in sorted(json_files)]
        
        # Generate synthetic case IDs
        return [f"case_{i:03d}" for i in range(1, 11)]
    
    # Alias for compatibility with existing code
    def list_nodules(self) -> List[str]:
        """Alias for list_cases() for backward compatibility."""
        return self.list_cases()
    
    def load_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a single case with image and report.
        
        Args:
            case_id: Case identifier
            
        Returns:
            Dictionary with:
            - image: numpy array (grayscale or RGB)
            - report: full radiology report text
            - findings: extracted findings section
            - impression: extracted impression section
            - features: extracted numeric features
            - malignancy: estimated malignancy (1-5)
        """
        if self.use_fallback:
            return self._load_fallback_case(case_id)
        
        return self._load_openi_case(case_id)
    
    # Alias for compatibility
    def load_nodule(self, nodule_id: str) -> Optional[Dict[str, Any]]:
        """Alias for load_case() for backward compatibility."""
        return self.load_case(nodule_id)
    
    def _load_openi_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Load case from Open-I data."""
        # Try to load image
        image = self._load_image(case_id)
        
        # Try to load report
        report_data = self._load_report(case_id)
        
        if image is None and report_data is None:
            logger.warning(f"[OpenILoader] No data found for {case_id}")
            return self._load_fallback_case(case_id)
        
        # Extract features from report
        features = self._extract_features_from_report(report_data or {})
        
        return {
            "case_id": case_id,
            "image": image,
            "report": report_data.get("full_text", "") if report_data else "",
            "findings": report_data.get("findings", "") if report_data else "",
            "impression": report_data.get("impression", "") if report_data else "",
            "indication": report_data.get("indication", "") if report_data else "",
            "features": features,
            "malignancy": features.get("malignancy", 3)
        }
    
    def _load_image(self, case_id: str) -> Optional[np.ndarray]:
        """Load image for a case."""
        images_dir = self.data_dir / "images"
        
        # Try different naming patterns
        patterns = [
            f"{case_id}.png",
            f"{case_id}_1.png",
            f"{case_id}-1001.png"
        ]
        
        for pattern in patterns:
            image_path = images_dir / pattern
            if image_path.exists():
                try:
                    img = Image.open(image_path)
                    return np.array(img)
                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {e}")
        
        return None
    
    def _load_report(self, case_id: str) -> Optional[Dict[str, str]]:
        """Load and parse XML report for a case."""
        reports_dir = self.data_dir / "reports"
        
        # Try different naming patterns
        patterns = [
            f"{case_id}.xml",
            f"{case_id}_report.xml"
        ]
        
        for pattern in patterns:
            report_path = reports_dir / pattern
            if report_path.exists():
                try:
                    return self._parse_xml_report(report_path)
                except Exception as e:
                    logger.warning(f"Failed to parse report {report_path}: {e}")
        
        # Check manifest for report text
        if self.manifest.get("cases"):
            for case in self.manifest["cases"]:
                if case.get("id") == case_id:
                    return {
                        "full_text": case.get("report", ""),
                        "findings": case.get("findings", ""),
                        "impression": case.get("impression", ""),
                        "indication": case.get("indication", "")
                    }
        
        return None
    
    def _parse_xml_report(self, xml_path: Path) -> Dict[str, str]:
        """
        Parse Open-I XML report format.
        
        Open-I XML structure:
        <eCitation>
            <MedlineCitation>
                <Article>
                    <Abstract>
                        <AbstractText Label="FINDINGS">...</AbstractText>
                        <AbstractText Label="IMPRESSION">...</AbstractText>
                    </Abstract>
                </Article>
            </MedlineCitation>
        </eCitation>
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        result = {
            "findings": "",
            "impression": "",
            "indication": "",
            "full_text": ""
        }
        
        # Find abstract sections
        for abstract_text in root.iter("AbstractText"):
            label = abstract_text.get("Label", "").upper()
            text = abstract_text.text or ""
            
            if "FINDING" in label:
                result["findings"] = text
            elif "IMPRESSION" in label:
                result["impression"] = text
            elif "INDICATION" in label:
                result["indication"] = text
        
        # Combine into full text
        parts = []
        if result["indication"]:
            parts.append(f"INDICATION: {result['indication']}")
        if result["findings"]:
            parts.append(f"FINDINGS: {result['findings']}")
        if result["impression"]:
            parts.append(f"IMPRESSION: {result['impression']}")
        
        result["full_text"] = "\n\n".join(parts)
        
        return result
    
    def _extract_features_from_report(self, report_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract numeric features from report text.
        
        EDUCATIONAL NOTE:
        This bridges the NLP output to structured features
        that can be used by the symbolic reasoning agent.
        """
        full_text = report_data.get("full_text", "")
        findings = report_data.get("findings", "")
        impression = report_data.get("impression", "")
        
        text = f"{findings} {impression}".lower()
        
        features = {
            "size_mm": self._extract_size(text),
            "texture": self._extract_texture(text),
            "location": self._extract_location(text),
            "malignancy": self._estimate_malignancy(text),
            "has_nodule_mention": self._has_nodule_mention(text),
            "calcification": self._extract_calcification(text)
        }
        
        return features
    
    def _extract_size(self, text: str) -> float:
        """Extract nodule size from text."""
        matches = re.findall(self.SIZE_PATTERN, text)
        if matches:
            # Take the first size mentioned
            size = float(matches[0])
            # Convert cm to mm if needed
            if "cm" in text[text.find(matches[0]):text.find(matches[0])+20]:
                size *= 10
            return size
        return 10.0  # Default size
    
    def _extract_texture(self, text: str) -> str:
        """Extract nodule texture from text."""
        if "ground glass" in text or "ground-glass" in text:
            return "ground_glass"
        elif "part solid" in text or "part-solid" in text or "partially solid" in text:
            return "part_solid"
        elif "calcif" in text:
            return "calcified"
        else:
            return "solid"
    
    def _extract_location(self, text: str) -> str:
        """Extract anatomical location from text."""
        locations = {
            "right upper": "right_upper_lobe",
            "right middle": "right_middle_lobe",
            "right lower": "right_lower_lobe",
            "left upper": "left_upper_lobe",
            "left lower": "left_lower_lobe",
            "lingula": "lingula",
            "hilum": "hilum",
            "mediastin": "mediastinum"
        }
        
        for pattern, location in locations.items():
            if pattern in text:
                return location
        return "unspecified"
    
    def _estimate_malignancy(self, text: str) -> int:
        """
        Estimate malignancy score (1-5) from report text.
        
        EDUCATIONAL NOTE:
        This is a simplified heuristic for educational purposes.
        Real systems would use trained NLP models.
        """
        # High suspicion indicators
        high_risk = [
            "malignant", "carcinoma", "cancer", "metasta",
            "highly suspicious", "concerning for malignancy",
            "biopsy recommended", "suspicious"
        ]
        
        # Moderate suspicion
        moderate_risk = [
            "indeterminate", "cannot exclude", "possible",
            "follow-up recommended", "further evaluation"
        ]
        
        # Low suspicion / benign
        low_risk = [
            "benign", "stable", "unchanged", "granuloma",
            "calcified", "no change", "resolved"
        ]
        
        high_count = sum(1 for term in high_risk if term in text)
        mod_count = sum(1 for term in moderate_risk if term in text)
        low_count = sum(1 for term in low_risk if term in text)
        
        if high_count >= 2:
            return 5
        elif high_count == 1:
            return 4
        elif mod_count >= 1:
            return 3
        elif low_count >= 2:
            return 1
        elif low_count == 1:
            return 2
        else:
            return 3  # Default to indeterminate
    
    def _has_nodule_mention(self, text: str) -> bool:
        """Check if report mentions nodule-related terms."""
        return any(kw in text for kw in self.NODULE_KEYWORDS)
    
    def _extract_calcification(self, text: str) -> str:
        """Extract calcification type from text."""
        if "popcorn" in text:
            return "popcorn"
        elif "central" in text and "calcif" in text:
            return "central"
        elif "laminated" in text or "concentric" in text:
            return "laminated"
        elif "calcif" in text:
            return "present"
        else:
            return "absent"
    
    def _load_fallback_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Load case from fallback/synthetic data."""
        # Try loading from JSON file
        json_path = self.data_dir / f"{case_id}.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Convert to expected format
            features = data.get("features", data)
            return {
                "case_id": case_id,
                "image": self._generate_synthetic_image(features),
                "report": data.get("report", self._generate_synthetic_report(features)),
                "findings": data.get("findings", ""),
                "impression": data.get("impression", ""),
                "features": features,
                "malignancy": features.get("malignancy", 3)
            }
        
        # Generate fully synthetic case
        return self._generate_synthetic_case(case_id)
    
    def _generate_synthetic_case(self, case_id: str) -> Dict[str, Any]:
        """Generate a synthetic case for demo purposes."""
        # Parse case number for reproducible randomness
        try:
            case_num = int(re.search(r'\d+', case_id).group())
        except:
            case_num = hash(case_id) % 100
        
        np.random.seed(case_num)
        
        # Generate features
        malignancy = (case_num % 5) + 1
        size_mm = 5 + malignancy * 3 + np.random.randint(-2, 3)
        
        textures = ["solid", "ground_glass", "part_solid", "calcified"]
        texture = textures[case_num % len(textures)]
        
        locations = ["right_upper_lobe", "right_lower_lobe", "left_upper_lobe", "left_lower_lobe"]
        location = locations[case_num % len(locations)]
        
        features = {
            "size_mm": float(size_mm),
            "texture": texture,
            "location": location,
            "malignancy": malignancy,
            "has_nodule_mention": True,
            "calcification": "present" if texture == "calcified" else "absent"
        }
        
        return {
            "case_id": case_id,
            "image": self._generate_synthetic_image(features),
            "report": self._generate_synthetic_report(features),
            "findings": "",
            "impression": "",
            "features": features,
            "malignancy": malignancy
        }
    
    def _generate_synthetic_image(self, features: Dict[str, Any]) -> np.ndarray:
        """Generate a synthetic chest X-ray-like image."""
        size = 256
        image = np.zeros((size, size), dtype=np.uint8)
        
        # Background gradient (simulating lung field)
        for i in range(size):
            image[i, :] = 20 + int(20 * (1 - abs(i - size//2) / (size//2)))
        
        # Add a "nodule"
        nodule_size = int(features.get("size_mm", 10) * 2)
        nodule_size = max(5, min(nodule_size, 50))
        
        # Position based on location
        location = features.get("location", "right_upper_lobe")
        if "right" in location:
            cx = size // 4 + np.random.randint(-20, 20)
        else:
            cx = 3 * size // 4 + np.random.randint(-20, 20)
        
        if "upper" in location:
            cy = size // 3 + np.random.randint(-20, 20)
        else:
            cy = 2 * size // 3 + np.random.randint(-20, 20)
        
        # Draw nodule
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        texture = features.get("texture", "solid")
        malignancy = features.get("malignancy", 3)
        
        # Nodule appearance based on texture
        if texture == "ground_glass":
            # Faint, diffuse
            mask = dist < nodule_size
            image[mask] = np.clip(image[mask] + 40, 0, 255).astype(np.uint8)
        elif texture == "calcified":
            # Bright central
            mask = dist < nodule_size
            image[mask] = 200
            inner_mask = dist < nodule_size // 2
            image[inner_mask] = 255
        else:
            # Solid
            mask = dist < nodule_size
            intensity = 120 + malignancy * 20
            image[mask] = min(intensity, 255)
        
        return image
    
    def _generate_synthetic_report(self, features: Dict[str, Any]) -> str:
        """Generate a synthetic radiology report."""
        size_mm = features.get("size_mm", 10)
        texture = features.get("texture", "solid")
        location = features.get("location", "right upper lobe")
        malignancy = features.get("malignancy", 3)
        calcification = features.get("calcification", "absent")
        
        # Convert location format
        location_text = location.replace("_", " ")
        
        # Texture description
        texture_map = {
            "solid": "solid",
            "ground_glass": "ground-glass opacity",
            "part_solid": "part-solid",
            "calcified": "calcified"
        }
        texture_text = texture_map.get(texture, texture)
        
        # Generate findings
        findings = f"A {size_mm:.0f}mm {texture_text} nodule is identified in the {location_text}."
        
        if calcification == "present":
            findings += " Central calcification is noted."
        
        # Generate impression based on malignancy
        if malignancy >= 4:
            impression = (
                f"Suspicious {texture_text} nodule in {location_text}. "
                "Recommend PET-CT or tissue sampling for further evaluation."
            )
        elif malignancy == 3:
            impression = (
                f"Indeterminate pulmonary nodule in {location_text}. "
                "Recommend follow-up CT in 3-6 months."
            )
        else:
            impression = (
                f"Likely benign nodule in {location_text}. "
                "Routine follow-up recommended."
            )
        
        report = f"FINDINGS: {findings}\n\nIMPRESSION: {impression}"
        
        return report
    
    def load_all(self) -> Iterator[Tuple[str, np.ndarray, Dict[str, Any]]]:
        """
        Iterate over all cases.
        
        Yields:
            Tuple of (case_id, image, features)
        """
        for case_id in self.list_cases():
            data = self.load_case(case_id)
            if data:
                yield (
                    case_id,
                    data.get("image"),
                    data.get("features", {})
                )
    
    def filter_nodule_cases(self) -> List[str]:
        """
        Get only cases that mention nodules.
        
        Returns:
            List of case IDs with nodule mentions
        """
        nodule_cases = []
        for case_id in self.list_cases():
            data = self.load_case(case_id)
            if data and data.get("features", {}).get("has_nodule_mention", False):
                nodule_cases.append(case_id)
        return nodule_cases


# Alias for backward compatibility
LIDCLoader = OpenILoader


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)
    
    print("=== Open-I Data Loader Test ===\n")
    
    loader = OpenILoader()
    cases = loader.list_cases()
    
    print(f"Found {len(cases)} cases\n")
    
    # Load and display first 3 cases
    for case_id in cases[:3]:
        print(f"--- {case_id} ---")
        data = loader.load_case(case_id)
        
        if data:
            print(f"  Image shape: {data['image'].shape if data.get('image') is not None else 'None'}")
            print(f"  Malignancy: {data.get('malignancy', 'N/A')}")
            print(f"  Features: {data.get('features', {})}")
            
            report = data.get("report", "")
            if report:
                print(f"  Report preview: {report[:100]}...")
        print()
