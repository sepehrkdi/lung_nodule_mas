"""
NLMCXR Data Loader

Multi-image data loader for the NLMCXR chest X-ray dataset.
Supports loading cases with multiple views (PA, Lateral, etc.).
Extracts ground truth labels from radiology reports using NLP.
Includes NLP-richness scoring to select cases with meaningful extractable content.
"""

import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from PIL import Image

from data.base_loader import BaseNoduleLoader, LoaderFactory
from data.nlmcxr_parser import parse_all_nlmcxr_cases, NLMCXRCase

# Import NLP extractor for ground truth derivation
try:
    from nlp.extractor import MedicalNLPExtractor
    HAS_NLP = True
except ImportError:
    HAS_NLP = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLMCXRLoader(BaseNoduleLoader):
    """
    Data loader for NLMCXR chest X-ray dataset.

    Supports multi-view analysis with variable numbers of images per case (1-5).
    Each case includes a radiology report with FINDINGS and IMPRESSION sections.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize NLMCXR loader.

        Args:
            data_dir: Base directory for NLMCXR data.
                     Defaults to PROJECT_ROOT/data/NLMCXR
        """
        self.project_root = self._find_project_root()
        self.data_dir = data_dir or (self.project_root / "data" / "NLMCXR")
        self.xml_dir = self.data_dir / "ecgen-radiology"
        self.image_dir = self.data_dir / "images"

        # Verify directories exist
        if not self.xml_dir.exists():
            raise FileNotFoundError(
                f"XML directory not found: {self.xml_dir}. "
                f"Run data/extract_nlmcxr.py first to extract the archives."
            )

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}. "
                f"Run data/extract_nlmcxr.py first to extract the archives."
            )

        # Cache parsed cases
        self._case_cache: Dict[str, NLMCXRCase] = {}
        self._index_cases()

        # Initialize NLP extractor for ground truth derivation
        self._nlp_extractor = None
        if HAS_NLP:
            try:
                self._nlp_extractor = MedicalNLPExtractor()
                logger.info("NLP extractor initialized for ground truth derivation")
            except Exception as e:
                logger.warning(f"Could not initialize NLP extractor: {e}")

        logger.info(
            f"NLMCXRLoader initialized with {len(self._case_cache)} cases"
        )

    @staticmethod
    def _find_project_root() -> Path:
        """Find the project root directory."""
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / "requirements.txt").exists():
                return parent
        return current.parent.parent

    def _index_cases(self):
        """Build case index from XML files."""
        logger.info(f"Indexing cases from {self.xml_dir}")
        self._case_cache = parse_all_nlmcxr_cases(self.xml_dir)
        logger.info(f"Indexed {len(self._case_cache)} cases")

    @property
    def supports_multi_image(self) -> bool:
        """Returns True - NLMCXR supports multiple images per case."""
        return True

    def _extract_ground_truth(
        self,
        findings: str,
        impression: str
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Extract ground truth label from radiology report text using NLP.
        
        Ground truth mapping (binary classification):
        - 1: Malignant (nodules, masses, suspicious findings)
        - 0: Benign (no significant findings, stable, benign)
        - -1: Indeterminate (cannot determine from text)
        
        Args:
            findings: FINDINGS section text
            impression: IMPRESSION section text
            
        Returns:
            Tuple of (ground_truth_label, nlp_features_dict)
        """
        # Prioritize IMPRESSION for Ground Truth
        # Only fall back to findings if impression is completely empty
        text_for_gt = impression.strip()
        if not text_for_gt:
            text_for_gt = findings.strip()
            
        if not text_for_gt or not self._nlp_extractor:
            return -1, {}
        
        try:
            result = self._nlp_extractor.extract(text_for_gt)
            features = result.to_dict() if hasattr(result, 'to_dict') else {}
            
            # Determine ground truth from malignancy assessment
            assessment = result.malignancy_assessment
            
            # Map assessment to binary label
            if assessment in ['highly_suspicious', 'moderately_suspicious']:
                ground_truth = 1  # Malignant
            elif assessment in ['benign', 'probably_benign']:
                ground_truth = 0  # Benign
            elif assessment == 'indeterminate':
                ground_truth = -1  # Indeterminate
            else:
                # Fallback: check for suspicious/abnormal keywords in text
                ground_truth = self._keyword_based_ground_truth(text_for_gt)
            
            return ground_truth, features
            
        except Exception as e:
            logger.warning(f"NLP extraction failed: {e}")
            return -1, {}
    
    def _keyword_based_ground_truth(self, text: str) -> int:
        """
        Fallback ground truth extraction using simple keyword matching.
        
        Args:
            text: Combined report text
            
        Returns:
            Ground truth label (1=malignant, 0=benign, -1=indeterminate)
        """
        text_lower = text.lower()
        
        # Abnormal/suspicious indicators
        abnormal_keywords = [
            'nodule', 'mass', 'tumor', 'carcinoma', 'malignant',
            'suspicious', 'concerning', 'opacity', 'consolidation',
            'infiltrate', 'effusion', 'pneumonia', 'lesion'
        ]
        
        # Normal/benign indicators
        normal_keywords = [
            'normal', 'unremarkable', 'no acute', 'no significant',
            'clear', 'negative', 'no evidence', 'within normal',
            'stable', 'unchanged', 'benign'
        ]
        
        # Negation patterns that negate abnormal findings
        negation_patterns = [
            'no nodule', 'no mass', 'no tumor', 'no evidence of',
            'without', 'negative for', 'ruled out'
        ]
        
        # Check for negated abnormal findings first
        has_negated_abnormal = any(neg in text_lower for neg in negation_patterns)
        
        # Count keyword matches
        abnormal_count = sum(1 for kw in abnormal_keywords if kw in text_lower)
        normal_count = sum(1 for kw in normal_keywords if kw in text_lower)
        
        if has_negated_abnormal and normal_count > 0:
            return 0  # Likely normal
        elif abnormal_count > normal_count:
            return 1  # Likely abnormal
        elif normal_count > abnormal_count:
            return 0  # Likely normal
        else:
            return -1  # Indeterminate
    
    @staticmethod
    def _ground_truth_label(ground_truth: int) -> str:
        """Convert numeric ground truth to descriptive label."""
        return {
            1: "malignant",
            0: "benign",
            -1: "indeterminate"
        }.get(ground_truth, "unknown")

    def get_case_ids(self) -> List[str]:
        """
        Get list of all available case IDs that have at least one valid image.

        Returns:
            List of case IDs sorted alphabetically
        """
        valid_cases = []
        for case_id, case in self._case_cache.items():
            # Check if at least one image exists for this case
            has_valid_image = False
            for img_info in case.images:
                file_path = self._find_image_file(img_info.image_id)
                if file_path and file_path.exists():
                    has_valid_image = True
                    break
            if has_valid_image:
                valid_cases.append(case_id)
        return sorted(valid_cases)

    def get_nodule_case_ids(self, limit: int = 50) -> List[str]:
        """
        Get list of case IDs that have a definite ground truth label (0 or 1).
        
        Selects cases where NLP can determine a clear normal/abnormal label,
        excluding indeterminate cases (ground_truth == -1).
        
        Args:
            limit: Maximum number of cases to return (default: 50)
            
        Returns:
            List of case IDs sorted alphabetically
        """
        labeled_cases = []
        
        # Iterate through all cases
        sorted_ids = sorted(self._case_cache.keys())
        
        for case_id in sorted_ids:
            if len(labeled_cases) >= limit:
                break
                
            case = self._case_cache[case_id]
            
            # Check for valid images first
            has_valid_image = False
            for img_info in case.images:
                file_path = self._find_image_file(img_info.image_id)
                if file_path and file_path.exists():
                    has_valid_image = True
                    break
            
            if not has_valid_image:
                continue
                
            # Extract ground truth from report
            ground_truth, _ = self._extract_ground_truth(
                case.findings, case.impression
            )
            
            # Include only cases with definite labels (0=normal or 1=abnormal)
            if ground_truth in (0, 1):
                labeled_cases.append(case_id)
                
        return labeled_cases

    # =========================================================================
    # NLP RICHNESS SCORING
    # =========================================================================

    # Target medical entity patterns for richness scoring
    _TARGET_ENTITY_PATTERN = re.compile(
        r'\b(nodule|nodules|mass|masses|opacity|opacities|lesion|lesions|'
        r'consolidation|granuloma|granulomas|atelectasis|tumor|neoplasm|carcinoma)\b',
        re.IGNORECASE
    )

    # Simple negation scope: entity preceded within 4 words by negation trigger
    _NEGATION_PREFIX_PATTERN = re.compile(
        r'\b(no|without|negative\s+for|absence\s+of|ruled\s+out|'
        r'no\s+evidence\s+of|deny|denies)\b',
        re.IGNORECASE
    )

    # Anatomical location patterns
    _LOCATION_PATTERN = re.compile(
        r'\b(upper|middle|lower|right|left)\s+(lobe|lung|hilum|hilus|base|apex)\b',
        re.IGNORECASE
    )

    @staticmethod
    def compute_nlp_richness(case: NLMCXRCase) -> Tuple[float, Dict[str, Any]]:
        """
        Compute a multi-factor NLP richness score for a case.

        Scores each case on 6 binary criteria (each worth 1 point):
          1. Text length: len(findings + impression) >= 80
          2. Non-normal MeSH: any mesh_major tag that is NOT just 'normal'
          3. Target entity present: regex match for clinical entities
          4. Entity NOT fully negated: at least one entity outside negation scope
          5. Both sections non-empty: findings and impression both have content
          6. Anatomical location specified: anatomy + laterality pattern

        Args:
            case: Parsed NLMCXRCase object

        Returns:
            Tuple of (score 0-6, breakdown dict)
        """
        breakdown = {}
        # METHODOLOGY UPDATE:
        # Agents only see FINDINGS, so richness must be calculated on FINDINGS.
        # We check IMPRESSION only for the 'both_sections' completeness check.
        combined_text = case.findings.strip()

        # 1. Text length (Findings should be substantial)
        breakdown['text_length'] = 1.0 if len(combined_text) >= 80 else 0.0

        # 2. Non-normal MeSH
        has_pathology_mesh = any(
            tag.lower() != 'normal' for tag in case.mesh_major
        ) if case.mesh_major else False
        breakdown['non_normal_mesh'] = 1.0 if has_pathology_mesh else 0.0

        # 3. Target entity present (in Findings)
        entity_matches = list(NLMCXRLoader._TARGET_ENTITY_PATTERN.finditer(combined_text))
        breakdown['has_target_entity'] = 1.0 if entity_matches else 0.0

        # 4. Entity NOT fully negated (at least one affirmed entity in Findings)
        has_affirmed_entity = False
        if entity_matches:
            for match in entity_matches:
                # Check ~40 chars before the entity for negation triggers
                start = max(0, match.start() - 40)
                preceding = combined_text[start:match.start()]
                if not NLMCXRLoader._NEGATION_PREFIX_PATTERN.search(preceding):
                    has_affirmed_entity = True
                    break
        breakdown['has_affirmed_entity'] = 1.0 if has_affirmed_entity else 0.0

        # 5. Both sections non-empty (still valuable to know if impression exists)
        both_filled = bool(case.findings.strip()) and bool(case.impression.strip())
        breakdown['both_sections'] = 1.0 if both_filled else 0.0

        # 6. Anatomical location specified (in Findings)
        has_location = bool(NLMCXRLoader._LOCATION_PATTERN.search(combined_text))
        breakdown['has_location'] = 1.0 if has_location else 0.0

        score = sum(breakdown.values())
        return score, breakdown

    def get_nlp_rich_case_ids(
        self,
        min_score: float = 3.0,
        limit: int = 50,
        offset: int = 0,
        require_valid_image: bool = True
    ) -> List[str]:
        """
        Get case IDs filtered by NLP richness score using pagination.

        Selects cases where the radiology report contains enough extractable
        NLP content for meaningful agent analysis. Cases are sorted by
        richness score descending (richest first).

        Args:
            min_score: Minimum richness score (0-6) to include. Default 3.0.
            limit: Maximum number of cases to return.
            offset: Number of cases to skip (for batch processing).
            require_valid_image: If True, only include cases with at least one image file.

        Returns:
            List of (case_id) sorted by richness score descending.
        """
        scored_cases = []

        for case_id, case in self._case_cache.items():
            # Check image availability
            if require_valid_image:
                has_valid_image = False
                for img_info in case.images:
                    file_path = self._find_image_file(img_info.image_id)
                    if file_path and file_path.exists():
                        has_valid_image = True
                        break
                if not has_valid_image:
                    continue

            score, breakdown = self.compute_nlp_richness(case)
            if score >= min_score:
                scored_cases.append((case_id, score, breakdown))

        # Sort by score descending, then by case_id for deterministic order
        scored_cases.sort(key=lambda x: (-x[1], x[0]))

        # Apply pagination (offset + limit)
        selected = [case_id for case_id, _, _ in scored_cases[offset : offset + limit]]

        logger.info(
            f"NLP richness filter: {len(scored_cases)} cases scored >= {min_score} "
            f"(out of {len(self._case_cache)} total), returning {len(selected)} cases "
            f"(offset {offset}, limit {limit})"
        )

        return selected

    def load_case(
        self,
        case_id: str
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Load case with all associated images.

        Args:
            case_id: Case identifier (e.g., "CXR1")

        Returns:
            Tuple of (images, metadata) where:
            - images: List[np.ndarray] - one array per view
            - metadata: Dict with case information and per-image metadata

        Raises:
            FileNotFoundError: If case_id is not found or images are missing
        """
        case = self._case_cache.get(case_id)
        if not case:
            raise FileNotFoundError(
                f"Case {case_id} not found. "
                f"Available cases: {len(self._case_cache)}"
            )

        # Load all images
        images = []
        image_metadata = []
        missing_images = []

        for img_info in case.images:
            # Construct PNG file path
            # Image ID format: CXR1_1_IM-0001-3001 -> CXR1_1_IM-0001-3001.png
            file_path = self._find_image_file(img_info.image_id)

            if not file_path or not file_path.exists():
                logger.warning(
                    f"Image not found for {case_id}: {img_info.image_id}"
                )
                missing_images.append(img_info.image_id)
                continue

            try:
                # Load PNG image
                img = Image.open(file_path).convert('L')  # Grayscale
                img_array = np.array(img).astype(np.float32) / 255.0
                images.append(img_array)

                # Store per-image metadata
                image_metadata.append({
                    "image_id": img_info.image_id,
                    "view_type": img_info.view_type,
                    "caption": img_info.caption,
                    "figure_id": img_info.figureId,
                    "shape": img_array.shape,
                    "file_path": str(file_path)
                })

            except Exception as e:
                logger.error(f"Error loading image {file_path}: {e}")
                missing_images.append(img_info.image_id)

        if not images:
            raise FileNotFoundError(
                f"No images found for case {case_id}. "
                f"Missing: {missing_images}"
            )

        # Assemble case metadata
        # Extract ground truth and NLP features from report text
        ground_truth, nlp_features = self._extract_ground_truth(
            case.findings, case.impression
        )
        
        metadata = {
            "case_id": case_id,
            "num_images": len(images),
            "images_metadata": image_metadata,
            "findings": case.findings,
            "impression": case.impression,
            "indication": case.indication,
            "comparison": case.comparison,
            "mesh_major": case.mesh_major,
            "mesh_automatic": case.mesh_automatic,
            "missing_images": missing_images,
            # Ground truth derived from report text via NLP
            "ground_truth": ground_truth,
            "ground_truth_label": self._ground_truth_label(ground_truth),
            "nlp_features": nlp_features
        }

        logger.debug(
            f"Loaded case {case_id}: {len(images)} images, "
            f"{len(missing_images)} missing"
        )

        return images, metadata

    def _find_image_file(self, image_id: str) -> Optional[Path]:
        """
        Find image file for a given image ID.

        The image files may be in subdirectories or directly in images/.
        Try multiple patterns to locate the file.

        Args:
            image_id: Image identifier (e.g., "CXR1_1_IM-0001-3001")

        Returns:
            Path to image file or None if not found
        """
        # Pattern 1: Direct PNG in images/ directory
        direct_path = self.image_dir / f"{image_id}.png"
        if direct_path.exists():
            return direct_path

        # Pattern 2: Search recursively
        matches = list(self.image_dir.rglob(f"{image_id}.png"))
        if matches:
            return matches[0]

        # Pattern 3: Try with underscores replaced
        alt_id = image_id.replace("_", "-")
        alt_path = self.image_dir / f"{alt_id}.png"
        if alt_path.exists():
            return alt_path

        return None

    def get_image_by_view(
        self,
        case_id: str,
        view_type: str
    ) -> Optional[np.ndarray]:
        """
        Get specific view from a case.

        Args:
            case_id: Case identifier
            view_type: Desired view type ("PA", "Lateral", "AP", etc.)

        Returns:
            Image array or None if view not found
        """
        images, metadata = self.load_case(case_id)

        for i, img_meta in enumerate(metadata["images_metadata"]):
            if img_meta["view_type"] == view_type:
                return images[i]

        return None

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the NLMCXR dataset.

        Returns:
            Dict with dataset statistics
        """
        total_images = sum(
            len(case.images) for case in self._case_cache.values()
        )

        from collections import Counter
        view_counts = Counter()
        images_per_case = []

        for case in self._case_cache.values():
            images_per_case.append(len(case.images))
            for img in case.images:
                view_counts[img.view_type] += 1

        return {
            "dataset": "NLMCXR",
            "total_cases": len(self._case_cache),
            "total_images": total_images,
            "avg_images_per_case": total_images / len(self._case_cache) if self._case_cache else 0,
            "min_images_per_case": min(images_per_case) if images_per_case else 0,
            "max_images_per_case": max(images_per_case) if images_per_case else 0,
            "view_distribution": dict(view_counts),
            "supports_multi_image": True
        }


# Register with factory
LoaderFactory.register_loader("NLMCXR", NLMCXRLoader)


def main():
    """Test the NLMCXR loader."""
    loader = NLMCXRLoader()

    # Print dataset info
    info = loader.get_dataset_info()
    print(f"\nDataset Info:")
    for key, value in info.items():
        if key != "view_distribution":
            print(f"  {key}: {value}")

    print(f"\n  View distribution:")
    for view, count in info["view_distribution"].items():
        print(f"    {view}: {count}")

    # Load a sample case
    case_ids = loader.get_case_ids()
    if case_ids:
        sample_id = case_ids[0]
        print(f"\nLoading sample case: {sample_id}")

        images, metadata = loader.load_case(sample_id)

        print(f"  Loaded {len(images)} images:")
        for i, img_meta in enumerate(metadata["images_metadata"]):
            print(
                f"    [{i}] {img_meta['image_id']} ({img_meta['view_type']}): "
                f"{img_meta['shape']}"
            )

        print(f"\n  Findings: {metadata['findings'][:150]}...")
        print(f"  Impression: {metadata['impression']}")


if __name__ == "__main__":
    main()
