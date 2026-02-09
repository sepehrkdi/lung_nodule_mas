"""
NLMCXR XML Parser

Parses NLMCXR XML report files to extract case metadata and image associations.
"""

import xml.etree.ElementTree as ET
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NLMCXRImage:
    """Represents a single image in an NLMCXR case."""
    image_id: str          # e.g., "CXR1_1_IM-0001-3001"
    figureId: str          # e.g., "F1", "F2"
    caption: str           # e.g., "Xray Chest PA and Lateral"
    url: str               # Original URL path
    view_type: str         # Classified view: "PA", "Lateral", "AP", "Unknown"
    img_modality: str = "7"  # Image modality code


@dataclass
class NLMCXRCase:
    """Represents a complete NLMCXR case with multiple images and report."""
    case_id: str           # e.g., "CXR1"
    images: List[NLMCXRImage] = field(default_factory=list)
    findings: str = ""     # Report FINDINGS section
    impression: str = ""   # Report IMPRESSION section
    indication: str = ""   # Report INDICATION section
    comparison: str = ""   # Report COMPARISON section
    metadata: Dict[str, Any] = field(default_factory=dict)


def classify_view(caption: str) -> str:
    """
    Classify view type from image caption.

    Args:
        caption: Image caption string

    Returns:
        View type: "PA", "Lateral", "AP", "Frontal", or "Unknown"
    """
    if not caption:
        return "Unknown"

    caption_lower = caption.lower()

    # Check for specific view types
    if "pa" in caption_lower and "lateral" not in caption_lower:
        return "PA"
    elif "lateral" in caption_lower:
        return "Lateral"
    elif "ap" in caption_lower:
        return "AP"
    elif "frontal" in caption_lower:
        return "Frontal"
    else:
        return "Unknown"


def parse_nlmcxr_xml(xml_path: Path) -> Optional[NLMCXRCase]:
    """
    Parse a single NLMCXR XML file.

    Args:
        xml_path: Path to the XML file

    Returns:
        NLMCXRCase object or None if parsing fails
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract case ID from <uId id="...">
        case_id = None
        uid_elem = root.find('.//uId')
        if uid_elem is not None and 'id' in uid_elem.attrib:
            case_id = uid_elem.attrib['id']

        if not case_id:
            logger.warning(f"No case ID found in {xml_path}")
            return None

        case = NLMCXRCase(case_id=case_id)

        # Extract report sections from <AbstractText Label="...">
        abstract_texts = root.findall('.//AbstractText')
        for abstract in abstract_texts:
            label = abstract.get('Label', '').upper()
            text = abstract.text or ""

            if label == 'FINDINGS':
                case.findings = text.strip()
            elif label == 'IMPRESSION':
                case.impression = text.strip()
            elif label == 'INDICATION':
                case.indication = text.strip()
            elif label == 'COMPARISON':
                case.comparison = text.strip()

        # Extract images from <parentImage id="...">
        parent_images = root.findall('.//parentImage')
        for parent_img in parent_images:
            image_id = parent_img.get('id')
            if not image_id:
                continue

            # Extract figure ID
            figure_id_elem = parent_img.find('.//figureId')
            figure_id = figure_id_elem.text if figure_id_elem is not None else ""

            # Extract caption
            caption_elem = parent_img.find('.//caption')
            caption = caption_elem.text if caption_elem is not None else ""

            # Extract URL
            url_elem = parent_img.find('.//url')
            url = url_elem.text if url_elem is not None else ""

            # Extract modality
            modality_elem = parent_img.find('.//imgModality')
            modality = modality_elem.text if modality_elem is not None else "7"

            # Classify view type from caption
            view_type = classify_view(caption)

            # Create image object
            image = NLMCXRImage(
                image_id=image_id,
                figureId=figure_id,
                caption=caption,
                url=url,
                view_type=view_type,
                img_modality=modality
            )

            case.images.append(image)

        # Store metadata
        case.metadata = {
            'xml_path': str(xml_path),
            'num_images': len(case.images)
        }

        return case

    except ET.ParseError as e:
        logger.error(f"XML parse error in {xml_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error parsing {xml_path}: {e}")
        return None


def parse_all_nlmcxr_cases(xml_dir: Path) -> Dict[str, NLMCXRCase]:
    """
    Parse all NLMCXR XML files in a directory.

    Args:
        xml_dir: Directory containing XML files

    Returns:
        Dictionary mapping case_id to NLMCXRCase objects
    """
    cases = {}
    xml_files = list(xml_dir.glob("*.xml"))

    logger.info(f"Parsing {len(xml_files)} XML files from {xml_dir}")

    for xml_file in xml_files:
        case = parse_nlmcxr_xml(xml_file)
        if case:
            cases[case.case_id] = case

    logger.info(f"Successfully parsed {len(cases)} cases")

    return cases


def main():
    """Test the parser on a sample file."""
    from pathlib import Path

    # Find project root
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "requirements.txt").exists():
            project_root = parent
            break
    else:
        project_root = current.parent.parent

    xml_dir = project_root / "data" / "NLMCXR" / "ecgen-radiology"

    if not xml_dir.exists():
        print(f"XML directory not found: {xml_dir}")
        return

    # Parse first file as test
    xml_files = list(xml_dir.glob("*.xml"))
    if xml_files:
        test_file = xml_files[0]
        print(f"\nTesting parser on {test_file.name}:")
        case = parse_nlmcxr_xml(test_file)

        if case:
            print(f"  Case ID: {case.case_id}")
            print(f"  Number of images: {len(case.images)}")
            for img in case.images:
                print(f"    - {img.image_id} ({img.view_type}): {img.caption}")
            print(f"  Findings: {case.findings[:100]}...")
            print(f"  Impression: {case.impression}")
        else:
            print("  Parsing failed")

    # Parse all and show statistics
    print(f"\nParsing all {len(xml_files)} files...")
    cases = parse_all_nlmcxr_cases(xml_dir)

    # Compute statistics
    from collections import Counter
    view_dist = Counter()
    images_per_case = []

    for case in cases.values():
        images_per_case.append(len(case.images))
        for img in case.images:
            view_dist[img.view_type] += 1

    print(f"\nStatistics:")
    print(f"  Total cases: {len(cases)}")
    print(f"  Total images: {sum(images_per_case)}")
    print(f"  Avg images/case: {sum(images_per_case) / len(cases):.2f}")
    print(f"  View distribution:")
    for view, count in view_dist.most_common():
        print(f"    {view}: {count}")


if __name__ == "__main__":
    main()
