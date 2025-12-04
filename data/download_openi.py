#!/usr/bin/env python3
"""
Open-I Dataset Download Helper
==============================

Helper script to download chest X-ray images and reports from the
Open-I Indiana University collection.

USAGE:
    python -m data.download_openi --n_cases 50
    python -m data.download_openi --search "pulmonary nodule"

NOTE:
    Open-I doesn't provide a direct bulk download API, so this script
    provides guidance and tools for manual/semi-automated download.

MANUAL DOWNLOAD STEPS:
    1. Go to https://openi.nlm.nih.gov/
    2. Search for relevant terms (e.g., "pulmonary nodule", "lung mass")
    3. Click on individual results to download images
    4. Save XML reports from the detailed view

For educational purposes, we provide a fallback dataset that
mimics the Open-I format.
"""

import argparse
import json
import os
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Open-I base URL
OPENI_BASE_URL = "https://openi.nlm.nih.gov"
OPENI_API_URL = "https://openi.nlm.nih.gov/api/search"


def get_project_root() -> Path:
    """Find the project root directory."""
    current = Path(__file__).parent.parent
    while current != current.parent:
        if (current / "requirements.txt").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent


def search_openi(query: str, n_results: int = 50) -> List[Dict[str, Any]]:
    """
    Search Open-I for chest X-rays matching query.
    
    Args:
        query: Search query (e.g., "pulmonary nodule")
        n_results: Maximum number of results
        
    Returns:
        List of search result dictionaries
    """
    logger.info(f"Searching Open-I for: {query}")
    
    params = {
        "q": query,
        "m": "1",  # Start from result 1
        "n": str(n_results),
        "coll": "iu",  # Indiana University collection
    }
    
    try:
        response = requests.get(OPENI_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("list", [])
        logger.info(f"Found {len(results)} results")
        return results
        
    except requests.RequestException as e:
        logger.error(f"Failed to search Open-I: {e}")
        return []


def download_image(image_id: str, output_dir: Path) -> Optional[Path]:
    """
    Download an image from Open-I.
    
    Args:
        image_id: Image identifier (e.g., "CXR111_IM-0076-1001")
        output_dir: Directory to save image
        
    Returns:
        Path to downloaded image or None
    """
    # Open-I image URL pattern
    image_url = f"{OPENI_BASE_URL}/imgs/{image_id}.png"
    
    output_path = output_dir / f"{image_id}.png"
    
    if output_path.exists():
        logger.debug(f"Image already exists: {output_path}")
        return output_path
    
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded: {image_id}")
        return output_path
        
    except requests.RequestException as e:
        logger.warning(f"Failed to download {image_id}: {e}")
        return None


def create_synthetic_dataset(output_dir: Path, n_cases: int = 20) -> None:
    """
    Create a synthetic dataset mimicking Open-I format.
    
    This is useful for development and testing when the actual
    Open-I data is not available.
    
    Args:
        output_dir: Directory to create dataset
        n_cases: Number of synthetic cases to create
    """
    logger.info(f"Creating {n_cases} synthetic cases in {output_dir}")
    
    images_dir = output_dir / "images"
    reports_dir = output_dir / "reports"
    images_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    import numpy as np
    from PIL import Image
    
    cases = []
    
    for i in range(1, n_cases + 1):
        case_id = f"case_{i:03d}"
        
        # Vary malignancy across cases
        malignancy = ((i - 1) % 5) + 1
        
        # Generate features
        np.random.seed(i)
        size_mm = 5 + malignancy * 4 + np.random.randint(-3, 4)
        
        textures = ["solid", "ground_glass", "part_solid"]
        texture = textures[i % 3]
        
        locations = [
            "right upper lobe", "right middle lobe", "right lower lobe",
            "left upper lobe", "left lower lobe"
        ]
        location = locations[i % 5]
        
        # Generate report text
        report = generate_synthetic_report(size_mm, texture, location, malignancy)
        
        # Generate synthetic image
        image = generate_synthetic_xray(size_mm, texture, location, malignancy)
        
        # Save image
        image_path = images_dir / f"{case_id}.png"
        Image.fromarray(image).save(image_path)
        
        # Save report as XML
        xml_content = generate_xml_report(case_id, report, location)
        report_path = reports_dir / f"{case_id}.xml"
        with open(report_path, 'w') as f:
            f.write(xml_content)
        
        cases.append({
            "id": case_id,
            "malignancy": malignancy,
            "size_mm": float(size_mm),
            "texture": texture,
            "location": location,
            "report": report["full_text"]
        })
    
    # Save manifest
    manifest = {
        "dataset": "synthetic_openi",
        "description": "Synthetic dataset mimicking Open-I format for educational use",
        "n_cases": n_cases,
        "cases": cases
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Created {n_cases} synthetic cases")
    logger.info(f"Manifest saved to {manifest_path}")


def generate_synthetic_report(
    size_mm: float,
    texture: str,
    location: str,
    malignancy: int
) -> Dict[str, str]:
    """Generate synthetic radiology report text."""
    
    texture_desc = {
        "solid": "solid",
        "ground_glass": "ground-glass",
        "part_solid": "part-solid"
    }.get(texture, texture)
    
    # Findings
    findings = f"A {size_mm:.0f}mm {texture_desc} nodule is identified in the {location}."
    
    if malignancy >= 4:
        findings += " The lesion demonstrates irregular margins with mild spiculation."
    elif malignancy == 3:
        findings += " The margins are somewhat indistinct."
    else:
        findings += " The nodule has smooth, well-defined margins."
    
    # Impression
    if malignancy >= 4:
        impression = (
            f"Suspicious pulmonary nodule in the {location}. "
            "Findings are concerning for malignancy. "
            "Recommend PET-CT or tissue sampling."
        )
    elif malignancy == 3:
        impression = (
            f"Indeterminate pulmonary nodule in the {location}. "
            "Recommend follow-up CT in 3-6 months."
        )
    else:
        impression = (
            f"Likely benign nodule in the {location}. "
            "Routine follow-up recommended."
        )
    
    full_text = f"FINDINGS: {findings}\n\nIMPRESSION: {impression}"
    
    return {
        "findings": findings,
        "impression": impression,
        "full_text": full_text
    }


def generate_synthetic_xray(
    size_mm: float,
    texture: str,
    location: str,
    malignancy: int
) -> 'np.ndarray':
    """Generate a synthetic chest X-ray image."""
    import numpy as np
    
    size = 512
    image = np.zeros((size, size), dtype=np.uint8)
    
    # Background (lung fields)
    for i in range(size):
        for j in range(size):
            # Simulate lung field darkness
            dist_center = abs(j - size//2) / (size//2)
            image[i, j] = int(30 + 20 * dist_center)
    
    # Add ribs (simplified)
    for rib_y in range(60, size-60, 50):
        for x in range(size):
            y_offset = int(15 * np.sin(x * np.pi / size))
            y = rib_y + y_offset
            if 0 <= y < size:
                image[max(0, y-2):min(size, y+3), x] = 80
    
    # Nodule position based on location
    if "right" in location:
        cx = size // 4
    else:
        cx = 3 * size // 4
    
    if "upper" in location:
        cy = size // 3
    elif "lower" in location:
        cy = 2 * size // 3
    else:
        cy = size // 2
    
    # Add some randomness
    np.random.seed(hash(location) % 1000)
    cx += np.random.randint(-30, 30)
    cy += np.random.randint(-30, 30)
    
    # Draw nodule
    nodule_radius = int(size_mm * 1.5)
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    if texture == "ground_glass":
        mask = dist < nodule_radius
        image[mask] = np.clip(image[mask] + 40, 0, 255).astype(np.uint8)
    elif texture == "part_solid":
        mask = dist < nodule_radius
        image[mask] = np.clip(image[mask] + 60, 0, 255).astype(np.uint8)
        inner = dist < nodule_radius * 0.6
        image[inner] = 150
    else:  # solid
        mask = dist < nodule_radius
        intensity = 100 + malignancy * 20
        image[mask] = min(intensity, 200)
    
    return image


def generate_xml_report(case_id: str, report: Dict[str, str], location: str) -> str:
    """Generate XML in Open-I format."""
    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<eCitation>
    <meta type="rr"/>
    <uId id="{case_id}"/>
    <pmcId id="{case_id}"/>
    <MedlineCitation>
        <Article>
            <Abstract>
                <AbstractText Label="FINDINGS">{report["findings"]}</AbstractText>
                <AbstractText Label="IMPRESSION">{report["impression"]}</AbstractText>
            </Abstract>
        </Article>
        <MeshHeadingList>
            <MeshHeading>
                <DescriptorName>Pulmonary Nodule</DescriptorName>
            </MeshHeading>
        </MeshHeadingList>
    </MedlineCitation>
</eCitation>'''
    return xml


def main():
    parser = argparse.ArgumentParser(
        description="Download or create Open-I chest X-ray dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m data.download_openi --synthetic --n_cases 20
    python -m data.download_openi --search "pulmonary nodule"
    python -m data.download_openi --help-download
        """
    )
    
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Create synthetic dataset (recommended for quick start)"
    )
    parser.add_argument(
        "--n_cases",
        type=int,
        default=20,
        help="Number of cases to create/download (default: 20)"
    )
    parser.add_argument(
        "--search",
        type=str,
        default="pulmonary nodule",
        help="Search query for Open-I"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/openi)"
    )
    parser.add_argument(
        "--help-download",
        action="store_true",
        help="Show manual download instructions"
    )
    
    args = parser.parse_args()
    
    if args.help_download:
        print("""
MANUAL DOWNLOAD INSTRUCTIONS:
=============================

The Open-I database doesn't provide a bulk download API, so you'll need
to download cases manually or semi-automatically.

1. Go to: https://openi.nlm.nih.gov/

2. Search for terms like:
   - "pulmonary nodule"
   - "lung nodule"
   - "lung mass"
   - "solitary pulmonary nodule"

3. For each result:
   a. Click on the image to view details
   b. Right-click and save the PNG image
   c. Click "MeSH/XML" to download the XML report
   
4. Save files to:
   - Images: data/openi/images/
   - Reports: data/openi/reports/

5. Use consistent naming: CXR###_IM-####.png and .xml

ALTERNATIVE - SYNTHETIC DATA:
=============================
For quick development, use synthetic data:

    python -m data.download_openi --synthetic --n_cases 50

This creates realistic test data without downloading.
        """)
        return
    
    # Set output directory
    project_root = get_project_root()
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / "data" / "openi"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.synthetic:
        # Create synthetic dataset
        create_synthetic_dataset(output_dir, args.n_cases)
        
        print(f"\n✓ Created {args.n_cases} synthetic cases in {output_dir}")
        print("\nTo use this data:")
        print("    python spade_main.py --demo")
        
    else:
        # Search and provide download guidance
        results = search_openi(args.search, args.n_cases)
        
        if results:
            print(f"\nFound {len(results)} results for '{args.search}'")
            print("\nTo download manually:")
            for i, result in enumerate(results[:10], 1):
                img_id = result.get("imgid", "unknown")
                print(f"  {i}. {OPENI_BASE_URL}/detailedresult?img={img_id}")
            
            if len(results) > 10:
                print(f"  ... and {len(results) - 10} more")
            
            print("\nOr create synthetic data instead:")
            print("    python -m data.download_openi --synthetic")
        else:
            print("No results found. Creating synthetic dataset instead...")
            create_synthetic_dataset(output_dir, args.n_cases)


if __name__ == "__main__":
    main()
