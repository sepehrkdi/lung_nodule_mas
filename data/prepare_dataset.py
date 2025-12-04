"""
Dataset Preparation Script
==========================

Prepares the LIDC-IDRI dataset for the lung nodule MAS project.

USAGE:
    # Use fallback dataset (10 pre-included nodules, no download required)
    python -m data.prepare_dataset --fallback
    
    # Prepare full dataset from LIDC-IDRI (requires pylidc setup)
    python -m data.prepare_dataset --full --n_nodules 100

LIDC-IDRI SETUP (for --full option):
1. Install pylidc: pip install pylidc
2. Download DICOM files from TCIA:
   https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
3. Configure pylidc to point to the DICOM directory:
   Create ~/.pylidcrc with:
   [dicom]
   path = /path/to/LIDC-IDRI
4. Run: python -m data.prepare_dataset --full

EDUCATIONAL PURPOSE:
This script demonstrates:
- Medical imaging data preprocessing
- Consensus annotation computation (multiple radiologists)
- Dataset curation for ML training
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from PIL import Image


def find_project_root() -> Path:
    """Find the project root directory."""
    current = Path(__file__).parent.parent
    while current != current.parent:
        if (current / "requirements.txt").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent


def prepare_fallback_dataset(project_root: Path) -> None:
    """
    Set up the fallback dataset for immediate use.
    
    The fallback data is already included in data/fallback/.
    This function just creates the subset directory and copies files.
    """
    fallback_dir = project_root / "data" / "fallback"
    subset_dir = project_root / "data" / "subset"
    
    # Create subset directory
    subset_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy fallback files to subset
    if fallback_dir.exists():
        for json_file in fallback_dir.glob("nodule_*.json"):
            shutil.copy(json_file, subset_dir / json_file.name)
            
        # Also copy any PNG files if they exist
        for png_file in fallback_dir.glob("nodule_*.png"):
            shutil.copy(png_file, subset_dir / png_file.name)
            
        print(f"[prepare_dataset] Copied fallback data to {subset_dir}")
        print(f"[prepare_dataset] {len(list(subset_dir.glob('*.json')))} nodules ready")
    else:
        print(f"[prepare_dataset] ERROR: Fallback directory not found at {fallback_dir}")
        sys.exit(1)


def compute_consensus(annotations: List[Any]) -> Dict[str, Any]:
    """
    Compute consensus features from multiple radiologist annotations.
    
    EDUCATIONAL NOTE:
    LIDC-IDRI has up to 4 radiologists annotating each nodule.
    We use the median of their ratings as the consensus value.
    This is a common approach in medical imaging research.
    
    Args:
        annotations: List of pylidc Annotation objects
        
    Returns:
        Dictionary with consensus feature values
    """
    if not annotations:
        return {}
    
    features = {
        "malignancy": [],
        "spiculation": [],
        "margin": [],
        "texture": [],
        "lobulation": [],
        "calcification": [],
        "sphericity": [],
        "subtlety": [],
        "internal_structure": [],
        "diameter_mm": []
    }
    
    for ann in annotations:
        features["malignancy"].append(ann.malignancy)
        features["spiculation"].append(ann.spiculation)
        features["margin"].append(ann.margin)
        features["texture"].append(ann.texture)
        features["lobulation"].append(ann.lobulation)
        features["calcification"].append(ann.calcification)
        features["sphericity"].append(ann.sphericity)
        features["subtlety"].append(ann.subtlety)
        features["internal_structure"].append(ann.internalStructure)
        features["diameter_mm"].append(ann.diameter)
    
    # Compute median for each feature
    consensus = {}
    for key, values in features.items():
        if values:
            consensus[key] = float(np.median(values))
            if key != "diameter_mm":
                consensus[key] = int(round(consensus[key]))
    
    consensus["consensus_annotations"] = len(annotations)
    
    return consensus


def get_semantic_labels(features: Dict[str, Any]) -> Dict[str, str]:
    """Add human-readable labels for numeric features."""
    labels = {}
    
    malignancy_map = {
        1: "Highly Unlikely",
        2: "Moderately Unlikely",
        3: "Indeterminate",
        4: "Moderately Suspicious",
        5: "Highly Suspicious"
    }
    
    spiculation_map = {
        1: "No Spiculation",
        2: "Nearly No Spiculation",
        3: "Medium Spiculation",
        4: "Near Marked Spiculation",
        5: "Marked Spiculation"
    }
    
    margin_map = {
        1: "Poorly Defined",
        2: "Near Poorly Defined",
        3: "Medium Margin",
        4: "Near Sharp",
        5: "Sharp"
    }
    
    texture_map = {
        1: "Non-Solid/GGO",
        2: "Non-Solid/Mixed",
        3: "Part Solid/Mixed",
        4: "Solid/Mixed",
        5: "Solid"
    }
    
    lobulation_map = {
        1: "No Lobulation",
        2: "Nearly No Lobulation",
        3: "Medium Lobulation",
        4: "Near Marked Lobulation",
        5: "Marked Lobulation"
    }
    
    calcification_map = {
        1: "Popcorn",
        2: "Laminated",
        3: "Solid",
        4: "Non-central",
        5: "Central",
        6: "Absent"
    }
    
    sphericity_map = {
        1: "Linear",
        2: "Ovoid/Linear",
        3: "Ovoid",
        4: "Ovoid/Round",
        5: "Round"
    }
    
    subtlety_map = {
        1: "Extremely Subtle",
        2: "Moderately Subtle",
        3: "Fairly Subtle",
        4: "Moderately Obvious",
        5: "Obvious"
    }
    
    internal_structure_map = {
        1: "Soft Tissue",
        2: "Fluid",
        3: "Fat",
        4: "Air"
    }
    
    labels["malignancy_label"] = malignancy_map.get(features.get("malignancy", 3), "Unknown")
    labels["spiculation_label"] = spiculation_map.get(features.get("spiculation", 1), "Unknown")
    labels["margin_label"] = margin_map.get(features.get("margin", 5), "Unknown")
    labels["texture_label"] = texture_map.get(features.get("texture", 5), "Unknown")
    labels["lobulation_label"] = lobulation_map.get(features.get("lobulation", 1), "Unknown")
    labels["calcification_label"] = calcification_map.get(features.get("calcification", 6), "Unknown")
    labels["sphericity_label"] = sphericity_map.get(features.get("sphericity", 5), "Unknown")
    labels["subtlety_label"] = subtlety_map.get(features.get("subtlety", 5), "Unknown")
    labels["internal_structure_label"] = internal_structure_map.get(features.get("internal_structure", 1), "Unknown")
    
    return labels


def prepare_full_dataset(project_root: Path, n_nodules: int = 100) -> None:
    """
    Prepare dataset from full LIDC-IDRI using pylidc.
    
    REQUIREMENTS:
    - pylidc installed
    - LIDC-IDRI DICOM files downloaded
    - ~/.pylidcrc configured with DICOM path
    
    Args:
        project_root: Path to project root
        n_nodules: Number of nodules to extract
    """
    try:
        import pylidc as pl
    except ImportError:
        print("[prepare_dataset] ERROR: pylidc not installed.")
        print("Run: pip install pylidc")
        print("Then configure ~/.pylidcrc with your LIDC-IDRI path")
        sys.exit(1)
    
    subset_dir = project_root / "data" / "subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[prepare_dataset] Querying LIDC-IDRI for nodules...")
    
    # Query for nodules with at least 3 annotations
    # We want a balanced dataset across malignancy scores
    nodules_by_malignancy = {1: [], 2: [], 3: [], 4: [], 5: []}
    
    # Get all annotations
    all_annotations = pl.query(pl.Annotation).all()
    
    # Group annotations by nodule
    nodule_annotations = {}
    for ann in all_annotations:
        scan_id = ann.scan.patient_id
        # Use centroid to identify same nodule
        centroid = tuple(ann.centroid.astype(int))
        key = (scan_id, centroid[2])  # Group by scan and z-slice
        
        if key not in nodule_annotations:
            nodule_annotations[key] = []
        nodule_annotations[key].append(ann)
    
    print(f"[prepare_dataset] Found {len(nodule_annotations)} unique nodule clusters")
    
    # Process nodules
    count = 0
    for key, annotations in nodule_annotations.items():
        if count >= n_nodules:
            break
            
        if len(annotations) < 3:
            continue
            
        # Compute consensus
        consensus = compute_consensus(annotations)
        malignancy = consensus.get("malignancy", 3)
        
        # Balance dataset
        if len(nodules_by_malignancy[malignancy]) >= n_nodules // 5 + 5:
            continue
        
        try:
            # Use first annotation for image extraction
            ann = annotations[0]
            scan = ann.scan
            
            # Get image patch
            vol = scan.to_volume()
            bbox = ann.bbox()
            
            # Extract central slice
            z_center = (bbox[2].start + bbox[2].stop) // 2
            patch = vol[bbox[0], bbox[1], z_center]
            
            # Resize to 64x64
            patch_normalized = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
            patch_uint8 = (patch_normalized * 255).astype(np.uint8)
            
            img = Image.fromarray(patch_uint8, mode='L')
            img = img.resize((64, 64), Image.Resampling.BILINEAR)
            
            # Create nodule ID
            nodule_id = f"{count + 1:03d}"
            
            # Save image
            img.save(subset_dir / f"nodule_{nodule_id}.png")
            
            # Get location from scan info
            location = "lung parenchyma"  # Default
            
            # Prepare features
            features = {
                "nodule_id": nodule_id,
                **consensus,
                **get_semantic_labels(consensus),
                "location": location,
                "scan_id": scan.patient_id,
                "notes": f"Extracted from LIDC-IDRI scan {scan.patient_id}"
            }
            
            # Save features
            with open(subset_dir / f"nodule_{nodule_id}.json", 'w') as f:
                json.dump(features, f, indent=4)
            
            nodules_by_malignancy[malignancy].append(nodule_id)
            count += 1
            
            if count % 10 == 0:
                print(f"[prepare_dataset] Processed {count} nodules...")
                
        except Exception as e:
            print(f"[prepare_dataset] Warning: Failed to process nodule: {e}")
            continue
    
    print(f"\n[prepare_dataset] Successfully extracted {count} nodules")
    print(f"[prepare_dataset] Distribution by malignancy:")
    for mal, nodules in nodules_by_malignancy.items():
        print(f"  Malignancy {mal}: {len(nodules)} nodules")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare lung nodule dataset for MAS project"
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use pre-included fallback dataset (10 nodules, no download)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Extract from full LIDC-IDRI dataset (requires pylidc setup)"
    )
    parser.add_argument(
        "--n_nodules",
        type=int,
        default=100,
        help="Number of nodules to extract (for --full, default: 100)"
    )
    
    args = parser.parse_args()
    
    project_root = find_project_root()
    print(f"[prepare_dataset] Project root: {project_root}")
    
    if args.full:
        print("[prepare_dataset] Preparing full dataset from LIDC-IDRI...")
        prepare_full_dataset(project_root, args.n_nodules)
    elif args.fallback:
        print("[prepare_dataset] Setting up fallback dataset...")
        prepare_fallback_dataset(project_root)
    else:
        print("[prepare_dataset] No option specified.")
        print("Use --fallback for quick setup or --full for LIDC-IDRI extraction")
        print("\nQuick start:")
        print("  python -m data.prepare_dataset --fallback")
        parser.print_help()


if __name__ == "__main__":
    main()
