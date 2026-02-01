#!/usr/bin/env python3
"""
LIDC-XML Import Utility
=======================

Parses LIDC-IDRI XML files (without DICOM images) to extract:
1. Nodule characteristics (malignancy, spiculation, etc.)
2. Nodule clusters (grouping multiple radiologist reads)
3. Consensus features (median of reads)

Outputs JSON files compatible with LIDCLoader.
"""

import argparse
import xml.etree.ElementTree as ET
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_project_root() -> Path:
    """Find the project root directory."""
    current = Path(__file__).parent.parent
    while current != current.parent:
        if (current / "requirements.txt").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent

class LIDCXmlParser:
    def __init__(self, xml_dir: str, output_dir: str):
        self.xml_dir = Path(xml_dir)
        self.output_dir = Path(output_dir)
        self.ns = {'ns': 'http://www.nih.gov'}  # XML Namespace
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self):
        """Main execution flow."""
        logger.info(f"Scanning {self.xml_dir} for XML files...")
        
        xml_files = list(self.xml_dir.rglob("*.xml"))
        logger.info(f"Found {len(xml_files)} XML files.")
        
        count = 0
        for xml_file in xml_files:
            try:
                nodules = self._parse_xml(xml_file)
                if nodules:
                    count += self._save_nodules(nodules, count)
            except Exception as e:
                logger.error(f"Failed to parse {xml_file.name}: {e}")
                
        logger.info(f"Successfully imported {count} nodules.")

    def _parse_xml(self, xml_path: Path) -> List[Dict[str, Any]]:
        """Parse single XML file and extract nodules."""
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        
        # Format: (scan_z, centroid_x, centroid_y) -> [annotations]
        # We group by Z-position and rough centroid to cluster reads of the same nodule
        clusters: Dict[Tuple[float, int, int], List[Dict]] = {}
        
        for sess in root.findall('ns:readingSession', self.ns):
            for nodule in sess.findall('ns:unblindedReadNodule', self.ns):
                chars = nodule.find('ns:characteristics', self.ns)
                roi = nodule.find('ns:roi', self.ns)
                
                # We only want nodules with full characteristics (not small nodules <3mm)
                if chars is None or roi is None:
                    continue
                
                # Extract features
                features = {
                    "malignancy": int(chars.find('ns:malignancy', self.ns).text),
                    "spiculation": int(chars.find('ns:spiculation', self.ns).text),
                    "texture": int(chars.find('ns:texture', self.ns).text),
                    "calcification": int(chars.find('ns:calcification', self.ns).text),
                    "margin": int(chars.find('ns:margin', self.ns).text),
                    "sphericity": int(chars.find('ns:sphericity', self.ns).text),
                    "lobulation": int(chars.find('ns:lobulation', self.ns).text),
                    "internal_structure": int(chars.find('ns:internalStructure', self.ns).text),
                    "subtlety": int(chars.find('ns:subtlety', self.ns).text),
                }
                
                # Get centroid from ROI
                z_pos = float(roi.find('ns:imageZposition', self.ns).text)
                edge_maps = roi.findall('ns:edgeMap', self.ns)
                if not edge_maps:
                    continue
                    
                xs = [int(e.find('ns:xCoord', self.ns).text) for e in edge_maps]
                ys = [int(e.find('ns:yCoord', self.ns).text) for e in edge_maps]
                
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                diameter = np.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2) * 0.7  # Approx pixel-to-mm (rough)
                features['diameter_mm'] = float(diameter)
                
                # Simple Clustering: Find existing cluster within threshold
                found_cluster = False
                for (cz, ccx, ccy) in list(clusters.keys()):
                    # Match if on same/close Z slice and close in XY plane
                    if abs(cz - z_pos) < 1.0 and np.sqrt((ccx-cx)**2 + (ccy-cy)**2) < 20:
                        clusters[(cz, ccx, ccy)].append(features)
                        found_cluster = True
                        break
                
                if not found_cluster:
                    clusters[(z_pos, cx, cy)] = [features]
                    
        # Compute consensus for clusters with > 1 read
        consensus_nodules = []
        for key, annotations in clusters.items():
            if len(annotations) < 2: # Require at least 2 reads for consensus
                continue
            
            consensus = self._compute_consensus(annotations)
            consensus_nodules.append(consensus)
            
        return consensus_nodules

    def _compute_consensus(self, annotations: List[Dict]) -> Dict:
        """Compute median consensus."""
        consensus = {}
        for key in annotations[0].keys():
            values = [a[key] for a in annotations]
            median_val = np.median(values)
            if key != 'diameter_mm':
                consensus[key] = int(round(median_val))
            else:
                consensus[key] = round(median_val, 1)
                
        # Add labels
        from prepare_dataset import get_semantic_labels
        consensus.update(get_semantic_labels(consensus))
        
        return consensus

    def _save_nodules(self, nodules: List[Dict], start_id: int) -> int:
        """Save nodules to JSON and generate placeholder images."""
        saved_count = 0
        from lidc_loader import LIDCLoader
        lidc = LIDCLoader(None) # Helper for image gen
        
        for i, features in enumerate(nodules):
            nodule_id = f"{start_id + i + 1:03d}"
            
            # 1. Save JSON
            json_path = self.output_dir / f"nodule_{nodule_id}.json"
            features['nodule_id'] = nodule_id
            features['source'] = "XML Import"
            
            with open(json_path, 'w') as f:
                json.dump(features, f, indent=4)
                
            # 2. Generate Synthetic Image (since DICOM unavailable)
            # We use the existing helper in existing LIDCLoader to keep system working
            img_arr = lidc._generate_synthetic_image(features)
            
            # Determine suitable intensity scale (0-255)
            img_uint8 = (img_arr * 255).astype(np.uint8)
            img = Image.fromarray(img_uint8, mode='L')
            img.save(self.output_dir / f"nodule_{nodule_id}.png")
            
            saved_count += 1
            
        return saved_count

def main():
    parser = argparse.ArgumentParser(description="Import LIDC XML data")
    parser.add_argument("--input", required=True, help="Path to XML directory")
    args = parser.parse_args()
    
    root = find_project_root()
    output_dir = root / "data" / "subset"
    
    # Clean previous subset
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    
    parser = LIDCXmlParser(args.input, str(output_dir))
    parser.run()

if __name__ == "__main__":
    sys.path.append(str(find_project_root())) # Ensure we can import from data.
    main()
