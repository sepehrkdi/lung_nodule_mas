"""
Nodule Case Selection Script
============================

EDUCATIONAL PURPOSE - DATASET CURATION:

This script filters the IU/Open-I Chest X-ray dataset to select
30-50 reports mentioning lung nodules for manual annotation and
evaluation of the NLP pipeline.

SELECTION CRITERIA:
1. Report must mention "nodule", "nodular", "mass", or "lesion"
2. At least one positive (non-negated) mention preferred
3. Diversity in:
   - Size mentions (small/large)
   - Location mentions (different lobes)
   - Certainty levels (affirmed/uncertain/negated)

OUTPUT:
- A manifest JSON file with selected case IDs and metadata
- Summary statistics about the selected cases
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter

# Try to import project modules
try:
    from nlp.negation_detector import NegExDetector, get_certainty
    from nlp.report_parser import ReportParser
    HAS_NLP = True
except ImportError:
    HAS_NLP = False
    print("[Warning] NLP modules not available. Using basic selection.")


@dataclass
class CaseInfo:
    """Information about a selected case."""
    case_id: str
    xml_path: str
    image_paths: List[str]
    report_text: str
    
    # Selection criteria results
    nodule_mentions: int
    certainty_distribution: Dict[str, int]  # {"affirmed": N, "negated": M, ...}
    has_size_mention: bool
    has_location_mention: bool
    sections_found: List[str]
    
    # Priority score for selection
    priority_score: float


class NoduleCaseSelector:
    """
    Select nodule-mentioning cases from IU/Open-I dataset.
    
    EDUCATIONAL PURPOSE:
    This demonstrates how to:
    1. Parse XML reports for content extraction
    2. Apply NLP-based filtering criteria
    3. Create balanced evaluation datasets
    """
    
    # Keywords indicating nodule presence
    NODULE_KEYWORDS = [
        r'\bnodule\b',
        r'\bnodules\b',
        r'\bnodular\b',
        r'\bmass\b',
        r'\bmasses\b',
        r'\blesion\b',
        r'\blesions\b',
        r'\bopacity\b',
        r'\bopacities\b',
    ]
    
    # Size patterns
    SIZE_PATTERNS = [
        r'\d+\s*mm',
        r'\d+\s*cm',
        r'\d+\.?\d*\s*x\s*\d+\.?\d*',
    ]
    
    # Location patterns
    LOCATION_PATTERNS = [
        r'(right|left)\s+(upper|middle|lower)\s+lobe',
        r'RUL|RML|RLL|LUL|LLL',
        r'lingula',
        r'apex',
        r'base',
        r'perihilar',
    ]
    
    def __init__(self, data_dir: str, target_count: int = 40):
        """
        Initialize selector.
        
        Args:
            data_dir: Path to IU/Open-I dataset directory
            target_count: Target number of cases to select (30-50)
        """
        self.data_dir = Path(data_dir)
        self.target_count = min(max(target_count, 30), 50)
        
        # Initialize NLP components if available
        if HAS_NLP:
            self.negation_detector = NegExDetector()
            self.report_parser = ReportParser()
        else:
            self.negation_detector = None
            self.report_parser = None
    
    def find_xml_files(self) -> List[Path]:
        """Find all XML report files in dataset."""
        xml_pattern = "*.xml"
        
        # Common IU/Open-I directory structures
        search_paths = [
            self.data_dir / "reports" / xml_pattern,
            self.data_dir / "ecgen-radiology" / xml_pattern,
            self.data_dir / xml_pattern,
        ]
        
        all_files = []
        for pattern in search_paths:
            parent = pattern.parent
            if parent.exists():
                all_files.extend(parent.glob(pattern.name))
        
        return list(set(all_files))
    
    def parse_xml_report(self, xml_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse IU/Open-I XML report format.
        
        Returns dict with:
        - findings: str
        - impression: str
        - indication: str
        - images: List[str]
        """
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            report_data = {
                "findings": "",
                "impression": "",
                "indication": "",
                "images": [],
                "case_id": xml_path.stem,
            }
            
            # Extract text sections
            for elem in root.iter():
                tag = elem.tag.lower()
                text = elem.text or ""
                
                if "finding" in tag:
                    report_data["findings"] += " " + text.strip()
                elif "impression" in tag:
                    report_data["impression"] += " " + text.strip()
                elif "indication" in tag or "history" in tag:
                    report_data["indication"] += " " + text.strip()
                elif "image" in tag:
                    # Look for image references
                    if elem.attrib.get("id"):
                        report_data["images"].append(elem.attrib["id"])
            
            # Combine into full report text
            report_data["full_text"] = f"""
FINDINGS: {report_data["findings"].strip()}

IMPRESSION: {report_data["impression"].strip()}

INDICATION: {report_data["indication"].strip()}
""".strip()
            
            return report_data
            
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return None
    
    def has_nodule_mention(self, text: str) -> Tuple[bool, int]:
        """Check if text mentions nodules. Returns (has_mention, count)."""
        count = 0
        text_lower = text.lower()
        
        for pattern in self.NODULE_KEYWORDS:
            matches = re.findall(pattern, text_lower)
            count += len(matches)
        
        return count > 0, count
    
    def analyze_case(self, xml_path: Path) -> Optional[CaseInfo]:
        """Analyze a case and return CaseInfo if it has nodule mentions."""
        report = self.parse_xml_report(xml_path)
        if not report:
            return None
        
        text = report.get("full_text", "")
        has_nodule, nodule_count = self.has_nodule_mention(text)
        
        if not has_nodule:
            return None
        
        # Analyze certainty distribution
        certainty_dist = {"affirmed": 0, "negated": 0, "uncertain": 0}
        if self.negation_detector:
            for pattern in self.NODULE_KEYWORDS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = (match.group(), match.start(), match.end())
                    results = self.negation_detector.detect(text, [entity])
                    if results:
                        cert = results[0].certainty.value
                        certainty_dist[cert] = certainty_dist.get(cert, 0) + 1
        
        # Check for size mentions
        has_size = any(re.search(p, text, re.IGNORECASE) for p in self.SIZE_PATTERNS)
        
        # Check for location mentions
        has_location = any(re.search(p, text, re.IGNORECASE) for p in self.LOCATION_PATTERNS)
        
        # Parse sections
        sections_found = []
        if self.report_parser:
            parsed = self.report_parser.parse(text)
            sections_found = list(parsed.sections.keys())
        
        # Calculate priority score
        # Higher priority for: affirmed mentions, size info, location info, varied certainty
        score = 0.0
        score += certainty_dist.get("affirmed", 0) * 2.0  # Prefer affirmed
        score += certainty_dist.get("uncertain", 0) * 1.5  # Uncertain is interesting
        score += 1.0 if has_size else 0
        score += 1.0 if has_location else 0
        score += len(sections_found) * 0.5
        
        return CaseInfo(
            case_id=report["case_id"],
            xml_path=str(xml_path),
            image_paths=report.get("images", []),
            report_text=text,
            nodule_mentions=nodule_count,
            certainty_distribution=certainty_dist,
            has_size_mention=has_size,
            has_location_mention=has_location,
            sections_found=sections_found,
            priority_score=score
        )
    
    def select_cases(self) -> List[CaseInfo]:
        """
        Select nodule cases from dataset.
        
        Strategy:
        1. Scan all XML files for nodule mentions
        2. Analyze each candidate
        3. Rank by priority score
        4. Select top N with diversity
        """
        xml_files = self.find_xml_files()
        print(f"Found {len(xml_files)} XML files to scan")
        
        candidates = []
        for xml_path in xml_files:
            case_info = self.analyze_case(xml_path)
            if case_info:
                candidates.append(case_info)
        
        print(f"Found {len(candidates)} cases with nodule mentions")
        
        if len(candidates) == 0:
            print("No nodule cases found. Creating synthetic test cases.")
            return self._create_synthetic_cases()
        
        # Sort by priority score
        candidates.sort(key=lambda x: x.priority_score, reverse=True)
        
        # Select with diversity
        selected = self._select_diverse(candidates)
        
        return selected
    
    def _select_diverse(self, candidates: List[CaseInfo]) -> List[CaseInfo]:
        """Select diverse set of cases."""
        selected = []
        selected_certainties = Counter()
        
        for case in candidates:
            if len(selected) >= self.target_count:
                break
            
            # Add diversity - don't over-represent any certainty type
            case_certainty = max(case.certainty_distribution.items(), 
                               key=lambda x: x[1], default=("unknown", 0))[0]
            
            if selected_certainties[case_certainty] < self.target_count // 3:
                selected.append(case)
                selected_certainties[case_certainty] += 1
        
        # Fill remaining slots with highest priority
        for case in candidates:
            if len(selected) >= self.target_count:
                break
            if case not in selected:
                selected.append(case)
        
        return selected
    
    def _create_synthetic_cases(self) -> List[CaseInfo]:
        """Create synthetic test cases if no real data available."""
        synthetic = []
        
        test_reports = [
            ("test_001", "FINDINGS: A 15mm solid nodule in the right upper lobe. IMPRESSION: Suspicious for malignancy."),
            ("test_002", "FINDINGS: No pulmonary nodules identified. IMPRESSION: Normal chest."),
            ("test_003", "FINDINGS: Possible 8mm nodule in LUL. IMPRESSION: Cannot exclude malignancy."),
            ("test_004", "FINDINGS: Multiple bilateral nodules. IMPRESSION: Consider metastatic disease."),
            ("test_005", "FINDINGS: Stable 12mm ground-glass nodule. IMPRESSION: Likely benign, recommend follow-up."),
        ]
        
        for case_id, text in test_reports:
            has_nodule, count = self.has_nodule_mention(text)
            
            # Default certainty analysis
            certainty_dist = {"affirmed": 0, "negated": 0, "uncertain": 0}
            if "no " in text.lower() or "without" in text.lower():
                certainty_dist["negated"] = 1
            elif "possible" in text.lower() or "cannot exclude" in text.lower():
                certainty_dist["uncertain"] = 1
            else:
                certainty_dist["affirmed"] = count
            
            synthetic.append(CaseInfo(
                case_id=case_id,
                xml_path=f"synthetic/{case_id}.xml",
                image_paths=[],
                report_text=text,
                nodule_mentions=count,
                certainty_distribution=certainty_dist,
                has_size_mention=bool(re.search(r'\d+\s*mm', text)),
                has_location_mention=bool(re.search(r'(right|left)', text, re.I)),
                sections_found=["FINDINGS", "IMPRESSION"],
                priority_score=1.0
            ))
        
        return synthetic
    
    def save_manifest(self, cases: List[CaseInfo], output_path: str) -> None:
        """Save selected cases to manifest file."""
        manifest = {
            "total_cases": len(cases),
            "selection_criteria": {
                "keywords": [p.replace(r'\b', '') for p in self.NODULE_KEYWORDS],
                "target_count": self.target_count,
            },
            "summary": {
                "with_size_mention": sum(1 for c in cases if c.has_size_mention),
                "with_location_mention": sum(1 for c in cases if c.has_location_mention),
                "certainty_counts": {
                    "affirmed": sum(c.certainty_distribution.get("affirmed", 0) for c in cases),
                    "negated": sum(c.certainty_distribution.get("negated", 0) for c in cases),
                    "uncertain": sum(c.certainty_distribution.get("uncertain", 0) for c in cases),
                },
            },
            "cases": [asdict(c) for c in cases]
        }
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Saved manifest to {output_path}")


def select_nodule_cases(
    data_dir: str,
    output_manifest: str = "nodule_cases_manifest.json",
    target_count: int = 40
) -> List[CaseInfo]:
    """
    Main function to select nodule cases from dataset.
    
    Args:
        data_dir: Path to IU/Open-I dataset
        output_manifest: Path for output manifest JSON
        target_count: Target number of cases (30-50)
        
    Returns:
        List of selected CaseInfo objects
    """
    selector = NoduleCaseSelector(data_dir, target_count)
    cases = selector.select_cases()
    
    if output_manifest:
        selector.save_manifest(cases, output_manifest)
    
    return cases


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Select nodule cases from IU/Open-I dataset")
    parser.add_argument("--data-dir", type=str, default="./data/openi",
                       help="Path to dataset directory")
    parser.add_argument("--output", type=str, default="nodule_cases_manifest.json",
                       help="Output manifest file path")
    parser.add_argument("--count", type=int, default=40,
                       help="Target number of cases to select")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Nodule Case Selection")
    print("=" * 60)
    
    cases = select_nodule_cases(
        data_dir=args.data_dir,
        output_manifest=args.output,
        target_count=args.count
    )
    
    print(f"\nSelected {len(cases)} cases:")
    for i, case in enumerate(cases[:5]):
        print(f"\n{i+1}. {case.case_id}")
        print(f"   Nodule mentions: {case.nodule_mentions}")
        print(f"   Certainty: {case.certainty_distribution}")
        print(f"   Size: {'Yes' if case.has_size_mention else 'No'}, "
              f"Location: {'Yes' if case.has_location_mention else 'No'}")
    
    if len(cases) > 5:
        print(f"\n... and {len(cases) - 5} more cases")
