"""
Report Parser - Section Splitting and Weighting
================================================

EDUCATIONAL PURPOSE - RADIOLOGY REPORT STRUCTURE:

Radiology reports follow a standardized format with distinct sections:

1. INDICATION: Why the exam was ordered (e.g., "cough, rule out pneumonia")
2. TECHNIQUE: How the exam was performed (e.g., "PA and lateral views")
3. COMPARISON: Previous exams compared to (e.g., "Comparison: CT 01/01/2025")
4. FINDINGS: Detailed observations from the images
5. IMPRESSION: Summary and clinical interpretation (most important!)

SECTION WEIGHTING RATIONALE (per Pons et al., 2016):
- IMPRESSION carries the radiologist's synthesized opinion → highest weight
- FINDINGS contain detailed observations → standard weight
- INDICATION provides context but not diagnostic info → lower weight

Reference:
    Pons et al. (2016). "Natural Language Processing in Radiology: A Systematic Review"
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class SectionData:
    """Data for a single report section."""
    name: str
    text: str
    weight: float
    start_pos: int = 0
    end_pos: int = 0
    
    def weighted_text(self) -> str:
        """Return text with weight metadata."""
        return self.text
    
    def __str__(self) -> str:
        return f"{self.name}: {self.text[:50]}..." if len(self.text) > 50 else f"{self.name}: {self.text}"


@dataclass
class ParsedReport:
    """Complete parsed radiology report."""
    sections: Dict[str, SectionData] = field(default_factory=dict)
    raw_text: str = ""
    
    @property
    def findings(self) -> str:
        return self.sections.get("FINDINGS", SectionData("FINDINGS", "", 1.0)).text
    
    @property
    def impression(self) -> str:
        return self.sections.get("IMPRESSION", SectionData("IMPRESSION", "", 1.5)).text
    
    @property
    def indication(self) -> str:
        return self.sections.get("INDICATION", SectionData("INDICATION", "", 0.5)).text
    
    def get_weighted_sections(self) -> List[tuple]:
        """Return sections sorted by weight (highest first)."""
        return sorted(
            [(s.name, s.text, s.weight) for s in self.sections.values()],
            key=lambda x: x[2],
            reverse=True
        )
    
    def get_combined_text(self, weighted: bool = False) -> str:
        """
        Get combined text from all sections.
        
        Args:
            weighted: If True, repeat high-weight sections to simulate importance
        """
        if not weighted:
            return " ".join(s.text for s in self.sections.values() if s.text)
        
        # Weight by repetition (simple approach)
        parts = []
        for section in self.sections.values():
            if section.text:
                # Repeat based on weight (1.5 -> text appears ~1.5x in combined)
                parts.append(section.text)
                if section.weight > 1.0:
                    parts.append(section.text)  # Extra copy for high-weight sections
        return " ".join(parts)


class ReportParser:
    """
    Parse radiology reports into weighted sections.
    
    EDUCATIONAL PURPOSE:
    This parser demonstrates how to:
    1. Split reports into semantic sections
    2. Assign differential weights for downstream processing
    3. Handle variations in report formatting
    
    Usage:
        parser = ReportParser()
        parsed = parser.parse("FINDINGS: nodule present. IMPRESSION: suspicious.")
        print(parsed.impression)  # "suspicious."
    """
    
    # Section headers to detect (order matters for parsing)
    SECTION_HEADERS = [
        "IMPRESSION:",
        "IMPRESSION",
        "FINDINGS:",
        "FINDINGS",
        "INDICATION:",
        "INDICATION",
        "CLINICAL INDICATION:",
        "CLINICAL HISTORY:",
        "HISTORY:",
        "TECHNIQUE:",
        "COMPARISON:",
        "COMPARISON",
        "CONCLUSION:",
        "RECOMMENDATION:",
        "RECOMMENDATIONS:",
    ]
    
    # Weights for each section type (higher = more important for malignancy assessment)
    SECTION_WEIGHTS = {
        "IMPRESSION": 1.5,       # Radiologist's synthesis - highest importance
        "CONCLUSION": 1.5,       # Alternative name for impression
        "FINDINGS": 1.0,         # Detailed observations - standard weight
        "RECOMMENDATION": 0.8,   # Follow-up recommendations
        "RECOMMENDATIONS": 0.8,
        "INDICATION": 0.5,       # Clinical context - lower weight
        "CLINICAL INDICATION": 0.5,
        "CLINICAL HISTORY": 0.5,
        "HISTORY": 0.5,
        "TECHNIQUE": 0.2,        # Technical details - minimal weight
        "COMPARISON": 0.3,       # Previous exam references
        "PREAMBLE": 0.3,         # Text before any labeled section
    }
    
    def __init__(self):
        """Initialize the parser."""
        # Build regex pattern for section detection
        self._header_pattern = self._build_header_pattern()
    
    def _build_header_pattern(self) -> re.Pattern:
        """Build regex pattern to match section headers."""
        # Sort by length (longest first) for greedy matching
        sorted_headers = sorted(self.SECTION_HEADERS, key=len, reverse=True)
        
        # Create pattern that matches headers at start of line or after newline
        escaped = [re.escape(h) for h in sorted_headers]
        pattern = r'(?:^|\n)\s*(' + '|'.join(escaped) + r')\s*'
        
        return re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    
    def parse(self, report: str) -> ParsedReport:
        """
        Parse a radiology report into sections.
        
        Args:
            report: Raw report text
            
        Returns:
            ParsedReport with sections dictionary
        """
        if not report or not report.strip():
            return ParsedReport(raw_text=report)
        
        result = ParsedReport(raw_text=report)
        
        # Find all section headers and their positions
        matches = list(self._header_pattern.finditer(report))
        
        if not matches:
            # No headers found - treat entire text as FINDINGS
            result.sections["FINDINGS"] = SectionData(
                name="FINDINGS",
                text=report.strip(),
                weight=self.SECTION_WEIGHTS.get("FINDINGS", 1.0),
                start_pos=0,
                end_pos=len(report)
            )
            return result
        
        # Handle text before first section header (preamble)
        first_match = matches[0]
        if first_match.start() > 0:
            preamble_text = report[:first_match.start()].strip()
            if preamble_text:
                result.sections["PREAMBLE"] = SectionData(
                    name="PREAMBLE",
                    text=preamble_text,
                    weight=self.SECTION_WEIGHTS.get("PREAMBLE", 0.3),
                    start_pos=0,
                    end_pos=first_match.start()
                )
        
        # Extract each section
        for i, match in enumerate(matches):
            header = match.group(1).upper().rstrip(':')
            section_start = match.end()
            
            # Section ends at next header or end of text
            if i + 1 < len(matches):
                section_end = matches[i + 1].start()
            else:
                section_end = len(report)
            
            section_text = report[section_start:section_end].strip()
            
            # Normalize header name
            header_normalized = self._normalize_header(header)
            weight = self.SECTION_WEIGHTS.get(header_normalized, 0.5)
            
            result.sections[header_normalized] = SectionData(
                name=header_normalized,
                text=section_text,
                weight=weight,
                start_pos=section_start,
                end_pos=section_end
            )
        
        return result
    
    def _normalize_header(self, header: str) -> str:
        """Normalize header name to standard form."""
        header = header.upper().rstrip(':').strip()
        
        # Map variations to standard names
        mappings = {
            "CLINICAL INDICATION": "INDICATION",
            "CLINICAL HISTORY": "INDICATION",
            "HISTORY": "INDICATION",
            "CONCLUSION": "IMPRESSION",
            "RECOMMENDATIONS": "RECOMMENDATION",
        }
        
        return mappings.get(header, header)
    
    def get_section_weight(self, section_name: str) -> float:
        """Get the weight for a section name."""
        normalized = self._normalize_header(section_name)
        return self.SECTION_WEIGHTS.get(normalized, 0.5)
    
    def extract_findings_and_impression(self, report: str) -> Dict[str, str]:
        """
        Quick extraction of just FINDINGS and IMPRESSION.
        
        Convenience method for common use case.
        """
        parsed = self.parse(report)
        return {
            "findings": parsed.findings,
            "impression": parsed.impression,
            "indication": parsed.indication
        }
    
    def calculate_section_scores(
        self, 
        report: str, 
        keyword_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate weighted scores per section based on keyword matches.
        
        Args:
            report: Raw report text
            keyword_scores: Dict mapping keywords to their scores
            
        Returns:
            Dict with section names and weighted scores
        """
        parsed = self.parse(report)
        section_scores = {}
        
        for section_name, section_data in parsed.sections.items():
            text_lower = section_data.text.lower()
            
            # Sum scores for keywords found in this section
            raw_score = sum(
                score for keyword, score in keyword_scores.items()
                if keyword.lower() in text_lower
            )
            
            # Apply section weight
            weighted_score = raw_score * section_data.weight
            section_scores[section_name] = weighted_score
        
        return section_scores


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def split_sections(report: str) -> Dict[str, str]:
    """
    Quick section splitting function.
    
    Args:
        report: Raw radiology report text
        
    Returns:
        Dict mapping section names to text content
    """
    parser = ReportParser()
    parsed = parser.parse(report)
    return {name: section.text for name, section in parsed.sections.items()}


def get_weighted_text(report: str, section_name: str) -> tuple:
    """
    Get section text with its weight.
    
    Returns:
        Tuple of (text, weight)
    """
    parser = ReportParser()
    parsed = parser.parse(report)
    section = parsed.sections.get(section_name.upper())
    if section:
        return section.text, section.weight
    return "", 0.0


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Report Parser Demo")
    print("=" * 60)
    
    # Sample radiology report
    sample_report = """
INDICATION: 65-year-old male with persistent cough. Rule out lung mass.

TECHNIQUE: PA and lateral chest radiographs.

COMPARISON: Chest CT from 01/15/2025.

FINDINGS: 
The lungs are clear bilaterally. No focal consolidation, pleural effusion, 
or pneumothorax is identified. A 12mm nodule is noted in the right upper lobe, 
unchanged from prior CT. The nodule appears to have spiculated margins. 
No lymphadenopathy.

IMPRESSION:
1. 12mm spiculated nodule in right upper lobe, stable compared to prior CT.
2. Recommend continued surveillance with follow-up CT in 3 months given 
   spiculated morphology.
"""
    
    parser = ReportParser()
    parsed = parser.parse(sample_report)
    
    print("\n--- Parsed Sections ---\n")
    for name, section in parsed.sections.items():
        print(f"[{name}] (weight={section.weight})")
        print(f"  {section.text[:100]}...")
        print()
    
    print("\n--- Quick Access ---\n")
    print(f"IMPRESSION: {parsed.impression[:80]}...")
    print(f"FINDINGS: {parsed.findings[:80]}...")
    
    print("\n--- Weighted Sections (sorted by importance) ---\n")
    for name, text, weight in parsed.get_weighted_sections():
        print(f"  {name}: weight={weight}")
