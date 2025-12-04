"""
Specialized Pathologist Agents
==============================

EDUCATIONAL PURPOSE - DIVERSE NLP APPROACHES:

This module implements two pathologist agents with different 
text analysis strategies:

1. PathologistRegex: Pattern matching with regular expressions
2. PathologistSpacy: spaCy NER + semantic rules

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │                 Two Pathologist Agents                      │
    ├───────────────────────────┬─────────────────────────────────┤
    │   Regex-Based             │   spaCy NER + Rules             │
    │   (Pattern Matching)      │   (Statistical NLP)             │
    ├───────────────────────────┼─────────────────────────────────┤
    │   Weight: 0.8             │   Weight: 0.9                   │
    │   Fast, Interpretable     │   More Robust, Contextual       │
    │   Exact Pattern Match     │   Entity Recognition            │
    └───────────────────────────┴─────────────────────────────────┘

WHY DIFFERENT APPROACHES?
- Regex is explicit and interpretable but brittle
- spaCy handles variations and context better
- Disagreement reveals ambiguous or complex cases
- Demonstrates hybrid symbolic + statistical NLP
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from agents.spade_base import MedicalAgentBase, Belief, get_asl_path

logger = logging.getLogger(__name__)


# =============================================================================
# BASE PATHOLOGIST CLASS
# =============================================================================

class PathologistBase(MedicalAgentBase):
    """
    Base class for all pathologist agents.
    
    Provides common interface and utilities for report analysis.
    """
    
    AGENT_TYPE = "pathologist"
    APPROACH = "base"
    WEIGHT = 0.8
    
    def __init__(self, name: str, asl_file: Optional[str] = None):
        if asl_file is None:
            asl_file = get_asl_path("pathologist")
        super().__init__(name=name, asl_file=asl_file)
        
    @abstractmethod
    def _analyze_report(self, report: str) -> Dict[str, Any]:
        """
        Analyze report text and extract findings.
        Must be implemented by subclasses.
        """
        pass
    
    def _estimate_malignancy(self, findings: Dict[str, Any]) -> float:
        """Estimate malignancy probability from extracted findings."""
        prob = 0.5  # Base probability
        
        # Size-based adjustment
        size = findings.get("size_mm", 10)
        if size < 6:
            prob -= 0.2
        elif size < 8:
            prob -= 0.1
        elif size < 15:
            prob += 0.1
        else:
            prob += 0.25
        
        # Texture adjustment
        texture = findings.get("texture", "").lower()
        if "ground" in texture or "glass" in texture:
            prob -= 0.1
        elif "spicul" in texture:
            prob += 0.2
        
        # Suspicious terms
        suspicious = findings.get("suspicious_terms", [])
        prob += len(suspicious) * 0.1
        
        # Benign terms
        benign = findings.get("benign_terms", [])
        prob -= len(benign) * 0.1
        
        return min(max(prob, 0.05), 0.95)
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process report analysis request."""
        nodule_id = request.get("nodule_id", "unknown")
        report = request.get("report", "")
        features = request.get("features", {})
        
        # Use report or generate from features
        if not report and features:
            report = self._generate_report_from_features(features)
        
        logger.info(f"[{self.name}] Analyzing report for {nodule_id}")
        
        findings = self._analyze_report(report)
        probability = self._estimate_malignancy(findings)
        predicted_class = self._prob_to_class(probability)
        
        self.add_belief(Belief(
            "analysis",
            (nodule_id, probability, findings),
            annotations={"source": self.name, "approach": self.APPROACH}
        ))
        
        return {
            "nodule_id": nodule_id,
            "agent": self.name,
            "agent_type": self.AGENT_TYPE,
            "approach": self.APPROACH,
            "weight": self.WEIGHT,
            "status": "success",
            "findings": {
                "text_malignancy_probability": probability,
                "predicted_class": predicted_class,
                **findings
            }
        }
    
    def _generate_report_from_features(self, features: Dict[str, Any]) -> str:
        """Generate synthetic report from features."""
        size = features.get("size_mm", features.get("diameter_mm", 10))
        texture = features.get("texture", "solid")
        location = features.get("location", "right upper lobe")
        malignancy = features.get("malignancy", 3)
        
        findings = f"A {size:.0f}mm {texture} nodule is identified in the {location}."
        
        if malignancy >= 4:
            impression = "Suspicious for malignancy. Recommend further evaluation."
        elif malignancy == 3:
            impression = "Indeterminate nodule. Recommend follow-up."
        else:
            impression = "Likely benign appearance. Routine follow-up."
        
        return f"FINDINGS: {findings}\n\nIMPRESSION: {impression}"
    
    def _prob_to_class(self, prob: float) -> int:
        if prob < 0.2: return 1
        elif prob < 0.4: return 2
        elif prob < 0.6: return 3
        elif prob < 0.8: return 4
        else: return 5


# =============================================================================
# REGEX-BASED PATHOLOGIST
# =============================================================================

class PathologistRegex(PathologistBase):
    """
    Pathologist using regular expression pattern matching.
    
    EDUCATIONAL NOTE - REGEX NLP:
    Regular expressions provide:
    - Explicit, interpretable patterns
    - Fast execution
    - Good for structured text with known patterns
    
    Limitations:
    - Brittle to variations ("5 mm" vs "5mm" vs "five millimeters")
    - No semantic understanding
    - Requires manual pattern engineering
    """
    
    AGENT_TYPE = "pathologist"
    APPROACH = "regex"
    WEIGHT = 0.8
    
    # Size patterns
    SIZE_PATTERNS = [
        r"(\d+(?:\.\d+)?)\s*(?:mm|millimeter)",
        r"(\d+(?:\.\d+)?)\s*(?:cm|centimeter)",  # Will multiply by 10
        r"(?:measures?|measuring|size[sd]?)\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*x\s*\d+(?:\.\d+)?\s*(?:mm|cm)?",  # Dimensions
    ]
    
    # Texture patterns
    TEXTURE_PATTERNS = {
        "ground_glass": [
            r"ground[\s-]?glass",
            r"GGO",
            r"GGN",
            r"non[\s-]?solid",
            r"hazy\s+opacity"
        ],
        "part_solid": [
            r"part[\s-]?solid",
            r"partially\s+solid",
            r"mixed",
            r"subsolid"
        ],
        "solid": [
            r"\bsolid\b",
            r"soft\s+tissue",
            r"dense"
        ],
        "calcified": [
            r"calcif(?:ied|ication)",
            r"calcium"
        ]
    }
    
    # Suspicious term patterns
    SUSPICIOUS_PATTERNS = [
        r"malignan(?:t|cy)",
        r"carcinom",
        r"cancer",
        r"suspicio(?:us|n)",
        r"metasta",
        r"spicul",
        r"irregular\s+margin",
        r"poorly\s+defined",
        r"rapid\s+growth",
        r"concerning"
    ]
    
    # Benign term patterns
    BENIGN_PATTERNS = [
        r"benign",
        r"stable",
        r"unchanged",
        r"granuloma",
        r"hamartoma",
        r"well[\s-]?defined",
        r"smooth\s+margin",
        r"calcified",
        r"no\s+change",
        r"resolved"
    ]
    
    # Location patterns
    LOCATION_PATTERNS = {
        "right_upper_lobe": r"right\s+upper\s+lobe|RUL",
        "right_middle_lobe": r"right\s+middle\s+lobe|RML",
        "right_lower_lobe": r"right\s+lower\s+lobe|RLL",
        "left_upper_lobe": r"left\s+upper\s+lobe|LUL",
        "left_lower_lobe": r"left\s+lower\s+lobe|LLL",
        "lingula": r"lingula",
    }
    
    def __init__(self, name: str = "pathologist_regex"):
        super().__init__(name=name)
        
    def _analyze_report(self, report: str) -> Dict[str, Any]:
        """Analyze report using regex patterns."""
        report_lower = report.lower()
        
        findings = {
            "size_mm": self._extract_size(report_lower),
            "texture": self._extract_texture(report_lower),
            "location": self._extract_location(report_lower),
            "suspicious_terms": self._find_terms(report_lower, self.SUSPICIOUS_PATTERNS),
            "benign_terms": self._find_terms(report_lower, self.BENIGN_PATTERNS),
            "approach": "regex"
        }
        
        logger.info(
            f"[{self.name}] Regex extraction: "
            f"size={findings['size_mm']}mm, "
            f"texture={findings['texture']}, "
            f"suspicious={len(findings['suspicious_terms'])}, "
            f"benign={len(findings['benign_terms'])}"
        )
        
        return findings
    
    def _extract_size(self, text: str) -> float:
        """Extract nodule size in mm."""
        for pattern in self.SIZE_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                size = float(matches[0])
                # Check if cm (multiply by 10)
                if "cm" in text[text.find(matches[0]):text.find(matches[0])+10]:
                    size *= 10
                return size
        return 10.0  # Default
    
    def _extract_texture(self, text: str) -> str:
        """Extract nodule texture."""
        for texture, patterns in self.TEXTURE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return texture
        return "solid"  # Default
    
    def _extract_location(self, text: str) -> str:
        """Extract anatomical location."""
        for location, pattern in self.LOCATION_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return location
        return "unspecified"
    
    def _find_terms(self, text: str, patterns: List[str]) -> List[str]:
        """Find all matching terms from patterns."""
        found = []
        for pattern in patterns:
            if re.search(pattern, text):
                found.append(pattern)
        return found


# =============================================================================
# SPACY NER PATHOLOGIST
# =============================================================================

class PathologistSpacy(PathologistBase):
    """
    Pathologist using spaCy Named Entity Recognition + rules.
    
    EDUCATIONAL NOTE - STATISTICAL NLP:
    spaCy provides:
    - Pre-trained NER models (including medical: scispaCy)
    - Robust to text variations
    - Contextual understanding
    - Entity relationships
    
    Uses scispaCy (en_core_sci_sm) for medical text when available.
    """
    
    AGENT_TYPE = "pathologist"
    APPROACH = "spacy_ner"
    WEIGHT = 0.9
    
    # Medical entity types to look for
    MEDICAL_ENTITIES = [
        "DISEASE", "SYMPTOM", "ANATOMICAL_STRUCTURE",
        "PROCEDURE", "CHEMICAL", "BODY_PART"
    ]
    
    # Custom rules for malignancy assessment
    MALIGNANCY_INDICATORS = {
        "high": ["carcinoma", "malignant", "metastatic", "invasive", "aggressive"],
        "moderate": ["suspicious", "concerning", "indeterminate", "atypical"],
        "low": ["benign", "stable", "unchanged", "granuloma", "resolved"]
    }
    
    def __init__(self, name: str = "pathologist_spacy"):
        super().__init__(name=name)
        self._nlp = None
        self._nlp_loaded = False
        
    def _load_nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is not None:
            return
            
        try:
            # Try scispaCy first (medical NLP)
            import spacy
            try:
                self._nlp = spacy.load("en_core_sci_sm")
                logger.info(f"[{self.name}] Loaded scispaCy medical model")
            except OSError:
                # Fall back to standard English model
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                    logger.info(f"[{self.name}] Loaded standard spaCy model")
                except OSError:
                    logger.warning(f"[{self.name}] No spaCy model available")
                    self._nlp = None
                    
            if self._nlp:
                self._nlp_loaded = True
                self.add_belief(Belief("nlp_loaded", ("spacy", True)))
                
        except ImportError:
            logger.warning(f"[{self.name}] spaCy not installed")
            self._nlp = None
    
    def _analyze_report(self, report: str) -> Dict[str, Any]:
        """Analyze report using spaCy NER + rules."""
        self._load_nlp()
        
        if self._nlp:
            return self._spacy_analysis(report)
        else:
            return self._fallback_analysis(report)
    
    def _spacy_analysis(self, report: str) -> Dict[str, Any]:
        """Analyze using spaCy NLP pipeline."""
        doc = self._nlp(report)
        
        # Extract named entities
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        # Extract size using dependency parsing
        size_mm = self._extract_size_spacy(doc)
        
        # Determine texture from context
        texture = self._extract_texture_spacy(doc)
        
        # Analyze sentiment/malignancy indicators
        malignancy_score = self._assess_malignancy_spacy(doc)
        
        # Extract location
        location = self._extract_location_spacy(doc)
        
        findings = {
            "size_mm": size_mm,
            "texture": texture,
            "location": location,
            "entities": entities,
            "malignancy_score": malignancy_score,
            "suspicious_terms": self._find_indicator_terms(doc, "high") + 
                               self._find_indicator_terms(doc, "moderate"),
            "benign_terms": self._find_indicator_terms(doc, "low"),
            "approach": "spacy_ner"
        }
        
        logger.info(
            f"[{self.name}] spaCy analysis: "
            f"size={size_mm}mm, texture={texture}, "
            f"entities={len(entities)}, malignancy_score={malignancy_score:.2f}"
        )
        
        return findings
    
    def _extract_size_spacy(self, doc) -> float:
        """Extract size using spaCy's token analysis."""
        import re
        
        # Look for number + unit patterns
        for i, token in enumerate(doc):
            if token.like_num:
                # Check next token for unit
                if i + 1 < len(doc):
                    next_token = doc[i + 1].text.lower()
                    if next_token in ["mm", "millimeter", "millimeters"]:
                        try:
                            return float(token.text)
                        except ValueError:
                            pass
                    elif next_token in ["cm", "centimeter", "centimeters"]:
                        try:
                            return float(token.text) * 10
                        except ValueError:
                            pass
        
        # Fallback to regex
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:mm|cm)", doc.text.lower())
        if matches:
            size = float(matches[0])
            if "cm" in doc.text.lower():
                size *= 10
            return size
        
        return 10.0
    
    def _extract_texture_spacy(self, doc) -> str:
        """Extract texture using semantic analysis."""
        text_lower = doc.text.lower()
        
        # Check for texture-related phrases
        texture_map = {
            "ground_glass": ["ground glass", "ground-glass", "ggo", "ggn", "hazy"],
            "part_solid": ["part-solid", "part solid", "partially solid", "subsolid"],
            "calcified": ["calcified", "calcification", "calcium"],
            "solid": ["solid", "dense"]
        }
        
        for texture, keywords in texture_map.items():
            for kw in keywords:
                if kw in text_lower:
                    return texture
        
        return "solid"
    
    def _assess_malignancy_spacy(self, doc) -> float:
        """Assess malignancy probability using semantic indicators."""
        score = 0.0
        count = 0
        
        text_lower = doc.text.lower()
        
        for level, terms in self.MALIGNANCY_INDICATORS.items():
            for term in terms:
                if term in text_lower:
                    count += 1
                    if level == "high":
                        score += 0.3
                    elif level == "moderate":
                        score += 0.1
                    else:  # low
                        score -= 0.2
        
        # Normalize
        if count > 0:
            return max(0, min(1, 0.5 + score))
        return 0.5
    
    def _extract_location_spacy(self, doc) -> str:
        """Extract anatomical location using NER."""
        text_lower = doc.text.lower()
        
        # Check for anatomical entities
        for ent in doc.ents:
            if ent.label_ in ["ANATOMICAL_STRUCTURE", "BODY_PART", "ORG"]:
                ent_lower = ent.text.lower()
                if "lobe" in ent_lower:
                    return ent.text.replace(" ", "_").lower()
        
        # Fallback patterns
        location_patterns = {
            "right_upper_lobe": ["right upper", "rul"],
            "right_middle_lobe": ["right middle", "rml"],
            "right_lower_lobe": ["right lower", "rll"],
            "left_upper_lobe": ["left upper", "lul"],
            "left_lower_lobe": ["left lower", "lll"],
        }
        
        for location, patterns in location_patterns.items():
            for p in patterns:
                if p in text_lower:
                    return location
        
        return "unspecified"
    
    def _find_indicator_terms(self, doc, level: str) -> List[str]:
        """Find malignancy indicator terms in text."""
        found = []
        text_lower = doc.text.lower()
        
        for term in self.MALIGNANCY_INDICATORS.get(level, []):
            if term in text_lower:
                found.append(term)
        
        return found
    
    def _fallback_analysis(self, report: str) -> Dict[str, Any]:
        """Fallback analysis without spaCy."""
        import re
        
        report_lower = report.lower()
        
        # Basic regex extraction
        size_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm|cm)", report_lower)
        size_mm = float(size_match.group(1)) if size_match else 10.0
        if size_match and "cm" in report_lower[size_match.start():size_match.end()+5]:
            size_mm *= 10
        
        # Texture
        texture = "solid"
        if "ground" in report_lower:
            texture = "ground_glass"
        elif "part" in report_lower and "solid" in report_lower:
            texture = "part_solid"
        
        # Malignancy terms
        suspicious = [t for t in self.MALIGNANCY_INDICATORS["high"] + 
                      self.MALIGNANCY_INDICATORS["moderate"] if t in report_lower]
        benign = [t for t in self.MALIGNANCY_INDICATORS["low"] if t in report_lower]
        
        return {
            "size_mm": size_mm,
            "texture": texture,
            "location": "unspecified",
            "entities": [],
            "malignancy_score": 0.5 + len(suspicious) * 0.1 - len(benign) * 0.1,
            "suspicious_terms": suspicious,
            "benign_terms": benign,
            "approach": "fallback"
        }
    
    def _estimate_malignancy(self, findings: Dict[str, Any]) -> float:
        """Override to use spaCy's malignancy score."""
        base_prob = findings.get("malignancy_score", 0.5)
        
        # Adjust based on size
        size = findings.get("size_mm", 10)
        if size < 6:
            base_prob -= 0.15
        elif size < 8:
            base_prob -= 0.05
        elif size >= 15:
            base_prob += 0.15
        
        return min(max(base_prob, 0.05), 0.95)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_pathologist_regex(name: str = "pathologist_regex"):
    """Create regex-based pathologist agent."""
    return PathologistRegex(name=name)

def create_pathologist_spacy(name: str = "pathologist_spacy"):
    """Create spaCy NER pathologist agent."""
    return PathologistSpacy(name=name)

def create_all_pathologists():
    """Create both pathologist agents."""
    return [
        PathologistRegex(name="pathologist_regex"),
        PathologistSpacy(name="pathologist_spacy")
    ]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== Pathologist Agents Test ===\n")
    
    # Test report
    test_report = """
    FINDINGS: A 15mm solid nodule is identified in the right upper lobe. 
    The lesion demonstrates mild spiculation with somewhat irregular margins.
    No calcification is present.
    
    IMPRESSION: Suspicious pulmonary nodule. Findings are concerning for 
    malignancy. Recommend PET-CT for further evaluation.
    """
    
    test_request = {
        "nodule_id": "test_001",
        "report": test_report
    }
    
    # Create both pathologists
    pathologists = create_all_pathologists()
    
    async def test():
        print("Test Report:")
        print(test_report[:100] + "...\n")
        
        for path in pathologists:
            result = await path.process_request(test_request)
            print(f"\n{path.name} ({path.APPROACH}):")
            print(f"  Probability: {result['findings']['text_malignancy_probability']:.3f}")
            print(f"  Class: {result['findings']['predicted_class']}")
            print(f"  Size: {result['findings'].get('size_mm', 'N/A')}mm")
            print(f"  Texture: {result['findings'].get('texture', 'N/A')}")
            print(f"  Suspicious terms: {result['findings'].get('suspicious_terms', [])}")
            print(f"  Benign terms: {result['findings'].get('benign_terms', [])}")
    
    asyncio.run(test())
