"""
Specialized Pathologist Agents
==============================

EDUCATIONAL PURPOSE - DIVERSE NLP APPROACHES:

This module implements two pathologist agents with different 
text analysis strategies:

1. PathologistRegex: Pattern matching with regular expressions
2. PathologistSpacy: spaCy NER + semantic rules

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
from models.dynamic_weights import BASE_WEIGHTS, get_base_weight

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
    WEIGHT = 0.8  # Base weight — dynamically scaled per-case by DynamicWeightCalculator
    
    def __init__(self, name: str, asl_file: Optional[str] = None):
        if asl_file is None:
            asl_file = get_asl_path("pathologist")
        super().__init__(name=name, asl_file=asl_file)
    
    def _register_actions(self) -> None:
        """Register internal actions for ASL plans."""
        self.internal_actions["load_nlp_model"] = self._action_load_nlp
        self.internal_actions["extract_all"] = self._action_extract_all
        
        # Individual extractors (for detailed plans)
        self.internal_actions["extract_size"] = lambda t, i="": (self._analyze_report(t).get("size_mm"),)
        self.internal_actions["extract_texture"] = lambda t, i="": (self._analyze_report(t).get("texture", "solid"),)
        self.internal_actions["extract_margin"] = lambda t, i="": ("smooth",) # Placeholder
        self.internal_actions["extract_spiculation"] = lambda t, i="": (1,) # Placeholder
        self.internal_actions["extract_malignancy"] = lambda t, i="": (self._analyze_report(t).get("malignancy_score", 0.5),)
        self.internal_actions["extract_lung_rads"] = lambda t, i="": ("3",)

    # =========================================================================
    # BDI Internal Actions
    # =========================================================================

    def _action_load_nlp(self) -> bool:
        """Internal action: Load NLP."""
        self.add_belief(Belief("nlp_loaded", (True,)))
        self.add_belief(Belief("nlp_model", (self.APPROACH,)))
        return True

    def _action_extract_all(
        self, 
        text: str, 
        nodule_id: str = "unknown"
    ) -> Tuple[float, str, str, int, float]:
        """
        Internal action: Extract all findings.
        Returns: (Size, Texture, Margin, Spiculation, Assessment)
        """
        logger.info(f"[{self.name}] Analyzing text for {nodule_id}")
        findings = self._analyze_report(text)
        
        size = findings.get("size_mm")
        if size is None:
            size = 0.0  # BDI action must return a number; 0.0 signals unknown
        texture = findings.get("texture", "solid")
        margin = "smooth" # Not currently extracted by variants
        spic = 1 # Not currently extracted
        
        # Calculate malignancy assessment
        # Variants return 'malignancy_score' or 'suspicious_terms'
        assessment = findings.get("malignancy_score", 0.5)
        if "probability" in findings: # Some variants might use different keys
             assessment = findings["probability"]
        
        return (size, texture, margin, spic, assessment)

        
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
        
        # Size-based adjustment (skip if size unknown)
        size = findings.get("size_mm")
        if size is not None:
            if size < 6:
                prob -= 0.2
            elif size < 8:
                prob -= 0.1
            elif size < 15:
                prob += 0.1
            else:
                prob += 0.25
        # If size is None, don't adjust — stay at base probability
        
        # Texture adjustment
        texture = findings.get("texture", "").lower()
        if "ground" in texture or "glass" in texture:
            prob -= 0.1
        elif "spicul" in texture:
            prob += 0.2
        
        # Suspicious terms
        suspicious = findings.get("suspicious_terms", [])
        prob += len(suspicious) * 0.1
        
        # Weak suspicious terms (nodule, mass, etc. without qualifiers)
        weak_suspicious = findings.get("weak_suspicious_terms", [])
        prob += len(weak_suspicious) * 0.05
        
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
        size = features.get("size_mm", features.get("diameter_mm"))
        texture = features.get("texture", "solid")
        location = features.get("location", "right upper lobe")
        malignancy = features.get("malignancy", 3)
        
        if size is not None:
            findings = f"A {size:.0f}mm {texture} nodule is identified in the {location}."
        else:
            findings = f"A {texture} nodule is identified in the {location}."
        
        if malignancy >= 1:
            impression = "Suspicious for malignancy. Recommend further evaluation."
        else:
            impression = "Likely benign appearance. Routine follow-up."
        
        return f"FINDINGS: {findings}\n\nIMPRESSION: {impression}"
    
    def _prob_to_class(self, prob: float, threshold: float = 0.5) -> int:
        """Convert probability to binary class (0=benign, 1=malignant)."""
        return 1 if prob >= threshold else 0


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
    WEIGHT = 0.8  # Base weight — dynamically scaled per-case
    
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
    
    # Weak Suspicious term patterns (nodule, mass -> slight prob boost)
    WEAK_SUSPICIOUS_PATTERNS = [
        r"nodule",
        r"mass",
        r"lesion",
        r"opacit(?:y|ies)",
        r"densit(?:y|ies)",
        r"infiltra(?:te|tion)",
        r"consolidat(?:ion|ed)",
        r"atelectas(?:is|ic)",
        r"fibrosis",
        r"thickening"
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
    
    def __init__(self, name: str = "pathologist_regex", asl_file: Optional[str] = None):
        super().__init__(name=name, asl_file=asl_file)
        
    def _analyze_report(self, report: str) -> Dict[str, Any]:
        """Analyze report using regex patterns."""
        report_lower = report.lower()
        
        size_mm, size_source = self._extract_size(report_lower)
        
        findings = {
            "size_mm": size_mm,
            "size_source": size_source,
            "texture": self._extract_texture(report_lower),
            "location": self._extract_location(report_lower),
            "suspicious_terms": self._find_terms(report_lower, self.SUSPICIOUS_PATTERNS),
            "weak_suspicious_terms": self._find_terms(report_lower, self.WEAK_SUSPICIOUS_PATTERNS),
            "benign_terms": self._find_terms(report_lower, self.BENIGN_PATTERNS),
            "approach": "regex"
        }
        
        size_str = f"{findings['size_mm']}mm" if findings['size_mm'] is not None else "unknown"
        logger.info(
            f"[{self.name}] Regex extraction: "
            f"size={size_str} ({size_source}), "
            f"texture={findings['texture']}, "
            f"suspicious={len(findings['suspicious_terms'])}, "
            f"benign={len(findings['benign_terms'])}"
        )
        
        return findings
    
    def _extract_size(self, text: str) -> Tuple[Optional[float], str]:
        """Extract nodule size in mm. Returns (size_mm, size_source)."""
        for pattern in self.SIZE_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                size = float(matches[0])
                # Check if cm (multiply by 10)
                if "cm" in text[text.find(matches[0]):text.find(matches[0])+10]:
                    size *= 10
                return size, "regex"
        return None, "unknown"
    
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
    WEIGHT = 0.9  # Base weight — dynamically scaled per-case
    
    # Medical entity types to look for
    MEDICAL_ENTITIES = [
        "DISEASE", "SYMPTOM", "ANATOMICAL_STRUCTURE",
        "PROCEDURE", "CHEMICAL", "BODY_PART"
    ]
    
    # Custom rules for malignancy assessment
    MALIGNANCY_INDICATORS = {
        "high": ["carcinoma", "malignant", "metastatic", "invasive", "aggressive"],
        "moderate": ["suspicious", "concerning", "indeterminate", "atypical"],
        "weak": ["nodule", "mass", "lesion", "opacity", "density", "infiltration", "consolidation"],
        "low": ["benign", "stable", "unchanged", "granuloma", "resolved", "calcified"]
    }
    
    def __init__(self, name: str = "pathologist_spacy", asl_file: Optional[str] = None):
        super().__init__(name=name, asl_file=asl_file)
        self._nlp = None
        self._nlp_loaded = False
        
    def _load_nlp(self):
        """Load spaCy medical model (required)."""
        if self._nlp is not None:
            return

        import spacy
        self._nlp = spacy.load("en_core_sci_sm")
        self._nlp_loaded = True
        self.add_belief(Belief("nlp_loaded", ("spacy", True)))
        logger.info(f"[{self.name}] Loaded scispaCy medical model")
    
    def _analyze_report(self, report: str) -> Dict[str, Any]:
        """Analyze report using spaCy NER + Dependency Frames (Module 2)."""
        self._load_nlp()
        
        # We need to leverage the full pipeline now, but PathologistSpacy 
        # usually just calls _spacy_analysis. 
        # Since we integrated DependencyFrameExtractor into MedicalNLPExtractor,
        # we can use the frame logic directly if we had a full extractor instance.
        # But here we are inside the agent which has its own _nlp model.
        # So we will replicate the frame extraction call here for the agent.
        
        from nlp.dependency_parser import DependencyFrameExtractor
        frame_extractor = DependencyFrameExtractor(self._nlp)
        
        doc = self._nlp(report)
        frames = frame_extractor.extract(doc)
        
        # 1. Standard spaCy Analysis (Legacy)
        findings = self._spacy_analysis(report, doc)
        
        # 2. Module 2: Structured Frame Integration
        findings["nodule_frames"] = [f.to_dict() for f in frames]
        findings["approach"] = "spacy_ner_v2_dependency"
        
        if frames:
            # INTELLIGENT SELECTION: 
            # If we found structured frames, pick the "Index Nodule" (most significant)
            # to populate the top-level fields, solving the "bag-of-words" ambiguity.
            
            # Sort by size (descending), then by suspicion
            # Define sort key: (has_size, size_mm, is_suspicious)
            def sort_key(f):
                size = f.size_mm if f.size_mm is not None else -1
                suspicious = 1 if (f.texture and "solid" in f.texture) else 0
                return (size, suspicious)
            
            # Get best candidate
            best_frame = sorted(frames, key=sort_key, reverse=True)[0]
            
            # Override with specific linked attributes
            if best_frame.size_mm:
                findings["size_mm"] = best_frame.size_mm
                findings["size_source"] = "dependency_frame"
            
            if best_frame.location:
                findings["location"] = best_frame.location
                
            if best_frame.texture:
                findings["texture"] = best_frame.texture
                
            logger.info(
                f"[{self.name}] Structured Frame Update: "
                f"Selected index nodule (size={best_frame.size_mm}mm, loc={best_frame.location})"
            )
            
        return findings
    
    def _spacy_analysis(self, report: str, doc=None) -> Dict[str, Any]:
        """Analyze using spaCy NLP pipeline."""
        if doc is None:
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
        size_mm, size_source = self._extract_size_spacy(doc)
        
        # Determine texture from context
        texture = self._extract_texture_spacy(doc)
        
        # Analyze sentiment/malignancy indicators
        malignancy_score = self._assess_malignancy_spacy(doc)
        
        # Extract location
        location = self._extract_location_spacy(doc)
        
        findings = {
            "size_mm": size_mm,
            "size_source": size_source,
            "texture": texture,
            "location": location,
            "entities": entities,
            "malignancy_score": malignancy_score,
            "suspicious_terms": self._find_indicator_terms(doc, "high") + 
                               self._find_indicator_terms(doc, "moderate"),
            "weak_suspicious_terms": self._find_indicator_terms(doc, "weak"),
            "benign_terms": self._find_indicator_terms(doc, "low"),
            "approach": "spacy_ner"
        }
        
        size_str = f"{size_mm}mm" if size_mm is not None else "unknown"
        logger.info(
            f"[{self.name}] spaCy analysis: "
            f"size={size_str} ({size_source}), texture={texture}, "
            f"entities={len(entities)}, malignancy_score={malignancy_score:.2f}"
        )
        
        return findings
    
    def _extract_size_spacy(self, doc) -> Tuple[Optional[float], str]:
        """Extract size using spaCy's token analysis. Returns (size_mm, size_source)."""
        import re
        
        # Look for number + unit patterns
        for i, token in enumerate(doc):
            if token.like_num:
                # Check next token for unit
                if i + 1 < len(doc):
                    next_token = doc[i + 1].text.lower()
                    if next_token in ["mm", "millimeter", "millimeters"]:
                        try:
                            return float(token.text), "spacy"
                        except ValueError:
                            pass
                    elif next_token in ["cm", "centimeter", "centimeters"]:
                        try:
                            return float(token.text) * 10, "spacy"
                        except ValueError:
                            pass
        
        # Fallback to regex
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:mm|cm)", doc.text.lower())
        if matches:
            size = float(matches[0])
            if "cm" in doc.text.lower():
                size *= 10
            return size, "regex_fallback"
        
        return None, "unknown"
    
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
        
        # Keep track of matched terms to avoid double counting same term
        matched_terms = set()
        
        for level, terms in self.MALIGNANCY_INDICATORS.items():
            for term in terms:
                if term in text_lower:
                    matched_terms.add(term)
                    count += 1
                    if level == "high":
                        score += 0.3
                    elif level == "moderate":
                        score += 0.1
                    elif level == "weak":
                         score += 0.05
                    else:  # low
                        score -= 0.2
        
        # Normalize
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
    
    
    def _estimate_malignancy(self, findings: Dict[str, Any]) -> float:
        """Override to use spaCy's malignancy score."""
        base_prob = findings.get("malignancy_score", 0.5)
        
        # Adjust based on size (skip if unknown)
        size = findings.get("size_mm")
        if size is not None:
            if size < 6:
                base_prob -= 0.15
            elif size < 8:
                base_prob -= 0.05
            elif size >= 15:
                base_prob += 0.15
        # If size is None, don't adjust
        
        return min(max(base_prob, 0.05), 0.95)


# =============================================================================
# CONTEXT SPECIALIST PATHOLOGIST (Pathologist-3)
# =============================================================================

class PathologistContext(PathologistBase):
    """
    Pathologist-3: Context specialist for negation and uncertainty detection.
    
    EDUCATIONAL PURPOSE - CLINICAL NLP CONTEXT ANALYSIS:
    
    This agent focuses on understanding the CERTAINTY of statements:
    - "No nodule" → NEGATED (entity is explicitly denied)
    - "Possible nodule" → UNCERTAIN (hedged language)
    - "12mm nodule" → AFFIRMED (positively stated)
    
    Uses NegEx-style algorithm (Chapman et al., 2001) with extensions
    for uncertainty detection (Harkema et al., 2009 - ConText).
    
    The certainty labels produced by this agent are critical for the
    Oncologist's conflict resolution, as negated/uncertain findings
    should be weighted differently than affirmed findings.
    
    Approach:
    1. Parse report into sections (FINDINGS, IMPRESSION)
    2. Find medical entity mentions (nodule, mass, etc.)
    3. Apply NegEx trigger/scope rules
    4. Assign certainty labels per entity
    5. Compute overall report certainty
    """
    
    AGENT_TYPE = "pathologist"
    APPROACH = "context"
    WEIGHT = 0.9  # Base weight — dynamically scaled per-case
    
    # Entity patterns to analyze
    ENTITY_PATTERNS = [
        r'\bnodule\b',
        r'\bnodules\b',
        r'\bmass\b',
        r'\bmasses\b',
        r'\blesion\b',
        r'\blesions\b',
        r'\bopacity\b',
        r'\bopacities\b',
        r'\btumor\b',
        r'\bneoplasm\b',
        r'\bcarcinoma\b',
        r'\bmalignancy\b',
        r'\bdensity\b',
        r'\bdensities\b',
        r'\binfiltra(?:te|tion)\b',
        r'\bconsolidat(?:ion|ed)\b',
    ]
    
    def __init__(self, name: str = "pathologist_context", asl_file: Optional[str] = None):
        super().__init__(name=name, asl_file=asl_file)
        self._negex_detector = None
        self._report_parser = None
        self._load_nlp_extensions()
    
    def _load_nlp_extensions(self):
        """Load NLP extension modules (required)."""
        from nlp.negation_detector import NegExDetector
        from nlp.report_parser import ReportParser
        self._negex_detector = NegExDetector()
        self._report_parser = ReportParser()
        logger.info(f"[{self.name}] Loaded NegEx detector and ReportParser")
    
    def _analyze_report(self, report: str) -> Dict[str, Any]:
        """
        Analyze report for negation and uncertainty.
        
        Returns:
            Dict with certainty analysis per entity and overall assessment.
        """
        
        findings = {
            "entity_certainties": [],
            "overall_certainty": "affirmed",
            "negated_count": 0,
            "uncertain_count": 0,
            "affirmed_count": 0,
            "section_analysis": {},
            "approach": "context",
            "suspicious_terms": [],
            "weak_suspicious_terms": []
        }
        
        # Parse into sections
        if self._report_parser:
            parsed = self._report_parser.parse(report)
            for section_name, section_data in parsed.sections.items():
                section_entities = self._analyze_section(section_data.text, section_name)
                findings["section_analysis"][section_name] = {
                    "weight": section_data.weight,
                    "entities": section_entities
                }
                # Add to overall counts
                for ent in section_entities:
                    findings["entity_certainties"].append(ent)
                    if ent["certainty"] == "negated":
                        findings["negated_count"] += 1
                    elif ent["certainty"] == "uncertain":
                        findings["uncertain_count"] += 1
                    else:
                        findings["affirmed_count"] += 1
        else:
            # Fallback: analyze full text
            entities = self._find_entities(report)
            certainties = self._negex_detector.detect(report, entities)
            
            for cert in certainties:
                cert_dict = cert.to_dict()
                findings["entity_certainties"].append(cert_dict)
                if cert_dict["certainty"] == "negated":
                    findings["negated_count"] += 1
                elif cert_dict["certainty"] == "uncertain":
                    findings["uncertain_count"] += 1
                else:
                    findings["affirmed_count"] += 1

        # Map entities to risk terms so base estimator can use them
        high_risk_keywords = ["tumor", "neoplasm", "carcinoma", "malignancy"]
        weak_risk_keywords = ["nodule", "mass", "lesion", "opacity", "density", "infiltra", "consolidat"]
        
        for ent in findings.get("entity_certainties", []):
            if ent["certainty"] == "negated":
                continue
            
            term_lower = ent["text"].lower()
            
            # Check for high risk terms
            for kw in high_risk_keywords:
                if kw in term_lower:
                    findings["suspicious_terms"].append(term_lower)
                    break
            
            # Check for weak risk terms
            for kw in weak_risk_keywords:
                if kw in term_lower:
                    findings["weak_suspicious_terms"].append(term_lower)
                    break
        
        # Determine overall certainty
        findings["overall_certainty"] = self._determine_overall_certainty(findings)
        
        # Extract basic features for malignancy estimation
        size_mm, size_source = self._extract_size(report)
        findings["size_mm"] = size_mm
        findings["size_source"] = size_source
        findings["texture"] = self._extract_texture(report)
        findings["location"] = self._extract_location(report)
        
        logger.info(
            f"[{self.name}] Context analysis: "
            f"affirmed={findings['affirmed_count']}, "
            f"negated={findings['negated_count']}, "
            f"uncertain={findings['uncertain_count']}, "
            f"overall={findings['overall_certainty']}"
        )
        
        return findings
    
    def _analyze_section(self, text: str, section_name: str) -> List[Dict]:
        """Analyze entities within a specific section."""
        entities = self._find_entities(text)
        if not entities:
            return []
        
        certainties = self._negex_detector.detect(text, entities)
        return [
            {**cert.to_dict(), "section": section_name}
            for cert in certainties
        ]
    
    def _find_entities(self, text: str) -> List[tuple]:
        """Find medical entity mentions in text."""
        import re
        entities = []
        for pattern in self.ENTITY_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append((match.group(), match.start(), match.end()))
        return entities
    
    def _determine_overall_certainty(self, findings: Dict) -> str:
        """
        Determine overall report certainty.
        
        Logic:
        - If ANY affirmed nodule mention exists → affirmed
        - If ALL mentions are negated → negated
        - If no affirmed but some uncertain → uncertain
        """
        affirmed = findings["affirmed_count"]
        negated = findings["negated_count"]
        uncertain = findings["uncertain_count"]
        
        if affirmed > 0:
            return "affirmed"
        elif negated > 0 and uncertain == 0:
            return "negated"
        elif uncertain > 0:
            return "uncertain"
        else:
            return "affirmed"  # Default if no entities found
    
    def _extract_size(self, text: str) -> Tuple[Optional[float], str]:
        """Extract nodule size. Returns (size_mm, size_source)."""
        import re
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mm|millimeter)', text, re.I)
        if match:
            return float(match.group(1)), "regex"
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:cm|centimeter)', text, re.I)
        if match:
            return float(match.group(1)) * 10, "regex"
        return None, "unknown"
    
    def _extract_texture(self, text: str) -> str:
        """Extract texture."""
        text_lower = text.lower()
        if "ground" in text_lower and "glass" in text_lower:
            return "ground_glass"
        elif "part" in text_lower and "solid" in text_lower:
            return "part_solid"
        elif "calcif" in text_lower:
            return "calcified"
        return "solid"
    
    def _extract_location(self, text: str) -> str:
        """Extract location."""
        import re
        text_lower = text.lower()
        patterns = {
            "right_upper_lobe": r"right\s+upper|rul",
            "right_middle_lobe": r"right\s+middle|rml",
            "right_lower_lobe": r"right\s+lower|rll",
            "left_upper_lobe": r"left\s+upper|lul",
            "left_lower_lobe": r"left\s+lower|lll",
        }
        for loc, pattern in patterns.items():
            if re.search(pattern, text_lower):
                return loc
        return "unspecified"
    
    
    def _estimate_malignancy(self, findings: Dict[str, Any]) -> float:
        """
        Estimate malignancy with certainty-aware adjustments.
        
        Key insight: Negated/uncertain findings should reduce confidence.
        """
        # Start with base probability based on size/texture
        base_prob = super()._estimate_malignancy(findings)
        
        # Adjust based on certainty
        overall = findings.get("overall_certainty", "affirmed")
        
        if overall == "negated":
            # Report says NO nodule - very low malignancy probability
            return 0.1
        elif overall == "uncertain":
            # Hedged language - reduce confidence, move toward 0.5
            return 0.3 + (base_prob - 0.5) * 0.5
        else:
            return base_prob
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process request with extra certainty information.
        """
        result = await super().process_request(request)
        
        # Add certainty info to response
        findings = result.get("findings", {})
        result["certainty_analysis"] = {
            "overall_certainty": findings.get("overall_certainty", "affirmed"),
            "negated_count": findings.get("negated_count", 0),
            "uncertain_count": findings.get("uncertain_count", 0),
            "affirmed_count": findings.get("affirmed_count", 0),
            "entity_certainties": findings.get("entity_certainties", [])
        }
        
        return result


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_pathologist_regex(name: str = "pathologist_regex"):
    """Create regex-based pathologist agent (Pathologist-1)."""
    return PathologistRegex(name=name)

def create_pathologist_spacy(name: str = "pathologist_spacy"):
    """Create spaCy NER pathologist agent (Pathologist-2)."""
    return PathologistSpacy(name=name)

def create_pathologist_context(name: str = "pathologist_context"):
    """Create context specialist pathologist agent (Pathologist-3)."""
    return PathologistContext(name=name)

def create_all_pathologists():
    """
    Create all three pathologist agents.
    
    Returns:
        List with:
        - Pathologist-1 (Regex): High-precision pattern matching
        - Pathologist-2 (spaCy): Statistical NLP with terminology mapping
        - Pathologist-3 (Context): Negation/uncertainty specialist
    """
    return [
        PathologistRegex(name="pathologist_regex"),
        PathologistSpacy(name="pathologist_spacy"),
        PathologistContext(name="pathologist_context")
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
