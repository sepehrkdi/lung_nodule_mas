"""
Medical NLP Extractor
=====================

EDUCATIONAL PURPOSE - NATURAL LANGUAGE PROCESSING CONCEPTS:

This module demonstrates fundamental NLP techniques for medical text analysis:

1. TOKENIZATION:
   Breaking text into meaningful units (words, sentences).
   Medical text requires special handling for abbreviations (CT, mm, GGO).

2. PART-OF-SPEECH (POS) TAGGING:
   Identifying word types (noun, verb, adjective).
   Helps find descriptive terms about nodules.

3. NAMED ENTITY RECOGNITION (NER):
   Identifying medical entities (diseases, anatomy, measurements).
   scispaCy provides biomedical-trained NER models.

4. PATTERN MATCHING (REGEX):
   Extracting structured information using regular expressions.
   Essential for measurements, staging, and specific terminology.

5. DEPENDENCY PARSING:
   Understanding grammatical relationships between words.
   Helps connect modifiers to the entities they describe.

NLP PIPELINE:
    Raw Text → Tokenization → POS Tagging → NER → Pattern Extraction → Structured Output

scispaCy MODELS:
- en_core_sci_sm: General biomedical NLP
- en_ner_bc5cdr_md: Disease/Chemical NER
- en_ner_bionlp13cg_md: Cancer genetics NER
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Import new NLP modules
from nlp.report_parser import ReportParser, ParsedReport
from nlp.negation_detector import NegExDetector, Certainty, EntityCertainty
from nlp.dependency_parser import DependencyFrameExtractor, NoduleFinding


@dataclass
class ExtractedEntity:
    """
    Represents an extracted entity from text.
    
    EDUCATIONAL NOTE:
    Named Entity Recognition (NER) identifies spans of text
    that represent specific concepts. Each entity has:
    - Text: The actual words
    - Label: The entity type (DISEASE, MEASUREMENT, etc.)
    - Span: Character positions in original text
    """
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    
    def __str__(self) -> str:
        return f"{self.text} [{self.label}]"


@dataclass
class ExtractionResult:
    """
    Complete extraction result from a pathology report.
    
    Contains all extracted information organized by category,
    plus the raw entities and tokens for further analysis.
    """
    # Extracted measurements
    size_mm: Optional[float] = None
    
    # Morphological features
    texture: Optional[str] = None  # solid, ground-glass, part-solid
    margin: Optional[str] = None   # well-defined, spiculated, etc.
    spiculation: Optional[str] = None
    lobulation: Optional[str] = None
    calcification: Optional[str] = None
    
    # Location
    location: Optional[str] = None  # right upper lobe, etc.
    
    # Clinical assessment
    malignancy_assessment: Optional[str] = None
    lung_rads_category: Optional[str] = None
    recommendation: Optional[str] = None
    
    # Raw extraction data
    entities: List[ExtractedEntity] = field(default_factory=list)
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    
    # NEW: Negation/Uncertainty detection
    certainty: str = "affirmed"  # "affirmed", "negated", "uncertain"
    
    # NEW: Structured Dependency Frames (Module 2)
    extracted_nodules: List[NoduleFinding] = field(default_factory=list)
    
    # NEW: Multiplicity detection
    multiplicity: bool = False
    nodule_count: Optional[int] = None
    
    # NEW: Section-based scores
    section_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "size_mm": self.size_mm,
            "texture": self.texture,
            "margin": self.margin,
            "spiculation": self.spiculation,
            "lobulation": self.lobulation,
            "calcification": self.calcification,
            "location": self.location,
            "malignancy_assessment": self.malignancy_assessment,
            "lung_rads_category": self.lung_rads_category,
            "recommendation": self.recommendation,
            # NEW fields
            "certainty": self.certainty,
            "multiplicity": self.multiplicity,
            "nodule_count": self.nodule_count,
            "section_scores": self.section_scores,
            # Entities and measurements
            "entities": [{"text": e.text, "label": e.label} for e in self.entities],
            "measurements": self.measurements
        }


class MedicalNLPExtractor:
    """
    NLP-based extractor for medical radiology reports.
    
    EDUCATIONAL PURPOSE - NLP PIPELINE DEMONSTRATION:
    
    This class implements a complete NLP pipeline:
    
    1. PREPROCESSING:
       - Lowercasing (optional, preserves acronyms)
       - Sentence segmentation
       
    2. TOKENIZATION:
       - spaCy's rule-based tokenizer
       - Handles medical abbreviations
       
    3. POS TAGGING:
       - Identifies nouns (nodule, mass), adjectives (solid, spiculated)
       
    4. NER:
       - scispaCy's trained models for biomedical entities
       - Custom patterns for radiology-specific terms
       
    5. PATTERN MATCHING:
       - Regex for measurements (15 mm, 1.5 cm)
       - Templates for Lung-RADS categories
       - Keywords for clinical impressions
       
    Usage:
        extractor = MedicalNLPExtractor()
        result = extractor.extract(report_text)
    """
    
    # Regex patterns for extraction
    # EDUCATIONAL NOTE: Regex is a powerful tool for structured extraction
    
    # Size patterns: "15 mm", "1.5 cm", "15mm", etc.
    SIZE_PATTERNS = [
        r'(\d+\.?\d*)\s*(mm|millimeter)',
        r'(\d+\.?\d*)\s*(cm|centimeter)',
        r'(\d+\.?\d*)\s*x\s*\d+\.?\d*\s*(mm|cm)',  # dimensions
    ]
    
    # Location patterns
    LOCATION_PATTERNS = [
        r'(right|left)\s+(upper|middle|lower)\s+lobe',
        r'(right|left)\s+lung',
        r'(upper|middle|lower)\s+lobe',
        r'lingula',
        r'lung\s+apex',
        r'lung\s+base',
        r'perihilar',
        r'subpleural',
    ]
    
    # Texture keywords
    TEXTURE_KEYWORDS = {
        'solid': ['solid', 'dense', 'high attenuation'],
        'ground_glass': ['ground-glass', 'ground glass', 'ggo', 'hazy', 'non-solid'],
        'part_solid': ['part-solid', 'part solid', 'subsolid', 'mixed'],
    }
    
    # Margin keywords
    MARGIN_KEYWORDS = {
        'well_defined': ['well-defined', 'well defined', 'sharp', 'smooth', 'circumscribed'],
        'poorly_defined': ['poorly-defined', 'poorly defined', 'indistinct', 'ill-defined'],
        'spiculated': ['spiculated', 'spiculation', 'corona radiata', 'stellate'],
        'lobulated': ['lobulated', 'lobulation', 'scalloped'],
    }
    
    # Calcification patterns
    CALCIFICATION_KEYWORDS = {
        'popcorn': ['popcorn', 'popcorn calcification'],
        'laminated': ['laminated', 'laminated calcification'],
        'central': ['central calcification', 'centrally calcified'],
        'eccentric': ['eccentric', 'non-central', 'peripheral calcification'],
        'absent': ['no calcification', 'without calcification', 'non-calcified'],
    }
    
    # Malignancy assessment keywords
    MALIGNANCY_KEYWORDS = {
        'highly_suspicious': ['highly suspicious', 'highly concerning', 'strongly suggests malignancy'],
        'moderately_suspicious': ['moderately suspicious', 'concerning', 'suspicious'],
        'indeterminate': ['indeterminate', 'uncertain', 'cannot exclude'],
        'probably_benign': ['probably benign', 'likely benign', 'low suspicion'],
        'benign': ['benign', 'highly unlikely', 'no concern'],
    }
    
    # Lung-RADS patterns
    LUNG_RADS_PATTERN = r'lung[-\s]?rads\s*(?:category)?\s*:?\s*(\d[a-z]?)'
    
    # NEW: Multiplicity patterns
    MULTIPLICITY_PATTERNS = [
        r'multiple\s+nodules?',
        r'several\s+nodules?',
        r'bilateral\s+nodules?',
        r'numerous\s+nodules?',
        r'diffuse\s+nodules?',
        r'\d+\s+nodules',  # "3 nodules"
        r'nodules\s+are\s+present',
        r'scattered\s+nodules?',
    ]
    
    # NEW: Medical abbreviation dictionary based on RadLex (Langlotz, 2006)
    RADLEX_COMMON_TERMS = {
        'RUL': 'right upper lobe',
        'RML': 'right middle lobe',
        'RLL': 'right lower lobe',
        'LUL': 'left upper lobe',
        'LLL': 'left lower lobe',
        'GGO': 'ground-glass opacity',
        'GGN': 'ground-glass nodule',
        'CT': 'computed tomography',
        'CXR': 'chest x-ray',
        'PET': 'positron emission tomography',
        'LDCT': 'low-dose computed tomography',
        'SPN': 'solitary pulmonary nodule',
        'LAD': 'lymphadenopathy',
        'PA': 'posteroanterior',
        'AP': 'anteroposterior',
        'LN': 'lymph node',
    }
    
    def __init__(self, use_scispacy: bool = True):
        """
        Initialize the NLP extractor.
        
        Args:
            use_scispacy: If True, try to load scispaCy model
        """
        self.nlp = None
        self.use_scispacy = use_scispacy
        
        # Initialize section parser and negation detector
        self.report_parser = ReportParser()
        self.negation_detector = NegExDetector()
        
        if use_scispacy:
            self._load_spacy_model()
    
    def _load_spacy_model(self) -> None:
        """
        Load spaCy/scispaCy model for NLP processing (required).
        
        EDUCATIONAL NOTE:
        scispaCy provides biomedical-trained models:
        - en_core_sci_sm: Small, efficient, general biomedical
        - en_ner_bc5cdr_md: Trained on BC5CDR corpus (diseases/chemicals)
        """
        import spacy
        self.nlp = spacy.load('en_core_sci_sm')
        self.negation_detector = NegExDetector()
        self.report_parser = ReportParser()
        self.dependency_extractor = DependencyFrameExtractor(self.nlp)
        print(f"[NLPExtractor] Loaded spaCy model: en_core_sci_sm")
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extract structured information from radiology report text.
        
        EDUCATIONAL NOTE - NLP PIPELINE:
        This method demonstrates the complete NLP pipeline:
        1. Text normalization
        2. SpaCy processing (tokenization, POS, NER)
        3. Pattern matching (regex)
        4. Feature aggregation
        
        Args:
            text: Raw pathology report text
            
        Returns:
            ExtractionResult with all extracted information
        """
        result = ExtractionResult()
        
        # Normalize text (preserve case for NER, use lowercase for patterns)
        text_lower = text.lower()
        
        # 1. Extract using spaCy if available
        if self.nlp:
            result = self._extract_with_spacy(text, result)
        
        # 2. Extract size measurements
        result = self._extract_size(text_lower, result)
        
        # 3. Extract location
        result = self._extract_location(text_lower, result)
        
        # 4. Extract texture
        result = self._extract_texture(text_lower, result)
        
        # 5. Extract margin characteristics
        result = self._extract_margins(text_lower, result)
        
        # 6. Extract calcification
        result = self._extract_calcification(text_lower, result)
        
        # 7. Extract malignancy assessment
        result = self._extract_malignancy(text_lower, result)
        
        # 8. Extract Lung-RADS category
        result = self._extract_lung_rads(text_lower, result)
        
        # 9. NEW: Parse sections and compute section scores
        result = self._extract_sections(text, result)
        
        # 10. NEW: Detect multiplicity
        result = self._extract_multiplicity(text_lower, result)
        
        # 11. NEW: Detect negation/uncertainty for main nodule mention
        result = self._detect_certainty(text, result)
        
        return result
    
        return result
        
    def _extract_with_spacy(self, text: str, result: ExtractionResult) -> ExtractionResult:
        """
        Use spaCy for tokenization, POS tagging, NER, and Dependency Parsing (Module 2).
        """
        doc = self.nlp(text)
        
        # Extract named entities (Standard NER)
        for ent in doc.ents:
            result.entities.append(ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char
            ))
            
        # EDUCATIONAL: Module 2 - Dependency-Anchored Frame Building
        # Extract structured nodule findings using grammatical dependencies
        try:
            frames = self.dependency_extractor.extract(doc)
            result.extracted_nodules = frames
            if frames:
                # Log for demonstration
                print(f"[NLPExtractor] Found {len(frames)} structured nodule frames")
        except Exception as e:
            print(f"[NLPExtractor] Dependency extraction error: {e}")
        
        return result
    
    def _extract_size(self, text: str, result: ExtractionResult) -> ExtractionResult:
        """
        Extract nodule size from text.
        
        EDUCATIONAL NOTE - REGEX PATTERNS:
        We use multiple patterns to catch different formats:
        - "15 mm" or "15mm"
        - "1.5 cm" (convert to mm)
        - "15 x 12 mm" (dimensions)
        """
        for pattern in self.SIZE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                unit = match.group(2).lower()
                
                # Convert cm to mm
                if 'cm' in unit or 'centimeter' in unit:
                    value *= 10
                
                result.size_mm = value
                result.measurements.append({
                    "value": value,
                    "unit": "mm",
                    "raw": match.group(0)
                })
                break
        
        return result
    
    def _extract_location(self, text: str, result: ExtractionResult) -> ExtractionResult:
        """Extract anatomical location."""
        for pattern in self.LOCATION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result.location = match.group(0).strip()
                break
        
        return result
    
    def _extract_texture(self, text: str, result: ExtractionResult) -> ExtractionResult:
        """
        Extract nodule texture (solid, ground-glass, part-solid).
        
        EDUCATIONAL NOTE - KEYWORD MATCHING:
        Medical terminology often has multiple ways to express
        the same concept. We group synonyms together.
        """
        for texture_type, keywords in self.TEXTURE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    result.texture = texture_type
                    return result
        
        return result
    
    def _extract_margins(self, text: str, result: ExtractionResult) -> ExtractionResult:
        """Extract margin characteristics (well-defined, spiculated, etc.)."""
        for margin_type, keywords in self.MARGIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    if margin_type == 'spiculated':
                        result.spiculation = 'present'
                        result.margin = 'spiculated'
                    elif margin_type == 'lobulated':
                        result.lobulation = 'present'
                        if not result.margin:
                            result.margin = 'lobulated'
                    else:
                        result.margin = margin_type
                    return result
        
        return result
    
    def _extract_calcification(self, text: str, result: ExtractionResult) -> ExtractionResult:
        """Extract calcification pattern."""
        for calc_type, keywords in self.CALCIFICATION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    result.calcification = calc_type
                    return result
        
        return result
    
    def _extract_malignancy(self, text: str, result: ExtractionResult) -> ExtractionResult:
        """
        Extract malignancy assessment from impression.
        
        EDUCATIONAL NOTE:
        Clinical impressions use specific terminology that maps
        to malignancy likelihood. We use keyword matching to
        classify the overall assessment.
        """
        for assessment, keywords in self.MALIGNANCY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    result.malignancy_assessment = assessment
                    return result
        
        return result
    
    def _extract_lung_rads(self, text: str, result: ExtractionResult) -> ExtractionResult:
        """Extract Lung-RADS category if mentioned."""
        match = re.search(self.LUNG_RADS_PATTERN, text, re.IGNORECASE)
        if match:
            result.lung_rads_category = match.group(1).upper()
        
        return result
    
    # =========================================================================
    # NEW METHODS: Section parsing, Multiplicity, Negation/Uncertainty
    # =========================================================================
    
    def _extract_sections(self, text: str, result: ExtractionResult) -> ExtractionResult:
        """
        Parse report into sections and compute section-weighted scores.
        
        EDUCATIONAL NOTE:
        Radiology reports have distinct sections (FINDINGS, IMPRESSION).
        The IMPRESSION carries more diagnostic weight as it represents
        the radiologist's synthesis of observations.
        """
        if not self.report_parser:
            return result
        
        parsed = self.report_parser.parse(text)
        
        # Store section-based scoring info
        for section_name, section_data in parsed.sections.items():
            result.section_scores[section_name] = section_data.weight
        
        return result
    
    def _extract_multiplicity(self, text: str, result: ExtractionResult) -> ExtractionResult:
        """
        Detect if multiple nodules are mentioned.
        
        EDUCATIONAL NOTE:
        Multiplicity is clinically significant:
        - "Multiple nodules" may indicate metastatic disease
        - "Bilateral nodules" affects staging and treatment
        """
        for pattern in self.MULTIPLICITY_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result.multiplicity = True
                
                # Try to extract count if mentioned
                count_match = re.search(r'(\d+)\s+nodules?', text, re.IGNORECASE)
                if count_match:
                    result.nodule_count = int(count_match.group(1))
                
                return result
        
        return result
    
    def _detect_certainty(self, text: str, result: ExtractionResult) -> ExtractionResult:
        """
        Detect negation and uncertainty for nodule mentions.
        
        EDUCATIONAL NOTE:
        This uses NegEx-style detection (Chapman et al., 2001):
        - "No nodule" → NEGATED
        - "Possible nodule" → UNCERTAIN
        - "12mm nodule" → AFFIRMED
        
        Critical for accurate information extraction!
        """
        if not self.negation_detector:
            return result
        
        # Find nodule mentions in text
        nodule_patterns = [
            r'\bnodule\b',
            r'\bnodules\b',
            r'\bmass\b',
            r'\blesion\b',
            r'\bopacity\b',
        ]
        
        entities = []
        for pattern in nodule_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append((match.group(), match.start(), match.end()))
        
        if not entities:
            return result
        
        # Detect certainty for each entity
        certainty_results = self.negation_detector.detect(text, entities)
        
        # Determine overall certainty (take most uncertain/negative)
        has_negated = any(r.certainty.value == "negated" for r in certainty_results)
        has_uncertain = any(r.certainty.value == "uncertain" for r in certainty_results)
        has_affirmed = any(r.certainty.value == "affirmed" for r in certainty_results)
        
        # Priority: if any affirmed mention exists, report is affirmed
        # unless ALL mentions are negated
        if has_negated and not has_affirmed:
            result.certainty = "negated"
        elif has_uncertain and not has_affirmed:
            result.certainty = "uncertain"
        else:
            result.certainty = "affirmed"
        
        return result
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand medical abbreviations in text.
        
        EDUCATIONAL NOTE:
        Radiology reports use many abbreviations:
        - RUL = Right Upper Lobe
        - GGO = Ground-Glass Opacity
        Expansion improves downstream NLP processing.
        """
        expanded = text
        for abbrev, full in self.RADLEX_COMMON_TERMS.items():
            # Match abbreviation as whole word
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            expanded = re.sub(pattern, f"{abbrev} ({full})", expanded)
        return expanded
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        EDUCATIONAL NOTE:
        Tokenization is the first step in NLP. For medical text,
        we need to handle:
        - Abbreviations (CT, mm, GGO)
        - Hyphenated terms (well-defined, ground-glass)
        - Numbers with units (15mm)
        """
        doc = self.nlp(text)
        return [token.text for token in doc]
    
    def get_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]
    
    def get_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text.
        
        EDUCATIONAL NOTE:
        Noun phrases often contain the key medical concepts:
        "solid nodule", "right upper lobe", "marked spiculation"
        """
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]
    
    def analyze_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze text structure for educational purposes.
        
        Returns detailed information about tokens, POS tags,
        entities, and dependencies.
        """
        if not self.nlp:
            return {"error": "spaCy not loaded"}
        
        doc = self.nlp(text)
        
        return {
            "tokens": [
                {
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "dep": token.dep_,
                    "head": token.head.text
                }
                for token in doc
            ],
            "entities": [
                {"text": ent.text, "label": ent.label_}
                for ent in doc.ents
            ],
            "noun_chunks": [chunk.text for chunk in doc.noun_chunks],
            "sentences": [sent.text for sent in doc.sents]
        }


# Convenience function
def extract_from_report(text: str) -> Dict[str, Any]:
    """
    Extract structured information from a radiology report.
    
    Args:
        text: Raw report text
        
    Returns:
        Dictionary with extracted features
    """
    extractor = MedicalNLPExtractor()
    result = extractor.extract(text)
    return result.to_dict()


if __name__ == "__main__":
    # Demo usage
    print("=== Medical NLP Extractor Demo ===\n")
    
    # Sample report
    sample_report = """
    CHEST CT - PULMONARY NODULE EVALUATION
    
    FINDINGS:
    A 15.2 mm solid pulmonary nodule is identified in the right upper lobe.
    The nodule demonstrates marked spiculation with poorly defined margins.
    No internal calcification is identified.
    
    IMPRESSION:
    Large solid nodule in the right upper lobe. Features are highly suspicious
    for malignancy. Estimated Lung-RADS Category: 4B.
    
    RECOMMENDATION:
    PET-CT strongly recommended. Consider CT-guided biopsy.
    """
    
    extractor = MedicalNLPExtractor()
    result = extractor.extract(sample_report)
    
    print("Extracted Information:")
    print(f"  Size: {result.size_mm} mm")
    print(f"  Location: {result.location}")
    print(f"  Texture: {result.texture}")
    print(f"  Margin: {result.margin}")
    print(f"  Spiculation: {result.spiculation}")
    print(f"  Calcification: {result.calcification}")
    print(f"  Malignancy: {result.malignancy_assessment}")
    print(f"  Lung-RADS: {result.lung_rads_category}")
    
    print("\n--- Entities ---")
    for entity in result.entities:
        print(f"  {entity}")
    
    print("\n--- Tokenization Demo ---")
    tokens = extractor.tokenize("A 15.2mm solid nodule with spiculated margins.")
    print(f"  Tokens: {tokens[:10]}...")
    
    print("\n--- Noun Phrases ---")
    phrases = extractor.get_noun_phrases(sample_report)
    print(f"  {phrases[:5]}...")
