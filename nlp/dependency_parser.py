"""
Dependency-Anchored Frame Extractor for Lung Nodule Analysis.

This module implements a grammar-based extraction approach (Module 2) that
links descriptive attributes (size, texture, location) to specific "anchor"
entities (nodule, mass, opacity) using the dependency tree.

This solves the "bag-of-words" problem where attributes in multi-nodule 
reports get mixed up.

ENHANCED LONG-DISTANCE DEPENDENCY RESOLUTION:
Handles complex syntactic constructions including:
- Appositive clauses (acl): "nodule representing granuloma"
- Relative clauses (relcl): "nodule which measures 5mm"
- Participial modifiers: "nodule, measuring 5mm"
- Reduced relative clauses: "nodule seen in the RUL"
- Comma-separated participial chains: "nodule, likely representing X, measuring Y"
- Parenthetical modifiers: "nodule (5mm)"
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple, TYPE_CHECKING
import spacy
from spacy.tokens import Token, Span, Doc

# Import uncertainty quantification
from nlp.uncertainty_quantification import (
    UncertaintyQuantification,
    UncertaintyQuantifier,
    CertaintyLabel,
    get_uncertainty_quantifier
)

logger = logging.getLogger(__name__)


# =============================================================================
# LONG-DISTANCE DEPENDENCY CONSTANTS
# =============================================================================

# Dependency relations that introduce clausal modifiers attached to nouns
CLAUSAL_MODIFIER_DEPS = {
    "acl",        # Clausal modifier of noun (e.g., "nodule measuring 5mm")
    "relcl",      # Relative clause modifier (e.g., "nodule which measures 5mm")
    "acl:relcl",  # Combined tag in some models
    "appos",      # Appositive (e.g., "nodule, a granuloma")
    "advcl",      # Adverbial clause (can modify noun in special cases)
}

# Verbs that commonly introduce measurements/descriptions in radiology
MEASUREMENT_VERBS = {
    "measure", "measuring", "measures", "measured",
    "size", "sizing", "sizes", "sized",
    "show", "showing", "shows", "showed",
    "demonstrate", "demonstrating", "demonstrates", "demonstrated",
    "reveal", "revealing", "reveals", "revealed",
    "represent", "representing", "represents", "represented",
    "suggest", "suggesting", "suggests", "suggested",
    "appear", "appearing", "appears", "appeared",
    "contain", "containing", "contains", "contained",
}

# Verbs indicating characterization (texture, margin, etc.)
CHARACTERIZATION_VERBS = {
    "represent", "representing", "represents",
    "consistent", "compatible", "suggestive",
    "likely", "probably", "possibly",  # Adverbs treated as verb-like in some parses
}

@dataclass
class NoduleFinding:
    """
    Structured representation of a single nodule finding.
    
    Attributes are strictly linked to this specific entity instance
    via grammatical dependency parsing, including long-distance relations.
    """
    anchor_text: str           # The word that triggered this frame (e.g., "nodule")
    anchor_idx: int            # Token index of the anchor
    
    # Attributes
    size_mm: Optional[float] = None
    size_source: str = "unknown"
    texture: Optional[str] = None
    location: Optional[str] = None
    margins: Optional[str] = None
    calcification: bool = False
    
    # Enhanced: characterization from clausal modifiers
    characterization: Optional[str] = None  # e.g., "granuloma" from "representing granuloma"
    
    # Context - categorical (backwards compatible)
    is_negated: bool = False
    is_uncertain: bool = False
    is_historical: bool = False  # e.g., "prior nodule"
    
    # Enhanced: Graded uncertainty quantification
    # Distinguishes aleatory (text ambiguity) from epistemic (knowledge gaps)
    uncertainty: Optional[UncertaintyQuantification] = None
    
    # Raw spans for verification
    text_span: str = ""
    
    # Enhanced: track which clausal paths contributed attributes
    extraction_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "anchor": self.anchor_text,
            "size_mm": self.size_mm,
            "size_source": self.size_source,
            "texture": self.texture,
            "location": self.location,
            "margins": self.margins,
            "calcification": self.calcification,
            "characterization": self.characterization,
            "negated": self.is_negated,
            "uncertain": self.is_uncertain,
            "span": self.text_span,
            "extraction_paths": self.extraction_paths
        }
        # Add graded uncertainty if computed
        if self.uncertainty is not None:
            result["uncertainty_quantification"] = self.uncertainty.to_dict()
        return result

class DependencyFrameExtractor:
    """
    Extracts structured NoduleFinding objects using dependency parsing trees.
    
    ENHANCED: Supports long-distance dependency resolution for complex
    syntactic constructions common in radiology reports:
    
    1. Participial chains: "A nodule, likely representing granuloma, measuring 5mm"
    2. Reduced relatives: "nodule measuring 5mm in the RUL"
    3. Appositive clauses: "a 5mm nodule, consistent with granuloma"
    4. Relative clauses: "nodule which measures approximately 5mm"
    """
    
    # Trigger terms that start a frame
    ANCHOR_TERMS = {
        "nodule", "nodules", "mass", "masses", "lesion", "lesions",
        "opacity", "opacities", "density", "densities", 
        "granuloma", "granulomas", "tumor", "neoplasm"
    }
    
    # Terms describing texture/type
    TEXTURE_TERMS = {
        "solid": "solid",
        "ground-glass": "ground_glass", "ground": "ground_glass", "ggo": "ground_glass",
        "part-solid": "part_solid", "subsolid": "part_solid", "mixed": "part_solid",
        "calcified": "calcified", "calcific": "calcified"
    }
    
    # Terms describing location
    LOCATION_TERMS = {
        "upper": "upper", "middle": "middle", "lower": "lower", "low": "lower",
        "right": "right", "left": "left", "bilateral": "bilateral",
        "lobe": "lobe", "lung": "lung", "apex": "apex", "base": "base",
        "hilum": "hilum", "hilus": "hilum",
        "rul": "right_upper_lobe", "lul": "left_upper_lobe",
        "rll": "right_lower_lobe", "lll": "left_lower_lobe",
        "rml": "right_middle_lobe"
    }
    
    # Characterization terms (what the nodule represents)
    # These can be anchor terms but also appear as descriptions of other nodules
    CHARACTERIZATION_TERMS = {
        "granuloma", "granulomas", "hamartoma", "carcinoma", "adenocarcinoma",
        "metastasis", "metastases", "lymphoma", "infection", "tuberculoma",
        "fungal", "malignancy", "benign", "malignant", "inflammatory",
        "fibrosis", "atelectasis", "pneumonia", "abscess"
    }
    
    # Dependency relations where a term is characterizing, not being described
    # These indicate the term is DESCRIBING something else, not standing alone
    CHARACTERIZING_DEP_RELATIONS = {
        "pobj",   # "with granuloma" (prep object)
        "nmod",   # "consistent with granuloma" (noun modifier - scispaCy)
        "dobj",   # "representing granuloma" (direct object)
        "attr",   # "is granuloma" (predicate attribute)
        "appos",  # "nodule, granuloma" (appositive)
        "conj",   # coordinated with a characterization verb
        "obl",    # oblique nominal (Universal Dependencies style)
    }

    def __init__(self, nlp=None):
        """
        Initialize with a spaCy model.
        Args:
            nlp: Loop-loaded spaCy model (e.g., en_core_sci_sm).
                 If None, caller must pass 'doc' found by their own model.
        """
        self.nlp = nlp

    def extract(self, doc) -> List[NoduleFinding]:
        """
        Extract nodule frames from a spaCy Doc object.
        
        ENHANCED: Skips characterization terms that appear in descriptive
        positions (e.g., "consistent with granuloma" - granuloma is not
        a separate finding, it's describing the nodule).
        """
        findings = []
        
        # 1. Find Anchors (skip characterization terms in descriptive positions)
        for token in doc:
            if token.lemma_.lower() in self.ANCHOR_TERMS:
                # Check if this is a characterization term being used descriptively
                if self._is_descriptive_characterization(token):
                    continue
                    
                finding = self._build_frame(token, doc)
                if finding:
                    findings.append(finding)
                    
        return findings
    
    def _is_descriptive_characterization(
        self, 
        token: Token, 
        relative_to: Optional[Token] = None
    ) -> bool:
        """
        Check if an anchor term is being used as a characterization
        of another finding rather than being a primary finding itself.
        
        Args:
            token: The potential characterization term to check
            relative_to: If provided, only return True if token characterizes this anchor
        
        Examples where we should skip creating a frame:
        - "consistent with granuloma" - granuloma is pobj, characterizing
        - "representing granuloma" - granuloma is dobj, characterizing  
        - "nodule, likely granuloma" - granuloma is appositive
        
        Examples where we should create a frame:
        - "A granuloma is seen" - granuloma is the subject, primary finding
        - "Multiple granulomas noted" - granulomas is the primary finding
        """
        lemma = token.lemma_.lower()
        
        # Only check for characterization terms
        if lemma not in self.CHARACTERIZATION_TERMS:
            return False
            
        # Check dependency relation
        if token.dep_ in self.CHARACTERIZING_DEP_RELATIONS:
            # Walk up the tree to see if we hit another ANCHOR_TERM
            head = token.head
            depth = 0
            while head != head.head and depth < 5:
                if head.lemma_.lower() in self.ANCHOR_TERMS:
                    # This characterization term is describing another anchor
                    # If relative_to is specified, check if it's the right anchor
                    if relative_to is not None:
                        return self._is_connected_to_anchor(head, relative_to)
                    return True
                head = head.head
                depth += 1
            
            # Check for common patterns like "with X" where head is prep
            if token.head.lemma_ in ["with", "consistent", "compatible", "represent", "suggest"]:
                if relative_to is not None:
                    # Check if this pattern ultimately connects to relative_to
                    return self._is_connected_to_anchor(token.head, relative_to)
                return True
                
        return False
    
    def _is_connected_to_anchor(self, token: Token, anchor: Token) -> bool:
        """Check if a token is in the dependency chain leading to a specific anchor."""
        current = token
        depth = 0
        while current != current.head and depth < 10:
            if current.i == anchor.i:
                return True
            current = current.head
            depth += 1
        return current.i == anchor.i

    def _build_frame(self, anchor: Token, doc) -> Optional[NoduleFinding]:
        """
        Build a findings frame starting from an anchor token.
        
        ENHANCED: Uses multi-pass traversal to handle:
        1. Direct modifiers (amod, compound, nummod)
        2. Clausal modifiers (acl, relcl) - long-distance deps
        3. Participial chains across commas
        4. Prepositional attachments at clause level
        """
        finding = NoduleFinding(
            anchor_text=anchor.text,
            anchor_idx=anchor.i
        )
        
        # Get the full subtree for context analysis
        subtree = list(anchor.subtree)
        subtree_start = min(t.i for t in subtree)
        subtree_end = max(t.i for t in subtree)
        finding.text_span = doc[subtree_start : subtree_end + 1].text
        
        visited = {anchor.i}
        
        # =================================================================
        # PASS 1: Direct modifiers (standard BFS on immediate subtree)
        # =================================================================
        queue = [anchor]
        while queue:
            node = queue.pop(0)
            
            if node != anchor:
                self._analyze_modifier(node, finding, path="direct")
            
            for child in node.children:
                # Skip conjuncts to avoid cross-contamination
                if child.dep_ in ["conj", "cc"]:
                    continue
                # Skip clausal modifiers in pass 1 (handled in pass 2)
                if child.dep_ in CLAUSAL_MODIFIER_DEPS:
                    continue
                    
                if child.i not in visited:
                    visited.add(child.i)
                    queue.append(child)
        
        # =================================================================
        # PASS 2: Clausal modifiers (long-distance dependencies)
        # Handles: "nodule, likely representing granuloma, measuring 5mm"
        # =================================================================
        self._process_clausal_modifiers(anchor, finding, visited)
        
        # =================================================================
        # PASS 3: Linear scan for comma-separated participial chains
        # Catches modifiers that may not be directly linked in dep tree
        # =================================================================
        self._process_participial_chain(anchor, doc, finding, visited)
        
        # =================================================================
        # PASS 4: Context-based fallbacks
        # =================================================================
        if finding.location is None:
            self._find_contextual_location(anchor, finding)

        self._check_negation(anchor, finding)
        
        # =================================================================
        # PASS 5: Graded Uncertainty Quantification
        # Computes aleatory (text ambiguity) vs epistemic (knowledge gaps)
        # =================================================================
        self._quantify_uncertainty(anchor, finding)
        
        return finding
    
    def _process_clausal_modifiers(
        self, 
        anchor: Token, 
        finding: NoduleFinding,
        visited: Set[int]
    ):
        """
        Process clausal modifiers (acl, relcl, appos) attached to the anchor.
        
        These create long-distance dependencies where attributes are nested
        inside participial or relative clauses.
        
        Example: "A nodule, likely representing granuloma, measuring 5mm"
        - "representing" is acl of "nodule"
        - "measuring" is acl of "nodule" (parallel)
        - "5mm" is inside the "measuring" clause
        """
        for child in anchor.children:
            if child.dep_ not in CLAUSAL_MODIFIER_DEPS:
                continue
                
            # Determine clause type for path tracking
            clause_type = child.dep_
            
            # Check if this is a measurement verb clause
            if child.lemma_.lower() in MEASUREMENT_VERBS:
                finding.extraction_paths.append(f"{clause_type}:{child.lemma_}")
                self._extract_from_measurement_clause(child, finding, visited)
                
            # Check if this is a characterization clause
            elif child.lemma_.lower() in CHARACTERIZATION_VERBS:
                finding.extraction_paths.append(f"{clause_type}:{child.lemma_}")
                self._extract_characterization(child, finding, visited)
                
            # Generic clausal modifier - traverse its subtree
            else:
                finding.extraction_paths.append(f"{clause_type}:{child.text}")
                self._traverse_clause_subtree(child, finding, visited, clause_type)
    
    def _extract_from_measurement_clause(
        self,
        clause_head: Token,
        finding: NoduleFinding,
        visited: Set[int]
    ):
        """
        Extract size/measurements from a measurement verb clause.
        
        Example: "measuring 5mm" or "measures approximately 12 mm"
        The clause head is the verb (measuring/measures).
        """
        for token in clause_head.subtree:
            if token.i in visited:
                continue
            visited.add(token.i)
            
            text = token.text.lower()
            
            # Look for numeric values
            if token.like_num or re.match(r'\d+(?:\.\d+)?$', text):
                size_val = self._parse_size_from_context(token)
                if size_val is not None:
                    finding.size_mm = size_val
                    finding.size_source = f"clausal_{clause_head.lemma_}"
                    
            # Also check for other modifiers in the clause
            self._analyze_modifier(token, finding, path=f"clause:{clause_head.lemma_}")
    
    def _extract_characterization(
        self,
        clause_head: Token,
        finding: NoduleFinding,
        visited: Set[int]
    ):
        """
        Extract characterization from clauses like "representing granuloma".
        
        These clauses describe what the finding is or represents.
        """
        for token in clause_head.subtree:
            if token.i in visited:
                continue
            visited.add(token.i)
            
            lemma = token.lemma_.lower()
            
            # Check for characterization terms
            if lemma in self.CHARACTERIZATION_TERMS:
                finding.characterization = lemma
                finding.extraction_paths.append(f"characterization:{lemma}")
                
            # Check for uncertainty markers
            if lemma in ["likely", "probable", "possible", "suspicious"]:
                finding.is_uncertain = True
                
            # Also extract any measurements/textures in the clause
            self._analyze_modifier(token, finding, path="characterization")
    
    def _traverse_clause_subtree(
        self,
        clause_head: Token,
        finding: NoduleFinding,
        visited: Set[int],
        clause_type: str
    ):
        """
        Generic traversal of a clausal modifier's subtree.
        """
        for token in clause_head.subtree:
            if token.i in visited:
                continue
            visited.add(token.i)
            
            self._analyze_modifier(token, finding, path=clause_type)
    
    def _process_participial_chain(
        self,
        anchor: Token,
        doc,
        finding: NoduleFinding,
        visited: Set[int]
    ):
        """
        Process comma-separated participial chains that may not be
        directly linked in the dependency tree.
        
        ENHANCED: Also captures standalone measurements in appositive structures.
        
        Example: "A nodule, possibly calcified, measuring 5 mm, located in RUL"
        Example: "A nodule, consistent with granuloma, approximately 5 mm in size"
        
        In some parses, these participials may attach to the sentence
        root rather than the anchor. This pass uses linear proximity
        to capture them.
        """
        # Define search window: from anchor to next noun/anchor or sentence end
        start_idx = anchor.i
        end_idx = self._find_chain_boundary(anchor, doc)
        
        # Look for participial verbs and standalone measurements in the window
        i = start_idx + 1
        while i < end_idx:
            token = doc[i]
            
            # Skip if already visited
            if token.i in visited:
                i += 1
                continue
            
            # Check if this looks like a participial modifier
            if self._is_participial_modifier(token, anchor):
                # Process this participial and its dependents
                finding.extraction_paths.append(f"linear_chain:{token.text}")
                
                for subtok in token.subtree:
                    if subtok.i not in visited:
                        visited.add(subtok.i)
                        self._analyze_modifier(subtok, finding, path="participial_chain")
                        
                        # Special handling for measurement verbs
                        if subtok.lemma_.lower() in MEASUREMENT_VERBS:
                            self._extract_from_measurement_clause(subtok, finding, visited)
            
            # ENHANCED: Check for standalone numeric measurements 
            # (e.g., "5 mm" not attached to clause verb)
            elif finding.size_mm is None and self._is_standalone_measurement(token, doc, anchor):
                size_val = self._parse_size_from_context(token)
                if size_val is not None:
                    finding.size_mm = size_val
                    finding.size_source = "linear_measurement"
                    finding.extraction_paths.append(f"linear_measurement:{token.text}")
                    visited.add(token.i)
            
            i += 1
        
        # ENHANCED: Second pass - look for measurements attached to appositive elements
        # that themselves are attached to the anchor
        if finding.size_mm is None:
            self._extract_from_appositive_measurements(anchor, doc, finding, visited)
    
    def _is_standalone_measurement(self, token: Token, doc, anchor: Token) -> bool:
        """
        Check if token is a standalone measurement that belongs to our anchor.
        
        This handles constructions like "A nodule, approximately 5 mm in size"
        where "5" is not directly descended from "nodule" but appears after it.
        """
        text = token.text.lower()
        
        # Must be numeric
        if not (token.like_num or re.match(r'\d+(?:\.\d+)?$', text)):
            return False
            
        # Must have a unit nearby
        if not self._is_measurement_chain(token):
            return False
            
        # Must not belong to a different anchor
        # Check if any ancestor is a different anchor term
        head = token.head
        depth = 0
        while head != head.head and depth < 10:
            if head.lemma_.lower() in self.ANCHOR_TERMS and head.i != anchor.i:
                return False  # Belongs to different anchor
            head = head.head
            depth += 1
            
        return True
    
    def _extract_from_appositive_measurements(
        self,
        anchor: Token,
        doc,
        finding: NoduleFinding,
        visited: Set[int]
    ):
        """
        Extract measurements from appositive structures.
        
        Handles: "A nodule, consistent with granuloma, approximately 5 mm in size"
        Where "5 mm" might attach to "in" -> "size" rather than directly to nodule.
        """
        # Check for measurements in the sentence that aren't claimed by other anchors
        for token in anchor.sent:
            if token.i in visited:
                continue
                
            # Check if this is a numeric token with measurement unit
            text = token.text.lower()
            if token.like_num or re.match(r'\d+(?:\.\d+)?$', text):
                if self._is_measurement_chain(token):
                    # Check it's not claimed by another STANDALONE anchor term
                    is_claimed = False
                    for other_tok in anchor.sent:
                        if other_tok.lemma_.lower() in self.ANCHOR_TERMS and other_tok.i != anchor.i:
                            # Skip characterization terms that are describing our anchor
                            # These don't "claim" measurements independently
                            if self._is_descriptive_characterization(other_tok, anchor):
                                continue
                            # Check if this number is in other anchor's subtree
                            if token in other_tok.subtree:
                                is_claimed = True
                                break
                    
                    if not is_claimed:
                        size_val = self._parse_size_from_context(token)
                        if size_val is not None:
                            finding.size_mm = size_val
                            finding.size_source = "appositive_fallback"
                            finding.extraction_paths.append(f"appositive_fallback:{token.text}")
                            return
    
    def _find_chain_boundary(self, anchor: Token, doc) -> int:
        """
        Find the end of a participial chain.
        Stops at: sentence boundary, another anchor term, or coordinating conjunction.
        """
        sent_end = anchor.sent.end
        
        for i in range(anchor.i + 1, sent_end):
            token = doc[i]
            
            # Stop at another anchor term (new finding) - but not characterization terms
            # that appear IN the anchor's description
            if token.lemma_.lower() in self.ANCHOR_TERMS and token.i != anchor.i:
                # Don't stop if it's a characterization (e.g., "granuloma" describing "nodule")
                if token.lemma_.lower() not in self.CHARACTERIZATION_TERMS:
                    return token.i
                # If it IS a characterization term, check if it's in a "representing X" clause
                if token.dep_ not in ["pobj", "dobj", "attr", "appos"]:
                    return token.i
                
            # Stop at sentence-level coordination
            if token.dep_ == "cc" and token.head.dep_ in ["ROOT", "conj"]:
                return token.i
                
        return sent_end
    
    def _is_participial_modifier(self, token: Token, anchor: Token) -> bool:
        """
        Check if a token looks like a participial modifier for our anchor.
        """
        # Must be a verb or adjective in participial form
        if token.pos_ not in ["VERB", "ADJ"]:
            return False
            
        # Check for -ing or -ed endings (participials)
        text = token.text.lower()
        if not (text.endswith("ing") or text.endswith("ed") or 
                token.tag_ in ["VBG", "VBN", "JJ"]):
            return False
            
        # Should not already be attached to a different anchor
        head = token.head
        while head != head.head:  # Traverse to root
            if head.lemma_.lower() in self.ANCHOR_TERMS and head.i != anchor.i:
                return False  # Belongs to different anchor
            head = head.head
            
        return True
    
    def _parse_size_from_context(self, num_token: Token) -> Optional[float]:
        """
        Parse size from a number token, looking at context for units.
        Enhanced to handle various patterns.
        """
        try:
            val = float(num_token.text)
        except ValueError:
            return None
            
        # Look for unit in multiple places
        doc = num_token.doc
        unit = "mm"  # Default
        
        # 1. Check immediate right neighbor
        if num_token.i + 1 < len(doc):
            next_tok = doc[num_token.i + 1]
            next_text = next_tok.text.lower()
            if "cm" in next_text or "centimeter" in next_text:
                unit = "cm"
            elif "mm" in next_text or "millimeter" in next_text:
                unit = "mm"
                
        # 2. Check children
        for child in num_token.children:
            child_text = child.text.lower()
            if "cm" in child_text:
                unit = "cm"
                break
                
        # 3. Check head
        head_text = num_token.head.text.lower()
        if "cm" in head_text:
            unit = "cm"
            
        if unit == "cm":
            val *= 10
            
        return val
        
    def _find_contextual_location(self, anchor: Token, finding: NoduleFinding):
        """Look up the dependency tree for locations attached to the governing verb."""
        # 1. Traverse up from conjunctions to find the primary noun
        current = anchor
        while current.dep_ == "conj":
            current = current.head
            
        # 2. Check if attached to a verb (e.g. subject, object, or attribute of 'is')
        if current.dep_ in ["nsubj", "nsubjpass", "attr", "dobj"] and current.head.pos_ in ["VERB", "AUX"]:
            verb = current.head
            # Check verb's children for location modifiers
            for child in verb.children:
                if child != current and child.dep_ in ["prep", "nmod"]:
                    # Only consider tokens that are NOT already visited/claimed
                    dummy_finding = NoduleFinding(anchor.text, anchor.i)
                    self._analyze_modifier(child, dummy_finding, path="contextual")
                    # Traverse the child's subtree to build the full phrase
                    for desc in child.subtree:
                        self._analyze_modifier(desc, dummy_finding, path="contextual")
                    
                    if dummy_finding.location:
                        finding.location = dummy_finding.location
                        finding.extraction_paths.append(f"contextual_verb:{verb.lemma_}")
                        return  # Stop after finding one location to avoid mixing

    def _analyze_modifier(self, token: Token, finding: NoduleFinding, path: str = "direct"):
        """
        Analyze a token to see if it modifies the finding.
        
        ENHANCED: Now tracks extraction path for debugging and validation.
        
        Args:
            token: The spaCy token to analyze
            finding: The NoduleFinding to update
            path: The extraction path for tracking (e.g., "direct", "clausal", "participial_chain")
        """
        text = token.text.lower()
        lemma = token.lemma_.lower()
        
        # 1. Check for Size (nummod usually)
        # matches "5 mm", "5mm", "1.2 cm"
        if token.like_num or re.match(r'\d+(?:\.\d+)?$', text):
            # Verify unit in children or next token
            if self._is_measurement_chain(token):
                size_val = self._parse_size(token)
                if size_val:
                    # Only update if we don't have a size yet, or this is from a measurement clause
                    if finding.size_mm is None or path.startswith("clause"):
                        finding.size_mm = size_val
                        finding.size_source = f"dependency_{path}"

        # 1b. Check for Compound Units (e.g. "5 mm mass" where 'mm' is compound of 'mass')
        if text in ['mm', 'cm', 'millimeter', 'centimeter']:
            # Look for the number modifying this unit
            for child in token.children:
                if child.like_num or re.match(r'\d+(?:\.\d+)?$', child.text):
                    try:
                        val = float(child.text)
                        if 'cm' in text or 'centimeter' in text:
                            val *= 10
                        if finding.size_mm is None:
                            finding.size_mm = val
                            finding.size_source = f"dependency_compound_{path}"
                    except ValueError:
                        pass

        # 2. Check for Texture (amod)
        if lemma in self.TEXTURE_TERMS:
            finding.texture = self.TEXTURE_TERMS[lemma]
        elif "spicul" in text:
            finding.margins = "spiculated"
        
        # 3. Check for Location (prep/pobj chain or compound)
        if lemma in self.LOCATION_TERMS or token.dep_ == "pobj":
            full_loc = self._expand_location_phrase(token)
            if full_loc and self._is_valid_location(full_loc):
                # If we already have a location, keep the more descriptive one
                if finding.location:
                    if len(full_loc) > len(finding.location):
                        finding.location = full_loc
                else:
                    finding.location = full_loc

        # 4. Check for Negation/Uncertainty terms directly attached
        if lemma in ["no", "not", "without", "absent"]:
            finding.is_negated = True
        if lemma in ["possible", "probable", "likely", "suspicious", "questionable"]:
            finding.is_uncertain = True
            
        # 5. Check for Characterization terms (for clausal paths)
        if lemma in self.CHARACTERIZATION_TERMS and finding.characterization is None:
            finding.characterization = lemma
            
        # 6. Check for historical indicators
        if lemma in ["prior", "previous", "old", "known", "stable", "unchanged"]:
            finding.is_historical = True

    def _is_measurement_chain(self, token: Token) -> bool:
        """Check if this number is part of a measurement (has 'mm' or 'cm')."""
        # Check immediate children/context for units
        # "5 mm" -> 5 is nummod of mm, or mm follows 5
        # "size 5mm" -> 5mm is one token
        
        # Case A: Unit is a child
        for child in token.children:
            if child.text.lower() in ['mm', 'cm', 'millimeter', 'centimeter']:
                return True
        
        # Case B: Unit is the head (in some parsers: 5 <- mm)
        if token.head.text.lower() in ['mm', 'cm']:
            return True
            
        # Case C: Unit is adjacent (flat structure fallback)
        doc = token.doc
        if token.i + 1 < len(doc):
            if doc[token.i + 1].text.lower() in ['mm', 'cm']:
                return True
                
        return False

    def _parse_size(self, token: Token) -> Optional[float]:
        """Convert number token to mm."""
        try:
            val = float(token.text)
            
            # Determine unit
            unit = "mm"
            # look ahead or at head
            doc = token.doc
            if token.i + 1 < len(doc) and "cm" in doc[token.i+1].text.lower():
                unit = "cm"
            elif token.head.text.lower() in ["cm", "centimeter"]:
                unit = "cm"
                
            if unit == "cm":
                val *= 10
            return val
        except ValueError:
            return None

    def _expand_location_phrase(self, token: Token) -> Optional[str]:
        """
        Reconstruct the full location string from a token (e.g., 'Lobe' -> 'Right Upper Lobe').
        """
        # Gather compounds and adjectives modifiers
        parts = []
        
        # recursive collector for compound/amod children
        def collect_modifiers(t):
            mods = []
            for child in t.children:
                if child.dep_ in ["compound", "amod", "nmod"] and child.lemma_.lower() in self.LOCATION_TERMS:
                     mods.extend(collect_modifiers(child))
                     mods.append(child.text)
            return mods

        # Check if this token is actually a location term
        lemma = token.lemma_.lower()
        if lemma not in self.LOCATION_TERMS:
            text_lower = token.text.lower()
            # fallback to text if lemma failed (e.g. LUL sometimes lemmas to LUL)
            if text_lower in self.LOCATION_TERMS:
               lemma = text_lower
            else:
               return None

        # Collect modifiers coming BEFORE the head (standard English)
        prefix_mods = collect_modifiers(token)
        parts.extend(prefix_mods)
        
        # Use mapped value if it's a full phrase (e.g. acronym expansion)
        mapped = self.LOCATION_TERMS[lemma]
        if "_" in mapped: # e.g. "right_upper_lobe"
            parts.append(mapped.replace("_", " "))
        else:
            parts.append(token.text)
        
        # Join and normalize
        full_loc = " ".join(parts).replace("-", " ") # standardize e.g. "Right-Upper"
        return full_loc
        
    def _is_valid_location(self, text: str) -> bool:
        """Heuristic to check if string looks like an anatomical location."""
        t = text.lower()
        return any(x in t for x in ["lobe", "lung", "hilum", "apex", "base", "right", "left"])

    def _check_negation(self, anchor: Token, finding: NoduleFinding):
        """Check for negation dependencies closer to the anchor."""
        # e.g. "No nodule is seen" -> 'No' is det of 'nodule'
        for child in anchor.children:
            if child.dep_ == "neg" or child.text.lower() in ["no", "not", "without"]:
                 finding.is_negated = True

    def _quantify_uncertainty(self, anchor: Token, finding: NoduleFinding):
        """
        Compute graded uncertainty quantification for the finding.
        
        Distinguishes between:
        - Aleatory uncertainty: Inherent ambiguity in the source text
        - Epistemic uncertainty: Knowledge gaps from incomplete extraction
        
        Updates finding.uncertainty with an UncertaintyQuantification object.
        Also syncs the categorical is_negated/is_uncertain flags.
        """
        quantifier = get_uncertainty_quantifier()
        
        # Get sentence context for broader analysis
        sent_text = anchor.sent.text if anchor.sent else finding.text_span
        
        # Build attributes dict
        attributes = {
            "size_mm": finding.size_mm,
            "size_source": finding.size_source,
            "location": finding.location,
            "texture": finding.texture,
            "margins": finding.margins,
        }
        
        # Quantify uncertainty
        finding.uncertainty = quantifier.quantify_uncertainty(
            text_span=finding.text_span,
            extracted_attributes=attributes,
            extraction_paths=finding.extraction_paths,
            context_window=sent_text
        )
        
        # Sync categorical labels (backwards compatibility)
        # Keep existing negation if already set, otherwise use quantified result
        if not finding.is_negated:
            finding.is_negated = (
                finding.uncertainty.categorical_label == CertaintyLabel.NEGATED
            )
        
        # Update uncertainty flag based on graded score
        # Consider uncertain if aleatory > 0.4 or total uncertainty > 0.5
        if not finding.is_uncertain:
            finding.is_uncertain = (
                finding.uncertainty.categorical_label == CertaintyLabel.UNCERTAIN or
                finding.uncertainty.aleatory_uncertainty > 0.4
            )
