"""
Dependency-Anchored Frame Extractor for Lung Nodule Analysis.

This module implements a grammar-based extraction approach (Module 2) that
links descriptive attributes (size, texture, location) to specific "anchor"
entities (nodule, mass, opacity) using the dependency tree.

This solves the "bag-of-words" problem where attributes in multi-nodule 
reports get mixed up.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import spacy
from spacy.tokens import Token, Span

logger = logging.getLogger(__name__)

@dataclass
class NoduleFinding:
    """
    Structured representation of a single nodule finding.
    
    Attributes are strictly linked to this specific entity instance
    via grammatical dependency parsing.
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
    
    # Context
    is_negated: bool = False
    is_uncertain: bool = False
    is_historical: bool = False  # e.g., "prior nodule"
    
    # Raw spans for verification
    text_span: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "anchor": self.anchor_text,
            "size_mm": self.size_mm,
            "texture": self.texture,
            "location": self.location,
            "margins": self.margins,
            "calcification": self.calcification,
            "negated": self.is_negated,
            "uncertain": self.is_uncertain,
            "span": self.text_span
        }

class DependencyFrameExtractor:
    """
    Extracts structured NoduleFinding objects using dependency parsing trees.
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
        """
        findings = []
        
        # 1. Find Anchors
        for token in doc:
            if token.lemma_.lower() in self.ANCHOR_TERMS:
                finding = self._build_frame(token, doc)
                if finding:
                    findings.append(finding)
                    
        return findings

    def _build_frame(self, anchor: Token, doc) -> Optional[NoduleFinding]:
        """
        Build a findings frame starting from an anchor token.
        Traverses the dependency tree to find modifiers.
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
        
        # Traverse children recursively to fill attributes
        # We use a queue for simple BFS traversal of the subtree
        queue = [anchor]
        visited = {anchor.i}
        
        # Traverse children recursively to fill attributes
        # We use a queue for simple BFS traversal of the subtree
        queue = [anchor]
        visited = {anchor.i}
        
        while queue:
            node = queue.pop(0)
            
            # Analyze node relation to parent
            if node != anchor:
                self._analyze_modifier(node, finding)
                # print(f"DEBUG: Analyzed {node.text} ({node.dep_}) -> Finding: {finding}")
            
            # recursive step
            for child in node.children:
                # CRITICAL FIX: Do not cross sentence boundaries or coordination (conjuncts)
                if child.dep_ in ["conj", "cc"]:
                    continue
                    
                if child.i not in visited:
                    visited.add(child.i)
                    queue.append(child)
                    
        # EXTENSION: context-based fallback
        # If location is missing, check the parent verb structure (common in "There is X in Y" phrases)
        if finding.location is None:
            self._find_contextual_location(anchor, finding)

        # Check parent for negation (sometimes "no" is the head, or attached to head)
        # e.g., "No module is seen" -> 'no' can be det 
        self._check_negation(anchor, finding)
        
        return finding
        
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
                     # Only consider tokens that are NOT already visited/claimed? 
                     # For simplicity, we just check if they provide a location
                     dummy_finding = NoduleFinding(anchor.text, anchor.i)
                     self._analyze_modifier(child, dummy_finding)
                     # Traverse the child's subtree to build the full phrase
                     for desc in child.subtree:
                         self._analyze_modifier(desc, dummy_finding)
                     
                     if dummy_finding.location:
                         finding.location = dummy_finding.location
                         return # Stop after finding one location to avoid mixing
                     dummy_finding = NoduleFinding(anchor.text, anchor.i)
                     self._analyze_modifier(child, dummy_finding)
                     # Traverse the child's subtree to build the full phrase
                     for desc in child.subtree:
                         self._analyze_modifier(desc, dummy_finding)
                     
                     if dummy_finding.location:
                         finding.location = dummy_finding.location
                         # print(f"DEBUG: Found contextual location {finding.location} from verb {verb.text}")
                         return # Stop after finding one location to avoid mixing
                    
        # Check parent for negation (sometimes "no" is the head, or attached to head)
        # e.g., "No module is seen" -> 'no' can be det 
        self._check_negation(anchor, finding)
        
        return finding

    def _analyze_modifier(self, token: Token, finding: NoduleFinding):
        """Analyze a token to see if it modifies the finding."""
        text = token.text.lower()
        lemma = token.lemma_.lower()
        
        # 1. Check for Size (nummod usually)
        # matches "5 mm", "5mm", "1.2 cm"
        if token.like_num or re.match(r'\d+(?:\.\d+)?$', text):
            # Verify unit in children or next token
            if self._is_measurement_chain(token):
                size_val = self._parse_size(token)
                if size_val:
                    finding.size_mm = size_val
                    finding.size_source = "dependency"

        # 1b. Check for Compound Units (e.g. "5 mm mass" where 'mm' is compound of 'mass')
        if text in ['mm', 'cm', 'millimeter', 'centimeter']:
            # Look for the number modifying this unit
            for child in token.children:
                if child.like_num or re.match(r'\d+(?:\.\d+)?$', child.text):
                     val = float(child.text)
                     if 'cm' in text or 'centimeter' in text:
                         val *= 10
                     finding.size_mm = val
                     finding.size_source = "dependency_compound"

        # 2. Check for Texture (amod)
        if lemma in self.TEXTURE_TERMS:
            finding.texture = self.TEXTURE_TERMS[lemma]
        elif "spicul" in text:
            finding.margins = "spiculated"
        
        # 3. Check for Location (prep/pobj chain or compound)
        # This is simplified; robust loc extraction needs full phrase re-assembly
        if lemma in self.LOCATION_TERMS or token.dep_ == "pobj":
            full_loc = self._expand_location_phrase(token)
            if full_loc and self._is_valid_location(full_loc):
                # If we already have a location, append (e.g. "Right Upper Lobe")
                if finding.location:
                    if len(full_loc) > len(finding.location): # Keep the more descriptive one
                        finding.location = full_loc
                else:
                    finding.location = full_loc

        # 4. Check for Negation/Uncertainty terms directly attached
        if lemma in ["no", "not", "without", "ruled out"]:
            finding.is_negated = True
        if lemma in ["possible", "probable", "likely", "suspicious"]:
            finding.is_uncertain = True

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
