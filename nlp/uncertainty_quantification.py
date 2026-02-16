"""
Uncertainty Quantification Module for Clinical NLP

This module implements graded uncertainty quantification that distinguishes between:

1. **Aleatory Uncertainty**: Inherent randomness/ambiguity in the source text itself.
   - Caused by: hedge words, conflicting evidence, ambiguous phrasing
   - Cannot be reduced by gathering more data from the same source
   - Example: "may represent granuloma" has high aleatory uncertainty

2. **Epistemic Uncertainty**: Uncertainty due to incomplete knowledge or extraction.
   - Caused by: missing size, few extraction paths, sparse modifiers
   - Can potentially be reduced by better extraction or additional data
   - Example: "nodule seen" with no size/location has high epistemic uncertainty

The module produces:
- certainty_score: Overall confidence in the extraction [0,1]
- aleatory_uncertainty: Degree of inherent ambiguity [0,1]
- epistemic_uncertainty: Degree of knowledge gaps [0,1]
- categorical_label: Backwards-compatible label (AFFIRMED/NEGATED/UNCERTAIN)

Reference: Der Kiureghian & Ditlevsen (2009) "Aleatory or epistemic? Does it matter?"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import re


class CertaintyLabel(Enum):
    """Categorical certainty labels for backwards compatibility."""
    AFFIRMED = "affirmed"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"


@dataclass
class UncertaintyQuantification:
    """
    Graded uncertainty quantification for a clinical finding.
    
    Attributes:
        certainty_score: Overall confidence in the assertion [0,1].
                        1.0 = highly certain, 0.0 = highly uncertain
        aleatory_uncertainty: Inherent ambiguity in the source text [0,1].
                             High = text is genuinely ambiguous
        epistemic_uncertainty: Knowledge gaps in extraction [0,1].
                              High = insufficient information extracted
        categorical_label: Backwards-compatible label
        contributing_factors: List of factors that influenced the scores
        hedge_count: Number of hedge phrases detected
        negation_strength: Strength of negation signal [0,1]
    """
    certainty_score: float = 1.0
    aleatory_uncertainty: float = 0.0
    epistemic_uncertainty: float = 0.0
    categorical_label: CertaintyLabel = CertaintyLabel.AFFIRMED
    contributing_factors: List[str] = field(default_factory=list)
    hedge_count: int = 0
    negation_strength: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "certainty_score": round(self.certainty_score, 3),
            "aleatory_uncertainty": round(self.aleatory_uncertainty, 3),
            "epistemic_uncertainty": round(self.epistemic_uncertainty, 3),
            "categorical_label": self.categorical_label.value,
            "contributing_factors": self.contributing_factors,
            "hedge_count": self.hedge_count,
            "negation_strength": round(self.negation_strength, 3)
        }
    
    @property
    def total_uncertainty(self) -> float:
        """Combined uncertainty (not simply additive due to different sources)."""
        # Use quadrature combination: sqrt(a² + e²) normalized to [0,1]
        combined = (self.aleatory_uncertainty**2 + self.epistemic_uncertainty**2)**0.5
        return min(1.0, combined)
    
    @property
    def is_high_uncertainty(self) -> bool:
        """Whether total uncertainty exceeds clinical threshold."""
        return self.total_uncertainty > 0.5
    
    @property
    def uncertainty_type(self) -> str:
        """Dominant type of uncertainty."""
        if self.aleatory_uncertainty > self.epistemic_uncertainty + 0.1:
            return "aleatory_dominant"
        elif self.epistemic_uncertainty > self.aleatory_uncertainty + 0.1:
            return "epistemic_dominant"
        else:
            return "mixed"


class UncertaintyQuantifier:
    """
    Computes graded uncertainty scores for clinical findings.
    
    The quantifier analyzes both the source text and the extraction results
    to produce calibrated uncertainty estimates.
    """
    
    # =========================================================================
    # ALEATORY UNCERTAINTY TRIGGERS (inherent text ambiguity)
    # =========================================================================
    
    # Strong hedge phrases - high aleatory uncertainty
    STRONG_HEDGES = {
        "may represent", "could represent", "might represent",
        "possibly", "possible", "probable", "probably",
        "suggestive of", "suspicious for", "concerning for",
        "cannot exclude", "cannot rule out", "differential includes",
        "questionable", "equivocal", "indeterminate"
    }
    
    # Weak hedge phrases - moderate aleatory uncertainty
    WEAK_HEDGES = {
        "likely", "appears", "seems", "consistent with",
        "compatible with", "favored", "presumed", "apparent",
        "suggests", "indicating", "representing"
    }
    
    # Conflicting evidence markers
    CONFLICT_MARKERS = {
        "however", "although", "but", "nevertheless",
        "on the other hand", "alternatively", "versus", "vs",
        "or", "either"
    }
    
    # =========================================================================
    # NEGATION TRIGGERS (affects certainty direction, not uncertainty level)
    # =========================================================================
    
    STRONG_NEGATION = {
        "no", "no evidence of", "without", "negative for",
        "denies", "absence of", "rules out", "excluded",
        "not seen", "not identified", "not demonstrated", "unremarkable"
    }
    
    WEAK_NEGATION = {
        "unlikely", "improbable", "doubtful", "low probability"
    }
    
    # =========================================================================
    # EPISTEMIC UNCERTAINTY FACTORS (knowledge gaps)
    # =========================================================================
    
    # Expected attributes for complete extraction
    EXPECTED_ATTRIBUTES = {"size_mm", "location", "texture", "margins"}
    
    # Minimum extraction paths for low epistemic uncertainty
    MIN_EXTRACTION_PATHS = 2
    
    def __init__(self):
        """Initialize the uncertainty quantifier."""
        # Compile regex patterns for efficiency
        self._strong_hedge_pattern = self._compile_pattern(self.STRONG_HEDGES)
        self._weak_hedge_pattern = self._compile_pattern(self.WEAK_HEDGES)
        self._conflict_pattern = self._compile_pattern(self.CONFLICT_MARKERS)
        self._strong_neg_pattern = self._compile_pattern(self.STRONG_NEGATION)
        self._weak_neg_pattern = self._compile_pattern(self.WEAK_NEGATION)
    
    def _compile_pattern(self, phrases: Set[str]) -> re.Pattern:
        """Compile a set of phrases into a regex pattern."""
        # Sort by length (longest first) to match longer phrases first
        sorted_phrases = sorted(phrases, key=len, reverse=True)
        escaped = [re.escape(p) for p in sorted_phrases]
        pattern = r'\b(' + '|'.join(escaped) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def quantify_uncertainty(
        self,
        text_span: str,
        extracted_attributes: Dict[str, Any],
        extraction_paths: List[str],
        context_window: Optional[str] = None
    ) -> UncertaintyQuantification:
        """
        Compute graded uncertainty for an extracted finding.
        
        Args:
            text_span: The text span from which the finding was extracted
            extracted_attributes: Dict of extracted attributes (size_mm, location, etc.)
            extraction_paths: List of extraction paths used
            context_window: Optional broader context around the span
            
        Returns:
            UncertaintyQuantification with graded scores
        """
        # Use context_window if provided, otherwise use text_span
        analysis_text = context_window if context_window else text_span
        
        # Compute aleatory uncertainty (text ambiguity)
        aleatory, aleatory_factors = self._compute_aleatory_uncertainty(analysis_text)
        
        # Compute epistemic uncertainty (knowledge gaps)
        epistemic, epistemic_factors = self._compute_epistemic_uncertainty(
            extracted_attributes, extraction_paths
        )
        
        # Compute negation strength
        negation_strength, is_negated = self._compute_negation_strength(analysis_text)
        
        # Determine categorical label
        categorical = self._determine_categorical_label(
            aleatory, negation_strength, is_negated
        )
        
        # Compute overall certainty score
        # High uncertainty (either type) reduces certainty
        certainty = self._compute_certainty_score(
            aleatory, epistemic, negation_strength, is_negated
        )
        
        # Count hedges for metadata
        hedge_count = len(self._strong_hedge_pattern.findall(analysis_text))
        hedge_count += len(self._weak_hedge_pattern.findall(analysis_text))
        
        return UncertaintyQuantification(
            certainty_score=certainty,
            aleatory_uncertainty=aleatory,
            epistemic_uncertainty=epistemic,
            categorical_label=categorical,
            contributing_factors=aleatory_factors + epistemic_factors,
            hedge_count=hedge_count,
            negation_strength=negation_strength
        )
    
    def _compute_aleatory_uncertainty(
        self,
        text: str
    ) -> Tuple[float, List[str]]:
        """
        Compute aleatory uncertainty from text ambiguity.
        
        Returns:
            (uncertainty_score, list_of_contributing_factors)
        """
        factors = []
        uncertainty = 0.0
        
        # Strong hedges contribute 0.3 each (capped contribution)
        strong_matches = self._strong_hedge_pattern.findall(text)
        if strong_matches:
            contribution = min(0.6, len(strong_matches) * 0.3)
            uncertainty += contribution
            factors.append(f"strong_hedges:{','.join(set(strong_matches))}")
        
        # Weak hedges contribute 0.15 each (capped contribution)
        weak_matches = self._weak_hedge_pattern.findall(text)
        if weak_matches:
            contribution = min(0.3, len(weak_matches) * 0.15)
            uncertainty += contribution
            factors.append(f"weak_hedges:{','.join(set(weak_matches))}")
        
        # Conflicting evidence markers contribute 0.2 each
        conflict_matches = self._conflict_pattern.findall(text)
        if conflict_matches:
            contribution = min(0.4, len(conflict_matches) * 0.2)
            uncertainty += contribution
            factors.append(f"conflict_markers:{','.join(set(conflict_matches))}")
        
        # Multiple question marks or ellipsis indicate uncertainty
        if text.count('?') > 0:
            uncertainty += 0.15
            factors.append("question_mark")
        
        # Cap at 1.0
        uncertainty = min(1.0, uncertainty)
        
        return uncertainty, factors
    
    def _compute_epistemic_uncertainty(
        self,
        attributes: Dict[str, Any],
        extraction_paths: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Compute epistemic uncertainty from extraction completeness.
        
        Returns:
            (uncertainty_score, list_of_contributing_factors)
        """
        factors = []
        uncertainty = 0.0
        
        # Missing critical attributes increase epistemic uncertainty
        missing_attrs = []
        for attr in self.EXPECTED_ATTRIBUTES:
            value = attributes.get(attr)
            if value is None or value == "":
                missing_attrs.append(attr)
        
        if missing_attrs:
            # Each missing attribute contributes 0.15
            contribution = min(0.6, len(missing_attrs) * 0.15)
            uncertainty += contribution
            factors.append(f"missing_attributes:{','.join(missing_attrs)}")
        
        # Size is particularly important - extra penalty if missing
        if attributes.get("size_mm") is None:
            uncertainty += 0.15
            if "missing_size" not in str(factors):
                factors.append("missing_size_critical")
        
        # Few extraction paths indicate shallow extraction
        if len(extraction_paths) < self.MIN_EXTRACTION_PATHS:
            contribution = 0.2 * (self.MIN_EXTRACTION_PATHS - len(extraction_paths))
            uncertainty += contribution
            factors.append(f"sparse_extraction_paths:{len(extraction_paths)}")
        
        # Unknown size source indicates extraction failure
        size_source = attributes.get("size_source", "")
        if size_source in ["unknown", "none_detected", ""]:
            uncertainty += 0.1
            factors.append(f"size_source_unknown:{size_source}")
        
        # Cap at 1.0
        uncertainty = min(1.0, uncertainty)
        
        return uncertainty, factors
    
    def _compute_negation_strength(
        self,
        text: str
    ) -> Tuple[float, bool]:
        """
        Compute negation strength and direction.
        
        Returns:
            (negation_strength [0,1], is_negated bool)
        """
        strength = 0.0
        
        # Strong negation signals
        strong_matches = self._strong_neg_pattern.findall(text)
        if strong_matches:
            strength = min(1.0, 0.8 + len(strong_matches) * 0.1)
        
        # Weak negation signals (only if no strong negation)
        if strength == 0.0:
            weak_matches = self._weak_neg_pattern.findall(text)
            if weak_matches:
                strength = min(0.6, len(weak_matches) * 0.3)
        
        is_negated = strength > 0.5
        
        return strength, is_negated
    
    def _determine_categorical_label(
        self,
        aleatory: float,
        negation_strength: float,
        is_negated: bool
    ) -> CertaintyLabel:
        """
        Determine categorical label for backwards compatibility.
        
        Priority: NEGATED > UNCERTAIN > AFFIRMED
        """
        if is_negated and negation_strength > 0.5:
            return CertaintyLabel.NEGATED
        elif aleatory > 0.4:
            return CertaintyLabel.UNCERTAIN
        else:
            return CertaintyLabel.AFFIRMED
    
    def _compute_certainty_score(
        self,
        aleatory: float,
        epistemic: float,
        negation_strength: float,
        is_negated: bool
    ) -> float:
        """
        Compute overall certainty score.
        
        Certainty represents confidence in the extracted information,
        regardless of whether it's affirmed or negated.
        """
        # Base certainty starts at 1.0
        certainty = 1.0
        
        # Aleatory uncertainty directly reduces certainty
        certainty -= aleatory * 0.5
        
        # Epistemic uncertainty reduces certainty (but less than aleatory)
        certainty -= epistemic * 0.3
        
        # Strong negation can increase certainty (confident negation)
        # Weak negation decreases certainty
        if is_negated:
            if negation_strength > 0.7:
                # Strong negation = confident in the negation
                certainty = max(certainty, 0.7)
            else:
                # Weak negation = uncertain
                certainty -= (1 - negation_strength) * 0.2
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, certainty))
    
    def quantify_from_nodule_finding(
        self,
        finding: 'NoduleFinding',  # Forward reference
        context_window: Optional[str] = None
    ) -> UncertaintyQuantification:
        """
        Convenience method to quantify uncertainty from a NoduleFinding object.
        
        Args:
            finding: A NoduleFinding dataclass instance
            context_window: Optional broader context
            
        Returns:
            UncertaintyQuantification with graded scores
        """
        # Extract attributes from finding
        attributes = {
            "size_mm": finding.size_mm,
            "size_source": getattr(finding, 'size_source', None),
            "location": finding.location,
            "texture": finding.texture,
            "margins": finding.margins,
        }
        
        extraction_paths = getattr(finding, 'extraction_paths', [])
        text_span = getattr(finding, 'text_span', "")
        
        return self.quantify_uncertainty(
            text_span=text_span,
            extracted_attributes=attributes,
            extraction_paths=extraction_paths,
            context_window=context_window
        )


# Module-level singleton for convenience
_default_quantifier: Optional[UncertaintyQuantifier] = None


def get_uncertainty_quantifier() -> UncertaintyQuantifier:
    """Get or create the default uncertainty quantifier instance."""
    global _default_quantifier
    if _default_quantifier is None:
        _default_quantifier = UncertaintyQuantifier()
    return _default_quantifier


def quantify_uncertainty(
    text_span: str,
    extracted_attributes: Dict[str, Any],
    extraction_paths: List[str],
    context_window: Optional[str] = None
) -> UncertaintyQuantification:
    """
    Module-level convenience function for uncertainty quantification.
    
    See UncertaintyQuantifier.quantify_uncertainty for details.
    """
    return get_uncertainty_quantifier().quantify_uncertainty(
        text_span, extracted_attributes, extraction_paths, context_window
    )
