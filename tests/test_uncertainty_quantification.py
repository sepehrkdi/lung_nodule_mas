"""
Tests for Uncertainty Quantification Module.

Tests the graded uncertainty scoring that distinguishes:
- Aleatory uncertainty: Inherent text ambiguity
- Epistemic uncertainty: Knowledge gaps from incomplete extraction
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.uncertainty_quantification import (
    UncertaintyQuantification,
    UncertaintyQuantifier,
    CertaintyLabel,
    quantify_uncertainty
)


class TestUncertaintyQuantification(unittest.TestCase):
    """Test basic uncertainty quantification functionality."""
    
    @classmethod
    def setUpClass(cls):
        cls.quantifier = UncertaintyQuantifier()
    
    def test_affirmed_complete_extraction(self):
        """Test: Complete extraction with no hedges -> low uncertainty."""
        result = self.quantifier.quantify_uncertainty(
            text_span="A 5mm solid nodule in the right upper lobe.",
            extracted_attributes={
                "size_mm": 5.0,
                "size_source": "dependency_frame",
                "location": "right upper lobe",
                "texture": "solid",
                "margins": None
            },
            extraction_paths=["direct:nummod", "direct:amod"]
        )
        
        print(f"\n[Test 1] Affirmed Complete: {result.to_dict()}")
        
        self.assertEqual(result.categorical_label, CertaintyLabel.AFFIRMED)
        self.assertLess(result.aleatory_uncertainty, 0.2)
        self.assertLess(result.epistemic_uncertainty, 0.3)  # margins missing
        self.assertGreater(result.certainty_score, 0.7)
    
    def test_high_aleatory_strong_hedges(self):
        """Test: Strong hedge phrases -> high aleatory uncertainty."""
        result = self.quantifier.quantify_uncertainty(
            text_span="Possible nodule, cannot exclude malignancy.",
            extracted_attributes={
                "size_mm": None,
                "size_source": "unknown",
                "location": None,
                "texture": None,
                "margins": None
            },
            extraction_paths=[]
        )
        
        print(f"\n[Test 2] High Aleatory (hedges): {result.to_dict()}")
        
        self.assertEqual(result.categorical_label, CertaintyLabel.UNCERTAIN)
        self.assertGreater(result.aleatory_uncertainty, 0.5)
        self.assertIn("strong_hedges", str(result.contributing_factors))
    
    def test_high_epistemic_missing_attributes(self):
        """Test: Missing critical attributes -> high epistemic uncertainty."""
        result = self.quantifier.quantify_uncertainty(
            text_span="Nodule noted.",  # Sparse text, no hedges
            extracted_attributes={
                "size_mm": None,
                "size_source": "unknown",
                "location": None,
                "texture": None,
                "margins": None
            },
            extraction_paths=[]
        )
        
        print(f"\n[Test 3] High Epistemic (missing attrs): {result.to_dict()}")
        
        # Should have high epistemic but low aleatory (text is clear, just sparse)
        self.assertGreater(result.epistemic_uncertainty, 0.6)
        self.assertLess(result.aleatory_uncertainty, 0.2)
        self.assertIn("missing_attributes", str(result.contributing_factors))
        self.assertIn("missing_size_critical", str(result.contributing_factors))
    
    def test_negation_detection(self):
        """Test: Negation triggers -> NEGATED label with high negation strength."""
        result = self.quantifier.quantify_uncertainty(
            text_span="No evidence of nodule.",
            extracted_attributes={
                "size_mm": None,
                "size_source": "unknown",
                "location": None,
                "texture": None,
                "margins": None
            },
            extraction_paths=[]
        )
        
        print(f"\n[Test 4] Negation: {result.to_dict()}")
        
        self.assertEqual(result.categorical_label, CertaintyLabel.NEGATED)
        self.assertGreater(result.negation_strength, 0.5)
    
    def test_weak_hedges_moderate_uncertainty(self):
        """Test: Weak hedges -> moderate aleatory uncertainty."""
        result = self.quantifier.quantify_uncertainty(
            text_span="Nodule, likely representing granuloma.",
            extracted_attributes={
                "size_mm": 5.0,
                "size_source": "dependency",
                "location": "right lung",
                "texture": None,
                "margins": None
            },
            extraction_paths=["acl:represent"]
        )
        
        print(f"\n[Test 5] Weak Hedges: {result.to_dict()}")
        
        # "likely" and "representing" are weak hedges
        self.assertGreater(result.aleatory_uncertainty, 0.1)
        self.assertLess(result.aleatory_uncertainty, 0.5)
        self.assertIn("weak_hedges", str(result.contributing_factors))
    
    def test_conflicting_evidence(self):
        """Test: Conflicting evidence markers -> increased aleatory."""
        result = self.quantifier.quantify_uncertainty(
            text_span="Nodule, likely benign but cannot exclude malignancy.",
            extracted_attributes={
                "size_mm": 8.0,
                "size_source": "regex",
                "location": "left lower lobe",
                "texture": None,
                "margins": None
            },
            extraction_paths=["direct"]
        )
        
        print(f"\n[Test 6] Conflicting Evidence: {result.to_dict()}")
        
        # "but" is conflict marker, "cannot exclude" is strong hedge
        self.assertGreater(result.aleatory_uncertainty, 0.4)
        self.assertEqual(result.categorical_label, CertaintyLabel.UNCERTAIN)
    
    def test_uncertainty_type_classification(self):
        """Test uncertainty type classification (dominant type)."""
        # High aleatory, low epistemic
        result1 = self.quantifier.quantify_uncertainty(
            text_span="Possibly malignant, cannot exclude metastasis.",
            extracted_attributes={
                "size_mm": 10.0,
                "size_source": "spacy",
                "location": "right lung",
                "texture": "solid",
                "margins": "spiculated"
            },
            extraction_paths=["direct:nummod", "direct:amod", "direct:location"]
        )
        
        # Low aleatory, high epistemic
        result2 = self.quantifier.quantify_uncertainty(
            text_span="Nodule identified.",  # Clear but sparse
            extracted_attributes={
                "size_mm": None,
                "size_source": "unknown",
                "location": None,
                "texture": None,
                "margins": None
            },
            extraction_paths=[]
        )
        
        print(f"\n[Test 7a] Aleatory Dominant: type={result1.uncertainty_type}")
        print(f"[Test 7b] Epistemic Dominant: type={result2.uncertainty_type}")
        
        self.assertEqual(result1.uncertainty_type, "aleatory_dominant")
        self.assertEqual(result2.uncertainty_type, "epistemic_dominant")
    
    def test_total_uncertainty_quadrature(self):
        """Test total uncertainty combines via quadrature."""
        result = self.quantifier.quantify_uncertainty(
            text_span="Possibly a nodule.",
            extracted_attributes={
                "size_mm": None,
                "size_source": "unknown",
                "location": None,
                "texture": None,
                "margins": None
            },
            extraction_paths=[]
        )
        
        print(f"\n[Test 8] Total Uncertainty: {result.total_uncertainty:.3f}")
        print(f"  Aleatory: {result.aleatory_uncertainty:.3f}")
        print(f"  Epistemic: {result.epistemic_uncertainty:.3f}")
        
        # Verify quadrature formula: sqrt(a² + e²), capped at 1.0
        expected = min(1.0, (result.aleatory_uncertainty**2 + result.epistemic_uncertainty**2)**0.5)
        self.assertAlmostEqual(result.total_uncertainty, expected, places=3)


class TestUncertaintyIntegrationWithFrames(unittest.TestCase):
    """Test uncertainty quantification integrated with dependency frame extraction."""
    
    @classmethod
    def setUpClass(cls):
        """Load spaCy model once for all tests."""
        from nlp.extractor import MedicalNLPExtractor
        cls.extractor = MedicalNLPExtractor()
    
    def test_uncertainty_in_extracted_frames(self):
        """Test that extracted frames include uncertainty quantification."""
        text = "A 5mm nodule in the right upper lobe, possibly granuloma."
        
        result = self.extractor.extract(text)
        frames = [f.to_dict() for f in result.extracted_nodules] if result.extracted_nodules else []
        
        print(f"\n[Integration Test 1] Frames: {len(frames)}")
        for frame in frames:
            print(f"  Frame: {frame}")
            if 'uncertainty_quantification' in frame:
                uq = frame['uncertainty_quantification']
                print(f"  -> Certainty: {uq['certainty_score']:.3f}")
                print(f"  -> Aleatory: {uq['aleatory_uncertainty']:.3f}")
                print(f"  -> Epistemic: {uq['epistemic_uncertainty']:.3f}")
                print(f"  -> Factors: {uq['contributing_factors']}")
        
        self.assertGreater(len(frames), 0)
        # Check that at least one frame has uncertainty
        has_uncertainty = any('uncertainty_quantification' in f for f in frames)
        self.assertTrue(has_uncertainty, "Frames should include uncertainty quantification")
    
    def test_complete_extraction_low_epistemic(self):
        """Test that complete extraction yields low epistemic uncertainty."""
        text = "A 10mm solid spiculated nodule in the left lower lobe."
        
        result = self.extractor.extract(text)
        frames = [f.to_dict() for f in result.extracted_nodules] if result.extracted_nodules else []
        
        print(f"\n[Integration Test 2] Complete extraction: {len(frames)} frames")
        
        self.assertGreater(len(frames), 0)
        frame = frames[0]
        
        if 'uncertainty_quantification' in frame:
            uq = frame['uncertainty_quantification']
            print(f"  Epistemic: {uq['epistemic_uncertainty']:.3f}")
            # Complete extraction should have low epistemic uncertainty
            self.assertLess(uq['epistemic_uncertainty'], 0.5)  # Relaxed threshold
    
    def test_hedged_text_high_aleatory(self):
        """Test that hedged text yields high aleatory uncertainty."""
        text = "Possible nodule, may represent infection, cannot exclude malignancy."
        
        result = self.extractor.extract(text)
        frames = [f.to_dict() for f in result.extracted_nodules] if result.extracted_nodules else []
        
        print(f"\n[Integration Test 3] Hedged text: {len(frames)} frames")
        
        self.assertGreater(len(frames), 0)
        frame = frames[0]
        
        if 'uncertainty_quantification' in frame:
            uq = frame['uncertainty_quantification']
            print(f"  Aleatory: {uq['aleatory_uncertainty']:.3f}")
            print(f"  Hedge count: {uq['hedge_count']}")
            # Multiple hedges should yield high aleatory uncertainty
            self.assertGreater(uq['aleatory_uncertainty'], 0.3)
            self.assertGreater(uq['hedge_count'], 0)


class TestModuleLevelFunctions(unittest.TestCase):
    """Test module-level convenience functions."""
    
    def test_quantify_uncertainty_function(self):
        """Test the module-level quantify_uncertainty function."""
        result = quantify_uncertainty(
            text_span="Suspicious nodule, concerning for malignancy.",
            extracted_attributes={"size_mm": 15.0, "location": "right lung"},
            extraction_paths=["direct"]
        )
        
        print(f"\n[Module Function Test] {result.to_dict()}")
        
        self.assertIsInstance(result, UncertaintyQuantification)
        self.assertGreater(result.aleatory_uncertainty, 0.2)  # "suspicious", "concerning" are hedges


if __name__ == '__main__':
    unittest.main(verbosity=2)
