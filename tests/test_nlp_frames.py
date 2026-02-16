import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from nlp.extractor import MedicalNLPExtractor
from agents.pathologist_variants import PathologistSpacy

class TestNLPFrames(unittest.TestCase):
    def setUp(self):
        self.extractor = MedicalNLPExtractor()
        self.agent = PathologistSpacy("test_pathologist")

    def test_single_nodule_frame(self):
        """Test simple single nodule extraction."""
        text = "A 5mm spiculated nodule is seen in the right upper lobe."
        result = self.extractor.extract(text)
        
        self.assertTrue(len(result.extracted_nodules) > 0)
        nodule = result.extracted_nodules[0]
        
        print(f"\n[Test 1] Single Nodule: {nodule.to_dict()}")
        self.assertEqual(nodule.size_mm, 5.0)
        self.assertEqual(nodule.texture, None) # "spiculated" is margin/texture, depending on term list
        self.assertEqual(nodule.margins, "spiculated")
        self.assertTrue("right upper" in nodule.location.lower())

    def test_multi_nodule_separation(self):
        """Test separation of attributes for two nodules."""
        text = "There is a 5mm nodule in the right upper lobe and a 12mm mass in the left lower lobe."
        result = self.extractor.extract(text)
        
        print(f"\n[Test 2] Multi Nodule Raw: {[n.to_dict() for n in result.extracted_nodules]}")
        
        self.assertEqual(len(result.extracted_nodules), 2)
        
        # Sort by size to identify
        nodules = sorted(result.extracted_nodules, key=lambda x: x.size_mm or 0)
        n1 = nodules[0] # 5mm
        n2 = nodules[1] # 12mm
        
        self.assertEqual(n1.size_mm, 5.0)
        self.assertTrue("right upper" in n1.location.lower())
        
        self.assertEqual(n2.size_mm, 12.0)
        self.assertTrue("left lower" in n2.location.lower())

    def test_agent_integration(self):
        """Test that PathologistSpacy selects the correct Index Nodule."""
        # Case: Small benign nodule vs Large suspicious mass
        text = "A tiny 2mm calcified nodule in the RLL. A large 25mm solid mass in the LUL."
        
        findings = self.agent._analyze_report(text)
        
        print(f"\n[Test 3] Agent Integration Findings: {findings}")
        
        # Should pick the 25mm mass as the "Index Nodule" for top-level stats
        self.assertEqual(findings["size_mm"], 25.0)
        self.assertEqual(findings["texture"], "solid")
        self.assertTrue("left upper" in findings["location"].lower())
        
        # But ensure frames are captured
        self.assertEqual(len(findings["nodule_frames"]), 2)


class TestLongDistanceDependencies(unittest.TestCase):
    """Test enhanced long-distance dependency resolution."""
    
    def setUp(self):
        self.extractor = MedicalNLPExtractor()
    
    def test_participial_chain_measuring(self):
        """Test: 'A nodule, likely representing granuloma, measuring 5mm'"""
        text = "A nodule, likely representing granuloma, measuring 5mm in the right upper lobe."
        result = self.extractor.extract(text)
        
        print(f"\n[Test 4] Participial Chain: {[n.to_dict() for n in result.extracted_nodules]}")
        
        self.assertTrue(len(result.extracted_nodules) >= 1)
        nodule = result.extracted_nodules[0]
        
        # Should extract size from the "measuring 5mm" clause
        self.assertEqual(nodule.size_mm, 5.0)
        # Should extract characterization from "representing granuloma"
        self.assertEqual(nodule.characterization, "granuloma")
        # Should mark as uncertain due to "likely"
        self.assertTrue(nodule.is_uncertain)
        # Should extract location
        self.assertIsNotNone(nodule.location)
        
    def test_relative_clause_which_measures(self):
        """Test: 'a nodule which measures approximately 12 mm'"""
        text = "There is a solid nodule which measures approximately 12 mm in the left lower lobe."
        result = self.extractor.extract(text)
        
        print(f"\n[Test 5] Relative Clause: {[n.to_dict() for n in result.extracted_nodules]}")
        
        self.assertTrue(len(result.extracted_nodules) >= 1)
        nodule = result.extracted_nodules[0]
        
        # Should extract size from relative clause
        self.assertEqual(nodule.size_mm, 12.0)
        # Should extract texture
        self.assertEqual(nodule.texture, "solid")
        
    def test_reduced_relative_clause(self):
        """Test: 'nodule seen in the RUL measuring 8mm'"""
        text = "A nodule seen in the right upper lobe measuring 8mm."
        result = self.extractor.extract(text)
        
        print(f"\n[Test 6] Reduced Relative: {[n.to_dict() for n in result.extracted_nodules]}")
        
        self.assertTrue(len(result.extracted_nodules) >= 1)
        nodule = result.extracted_nodules[0]
        
        # Should extract size
        self.assertEqual(nodule.size_mm, 8.0)
        # Should extract location
        self.assertTrue("right upper" in nodule.location.lower())
        
    def test_appositive_with_size(self):
        """Test: 'a nodule, consistent with granuloma, 5mm'"""
        text = "A nodule, consistent with granuloma, approximately 5 mm in size."
        result = self.extractor.extract(text)
        
        print(f"\n[Test 7] Appositive: {[n.to_dict() for n in result.extracted_nodules]}")
        
        self.assertTrue(len(result.extracted_nodules) >= 1)
        nodule = result.extracted_nodules[0]
        
        # Should extract characterization
        self.assertEqual(nodule.characterization, "granuloma")
        # Should extract size
        self.assertEqual(nodule.size_mm, 5.0)
        
    def test_complex_chain_multiple_modifiers(self):
        """Test complex construction with multiple participial modifiers."""
        text = "A part-solid nodule, possibly calcified, measuring 15 mm, located in the left upper lobe, suspicious for malignancy."
        result = self.extractor.extract(text)
        
        print(f"\n[Test 8] Complex Chain: {[n.to_dict() for n in result.extracted_nodules]}")
        
        self.assertTrue(len(result.extracted_nodules) >= 1)
        nodule = result.extracted_nodules[0]
        
        # Should extract all attributes
        self.assertEqual(nodule.texture, "part_solid")
        self.assertEqual(nodule.size_mm, 15.0)
        self.assertTrue("left upper" in nodule.location.lower())
        self.assertTrue(nodule.is_uncertain)  # "possibly", "suspicious"
        
    def test_centimeter_conversion_in_clause(self):
        """Test cm to mm conversion within clausal modifier."""
        text = "A nodule measuring 1.5 cm in the right lung."
        result = self.extractor.extract(text)
        
        print(f"\n[Test 9] CM Conversion in Clause: {[n.to_dict() for n in result.extracted_nodules]}")
        
        self.assertTrue(len(result.extracted_nodules) >= 1)
        nodule = result.extracted_nodules[0]
        
        # Should convert 1.5 cm to 15 mm
        self.assertEqual(nodule.size_mm, 15.0)
        
    def test_extraction_paths_tracking(self):
        """Test that extraction paths are properly tracked."""
        text = "A nodule, representing granuloma, measuring 5mm."
        result = self.extractor.extract(text)
        
        print(f"\n[Test 10] Extraction Paths: {[n.to_dict() for n in result.extracted_nodules]}")
        
        self.assertTrue(len(result.extracted_nodules) >= 1)
        nodule = result.extracted_nodules[0]
        
        # Should have recorded extraction paths
        self.assertTrue(len(nodule.extraction_paths) > 0)
        print(f"  Extraction paths: {nodule.extraction_paths}")


if __name__ == '__main__':
    unittest.main()
