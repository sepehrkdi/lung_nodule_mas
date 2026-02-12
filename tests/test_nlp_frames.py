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

if __name__ == '__main__':
    unittest.main()
