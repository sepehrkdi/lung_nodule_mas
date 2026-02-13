
import unittest
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from knowledge.prolog_engine import PrologEngine

class TestConsensusPatterns(unittest.TestCase):
    def setUp(self):
        self.engine = PrologEngine()
        
    def test_visual_text_conflict_recheck(self):
        """
        Test Pattern 1: CV sees malignancy (0.8), NLP sees benign (0.1).
        Expect: 'visual_text_conflict_recheck', prob ~0.45, low confidence.
        """
        case_id = "test_conflict_1"
        
        findings = [
            {"agent_name": "radiologist_densenet", "probability": 0.85, "predicted_class": 1},
            {"agent_name": "radiologist_resnet", "probability": 0.75, "predicted_class": 1},
            {"agent_name": "pathologist_spacy", "probability": 0.1, "predicted_class": 0},
            {"agent_name": "pathologist_regex", "probability": 0.1, "predicted_class": 0}
        ]
        
        print(f"\n[Test] Visual-Text Conflict ({case_id})")
        
        # Compute consensus
        result = self.engine.compute_consensus(case_id, findings)
        print(f"Result: {result}")
        
        self.assertEqual(result['strategy'], 'visual_text_conflict_recheck')
        self.assertAlmostEqual(result['probability'], 0.45, delta=0.05) # (0.8+0.1)/2 = 0.45
        self.assertLess(result['confidence'], 0.5)

    def test_text_override_missed_visual(self):
        """
        Test Pattern 2: CV sees benign (0.1), NLP sees malignancy (0.9).
        Expect: 'text_override_missed_visual', prob ~0.9, high confidence.
        """
        case_id = "test_override_1"
        
        findings = [
            {"agent_name": "radiologist_densenet", "probability": 0.1, "predicted_class": 0},
            {"agent_name": "radiologist_resnet", "probability": 0.1, "predicted_class": 0},
            {"agent_name": "pathologist_spacy", "probability": 0.95, "predicted_class": 1},
            {"agent_name": "pathologist_regex", "probability": 0.85, "predicted_class": 1}
        ]
        
        print(f"\n[Test] Text Override ({case_id})")
        
        # Compute consensus
        result = self.engine.compute_consensus(case_id, findings)
        print(f"Result: {result}")
        
        self.assertEqual(result['strategy'], 'text_override_missed_visual')
        self.assertGreater(result['probability'], 0.8)
        self.assertGreater(result['confidence'], 0.7)

    def test_standard_consensus(self):
        """
        Test Standard: All agree (0.8).
        Expect: 'weighted_average' (or similar), high prob, high confidence.
        """
        case_id = "test_agree_1"
        
        findings = [
            {"agent_name": "radiologist_densenet", "probability": 0.8, "predicted_class": 1},
            {"agent_name": "pathologist_spacy", "probability": 0.8, "predicted_class": 1}
        ]
        
        print(f"\n[Test] Standard Consensus ({case_id})")
        
        result = self.engine.compute_consensus(case_id, findings)
        print(f"Result: {result}")
        
        # Should NOT use special strategies
        self.assertNotIn(result['strategy'], ['visual_text_conflict_recheck', 'text_override_missed_visual'])
        self.assertGreater(result['probability'], 0.7)

if __name__ == '__main__':
    unittest.main()
