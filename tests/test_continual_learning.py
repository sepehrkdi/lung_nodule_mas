
import sys
import os
import json
import logging
import unittest
from pathlib import Path
import tempfile
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dynamic_weights import DynamicWeightCalculator, BASE_WEIGHTS, MIN_WEIGHT, MAX_WEIGHT, LEARNING_RATE

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestContinualLearning(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.weights_file = Path(self.test_dir) / "test_learned_weights.json"
        
        # Patch the global constant in the module (rough but effective for this script)
        import models.dynamic_weights
        models.dynamic_weights.LEARNED_WEIGHTS_FILE = self.weights_file
        
        self.calculator = DynamicWeightCalculator()

    def tearDown(self):
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test that weights file is created on init."""
        self.assertTrue(self.weights_file.exists())
        with open(self.weights_file, 'r') as f:
            data = json.load(f)
        self.assertEqual(data, BASE_WEIGHTS)

    def test_weight_update_reward(self):
        """Test rewarding an accurate agent."""
        agent_name = "radiologist_densenet"
        initial_weight = BASE_WEIGHTS[agent_name]
        
        findings = [
            {"agent_name": agent_name, "probability": 0.9, "predicted_class": 1}
        ]
        ground_truth = 1 # Match
        
        new_weights = self.calculator.update_weights(findings, ground_truth)
        
        expected_weight = min(MAX_WEIGHT, initial_weight + LEARNING_RATE)
        self.assertAlmostEqual(new_weights[agent_name], expected_weight, places=4)
        
        # Verify persistence
        with open(self.weights_file, 'r') as f:
            saved_weights = json.load(f)
        self.assertAlmostEqual(saved_weights[agent_name], expected_weight, places=4)

    def test_weight_update_penalize(self):
        """Test penalizing an inaccurate agent."""
        agent_name = "pathologist_spacy"
        initial_weight = BASE_WEIGHTS[agent_name]
        
        findings = [
            {"agent_name": agent_name, "probability": 0.8, "predicted_class": 1}
        ]
        ground_truth = 0 # Mismatch
        
        new_weights = self.calculator.update_weights(findings, ground_truth)
        
        expected_weight = max(MIN_WEIGHT, initial_weight - LEARNING_RATE)
        self.assertAlmostEqual(new_weights[agent_name], expected_weight, places=4)

    def test_multiple_updates(self):
        """Test cumulative updates."""
        agent_name = "radiologist_resnet"
        
        # 3 correct predictions
        for _ in range(3):
            findings = [{"agent_name": agent_name, "predicted_class": 1}]
            self.calculator.update_weights(findings, ground_truth=1)
            
        with open(self.weights_file, 'r') as f:
            saved_weights = json.load(f)
            
        expected_increase = 3 * LEARNING_RATE
        self.assertAlmostEqual(saved_weights[agent_name], BASE_WEIGHTS[agent_name] + expected_increase, places=4)

if __name__ == '__main__':
    unittest.main()
