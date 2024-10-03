import unittest
import numpy as np
from niaarmts.rule import build_rule, calculate_border, calculate_selected_category

class TestBuildRule(unittest.TestCase):

    def test_numerical_feature_rule(self):
        """Test building a rule with numerical features."""
        solution = [0.2, 0.8, 0.1, 0.9]  # Solution array
        features = {
            'num_feature1': {'type': 'numerical', 'min': 0.0, 'max': 10.0, 'position': 0},
            'num_feature2': {'type': 'numerical', 'min': 0.0, 'max': 100.0, 'position': 1}
        }
        rules = build_rule(solution, features)
        self.assertEqual(len(rules), 2)
        self.assertEqual(rules[0]['feature'], 'num_feature2')
        self.assertEqual(rules[1]['feature'], 'num_feature1')

    def test_categorical_feature_rule(self):
        """Test building a rule with categorical features."""
        solution = [0.6, 0.1, 0.7, 0.9]  # Solution array
        features = {
            'cat_feature1': {'type': 'categorical', 'categories': ['A', 'B', 'C'], 'position': 0},
            'cat_feature2': {'type': 'categorical', 'categories': ['X', 'Y', 'Z'], 'position': 1}
        }
        rules = build_rule(solution, features)
        self.assertEqual(len(rules), 2)
        self.assertEqual(rules[0]['feature'], 'cat_feature2')
        self.assertEqual(rules[0]['category'], 'Z')
        self.assertEqual(rules[1]['feature'], 'cat_feature1')
        self.assertEqual(rules[1]['category'], 'C')

    def test_time_series_rule(self):
        """Test building a rule with time-series data."""
        solution = [0.3, 0.9, 0.2, 0.8]  # Solution array
        features = {
            'ts_feature1': {'type': 'numerical', 'min': 0.0, 'max': 100.0, 'position': 0},
            'ts_feature2': {'type': 'numerical', 'min': 0.0, 'max': 50.0, 'position': 1}
        }
        rules = build_rule(solution, features, is_time_series=True)
        self.assertEqual(len(rules), 2)
        self.assertEqual(rules[0]['feature'], 'ts_feature2')
        self.assertEqual(rules[1]['feature'], 'ts_feature1')

    def test_mixed_feature_rule(self):
        """Test building a rule with both categorical and numerical features."""
        solution = [0.1, 0.5, 0.3, 0.7]  # Solution array
        features = {
            'num_feature1': {'type': 'numerical', 'min': 0.0, 'max': 10.0, 'position': 0},
            'cat_feature2': {'type': 'categorical', 'categories': ['A', 'B', 'C'], 'position': 1}
        }
        rules = build_rule(solution, features)
        self.assertEqual(len(rules), 2)
        self.assertEqual(rules[0]['feature'], 'cat_feature2')
        self.assertEqual(rules[1]['feature'], 'num_feature1')

    def test_invalid_solution_length(self):
        solution = [0.2, 0.8]  # Too short solution array for 3 features
        features = {
            'num_feature1': {'type': 'numerical', 'min': 0.0, 'max': 10.0, 'position': 0},
            'num_feature2': {'type': 'numerical', 'min': 0.0, 'max': 100.0, 'position': 1},
            'num_feature3': {'type': 'numerical', 'min': 0.0, 'max': 50.0, 'position': 2}
        }
        with self.assertRaises(IndexError):
            build_rule(solution, features)

if __name__ == '__main__':
    unittest.main()
