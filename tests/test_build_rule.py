import unittest
import numpy as np
from niaarmts import Dataset
from niaarmts.rule import build_rule, calculate_border, calculate_selected_category, feature_position
import pytest

class TestBuildRule(unittest.TestCase):

    def test_rule_building(self):
        # Create an instance of Dataset
        dataset = Dataset()

        # Load data from CSV with our test file
        dataset.load_data_from_csv('ts.csv', timestamp_col='timestamp')

        # Calculate the problem dimension

        # Dataset outline
        # temperature - numerical - + 4
        # humidity - numerical - + 4
        # moisture - numerical - + 4
        # light - numerical - + 4
        # time series - + 2
        # cut point - + 1

        dim = dataset.calculate_problem_dimension()
        self.assertEqual(dim, 22)

        # Sample solution array
        solution = [
            0.93186346, 0.62861471, 0.34720034, 0.51736529, 0.34957089, 0.06362278,
            0.52224747, 0.31581756, 0.78154328, 0.60825901, 0.81263313, 0.4070408,
            0.93014498, 0.46848055, 0.15840165, 0.14308865, 0.86379166, 0.46777855,
            0.60746777, 0.13133695, 0.23055155, 0.60543971
        ]

        # Get feature metadata
        features = dataset.get_all_features_with_metadata()

        # minor checks - check dataset properties
        first_feature_name = list(features.keys())[0]
        self.assertEqual(list(features.keys())[0], "temperature")

        # Build the rule using the solution and feature metadata
        rule = build_rule(solution, features)

        # Check the length of the rule
        self.assertEqual(len(rule), 4)

        # Detailed check
        self.assertEqual(rule[0]['feature'], 'humidity')
        self.assertEqual(rule[0]['type'], 'Numerical')
        self.assertAlmostEqual(rule[0]['border1'], 62.5865, places=4)
        self.assertAlmostEqual(rule[0]['border2'], 65.8921, places=4)
        self.assertEqual(rule[0]['category'], 'EMPTY')

        self.assertEqual(rule[1]['feature'], 'weather')
        self.assertEqual(rule[1]['type'], 'Categorical')
        self.assertAlmostEqual(rule[1]['border1'], 1.0)
        self.assertAlmostEqual(rule[1]['border2'], 1.0)
        self.assertEqual(rule[1]['category'], 'sun')

        self.assertEqual(rule[2]['feature'], 'temperature')
        self.assertEqual(rule[2]['type'], 'Numerical')
        self.assertAlmostEqual(rule[2]['border1'], 29.0172, places=4)
        self.assertAlmostEqual(rule[2]['border2'], 29.4114, places=4)
        self.assertEqual(rule[2]['category'], 'EMPTY')

        self.assertEqual(rule[3]['feature'], 'light')
        self.assertEqual(rule[3]['type'], 'Numerical')
        self.assertAlmostEqual(rule[3]['border1'], 9.6287, places=4)
        self.assertAlmostEqual(rule[3]['border2'], 12.864, places=4)
        self.assertEqual(rule[3]['category'], 'EMPTY')


        # check also permutations, etc
        num_features = len(features)
        self.assertEqual(num_features,5)
        len_solution = len(solution)
        self.assertEqual(len_solution, 22)

        permutation_part = solution[-num_features:]
        self.assertEqual(permutation_part, [0.46777855, 0.60746777, 0.13133695, 0.23055155, 0.60543971])
        solution_part = solution[:-num_features]
        sol = [
            0.93186346, 0.62861471, 0.34720034, 0.51736529, 0.34957089, 0.06362278,
            0.52224747, 0.31581756, 0.78154328, 0.60825901, 0.81263313, 0.4070408,
            0.93014498, 0.46848055, 0.15840165, 0.14308865, 0.86379166]
        self.assertEqual(solution_part, sol)



# TODOS - check border calculations
