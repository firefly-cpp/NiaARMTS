import unittest
import pandas as pd
import numpy as np
from niaarmts.feature import Feature

class TestFeature(unittest.TestCase):

    def test_empty_data_raises_value_error(self):
        """Test if passing an empty DataFrame raises ValueError."""
        empty_data = pd.DataFrame()
        with self.assertRaises(ValueError):
            Feature(empty_data)

    def test_get_numerical_features(self):
        """Test if numerical features are identified correctly."""
        data = pd.DataFrame({
            'num_col1': [1.5, 2.3, 3.8],
            'num_col2': [10, 20, 30],
            'cat_col': ['A', 'B', 'A'],
            'datetime_col': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03'])
        })
        feature = Feature(data)
        numerical_features = feature.get_numerical_features()
        self.assertEqual(numerical_features, ['num_col1', 'num_col2'])

    def test_get_categorical_features(self):
        """Test if categorical features are identified correctly."""
        data = pd.DataFrame({
            'num_col': [1.5, 2.3, 3.8],
            'cat_col': ['A', 'B', 'A'],
            'datetime_col': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03'])
        })
        feature = Feature(data)
        categorical_features = feature.get_categorical_features()
        self.assertEqual(categorical_features, ['cat_col'])

    def test_get_datetime_features(self):
        """Test if datetime features are identified correctly."""
        data = pd.DataFrame({
            'num_col': [1.5, 2.3, 3.8],
            'cat_col': ['A', 'B', 'A'],
            'datetime_col': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03'])
        })
        feature = Feature(data)
        datetime_features = feature.get_datetime_features()
        self.assertEqual(datetime_features, ['datetime_col'])

    def test_get_feature_numerical_stats(self):
        """Test if feature statistics are calculated correctly for numerical columns."""
        data = pd.DataFrame({
            'num_col': [1.5, 2.3, 3.8],
            'cat_col': ['A', 'B', 'A']
        })
        feature = Feature(data)
        stats = feature.get_feature_stats('num_col')
        self.assertEqual(stats['type'], 'Numerical')
        self.assertAlmostEqual(stats['min'], 1.5)
        self.assertAlmostEqual(stats['max'], 3.8)
        self.assertAlmostEqual(stats['mean'], 2.533333, places=5)
        self.assertAlmostEqual(stats['std_dev'], 1.1676, places=4)

    def test_get_feature_stats_categorical(self):
        """Test if feature statistics are calculated correctly for categorical columns."""
        data = pd.DataFrame({
            'num_col': [1.5, 2.3, 3.8],
            'cat_col': ['A', 'B', 'A']  # 2 unique categories
        })
        feature = Feature(data)
        stats = feature.get_feature_stats('cat_col')

        # Check that the feature is correctly identified as 'Categorical'
        self.assertEqual(stats['type'], 'Categorical')
        # Ensure that the number of unique categories is correctly identified
        self.assertEqual(stats['unique_classes'], 2)
        # Ensure that the correct categories are returned (sorted)
        self.assertCountEqual(stats['classes'], ['A', 'B'])  # Check if categories match
