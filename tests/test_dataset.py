import unittest
import pandas as pd
from unittest.mock import patch
from niaarmts.dataset import Dataset
from niaarmts.feature import Feature

class TestDataset(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_load_data_from_csv(self, mock_read_csv):
        # Mock data for CSV file
        mock_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['A', 'B', 'A'],
            'timestamp': ['2021-01-01', '2021-01-02', '2021-01-03']
        })
        mock_read_csv.return_value = mock_data

        dataset = Dataset()
        dataset.load_data_from_csv('mock_file.csv', timestamp_col='timestamp')

        # Test if the data is loaded properly
        self.assertFalse(dataset.data.empty)
        self.assertEqual(dataset.data.shape, (3, 3))
        self.assertEqual(dataset.timestamp_col, 'timestamp')
        self.assertTrue('timestamp' in dataset.data.columns)
        self.assertTrue('col2' in dataset.data.columns)

    def test_get_feature_summary(self):
        dataset = Dataset()
        # Manually set data for testing
        dataset.data = pd.DataFrame({
            'num_col': [1.5, 2.3, 3.8],
            'cat_col': ['A', 'B', 'A'],
            'timestamp': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03'])
        })

        # Initialize feature analysis
        dataset.feature_analysis = Feature(dataset.data)

        feature_summary = dataset.get_feature_summary()

        # Test feature types
        self.assertEqual(feature_summary['num_col']['type'], 'Numerical')
        self.assertEqual(feature_summary['cat_col']['type'], 'Categorical')
        self.assertEqual(feature_summary['timestamp']['type'], 'Datetime')

    def test_calculate_problem_dimension(self):
        dataset = Dataset()
        # Manually set data for testing
        dataset.data = pd.DataFrame({
            'num_col': [1.5, 2.3, 3.8],
            'cat_col': ['A', 'B', 'A'],
            'interval': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03']),
            'timestamp': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03'])
        })

        # Initialize feature analysis
        dataset.feature_analysis = Feature(dataset.data)

        # Calculate problem dimension
        dimension = dataset.calculate_problem_dimension()

        # Test the correct calculation of dimensions
        self.assertEqual(dimension, 9)  # 3 for num_col, 2 for cat_col, 1 for interval, 2 for timestamp, 1 for cut point

if __name__ == '__main__':
    unittest.main()
