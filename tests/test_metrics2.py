import unittest
import os
import pandas as pd
from niaarmts import Dataset
from niaarmts.metrics import (
    calculate_support,
    calculate_confidence,
    calculate_inclusion_metric,
    calculate_amplitude_metric,
    calculate_coverage_metric
)

class TestSpecificRule(unittest.TestCase):

    def setUp(self):
        dataset = Dataset()
        dataset.load_data_from_csv(os.path.join(os.path.dirname(__file__), "test_data", "september24.csv"), timestamp_col='timestamp')
        self.transactions = dataset.get_all_transactions()
        self.features = dataset.get_all_features_with_metadata()

        self.antecedent = [
            {'feature': 'light', 'type': 'Numerical', 'border1': 632.696, 'border2': 636.0194, 'category': 'EMPTY'},
            {'feature': 'moisture', 'type': 'Numerical', 'border1': 1820.3302, 'border2': 1946.4403, 'category': 'EMPTY'}
        ]

        self.consequent = [
            {'feature': 'humidity', 'type': 'Numerical', 'border1': 34.3471, 'border2': 62.8, 'category': 'EMPTY'}
        ]

        self.start = pd.Timestamp("2024-09-14 12:46:57")
        self.end = pd.Timestamp("2024-09-21 16:57:32")

        self.filtered = self.transactions[
            (self.transactions['timestamp'] >= self.start) &
            (self.transactions['timestamp'] <= self.end)
        ]

    def test_filtered_data_length(self):
        count = len(self.filtered)
        print(f"Filtered dataset length: {count}")
        self.assertEqual(count, 61929)

    def test_support_confidence_inclusion_amplitude(self):
        support = calculate_support(self.transactions, self.antecedent, self.consequent, self.start, self.end)
        confidence = calculate_confidence(self.transactions, self.antecedent, self.consequent, self.start, self.end)
        inclusion = calculate_inclusion_metric(self.features, self.antecedent, self.consequent)
        amplitude = calculate_amplitude_metric(self.transactions, self.features, self.antecedent, self.consequent, self.start, self.end, use_interval=False)

        self.assertEqual(0.0008235237126386669, support)
        self.assertEqual(1.0, confidence)
        self.assertEqual(0.75, inclusion)
        self.assertEqual(0.7400385939748908, amplitude)
