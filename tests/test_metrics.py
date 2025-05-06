import unittest
import os
import pandas as pd
import numpy as np
from niaarmts import Dataset
from niaarmts.NiaARMTS import NiaARMTS
from niaarmts.metrics import calculate_support, calculate_confidence, calculate_inclusion_metric, calculate_amplitude_metric

class TestNiaARMTS(unittest.TestCase):

    def setUp(self):
        dataset = Dataset()
        dataset.load_data_from_csv(os.path.join(os.path.dirname(__file__), "test_data", "ts.csv"), timestamp_col='timestamp')
        dim = dataset.calculate_problem_dimension()
        self.assertEqual(dim, 22)

        self.solution = [
            0.93186346, 0.62861471, 0.34720034, 0.51736529, 0.34957089, 0.06362278,
            0.52224747, 0.31581756, 0.78154328, 0.60825901, 0.81263313, 0.4070408,
            0.93014498, 0.46848055, 0.15840165, 0.14308865, 0.86379166, 0.46777855,
            0.60746777, 0.13133695, 0.23055155, 0.60543971
        ]

        self.features = dataset.get_all_features_with_metadata()

        self.niaarmts = NiaARMTS(
            dimension=dim,
            lower=0,
            upper=1,
            features=self.features,
            transactions=dataset.get_all_transactions(),
            interval='false',
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
            delta=1.0
        )

        self.rule = [
            {'feature': 'humidity', 'type': 'Numerical', 'border1': 62.5865, 'border2': 65.8921, 'category': 'EMPTY'},
            {'feature': 'weather', 'type': 'Categorical', 'border1': 1.0, 'border2': 1.0, 'category': 'sun'},
            {'feature': 'temperature', 'type': 'Numerical', 'border1': 29.0172, 'border2': 29.4114, 'category': 'EMPTY'},
            {'feature': 'light', 'type': 'Numerical', 'border1': 9.6287, 'border2': 12.864, 'category': 'EMPTY'}
        ]

        # Store ant and con once in setup for reuse
        self.ant = self.rule[:2]
        self.con = self.rule[2:]

    def test_cut_point(self):
        num_attributes = len(self.rule)
        cut = self.niaarmts.cut_point(0.60543971, num_attributes)
        self.assertEqual(cut, 2)

        antecedent = self.rule[:cut]
        consequent = self.rule[cut:]

        self.assertEqual(antecedent, self.ant)
        self.assertEqual(consequent, self.con)

    def test_start_end_interval(self):
        upper = self.solution[-2]
        lower = self.solution[-3]

        min_interval, max_interval = self.niaarmts.map_to_ts(lower, upper)

        self.assertEqual(min_interval, 13)
        self.assertEqual(max_interval, 22)

        start = self.niaarmts.transactions.loc[min_interval, 'timestamp']
        end = self.niaarmts.transactions.loc[max_interval, 'timestamp']

        self.assertEqual(str(start), str('2024-09-08 20:16:21'))
        self.assertEqual(str(end), str('2024-09-08 20:17:51'))

    def test_calculate_support(self):
        upper = self.solution[-2]
        lower = self.solution[-3]

        min_interval, max_interval = self.niaarmts.map_to_ts(lower, upper)
        start = self.niaarmts.transactions.loc[min_interval, 'timestamp']
        end = self.niaarmts.transactions.loc[max_interval, 'timestamp']

        support = calculate_support(self.niaarmts.transactions, self.ant, self.con, start, end)
        self.assertEqual(support, 0.0)

        confidence = calculate_confidence(self.niaarmts.transactions, self.ant, self.con, start, end)
        self.assertEqual(confidence, 0.0)

    def test_calculate_support_2(self):
        ant = [{'feature': 'weather', 'type': 'Categorical', 'border1': 1.0, 'border2': 1.0, 'category': 'clouds'}]
        ant2 = ant + [{'feature': 'humidity', 'type': 'Numerical', 'border1': 60.23, 'border2': 65.8921, 'category': 'EMPTY'}]
        ant3 = ant2 + [{'feature': 'light', 'type': 'Numerical', 'border1': 13.00, 'border2': 20.8921, 'category': 'EMPTY'}]

        con = [{'feature': 'temperature', 'type': 'Numerical', 'border1': 28.5, 'border2': 28.5, 'category': 'EMPTY'}]
        con2 = [{'feature': 'temperature', 'type': 'Numerical', 'border1': 0, 'border2': 100, 'category': 'EMPTY'}]

        upper = self.solution[-2]
        lower = self.solution[-3]
        min_interval, max_interval = self.niaarmts.map_to_ts(lower, upper)
        start = self.niaarmts.transactions.loc[min_interval, 'timestamp']
        end = self.niaarmts.transactions.loc[max_interval, 'timestamp']

        self.assertEqual(calculate_support(self.niaarmts.transactions, ant, con, start, end), 0.2)
        self.assertEqual(calculate_support(self.niaarmts.transactions, ant2, con, start, end), 0.0)
        self.assertEqual(calculate_support(self.niaarmts.transactions, ant3, con, start, end), 0.0)

        self.assertEqual(calculate_support(self.niaarmts.transactions, ant, con2, start, end), 1.0)
        self.assertEqual(calculate_support(self.niaarmts.transactions, ant2, con2, start, end), 0.3)
        self.assertEqual(calculate_support(self.niaarmts.transactions, ant3, con2, start, end), 0.1)

        self.assertEqual(calculate_confidence(self.niaarmts.transactions, ant, con, start, end), 0.2)
        self.assertEqual(calculate_confidence(self.niaarmts.transactions, ant2, con, start, end), 0.0)
        self.assertEqual(calculate_confidence(self.niaarmts.transactions, ant3, con, start, end), 0.0)

        self.assertEqual(calculate_confidence(self.niaarmts.transactions, ant, con2, start, end), 1.0)
        self.assertEqual(calculate_confidence(self.niaarmts.transactions, ant2, con2, start, end), 1.0)
        self.assertEqual(calculate_confidence(self.niaarmts.transactions, ant3, con2, start, end), 1.0)

    def test_calculate_inclusion(self):
        con2 = [{'feature': 'temperature', 'type': 'Numerical', 'border1': 0, 'border2': 100, 'category': 'EMPTY'}]
        ant = [{'feature': 'weather', 'type': 'Categorical', 'border1': 1.0, 'border2': 1.0, 'category': 'clouds'}]
        ant2 = ant + [{'feature': 'humidity', 'type': 'Numerical', 'border1': 60.23, 'border2': 65.8921, 'category': 'EMPTY'}]

        inclusion1 = calculate_inclusion_metric(self.features, ant, con2)
        inclusion2 = calculate_inclusion_metric(self.features, ant2, con2)

        self.assertEqual(inclusion1, 0.4)
        self.assertEqual(inclusion2, 0.6)


    def test_calculate_amplitude_2(self):
        upper = self.solution[-2]
        lower = self.solution[-3]
        min_interval, max_interval = self.niaarmts.map_to_ts(lower, upper)
        start = self.niaarmts.transactions.loc[min_interval, 'timestamp']
        end = self.niaarmts.transactions.loc[max_interval, 'timestamp']

        ant = [{'feature': 'weather', 'type': 'Categorical', 'border1': 1.0, 'border2': 1.0, 'category': 'clouds'}]
        ant2 = ant + [{'feature': 'humidity', 'type': 'Numerical', 'border1': 59.5, 'border2': 60.5, 'category': 'EMPTY'}]
        con = [{'feature': 'temperature', 'type': 'Numerical', 'border1': 28.4, 'border2': 28.5, 'category': 'EMPTY'}]

        amplitude1 = calculate_amplitude_metric(self.niaarmts.transactions, self.features, ant, con, start, end, use_interval=False)
        amplitude2 = calculate_amplitude_metric(self.niaarmts.transactions, self.features, ant2, con, start, end, use_interval=False)

        self.assertEqual(amplitude1, 0.4999999999999911)
        self.assertEqual(amplitude2, 0.5326086956521698)



