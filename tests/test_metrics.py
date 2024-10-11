import unittest
import os
import pandas as pd
import numpy as np
from niaarmts import Dataset
from niaarmts.NiaARMTS import NiaARMTS

class TestNiaARMTS(unittest.TestCase):

    def setUp(self):
        dataset = Dataset()

        # Load data from CSV with the test file
        dataset.load_data_from_csv(os.path.join(os.path.dirname(__file__), "test_data", "ts.csv"), timestamp_col='timestamp')

        # Calculate the problem dimension
        dim = dataset.calculate_problem_dimension()
        self.assertEqual(dim, 22)

        # Sample solution array
        self.solution = [
            0.93186346, 0.62861471, 0.34720034, 0.51736529, 0.34957089, 0.06362278,
            0.52224747, 0.31581756, 0.78154328, 0.60825901, 0.81263313, 0.4070408,
            0.93014498, 0.46848055, 0.15840165, 0.14308865, 0.86379166, 0.46777855,
            0.60746777, 0.13133695, 0.23055155, 0.60543971
        ]

        # Get feature metadata
        self.features = dataset.get_all_features_with_metadata()

        # Initialize NiaARMTS
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
            delta=1.0,
            output=None
        )

        # Sample rule created from build_rule
        self.rule = [
            {'feature': 'humidity', 'type': 'Numerical', 'border1': 62.5865, 'border2': 65.8921, 'category': 'EMPTY'},
            {'feature': 'weather', 'type': 'Categorical', 'border1': 1.0, 'border2': 1.0, 'category': 'sun'},
            {'feature': 'temperature', 'type': 'Numerical', 'border1': 29.0172, 'border2': 29.4114, 'category': 'EMPTY'},
            {'feature': 'light', 'type': 'Numerical', 'border1': 9.6287, 'border2': 12.864, 'category': 'EMPTY'}
        ]

        self.ant = [
            {'feature': 'humidity', 'type': 'Numerical', 'border1': 62.5865, 'border2': 65.8921, 'category': 'EMPTY'},
            {'feature': 'weather', 'type': 'Categorical', 'border1': 1.0, 'border2': 1.0, 'category': 'sun'}
             ]

        self.con = [{'feature': 'temperature', 'type': 'Numerical', 'border1': 29.0172, 'border2': 29.4114, 'category': 'EMPTY'}, {'feature': 'light', 'type': 'Numerical', 'border1': 9.6287, 'border2': 12.864, 'category': 'EMPTY'}]

    def test_cut_point(self):
        # Test for cut_point method
        num_attributes = len(self.rule)  # Number of attributes in the rule

        # Calculate the cut
        cut = self.niaarmts.cut_point(0.60543971, num_attributes)

        self.assertEqual(cut, 2)

        antecedent = self.rule[:cut]
        consequent = self.rule[cut:]

        # check antecedents and consequents
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
        # Get start and end timestamps from the solution
        upper = self.solution[-2]
        lower = self.solution[-3]

        min_interval, max_interval = self.niaarmts.map_to_ts(lower, upper)

        start = self.niaarmts.transactions.loc[min_interval, 'timestamp']
        end = self.niaarmts.transactions.loc[max_interval, 'timestamp']

        support = self.niaarmts.calculate_support(self.niaarmts.transactions, self.ant, self.con, start, end)

        self.assertEqual(support, 0.0)

        confidence = self.niaarmts.calculate_confidence(self.niaarmts.transactions, self.ant, self.con, start, end)

        self.assertEqual(confidence, 0.0)

    def test_calculate_support_2(self):

        ant = [
            {'feature': 'weather', 'type': 'Categorical', 'border1': 1.0, 'border2': 1.0, 'category': 'clouds'}
             ]

        ant2 = [
            {'feature': 'weather', 'type': 'Categorical', 'border1': 1.0, 'border2': 1.0, 'category': 'clouds'},
            {'feature': 'humidity', 'type': 'Numerical', 'border1': 60.23, 'border2': 65.8921, 'category': 'EMPTY'},
             ]

        ant3 = [
            {'feature': 'weather', 'type': 'Categorical', 'border1': 1.0, 'border2': 1.0, 'category': 'clouds'},
            {'feature': 'humidity', 'type': 'Numerical', 'border1': 60.23, 'border2': 65.8921, 'category': 'EMPTY'},
            {'feature': 'light', 'type': 'Numerical', 'border1': 13.00, 'border2': 20.8921, 'category': 'EMPTY'},
             ]

        con = [{'feature': 'temperature', 'type': 'Numerical', 'border1': 28.5, 'border2': 28.5, 'category': 'EMPTY'}]

        con2 = [{'feature': 'temperature', 'type': 'Numerical', 'border1': 0, 'border2': 100, 'category': 'EMPTY'}]

        upper = self.solution[-2]
        lower = self.solution[-3]

        min_interval, max_interval = self.niaarmts.map_to_ts(lower, upper)

        start = self.niaarmts.transactions.loc[min_interval, 'timestamp']
        end = self.niaarmts.transactions.loc[max_interval, 'timestamp']

        support = self.niaarmts.calculate_support(self.niaarmts.transactions, ant, con, start, end)
        support2 = self.niaarmts.calculate_support(self.niaarmts.transactions, ant2, con, start, end)
        support3 = self.niaarmts.calculate_support(self.niaarmts.transactions, ant3, con, start, end)

        support4 = self.niaarmts.calculate_support(self.niaarmts.transactions, ant, con2, start, end)
        support5 = self.niaarmts.calculate_support(self.niaarmts.transactions, ant2, con2, start, end)
        support6 = self.niaarmts.calculate_support(self.niaarmts.transactions, ant3, con2, start, end)

        self.assertEqual(support, 0.2) # 2 transactions out of 10
        self.assertEqual(support2, 0.00) # 0 transactions out of 100
        self.assertEqual(support3, 0.00) # 1 transaction out of 100

        self.assertEqual(support4, 1) # 10 transactions out of 10
        self.assertEqual(support5, 0.3) # 3 transactions out of 100
        self.assertEqual(support6, 0.1) # 1 transaction out of 100

        confidence = self.niaarmts.calculate_confidence(self.niaarmts.transactions, self.ant, self.con, start, end)

        self.assertEqual(confidence, 0.01)
