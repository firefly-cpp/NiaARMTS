import pandas as pd
import numpy as np
from niaarmts.feature import Feature

class Dataset:
    def __init__(self):
        """
        Initializes the Dataset class.
        """
        self.data = pd.DataFrame()
        self.timestamp_col = None
        self.feature_analysis = None

    def load_data_from_csv(self, file_path: str, timestamp_col: str = None):
        """
        Load the dataset from a CSV file.

        :param file_path: Path to the CSV file.
        :param timestamp_col: Optional, the name of the column containing timestamps (if applicable).
        """
        self.data = pd.read_csv(file_path)

        if timestamp_col:
            self.timestamp_col = timestamp_col
            # Convert timestamp column to datetime (if present)
            try:
                self.data[timestamp_col] = pd.to_datetime(self.data[timestamp_col])
            except (KeyError, ValueError, TypeError):
                pass

        # Initialize FeatureAnalysis after data loading
        self.feature_analysis = Feature(self.data)

    def get_feature_summary(self):
        """
        Get a summary of features, categorized by type.

        :return: A dictionary with feature summaries.
        """
        if self.feature_analysis is None:
            raise ValueError("Data has not been loaded yet.")
        return self.feature_analysis.get_feature_summary()

    def get_numerical_features(self):
        """
        Get a list of numerical features.

        :return: A list of numerical feature names.
        """
        if self.feature_analysis is None:
            raise ValueError("Data has not been loaded yet.")
        return self.feature_analysis.get_numerical_features()

    def get_categorical_features(self):
        """
        Get a list of categorical features.

        :return: A list of categorical feature names.
        """
        if self.feature_analysis is None:
            raise ValueError("Data has not been loaded yet.")
        return self.feature_analysis.get_categorical_features()

    def get_datetime_features(self):
        """
        Get a list of datetime features.

        :return: A list of datetime feature names.
        """
        if self.feature_analysis is None:
            raise ValueError("Data has not been loaded yet.")
        return self.feature_analysis.get_datetime_features()

    def get_feature_stats(self, feature_name: str):
        """
        Get detailed statistics for a given feature.

        :param feature_name: The name of the feature to analyze.
        :return: A dictionary with statistics about the feature.
        """
        if self.feature_analysis is None:
            raise ValueError("Data has not been loaded yet.")
        return self.feature_analysis.get_feature_stats(feature_name)

    def calculate_problem_dimension(self):
        """
        Calculates the dimension of the problem based on the type of features.

        - Adds 4 for each numerical attribute (lower and upper bound, threshold and permutation).
        - Adds 3 for each categorical attribute (category, threshold and permutation).
        - Adds 1 if an interval (datetime) attribute is present.
        - Adds 2 if time series data (timestamp) is present.
        - Adds 1 for cut point value.

        :return: The calculated dimension of the problem.
        """
        dimension = 0
        numerical_features = self.get_numerical_features()
        categorical_features = self.get_categorical_features()

        # Add to dimension based on numerical features
        dimension += len(numerical_features) * 4

        # Add to dimension based on categorical features
        dimension += len(categorical_features) * 3

        # Add to dimension if interval (datetime) attribute is present
        if 'interval' in self.data.columns:
            dimension += 1

        # Add to dimension if time series data (timestamp) is present (assuming it's datetime feature)
        if 'timestamp' in self.data.columns or self.data.select_dtypes(include=[np.datetime64]).shape[1] > 0:
            dimension += 2

        # cut point
        dimension = dimension + 1

        return dimension

    def get_all_features_with_metadata(self):
        if self.feature_analysis is None:
            raise ValueError("Data has not been loaded yet.")

        # Prepare feature metadata
        features_metadata = {}

        for idx, column in enumerate(self.data.columns):
            if column == 'timestamp' or column == 'interval':  # Skip the timestamp or interval column
                continue

            feature_type = 'Unknown'
            col_data = self.data[column]

            if np.issubdtype(col_data.dtype, np.number):
                feature_type = 'Numerical'
            elif col_data.dtype == 'object':
                feature_type = 'Categorical'

            # Create metadata for each feature
            features_metadata[column] = {
                'type': feature_type,
                'min': col_data.min() if np.issubdtype(col_data.dtype, np.number) else None,
                'max': col_data.max() if np.issubdtype(col_data.dtype, np.number) else None,
                'categories': col_data.unique() if feature_type == 'Categorical' else None,
                'position': idx  # Position in the dataset
            }

        return features_metadata


    def get_all_transactions(self):
        """
        Get all transactions (rows) from the dataset.

        :return: A DataFrame containing all transactions (rows).
        """
        if self.data.empty:
            raise ValueError("Data has not been loaded yet.")

        return self.data
