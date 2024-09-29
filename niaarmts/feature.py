import pandas as pd
import numpy as np

class Feature:
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the Feature class.

        :param data: A Pandas DataFrame containing the dataset.
        """
        self.data = data
        if self.data.empty:
            raise ValueError("No data available for feature analysis.")

    def get_feature_summary(self):
        """
        Provides a summary of features in the dataset, categorized by type.

        :return: A dictionary with feature summaries.
        """
        summary = {}
        for column in self.data.columns:
            col_data = self.data[column]

            if np.issubdtype(col_data.dtype, np.number):
                summary[column] = {
                    'type': 'Numerical',
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'mean': col_data.mean(),
                    'std_dev': col_data.std(),
                }
            elif col_data.dtype == 'object':
                summary[column] = {
                    'type': 'Categorical',
                    'unique_classes': col_data.nunique(),
                    'classes': col_data.unique()
                }
            elif np.issubdtype(col_data.dtype, np.datetime64):
                summary[column] = {
                    'type': 'Datetime',
                    'min': col_data.min(),
                    'max': col_data.max(),
                }
            else:
                summary[column] = {
                    'type': 'Unknown',
                }

        return summary

    def get_numerical_features(self):
        """
        Returns a list of numerical features in the dataset.

        :return: A list of numerical feature names.
        """
        numerical_features = [col for col in self.data.columns if np.issubdtype(self.data[col].dtype, np.number)]
        return numerical_features

    def get_categorical_features(self):
        """
        Returns a list of categorical features in the dataset.

        :return: A list of categorical feature names.
        """
        categorical_features = [col for col in self.data.columns if self.data[col].dtype == 'object']
        return categorical_features

    def get_datetime_features(self):
        """
        Returns a list of datetime features in the dataset.

        :return: A list of datetime feature names.
        """
        datetime_features = [col for col in self.data.columns if np.issubdtype(self.data[col].dtype, np.datetime64)]
        return datetime_features

    def get_feature_stats(self, feature_name: str):
        """
        Provides detailed statistics for a given feature.

        :param feature_name: The name of the feature to analyze.
        :return: A dictionary with statistics about the feature.
        """
        if feature_name not in self.data.columns:
            raise ValueError(f"Feature '{feature_name}' not found in the dataset.")

        col_data = self.data[feature_name]
        if np.issubdtype(col_data.dtype, np.number):
            return {
                'type': 'Numerical',
                'min': col_data.min(),
                'max': col_data.max(),
                'mean': col_data.mean(),
                'std_dev': col_data.std(),
            }
        elif col_data.dtype == 'object':
            return {
                'type': 'Categorical',
                'unique_classes': col_data.nunique(),
                'classes': col_data.unique(),
            }
        elif np.issubdtype(col_data.dtype, np.datetime64):
            return {
                'type': 'Datetime',
                'min': col_data.min(),
                'max': col_data.max(),
            }
        else:
            return {
                'type': 'Unknown',
            }
