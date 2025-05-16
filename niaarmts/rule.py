import numpy as np

def build_rule(solution, features, is_time_series=False, start=None, end=None, transactions=None):
    """
    Build association rules based on a given solution and feature metadata.

    Args:
        solution (list[float]): The solution array containing encoded thresholds, permutations and feature values.
        features (dict): A dictionary where keys are feature names, and values contain metadata about the feature.
        is_time_series (bool): Whether the dataset contains time series data.
        start (datetime): Start timestamp for filtering time series data.
        end (datetime): End timestamp for filtering time series data.
        transactions (pd.DataFrame): Transaction data for calculating time-based feature bounds.

    Returns:
        list: A list of rules constructed from the solution and features.
    """
    is_first_attribute = True
    attributes = []

    # Extract the number of features and the permutation part of the solution
    num_features = len(features)
    len_solution = len(solution)

    if len_solution < num_features:
        raise ValueError("Solution length is smaller than the number of features.")

    # Separate the permutation part
    permutation_part = solution[-num_features:]
    solution_part = solution[:-num_features]

    # Sort the permutation in descending order
    permutation_indices = np.argsort(permutation_part)[::-1]

    # Filter transactions if time series is active
    ts_filtered = None
    if is_time_series and transactions is not None and start is not None and end is not None:
        ts_filtered = transactions[(transactions['timestamp'] >= start) & (transactions['timestamp'] <= end)]

    for i in permutation_indices:
        feature_name = list(features.keys())[i]
        feature_meta = features[feature_name]
        feature_type = feature_meta['type']

        # Calculate the position of the vector from solution
        vector_position = feature_position(features, feature_name)
        threshold_position = vector_position + 2 if feature_type != 'Categorical' else vector_position + 1

        if solution_part[vector_position] > solution_part[threshold_position]:
            if feature_type != 'Categorical':
                # Use filtered bounds if available
                if ts_filtered is not None and feature_name in ts_filtered.columns:
                    series = ts_filtered[feature_name].dropna()
                    if not series.empty:
                        temp_min = series.min()
                        temp_max = series.max()
                    else:
                        temp_min = feature_meta['min']
                        temp_max = feature_meta['max']
                else:
                    temp_min = feature_meta['min']
                    temp_max = feature_meta['max']

                border1 = np.round(calculate_border(temp_min, temp_max, solution_part[vector_position]), 4)
                border2 = np.round(calculate_border(temp_min, temp_max, solution_part[vector_position + 1]), 4)

                if border1 > border2:
                    border1, border2 = border2, border1

                if is_first_attribute:
                    attributes = add_attribute([], feature_name, feature_type, border1, border2, "EMPTY")
                    is_first_attribute = False
                else:
                    attributes = add_attribute(attributes, feature_name, feature_type, border1, border2, "EMPTY")
            else:
                categories = feature_meta['categories']
                selected_category = calculate_selected_category(solution_part[vector_position], len(categories))
                if is_first_attribute:
                    attributes = add_attribute([], feature_name, feature_type, 1.0, 1.0, categories[selected_category])
                    is_first_attribute = False
                else:
                    attributes = add_attribute(attributes, feature_name, feature_type, 1.0, 1.0, categories[selected_category])

    return attributes

def feature_position(features, feature_name):
    position = 0
    for feat_name, feat_meta in features.items():
        if feat_name == feature_name:
            break
        position += 2 if feat_meta['type'] == 'Categorical' else 3
    return position

def calculate_border(feature_min, feature_max, value):
    return feature_min + (feature_max - feature_min) * value

def calculate_selected_category(value, num_categories):
    return int(value * (num_categories - 1))

def add_attribute(attributes, feature_name, feature_type, border1, border2, category):
    attribute = {
        'feature': feature_name,
        'type': feature_type,
        'border1': border1,
        'border2': border2,
        'category': category
    }
    attributes.append(attribute)
    return attributes
