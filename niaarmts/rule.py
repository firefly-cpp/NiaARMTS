import numpy as np

def build_rule(solution, features, is_time_series=False):
    """
    Build association rules based on a given solution and feature metadata.

    Args:
        solution (list[float]): The solution array containing encoded thresholds, permutations and feature values.
        features (dict): A dictionary where keys are feature names, and values contain metadata about the feature.
        is_time_series (bool): Whether the dataset contains time series data.

    Returns:
        list: A list of rules constructed from the solution and features.
    """
    is_first_attribute = True
    attributes = []

    # Extract the number of features and the permutation part of the solution
    num_features = len(features)
    len_solution = len(solution)

    # Safety check: Ensure solution length matches expected dimensions
    if len_solution < num_features:
        raise ValueError("Solution length is smaller than the number of features.")

    # Separate the permutation part
    permutation_part = solution[-num_features:]
    solution_part = solution[:-num_features]

    # Sort the permutation in descending order
    permutation_indices = np.argsort(permutation_part)[::-1]

    # Iterate over features based on the permutation order
    for i in permutation_indices:
        feature_name = list(features.keys())[i]
        feature_meta = features[feature_name]
        feature_type = feature_meta['type']

        # Calculate the position of the vector from solution
        vector_position = feature_position(features, feature_name)

        # Determine threshold position based on feature type
        threshold_position = vector_position + 2 if feature_type == "Numerical" else vector_position + 1

        if solution_part[vector_position] > solution_part[threshold_position]:
            # Handle numerical or time-series features
            if feature_type != "Categorical":
                border1 = np.round(calculate_border(feature_meta, solution_part[vector_position]), 4)
                border2 = np.round(calculate_border(feature_meta, solution_part[vector_position + 1]), 4)

                # Ensure correct border ordering
                if border1 > border2:
                    border1, border2 = border2, border1

                # Add the numerical feature to the attribute list
                if is_first_attribute:
                    attributes = add_attribute([], feature_name, feature_type, border1, border2, "EMPTY")
                    is_first_attribute = False
                else:
                    attributes = add_attribute(attributes, feature_name, feature_type, border1, border2, "EMPTY")
            else:
                # Handle categorical features
                categories = feature_meta['categories']
                selected_category = calculate_selected_category(solution_part[vector_position], len(categories))

                # Add the categorical feature to the attribute list
                if is_first_attribute:
                    attributes = add_attribute([], feature_name, feature_type, 1.0, 1.0, categories[selected_category])
                    is_first_attribute = False
                else:
                    attributes = add_attribute(attributes, feature_name, feature_type, 1.0, 1.0, categories[selected_category])

    return attributes

def feature_position(features, feature_name):
    """
    Find the position of a feature in the solution vector based on its type.
    Args:
        features (dict): The dictionary containing metadata about the features.
        feature_name (str): The name of the feature to find the position for.

    Returns:
        int: The position of the feature in the solution vector.
    """
    position = 0
    for feat_name, feat_meta in features.items():
        if feat_name == feature_name:
            break
        # Categorical features take 2 slots, others take 3
        position += 2 if feat_meta['type'] == 'Categorical' else 3
    return position

def calculate_border(feature_meta, value):
    """
    Calculate the border (threshold) value for a numerical or time series feature.

    Args:
        feature_meta (dict): Metadata for the feature (including min/max values).
        value (float): The value to map to the feature's range.

    Returns:
        float: The calculated border value.
    """
    feature_min = feature_meta['min']
    feature_max = feature_meta['max']
    return feature_min + (feature_max - feature_min) * value

def calculate_selected_category(value, num_categories):
    """
    Calculate the index of the selected category based on the solution vector value.

    Args:
        value (float): The encoded value representing a category.
        num_categories (int): The number of categories for the feature.

    Returns:
        int: The index of the selected category.
    """
    return int(value * (num_categories - 1))  # Ensuring category selection is within bounds

def add_attribute(attributes, feature_name, feature_type, border1, border2, category):
    """
    Add a new attribute to the rule being constructed.

    Args:
        attributes (list): The list of attributes constructed so far.
        feature_name (str): The name of the feature to add.
        feature_type (str): The type of the feature (Numerical, Categorical, etc.).
        border1 (float): Lower bound for numerical features (or 1.0 for categorical).
        border2 (float): Upper bound for numerical features (or 1.0 for categorical).
        category (str): The selected category for categorical features (or "EMPTY" for numerical).

    Returns:
        list: The updated list of attributes.
    """
    attribute = {
        'feature': feature_name,
        'type': feature_type,
        'border1': border1,
        'border2': border2,
        'category': category
    }
    attributes.append(attribute)
    return attributes
