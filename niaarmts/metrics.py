import pandas as pd
import numpy as np

def calculate_support(df, antecedents, consequents, start=0, end=0, use_interval=False):
    """
    Calculate the support for the given list of antecedents and consequents within the specified time range or interval range.

    Args:
        df (pd.DataFrame): The dataset containing the transactions.
        antecedents (list): A list of dictionaries defining the antecedent conditions.
        consequents (list): A list of dictionaries defining the consequent conditions.
        start (int or datetime): The start of the interval (if use_interval is True) or timestamp range.
        end (int or datetime): The end of the interval (if use_interval is True) or timestamp range.
        use_interval (bool): Whether to filter by 'interval' (True) or 'timestamp' (False) for time-based filtering.

    Returns:
        float: The support value, which is the ratio of transactions matching both antecedents and consequents
        to the total transactions in the filtered range. If no transactions exist, returns 0.
    """
    if use_interval:
        df_filtered = df[(df['interval'] >= start) & (df['interval'] <= end)]
    else:
        df_filtered = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

    filtered = len(df_filtered)

    # Apply each antecedent condition
    for antecedent in antecedents:
        if antecedent['type'] == 'Categorical':
            df_filtered = df_filtered[df_filtered[antecedent['feature']] == antecedent['category']]
        elif antecedent['type'] == 'Numerical':
            df_filtered = df_filtered[
                (df_filtered[antecedent['feature']] >= antecedent['border1']) &
                (df_filtered[antecedent['feature']] <= antecedent['border2'])
            ]

    # Apply each consequent condition
    for consequent in consequents:
        if consequent['type'] == 'Categorical':
            df_filtered = df_filtered[df_filtered[consequent['feature']] == consequent['category']]
        elif consequent['type'] == 'Numerical':
            df_filtered = df_filtered[
                (df_filtered[consequent['feature']] >= consequent['border1']) &
                (df_filtered[consequent['feature']] <= consequent['border2'])
            ]

    # Calculate support: the ratio of rows matching both antecedents and consequents to total filtered rows
    return len(df_filtered) / filtered if len(df) > 0 else 0


def calculate_confidence(df, antecedents, consequents, start, end, use_interval=False):
    """
    Calculate the confidence for the given list of antecedents and consequents within the specified time range or interval range.

    Args:
        df (pd.DataFrame): The dataset containing the transactions.
        antecedents (list): A list of dictionaries defining the antecedent conditions.
        consequents (list): A list of dictionaries defining the consequent conditions.
        start (int or datetime): The start of the interval (if use_interval is True) or timestamp range.
        end (int or datetime): The end of the interval (if use_interval is True) or timestamp range.
        use_interval (bool): Whether to filter by 'interval' (True) or 'timestamp' (False) for time-based filtering.

    Returns:
        float: The confidence value, which is the ratio of rows matching both antecedents and consequents
        to the rows matching antecedents. If no antecedent-matching rows exist, returns 0.
    """
    if use_interval:
        df_filtered = df[(df['interval'] >= start) & (df['interval'] <= end)]
    else:
        df_filtered = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

    filtered = len(df_filtered)

    # Apply each antecedent condition
    for antecedent in antecedents:
        if antecedent['type'] == 'Categorical':
            df_filtered = df_filtered[df_filtered[antecedent['feature']] == antecedent['category']]
        elif antecedent['type'] == 'Numerical':
            if 'border1' in antecedent and 'border2' in antecedent:
                df_filtered = df_filtered[
                    (df_filtered[antecedent['feature']] >= antecedent['border1']) &
                    (df_filtered[antecedent['feature']] <= antecedent['border2'])
                ]
            else:
                raise ValueError("Numerical antecedent must have 'border1' and 'border2'")

    antecedent_support = df_filtered.copy()

    # Apply consequent conditions to the antecedent-supporting rows
    for consequent in consequents:
        if consequent['type'] == 'Categorical':
            antecedent_support = antecedent_support[antecedent_support[consequent['feature']] == consequent['category']]
        elif consequent['type'] == 'Numerical':
            if 'border1' in consequent and 'border2' in consequent:
                antecedent_support = antecedent_support[
                    (antecedent_support[consequent['feature']] >= consequent['border1']) &
                    (antecedent_support[consequent['feature']] <= consequent['border2'])
                ]
            else:
                raise ValueError("Numerical consequent must have 'border1' and 'border2'")

    antecedent_count = len(df_filtered)
    consequent_count = len(antecedent_support)

    # Calculate confidence: the ratio of rows that match both antecedents and consequents to the rows matching antecedents
    return consequent_count / antecedent_count if antecedent_count > 0 else 0.0


def calculate_inclusion_metric(features, antecedents, consequents):
    """
    Calculate the inclusion metric, which measures how many attributes appear in both the antecedent and consequent
    relative to the total number of features in the dataset.

    Args:
        features (dict): A dictionary of feature metadata for the dataset.
        antecedents (list): A list of dictionaries defining the antecedent conditions.
        consequents (list): A list of dictionaries defining the consequent conditions.

    Returns:
        float: The inclusion metric value, normalized between 0 and 1.
    """
    all_dataset_features = len(features)
    antecedent_features = {feature['feature'] for feature in antecedents}
    consequent_features = {feature['feature'] for feature in consequents}

    common_features = len(consequent_features) + len(antecedent_features)

    if common_features == 0:
        return 0.0

    # Calculate inclusion metric normalized by total features
    inclusion_metric = common_features / all_dataset_features

    return inclusion_metric


def calculate_amplitude_metric(df, features, antecedents, consequents, start=0, end=0, use_interval=False):
    """
    Calculate the amplitude metric for the given rule, incorporating both numerical and categorical attributes.
    For numerical attributes, it is based on the normalized range; for categorical, based on inverse frequency
    in the filtered dataset.

    Args:
        df (pd.DataFrame): The dataset containing the transactions (TRANSACTION DATABASE).
        features (dict): A dictionary of feature metadata for the dataset.
        antecedents (list): A list of dictionaries defining the antecedent conditions.
        consequents (list): A list of dictionaries defining the consequent conditions.
        start (int or datetime): The start of the interval (if use_interval is True) or timestamp range.
        end (int or datetime): The end of the interval (if use_interval is True) or timestamp range.
        use_interval (bool): Whether to filter by 'interval' (True) or 'timestamp' (False) for time-based filtering.

    Returns:
        float: The amplitude metric value, normalized between 0 and 1.
    """

    # Filter the dataframe based on time or interval
    if use_interval:
        df_filtered = df[(df['interval'] >= start) & (df['interval'] <= end)]
    else:
        df_filtered = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

    total_metric = 0.0
    total_attributes = 0

    # Combine antecedents and consequents
    rule_parts = antecedents + consequents

    for feature in rule_parts:
        feature_name = feature['feature']
        feature_type = feature['type']

        if feature_type == 'Numerical':
            border1 = feature['border1']
            border2 = feature['border2']

            if feature_name in df_filtered.columns and not df_filtered[feature_name].empty:
                feature_min = df_filtered[feature_name].min()
                feature_max = df_filtered[feature_name].max()
            else:
                continue  # Skip if feature is missing or empty

            if feature_max != feature_min:
                normalized_range = (border2 - border1) / (feature_max - feature_min)
            else:
                normalized_range = 0.0

            total_metric += (1 - normalized_range)
            total_attributes += 1

        elif feature_type == 'Categorical':
            value = feature['category']
            if feature_name in df_filtered.columns and not df_filtered[feature_name].empty:
                value_count = df_filtered[feature_name].value_counts(normalize=True).get(value, 0.0)
                inverse_frequency = 1.0 - value_count
                total_metric += inverse_frequency
                total_attributes += 1

    if total_attributes == 0:
        return 0.0

    amplitude_metric = total_metric / total_attributes
    return amplitude_metric

def calculate_coverage_metric(df, conditions, start, end, use_interval):
    df_filtered = df[(df['interval'] >= start) & (df['interval'] <= end)] if use_interval else \
                  df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

    mask = np.ones(len(df_filtered), dtype=bool)
    for cond in conditions:
        feature = cond['feature']
        if cond['type'].lower() == 'numerical':
            mask &= (df_filtered[feature] >= cond['border1']) & (df_filtered[feature] <= cond['border2'])
        elif cond['type'].lower() == 'categorical':
            mask &= (df_filtered[feature] == cond['category'])

    coverage = mask.sum() / len(df_filtered) if len(df_filtered) > 0 else 0.0
    return coverage


def calculate_fitness(supp, conf, incl, ampl, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
    """
    Calculate the fitness score of a rule using the weighted sum of support, confidence, inclusion and amplitude.

    The fitness function is used to evaluate how good a particular rule is, based on its support, confidence,
    and inclusion metrics. The function allows weighting of each metric through the alpha, beta, and delta parameters.

    Args:
        supp (float): The support value of the rule.
        conf (float): The confidence value of the rule.
        incl (float): The inclusion value of the rule.
        alpha (float): Weight for the support metric.
        beta (float): Weight for the confidence metric.
        gamma (float): Weight for the inclusion metric.
        delta (float): Weight for the amplitude metric.

    Returns:
        float: The fitness score, normalized between 0 and 1.
    """
    return ((alpha * supp) + (beta * conf) + (gamma * incl) + (delta * ampl)) / 4
