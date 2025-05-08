from niaarmts.metrics import calculate_support, calculate_confidence, calculate_inclusion_metric, calculate_amplitude_metric, calculate_coverage_metric
import matplotlib.pyplot as plt

def explain_rule_features(
    df,
    features,
    conditions,
    counterpart=None,
    start=0,
    end=0,
    use_interval=False,
    show_plot=True,
    part_name="Antecedent"
):
    """
    Rank features by their contribution using coverage, inclusion, and amplitude.

    Args:
        df (pd.DataFrame): Transactions.
        features (dict): Feature metadata.
        conditions (list): Feature conditions (antecedent or consequent).
        counterpart (list): The opposite side of the rule (optional, for inclusion).
        start (int or datetime): Start of time/interval range.
        end (int or datetime): End of time/interval range.
        use_interval (bool): Whether to use 'interval' or 'timestamp'.
        part_name (str): Label for the rule part ("Antecedent" or "Consequent").

    Returns:
        list: Ranked list of (feature_name, importance_score).
    """
    contributions = []

    for attr in conditions:
        single_condition = [attr]
        feature_name = attr['feature']

        # Compute coverage
        coverage = calculate_coverage_metric(df, single_condition, start, end, use_interval)

        # Compute amplitude
        if attr['type'] == 'Numerical':
            df_filtered = df[(df['interval'] >= start) & (df['interval'] <= end)] if use_interval else \
                          df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
            feature_min = df_filtered[feature_name].min()
            feature_max = df_filtered[feature_name].max()
            if feature_max != feature_min:
                normalized_range = (attr['border2'] - attr['border1']) / (feature_max - feature_min)
                amplitude_score = 1 - normalized_range
            else:
                amplitude_score = 1.0  # No variation
        else:
            amplitude_score = 0.0

        # Inclusion (used for both antecedents and consequents)
        incl = calculate_inclusion_metric(features, conditions, counterpart or [])

        # Final score
        score = 0.5 * coverage + 0.3 * incl + 0.2 * amplitude_score ## revise TODO
        contributions.append((feature_name, score))

    contributions.sort(key=lambda x: x[1], reverse=True)

    # Display results
    print(f"\n Critical {part_name} Attributes:")
    for i, (name, score) in enumerate(contributions, 1):
        print(f"{i}. {name}: {score:.4f}")

    return contributions
