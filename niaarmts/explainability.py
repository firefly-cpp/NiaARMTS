import numpy as np
import matplotlib.pyplot as plt
from niaarmts.metrics import (
    calculate_support,
    calculate_confidence,
    calculate_inclusion_metric,
    calculate_amplitude_metric,
    calculate_coverage_metric
)

def format_rule(antecedent, consequent):
    def format_part(part):
        return " ∧ ".join([f"{c['feature']} ∈ [{c['border1']}, {c['border2']}]" for c in part])
    return f"{format_part(antecedent)} ⇒ {format_part(consequent)}"

def _explain_rule_part(
    df,
    features,
    conditions,
    counterpart=None,
    start=0,
    end=0,
    use_interval=False,
    part_name="Antecedent",
    weights=None
):
    contributions = []

    df_filtered = df[(df['interval'] >= start) & (df['interval'] <= end)] if use_interval else \
                  df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

    if weights is None:
        weights = {}
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}

    for attr in conditions:
        single_condition = [attr]
        feature_name = attr['feature']
        feature_type = attr['type'].lower()

        coverage = calculate_coverage_metric(df, single_condition, start, end, use_interval) if 'coverage' in weights else None
        inclusion = calculate_inclusion_metric(features, conditions, counterpart or []) if 'inclusion' in weights else None

        amplitude = None
        if 'amplitude' in weights:
            if feature_type == 'numerical':
                if feature_name in df_filtered.columns:
                    feature_min = df_filtered[feature_name].min()
                    feature_max = df_filtered[feature_name].max()
                    if feature_max != feature_min:
                        normalized_range = (attr['border2'] - attr['border1']) / (feature_max - feature_min)
                        amplitude = 1 - normalized_range
                    else:
                        amplitude = 1.0
                else:
                    amplitude = 0.0
            elif feature_type == 'categorical':
                value = attr['category']
                if feature_name in df_filtered.columns and not df_filtered[feature_name].empty:
                    value_count = df_filtered[feature_name].value_counts(normalize=True).get(value, 0.0)
                    amplitude = 1.0 - value_count
                else:
                    amplitude = 0.0
            else:
                amplitude = 0.0

        # Score calculation
        score = 0.0
        if part_name.lower() == "antecedent":
            score = sum([
                weights.get('coverage', 0) * (coverage if coverage is not None else 0),
                weights.get('inclusion', 0) * (inclusion if inclusion is not None else 0),
                weights.get('amplitude', 0) * (amplitude if amplitude is not None else 0)
            ])
        else:  # Consequence
            score = sum([
                weights.get('coverage', 0) * ((1 - coverage) if coverage is not None else 0),
                weights.get('amplitude', 0) * (amplitude if amplitude is not None else 0)
            ])

        contributions.append({
            'feature': feature_name,
            'coverage': coverage,
            'inclusion': inclusion,
            'amplitude': amplitude,
            'score': score
        })

    contributions.sort(key=lambda x: x['score'], reverse=True)

    print(f"\nCritical {part_name} Attributes:")
    for i, c in enumerate(contributions, 1):
        print(f"{i}. {c['feature']}: {c['score']:.4f}")

    return contributions

def generate_latex_table(results, antecedent_weights, consequent_weights, antecedent, consequent):
    def part_to_latex(data, part_name, weights):
        used_metrics = [m for m in ['coverage', 'inclusion', 'amplitude'] if m in weights]
        latex = f"\\begin{{table}}[htbp]\n\\centering\n"
        latex += f"\\caption{{{part_name} Feature Contributions}}\n"
        latex += f"\\begin{{tabular}}{{l{'r' * (len(used_metrics) + 2)}}}\n"
        latex += "\\toprule\n"
        headers = ["Rank", "Feature"] + [m.title() for m in used_metrics] + ["Score"]
        latex += " & ".join(headers) + " \\\\\n\\midrule\n"

        for idx, row in enumerate(data, 1):
            row_vals = [f"{idx}", row['feature']] + \
                       [f"{row[m]:.2f}" if row[m] is not None else "-" for m in used_metrics] + \
                       [f"{row['score']:.4f}"]
            latex += " & ".join(row_vals) + " \\\\\n"

        latex += "\\bottomrule\n\\end{tabular}\n"
        latex += f"\\label{{{{tab:{part_name.lower()}}}}}\n\\end{{table}}\n\n"
        return latex

    def format_condition_latex(part):
        parts = []
        for cond in part:
            feature = cond['feature']
            b1 = cond['border1']
            b2 = cond['border2']
            if cond['type'].lower() == 'categorical':
                val = cond['category']
                parts.append(f"{feature} = \\texttt{{{val}}}")
            else:
                parts.append(f"{b1:.2f} \\leq {feature} \\leq {b2:.2f}")
        return " \\wedge ".join(parts)

    rule_latex = f"\\[\n{format_condition_latex(antecedent)} \\Rightarrow {format_condition_latex(consequent)}\n\\]\n"

    latex_code = "\\documentclass{article}\n\\usepackage{booktabs}\n\\usepackage{amsmath}\n\\begin{document}\n\n"
    latex_code += "\\section*{Explained Rule}\n"
    latex_code += rule_latex + "\n"
    latex_code += part_to_latex(results["Antecedent"], "Antecedent", antecedent_weights)
    latex_code += part_to_latex(results["Consequent"], "Consequent", consequent_weights)
    latex_code += "\\end{document}"
    return latex_code

def explain_rule(
    df,
    features,
    antecedent,
    consequent,
    start=0,
    end=0,
    use_interval=False,
    show_plot=True,
    antecedent_weights={'coverage': 0.5, 'inclusion': 0.3, 'amplitude': 0.2},
    consequent_weights={'coverage': 0.5, 'amplitude': 0.5}
):
    print("=== Explaining Antecedent ===")
    antecedent_data = _explain_rule_part(
        df, features, antecedent, counterpart=consequent,
        start=start, end=end, use_interval=use_interval,
        part_name="Antecedent", weights=antecedent_weights
    )

    print("\n=== Explaining Consequent ===")
    consequent_data = _explain_rule_part(
        df, features, consequent, counterpart=antecedent,
        start=start, end=end, use_interval=use_interval,
        part_name="Consequent", weights=consequent_weights
    )

    results = {
        "Antecedent": antecedent_data,
        "Consequent": consequent_data
    }

    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("Feature Metric Contributions with Final Scores", fontsize=16)

        for idx, part_name in enumerate(["Antecedent", "Consequent"]):
            part_data = results[part_name]

            # Simple fix: invert coverage for plotting in Consequent part
            if part_name == "Consequent" and 'coverage' in weights:
                for x in part_data:
                    if x['coverage'] is not None:
                        x['coverage'] = 1 - x['coverage']

            weights = antecedent_weights if part_name == "Antecedent" else consequent_weights
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items() if v > 0}

            if not part_data:
                continue

            features_list = [x['feature'] for x in part_data]
            y_pos = np.arange(len(features_list))
            left = np.zeros(len(features_list))

            for metric in ['coverage', 'inclusion', 'amplitude']:
                if metric in weights:
                    contrib_vals = [x[metric] * weights[metric] if x[metric] is not None else 0 for x in part_data]
                    axes[idx].barh(y_pos, contrib_vals, left=left, label=f"{metric.title()} ({weights[metric]*100:.0f}%)")
                    left += contrib_vals

            for i, score in enumerate([x['score'] for x in part_data]):
                axes[idx].text(score + 0.01, y_pos[i], f"{score:.2f}", va='center', fontsize=9, color='black')

            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(features_list)
            axes[idx].invert_yaxis()
            axes[idx].set_title(part_name)
            axes[idx].set_xlabel("Importance Score")
            axes[idx].legend()
            axes[idx].grid(axis='x', linestyle='--', alpha=0.6)

        full_rule = format_rule(antecedent, consequent)
        fig.text(0.5, 0.01, full_rule,
                 ha='center', va='bottom', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.4", edgecolor='black', facecolor='#f0f0f0'))

        plt.tight_layout(rect=[0, 0.05, 1, 0.92])
        plt.show()

    latex_code = generate_latex_table(results, antecedent_weights, consequent_weights, antecedent, consequent)
    return results, latex_code
