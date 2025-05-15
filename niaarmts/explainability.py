import numpy as np
import matplotlib.pyplot as plt
from niaarmts.metrics import (
    calculate_support,
    calculate_confidence,
    calculate_inclusion_metric,
    calculate_amplitude_metric,
    calculate_coverage_metric
)

# for showing on plot in rectangle
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
    part_name="Antecedent"
):
    contributions = []

    df_filtered = df[(df['interval'] >= start) & (df['interval'] <= end)] if use_interval else \
                  df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

    for attr in conditions:
        single_condition = [attr]
        feature_name = attr['feature']
        feature_type = attr['type'].lower()

        print ("Feature_name: ", feature_name)
        # Coverage
        coverage = calculate_coverage_metric(df, single_condition, start, end, use_interval)
        print ("Coverage: ", coverage)
        # Amplitude
        if feature_type == 'numerical':
            if feature_name in df_filtered.columns:
                feature_min = df_filtered[feature_name].min()
                feature_max = df_filtered[feature_name].max()
                print ("Feature_max: ", feature_max)
                print ("Feature_min: ", feature_min)
                if feature_max != feature_min:
                    normalized_range = (attr['border2'] - attr['border1']) / (feature_max - feature_min)
                    print ("normalized_range: ", normalized_range)
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
                print ("Value count categorical: ", value_count)
            else:
                amplitude = 0.0
        else:
            amplitude = 0.0

        print ("Amplitude: ", amplitude)
        # Inclusion
        inclusion = calculate_inclusion_metric(features, conditions, counterpart or [])

        # Final score: different formula for antecedent vs. consequent
        if part_name.lower() == "antecedent":
            score = 0.5 * coverage + 0.3 * inclusion + 0.2 * amplitude
        else:  # Consequent
            score = 0.5 * (1 - coverage) + 0.5 * amplitude

        #score = 0.5 * coverage + 0.3 * inclusion + 0.2 * (1 / amplitude)

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

def generate_latex_table(results):
    def part_to_latex(data, part_name):
        latex = f"\\begin{{table}}[htbp]\n\\centering\n"
        latex += f"\\caption{{{part_name} Feature Contributions}}\n"
        latex += f"\\begin{{tabular}}{{lrrrrr}}\n"
        latex += "\\toprule\n"
        latex += "Rank & Feature & Coverage & Inclusion & Amplitude & Score \\\\\n"
        latex += "\\midrule\n"

        for idx, row in enumerate(data, 1):
            latex += f"{idx} & {row['feature']} & {row['coverage']:.2f} & {row['inclusion']:.2f} & {row['amplitude']:.2f} & {row['score']:.4f} \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += f"\\label{{tab:{part_name.lower()}}}\n"
        latex += "\\end{table}\n\n"
        return latex

    latex_code = "\\documentclass{article}\n\\usepackage{booktabs}\n\\begin{document}\n\n"
    latex_code += part_to_latex(results["Antecedent"], "Antecedent")
    latex_code += part_to_latex(results["Consequent"], "Consequent")
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
    show_plot=True
):
    print("=== Explaining Antecedent ===")
    antecedent_data = _explain_rule_part(
        df, features, antecedent, counterpart=consequent,
        start=start, end=end, use_interval=use_interval,
        part_name="Antecedent"
    )

    print("\n=== Explaining Consequent ===")
    consequent_data = _explain_rule_part(
        df, features, consequent, counterpart=antecedent,
        start=start, end=end, use_interval=use_interval,
        part_name="Consequent"
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
            if not part_data:
                continue

            features = [x['feature'] for x in part_data]
            coverage_vals = [x['coverage'] * 0.5 for x in part_data]
            inclusion_vals = [x['inclusion'] * 0.3 for x in part_data]
            amplitude_vals = [x['amplitude'] * 0.2 for x in part_data]
            scores = [x['score'] for x in part_data]
            y_pos = np.arange(len(features))

            axes[idx].barh(y_pos, coverage_vals, label='Coverage (50%)')
            axes[idx].barh(y_pos, inclusion_vals, left=coverage_vals, label='Inclusion (30%)')
            axes[idx].barh(
                y_pos,
                amplitude_vals,
                left=np.array(coverage_vals) + np.array(inclusion_vals),
                label='Amplitude (20%)'
            )

            for i, score in enumerate(scores):
                axes[idx].text(score + 0.01, y_pos[i], f"{score:.2f}", va='center', fontsize=9, color='black')

            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(features)
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

    latex_code = generate_latex_table(results)
    return results, latex_code
