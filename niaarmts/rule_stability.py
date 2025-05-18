import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from niaarmts.metrics import calculate_support, calculate_confidence

def rule_quality(df, antecedent, consequent, start, end, use_interval=False):
    """
    Compute the average of support and confidence (quality_score) for a rule within a given time interval.
    """
    support = calculate_support(df, antecedent, consequent, start, end, use_interval=use_interval)
    confidence = calculate_confidence(df, antecedent, consequent, start, end, use_interval=use_interval)
    return 0.5 * (support + confidence)

def calculate_stability_score(df, antecedent, consequent, start, end, delta=pd.Timedelta(hours=6), use_interval=False):
    """
    Calculate the stability score of a rule over three intervals: I^-, I, I^+

    Args:
        df (pd.DataFrame): Transaction dataset.
        antecedent (list): Antecedent of the rule.
        consequent (list): Consequent of the rule.
        start (pd.Timestamp): Start timestamp of the original rule interval.
        end (pd.Timestamp): End timestamp of the original rule interval.
        delta (pd.Timedelta): Time offset for generating I^- and I^+ intervals.
        use_interval (bool): Whether to use 'interval' column instead of 'timestamp'.

    Returns:
        float: Stability score in the range [0.0, 1.0]. Higher means more stable.
    """
    start_minus = start - delta
    end_minus = end - delta
    start_plus = start + delta
    end_plus = end + delta

    score_current = rule_quality(df, antecedent, consequent, start, end, use_interval)
    score_minus = rule_quality(df, antecedent, consequent, start_minus, end_minus, use_interval)
    score_plus = rule_quality(df, antecedent, consequent, start_plus, end_plus, use_interval)

    score = np.sqrt((score_minus - score_current)**2 + (score_plus - score_current)**2)
    return round(score, 4)

def plot_rule_stability(df, antecedent, consequent, start, end, delta=pd.Timedelta(hours=6), use_interval=False):
    """
    Plot the stability of a rule over three time intervals (I⁻, I, I⁺), showing support, confidence, and quality score.

    Args:
        df (pd.DataFrame): Transaction dataset.
        antecedent (list): Antecedent of the rule.
        consequent (list): Consequent of the rule.
        start (pd.Timestamp): Start of the original time window.
        end (pd.Timestamp): End of the original time window.
        delta (pd.Timedelta): Time offset for I⁻ and I⁺.
        use_interval (bool): Whether to use the 'interval' column instead of 'timestamp'.
    """
    start_minus, end_minus = start - delta, end - delta
    start_plus, end_plus = start + delta, end + delta

    intervals = [("I⁻", start_minus, end_minus), ("I", start, end), ("I⁺", start_plus, end_plus)]
    metrics = []

    print("\n[INFO] Interval Diagnostics:")
    for label, s, e in intervals:
        if use_interval:
            count = len(df[(df['interval'] >= s) & (df['interval'] <= e)])
        else:
            count = len(df[(df['timestamp'] >= s) & (df['timestamp'] <= e)])
        print(f"{label} | Start: {s} | End: {e} | Rows in interval: {count}")

    for label, s, e in intervals:
        support = calculate_support(df, antecedent, consequent, s, e, use_interval)
        confidence = calculate_confidence(df, antecedent, consequent, s, e, use_interval)
        quality = 0.5 * (support + confidence)
        print(f"{label} | Support: {support:.4f} | Confidence: {confidence:.4f} | Quality: {quality:.4f}")
        metrics.append((label, support, confidence, quality))

    stability_score = calculate_stability_score(df, antecedent, consequent, start, end, delta, use_interval)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle("Rule Stability Visualization", fontsize=14, fontweight='bold')

    # --- Timeline Plot ---
    axs[0].set_title("Time Intervals")
    for idx, (label, s, e) in enumerate(intervals):
        axs[0].barh(0, (e - s).total_seconds(), left=mdates.date2num(s), height=0.5, label=label)
        axs[0].text(mdates.date2num(s), 0.1, f"{label}", fontsize=10, color='black')

    axs[0].set_yticks([])

    min_time = min([s for _, s, _ in intervals])
    max_time = max([e for _, _, e in intervals])
    axs[0].set_xlim(mdates.date2num(min_time), mdates.date2num(max_time))

    axs[0].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    axs[0].legend()

    # --- Bar Chart for Metrics ---
    labels = [label for label, _, _, _ in metrics]
    support_vals = [m[1] for m in metrics]
    confidence_vals = [m[2] for m in metrics]
    quality_vals = [m[3] for m in metrics]

    x = range(len(labels))
    width = 0.25

    axs[1].bar([i - width for i in x], support_vals, width=width, label='Support')
    axs[1].bar(x, confidence_vals, width=width, label='Confidence')
    axs[1].bar([i + width for i in x], quality_vals, width=width, label='Quality')

    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels)
    axs[1].set_ylim(0, 1.05)
    axs[1].set_ylabel("Metric Value")
    axs[1].legend()
    axs[1].set_title(f"Stability Score: {stability_score:.3f}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def create_latex_table(df, antecedent, consequent, start, end, delta=pd.Timedelta(hours=6),
                       use_interval=False, file_path="rule_stability_table.tex"):
    """
    Generate a LaTeX table showing support, confidence, and quality scores over three intervals (I⁻, I, I⁺),
    and write it to a .tex file.

    Args:
        df (pd.DataFrame): The dataset.
        antecedent (list): Antecedent of the rule.
        consequent (list): Consequent of the rule.
        start (pd.Timestamp): Start of the main interval.
        end (pd.Timestamp): End of the main interval.
        delta (pd.Timedelta): Time delta for offset intervals.
        use_interval (bool): Whether to use 'interval' instead of 'timestamp'.
        file_path (str): Path to save the LaTeX file.
    """
    start_minus, end_minus = start - delta, end - delta
    start_plus, end_plus = start + delta, end + delta

    intervals = [("I$^{-}$", start_minus, end_minus), ("I", start, end), ("I$^{+}$", start_plus, end_plus)]
    rows = []

    for label, s, e in intervals:
        support = calculate_support(df, antecedent, consequent, s, e, use_interval)
        confidence = calculate_confidence(df, antecedent, consequent, s, e, use_interval)
        quality = 0.5 * (support + confidence)
        rows.append((label, s.strftime('%Y-%m-%d %H:%M'), e.strftime('%Y-%m-%d %H:%M'), support, confidence, quality))

    # Format rule in LaTeX
    antecedent_str = ", ".join([f"\\texttt{{{item}}}" for item in antecedent])
    consequent_str = ", ".join([f"\\texttt{{{item}}}" for item in consequent])
    rule_latex = f"\\textbf{{Rule:}} $\\{{{antecedent_str}\\}} \\Rightarrow \\{{{consequent_str}\\}}$\n\n"

    # Build LaTeX table string
    table = "\\begin{tabular}{lccccc}\n"
    table += "\\toprule\n"
    table += "Interval & Start & End & Support & Confidence & Quality \\\\\n"
    table += "\\midrule\n"
    for label, s, e, sup, conf, qual in rows:
        table += f"{label} & {s} & {e} & {sup:.4f} & {conf:.4f} & {qual:.4f} \\\\\n"
    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"

    # Write to file
    with open(file_path, "w") as f:
        f.write(rule_latex)
        f.write("\n")
        f.write(table)

    print(f"[INFO] LaTeX rule and table written to: {file_path}")

