import pandas as pd
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class NarmViz:
    def __init__(self, transactions, interval_column='interval', timestamp_column='timestamp'):
        self.transactions = transactions
        self.interval_column = interval_column
        self.timestamp_column = timestamp_column

    def describe_rule(self, rule):
        def format_condition(attr):
            if attr["type"] == "Categorical":
                return f"{attr['feature']} == '{attr['category']}'"
            else:
                return f"{attr['feature']} ∈ [{attr['border1']:.4f}, {attr['border2']:.4f}]"

        antecedents = [format_condition(a) for a in rule["antecedent"]]
        consequents = [format_condition(c) for c in rule["consequent"]]

        description = "IF " + " AND ".join(antecedents)
        description += "\nTHEN " + " AND ".join(consequents)

        description += f"\n\nSupport: {rule['support']:.3f}"
        description += f"\nConfidence: {rule['confidence']:.3f}"
        description += f"\nFitness: {rule['fitness']:.3f}"

        if "inclusion" in rule:
            description += f"\nInclusion: {rule['inclusion']:.3f}"
        if "amplitude" in rule:
            description += f"\nAmplitude: {rule['amplitude']:.3f}"

        if "start" in rule and "end" in rule:
            description += f"\nTime window: {rule['start']} → {rule['end']}"

        return description

    def visualize_rule(self, rule_entry, interval_data=True, show_all_features=False,
                       plot_full_data=True, save_path=None, pdf_path=None, show=True,
                       describe=False):
        df = self.transactions.copy()

        if not plot_full_data:
            if interval_data and self.interval_column in df.columns:
                df = df[df[self.interval_column] == rule_entry['start']]
            elif not interval_data and self.timestamp_column in df.columns:
                df = df[(df[self.timestamp_column] >= rule_entry['start']) &
                        (df[self.timestamp_column] <= rule_entry['end'])]

        plotted_features = set()
        plot_tasks = []

        def is_categorical(series):
            return pd.api.types.is_categorical_dtype(series) or series.dtype == object

        def draw_numerical(ax, series, attr, is_antecedent=None):
            if self.timestamp_column in df.columns and len(df[self.timestamp_column]) == len(series):
                x = df[self.timestamp_column].values
            else:
                x = np.arange(len(series))

            y = series.values
            color = 'purple' if is_antecedent else 'green' if is_antecedent is not None else 'gray'
            title_suffix = "(Antecedent)" if is_antecedent else "(Consequent)" if is_antecedent is not None else ""

            # TODO - higlight colors
            if is_antecedent is not None and attr['border1'] is not None and attr['border2'] is not None:
                highlight_color = '#FF69B4' if is_antecedent else '#00CED1'
                ax.axhspan(attr['border1'], attr['border2'], color=highlight_color, alpha=0.3)

            ax.scatter(x, y, s=10, color=color)
            ax.set_title(f"{attr['feature']} {title_suffix}".strip(), fontsize=9)
            ax.set_xlabel("Time" if isinstance(x[0], (np.datetime64, pd.Timestamp)) else "Index", fontsize=8)
            ax.set_ylabel("Value", fontsize=8)
            ax.tick_params(labelsize=6)

            if isinstance(x[0], (np.datetime64, pd.Timestamp)):
                ax.tick_params(axis='x', rotation=30)

        def draw_mosaic(ax, series, attr, is_antecedent=None):
            counts = series.value_counts(normalize=True).sort_index()
            labels = counts.index.tolist()
            values = counts.values

            x = 0
            for label, val in zip(labels, values):
                if is_antecedent and label == attr.get('category'):
                    color = '#FF69B4'
                elif is_antecedent is False and label == attr.get('category'):
                    color = '#00CED1'
                else:
                    color = 'grey'
                rect = patches.Rectangle((x, 0), val, 1, color=color)
                ax.add_patch(rect)
                ax.text(x + val / 2, 0.5, str(label), ha='center', va='center', fontsize=6)
                x += val

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            label = attr['feature']
            suffix = "(Antecedent)" if is_antecedent else "(Consequent)" if is_antecedent is not None else ""
            ax.set_title(f"{label} {suffix}".strip(), fontsize=9)
            ax.axis('off')

        # 1. Antecedents
        for attr in rule_entry['antecedent']:
            name = attr['feature']
            if name not in df.columns or name in ['timestamp', 'interval']:
                continue
            series = df[name]
            plotted_features.add(name)
            plot_tasks.append((series, attr, True))

        # 2. Consequents
        for attr in rule_entry['consequent']:
            name = attr['feature']
            if name in plotted_features or name not in df.columns or name in ['timestamp', 'interval']:
                continue
            series = df[name]
            plotted_features.add(name)
            plot_tasks.append((series, attr, False))

        # 3. Other features
        if show_all_features:
            for col in df.columns:
                if col in plotted_features or col in ['timestamp', 'interval']:
                    continue
                series = df[col]
                dummy_attr = {
                    'feature': col,
                    'border1': None,
                    'border2': None,
                    'type': 'Categorical' if is_categorical(series) else 'Numerical',
                    'category': None
                }
                plot_tasks.append((series, dummy_attr, None))

        # 4. Layout and plotting
        n = len(plot_tasks)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
        axs = axs.flatten() if n > 1 else [axs]

        for i, (series, attr, is_antecedent) in enumerate(plot_tasks):
            ax = axs[i]
            if attr['type'] == 'Categorical':
                draw_mosaic(ax, series, attr, is_antecedent)
            else:
                draw_numerical(ax, series, attr, is_antecedent)

        for j in range(len(plot_tasks), len(axs)):
            axs[j].axis('off')

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        if pdf_path:
            fig.savefig(pdf_path, bbox_inches='tight', format='pdf')
            print(f"Visualization saved as PDF to {pdf_path}")

        if show:
            plt.show()

        if describe:
            print("\n" + self.describe_rule(rule_entry))
