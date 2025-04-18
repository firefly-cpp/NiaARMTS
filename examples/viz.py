from niaarmts import Dataset
import pandas as pd
from niaarmts.visualization import NarmViz

# Load dataset
dataset = Dataset()
dataset.load_data_from_csv('datasets/september24.csv', timestamp_col='timestamp')

# Get all transactions
transactions=dataset.get_all_transactions()

viz = NarmViz(transactions)

# This is an example of association rule
rule = {
    'full_rule': [
        {'feature': 'temperature', 'type': 'Numerical', 'border1': 17.5001, 'border2': 25.6032, 'category': 'EMPTY'},
        {'feature': 'moisture', 'type': 'Numerical', 'border1': 2015.0293, 'border2': 2385.0, 'category': 'EMPTY'}
    ],
    'antecedent': [
        {'feature': 'temperature', 'type': 'Numerical', 'border1': 17.5001, 'border2': 25.6032, 'category': 'EMPTY'}
    ],
    'consequent': [
        {'feature': 'moisture', 'type': 'Numerical', 'border1': 2015.0293, 'border2': 2385.0, 'category': 'EMPTY'}
    ],
    'fitness': 0.6324151935703214,
    'support': 0.8240493662441628,
    'confidence': 0.9887932759655793,
    'inclusion': 0.5,
    'amplitude': 0.4956648490696878,
    'start': pd.Timestamp('2024-09-11 08:39:53'),
    'end': pd.Timestamp('2024-09-11 08:42:53') # 01:19:33
}

viz.visualize_rule(
    rule_entry=rule,
    interval_data=False,
    show_all_features=False,
    plot_full_data=False,
    save_path="plot.png",
    pdf_path="plot.pdf",
    describe=True
)
