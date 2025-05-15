from niaarmts.explainability import explain_rule
from niaarmts import Dataset
import pandas as pd

# Load dataset
dataset = Dataset()
dataset.load_data_from_csv('datasets/ts.csv', timestamp_col='timestamp')
transactions = dataset.get_all_transactions()

# Rule definition
antecedent = [
    {'feature': 'weather', 'type': 'Categorical', 'border1': 1.0, 'border2': 1.0, 'category': 'clouds'},
    {'feature': 'light', 'type': 'numerical', 'border1': 0.0, 'border2': 7.0, 'category': 'EMPTY'},
    {'feature': 'moisture', 'type': 'numerical', 'border1': 2340.0, 'border2': 2341.0, 'category': 'EMPTY'}
]

consequent = [
    {'feature': 'temperature', 'type': 'Numerical', 'border1': 28.3, 'border2': 28.4, 'category': 'EMPTY'},
    {'feature': 'humidity', 'type': 'numerical', 'border1': 60.2, 'border2': 60.3, 'category': 'EMPTY'}
]

# Run explanation and get LaTeX code
results, latex_code = explain_rule(
    df=transactions,
    features=dataset.get_all_features_with_metadata(),
    antecedent=antecedent,
    consequent=consequent,
    start=pd.Timestamp("2024-09-08 20:16:21"),
    end=pd.Timestamp("2024-09-08 20:17:51"),
    use_interval=False
)

# Save LaTeX output
with open("rule_explanation_tables.tex", "w") as f:
    f.write(latex_code)

print("LaTeX table saved to 'rule_explanation_tables.tex'")
