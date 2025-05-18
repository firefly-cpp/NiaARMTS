from niaarmts.rule_stability import calculate_stability_score, plot_rule_stability
from niaarmts import Dataset
import pandas as pd

dataset = Dataset()
dataset.load_data_from_csv('datasets/september24.csv', timestamp_col='timestamp')
transactions = dataset.get_all_transactions()

antecedent = [
    {'feature': 'temperature', 'type': 'Numerical', 'border1': 22.3382, 'border2': 24.8301, 'category': 'EMPTY'}
]

consequent = [
   {'feature': 'light', 'type': 'Numerical', 'border1': 0.0, 'border2': 871.1136, 'category': 'EMPTY'}
]

start = pd.Timestamp("2024-09-18 13:18:07")
end=pd.Timestamp("2024-09-20 13:18:07")

stability = calculate_stability_score(
    df=transactions,
    antecedent=antecedent,
    consequent=consequent,
    start=start,
    end=end,
    delta=pd.Timedelta(hours=48))

print("Stability score:", stability)

# --- Plot Rule Stability ---
plot_rule_stability(transactions, antecedent, consequent, start, end, pd.Timedelta(hours=48))
