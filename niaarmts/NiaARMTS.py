import numpy as np
import pandas as pd
import json
from niapy.problems import Problem
from niaarmts.rule import build_rule
from niaarmts.metrics import calculate_support, calculate_confidence, calculate_inclusion_metric, calculate_amplitude_metric, calculate_fitness

class NiaARMTS(Problem):
    def __init__(
        self,
        dimension,
        lower,
        upper,
        features,
        transactions,
        interval,
        alpha,
        beta,
        gamma,
        delta
    ):
        """
        Initialize instance of NiaARMTS.

        Arguments:
            dimension (int): Dimension of the optimization problem.
            lower (float): Lower bound of the solution space.
            upper (float): Upper bound of the solution space.
            features (dict): A dictionary of feature metadata.
            transactions (df): Transaction data in data frame.
            interval (str): 'true' if dealing with interval data, 'false' if pure time series.
            alpha (float): Weight for support in fitness function.
            beta (float): Weight for confidence in fitness function.
            gamma (float): Weight for inclusion in fitness function.
            delta (float): Weight for amplitude in fitness function.

        Raises:
            KeyError: Timestamp column is required when interval is set to false.
        """
        if interval == 'false' and 'timestamp' not in transactions:
            raise KeyError('Timestamp column is required when interval is set to false.')

        self.dim = dimension
        self.features = features
        self.transactions = transactions
        self.interval = interval  # 'true' if we deal with interval data, 'false' if we deal with pure time series data
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # Archive for storing all unique rules with fitness > 0.0
        self.rule_archive = []

        # Store the best fitness value
        self.best_fitness = np.NINF
        super().__init__(dimension, lower, upper)

    # NiaPy evaluation function
    def _evaluate(self, solution):
        # get cut point
        cut_point_val = solution[-1]
        solution = np.delete(solution, -1)

        if self.interval == 'true':
            interval = solution[-1]
            solution = np.delete(solution, -1)
            curr_interval = self.map_to_interval(interval)

            # if we deal only with one interval start == end
            start = curr_interval
            end = curr_interval

        else:  # if time series
            upper = solution[-1]
            solution = np.delete(solution, -1)
            lower = solution[-1]
            solution = np.delete(solution, -1)
            min_interval, max_interval = self.map_to_ts(lower, upper)

            # Get time bounds for filtering transactions
            # Fetching the actual timestamps from the dataset
            start = self.transactions['timestamp'].iloc[min_interval]
            end = self.transactions['timestamp'].iloc[max_interval]

        # Step 1: Build the rules using the solution and features
        rule = build_rule(solution, self.features, is_time_series=(self.interval == "false"))

        # Step 2: Split the rule into antecedents and consequents based on the cut point
        cut = self.cut_point(cut_point_val, len(rule))
        antecedent = rule[:cut]  # From the start to the 'cut' index (not inclusive)
        consequent = rule[cut:]  # From 'cut' index (inclusive) to the end of the array

        # Step 3: Calculate support, confidence, and other arbitrary metrics for the rules
        if len(antecedent) > 0 and len(consequent) > 0:
            # Calculate support and confidence always

            if self.interval != "true":
                support = calculate_support(self.transactions, antecedent, consequent, start, end)
                confidence = calculate_confidence(self.transactions, antecedent, consequent, start, end)

            else:
                support = calculate_support(self.transactions, antecedent, consequent, start, end, use_interval=True)
                confidence = calculate_confidence(self.transactions, antecedent, consequent, start, end, use_interval=True)

            inclusion = 0.0
            if self.gamma > 0.0:
                inclusion = calculate_inclusion_metric(self.features, antecedent, consequent)

            amplitude = 0.0
            if self.delta > 0.0:
                amplitude = calculate_amplitude_metric(self.features, antecedent, consequent)

            # Step 4: Calculate the fitness of the rules using weights for support, confidence, inclusion and amplitude
            fitness = calculate_fitness(support, confidence, inclusion, amplitude)

            # Step 5: Store the rule if it has fitness > 0 and it's unique
            if fitness > 0:
                self.add_rule_to_archive(rule, antecedent, consequent, fitness, start, end, support, confidence, inclusion, amplitude)

            return fitness
        else:
            return 0.0

    def add_rule_to_archive(self, full_rule, antecedent, consequent, fitness, start, end, support, confidence, inclusion, amplitude):
        """
        Add the rule to the archive if its fitness is greater than zero and it's not already present.
        Args:
            full_rule (list): The full rule generated from the solution.
            antecedent (list): The antecedent part of the rule.
            consequent (list): The consequent part of the rule.
            fitness (float): The fitness value of the rule.
            start (timestamp): The start timestamp for the rule.
            end (timestamp): The end timestamp for the rule.
            support (float): Support value for the rule.
            confidence (float): Confidence value for the rule.
            inclusion (float): Inclusion metric for the rule.
            amplitude (float): Amplitude metric for the rule.
        """
        rule_repr = self.rule_representation(full_rule)
        # Check if the rule is already in the archive (by its string representation)
        if rule_repr not in [self.rule_representation(r['full_rule']) for r in self.rule_archive]:
            # Add the rule, its antecedent, consequent, fitness, support, confidence, inclusion, amplitude and timestamps to the archive
            self.rule_archive.append({
                'full_rule': full_rule,
                'antecedent': antecedent,
                'consequent': consequent,
                'fitness': fitness,
                'support': support,
                'confidence': confidence,
                'inclusion': inclusion,
                'amplitude': amplitude,
                'start': start,
                'end': end
            })

    def rule_representation(self, rule):
        """
        Generate a string representation of a rule for easier comparison and to avoid duplicates.
        Args:
            rule (list): The rule to represent as a string.

        Returns:
            str: A string representation of the rule.
        """
        return str(sorted([str(attr) for attr in rule]))

    def cut_point(self, sol, num_attr):
        """
        Calculate cut point based on the solution and the number of attributes.
        """
        cut = int(np.trunc(sol * num_attr))

        # Ensure cut is at least 1
        if cut == 0:
            cut = 1

        # Ensure cut does not exceed num_attr - 2
        if cut > (num_attr - 1):
            cut = num_attr - 2

        return cut

    def map_to_interval(self, val):
        min_interval = self.transactions['interval'].min()
        max_interval = self.transactions['interval'].max()

        if not 0.0 <= val <= 1.0:
            raise ValueError("The random solution must be between 0 and 1.")

        curr_interval = int(min_interval + (max_interval - min_interval) * val)

        return curr_interval

    def map_to_ts(self, lower, upper):
        total_transactions = len(self.transactions) - 1
        low = int(total_transactions * lower)
        up = int(total_transactions * upper)

        if low > up:
            low, up = up, low

        return low, up

    def get_rule_archive(self):
        """
        Return the archive of all valid rules (those with fitness > 0), sorted by fitness in descending order.
        """
        # Sort the archive by fitness in descending order
        self.rule_archive.sort(key=lambda x: x['fitness'], reverse=True)
        return self.rule_archive

    def save_rules_to_csv(self, file_path):
        """
        Save the archived rules to a CSV file, sorted by fitness (descending).

        Args:
            file_path (str): The path to save the CSV file.
        """
        # Ensure archive is sorted by fitness
        self.get_rule_archive()

        # Prepare data for the CSV
        rule_data = []
        for entry in self.rule_archive:
            rule_info = {
                'fitness': entry['fitness'],
                'support': entry['support'],
                'confidence': entry['confidence'],
                'inclusion': entry['inclusion'],
                'amplitude': entry['amplitude'],
                'antecedent': str(entry['antecedent']),
                'consequent': str(entry['consequent']),
                'start_timestamp': entry['start'],
                'end_timestamp': entry['end']
            }
            rule_data.append(rule_info)

        # Create a DataFrame and save to CSV
        df = pd.DataFrame(rule_data)
        df.to_csv(file_path, index=False)
        print(f"Rules saved to {file_path}.")

    def save_rules_to_json(self, file_path):
        """
        Save the archived rules to a JSON file, sorted by fitness (descending).

        Args:
            file_path (str): The path to save the JSON file.
        """
        # Ensure archive is sorted by fitness
        self.get_rule_archive()

        # Prepare the archive as a JSON-friendly format
        archive_dict = {'rules': []}
        for entry in self.rule_archive:
            archive_dict['rules'].append({
                'fitness': entry['fitness'],
                'support': entry['support'],
                'confidence': entry['confidence'],
                'inclusion': entry['inclusion'],
                'antecedent': entry['antecedent'],
                'amplitude': entry['amplitude'],
                'consequent': entry['consequent'],
                'start_timestamp': str(entry['start']),
                'end_timestamp': str(entry['end'])
            })

        # Save to JSON
        with open(file_path, 'w') as f:
            json.dump(archive_dict, f, indent=4)
        print(f"Rules saved to {file_path}.")
