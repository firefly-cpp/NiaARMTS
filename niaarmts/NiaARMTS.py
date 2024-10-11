import numpy as np
import pandas as pd
import json
from niapy.problems import Problem
from niaarmts.rule import build_rule

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
        delta,
        output
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
            gamma (float): Weight for amplitude in fitness function.
            delta (float): Weight for inclusion in fitness function.
            output: Output file or method for logging the results.
        """
        self.dim = dimension
        self.features = features
        self.transactions = transactions
        self.interval = interval  # 'true' if we deal with interval data, 'false' if we deal with pure time series data
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.output = output

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
            min_interval, max_interval = self.map_to_interval(interval)
        else:  # if time series
            upper = solution[-1]
            solution = np.delete(solution, -1)
            lower = solution[-1]
            solution = np.delete(solution, -1)
            min_interval, max_interval = self.map_to_ts(lower, upper)

            # Get time bounds for filtering transactions
            start = self.transactions.loc[min_interval, 'timestamp']
            end = self.transactions.loc[max_interval, 'timestamp']

        # Step 1: Build the rules using the solution and features
        rule = build_rule(solution, self.features, is_time_series=(self.interval == "false"))

        # Step 2: Split the rule into antecedents and consequents based on the cut point
        cut = self.cut_point(cut_point_val, len(rule))
        antecedent = rule[:cut]  # From the start to the 'cut' index (not inclusive)
        consequent = rule[cut:]  # From 'cut' index (inclusive) to the end of the array

        # Step 3: Calculate support, confidence, and other arbitrary metrics for the rules
        if len(antecedent) > 0 and len(consequent) > 0:
            # Calculate support and confidence always
            support = self.calculate_support(self.transactions, antecedent, start, end)
            confidence = self.calculate_confidence(self.transactions, antecedent, consequent, start, end)

            if self.gamma > 0.0:
                inclusion = self.calculate_inclusion_metric(antecedent, consequent)

            # Step 4: Calculate the fitness of the rules using weights for support, confidence, and inclusion
            fitness = self.calculate_fitness(support, confidence, inclusion)

            # Step 5: Store the rule if it has fitness > 0 and it's unique
            if fitness > 0:
                self.add_rule_to_archive(rule, antecedent, consequent, fitness, start, end, support, confidence, inclusion)

            return fitness
        else:
            return 0.0

    def add_rule_to_archive(self, full_rule, antecedent, consequent, fitness, start, end, support, confidence, inclusion):
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
        """
        rule_repr = self.rule_representation(full_rule)
        # Check if the rule is already in the archive (by its string representation)
        if rule_repr not in [self.rule_representation(r['full_rule']) for r in self.rule_archive]:
            # Add the rule, its antecedent, consequent, fitness, support, confidence, inclusion, and timestamps to the archive
            self.rule_archive.append({
                'full_rule': full_rule,
                'antecedent': antecedent,
                'consequent': consequent,
                'fitness': fitness,
                'support': support,
                'confidence': confidence,
                'inclusion': inclusion,
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

    def calculate_support(self, df, antecedents, consequents, start=0, end=0, use_interval=False):
        """
        Calculate the support for the given list of antecedents and consequents within the specified time range or interval range.

        Parameters:
            df (pd.DataFrame): The dataset containing the transactions.
            antecedents (list): A list of dictionaries defining the antecedent conditions.
            consequents (list): A list of dictionaries defining the consequent conditions.
            start (int or datetime): The start of the interval (if use_interval is True) or timestamp range.
            end (int or datetime): The end of the interval (if use_interval is True) or timestamp range.
            use_interval (bool): Whether to use 'interval' (True) or 'timestamp' (False) for filtering.

        Returns:
            float: Support value.
        """
        # Filter data by interval or timestamp
        if use_interval:
            df_filtered = df[(df['interval'] >= start) & (df['interval'] <= end)]
        else:
            df_filtered = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

        # Get the number of filtered transactions
        filtered = len(df_filtered)

        # Apply each antecedent condition (both categorical and numerical)
        for antecedent in antecedents:
            if antecedent['type'] == 'Categorical':
                df_filtered = df_filtered[df_filtered[antecedent['feature']] == antecedent['category']]
            elif antecedent['type'] == 'Numerical':
                df_filtered = df_filtered[
                    (df_filtered[antecedent['feature']] >= antecedent['border1']) &
                    (df_filtered[antecedent['feature']] <= antecedent['border2'])
                ]

        # Apply each consequent condition (both categorical and numerical)
        for consequent in consequents:
            if consequent['type'] == 'Categorical':
                df_filtered = df_filtered[df_filtered[consequent['feature']] == consequent['category']]
            elif consequent['type'] == 'Numerical':
                df_filtered = df_filtered[
                    (df_filtered[consequent['feature']] >= consequent['border1']) &
                    (df_filtered[consequent['feature']] <= consequent['border2'])
                ]

        # Support is the ratio of rows matching both antecedents and consequents to total filtered rows
        return len(df_filtered) / filtered if len(df) > 0 else 0


    def calculate_confidence(self, df, antecedents, consequents, start, end, use_interval=False):
        """
        Calculate the confidence for the given list of antecedents and consequents within the specified time range or interval range.

        Parameters:
            df (pd.DataFrame): The dataset containing the transactions.
            antecedents (list): A list of dictionaries defining the antecedent conditions.
            consequents (list): A list of dictionaries defining the consequent conditions.
            start (int or datetime): The start of the interval (if use_interval is True) or timestamp range.
            end (int or datetime): The end of the interval (if use_interval is True) or timestamp range.
            use_interval (bool): Whether to use 'interval' (True) or 'timestamp' (False) for filtering.

        Returns:
            float: Confidence value.
        """
        # Filter data by interval or timestamp
        if use_interval:
            df_filtered = df[(df['interval'] >= start) & (df['interval'] <= end)]
        else:
            df_filtered = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

        # Get the number of filtered transactions
        filtered = len(df_filtered)

        # Apply each antecedent condition (both categorical and numerical)
        for antecedent in antecedents:
            if antecedent['type'] == 'Categorical':
                df_filtered = df_filtered[df_filtered[antecedent['feature']] == antecedent['category']]
            elif antecedent['type'] == 'Numerical':
                df_filtered = df_filtered[
                    (df_filtered[antecedent['feature']] >= antecedent['border1']) &
                    (df_filtered[antecedent['feature']] <= antecedent['border2'])
                ]

        # Now we have the filtered data where all antecedents are true
        antecedent_support = df_filtered

        # Apply consequent conditions (both categorical and numerical) to the antecedent-supporting rows
        for consequent in consequents:
            if consequent['type'] == 'Categorical':
                df_filtered = df_filtered[df_filtered[consequent['feature']] == consequent['category']]
            elif consequent['type'] == 'Numerical':
                df_filtered = df_filtered[
                    (df_filtered[consequent['feature']] >= consequent['border1']) &
                    (df_filtered[consequent['feature']] <= consequent['border2'])
                ]

        # Confidence is the ratio of rows that match both antecedents and consequents to the rows matching antecedents
        return len(df_filtered) / len(antecedent_support) if len(antecedent_support) > 0 else 0

    def calculate_inclusion_metric(self, antecedents, consequents):
        """
        Calculate the inclusion metric, which counts the number of attributes present
        in both the antecedent and consequent. The value is normalized between 0 and 1.

        This metric is based on the paper:
        I. Fister Jr., V. Podgorelec, I. Fister. Improved Nature-Inspired Algorithms for
        Numeric Association Rule Mining. In: Vasant P., Zelinka I., Weber GW. (eds)
        Intelligent Computing and Optimization. ICO 2020. Advances in Intelligent
        Systems and Computing, vol 1324. Springer, Cham.

        Args:
            antecedents (list): A list of dictionaries defining the antecedent conditions.
            consequents (list): A list of dictionaries defining the consequent conditions.

        Returns:
            float: The inclusion metric (0 to 1).
        """
        # Get feature names in antecedents and consequents
        antecedent_features = {feature['feature'] for feature in antecedents}
        consequent_features = {feature['feature'] for feature in consequents}

        # Find common features
        common_features = antecedent_features.intersection(consequent_features)

        # Total unique features in antecedent and consequent
        total_unique_features = len(antecedent_features.union(consequent_features))

        # Calculate inclusion metric
        if total_unique_features == 0:
            return 0.0  # Avoid division by zero

        inclusion_metric = len(common_features) / total_unique_features
        return inclusion_metric

    def calculate_amplitude_metric(self, antecedents, consequents):
        """
        Calculate the amplitude metric for the given rule, based on the numerical attributes' borders.
        The metric rewards smaller ranges and is normalized between 0 and 1.

        Args:
            antecedents (list): A list of dictionaries defining the antecedent conditions.
            consequents (list): A list of dictionaries defining the consequent conditions.

        Returns:
            float: The amplitude metric (0 to 1), where smaller ranges are preferred.
        """
        total_range = 0.0  # Sum of normalized ranges for all numerical attributes
        num_numerical_attributes = 0  # Counter for the number of numerical attributes

        # Combine antecedents and consequents
        rule_parts = antecedents + consequents

        for feature in rule_parts:
            if feature['type'] == 'Numerical':
                # Get the feature's borders
                border1 = feature['border1']
                border2 = feature['border2']

                # Retrieve the original feature min and max from the dataset metadata
                feature_min = self.features[feature['feature']]['min']
                feature_max = self.features[feature['feature']]['max']

                # Calculate the normalized range (border2 - border1) / (feature_max - feature_min)
                if feature_max != feature_min:  # Avoid division by zero
                    normalized_range = (border2 - border1) / (feature_max - feature_min)
                else:
                    normalized_range = 0.0  # If the feature has no variation, set range to 0

                # Accumulate the total normalized range and count numerical attributes
                total_range += normalized_range
                num_numerical_attributes += 1

        # Calculate the average normalized range
        if num_numerical_attributes == 0:
            return 0.0  # No numerical attributes, return 0 as amplitude metric

        # Normalize the amplitude metric (the smaller the range, the better)
        amplitude_metric = 1 - (total_range / num_numerical_attributes)

        return amplitude_metric

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

    def calculate_fitness(self, supp, conf, incl, alpha=1.0, beta=1.0, delta=1.0):
        """
        Calculate the fitness of the rules based on support, confidence, and inclusion.
        """
        return ((self.alpha * supp) + (self.beta * conf) + (self.delta * incl)) / 3

    def map_to_interval(self, val):
        min_interval = self.features['interval']['min']
        max_interval = self.features['interval']['max']
        return min_interval, max_interval

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
                'consequent': entry['consequent'],
                'start_timestamp': str(entry['start']),
                'end_timestamp': str(entry['end'])
            })

        # Save to JSON
        with open(file_path, 'w') as f:
            json.dump(archive_dict, f, indent=4)
        print(f"Rules saved to {file_path}.")
