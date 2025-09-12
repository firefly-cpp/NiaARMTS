from niapy.algorithms.basic import ParticleSwarmAlgorithm
from niapy.task import Task
from niaarmts import Dataset
from niaarmts.NiaARMTS import NiaARMTS

# Load dataset
dataset = Dataset()
dataset.load_data_from_csv('datasets/september24.csv', timestamp_col='timestamp')

# Create an instance of NiaARMTS
niaarmts_problem = NiaARMTS(
    dimension=dataset.calculate_problem_dimension(),  # Adjust dimension dynamically
    lower=0.0,  # Lower bound of solution space
    upper=1.0,  # Upper bound of solution space
    features=dataset.get_all_features_with_metadata(),  # Pass feature metadata
    transactions=dataset.get_all_transactions(),  # Dataframe containing all transactions
    interval='false',  # Whether we're dealing with interval data
    alpha=1.0,  # Weight for support in fitness calculation
    beta=1.0,  # Weight for confidence in fitness calculation
    gamma=1.0,  # Weight for inclusion in fitness calculation # if 0.0 then inclusion metric is omitted
    delta=1.0,  # Weight for amplitude in fitness calculation # if 0.0 then amplitude metric is omitted
    epsilon=1.0 # Weight for timestamp metric in fitness calculation # if 0.0 then tsm is omitted
)

# Define the optimization task
task = Task(problem=niaarmts_problem, max_iters=100)  # Run for 100 iterations

# Initialize the Particle Swarm Optimization algorithm
pso = ParticleSwarmAlgorithm(population_size=40, min_velocity=-1.0, max_velocity=1.0, c1=2.0, c2=2.0)

# Run the algorithm
best_solution = pso.run(task)

# Output the best solution and its fitness value
print(f"Best solution: {best_solution[0]}")
print(f"Fitness value: {best_solution[1]}")

# Save all discovered rules to a CSV file
niaarmts_problem.save_rules_to_csv("discovered_rules.csv")

# Print all rules to the terminal
print("\n=== All Identified Rules (Sorted by Fitness) ===")
for idx, rule in enumerate(niaarmts_problem.get_rule_archive(), 1):
    print(f"\nRule #{idx}:")
    print(f"  Antecedent: {rule['antecedent']}")
    print(f"  Consequent: {rule['consequent']}")
    print(f"  Support: {rule['support']:.4f}")
    print(f"  Confidence: {rule['confidence']:.4f}")
    print(f"  Inclusion: {rule['inclusion']:.4f}")
    print(f"  Amplitude: {rule['amplitude']:.4f}")
    print(f"  TSM: {rule['tsm']:.4f}")
    print(f"  Fitness: {rule['fitness']:.4f}")
    print(f"  Time window: {rule['start']} to {rule['end']}")
