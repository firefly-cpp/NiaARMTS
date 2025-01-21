Getting started
===============

Installation
------------

To install ``NiaARMTS`` with pip, use:

..  code:: bash

    pip install niaarmts

Usage
-----

Fixed Interval Time Series Numerical Association Rule Mining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

..  code:: python

    from niapy.algorithms.basic import ParticleSwarmAlgorithm
    from niapy.task import Task
    from niaarmts import Dataset
    from niaarmts.NiaARMTS import NiaARMTS

    # Load dataset
    dataset = Dataset()
    dataset.load_data_from_csv('intervals.csv', timestamp_col='timestamp')

    # Create an instance of NiaARMTS
    niaarmts_problem = NiaARMTS(
        dimension=dataset.calculate_problem_dimension(),  # Adjust dimension dynamically
        lower=0.0,  # Lower bound of solution space
        upper=1.0,  # Upper bound of solution space
        features=dataset.get_all_features_with_metadata(),  # Pass feature metadata
        transactions=dataset.get_all_transactions(),  # Dataframe containing all transactions
        interval='true',  # Whether we're dealing with interval data
        alpha=1.0,  # Weight for support in fitness calculation
        beta=1.0,  # Weight for confidence in fitness calculation
        gamma=1.0,  # Weight for inclusion in fitness calculation # if 0.0 then inclusion metric is omitted
        delta=1.0  # Weight for amplitude in fitness calculation # if 0.0 then amplitude metric is omitted
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

Segmented Interval Time Series Numerical Association Rule Mining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

..  code:: python

    from niapy.algorithms.basic import ParticleSwarmAlgorithm
    from niapy.task import Task
    from niaarmts import Dataset
    from niaarmts.NiaARMTS import NiaARMTS

    # Load dataset
    dataset = Dataset()
    dataset.load_data_from_csv('ts.csv', timestamp_col='timestamp')

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
        delta=1.0  # Weight for amplitude in fitness calculation # if 0.0 then amplitude metric is omitted
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