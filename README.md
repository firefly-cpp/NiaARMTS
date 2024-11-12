<h1 align="center">
    Nature-Inspired Algorithms for Time Series Numerical Association Rule Mining
</h1>

<p align="center">
    <a href="#-features">✨ Features</a> •
    <a href="#-installation">📦 Installation</a> •
    <a href="#-basic-example">🚀 Basic example</a> •
    <a href="#-reference-papers">📚 Reference Papers</a> •
    <a href="#-license">🔑 License</a> •
    <a href="#-cite-us">📄 Cite us</a>
</p>

This framework is designed for **numerical association rule mining in time series data** using **stochastic population-based nature-inspired algorithms**[^1]. It provides tools to extract association rules from time series datasets while incorporating key metrics such as **support**, **confidence**, **inclusion**, and **amplitude**. Although independent from the NiaARM framework, this software can be viewed as an extension, with additional support for time series numerical association rule mining.

## ✨ Features

The current version of the framework supports two types of time series numerical association rule mining:

- **Fixed Interval Time Series Numerical Association Rule Mining**
- **Segmented Interval Time Series Numerical Association Rule Mining**

## 📦 Installation

To install `NiaARMTS` with pip, use:

```sh
pip install niaarmts
```

## 🚀 Basic example

### Fixed Interval Time Series Numerical Association Rule Mining example

```python
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
```

### Segmented Interval Time Series Numerical Association Rule Mining example

```python
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
```

## 📚 Reference Papers

Ideas are based on the following research papers:

[1] Iztok Fister Jr., Dušan Fister, Iztok Fister, Vili Podgorelec, Sancho Salcedo-Sanz. [Time series numerical association rule mining variants in smart agriculture](https://iztok.link/static/publications/314.pdf). Journal of Ambient Intelligence and Humanized Computing (2023): 1-14.

[2] Iztok Fister Jr., Iztok Fister, Sancho Salcedo-Sanz. [Time Series Numerical Association Rule Mining for assisting Smart Agriculture](https://iztok.link/static/publications/298.pdf). In: International Conference on Electrical, Computer and Energy Technologies (ICECET). IEEE, 2022.

[3] I. Fister Jr., A. Iglesias, A. Gálvez, J. Del Ser, E. Osaba, I Fister. [Differential evolution for association rule mining using categorical and numerical attributes](http://www.iztok-jr-fister.eu/static/publications/231.pdf) In: Intelligent data engineering and automated learning - IDEAL 2018, pp. 79-88, 2018.

[4] I. Fister Jr., V. Podgorelec, I. Fister. [Improved Nature-Inspired Algorithms for Numeric Association Rule Mining](https://iztok-jr-fister.eu/static/publications/324.pdf). In: Vasant P., Zelinka I., Weber GW. (eds) Intelligent Computing and Optimization. ICO 2020. Advances in Intelligent Systems and Computing, vol 1324. Springer, Cham.

[5] I. Fister Jr., I. Fister [A brief overview of swarm intelligence-based algorithms for numerical association rule mining](https://arxiv.org/abs/2010.15524). arXiv preprint arXiv:2010.15524 (2020).

[6] Fister, I. et al. (2020). [Visualization of Numerical Association Rules by Hill Slopes](http://www.iztok-jr-fister.eu/static/publications/280.pdf).
    In: Analide, C., Novais, P., Camacho, D., Yin, H. (eds) Intelligent Data Engineering and Automated Learning – IDEAL 2020.
    IDEAL 2020. Lecture Notes in Computer Science(), vol 12489. Springer, Cham. https://doi.org/10.1007/978-3-030-62362-3_10

[7] I. Fister, S. Deb, I. Fister, [Population-based metaheuristics for Association Rule Text Mining](http://www.iztok-jr-fister.eu/static/publications/260.pdf),
    In: Proceedings of the 2020 4th International Conference on Intelligent Systems, Metaheuristics & Swarm Intelligence,
    New York, NY, USA, mar. 2020, pp. 19–23. doi: [10.1145/3396474.3396493](https://dl.acm.org/doi/10.1145/3396474.3396493).

[8] I. Fister, I. Fister Jr., D. Novak and D. Verber, [Data squashing as preprocessing in association rule mining](https://iztok-jr-fister.eu/static/publications/300.pdf), 2022 IEEE Symposium Series on Computational Intelligence (SSCI), Singapore, Singapore, 2022, pp. 1720-1725, doi: [10.1109/SSCI51031.2022.10022240](https://doi.org/10.1109/SSCI51031.2022.10022240).

## See also

[1] [NiaARM.jl: Numerical Association Rule Mining in Julia](https://github.com/firefly-cpp/NiaARM.jl)

[2] [arm-preprocessing: Implementation of several preprocessing techniques for Association Rule Mining (ARM)](https://github.com/firefly-cpp/arm-preprocessing)

## 🔑 License

This package is distributed under the MIT License. This license can be found online at <http://www.opensource.org/licenses/MIT>.

## Disclaimer

This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. Use it at your own risk!

## 📄 Cite us

[^1] Fister Jr, I., Yang, X. S., Fister, I., Brest, J., & Fister, D. (2013). [A brief review of nature-inspired algorithms for optimization](https://arxiv.org/abs/1307.4186). arXiv preprint arXiv:1307.4186.

[^2] Iztok Fister Jr., Dušan Fister, Iztok Fister, Vili Podgorelec, Sancho Salcedo-Sanz. [Time series numerical association rule mining variants in smart agriculture](https://iztok.link/static/publications/314.pdf). Journal of Ambient Intelligence and Humanized Computing (2023): 1-14.
