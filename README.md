# Fuzzy Logic Projects

A collection of practical implementations exploring fuzzy logic systems for time series forecasting and control applications.

## Overview

This repository contains two main projects developed as part of a fuzzy logic course. The first focuses on building a fuzzy inference system for time series prediction using the Wang & Mendel rule generation method. The second explores fuzzy control systems using MATLAB/Simulink.

## Project Structure

```
fuzzy_logic/
├── 1st-exercise_rule-generation/    # Time series forecasting with fuzzy systems
│   ├── data/                         # Dataset files (stock prices, temperature)
│   ├── notebooks/                    # Jupyter notebooks with implementations
│   ├── src/                          # Core fuzzy logic utilities and algorithms
│   └── requirements.txt
│  
│
├── 2nd-exercise_fuzzy-control/       # Fuzzy control systems
│   ├── data/                         # Control system datasets
│   └── simulink/                     # MATLAB/Simulink models
│
└── README.md
```

## First Project: Fuzzy Rule Generation for Time Series

This project implements a complete fuzzy inference system for time series forecasting. The system uses the Wang & Mendel method to automatically extract fuzzy rules from historical data.

### Key Features

- Automatic fuzzy rule generation from time series data
- Multiple defuzzification methods (centroid, mean of maxima, weighted average)
- Configurable intersection and implication operations
- Comprehensive hyperparameter optimization
- Performance visualization and analysis tools

### Implementation Details

The fuzzy system workflow includes:

1. **Data Preparation**: Creating time-lagged features from the original series
2. **Fuzzy Variable Definition**: Automatic generation of linguistic variables with adjustable membership functions
3. **Rule Extraction**: Wang & Mendel algorithm for learning rules from training data
4. **Inference**: Forward prediction using the constructed rule base
5. **Evaluation**: MSE/RMSE metrics and residual analysis

### Hyperparameter Search

The implementation includes an exhaustive search across multiple dimensions:

- Number of fuzzy sets (3, 5, 7)
- Window sizes for lag features (1, 2, 3, 5)
- Intersection operations (minimum, product, average)
- Implication methods (minimum, product, bounded sum)
- Defuzzification techniques (centroid, mean of maxima, weighted average)

Results are automatically saved and visualized through boxplots, heatmaps, and sensitivity analysis charts.

### Available Datasets

- **a1621595_data.csv**: Stock market price data (AAPL)
- **s_temp_ubatuba.csv**: Temperature time series
- **stock_data_part_10.csv**: Additional financial data

### Usage Example

```python
from src.fuzzy_utils import lags_create, create_fuzzy_variables, avaliar_modelo
from src.rule_generation import create_fuzzy_system

# Load and prepare data
df = lags_create(window_size=7, file_path='data/a1621595_data.csv', column='Close')

# Create fuzzy variables
variable_list = create_fuzzy_variables(df, num_fuzzy_sets=5, defuzz_method=defuzzificacao_centroid)

# Split data
train_df = df.iloc[:int(len(df) * 0.9)]
test_df = df.iloc[int(len(df) * 0.9):]

# Build and evaluate system
fuzzy_system = create_fuzzy_system(train_df, variable_list, operacao_intersecao_min, implicacao_prod, 1)
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_system)

mse_test, rmse_test, predictions = avaliar_modelo(fuzzy_sim, test_df, defuzzificacao_centroid)
```

## Second Project: Fuzzy Control Systems

This project contains MATLAB/Simulink implementations of fuzzy controllers, focusing on tank level control applications.

The Simulink models demonstrate practical fuzzy control strategies with visualization tools for analyzing system behavior and performance.

## Installation

### Requirements

- Python 3.8 or higher
- MATLAB R2018 or higher (for the second project)

### Setup

```bash
# Clone the repository
git clone https://github.com/palomaflsette/fuzzy-logic.git
cd fuzzy-logic

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The main Python packages used are:

- **scikit-fuzzy**: Fuzzy logic toolkit and control systems
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Visualization and plotting
- **scikit-learn**: Machine learning metrics

## Notebooks

The `1st-exercise_rule-generation/notebooks/` directory contains detailed implementations:

- **solution.ipynb**: Complete implementation with explanations
- **examples/**: Reference implementations for different datasets

Each notebook includes:

- Step-by-step model construction
- Performance evaluation metrics
- Visualization of predictions and residuals
- Hyperparameter optimization results

## Results and Analysis

The system generates comprehensive analysis including:

- **Prediction plots**: Visual comparison of actual vs predicted values
- **Residual analysis**: Error distribution and temporal patterns
- **Performance metrics**: MSE, RMSE for training and testing sets
- **Sensitivity analysis**: Impact of each hyperparameter on model performance
- **Overfitting detection**: Training vs testing error comparison

## Contributing

This is an educational project developed for academic purposes. Feel free to use it as a reference for your own fuzzy logic implementations.

## License

This project is available for educational and research purposes.

## Author

Paloma F. L. Sette

## Acknowledgments

This project was developed as part of a fuzzy logic course, implementing concepts from fuzzy set theory, inference systems, and the Wang & Mendel rule generation algorithm.
