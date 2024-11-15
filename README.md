# UFC Fight Outcome Prediction

## Description

This project aims to predict the outcomes of UFC (Ultimate Fighting Championship) fights using machine learning models. It encompasses data loading, preprocessing, feature engineering, model training, and evaluation using various algorithms, including Random Forest, Gradient Boosting, Support Vector Machines, Logistic Regression, K-Nearest Neighbors, Neural Networks (PyTorch), and Naive Bayes.

## Directory Structure

"""
project/
│
├── main.py                # Entry point
├── config.py              # Configuration and constants
├── data_loader.py         # Data loading and cleaning
├── preprocessing.py       # Feature engineering and preprocessing
├── models.py              # Model definitions
├── training.py            # Training logic
├── evaluation.py          # Evaluation and metrics
├── utils.py               # Utility functions (optional)
│
└── outputs/               # Save outputs like models, logs, plots
"""

## Installation

1. **Clone the repository:**

"""
git clone https://github.com/yourusername/ufc-prediction.git
"""

2. **Navigate to the project directory:**

"""
cd ufc-prediction
"""

3. **Install the required dependencies:**

"""
pip install -r requirements.txt
"""

*Note: Ensure that `requirements.txt` includes all necessary packages such as `pandas`, `numpy`, `torch`, `scikit-learn`, `matplotlib`, etc.*

## Configuration

All configuration settings, such as file paths and device configurations, are located in the `config.py` file. Modify this file as needed to suit your environment.

## Usage

Run the main script to execute the entire workflow:

"""
python main.py
"""

## Modules

### `config.py`

Handles configurations and global variables such as paths, parameters, and device setup.

### `data_loader.py`

Handles data loading and preprocessing logic. It includes functions for reading raw data, cleaning it, and generating features.

### `preprocessing.py`

Contains functions for parsing and preprocessing data, such as parsing heights, reach, strikes, and calculating fight times.

### `models.py`

Contains model definitions and initialization logic, including both scikit-learn models and the custom PyTorch neural network.

### `training.py`

Handles the training logic for models. It trains both scikit-learn models and the PyTorch neural network.

### `evaluation.py`

Handles evaluation logic for models. It generates predictions, calculates metrics like accuracy and confusion matrices, and plots and saves graphs.

### `utils.py` (Optional)

Contains utility functions that can be reused across different modules, such as plotting utilities or helper functions.

## Outputs

All results, including trained models, performance metrics, and plots, are saved in the `outputs/` directory. Each run creates a timestamped subdirectory to organize the outputs.

## Dependencies

The project relies on the following major libraries:

- Python 3.7+
- pandas
- numpy
- torch
- scikit-learn
- matplotlib
- seaborn

Ensure all dependencies are installed via the `requirements.txt` file.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please contact [your.email@example.com](mailto:your.email@example.com).
