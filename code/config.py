import os
import torch

# Path for local results before uploading to S3
RESULTS_PATH = os.path.join(os.getcwd(), "results")
os.makedirs(RESULTS_PATH, exist_ok=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

HYPERPARAMETERS = {
    "Random Forest": {
        # Tree-related parameters
        "n_estimators": 200,  # Number of trees in the forest
        "criterion": "gini",  # Function to measure split quality
        "max_depth": 8,  # Maximum depth of the tree
        "min_samples_split": 10,  # Minimum samples required to split
        "min_samples_leaf": 4,  # Minimum samples required in a leaf
        "min_weight_fraction_leaf": 0.0,  # Minimum weighted fraction of leaf
        "max_features": "sqrt",  # Number of features to consider for split
        "max_leaf_nodes": None,  # Maximum number of leaf nodes
        "min_impurity_decrease": 0.0,  # Minimum impurity decrease for split
        # Ensemble-related parameters
        "bootstrap": True,  # Whether to bootstrap samples
        "oob_score": False,  # Whether to use out-of-bag samples
        # Computational settings
        "n_jobs": -1,  # Number of jobs to run in parallel
        "random_state": 21,  # Random seed for reproducibility
        "verbose": 0,  # Verbosity level
        "warm_start": False,  # Reuse previous solutions to add trees
        # Regularization
        "class_weight": "balanced",  # Weights associated with classes
        "ccp_alpha": 0.01,  # Complexity parameter for pruning
        "max_samples": None,  # Maximum samples to draw for bootstrap
    },
    "Gradient Boosting": {
        # Loss and learning parameters
        "loss": "log_loss",  # Loss function to optimize
        "learning_rate": 0.1,  # Step size for each update
        "n_estimators": 100,  # Number of boosting stages
        "subsample": 1.0,  # Fraction of samples for fitting base learners
        # Tree-related parameters
        "criterion": "friedman_mse",  # Split quality function
        "min_samples_split": 2,  # Minimum samples required to split
        "min_samples_leaf": 1,  # Minimum samples required in a leaf
        "min_weight_fraction_leaf": 0.0,  # Minimum weighted fraction of leaf
        "max_depth": 2,  # Maximum depth of the tree
        "min_impurity_decrease": 0.0,  # Minimum impurity decrease for split
        "init": None,  # Estimator to use for initial predictions
        # Computational settings
        "random_state": 21,  # Random seed for reproducibility
        "max_features": None,  # Maximum number of features to consider
        "verbose": 0,  # Verbosity level
        "max_leaf_nodes": None,  # Maximum number of leaf nodes
        "warm_start": False,  # Reuse previous solutions to add trees
        # Early stopping parameters
        "validation_fraction": 0.1,  # Fraction of training data for validation
        "n_iter_no_change": None,  # Iterations with no improvement
        "tol": 1e-4,  # Tolerance for early stopping
        # Regularization
        "ccp_alpha": 0.01,  # Complexity parameter for pruning
    },
    "SVM": {
        # Model settings
        "C": 0.5,  # Regularization parameter
        "kernel": "rbf",  # Kernel type for SVM
        "degree": 3,  # Degree for polynomial kernel
        "gamma": "scale",  # Kernel coefficient
        "coef0": 0.0,  # Independent term in kernel function
        # Optimization
        "shrinking": True,  # Whether to use shrinking heuristic
        "probability": True,  # Whether to enable probability estimates
        "tol": 1e-4,  # Tolerance for stopping criterion
        # Computational settings
        "cache_size": 500,  # Size of kernel cache (MB)
        "class_weight": "balanced",  # Weights associated with classes
        "verbose": False,  # Verbosity level
        "max_iter": -1,  # Maximum number of iterations
        # Decision function
        "decision_function_shape": "ovr",  # One-vs-rest decision function
        "break_ties": False,  # Whether to break ties
        "random_state": 21,  # Random seed for reproducibility
    },
    "Logistic Regression": {
        # Regularization
        "penalty": "l2",  # Regularization penalty
        "dual": False,  # Solve dual problem (only for l2)
        # Optimization
        "tol": 1e-5,  # Tolerance for stopping criteria
        "C": 0.5,  # Inverse of regularization strength
        # Model settings
        "fit_intercept": True,  # Whether to fit intercept
        "intercept_scaling": 1,  # Scaling of intercept
        "class_weight": "balanced",  # Weights associated with classes
        # Computational settings
        "random_state": 21,  # Random seed for reproducibility
        "solver": "lbfgs",  # Optimization solver
        "max_iter": 3000,  # Maximum iterations
        "verbose": 0,  # Verbosity level
        "warm_start": False,  # Reuse previous solution
        "n_jobs": -1,  # Number of jobs to run in parallel
        "l1_ratio": None,  # Elastic-net mixing parameter
    },
    "KNN": {
        # Model settings
        "n_neighbors": 5,  # Number of neighbors
        "weights": "uniform",  # Weight function used in prediction
        "algorithm": "auto",  # Algorithm for neighbor search
        "leaf_size": 30,  # Leaf size for tree-based algorithms
        # Distance metric
        "p": 2,  # Power parameter for Minkowski metric
        "metric": "minkowski",  # Distance metric
        "metric_params": None,  # Additional parameters for metric
        # Computational settings
        "n_jobs": -1,  # Number of jobs to run in parallel
    },
    "Naive Bayes": {
        "var_smoothing": 1e-8,  # Portion of variance added to avoid zero
    },
    "Decision Tree": {
        # Tree-related parameters
        "criterion": "gini",  # Function to measure split quality
        "splitter": "best",  # Strategy used to choose the split
        "max_depth": 10,  # Maximum depth of the tree
        "min_samples_split": 2,  # Minimum samples required to split
        "min_samples_leaf": 5,  # Minimum samples required in a leaf
        "min_weight_fraction_leaf": 0.0,  # Minimum weighted fraction of leaf
        "max_features": None,  # Number of features to consider for split
        "random_state": 21,  # Random seed for reproducibility
        "max_leaf_nodes": None,  # Maximum number of leaf nodes
        "min_impurity_decrease": 0.0,  # Minimum impurity decrease for split
        "class_weight": None,  # Weights associated with classes
        "ccp_alpha": 0.01,  # Complexity parameter for pruning
    },
    "FNN": {
        # Model architecture
        "hidden_size": 256,  # Number of units in hidden layer
        # Training hyperparameters common to all
        "optimizer": "Adam",  # Optimization algorithm
        "learning_rate": 0.0005,  # Step size for optimizer
        "weight_decay": 0.05,  # Weight decay for regularization
        "num_epochs": 300,  # Number of training epochs
        "batch_size": 32,  # Batch size for training
        # SGD parameters
        "momentum": 0.0,  # Momentum factor
        "dampening": 0.0,  # Dampening for momentum
        "nesterov": False,  # Whether to enable Nesterov momentum
        "maximize": False,  # Maximize instead of minimize
        # Adam/AdamW parameters
        "betas": (0.9, 0.999),  # Coefficients for Adam optimizer
        "eps": 1e-8,  # Term for numerical stability
        "amsgrad": False,  # Whether to use AMSGrad variant
        # RMSprop parameters
        "alpha": 0.99,  # Smoothing constant for RMSprop
        "centered": False,  # Whether to center RMSprop
    },
    "Transformer": {
        # Model architecture
        "embedding_dim": 128,  # Size of embedding vectors
        "num_layers": 4,  # Number of transformer layers
        "nhead": 8,  # Number of attention heads
        # Training hyperparameters
        "optimizer": "Adam",  # Optimization algorithm
        "learning_rate": 0.0005,  # Step size for optimizer
        "weight_decay": 0.05,  # Weight decay for regularization
        "num_epochs": 300,  # Number of training epochs
        "batch_size": 32,  # Batch size for training
        # SGD parameters
        "momentum": 0.0,  # Momentum factor
        "dampening": 0.0,  # Dampening for momentum
        "nesterov": False,  # Whether to enable Nesterov momentum
        "maximize": False,  # Maximize instead of minimize
        # Adam/AdamW parameters
        "betas": (0.9, 0.999),  # Coefficients for Adam optimizer
        "eps": 1e-8,  # Term for numerical stability
        "amsgrad": False,  # Whether to use AMSGrad variant
        # RMSprop parameters
        "alpha": 0.99,  # Smoothing constant for RMSprop
        "centered": False,  # Whether to center RMSprop
    },
}


def is_experimental():
    """
    Check if this is an experimental run based on S3 results prefix.
    This is the single source of truth for experimental vs main runs.

    Returns:
        bool: True if experimental run, False if main/production run
    """
    s3_results_prefix = os.environ.get("S3_RESULTS_PREFIX", "")
    return s3_results_prefix.startswith("experiments/")