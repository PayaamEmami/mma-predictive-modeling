# config.py

import os
from datetime import datetime
import torch

# paths
BASE_PATH = "/content/drive/MyDrive/dev"
DATA_PATH = os.path.join(BASE_PATH, "data/ufc")
OUTPUT_PATH = os.path.join(
    BASE_PATH,
    "output/mma-predictive-modeling",
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
)

# ensure directories exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

HYPERPARAMETERS = {
    "Random Forest": {
        # tree-related parameters
        "n_estimators": 100, # number of trees in the forest
        "criterion": "gini", # function to measure split quality
        "max_depth": None, # maximum depth of the tree
        "min_samples_split": 2, # minimum samples required to split
        "min_samples_leaf": 1, # minimum samples required in a leaf
        "min_weight_fraction_leaf": 0.0, # minimum weighted fraction of leaf
        "max_features": "sqrt", # number of features to consider for split
        "max_leaf_nodes": None, # maximum number of leaf nodes
        "min_impurity_decrease": 0.0, # minimum impurity decrease for split
        # ensemble-related parameters
        "bootstrap": True, # whether to bootstrap samples
        "oob_score": False, # whether to use out-of-bag samples
        # computational settings
        "n_jobs": None, # number of jobs to run in parallel
        "random_state": 21, # random seed for reproducibility
        "verbose": 0, # verbosity level
        "warm_start": False, # reuse previous solutions to add trees
        # regularization
        "class_weight": None, # weights associated with classes
        "ccp_alpha": 0.0, # complexity parameter for pruning
        "max_samples": None, # maximum samples to draw for bootstrap
    },
    "Gradient Boosting": {
        # loss and learning parameters
        "loss": "log_loss", # loss function to optimize
        "learning_rate": 0.1, # step size for each update
        "n_estimators": 100, # number of boosting stages
        "subsample": 1.0, # fraction of samples for fitting base learners
        # tree-related parameters
        "criterion": "friedman_mse", # split quality function
        "min_samples_split": 2, # minimum samples required to split
        "min_samples_leaf": 1, # minimum samples required in a leaf
        "min_weight_fraction_leaf": 0.0, # minimum weighted fraction of leaf
        "max_depth": 3, # maximum depth of the tree
        "min_impurity_decrease": 0.0, # minimum impurity decrease for split
        "init": None, # estimator to use for initial predictions
        # computational settings
        "random_state": 21, # random seed for reproducibility
        "max_features": None, # maximum number of features to consider
        "verbose": 0, # verbosity level
        "max_leaf_nodes": None, # maximum number of leaf nodes
        "warm_start": False, # reuse previous solutions to add trees
        # early stopping parameters
        "validation_fraction": 0.1, # fraction of training data for validation
        "n_iter_no_change": None, # iterations with no improvement
        "tol": 1e-4, # tolerance for early stopping
        # regularization
        "ccp_alpha": 0.0, # complexity parameter for pruning
    },
    "SVM": {
        # model settings
        "C": 1.0, # regularization parameter
        "kernel": "rbf", # kernel type for svm
        "degree": 3, # degree for polynomial kernel
        "gamma": "scale", # kernel coefficient
        "coef0": 0.0, # independent term in kernel function
        # optimization
        "shrinking": True, # whether to use shrinking heuristic
        "probability": True, # whether to enable probability estimates
        "tol": 1e-3, # tolerance for stopping criterion
        # computational settings
        "cache_size": 200, # size of kernel cache (mb)
        "class_weight": None, # weights associated with classes
        "verbose": False, # verbosity level
        "max_iter": -1, # maximum number of iterations
        # decision function
        "decision_function_shape": "ovr", # one-vs-rest decision function
        "break_ties": False, # whether to break ties
        "random_state": 21, # random seed for reproducibility
    },
    "Logistic Regression": {
        # regularization
        "penalty": "l2", # regularization penalty
        "dual": False, # solve dual problem (only for l2)
        # optimization
        "tol": 1e-4, # tolerance for stopping criteria
        "C": 1.0, # inverse of regularization strength
        # model settings
        "fit_intercept": True, # whether to fit intercept
        "intercept_scaling": 1, # scaling of intercept
        "class_weight": None, # weights associated with classes
        # computational settings
        "random_state": 21, # random seed for reproducibility
        "solver": "lbfgs", # optimization solver
        "max_iter": 1000, # maximum iterations
        "verbose": 0, # verbosity level
        "warm_start": False, # reuse previous solution
        "n_jobs": None, # number of jobs to run in parallel
        "l1_ratio": None, # elastic-net mixing parameter
    },
    "KNN": {
        # model settings
        "n_neighbors": 5, # number of neighbors
        "weights": "uniform", # weight function used in prediction
        "algorithm": "auto", # algorithm for neighbor search
        "leaf_size": 30, # leaf size for tree-based algorithms
        # distance metric
        "p": 2, # power parameter for minkowski metric
        "metric": "minkowski", # distance metric
        "metric_params": None, # additional parameters for metric
        # computational settings
        "n_jobs": None, # number of jobs to run in parallel
    },
    "Naive Bayes": {
        "var_smoothing": 1e-9, # portion of variance added to avoid zero
    },
    "FCNN": {
        # model architecture
        "hidden_size": 512, # number of units in hidden layer
        # training hyperparams common to all
        "optimizer": "SGD", # optimization algorithm
        "learning_rate": 0.001, # step size for optimizer
        "weight_decay": 0.1, # weight decay for regularization
        "num_epochs": 500, # number of training epochs
        "batch_size": None, # batch size for training
        # sgd parameters
        "momentum": 0.0, # momentum factor
        "dampening": 0.0, # dampening for momentum
        "nesterov": False, # whether to enable nesterov momentum
        "maximize": False, # maximize instead of minimize
        # adam/adamw parameters
        "betas": (0.9, 0.999), # coefficients for adam optimizer
        "eps": 1e-8, # term for numerical stability
        "amsgrad": False, # whether to use amsgrad variant
        # rmsprop parameters
        "alpha": 0.99, # smoothing constant for rmsprop
        "centered": False, # whether to center rmsprop
    },
    "RNN": {
        # model architecture
        "hidden_size": 64, # number of units in hidden layer
        "num_layers": 1, # number of recurrent layers
        # training hyperparams
        "optimizer": "SGD", # optimization algorithm
        "learning_rate": 0.001, # step size for optimizer
        "weight_decay": 0.1, # weight decay for regularization
        "num_epochs": 500, # number of training epochs
        "batch_size": None, # batch size for training
        # sgd parameters
        "momentum": 0.0, # momentum factor
        "dampening": 0.0, # dampening for momentum
        "nesterov": False, # whether to enable nesterov momentum
        "maximize": False, # maximize instead of minimize
        # adam/adamw parameters
        "betas": (0.9, 0.999), # coefficients for adam optimizer
        "eps": 1e-8, # term for numerical stability
        "amsgrad": False, # whether to use amsgrad variant
        # rmsprop parameters
        "alpha": 0.99, # smoothing constant for rmsprop
        "centered": False, # whether to center rmsprop
    },
    "LSTM": {
        # model architecture
        "hidden_size": 64, # number of units in hidden layer
        "num_layers": 1, # number of recurrent layers
        # training hyperparams
        "optimizer": "SGD", # optimization algorithm
        "learning_rate": 0.001, # step size for optimizer
        "weight_decay": 0.1, # weight decay for regularization
        "num_epochs": 500, # number of training epochs
        "batch_size": None, # batch size for training
        # sgd parameters
        "momentum": 0.0, # momentum factor
        "dampening": 0.0, # dampening for momentum
        "nesterov": False, # whether to enable nesterov momentum
        "maximize": False, # maximize instead of minimize
        # adam/adamw parameters
        "betas": (0.9, 0.999), # coefficients for adam optimizer
        "eps": 1e-8, # term for numerical stability
        "amsgrad": False, # whether to use amsgrad variant
        # rmsprop parameters
        "alpha": 0.99, # smoothing constant for rmsprop
        "centered": False, # whether to center rmsprop
    },
    "Transformer": {
        # model architecture
        "embedding_dim": 64, # size of embedding vectors
        "num_layers": 2, # number of transformer layers
        "nhead": 8, # number of attention heads
        # training hyperparams
        "optimizer": "SGD", # optimization algorithm
        "learning_rate": 0.001, # step size for optimizer
        "weight_decay": 0.1, # weight decay for regularization
        "num_epochs": 500, # number of training epochs
        "batch_size": None, # batch size for training
        # sgd parameters
        "momentum": 0.0, # momentum factor
        "dampening": 0.0, # dampening for momentum
        "nesterov": False, # whether to enable nesterov momentum
        "maximize": False, # maximize instead of minimize
        # adam/adamw parameters
        "betas": (0.9, 0.999), # coefficients for adam optimizer
        "eps": 1e-8, # term for numerical stability
        "amsgrad": False, # whether to use amsgrad variant
        # rmsprop parameters
        "alpha": 0.99, # smoothing constant for rmsprop
        "centered": False, # whether to center rmsprop
    },
}
