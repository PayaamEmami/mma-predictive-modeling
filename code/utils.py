import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.model_selection import learning_curve, KFold
import torch
from config import HYPERPARAMETERS
from training import train_model
from models import FCNN, RNN, LSTM, Transformer


def plot_model_accuracies(performance_df, output_path):
    """
    Generate and save a bar plot comparing model accuracies.
    
    Args:
        performance_df: DataFrame containing model names and their accuracies
        output_path: Path to save the plot
        
    Returns:
        None. Saves the plot to output_path.
    """
    # Sort by accuracy for cleaner display
    performance_df = performance_df.sort_values(by="Accuracy", ascending=True)

    # Normalize accuracy for color mapping
    norm = mcolors.Normalize(vmin=performance_df["Accuracy"].min(),
                             vmax=performance_df["Accuracy"].max())
    
    # Use a pastel-friendly version of RdYlGn
    cmap = plt.get_cmap("RdYlGn")
    colors = [cmap(norm(score)) for score in performance_df["Accuracy"]]

    plt.figure(figsize=(10, 6))
    plt.barh(performance_df["Model"], performance_df["Accuracy"], color=colors)
    plt.xlabel("Mean Cross-Validation Accuracy")
    plt.title("Model Performance Comparison")
    plt.xlim([0, 1])
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "model_accuracy_comparison.png"))
    plt.close()


def plot_learning_curve(
    model,
    X,
    y,
    model_name,
    output_path,
    device,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
    random_state=42,
    verbose=False
):
    """
    Generate and plot learning curves for a model.
    
    Args:
        model: The trained model to evaluate
        X: Training features
        y: Training labels
        model_name: Name of the model for plot title
        output_path: Path to save the plot
        device: PyTorch device to use
        cv: Number of cross-validation folds
        n_jobs: Number of jobs to run in parallel
        train_sizes: Array of training set sizes to evaluate
        random_state: Random seed for reproducibility
        verbose: Whether to print training progress
        
    Returns:
        train_scores: Array of training scores for each training size and fold
        test_scores: Array of validation scores for each training size and fold
    """
    if isinstance(model, torch.nn.Module):
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        train_scores = np.zeros((len(train_sizes), cv))
        test_scores = np.zeros((len(train_sizes), cv))
        
        for size_idx, size in enumerate(train_sizes):
            subset_size = int(len(X) * size)
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                # Get subset of training data
                X_train = X[train_idx[:subset_size]]
                y_train = y[train_idx[:subset_size]]
                X_val = X[val_idx]
                y_val = y[val_idx]
                
                # Create a fresh copy of the model for this fold
                if isinstance(model, LSTM):
                    fresh_model = LSTM(
                        model.input_size,
                        hidden_size=HYPERPARAMETERS["LSTM"]["hidden_size"],
                        num_layers=HYPERPARAMETERS["LSTM"]["num_layers"]
                    ).to(device)
                elif isinstance(model, RNN):
                    fresh_model = RNN(
                        model.input_size,
                        hidden_size=HYPERPARAMETERS["RNN"]["hidden_size"],
                        num_layers=HYPERPARAMETERS["RNN"]["num_layers"]
                    ).to(device)
                elif isinstance(model, Transformer):
                    fresh_model = Transformer(
                        model.num_features,
                        embedding_dim=HYPERPARAMETERS["Transformer"]["embedding_dim"],
                        num_layers=HYPERPARAMETERS["Transformer"]["num_layers"],
                        nhead=HYPERPARAMETERS["Transformer"]["nhead"]
                    ).to(device)
                elif isinstance(model, FCNN):
                    fresh_model = FCNN(
                        next(model.parameters()).shape[1],
                        hidden_size=HYPERPARAMETERS["FCNN"]["hidden_size"]
                    ).to(device)

                # Train the model
                fresh_model = train_model(model_name, fresh_model, X_train, y_train, device, verbose=verbose)
                
                # Evaluate on training and validation sets
                fresh_model.eval()
                with torch.no_grad():
                    # Training score
                    X_train_tensor = torch.tensor(X_train.astype(np.float32)).to(device)
                    outputs = fresh_model(X_train_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    train_scores[size_idx, fold_idx] = (predicted == torch.tensor(y_train).to(device)).sum().item() / len(y_train)
                    
                    # Validation score
                    X_val_tensor = torch.tensor(X_val.astype(np.float32)).to(device)
                    outputs = fresh_model(X_val_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    test_scores[size_idx, fold_idx] = (predicted == torch.tensor(y_val).to(device)).sum().item() / len(y_val)

        # Convert fractional train_sizes to actual sample counts for plotting
        train_sizes = (train_sizes * len(X)).astype(int)
    else:
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            X,
            y,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            scoring="accuracy",
            random_state=random_state
        )
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    
    plt.plot(
        train_sizes,
        train_scores.mean(axis=1),
        "o-",
        color="C0",
        label="Training Accuracy",
    )
    plt.plot(
        train_sizes,
        test_scores.mean(axis=1),
        "o-",
        color="C1",
        label="Cross-Validation Accuracy",
    )
    plt.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.1,
        color="C0",
    )
    plt.fill_between(
        train_sizes,
        test_scores.mean(axis=1) - test_scores.std(axis=1),
        test_scores.mean(axis=1) + test_scores.std(axis=1),
        alpha=0.1,
        color="C1",
    )
    plt.title(f"Learning Curve for {model_name}\n(Mean Accuracy ± 1 Standard Deviation)")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Accuracy")
    
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f"learning_curve_{model_name.replace(' ', '_')}.png"))
    plt.close()
    
    return train_scores, test_scores
