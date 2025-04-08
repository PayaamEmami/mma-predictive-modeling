import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve, KFold
import torch
from config import HYPERPARAMETERS
from training import train_model

train_color = "#1f77b4"
cross_validation_color = "#ff7f0e"


def plot_model_accuracies(performance_df, output_path):
    """
    Generate and save a bar plot comparing model accuracies.
    
    Args:
        performance_df: DataFrame containing model names and their accuracies
        output_path: Path to save the plot
        
    Returns:
        None. Saves the plot to output_path.
    """
    performance_df = performance_df.sort_values(by="Accuracy", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(performance_df["Model"], performance_df["Accuracy"], color="skyblue")
    plt.xlabel("Accuracy")
    plt.title("Comparison of Model Accuracies")
    plt.xlim([0, 1])
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.yticks(rotation=45)
    plt.savefig(os.path.join(output_path, "model_accuracy_comparison.png"))
    plt.show()
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
    train_sizes=np.linspace(0.1, 1.0, 10),
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
        
    Returns:
        train_scores: Array of training scores for each training size and fold
        test_scores: Array of validation scores for each training size and fold
    """
    if isinstance(model, torch.nn.Module):
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
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
                
                # Evaluate on training and validation sets
                model.eval()
                with torch.no_grad():
                    # Training score
                    X_train_tensor = torch.tensor(X_train.astype(np.float32)).to(device)
                    outputs = model(X_train_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    train_scores[size_idx, fold_idx] = (predicted == torch.tensor(y_train).to(device)).sum().item() / len(y_train)
                    
                    # Validation score
                    X_val_tensor = torch.tensor(X_val.astype(np.float32)).to(device)
                    outputs = model(X_val_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    test_scores[size_idx, fold_idx] = (predicted == torch.tensor(y_val).to(device)).sum().item() / len(y_val)
    else:
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            X,
            y,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            scoring="accuracy",
        )
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(
        train_sizes,
        train_scores.mean(axis=1),
        "o-",
        color=train_color,
        label="Training score",
    )
    plt.plot(
        train_sizes,
        test_scores.mean(axis=1),
        "o-",
        color=cross_validation_color,
        label="Cross-validation score",
    )
    plt.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.1,
        color=train_color,
    )
    plt.fill_between(
        train_sizes,
        test_scores.mean(axis=1) - test_scores.std(axis=1),
        test_scores.mean(axis=1) + test_scores.std(axis=1),
        alpha=0.1,
        color=cross_validation_color,
    )
    plt.title(f"Learning Curve for {model_name}")
    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f"learning_curve_{model_name.replace(' ', '_')}.png"))
    plt.close()
    
    return train_scores, test_scores
