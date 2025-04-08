import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve, KFold
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import HYPERPARAMETERS

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
        model: The model to evaluate
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
                
                # Create datasets and dataloaders
                train_dataset = TensorDataset(
                    torch.tensor(X_train.astype(np.float32)).to(device),
                    torch.tensor(y_train.astype(np.longlong)).to(device)
                )
                val_dataset = TensorDataset(
                    torch.tensor(X_val.astype(np.float32)).to(device),
                    torch.tensor(y_val.astype(np.longlong)).to(device)
                )
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                # Configure optimizer and criterion
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=0.001, weight_decay=0.1
                )
                criterion = torch.nn.CrossEntropyLoss()
                
                # Train model
                num_epochs = HYPERPARAMETERS[model_name]["num_epochs"]
                for epoch in range(num_epochs):
                    model.train()
                    for inputs, labels in train_loader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate on training and validation sets
                model.eval()
                with torch.no_grad():
                    # Training score
                    train_correct = 0
                    train_total = 0
                    for inputs, labels in train_loader:
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += labels.size(0)
                        train_correct += (predicted == labels).sum().item()
                    train_scores[size_idx, fold_idx] = train_correct / train_total
                    
                    # Validation score
                    val_correct = 0
                    val_total = 0
                    for inputs, labels in val_loader:
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                    test_scores[size_idx, fold_idx] = val_correct / val_total
                    
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
        plt.savefig(os.path.join(output_path, f"{model_name}_learning_curve.png"))
        plt.close()
        
        return train_scores, test_scores
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
        plt.savefig(os.path.join(output_path, f"{model_name}_learning_curve.png"))
        plt.close()
        
        return train_scores, test_scores
