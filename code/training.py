import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, List, Dict, Any
from config import HYPERPARAMETERS, DATA_PATH

def create_time_series_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.2,
    min_train_size: int = 1000
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create time series splits for cross-validation.
    
    Args:
        data: DataFrame containing fight data sorted by EventDate
        n_splits: Number of splits for cross-validation
        test_size: Proportion of data to use for testing in each split
        min_train_size: Minimum number of training samples required
        
    Returns:
        List of tuples containing (train_data, test_data) for each split
    """
    total_size = len(data)
    test_size_samples = int(total_size * test_size)
    
    # Calculate step size to ensure sufficient training samples
    available_size = total_size - test_size_samples - min_train_size
    step_size = available_size // (n_splits - 1)
    
    splits = []
    for i in range(n_splits):
        if i == 0:
            # First split uses min_train_size for training
            train_start = 0
            train_end = min_train_size
            test_start = train_end
            test_end = test_start + test_size_samples
        else:
            # Move training and test windows forward for subsequent splits
            train_start = min_train_size + (i - 1) * step_size
            train_end = train_start + min_train_size
            test_start = train_end
            test_end = test_start + test_size_samples
        
        train_data = data.iloc[train_start:train_end].copy()
        test_data = data.iloc[test_start:test_end].copy()
        splits.append((train_data, test_data))
    
    return splits

def prepare_data_for_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    device: torch.device
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare data for PyTorch models by converting to tensors and creating DataLoaders.
    
    Args:
        train_data: Training DataFrame
        test_data: Test DataFrame
        feature_columns: List of feature column names
        target_column: Name of target column
        device: PyTorch device to use
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Convert data to appropriate types
    X_train = train_data[feature_columns].values.astype(np.float32)
    y_train = train_data[target_column].values.astype(np.longlong)
    X_test = test_data[feature_columns].values.astype(np.float32)
    y_test = test_data[target_column].values.astype(np.longlong)
    
    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train).to(device),
        torch.tensor(y_train).to(device)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test).to(device),
        torch.tensor(y_test).to(device)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

def evaluate_model_on_split(
    model: Any,
    test_loader: DataLoader,
    device: torch.device,
    is_pytorch: bool = True
) -> float:
    """
    Evaluate model performance on a single split.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader containing test data
        device: PyTorch device to use
        is_pytorch: Whether the model is a PyTorch model
        
    Returns:
        Accuracy score
    """
    if is_pytorch:
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    else:
        # Convert PyTorch tensors to numpy arrays for scikit-learn models
        X_test = []
        y_test = []
        for inputs, labels in test_loader:
            X_test.extend(inputs.cpu().numpy())
            y_test.extend(labels.cpu().numpy())
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        return model.score(X_test, y_test)

def print_cross_validation_results(results: Dict[str, List[float]]) -> None:
    """
    Print cross-validation results in a formatted way.
    
    Args:
        results: Dictionary containing accuracy scores for each model
    """
    print("\nCross-Validation Results:")
    print("-" * 50)
    
    for model_name, scores in results.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{model_name}:")
        print(f"  Mean Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
        print(f"  Individual Scores: {[f'{score:.4f}' for score in scores]}")
        print("-" * 50)

def time_series_cross_validate(
    data: pd.DataFrame,
    models: Dict[str, Any],
    feature_columns: List[str],
    target_column: str,
    device: torch.device,
    n_splits: int = 5,
    test_size: float = 0.2,
    min_train_size: int = 1000
) -> Tuple[Dict[str, List[float]], pd.DataFrame, pd.DataFrame]:
    """
    Perform time series cross-validation on multiple models.
    
    Args:
        data: DataFrame containing fight data
        models: Dictionary of model names and their instances
        feature_columns: List of feature column names
        target_column: Name of target column
        device: PyTorch device to use
        n_splits: Number of splits for cross-validation
        test_size: Proportion of data to use for testing in each split
        min_train_size: Minimum number of training samples required
        
    Returns:
        Tuple containing:
        - Dictionary containing accuracy scores for each model across all splits
        - Training data for final evaluation
        - Test data for final evaluation
    """
    splits = create_time_series_splits(data, n_splits, test_size, min_train_size)
    results = {name: [] for name in models.keys()}
    
    for split_idx, (train_data, test_data) in enumerate(splits):
        print(f"\nProcessing split {split_idx + 1}/{n_splits}")
        print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        
        for name, model in models.items():
            is_pytorch = name in ["FCNN", "RNN", "LSTM", "Transformer"]
            
            # Prepare data and train model
            train_loader, test_loader = prepare_data_for_model(
                train_data, test_data, feature_columns, target_column, device
            )
            
            if is_pytorch:
                model.train()
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters())
                
                num_epochs = HYPERPARAMETERS[name]["num_epochs"]
                for epoch in range(num_epochs):
                    for inputs, labels in train_loader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
            else:
                X_train = train_data[feature_columns].values
                y_train = train_data[target_column].values
                model.fit(X_train, y_train)
            
            # Evaluate and store results
            accuracy = evaluate_model_on_split(model, test_loader, device, is_pytorch)
            results[name].append(accuracy)
            print(f"{name} accuracy: {accuracy:.4f}")
    
    print_cross_validation_results(results)
    return results, splits[-1][0], splits[-1][1]

def train_model(name, model, X_train, y_train, device):
    """
    Train a model and save its state.
    
    Args:
        name: Name of the model
        model: Model instance to train
        X_train: Training features
        y_train: Training labels
        device: PyTorch device to use
    """
    if name in ["FCNN", "RNN", "LSTM", "Transformer"]:
        params = HYPERPARAMETERS[name]
        criterion = torch.nn.CrossEntropyLoss()

        # Configure optimizer based on model parameters
        if params["optimizer"].lower() == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
                momentum=params["momentum"],
            )
        elif params["optimizer"].lower() == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
                betas=params["betas"],
                eps=params["eps"],
            )
        elif params["optimizer"].lower() == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
                betas=params["betas"],
                eps=params["eps"],
                amsgrad=params["amsgrad"],
            )
        elif params["optimizer"].lower() == "rmsprop":
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
                alpha=params["alpha"],
                eps=params["eps"],
                momentum=params["momentum"],
                centered=params["centered"],
            )
        else:
            print(f"Warning: Unknown optimizer '{params['optimizer']}'. Falling back to SGD.")
            optimizer = optim.SGD(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
            )

        # Prepare data and train model
        num_epochs = params["num_epochs"]
        batch_size = params["batch_size"]

        X_train_tensor = torch.tensor(X_train.astype(np.float32)).to(device)
        y_train_tensor = torch.tensor(y_train.astype(np.longlong)).to(device)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        # Save model state
        torch.save(model.state_dict(), os.path.join(DATA_PATH, f"{name}.pth"))
        print(f"Model {name} saved successfully.")
    else:
        model.fit(X_train, y_train)
