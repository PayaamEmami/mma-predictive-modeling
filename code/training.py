import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from config import HYPERPARAMETERS, DATA_PATH, OUTPUT_PATH
from utils import plot_learning_curve


def train_model(name, model, X_train, y_train, device):
    """
    Train a model and generate learning curves.
    
    Args:
        name: Name of the model
        model: Model instance to train (PyTorch or sklearn)
        X_train: Training features
        y_train: Training labels
        device: PyTorch device to use
        
    Returns:
        The trained model and learning curve metrics (train_scores, test_scores)
    """
    print(f"\nTraining {name}...")
    
    params = HYPERPARAMETERS[name]
    train_scores = None
    test_scores = None
    
    if not isinstance(model, torch.nn.Module):
        # Handle sklearn models
        model.fit(X_train, y_train)
        train_scores, test_scores = plot_learning_curve(
            model, X_train, y_train, name, OUTPUT_PATH, device
        )
    else:
        # PyTorch model training
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

        X_train_tensor = torch.tensor(X_train.astype(np.float32)).to(device)
        y_train_tensor = torch.tensor(y_train.astype(np.longlong)).to(device)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

        for epoch in range(params["num_epochs"]):
            model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        # Generate learning curves for PyTorch models
        train_scores, test_scores = plot_learning_curve(
            model, X_train, y_train, name, OUTPUT_PATH, device
        )
    
    return model, train_scores, test_scores
