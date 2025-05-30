import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from config import HYPERPARAMETERS


class FCNN(nn.Module):
    """
    Fully Connected Neural Network for fight outcome prediction.

    Architecture:
        - Input layer -> Hidden layer (ReLU) -> Output layer
    """

    def __init__(self, input_size, hidden_size):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # Input shape: (batch_size, input_size)
        out = self.fc1(x)  # Shape: (batch_size, hidden_size)
        out = self.relu(out)  # Shape: (batch_size, hidden_size)
        out = self.fc2(out)  # Shape: (batch_size, 2)
        return out


class Transformer(nn.Module):
    """
    Transformer Network for fight outcome prediction.

    Architecture:
        - Input embedding -> Positional encoding -> Transformer encoder -> Classification layer
    """

    def __init__(self, input_size, embedding_dim, num_layers, nhead):
        super(Transformer, self).__init__()
        self.num_features = input_size
        self.embedding_dim = embedding_dim

        # Network components
        self.embedding = nn.Linear(1, self.embedding_dim)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.num_features, self.embedding_dim)
        )

        # Transformer encoder setup
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        self.fc = nn.Linear(self.num_features * self.embedding_dim, 2)

    def forward(self, x):
        # Input shape: (batch_size, num_features)
        x = x.unsqueeze(-1)  # Shape: (batch_size, num_features, 1)
        x = self.embedding(x)  # Shape: (batch_size, num_features, embedding_dim)
        x = x + self.positional_encoding
        x = self.transformer_encoder(
            x
        )  # Shape: (batch_size, num_features, embedding_dim)
        x = x.flatten(1)  # Shape: (batch_size, num_features * embedding_dim)
        x = self.fc(x)  # Shape: (batch_size, 2)
        return x


def initialize_models(input_size, device):
    """
    Initialize all models with their respective hyperparameters.

    Args:
        input_size: Number of input features
        device: PyTorch device to use

    Returns:
        Dictionary of initialized models
    """
    models = {}

    # Initialize scikit-learn models
    models["Random Forest"] = RandomForestClassifier(**HYPERPARAMETERS["Random Forest"])
    models["Gradient Boosting"] = GradientBoostingClassifier(
        **HYPERPARAMETERS["Gradient Boosting"]
    )
    models["SVM"] = SVC(**HYPERPARAMETERS["SVM"])
    models["Logistic Regression"] = LogisticRegression(
        **HYPERPARAMETERS["Logistic Regression"]
    )
    models["KNN"] = KNeighborsClassifier(**HYPERPARAMETERS["KNN"])
    models["Naive Bayes"] = GaussianNB(**HYPERPARAMETERS["Naive Bayes"])

    # Initialize PyTorch models
    models["FCNN"] = FCNN(
        input_size, hidden_size=HYPERPARAMETERS["FCNN"]["hidden_size"]
    ).to(device)

    models["Transformer"] = Transformer(
        input_size,
        embedding_dim=HYPERPARAMETERS["Transformer"]["embedding_dim"],
        num_layers=HYPERPARAMETERS["Transformer"]["num_layers"],
        nhead=HYPERPARAMETERS["Transformer"]["nhead"],
    ).to(device)

    return models
