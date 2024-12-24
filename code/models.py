# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from config.config import HYPERPARAMETERS


# fully connected neural network
class FCNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FCNN, self).__init__()
        # fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # relu activation layer
        self.relu = nn.ReLU()
        # fully connected layer
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        out = self.fc1(x) # (batch_size, 512)
        out = self.relu(out) # (batch_size, 512)
        out = self.fc2(out) # (batch_size, 2)
        return out


# recurrent neural network
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(-1) # (batch_size, input_size, 1)
        out, _ = self.rnn(x) # out: (batch_size, input_size, hidden_size)
        out = out[:, -1, :] # (batch_size, hidden_size)
        out = self.fc(out) # (batch_size, 2)
        return out


# long short-term memory neural network
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(-1) # (batch_size, input_size, 1)
        out, (hn, cn) = self.lstm(x) # out: (batch_size, input_size, hidden_size)
        # take the last timestep
        out = out[:, -1, :] # (batch_size, hidden_size)
        out = self.fc(out) # (batch_size, 2)
        return out


# transformer neural network
class Transformer(nn.Module):
    def __init__(self, input_size):
        super(Transformer, self).__init__()
        self.num_features = input_size
        self.embedding_dim = 64
        # embedding layer
        self.embedding = nn.Linear(1, self.embedding_dim)
        # positional encoding
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.num_features, self.embedding_dim)
        )
        # transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=8, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # final classification layer
        self.fc = nn.Linear(self.num_features * self.embedding_dim, 2)

    def forward(self, x):
        x = x.unsqueeze(-1) # (batch_size, num_features, 1)
        x = self.embedding(x) # (batch_size, num_features, embedding_dim)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        x = x.flatten(1) # (batch_size, num_features * embedding_dim)
        x = self.fc(x) # (batch_size, 2)
        return x


def initialize_models(input_size, device):
    models = {}

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

    models["FCNN"] = FCNN(
        input_size, hidden_size=HYPERPARAMETERS["FCNN"]["hidden_size"]
    ).to(device)

    models["RNN"] = RNN(
        input_size,
        hidden_size=HYPERPARAMETERS["RNN"]["hidden_size"],
        num_layers=HYPERPARAMETERS["RNN"],
    ).to(device)

    models["LSTM"] = LSTM(
        input_size,
        hidden_size=HYPERPARAMETERS["LSTM"]["hidden_size"],
        num_layers=HYPERPARAMETERS["LSTM"]["num_layers"],
    ).to(device)

    models["Transformer"] = Transformer(
        input_size,
        embedding_dim=HYPERPARAMETERS["Transformer"]["embedding_dim"],
        num_layers=HYPERPARAMETERS["Transformer"]["num_layers"],
        nhead=HYPERPARAMETERS["Transformer"]["nhead"],
    ).to(device)

    return models
