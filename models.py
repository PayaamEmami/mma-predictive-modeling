# models.py

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import torch.nn as nn


# transformer
class MMATransformerNet(nn.Module):
    def __init__(self, input_size):
        super(MMATransformerNet, self).__init__()
        self.num_features = input_size
        self.embedding_dim = 64
        # embedding layer for numerical features
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
        x = x.unsqueeze(-1)  # (batch_size, num_features, 1)
        x = self.embedding(x)  # (batch_size, num_features, embedding_dim)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        x = x.flatten(1)  # (batch_size, num_features * embedding_dim)
        x = self.fc(x)  # (batch_size, 2)
        return x


# feedforward neural network
class MMANet(nn.Module):
    def __init__(self, input_size):
        super(MMANet, self).__init__()
        # fully connected layer
        self.fc1 = nn.Linear(input_size, 512)
        # ReLU activation layer
        self.relu = nn.ReLU()
        # fully connected layer
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        out = self.fc1(x)  # (batch_size, 512)
        out = self.relu(out)  # (batch_size, 512)
        out = self.fc2(out)  # (batch_size, 2)
        return out


def initialize_models(input_size, device):
    models = {}

    models["Random Forest"] = RandomForestClassifier(random_state=21)

    models["Gradient Boosting"] = GradientBoostingClassifier(random_state=21)

    models["Support Vector Machine"] = SVC(probability=True, random_state=21)

    models["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=21)

    models["K-Nearest Neighbors"] = KNeighborsClassifier()

    models["Naive Bayes"] = GaussianNB()

    models["Neural Network"] = MMANet(input_size).to(device)

    models["Transformer"] = MMATransformerNet(input_size).to(device)

    return models
