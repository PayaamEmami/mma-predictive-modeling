# training.py

import torch
import torch.optim as optim
import numpy as np


def train_model(name, model, X_train, y_train, device):
    if name in ["FCNN", "RNN", "LSTM", "Transformer"]:
        # training logic for pytorch models
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.1)
        num_epochs = 500

        X_train_tensor = torch.tensor(X_train.astype(np.float32)).to(device)
        y_train_tensor = torch.tensor(y_train.astype(np.longlong)).to(device)

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
    else:
        # training logic for scikit-learn models
        model.fit(X_train, y_train)
