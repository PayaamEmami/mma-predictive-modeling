# training.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator
from models import UFCNet

def train_sklearn_model(name, model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def train_pytorch_model(model, X_train, y_train, device, num_epochs=500, lr=0.001, weight_decay=0.1):
    # convert data to pytorch tensors
    X_train_tensor = torch.tensor(X_train.astype(float)).to(device)
    y_train_tensor = torch.tensor(y_train.astype(int)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model
