# models.py

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import torch
import torch.nn as nn

class UFCNet(nn.Module):
    def __init__(self, input_size):
        super(UFCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def initialize_models(input_size, device):
    models = {}

    # random forest
    models['Random Forest'] = RandomForestClassifier(random_state=42)

    # gradient boosting
    models['Gradient Boosting'] = GradientBoostingClassifier(random_state=42)

    # support vector machine
    models['Support Vector Machine'] = SVC(probability=True, random_state=42)

    # logistic regression
    models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42)

    # k-nearest neighbors
    models['K-Nearest Neighbors'] = KNeighborsClassifier()

    # naive bayes
    models['Naive Bayes'] = GaussianNB()

    # neural network
    models['Neural Network'] = UFCNet(input_size).to(device)

    return models
