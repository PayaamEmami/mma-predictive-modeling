# utils.py

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
import torch
from torch.utils.data import DataLoader, TensorDataset

def plot_confusion_matrix(model_name, y_test, y_pred, label_encoder, output_path):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot()
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(os.path.join(output_path, f'confusion_matrix_{model_name.replace(" ", "_")}.png'))
    plt.show()
    plt.close()

def plot_model_accuracies(performance_df, output_path):
    performance_df = performance_df.sort_values(by='Accuracy', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(performance_df['Model'], performance_df['Accuracy'], color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Comparison of Model Accuracies')
    plt.xlim([0, 1])
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.yticks(rotation=45)
    plt.savefig(os.path.join(output_path, 'model_accuracy_comparison.png'))
    plt.show()
    plt.close()


def plot_learning_curve(model, X, y, model_name, output_path, device, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), n_trials=5):
    if isinstance(model, torch.nn.Module):
        train_scores = {size: [] for size in train_sizes}
        test_scores = {size: [] for size in train_sizes}

        for size in train_sizes:
            subset_size = int(len(X) * size)
            for trial in range(n_trials):
                indices = np.random.permutation(len(X))
                X_subset = X[indices[:subset_size]]
                y_subset = y[indices[:subset_size]]

                dataset = TensorDataset(torch.tensor(X_subset.astype(np.float32)).to(device), torch.tensor(y_subset.astype(np.longlong)).to(device))
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

                optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.1)
                criterion = torch.nn.CrossEntropyLoss()

                model.train()
                for epoch in range(10):
                    for inputs, labels in dataloader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    train_outputs = model(torch.tensor(X_subset.astype(np.float32)).to(device))
                    train_pred = torch.argmax(train_outputs, dim=1).cpu().numpy()
                    train_accuracy = (train_pred == y_subset).sum() / len(y_subset)
                    train_scores[size].append(train_accuracy)

                    test_outputs = model(torch.tensor(X.astype(np.float32)).to(device))
                    test_pred = torch.argmax(test_outputs, dim=1).cpu().numpy()
                    test_accuracy = (test_pred == y).sum() / len(y)
                    test_scores[size].append(test_accuracy)

        train_scores_mean = [np.mean(train_scores[size]) for size in train_sizes]
        train_scores_std = [np.std(train_scores[size]) for size in train_sizes]
        test_scores_mean = [np.mean(test_scores[size]) for size in train_sizes]
        test_scores_std = [np.std(test_scores[size]) for size in train_sizes]

        train_sizes_abs = [int(len(X) * size) for size in train_sizes]
        plt.figure()
        plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes_abs, test_scores_mean, 'o-', color='g', label='Cross-validation score')
        plt.fill_between(train_sizes_abs, np.array(train_scores_mean) - np.array(train_scores_std), np.array(train_scores_mean) + np.array(train_scores_std), color='r', alpha=0.1)
        plt.fill_between(train_sizes_abs, np.array(test_scores_mean) - np.array(test_scores_std), np.array(test_scores_mean) + np.array(test_scores_std), color='g', alpha=0.1)
    else:
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color='r', alpha=0.1)
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color='g', alpha=0.1)

    plt.title(f'Learning Curve for {model_name}')
    plt.xlabel('Training examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(os.path.join(output_path, f'learning_curve_{model_name.replace(" ", "_")}.png'))
    plt.show()
    plt.close()