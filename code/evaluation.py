import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report

from config import OUTPUT_PATH
from utils import plot_confusion_matrix, plot_model_accuracies, plot_learning_curve


def evaluate_models(models, X_train, X_test, y_train, y_test, label_encoder, device):
    """
    Evaluate multiple models and generate performance metrics and visualizations.
    
    Args:
        models: Dictionary of model names and their instances
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        label_encoder: Label encoder for class names
        device: PyTorch device to use
        
    Returns:
        None. Saves results and plots to OUTPUT_PATH.
    """
    model_performances = {}

    for name, model in models.items():
        # Handle PyTorch models
        if name in ["FCNN", "RNN", "LSTM", "Transformer"]:
            model.eval()
            with torch.no_grad():
                # Convert test data to tensor and get predictions
                X_test_tensor = torch.tensor(X_test.astype(np.float32)).to(device)
                outputs = model(X_test_tensor)
                _, y_pred_tensor = torch.max(outputs.data, 1)
                y_pred = y_pred_tensor.cpu().numpy()
                accuracy = (y_pred == y_test).sum() / y_test.size
        # Handle scikit-learn models
        else:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

        # Store and display model performance
        model_performances[name] = accuracy
        print(
            f"Classification Report for {name}:\n",
            classification_report(y_test, y_pred, target_names=label_encoder.classes_),
        )

        # Generate visualizations
        plot_confusion_matrix(name, y_test, y_pred, label_encoder, OUTPUT_PATH)
        plot_learning_curve(model, X_train, y_train, name, OUTPUT_PATH, device)

    # Save and display results
    performance_df = pd.DataFrame(
        list(model_performances.items()),
        columns=["Model", "Accuracy"]
    )
    performance_df.to_csv(
        os.path.join(OUTPUT_PATH, "model_performances.csv"),
        index=False
    )
    plot_model_accuracies(performance_df, OUTPUT_PATH)
    print(performance_df)

    print(f"All tasks completed. Results and plots saved in {OUTPUT_PATH}.")
