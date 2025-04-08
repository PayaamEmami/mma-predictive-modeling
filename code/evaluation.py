import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from config import OUTPUT_PATH
from utils import plot_model_accuracies, plot_learning_curve


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
        # Generate learning curves and get all scores
        train_scores, test_scores = plot_learning_curve(
            model, X_train, y_train, name, OUTPUT_PATH, device
        )
        
        # Calculate performance metrics from learning curves
        final_train_accuracy = train_scores[-1].mean()
        final_val_accuracy = test_scores[-1].mean()
        train_std = train_scores[-1].std()
        val_std = test_scores[-1].std()
        
        # Store comprehensive performance metrics
        model_performances[name] = {
            'Final Train Accuracy': final_train_accuracy,
            'Final Validation Accuracy': final_val_accuracy,
            'Train Std': train_std,
            'Validation Std': val_std,
            'Learning Rate': (final_val_accuracy - final_train_accuracy) / final_train_accuracy
        }

        # Get predictions for classification report
        if name in ["FCNN", "RNN", "LSTM", "Transformer"]:
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test.astype(np.float32)).to(device)
                outputs = model(X_test_tensor)
                _, y_pred = torch.max(outputs.data, 1)
                y_pred = y_pred.cpu().numpy()
        else:
            y_pred = model.predict(X_test)

        # Print classification report
        print(
            f"Classification Report for {name}:\n",
            classification_report(y_test, y_pred, target_names=label_encoder.classes_),
        )
        print(
            f"Learning Curve Metrics for {name}:\n"
            f"  Final Training Accuracy: {final_train_accuracy:.4f} (+/- {train_std:.4f})\n"
            f"  Final Validation Accuracy: {final_val_accuracy:.4f} (+/- {val_std:.4f})\n"
            f"  Learning Rate: {model_performances[name]['Learning Rate']:.4f}"
        )

    # Save and display results
    performance_df = pd.DataFrame(
        [(name, metrics['Final Validation Accuracy']) 
         for name, metrics in model_performances.items()],
        columns=["Model", "Accuracy"]
    )
    performance_df.to_csv(
        os.path.join(OUTPUT_PATH, "model_performances.csv"),
        index=False
    )
    
    # Save detailed metrics
    detailed_metrics_df = pd.DataFrame(model_performances).T
    detailed_metrics_df.to_csv(
        os.path.join(OUTPUT_PATH, "detailed_model_metrics.csv")
    )
    
    plot_model_accuracies(performance_df, OUTPUT_PATH)
    print("\nModel Performance Summary:")
    print(performance_df)
    print("\nDetailed Metrics:")
    print(detailed_metrics_df)

    print(f"\nAll tasks completed. Results and plots saved in {OUTPUT_PATH}.")
