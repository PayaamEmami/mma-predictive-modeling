import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, accuracy_score

from config import RESULTS_PATH
from plots import plot_model_comparisons, plot_learning_curve


def evaluate_models(models, X_train, X_test, y_train, y_test, label_encoder, device):
    """
    Evaluate trained models and generate performance metrics and visualizations.

    Args:
        models: Dictionary of model names and their trained instances
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        label_encoder: Label encoder for class names
        device: PyTorch device to use

    Returns:
        None. Saves results and plots to RESULTS_PATH.
    """
    model_performances = {}

    # Prepare a string to collect all printed metrics and reports
    report_lines = []

    print("\nGenerating evaluation plots and metrics...")

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        # Generate learning curves
        train_scores, test_scores = plot_learning_curve(
            model,
            X_train,
            y_train,
            name,
            RESULTS_PATH,
            device,
            train_sizes=np.linspace(0.2, 1.0, 5),
            verbose=False,
        )

        # Calculate performance metrics from learning curves
        final_train_accuracy = train_scores[-1].mean()
        train_std = train_scores[-1].std()
        val_std = test_scores[-1].std()

        # Get predictions for classification report
        if name in ["FNN", "RNN", "LSTM", "Transformer"]:
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test.astype(np.float32)).to(device)
                outputs = model(X_test_tensor)
                _, y_pred = torch.max(outputs.data, 1)
                y_pred = y_pred.cpu().numpy()
        else:
            y_pred = model.predict(X_test)

        # Calculate final test accuracy
        final_test_accuracy = accuracy_score(y_test, y_pred)

        # Store comprehensive performance metrics using final test accuracy
        model_performances[name] = {
            "Final Train Accuracy": final_train_accuracy,
            "Final Test Accuracy": final_test_accuracy,
            "Train Std": train_std,
            "Validation Std": val_std,
            "Learning Rate": (final_test_accuracy - final_train_accuracy)
            / final_train_accuracy,
        }

        # Prepare learning curve metrics string
        learning_curve_str = (
            f"Learning Curve Metrics for {name}:\n"
            f"  Final Training Accuracy: {final_train_accuracy:.4f} (+/- {train_std:.4f})\n"
            f"  Final Test Accuracy: {final_test_accuracy:.4f} (+/- {val_std:.4f})\n"
            f"  Learning Rate: {model_performances[name]['Learning Rate']:.4f}"
        )
        print(learning_curve_str)

        # Prepare classification report string
        class_report_str = (
            f"Classification Report for {name}:\n"
            + classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        )
        print(class_report_str)

        # Add learning curve metrics and classification report to report_lines
        report_lines.append(learning_curve_str + "\n" + class_report_str + "\n")

    # Save and display results using final test accuracy
    performance_df = pd.DataFrame(
        [
            (name, metrics["Final Test Accuracy"])
            for name, metrics in model_performances.items()
        ],
        columns=["Model", "Accuracy"],
    )
    performance_df.to_csv(
        os.path.join(RESULTS_PATH, "model_performances.csv"), index=False
    )

    # Save the report to a txt file
    report_path = os.path.join(RESULTS_PATH, "model_metrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    plot_model_comparisons(performance_df, RESULTS_PATH)
    print("\nModel Performance Summary:")
    print(performance_df)

    print(f"\nAll tasks completed. Results and plots saved in {RESULTS_PATH}.")
