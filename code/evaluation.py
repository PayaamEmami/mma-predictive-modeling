import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, accuracy_score

from config import RESULTS_PATH
from plots import plot_model_comparisons, plot_learning_curve


def _predict(model, X, device):
    """Run inference with a trained model, handling both PyTorch and sklearn models."""
    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.astype(np.float32)).to(device)
            outputs = model(X_tensor)
            _, y_pred = torch.max(outputs.data, 1)
            return y_pred.cpu().numpy()
    return model.predict(X)


def evaluate_validation(models, X_train, X_val, y_train, y_val, label_encoder, device):
    """
    Evaluate trained models on the validation set and generate performance metrics.

    Generates learning curves, classification reports, and a comparison bar plot.

    Args:
        models: Dictionary of model names and their trained instances
        X_train: Training features (used for learning curves)
        X_val: Validation features
        y_train: Training labels (used for learning curves)
        y_val: Validation labels
        label_encoder: Label encoder for class names
        device: PyTorch device to use

    Returns:
        None. Saves results and plots to RESULTS_PATH.
    """
    model_performances = {}

    report_lines = []

    print("\nGenerating evaluation plots and metrics...")

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        # Generate learning curves using training data with temporal CV
        train_scores, val_scores = plot_learning_curve(
            model,
            X_train,
            y_train,
            name,
            RESULTS_PATH,
            device,
            train_sizes=np.linspace(0.2, 1.0, 5),
            verbose=False,
        )

        # Extract final fold scores for summary statistics
        final_train_accuracy = train_scores[-1].mean()
        train_std = train_scores[-1].std()
        val_std = val_scores[-1].std()

        # Get predictions on validation set
        y_pred = _predict(model, X_val, device)

        final_val_accuracy = accuracy_score(y_val, y_pred)

        model_performances[name] = {
            "Final Train Accuracy": final_train_accuracy,
            "Final Validation Accuracy": final_val_accuracy,
            "Train Std": train_std,
            "Validation Std": val_std,
            "Learning Rate": (final_val_accuracy - final_train_accuracy)
            / final_train_accuracy,
        }

        learning_curve_str = (
            f"Learning Curve Metrics for {name}:\n"
            f"  Final Training Accuracy: {final_train_accuracy:.4f} (+/- {train_std:.4f})\n"
            f"  Final Validation Accuracy: {final_val_accuracy:.4f} (+/- {val_std:.4f})\n"
            f"  Learning Rate: {model_performances[name]['Learning Rate']:.4f}"
        )
        print(learning_curve_str)

        class_report_str = (
            f"Classification Report for {name}:\n"
            + classification_report(y_val, y_pred, target_names=label_encoder.classes_)
        )
        print(class_report_str)

        report_lines.append(learning_curve_str + "\n" + class_report_str + "\n")

    # Save validation accuracies to CSV
    performance_df = pd.DataFrame(
        [
            (name, metrics["Final Validation Accuracy"])
            for name, metrics in model_performances.items()
        ],
        columns=["Model", "Accuracy"],
    )
    performance_df.to_csv(
        os.path.join(RESULTS_PATH, "validation_performances.csv"), index=False
    )

    # Save detailed metrics report
    report_path = os.path.join(RESULTS_PATH, "validation_metrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Generate validation comparison bar plot
    plot_model_comparisons(
        performance_df,
        RESULTS_PATH,
        filename="validation_performances.png",
        label="Validation",
    )
    print("\nModel Performance Summary:")
    print(performance_df)

    print(f"\nAll tasks completed. Results and plots saved in {RESULTS_PATH}.")


def evaluate_test(models, X_test, y_test, label_encoder, device):
    """
    Evaluate all models on the held-out test set.

    Produces an accuracy summary CSV and comparison bar plot.

    Args:
        models: Dictionary of model names and their trained instances
        X_test: Test features
        y_test: Test labels
        label_encoder: Label encoder for class names
        device: PyTorch device to use

    Returns:
        None. Saves results and plot to RESULTS_PATH.
    """
    test_results = []
    print("\n--- Held-out Test Set Evaluation ---")

    # Get predictions and accuracy for each model
    for name, model in models.items():
        y_pred = _predict(model, X_test, device)
        acc = accuracy_score(y_test, y_pred)
        test_results.append((name, acc))
        print(f"  {name}: {acc:.4f}")

    # Save test accuracies to CSV and generate comparison bar plot
    test_df = pd.DataFrame(test_results, columns=["Model", "Accuracy"])
    test_df.to_csv(os.path.join(RESULTS_PATH, "test_performances.csv"), index=False)
    plot_model_comparisons(
        test_df, RESULTS_PATH, filename="test_performances.png", label="Test"
    )
    print(f"Test set results saved to {RESULTS_PATH}/test_performances.csv")
