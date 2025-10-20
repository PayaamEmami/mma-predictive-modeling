import os
import numpy as np
import pandas as pd
import torch
import boto3
import json
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score

from config import RESULTS_PATH, is_experimental
from plots import plot_model_comparisons, plot_learning_curve


def copy_baseline_metrics(s3_bucket):
    """
    Copy current baseline metrics from S3 for comparison.
    Only runs during experimental training runs.

    Args:
        s3_bucket: S3 bucket name where baseline results are stored

    Returns:
        None. Saves baseline_metrics.json to RESULTS_PATH.
    """
    s3_client = boto3.client("s3")

    try:
        print("Copying baseline metrics from S3...")

        # Download baseline performance CSV
        baseline_key = "results/model_performances.csv"
        response = s3_client.get_object(Bucket=s3_bucket, Key=baseline_key)
        baseline_df = pd.read_csv(response["Body"])

        # Convert to simple dict format for easy comparison
        baseline_metrics = {
            row["Model"]: float(row["Accuracy"])
            for _, row in baseline_df.iterrows()
        }

        # Save locally
        baseline_path = os.path.join(RESULTS_PATH, "baseline_metrics.json")
        with open(baseline_path, "w") as f:
            json.dump(baseline_metrics, f, indent=2)

        print(f"✓ Baseline metrics captured: {len(baseline_metrics)} models")

    except Exception as e:
        print(f"⚠ Could not copy baseline metrics: {e}")
        print("Continuing without baseline comparison...")


def generate_experiment_report(performance_df, output_path):
    """
    Auto-generate markdown report comparing experiment to baseline.

    Args:
        performance_df: DataFrame with experiment results (Model, Accuracy columns)
        output_path: Path where report should be saved

    Returns:
        str: Generated markdown report content
    """
    # Load baseline metrics if available
    baseline_path = os.path.join(output_path, "baseline_metrics.json")
    baseline_metrics = {}

    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            baseline_metrics = json.load(f)

    # Load experiment config if available
    # Try to find experiment_config.json in parent experiments/ directory
    experiment_config = {}
    s3_results_prefix = os.environ.get("S3_RESULTS_PREFIX", "")

    if s3_results_prefix.startswith("experiments/"):
        # Extract experiment name from path like "experiments/results/" or "experiments/my-exp/results/"
        parts = s3_results_prefix.strip("/").split("/")
        if len(parts) >= 2:
            experiment_name = parts[1] if parts[1] != "results" else parts[0]
            config_path = os.path.join("experiments", experiment_name, "experiment_config.json")

            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    experiment_config = json.load(f)

    # Calculate comparisons
    comparisons = []
    improvements = 0
    regressions = 0

    for _, row in performance_df.iterrows():
        model = row["Model"]
        exp_acc = float(row["Accuracy"])
        base_acc = baseline_metrics.get(model, None)

        if base_acc is not None:
            delta = exp_acc - base_acc
            percent = (delta / base_acc * 100) if base_acc > 0 else 0

            if delta > 0.001:  # Small threshold for floating point
                improvements += 1
            elif delta < -0.001:
                regressions += 1
        else:
            delta = None
            percent = None

        comparisons.append({
            "model": model,
            "baseline": base_acc,
            "experiment": exp_acc,
            "delta": delta,
            "percent": percent,
        })

    # Generate markdown report
    report_lines = []

    # Header
    exp_name = experiment_config.get("experiment_name", "Experiment")
    report_lines.append(f"## {exp_name}\n")

    # Hypothesis
    if "hypothesis" in experiment_config:
        report_lines.append(f"{experiment_config['hypothesis']}\n")

    # Metadata
    exp_date = experiment_config.get("date", datetime.now().strftime("%Y-%m-%d"))
    report_lines.append(f"**Date:** {exp_date}\n")

    # Results summary
    if baseline_metrics:
        report_lines.append("\n**Summary:**")
        report_lines.append(f"- Improvements: {improvements} models")
        report_lines.append(f"- Regressions: {regressions} models")
        report_lines.append(f"- Unchanged: {len(comparisons) - improvements - regressions} models\n")

    # Results table
    report_lines.append("\n**Results:**\n")

    if baseline_metrics:
        report_lines.append("| Model | Baseline Accuracy | Experiment Accuracy | Change | % Change |")
        report_lines.append("|-------|------------------|---------------------|--------|----------|")

        for comp in comparisons:
            if comp["delta"] is not None:
                sign = "+" if comp["delta"] >= 0 else ""
                report_lines.append(
                    f"| {comp['model']} | {comp['baseline']:.4f} | {comp['experiment']:.4f} | "
                    f"{sign}{comp['delta']:.4f} | {sign}{comp['percent']:.2f}% |"
                )
            else:
                report_lines.append(
                    f"| {comp['model']} | N/A | {comp['experiment']:.4f} | N/A | N/A |"
                )
    else:
        # No baseline comparison available
        report_lines.append("| Model | Accuracy |")
        report_lines.append("|-------|----------|")
        for comp in comparisons:
            report_lines.append(f"| {comp['model']} | {comp['experiment']:.4f} |")

    # Changed hyperparameters
    if "changed_hyperparameters" in experiment_config:
        report_lines.append("\n**Changed Hyperparameters:**\n")
        report_lines.append("```json")
        report_lines.append(json.dumps(experiment_config["changed_hyperparameters"], indent=2))
        report_lines.append("```\n")

    # Placeholder for discussion
    report_lines.append("**Discussion:**\n")
    report_lines.append("_[Add your analysis and conclusions here]_\n")

    # Join all lines
    report = "\n".join(report_lines)

    # Save report
    report_path = os.path.join(output_path, "experiment_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"✓ Experiment report generated: {report_path}")

    return report


def update_experiments_index(experiment_config, performance_df, output_path):
    """
    Update the EXPERIMENTS.md index file with a new experiment entry.

    Args:
        experiment_config: Dict with experiment configuration
        performance_df: DataFrame with experiment results
        output_path: Path where experiment results are saved

    Returns:
        None. Updates experiments/EXPERIMENTS.md file.
    """
    # Get experiment name from path
    s3_results_prefix = os.environ.get("S3_RESULTS_PREFIX", "")
    parts = s3_results_prefix.strip("/").split("/")
    experiment_name = parts[1] if len(parts) >= 2 and parts[1] != "results" else "unknown"

    exp_name = experiment_config.get("experiment_name", experiment_name)
    exp_date = experiment_config.get("date", datetime.now().strftime("%Y-%m-%d"))
    exp_hypothesis = experiment_config.get("hypothesis", "")

    # Create experiment entry
    entry = f"""
### [{exp_name}]({experiment_name}/experiment_report.md)

**Date:** {exp_date}

**Hypothesis:** {exp_hypothesis}

[View Full Report →]({experiment_name}/experiment_report.md)

---
"""

    # Path to EXPERIMENTS.md (in experiments/ directory)
    experiments_index_path = os.path.join(
        os.path.dirname(os.path.dirname(output_path)),
        "EXPERIMENTS.md"
    )

    # If file doesn't exist yet, we're probably in AWS, so create relative path
    if not os.path.exists(experiments_index_path):
        # Try one level up
        experiments_index_path = os.path.join(
            os.path.dirname(output_path),
            "EXPERIMENTS.md"
        )

    try:
        if os.path.exists(experiments_index_path):
            with open(experiments_index_path, "r") as f:
                content = f.read()

            # Find the markers
            start_marker = "<!-- EXPERIMENTS_LIST_START -->"
            end_marker = "<!-- EXPERIMENTS_LIST_END -->"

            if start_marker in content and end_marker in content:
                start_idx = content.index(start_marker) + len(start_marker)
                end_idx = content.index(end_marker)

                # Get existing entries
                existing = content[start_idx:end_idx].strip()

                # Remove placeholder text if it exists
                if "No experiments completed yet" in existing:
                    existing = ""

                # Add new entry at the top
                new_list = f"\n{entry}\n{existing}\n"

                # Reconstruct content
                new_content = (
                    content[:start_idx] +
                    new_list +
                    content[end_idx:]
                )

                # Write back
                with open(experiments_index_path, "w") as f:
                    f.write(new_content)

                print(f"✓ Updated EXPERIMENTS.md index with new experiment")
            else:
                print(f"⚠ EXPERIMENTS.md markers not found")
        else:
            print(f"⚠ EXPERIMENTS.md not found at {experiments_index_path}")

    except Exception as e:
        print(f"⚠ Could not update EXPERIMENTS.md: {e}")
        print("Continuing without index update...")


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

    # For experimental runs, copy baseline metrics first
    if is_experimental():
        s3_bucket = os.environ.get("S3_BUCKET")
        if s3_bucket:
            copy_baseline_metrics(s3_bucket)

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

    # For experimental runs, generate comparison report and update index
    if is_experimental():
        generate_experiment_report(performance_df, RESULTS_PATH)

        # Update EXPERIMENTS.md index
        baseline_path = os.path.join(RESULTS_PATH, "baseline_metrics.json")
        experiment_config = {}

        # Try to load experiment config
        s3_results_prefix = os.environ.get("S3_RESULTS_PREFIX", "")
        if s3_results_prefix.startswith("experiments/"):
            parts = s3_results_prefix.strip("/").split("/")
            if len(parts) >= 2:
                experiment_name = parts[1] if parts[1] != "results" else parts[0]
                config_path = os.path.join("experiments", experiment_name, "experiment_config.json")

                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        experiment_config = json.load(f)

        update_experiments_index(experiment_config, performance_df, RESULTS_PATH)

    print(f"\nAll tasks completed. Results and plots saved in {RESULTS_PATH}.")
