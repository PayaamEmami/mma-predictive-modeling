# Quick Start: Running Your First Enhanced Experiment

This guide walks through running an experiment with the new automated tracking system.

## Step 1: Create Experiment Config

Create a new folder and config file for your experiment:

```bash
cd experiments
mkdir my-new-experiment
```

Create `experiments/my-new-experiment/experiment_config.json`:

```json
{
  "experiment_name": "My New Experiment",
  "date": "2024-10-20",
  "hypothesis": "Testing whether [CHANGE] improves model performance",
  "changed_hyperparameters": {
    "ModelName": {
      "parameter_name": {
        "old": "old_value",
        "new": "new_value"
      }
    }
  }
}
```

## Step 2: Update Code

Make your changes (usually in `code/config.py`):

```python
# Example: Change learning rate for FNN
HYPERPARAMETERS = {
    "FNN": {
        "learning_rate": 0.001,  # Changed from 0.0005
        # ... other params
    }
}
```

## Step 3: Commit to Experimental Branch

```bash
git checkout experimental
git add .
git commit -m "Experiment: Testing higher learning rate for FNN"
git push origin experimental
```

GitHub Actions will automatically upload your code to S3.

## Step 4: Trigger AWS Training

Trigger the SageMaker training job (your usual process).

The training will automatically:

- ✅ Detect it's an experimental run (from S3 path)
- ✅ Copy baseline metrics from production
- ✅ Train all models with your changes
- ✅ Generate comparison report
- ✅ Save model files locally
- ✅ Upload everything to S3
- ✅ Create GitHub PR with results

## Step 5: Review Results

Once the PR is created, review your experiment folder:

```
experiments/my-new-experiment/
  experiment_config.json       ← Your input
  baseline_metrics.json        ← Baseline accuracies
  experiment_report.md         ← 🆕 Auto-generated comparison!
  model_performances.csv       ← Experiment results
  model_metrics_report.txt     ← Detailed metrics
  learning_curve_*.png         ← Training plots
  models/                      ← 🆕 Saved models!
    FNN.pth
    Transformer.pth
    ...
```
