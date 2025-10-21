# Experiments Index

This file is **automatically updated** when new experiments are completed. Each experiment folder is listed below with a link to its detailed report.

See `QUICKSTART.md` for how to run experiments with automated tracking.

---

## Where to View Experiment Reports

**📋 During Development (PR):** Review `experiments/your-experiment-name/experiment_report.md` in the auto-generated PR

**🌐 On Production Website:** Visit [payaam.dev/projects/mma-predictive-modeling](https://payaam.dev/projects/mma-predictive-modeling) after merging

**📚 In This File:** Links to all completed experiments below

---

## Completed Experiments

<!-- EXPERIMENTS_LIST_START -->

### 1. [Increased Regularization Across Models](./increased-regularization/experiment_report.md)

**Date:** October 20, 2024
**Status:** ⚠️ Mixed Results

**Hypothesis:** Increasing regularization strength should reduce overfitting, particularly for Decision Tree which showed 100% training accuracy.

**Key Findings:**

- ❌ Tree-based models (Decision Tree, Random Forest, Gradient Boosting) over-regularized with accuracy drops of 7-13%
- 🟡 Neural networks (FNN, Transformer) showed minor performance loss (~1%)
- ✅ Successfully confirmed Decision Tree overfitting issue
- 💡 Learned that combined regularization techniques compound effects

**Outcome:** Regularization approach validated but applied too aggressively. Follow-up experiment recommended with moderate settings.

[View Full Report →](./increased-regularization/experiment_report.md)

---

<!-- EXPERIMENTS_LIST_END -->

---

## How This Works

When an experiment completes and generates `experiment_report.md`, this file is automatically updated to include a link and summary of that experiment. The list above is maintained between the `<!-- EXPERIMENTS_LIST_START -->` and `<!-- EXPERIMENTS_LIST_END -->` markers.
