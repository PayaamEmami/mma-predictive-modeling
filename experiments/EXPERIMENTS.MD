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

### 2. [Dropout Regularization for Neural Networks](./dropout-regularization/experiment_report.md)

**Status:** ❌ Negative Results

**Hypothesis:** Adding dropout regularization to FNN and Transformer models should reduce overfitting and improve generalization by preventing neuron co-adaptation.

**Key Findings:**

- ❌ FNN performance decreased from 62.09% to 61.46% (-1.0%)
- ❌ Transformer performance decreased from 60.82% to 60.69% (-0.2%)
- 💡 Current models are already appropriately regularized with weight decay
- 💡 Small dataset + simple architectures mean capacity matters more than additional regularization
- ✅ Validated that baseline hyperparameters are well-tuned for this problem

**Outcome:** Dropout provides no benefit for our MMA prediction task. The baseline models with weight decay alone perform better. Change will not be merged. Experiment demonstrates that aggressive regularization is counterproductive when models aren't severely overfitting.

[View Full Report →](./dropout-regularization/experiment_report.md)

---

### 3. [Differential Features](./differential-features/)

**Status:** ❌ Negative Results

**Hypothesis:** Replacing absolute Fighter1/Fighter2 features with differential (Fighter1 - Fighter2) features plus derived stats (WinRate, TakedownAccuracy, TotalFights, AvgGroundStrikes, AvgHeadStrikes) should improve model accuracy by providing the comparison signal directly and reducing feature dimensionality.

**Key Findings:**

- ❌ 7 of 9 models degraded compared to baseline
- ❌ Gradient Boosting (previously best model at 62.95%) dropped to 59.64% (-3.31%)
- ❌ Random Forest, SVM, and FNN all declined by ~1-1.5%
- 🟡 Transformer showed marginal gain to 62.04% (+0.26%), becoming the top model
- 🟡 Decision Tree marginally improved to 54.61% (+0.58%), though still the weakest model
- 💡 Removing absolute feature values loses information that models were leveraging
- 💡 Dimensionality reduction did not compensate for the loss of individual fighter statistics

**Outcome:** Differential feature transformation does not improve prediction accuracy. Models perform better with access to both fighters' raw statistics rather than only the difference between them. The absolute features likely encode useful signals (e.g., overall fighter quality) that are lost when only the differential is kept. Change will not be merged.

[View Full Report →](./differential-features/)

---

<!-- EXPERIMENTS_LIST_END -->

---

## How This Works

When an experiment completes and generates `experiment_report.md`, this file is automatically updated to include a link and summary of that experiment. The list above is maintained between the `<!-- EXPERIMENTS_LIST_START -->` and `<!-- EXPERIMENTS_LIST_END -->` markers.
