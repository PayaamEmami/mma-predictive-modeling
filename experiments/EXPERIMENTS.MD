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

**Status:** ⚠️ Mixed Results

**Hypothesis:** Replacing absolute Fighter1/Fighter2 features with differential (Fighter1 - Fighter2) features plus derived stats (WinRate, TakedownAccuracy, TotalFights, AvgGroundStrikes, AvgHeadStrikes) should improve model accuracy by providing the comparison signal directly and reducing feature dimensionality.

**Key Findings:**

- ⚠️ 5 of 9 models improved, 1 flat, 3 degraded — most changes within statistical noise (~1.2% standard error)
- ❌ Gradient Boosting (previously best model at 62.53%) dropped to 59.64% (-2.89%)
- ❌ Random Forest declined from 61.73% to 60.63% (-1.10%), SVM from 61.00% to 60.07% (-0.93%)
- ✅ Transformer improved from 60.63% to 62.04% (+1.41%), becoming the new top model with the healthiest learning curve
- ✅ Decision Tree improved from 52.09% to 54.61% (+2.52%), KNN from 52.83% to 53.87% (+1.04%)
- 🟡 Logistic Regression improved from 60.44% to 61.18% (+0.74%), FNN from 59.89% to 60.44% (+0.55%)
- 💡 Tree-based ensembles (GB, RF) degraded — they can learn feature interactions natively and lost information from the transformation
- 💡 Neural networks and linear models benefited from having the comparison signal pre-computed

**Outcome:** Results are inconclusive rather than clearly negative. The differential representation benefits models that cannot easily learn feature interactions (Transformer, Logistic Regression, FNN) while hurting tree-based ensembles that could already discover comparisons from absolute values. Change merged into main as the theoretical foundation is sound for a matchup prediction task and the new top model (Transformer) shows the healthiest train/test gap. A hybrid approach (differential + select absolute features) may capture the best of both worlds.

[View Full Report →](./differential-features/)

---

<!-- EXPERIMENTS_LIST_END -->

---

## How This Works

When an experiment completes and generates `experiment_report.md`, this file is automatically updated to include a link and summary of that experiment. The list above is maintained between the `<!-- EXPERIMENTS_LIST_START -->` and `<!-- EXPERIMENTS_LIST_END -->` markers.
