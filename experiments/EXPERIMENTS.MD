# Experiments Index

## 1. [Increased Regularization Across Models](./increased-regularization/)

**Status:** ⚠️ Mixed Results

**Hypothesis:** Increasing regularization strength should reduce overfitting, particularly for Decision Tree which showed 100% training accuracy.

**Key Findings:**

- ❌ Tree-based models (Decision Tree, Random Forest, Gradient Boosting) over-regularized with accuracy drops of 7-13%
- 🟡 Neural networks (FNN, Transformer) showed minor performance loss (~1%)
- ✅ Successfully confirmed Decision Tree overfitting issue
- 💡 Learned that combined regularization techniques compound effects

**Outcome:** Regularization approach validated but applied too aggressively. Follow-up experiment recommended with moderate settings.

---

## 2. [Dropout Regularization for Neural Networks](./dropout-regularization/)

**Status:** ❌ Negative Results

**Hypothesis:** Adding dropout regularization to FNN and Transformer models should reduce overfitting and improve generalization by preventing neuron co-adaptation.

**Key Findings:**

- ❌ FNN performance decreased from 62.09% to 61.46% (-1.0%)
- ❌ Transformer performance decreased from 60.82% to 60.69% (-0.2%)
- 💡 Current models are already appropriately regularized with weight decay
- 💡 Small dataset + simple architectures mean capacity matters more than additional regularization
- ✅ Validated that baseline hyperparameters are well-tuned for this problem

**Outcome:** Dropout provides no benefit for our MMA prediction task. The baseline models with weight decay alone perform better. Change will not be merged. Experiment demonstrates that aggressive regularization is counterproductive when models aren't severely overfitting.

---

## 3. [Differential Features](./differential-features/)

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

---

## 4. [Temporal Split](./temporal-split/)

**Status:** ✅ Methodology Improvement

**Hypothesis:** Replacing random train/test split with chronological (temporal) split will eliminate data leakage where test fight outcomes leak into training features via `compute_historical_stats`. Additionally, fitting StandardScaler only on training data and using TimeSeriesSplit for learning curve CV will remove remaining leakage sources. Accuracy numbers are expected to drop, reflecting a more honest evaluation that matches the production prediction scenario.

**Key Findings:**

- ✅ Confirmed temporal leakage was inflating results — tree-based ensembles hit hardest: Random Forest -3.68%, SVM -2.58%, Gradient Boosting -2.52%
- ✅ Logistic Regression became the top model at 61.00% (+0.56%), proving its performance was genuine and not leakage-driven
- ✅ FNN held steady at 60.44% (+0.55%), also resilient to the leakage fix
- 🟡 Transformer dropped marginally from 60.63% to 60.44% (-0.19%), essentially flat
- 🟡 KNN improved from 52.83% to 54.61% (+1.78%), Decision Tree from 52.09% to 53.50% (+1.41%)
- 💡 Models with high train/test gaps (RF: 91.9%/58.1%, DT: 92.8%/53.5%) reveal severe overfitting now visible without leakage masking it
- 💡 Simpler models (LR, NB) and well-regularized neural networks (FNN, Transformer) were least affected, confirming they weren't relying on leaked information

**Outcome:** The temporal split produces more honest evaluation metrics that reflect real-world production performance. The 2-4% drop in top models is not a regression — it reveals previously hidden leakage inflation. Change will be merged as the correct evaluation methodology going forward.

---