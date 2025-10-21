# Increased Regularization Across Models

**Date:** October 20, 2024
**Status:** ⚠️ Mixed Results - Partial Success

---

## Hypothesis

Increasing regularization strength across models should reduce overfitting and improve generalization on the test set, given the relatively small dataset (~8k examples). Decision Tree in particular has no depth limit and may be severely overfitting.

---

## Changed Hyperparameters

### Decision Tree

- `max_depth`: `null` → `10`
- `ccp_alpha`: `0.0` → `0.01`
- `min_samples_leaf`: `1` → `5`

### Random Forest

- `ccp_alpha`: `0.0` → `0.01`
- `max_depth`: `10` → `8`

### Gradient Boosting

- `ccp_alpha`: `0.0` → `0.01`
- `max_depth`: `3` → `2`

### FNN (Neural Network)

- `weight_decay`: `0.01` → `0.05`

### Transformer

- `weight_decay`: `0.01` → `0.05`

---

## Results Summary

| Model                 | Baseline Accuracy | Experiment Accuracy | Change      | Status                 |
| --------------------- | ----------------- | ------------------- | ----------- | ---------------------- |
| **Random Forest**     | 61.39%            | 54.12%              | **-7.27%**  | ❌ Degraded            |
| **Gradient Boosting** | 62.60%            | 49.39%              | **-13.21%** | ❌❌ Severely Degraded |
| **Decision Tree**     | 56.92%            | 49.39%              | **-7.53%**  | ❌ Degraded            |
| **FNN**               | 62.28%            | 61.07%              | **-1.21%**  | 🟡 Minor Loss          |
| **Transformer**       | 62.35%            | 61.39%              | **-0.96%**  | 🟡 Minor Loss          |
| SVM                   | 62.35%            | 62.35%              | 0.00%       | ✅ Unchanged           |
| Logistic Regression   | 61.07%            | 61.07%              | 0.00%       | ✅ Unchanged           |
| KNN                   | 52.27%            | 52.27%              | 0.00%       | ✅ Unchanged           |
| Naive Bayes           | 55.90%            | 55.90%              | 0.00%       | ✅ Unchanged           |

---

## Detailed Analysis

### 🔴 Critical Issues

#### Gradient Boosting - Catastrophic Failure

- **Training Accuracy:** 51.23% (baseline: 72.55%)
- **Test Accuracy:** 49.39% (baseline: 62.60%)
- **Problem:** Model collapsed to predicting only one class
  - Precision for class 2: **0.00** (no predictions made)
  - Recall for class 1: **100%** (predicts everything as class 1)
- **Root Cause:** Over-regularization from combination of `max_depth: 2` and `ccp_alpha: 0.01`

#### Decision Tree - Severe Underfitting

- **Training Accuracy:** 51.23% (baseline: 100%)
- **Test Accuracy:** 49.39% (baseline: 56.92%)
- **Problem:** Same collapse as Gradient Boosting
- **Analysis:** Successfully eliminated overfitting (100% → 51% training acc), but regularization too aggressive
- **Original Issue Confirmed:** Was severely overfitting with perfect training accuracy

#### Random Forest - Significant Performance Drop

- **Training Accuracy:** 54.32% (baseline: 90.70%)
- **Test Accuracy:** 54.12% (baseline: 61.39%)
- **Problem:** Over-constrained, now underfitting
- **Analysis:**
  - Precision drop for class 1: 61% → 58%
  - Recall drop for class 1: 63% → 27% (**severe**)
  - Model became too conservative

### 🟡 Minor Degradations

#### FNN (Feedforward Neural Network)

- **Training Accuracy:** 60.43% (baseline: 67.96%)
- **Test Accuracy:** 61.07% (baseline: 62.28%)
- **Change:** -1.21%
- **Analysis:** Weight decay increase (0.01 → 0.05) had minimal impact, slightly more regularization than needed

#### Transformer

- **Training Accuracy:** 62.96% (baseline: 63.43%)
- **Test Accuracy:** 61.39% (baseline: 62.35%)
- **Change:** -0.96%
- **Analysis:** Similar to FNN, weight decay increase slightly over-regularized but not critically

### ✅ Unchanged Models

Models without hyperparameter changes maintained baseline performance:

- **SVM**, **Logistic Regression**, **KNN**, **Naive Bayes**

---

## Key Insights

### ✅ Hypothesis Validation

1. **Decision Tree overfitting confirmed:** Baseline showed 100% training accuracy vs 56.92% test accuracy
2. **Regularization approach was correct** in principle
3. **Magnitude was the problem:** Applied too aggressively

### ⚠️ Critical Learnings

1. **Tree-based models are highly sensitive to regularization:**

   - `max_depth: 3 → 2` in Gradient Boosting caused collapse
   - `max_depth: 10` + `ccp_alpha: 0.01` + `min_samples_leaf: 5` over-constrained Decision Tree
   - Random Forest depth reduction (10 → 8) combined with pruning too restrictive

2. **Neural networks are more robust:**

   - 5x increase in weight_decay (0.01 → 0.05) only caused ~1% accuracy loss
   - Suggests these models can handle stronger regularization

3. **Combined regularization effects compound:**
   - Using multiple regularization techniques simultaneously (depth limits, pruning, minimum samples) can over-constrain

---

## Recommendations

### For Tree-Based Models

#### Decision Tree

```json
{
  "max_depth": 15, // More breathing room (was 10)
  "min_samples_leaf": 3, // Less restrictive (was 5)
  "ccp_alpha": 0.005 // Lighter pruning (was 0.01)
}
```

#### Gradient Boosting

```json
{
  "max_depth": 4, // Increase from 2 (baseline was 3)
  "ccp_alpha": 0.0, // Remove pruning entirely for now
  "learning_rate": 0.05 // Alternative: reduce learning rate instead
}
```

#### Random Forest

```json
{
  "max_depth": 12, // Moderate reduction from baseline 10
  "ccp_alpha": 0.0, // Remove pruning for now
  "min_samples_leaf": 2 // Slight increase from baseline 1
}
```

### For Neural Networks

Current regularization is acceptable but could be fine-tuned:

```json
{
  "FNN": {
    "weight_decay": 0.03 // Between 0.01 and 0.05
  },
  "Transformer": {
    "weight_decay": 0.03 // Between 0.01 and 0.05
  }
}
```

### Next Experiment Ideas

1. **Moderate Regularization:** Apply above recommendations
2. **Incremental Approach:** Test one regularization technique at a time
3. **Cross-validation tuning:** Use GridSearchCV to find optimal regularization per model
4. **Dropout for Neural Networks:** Add dropout layers instead of just weight decay

---

## Conclusion

This experiment successfully identified and confirmed the Decision Tree overfitting problem but applied regularization too aggressively, particularly for tree-based models. The results demonstrate that:

- ✅ Regularization is needed for tree-based models
- ❌ Current settings are too restrictive
- 🔄 A more gradual approach is recommended

The next experiment should apply moderate regularization with incremental testing to find the sweet spot between underfitting and overfitting.
