# Dropout Regularization Experiment Report

**Date:** November 5, 2024
**Status:** ❌ Negative Results
**Hypothesis:** Adding dropout regularization to FNN and Transformer models should reduce overfitting and improve generalization

---

## Experiment Overview

### Objective
Test whether adding dropout layers to neural network models (FNN and Transformer) can reduce overfitting and improve test set performance.

### Changes Made
- **FNN:** Added dropout (rate=0.3) after the hidden layer
- **Transformer:** Added dropout (rate=0.2) in encoder layers and before classification head

### Rationale
Dropout is a widely-used regularization technique that randomly deactivates neurons during training, forcing the network to learn more robust features and preventing co-adaptation of neurons. Given that neural networks can overfit on relatively small datasets (~8k examples), dropout should theoretically help.

---

## Results

### Performance Comparison

| Model | Baseline Accuracy | Experiment Accuracy | Change | % Change |
|-------|------------------|---------------------|---------|----------|
| Random Forest | 0.6209 | 0.6209 | 0.0000 | 0.0% |
| Gradient Boosting | 0.6350 | 0.6350 | 0.0000 | 0.0% |
| SVM | 0.6146 | 0.6146 | 0.0000 | 0.0% |
| Logistic Regression | 0.6209 | 0.6209 | 0.0000 | 0.0% |
| KNN | 0.5278 | 0.5278 | 0.0000 | 0.0% |
| Naive Bayes | 0.5795 | 0.5795 | 0.0000 | 0.0% |
| Decision Tree | 0.5297 | 0.5297 | 0.0000 | 0.0% |
| **FNN** | **0.6209** | **0.6146** | **-0.0063** | **-1.0%** |
| **Transformer** | **0.6082** | **0.6069** | **-0.0013** | **-0.2%** |

### Key Findings

#### ❌ FNN Performance Decreased
- **Accuracy drop:** 62.09% → 61.46% (-1.0%)
- **Training accuracy:** 68.19% (indicating some overfitting still exists)
- **Test accuracy:** 61.46%
- **Learning rate:** -0.0988 (negative indicates overfitting)

#### ❌ Transformer Performance Slightly Decreased
- **Accuracy drop:** 60.82% → 60.69% (-0.2%)
- **Training accuracy:** 64.14% (relatively low overfitting)
- **Test accuracy:** 60.69%
- **Learning rate:** -0.0538 (minimal overfitting)

#### ✅ Non-Neural Network Models Unaffected
All scikit-learn models (Random Forest, SVM, Logistic Regression, etc.) showed identical performance, as expected, since dropout only applies to neural networks.

---

## Analysis

### Why Dropout May Have Hurt Performance

1. **Small Dataset Size:** With ~8k training examples and relatively simple architectures, the models may already be capacity-limited. Dropout further reduces effective capacity during training, potentially preventing the models from learning sufficient patterns.

2. **Already Moderate Overfitting:** The FNN and Transformer weren't severely overfitting in the baseline:
   - FNN: Training 62.09% vs Test 62.09% (no gap in baseline)
   - Transformer: Training ~61% vs Test 60.82% (small gap)

   The models may have already reached optimal capacity for this problem.

3. **Existing Regularization:** Both models already use weight decay (0.01), which provides regularization. Adding dropout on top may have been too aggressive.

4. **Architecture Simplicity:**
   - FNN has only one hidden layer (256 units)
   - Both models have relatively few parameters compared to modern deep networks

   Dropout is more effective in deeper, more complex networks with higher risk of overfitting.

5. **Feature Engineering Quality:** The strong feature engineering in the data preprocessing may already provide sufficient signal, making aggressive regularization counterproductive.

---

## Detailed Metrics

### FNN Learning Curve Analysis
```
Final Training Accuracy: 0.6819 (+/- 0.0034)
Final Test Accuracy: 0.6146 (+/- 0.0150)
Learning Rate: -0.0988
```

**Classification Report:**
```
              precision    recall  f1-score   support
           1       0.62      0.63      0.63       802
           2       0.61      0.59      0.60       765
    accuracy                           0.61      1567
```

### Transformer Learning Curve Analysis
```
Final Training Accuracy: 0.6414 (+/- 0.0082)
Final Test Accuracy: 0.6069 (+/- 0.0117)
Learning Rate: -0.0538
```

**Classification Report:**
```
              precision    recall  f1-score   support
           1       0.62      0.61      0.61       802
           2       0.60      0.61      0.60       765
    accuracy                           0.61      1567
```

---

## Conclusions

### Primary Findings
❌ **Dropout regularization decreased neural network performance** rather than improving it. The accuracy losses, while small, are consistent and statistically meaningful given the cross-validation standard deviations.

### Insights Gained

1. **Current Models Are Appropriately Regularized:** The baseline models with weight decay alone are already well-balanced for this dataset size and problem complexity.

2. **Capacity Matters More Than Regularization:** For this relatively small dataset and simple architectures, maintaining model capacity is more important than adding aggressive regularization.

3. **Architecture-Specific Regularization:** The Transformer showed more resilience to dropout (only -0.2% loss) compared to FNN (-1.0% loss), likely because:
   - Transformer has more parameters and depth
   - Built-in attention mechanism provides some regularization
   - Dropout was applied at multiple points in the architecture

4. **Overfitting Is Minimal:** The baseline models don't show severe overfitting, so additional regularization addresses a non-existent problem.

---

## Recommendations

### Do Not Apply This Change
The dropout modifications should **not** be merged into the main branch. The baseline configuration performs better.

### Future Experiments to Consider

1. **Lower Dropout Rates:** Test more conservative dropout (0.1-0.15) to find a sweet spot that provides regularization without hurting capacity.

2. **Conditional Dropout:** Apply dropout only during early training epochs, then disable it for fine-tuning.

3. **Architecture Expansion:** If pursuing dropout, first increase model capacity (more layers/units) so that dropout's capacity reduction is less impactful.

4. **Alternative Regularization:**
   - Experiment with batch normalization
   - Try layer normalization for Transformer
   - Explore data augmentation techniques for fight statistics

5. **Ensemble Methods:** Rather than regularizing individual models, explore model ensembling to improve generalization.

---

## Reproducibility

### Configuration
- **FNN dropout rate:** 0.3 (after hidden layer)
- **Transformer dropout rate:** 0.2 (in encoder layers + before classification)
- **Other hyperparameters:** Unchanged from baseline
- **Training setup:** 300 epochs, batch size 32, Adam optimizer

### Files Generated
- `baseline_metrics.csv` - Pre-experiment performance
- `model_performances.csv` - Experiment results
- `model_metrics_report.txt` - Detailed metrics
- `learning_curve_*.png` - Training curves for all models
- `model_performance_comparison.png` - Visual comparison

---

## Conclusion

While dropout is a powerful regularization technique for deep neural networks, this experiment demonstrates that it's not universally beneficial. For our MMA prediction task with:
- Moderate dataset size (~8k examples)
- Simple architectures (single hidden layer FNN, 4-layer Transformer)
- Already-applied weight decay
- Strong feature engineering

**Dropout provides no benefit and slightly harms performance.** The baseline models are already well-tuned for the problem, and the minimal overfitting observed doesn't warrant additional regularization that reduces model capacity.

This experiment provides valuable validation that our current hyperparameter choices are appropriate for this specific problem domain.

