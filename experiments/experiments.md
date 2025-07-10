<!--
## Experiment Name (A short, descriptive title for the experiment).

Briefly explain the motivation, hypothesis, and setup for the experiment. What question are you trying to answer or what are you testing?

**Results:**
Summarize the key findings and metrics. Include relevant plots, tables, or images.
![Description](path/to/image.png)

**Discussion:**
Interpret the results, discuss implications, limitations, and possible next steps or follow-up experiments.
-->

# Experiments

## Table of Contents

- [Scalar Normalization](#scalar-normalization)

This file documents structured machine learning experiments. Each entry includes the experiment name, a description of the setup and motivation, results with key metrics and visuals, and a discussion of findings, limitations, and next steps. This format supports reproducibility, comparison, and ongoing project development.

## Scalar Normalization

Scalar normalization (such as standardization or min-max scaling) ensures that all input features are on a comparable scale, which is especially important for distance-based models (like KNN), gradient-based optimizers (neural networks), and regularized models. To assess the impact of normalization, I trained all models with and without applying scalar normalization to the input features and compared their performance.

**Results:**

During data preprocessing, omitting scalar normalization consistently led to approximately a 10% decline in accuracy across all models.

![Model Accuracy Comparison](scalar-normalization/model_accuracy_comparison.png)

**Discussion:**

Models trained on unnormalized data struggled to learn meaningful patterns, likely due to imbalanced feature influence and unstable gradients. Without normalization, features with larger numeric ranges can dominate the learning process, resulting in suboptimal performance and poor convergence. The observed accuracy drop highlights the crucial role of normalization in ensuring model effectiveness across both classical ML and deep learning approaches. These findings underscore that normalization is not just beneficial but essential for reliable, stable, and comparable model performance in the MMA prediction task.
