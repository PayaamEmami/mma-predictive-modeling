# MMA Predictive Modeling

A machine learning project for predicting fight outcomes in Mixed Martial Arts (MMA) events. This project leverages classical machine learning algorithms, neural networks, and ensemble methods to analyze fighter statistics and historical fight data to generate insights and visualizations.

## Current Status

✅ Ongoing – The project is functional and actively maintained. It is periodically updated with the latest fight data after events.

## Features

### Machine Learning Models

- **Classical ML:** K-Nearest Neighbors (KNN), Naive Bayes, Logistic Regression, Support Vector Machines (SVM), Decision Trees
- **Neural Networks:** Fully Connected Neural Network (FCNN), Transformers
- **Ensemble Methods:** Gradient Boosting, Random Forest

### Data Processing & Preprocessing

- Extracts fighter statistics, fight history, and event-based features
- Feature engineering for significant fight attributes such as striking accuracy, takedown success, and fight duration
- Handles missing data and normalizes input features

### Training & Evaluation

- Implements multiple machine learning models for comparison
- Trains models using fight event data
- Generates learning curve plots
- Evaluates models with accuracy metrics and classification reports

## Development Notes

- Written in Python using Scikit-learn and PyTorch
- Data stored in structured CSV format for easy updates
- Outputs include predictive plots and model performance summaries

## Updating the Dataset

New fight event data is added periodically to keep the model updated. The pipeline processes new event data, retrains models, and generates updated performance evaluations.

## Contributing

🔒 Currently a solo project – external contributions are not planned at this time.

## Automated Training & Deployment Pipeline

This project includes a fully automated pipeline that handles data ingestion, model training, result generation, and repository updates. It scrapes the latest MMA event data, retrains all models, compares performance, and pushes the updated results to GitHub - all with no manual intervention.

![MMA Predictive Modeling Pipeline](aws/aws-pipeline.png)

**Pipeline Overview:**

1. **EventBridge Rule** triggers every Sunday.
2. **ECS Task** runs a .NET + Playwright data scraper inside a Docker container.
3. Scraped data is uploaded to **S3** as a CSV file.
4. An **S3-triggered Lambda** starts a **SageMaker training job**.
5. The job runs **Python modeling code** using PyTorch + scikit-learn, and generates metrics and plots.
6. Results are uploaded to **S3** in a `results/` folder.
7. Another **Lambda function** detects the results and creates a **GitHub Pull Request** with the updated output.

## Results

![Comparison of Model Accuracies](results/model_accuracy_comparison.png)
![Learning Curve for Random Forest](results/learning_curve_Random_Forest.png)
![Learning Curve for Gradient Boosting](results/learning_curve_Gradient_Boosting.png)
![Learning Curve for Support Vector Machine](results/learning_curve_SVM.png)
![Learning Curve for Logistic Regression](results/learning_curve_Logistic_Regression.png)
![Learning Curve for K-Nearest Neighbors](results/learning_curve_KNN.png)
![Learning Curve for Naive Bayes](results/learning_curve_Naive_Bayes.png)
![Learning Curve for Decision Tree](results/learning_curve_Decision_Tree.png)
![Learning Curve for Fully Connected Neural Network](results/learning_curve_FCNN.png)
![Learning Curve for Transformer](results/learning_curve_Transformer.png)
