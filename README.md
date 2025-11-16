# MMA Predictive Modeling

MMA Predictive Modeling (MPM) is a machine learning system designed to predict Mixed Martial Arts (MMA) fight outcomes using various models.

MMA fight prediction is impossible to perfect because real outcomes depend on unpredictable human behavior, hidden factors like injuries or mindset, and inherent randomness, including lucky shots and judging errors. The sport is chaotic and the data is incomplete, so no model can guarantee perfect accuracy.

## Project Overview

### Key Features

Fully automated end-to-end pipeline featuring:

- **Data collection**: Scraping of latest fight results and upcoming matchups
- **Model retraining**: Continuous updates using new fight data
- **Fight predictions**: Upcoming matchups analyzed with confidence scores for each model
- **Live deployment**: Results published to website

### 🧠 **Machine Learning Models**

The system implements the following models for predictive analysis:

- **Classical ML:** K-Nearest Neighbors (KNN), Naive Bayes, Logistic Regression, Support Vector Machines (SVM), Decision Trees
- **Neural Networks:** Feedforward Neural Network (FNN), Transformer
- **Ensemble Methods:** Gradient Boosting, Random Forest

### 📊 **Data Processing & Feature Engineering**

The data pipeline transforms raw fight statistics into predictive features:

- **Physical attributes:** Height, reach, age, and fighting stance
- **Historical performance:** Chronologically computed statistics including win/loss records, finish rates, average fight time, and time since last fight
- **Fighting metrics:** Strike accuracy, average strikes landed/attempted, takedown rates, control time, submission attempts, and reversals
- **Preprocessing:** StandardScaler normalization for numerical features and OneHotEncoder for categorical features using scikit-learn pipelines

### 🎯 **Training & Model Evaluation**

Models are trained and evaluated using a rigorous validation framework:

- **Train/test split:** 80/20 split on historical fight data
- **Learning curves:** 5-fold cross-validation across training sizes from 20% to 100% of data
- **Optimization:** PyTorch models trained with Adam optimizer and CrossEntropyLoss
- **Metrics:** Train/test accuracy with standard deviations, plus precision, recall, and F1-score via classification reports
- **Visualization:** Learning curves and accuracy progression plots saved for each model

## Automated ML Pipeline

This project features a complete end-to-end automated machine learning pipeline that handles **data ingestion**, **model training**, and **inference** for MMA fight prediction. The system operates on AWS infrastructure with three main automated workflows:

### 📥 **Data Ingestion Pipeline**

Automatically collects the latest MMA fight data:

1. **EventBridge Rule** triggers scheduled scraping tasks
2. **ECS Task** runs containerized web scraper in Docker
3. Historical fight results and upcoming matchups are extracted
4. Data is uploaded to **S3** (CSV for historical, JSON for upcoming)
5. **S3 upload events** trigger downstream training or inference pipelines

### 🔄 **Training Pipeline**

Maintains and updates the machine learning models with the latest fight data:

1. **S3-triggered Lambda** initiates **SageMaker training job**
2. **SageMaker** loads historical fight data and retrains all models using PyTorch + scikit-learn
3. Model metrics, learning curves, and updated models are saved to **S3**
4. **Lambda function** creates a **GitHub Pull Request** with new results

### 🎯 **Inference Pipeline**

Generates predictions for upcoming MMA fights:

1. **S3-triggered Lambda** starts **SageMaker inference job**
2. **SageMaker** loads trained models and generates predictions for upcoming matchups
3. Predictions with confidence scores are saved to **S3**
4. Results are automatically displayed on the project website

## Results

Learning curves, past results, future predictions, and model comparisons can be viewed here: [**payaam.dev/projects/mma-predictive-modeling**](https://payaam.dev/projects/mma-predictive-modeling)
