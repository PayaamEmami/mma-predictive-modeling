# MMA Predictive Modeling

MMA Predictive Modeling (MPM) is a fully automated end-to-end machine learning system designed to predict mixed martial arts (MMA) fight outcomes based on historical data and fighter statistics.

MPM features a fully automated end-to-end pipeline that scrapes the latest fight results and upcoming matchups, continuously retrains models using new fight data, generates predictions with confidence scores for each model, and publishes results to the project website.

MMA fight prediction is impossible to perfect because real outcomes depend on unpredictable human behavior, hidden factors like injuries or mindset, and inherent randomness, including lucky shots and judging errors. The sport is chaotic and the data is incomplete, so no model can guarantee perfect accuracy.

## Results

Learning curves, past results, future predictions, and model comparisons can be viewed here: [**payaam.dev/projects/mpm**](https://payaam.dev/projects/mpm)

## Technology Stack

### Core Technologies
- **Python 3.12** - Primary programming language
- **PyTorch** - Deep learning framework for neural networks
- **scikit-learn** - Classical machine learning algorithms

### Machine Learning & Data Science
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing and array operations
- **matplotlib** - Data visualization and learning curves
- **BeautifulSoup4** - HTML parsing for web scraping

### Cloud & Infrastructure (AWS)
- **AWS SageMaker** - Model training and inference jobs
- **AWS Lambda** - Serverless function orchestration
- **AWS S3** - Data storage and model persistence
- **AWS EventBridge** - Scheduled task automation
- **AWS ECS (Fargate)** - Containerized scraper execution
- **AWS CloudWatch** - Monitoring and logging
- **AWS Parameter Store** - Secrets management

### DevOps & Deployment
- **Docker** - Containerization for scraper tasks
- **GitHub Actions** - Automated pull requests for results
- **pytest** - Unit and integration testing

### Web Scraping
- **HTTP Client** - Direct HTML fetching with rate limiting
- **Functional Design** - Pure functions for data extraction
- **Incremental Updates** - Event tracking to avoid duplicates

## Project Overview

### 🧠 **Machine Learning Models**

The system implements the following models for predictive analysis:

- **Classical ML:** K-Nearest Neighbors (KNN), Naive Bayes, Logistic Regression, Support Vector Machines (SVM), Decision Trees
- **Neural Networks:** Feedforward Neural Network (FNN), Transformer
- **Ensemble Methods:** Gradient Boosting, Random Forest

### 📊 **Data Processing & Feature Engineering**

The data pipeline transforms raw fight statistics into predictive features:

- **Differential features:** Instead of raw Fighter1/Fighter2 stats, the pipeline computes Fighter1 - Fighter2 differences for all numerical features, directly encoding the matchup comparison signal
- **Physical attributes:** Height, reach, and age differentials between fighters, plus individual fighting stances
- **Historical performance:** Chronologically computed statistics including win/loss records, finish rates, average fight time, and time since last fight
- **Fighting metrics:** Strike accuracy, average strikes landed/attempted, takedown rates, control time, submission attempts, and reversals
- **Preprocessing:** StandardScaler normalization and OneHotEncoder for stances, fit on training data only to prevent data leakage

### 🎯 **Training & Model Evaluation**

Models are trained and evaluated using a rigorous validation framework:

- **Chronological data split:** 70/20/10 training/validation/test split ordered by event date to prevent temporal data leakage
- **Learning curves:** 5-fold temporal cross-validation (TimeSeriesSplit) across training sizes from 20% to 100%
- **Early stopping:** FNN and Transformer use patience-based early stopping on validation loss; Gradient Boosting uses sklearn's built-in `n_iter_no_change`
- **Optimization:** PyTorch models trained with Adam optimizer and CrossEntropyLoss
- **Metrics:** Training/validation accuracy with standard deviations, plus precision, recall, and F1-score via classification reports. Held-out test set evaluated separately for unbiased final metrics.
- **Visualization:** Learning curves and model comparison plots saved for each model

## Automated ML Pipeline

This project features a complete end-to-end automated machine learning pipeline that handles **data ingestion**, **model training**, and **inference** for MMA fight prediction. The system operates on AWS infrastructure with three main automated workflows:

### 📥 **Data Ingestion Pipeline**

Automatically collects the latest MMA fight data:

1. **EventBridge Rule** triggers scheduled scraping tasks
2. **ECS Task** runs containerized scraper (`scraper/` package)
3. Historical fight results and upcoming matchups are extracted from data sources
4. Data is uploaded to **S3** (CSV for historical data, JSON for upcoming fights)
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

## Data Scraper

The project includes a data scraper that automatically collects fight data from public sources.

### ✨ **Features**

- **Historical Data Mode:** Scrapes all completed MMA events with comprehensive fight statistics
- **Upcoming Fights Mode:** Extracts matchup information for future events to generate predictions
- **S3 Integration:** Downloads existing data, updates it, and uploads back to cloud storage
- **Incremental Updates:** Only processes new events that haven't been scraped yet

### 📋 **Data Collected**

Each fight record includes 39 fields:
- **Event details:** Name, date, location
- **Fighter profiles:** Name, DOB, height, reach, stance, profile URL
- **Fight statistics:** Knockdowns, takedowns, submissions, reversals, control time
- **Strike statistics:** Significant strikes, head/body/leg strikes, distance/clinch/ground strikes
- **Fight outcome:** Method, round, time, winner
