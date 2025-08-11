# MMA Predictive Modeling

MMA Predictive Modeling (MPM) is an advanced machine learning system that predicts Mixed Martial Arts (MMA) fight outcomes using ensemble methods and comprehensive statistical analysis. This project combines classical machine learning algorithms, deep neural networks, and modern ensemble techniques to analyze fighter performance data and generate intelligent predictions for upcoming MMA events.

Built with Python using frameworks including scikit-learn and PyTorch, the system features automated data processing, model training, and real-time inference capabilities. The architecture maintains all fight data in structured CSV format for efficient updates and management, while generating comprehensive visualizations including model comparison charts, learning curves, and detailed performance analytics.

MMA fight prediction is, in theory, a well-posed input-output problem: given data, output a prediction (e.g., Fighter A wins or Fighter B wins). However, in practice, accurate fight prediction is extremely challenging. Real-world outcomes depend on human behavior (which is non-deterministic and chaotic), incomplete data (such as injuries, mindset, or training changes), and inherent randomness (like lucky punches or judging errors). This introduces uncertainty and irreducible complexity, making perfect prediction impossible. MMA fight prediction is practically intractable to solve perfectly. No model can guarantee perfect accuracy due to real-world randomness and incomplete information.

## Project Overview

This comprehensive machine learning system employs a diverse array of algorithms and sophisticated data processing techniques to analyze MMA fight outcomes. The project combines traditional statistical methods with modern deep learning approaches to create robust predictive models.

### 🧠 **Machine Learning Models**

The system implements **9 distinct algorithms** across multiple paradigms for comprehensive predictive analysis:

- **Classical ML:** K-Nearest Neighbors (KNN), Naive Bayes, Logistic Regression, Support Vector Machines (SVM), Decision Trees
- **Neural Networks:** Fully Connected Neural Network (FCNN), Transformers
- **Ensemble Methods:** Gradient Boosting, Random Forest

Each model contributes unique strengths to the ensemble, from the interpretability of decision trees to the pattern recognition capabilities of neural networks.

### 📊 **Data Processing & Feature Engineering**

The data pipeline transforms raw fight statistics into meaningful predictive features through:

- **Multi-source data extraction:** Fighter profiles, historical performance metrics, and event-specific contextual data
- **Advanced feature engineering:** Derivation of key performance indicators including striking accuracy, takedown defense rates, fight finish percentages, and momentum-based metrics
- **Intelligent preprocessing:** Automated handling of missing values, feature normalization, and outlier detection to ensure robust model training
- **Temporal analysis:** Integration of career progression trends and recent performance patterns

### 🎯 **Training & Model Evaluation**

The training framework ensures rigorous model development and validation:

- **Comparative analysis:** Systematic evaluation across all 9 algorithms using consistent training protocols
- **Comprehensive datasets:** Models trained on extensive historical fight data with careful train/validation/test splits
- **Performance visualization:** Generation of detailed learning curves and accuracy progression plots
- **Multi-metric evaluation:** Assessment using accuracy, precision, recall, and F1-score
- **Cross-validation:** Robust validation strategies to ensure generalization and prevent overfitting

## Automated ML Pipeline

This project features a complete end-to-end automated machine learning pipeline that handles both **model training** and **inference** for MMA fight prediction. The system operates on AWS infrastructure and provides continuous updates with minimal manual intervention.

The system consists of two main automated workflows:

### 🔄 **Training Pipeline** (Weekly - Sundays)

Maintains and updates the machine learning models with the latest fight data:

1. **EventBridge Rule** triggers every Sunday
2. **ECS Task** runs a .NET + Playwright data scraper in Docker
3. Latest fight data is uploaded to **S3** as CSV
4. **S3-triggered Lambda** initiates **SageMaker training job**
5. **SageMaker** retrains all 9 ML models using PyTorch + scikit-learn
6. Model metrics, learning curves, and updated models are saved to **S3**
7. **Lambda function** creates a **GitHub Pull Request** with new results

### 🎯 **Inference Pipeline** (Weekly - Fridays)

Generates predictions for upcoming MMA fights:

1. **EventBridge Rule** triggers every Friday
2. **ECS Task** scrapes upcoming fight data in "prediction mode"
3. Fighter matchup data is uploaded to **S3** as JSON
4. **S3-triggered Lambda** starts **SageMaker inference job**
5. **SageMaker** loads trained models and generates ensemble predictions
6. Predictions with confidence scores are saved to **S3**
7. Results are automatically displayed on the project website

### Key Features

- **Multi-model ensemble**: Predictions from 9 different ML algorithms
- **Confidence scoring**: Each prediction includes model agreement and confidence levels
- **Automated data ingestion**: Continuous scraping of latest MMA fight data
- **Real-time deployment**: Predictions automatically published to live website
- **Model persistence**: Trained models and latest predictions stored in S3
- **Zero manual intervention**: Complete automation from data collection to result publication

## Results

Explore the comprehensive outcomes and insights from the MMA predictive modeling system: [**payaam.dev/projects/mma-predictive-modeling**](https://payaam.dev/projects/mma-predictive-modeling)

This interactive results page showcases both aspects of the machine learning pipeline:

### 🤖 **Training Results**

- **Learning curves** for all 9 machine learning models showing training progression
- **Model performance comparisons** with accuracy metrics and visual analytics
- **Interactive plot viewer** for detailed examination of model behavior
- **Real-time updates** reflecting the latest model training cycles

### 🥊 **Live Fight Predictions**

- **Current predictions** for upcoming MMA events with ensemble model consensus
- **Confidence scores** and individual model breakdowns for each fight
- **Fighter matchup analysis** displaying predicted winners and probability distributions
- **Multi-model voting** showing agreement levels across different algorithms

The results page provides a complete view of both the machine learning development process and the practical application of the trained models to real-world fight predictions.
