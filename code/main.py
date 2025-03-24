# main.py

import torch
import pandas as pd
from data import load_ufc_data
from models import initialize_models
from evaluation import evaluate_models
from training import time_series_cross_validate
from config import DEVICE

def main():
    # load and preprocess data
    X_ufc, y_ufc, label_encoder = load_ufc_data()

    if X_ufc is None:
        print("Failed to load UFC data.")
        exit(1)

    # Convert numpy arrays to DataFrame for time series cross-validation
    fight_data = pd.DataFrame(X_ufc)
    fight_data['Winner'] = y_ufc
    fight_data['EventDate'] = pd.date_range(start='1994-03-11', periods=len(fight_data), freq='D')  # Approximate dates

    # Get feature columns and target column
    feature_columns = [col for col in fight_data.columns if col not in ['Winner', 'EventDate']]
    target_column = 'Winner'

    # input_size is the number of features in the training dataset
    input_size = len(feature_columns)

    # initialize models to train and analyze
    models = initialize_models(input_size, DEVICE)

    # Perform time series cross-validation and get final evaluation data
    print("\nStarting Time Series Cross-Validation...")
    _, train_data, test_data = time_series_cross_validate(
        data=fight_data,
        models=models,
        feature_columns=feature_columns,
        target_column=target_column,
        device=DEVICE,
        n_splits=5,
        test_size=0.2,
        min_train_size=1000
    )
    
    # Prepare data for final evaluation
    X_train = train_data[feature_columns].values
    y_train = train_data[target_column].values
    X_test = test_data[feature_columns].values
    y_test = test_data[target_column].values

    # Evaluate models and generate plots
    print("\nGenerating final evaluation plots...")
    evaluate_models(models, X_train, X_test, y_train, y_test, label_encoder, DEVICE)

if __name__ == "__main__":
    main()
