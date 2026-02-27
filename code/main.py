import argparse
import numpy as np
from data import (
    load_fight_data,
    build_preprocessor,
    upload_results_to_s3,
    DIFF_FEATURE_SUFFIXES,
)
from models import initialize_models
from evaluation import evaluate_validation, evaluate_test
from training import train_model, save_models, save_label_encoder
from config import DEVICE, set_global_seed


def main(s3_bucket, s3_data_key, s3_results_prefix):
    """
    Main execution function for the fight prediction pipeline.

    This function:
    1. Loads and preprocesses the fight data
    2. Splits data chronologically into training/validation/test sets
    3. Fits preprocessing on training data only
    4. Trains and evaluates models on validation set
    5. Reports final unbiased metrics on held-out test set
    """
    set_global_seed()

    # Load and validate data from S3
    X_df, y, label_encoder, event_dates = load_fight_data(
        s3_bucket, s3_data_key, s3_results_prefix
    )
    if X_df is None:
        print("Failed to load fight event data.")
        exit(1)

    # Chronological three-way split: 70% training, 20% validation, 10% test
    sorted_indices = np.argsort(event_dates.values)
    split_train = int(len(X_df) * 0.7)
    split_val = int(len(X_df) * 0.9)
    train_idx = sorted_indices[:split_train]
    val_idx = sorted_indices[split_train:split_val]
    test_idx = sorted_indices[split_val:]

    X_train_df = X_df.iloc[train_idx]
    X_val_df = X_df.iloc[val_idx]
    X_test_df = X_df.iloc[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    print(f"Temporal split: {len(train_idx)} training, {len(val_idx)} validation, {len(test_idx)} test")
    print(
        f"Training period:   {event_dates.iloc[train_idx[0]]} to {event_dates.iloc[train_idx[-1]]}"
    )
    print(
        f"Validation period: {event_dates.iloc[val_idx[0]]} to {event_dates.iloc[val_idx[-1]]}"
    )
    print(
        f"Test period:       {event_dates.iloc[test_idx[0]]} to {event_dates.iloc[test_idx[-1]]}"
    )

    # Fit preprocessor on training data only to prevent leakage
    numerical_columns = [f"{s}_Diff" for s in DIFF_FEATURE_SUFFIXES]
    categorical_columns = ["Fighter1_Stance", "Fighter2_Stance"]
    preprocessor = build_preprocessor(X_train_df, numerical_columns, categorical_columns)
    X_train = preprocessor.transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    # Initialize models
    input_size = X_train.shape[1]
    models = initialize_models(input_size, DEVICE)

    # Train models
    trained_models = {}
    for name, model in models.items():
        trained_model = train_model(name, model, X_train, y_train, DEVICE)
        trained_models[name] = trained_model

    # Evaluate on validation set (used for experiments and model comparison)
    evaluate_validation(
        trained_models, X_train, X_val, y_train, y_val, label_encoder, DEVICE
    )

    # Evaluate on held-out test set (unbiased final metric)
    evaluate_test(trained_models, X_test, y_test, label_encoder, DEVICE)

    # Save trained models and label encoder to S3
    save_models(trained_models, input_size, s3_bucket)
    save_label_encoder(label_encoder, s3_bucket)

    # Upload results to S3
    upload_results_to_s3("results", s3_bucket, s3_results_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMA Predictive Modeling Pipeline")
    parser.add_argument("--s3_bucket", required=True)
    parser.add_argument("--s3_data_key", required=True)
    parser.add_argument("--s3_results_prefix", required=True)
    args = parser.parse_args()
    main(args.s3_bucket, args.s3_data_key, args.s3_results_prefix)
