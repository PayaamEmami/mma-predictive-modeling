import argparse
from sklearn.model_selection import train_test_split
from data import load_fight_data, upload_results_to_s3
from models import initialize_models
from evaluation import evaluate_models
from training import train_model, save_models, save_label_encoder
from config import DEVICE


def main(s3_bucket, s3_data_key, s3_results_prefix):
    """
    Main execution function for the fight prediction pipeline.

    This function:
    1. Loads and preprocesses the fight data
    2. Splits data into train and test sets
    3. Trains and evaluates models
    4. Generates performance visualizations
    """

    # Load and validate data from S3
    X_fight, y_fight, label_encoder = load_fight_data(
        s3_bucket, s3_data_key, s3_results_prefix
    )
    if X_fight is None:
        print("Failed to load fight event data.")
        exit(1)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_fight, y_fight, test_size=0.2, random_state=42
    )

    # Initialize models
    input_size = X_train.shape[1]
    models = initialize_models(input_size, DEVICE)

    # Train models
    trained_models = {}
    for name, model in models.items():
        trained_model = train_model(name, model, X_train, y_train, DEVICE)
        trained_models[name] = trained_model

    # Generate evaluation metrics and visualizations
    evaluate_models(
        trained_models, X_train, X_test, y_train, y_test, label_encoder, DEVICE
    )

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
