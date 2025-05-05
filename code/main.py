from sklearn.model_selection import train_test_split
from data import load_ufc_data
from models import initialize_models
from evaluation import evaluate_models
from training import train_model
from config import DEVICE


def main():
    """
    Main execution function for the UFC fight prediction pipeline.

    This function:
    1. Loads and preprocesses the UFC fight data
    2. Splits data into train and test sets
    3. Trains and evaluates models
    4. Generates performance visualizations
    """
    # Load and validate data
    X_ufc, y_ufc, label_encoder = load_ufc_data()
    if X_ufc is None:
        print("Failed to load UFC data.")
        exit(1)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_ufc, y_ufc, test_size=0.2, random_state=42
    )

    # Initialize models
    input_size = X_train.shape[1]
    models = initialize_models(input_size, DEVICE)

    # Train models
    trained_models = {}
    for name, model in models.items():
        trained_model = train_model(
            name, model, X_train, y_train, DEVICE
        )
        trained_models[name] = trained_model

    # Generate evaluation metrics and visualizations
    evaluate_models(trained_models, X_train, X_test, y_train, y_test, label_encoder, DEVICE)


if __name__ == "__main__":
    main()
