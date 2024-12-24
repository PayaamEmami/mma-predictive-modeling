# main.py

from sklearn.model_selection import train_test_split

from data import load_ufc_data
from models import initialize_models
from training import train_model
from evaluation import evaluate_models
from configs.config import DEVICE


def main():
    # load and preprocess data
    X_ufc, y_ufc, label_encoder = load_ufc_data()

    if X_ufc is None:
        print("Failed to load UFC data.")
        exit(1)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_ufc, y_ufc, test_size=0.2, random_state=21, stratify=y_ufc
    )

    # input_size is the number of features in the training dataset
    input_size = X_train.shape[1]

    # initialize models to train and analyze
    models = initialize_models(input_size, DEVICE)

    # train models
    for name, model in models.items():
        print(f"Training {name}...")
        train_model(name, model, X_train, y_train, DEVICE)

    # evaluate models
    evaluate_models(models, X_train, X_test, y_train, y_test, label_encoder, DEVICE)


if __name__ == "__main__":
    main()
