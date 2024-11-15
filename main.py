# main.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import DATA_PATH, OUTPUT_PATH, DEVICE, DRIVE_MOUNT_PATH
from data import load_ufc_data
from models import initialize_models
from training import train_sklearn_model, train_pytorch_model
from evaluation import evaluate_sklearn_model, evaluate_pytorch_model, save_performance, save_classification_reports, save_confusion_matrices
from sklearn.metrics import classification_report
from google.colab import drive

def main():
    # mount google drive
    drive.mount(DRIVE_MOUNT_PATH)

    # load and preprocess data
    X, y, label_encoder, preprocessor = load_ufc_data(DATA_PATH)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # initialize models
    input_size = X_train.shape[1]
    models = initialize_models(input_size, DEVICE)

    # dictionary to store model performances
    model_performances = {}
    classification_reports = {}
    confusion_matrices = {}

    # train and evaluate scikit-learn models
    sklearn_models = ['Random Forest', 'Gradient Boosting', 'Support Vector Machine',
                      'Logistic Regression', 'K-Nearest Neighbors', 'Naive Bayes']

    for model_name in sklearn_models:
        print(f"\nTraining {model_name}...")
        model = models[model_name]
        trained_model = train_sklearn_model(model_name, model, X_train, y_train)
        y_pred, accuracy, report, cm = evaluate_sklearn_model(model_name, trained_model, X_test, y_test)
        model_performances[model_name] = accuracy
        classification_reports[model_name] = {'y_true': y_test, 'y_pred': y_pred}
        confusion_matrices[model_name] = {'y_true': y_test, 'y_pred': y_pred}
        print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")

    # train and evaluate pytorch neural network
    print("\nTraining Neural Network...")
    nn_model = models['Neural Network']
    trained_nn_model = train_pytorch_model(nn_model, X_train, y_train, DEVICE, num_epochs=500, lr=0.001, weight_decay=0.1)
    y_pred_nn, accuracy_nn = evaluate_pytorch_model(trained_nn_model, X_test, y_test, DEVICE)
    model_performances['Neural Network'] = accuracy_nn
    classification_reports['Neural Network'] = {'y_true': y_test, 'y_pred': y_pred_nn}
    confusion_matrices['Neural Network'] = {'y_true': y_test, 'y_pred': y_pred_nn}
    print(f"Neural Network Accuracy: {accuracy_nn * 100:.2f}%")

    # save model performances
    performance_df = save_performance(model_performances, OUTPUT_PATH)
    print("\nModel Performances:")
    print(performance_df)

    # save classification reports
    save_classification_reports(classification_reports, label_encoder, OUTPUT_PATH)

    # save confusion matrices
    save_confusion_matrices(confusion_matrices, label_encoder, OUTPUT_PATH)

    print(f"\nAll tasks completed. Results and plots saved in {OUTPUT_PATH}.")

if __name__ == "__main__":
    main()
