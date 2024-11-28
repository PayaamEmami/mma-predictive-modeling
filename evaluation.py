# evaluation.py

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report

from config import OUTPUT_PATH
from utils import plot_confusion_matrix, plot_model_accuracies, plot_learning_curve

def evaluate_models(models, X_train, X_test, y_train, y_test, label_encoder, device):
    # dictionary to store model performances
    model_performances = {}

    # iterate over each model
    for name, model in models.items():
        if name in ['Neural Network', 'Transformer']:
            # set model to evaluation mode
            model.eval()
            with torch.no_grad():
                # convert test data to tensor and move to device
                X_test_tensor = torch.tensor(X_test.astype(np.float32)).to(device)
                # get model predictions
                outputs = model(X_test_tensor)
                _, y_pred_tensor = torch.max(outputs.data, 1)
                y_pred = y_pred_tensor.cpu().numpy()
                # calculate accuracy
                accuracy = (y_pred == y_test).sum() / y_test.size
        else:
            # get model predictions
            y_pred = model.predict(X_test)
            # calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

        # store model performance
        model_performances[name] = accuracy
        print(f"Classification Report for {name}:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

        # plot confusion matrix
        plot_confusion_matrix(name, y_test, y_pred, label_encoder, OUTPUT_PATH)

        # plot learning curve
        plot_learning_curve(model, X_train, y_train, name, OUTPUT_PATH, device)

    # convert model performances to DataFrame
    performance_df = pd.DataFrame(list(model_performances.items()), columns=['Model', 'Accuracy'])
    # save model performances to CSV
    performance_df.to_csv(os.path.join(OUTPUT_PATH, 'model_performances.csv'), index=False)
    # plot model accuracies
    plot_model_accuracies(performance_df, OUTPUT_PATH)
    # print model accuracies
    print(performance_df)

    print(f"All tasks completed. Results and plots saved in {OUTPUT_PATH}.")
