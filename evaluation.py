# evaluation.py
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
from utils import plot_confusion_matrix, plot_model_accuracies
import pandas as pd
import os

def evaluate_sklearn_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return y_pred, accuracy, report, cm

def evaluate_pytorch_model(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.astype(float)).to(device)
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted.cpu().numpy() == y_test).mean()
    return predicted.cpu().numpy(), accuracy

def save_performance(performance_dict, output_path):
    performance_df = pd.DataFrame(list(performance_dict.items()), columns=['Model', 'Accuracy'])
    performance_df.to_csv(os.path.join(output_path, 'model_performances.csv'), index=False)
    plot_model_accuracies(performance_df, output_path)
    return performance_df

def save_classification_reports(reports, label_encoder, output_path):
    for model_name, report in reports.items():
        print(f"Classification Report for {model_name}:\n")
        print(classification_report(report['y_true'], report['y_pred'], target_names=label_encoder.classes_))
        # optionally, save the report to a file
        report_path = os.path.join(output_path, f'classification_report_{model_name.replace(" ", "_")}.txt')
        with open(report_path, 'w') as f:
            f.write(classification_report(report['y_true'], report['y_pred'], target_names=label_encoder.classes_))

def save_confusion_matrices(confusion_matrices, label_encoder, output_path):
    for model_name, cm in confusion_matrices.items():
        plot_confusion_matrix(model_name, confusion_matrices[model_name]['y_true'], confusion_matrices[model_name]['y_pred'], label_encoder, output_path)

