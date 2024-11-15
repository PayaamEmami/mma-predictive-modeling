# utils.py
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import os

def parse_height(height_str):
    if pd.isnull(height_str):
        return 177
    pattern = r"(?:(\d+)ft)?\s*(\d+)in"
    match = re.match(pattern, height_str)
    if match:
        feet = int(match.group(1)) if match.group(1) else 0
        inches = int(match.group(2))
        total_inches = feet * 12 + inches
        return total_inches * 2.54
    else:
        return 177

def parse_reach(reach_str):
    if pd.isnull(reach_str):
        return 183
    pattern = r"(\d+)in"
    match = re.match(pattern, reach_str)
    if match:
        inches = int(match.group(1))
        return inches * 2.54
    else:
        return 183

def parse_strike(strike_str):
    if pd.isnull(strike_str):
        return [0, 0]
    pattern = r"(\d+)\s+of\s+(\d+)"
    match = re.match(pattern, strike_str)
    if match:
        landed = int(match.group(1))
        attempted = int(match.group(2))
        return [landed, attempted]
    else:
        return [0, 0]

def parse_time(time_str):
    if pd.isnull(time_str):
        return 0
    pattern = r"(\d+):(\d+)"
    match = re.match(pattern, time_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        total_seconds = minutes * 60 + seconds
        return total_seconds
    else:
        return 0

def plot_confusion_matrix(model_name, y_test, y_pred, label_encoder, output_path):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot()
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(os.path.join(output_path, f'confusion_matrix_{model_name.replace(" ", "_")}.png'))
    plt.close()

def plot_model_accuracies(performance_df, output_path):
    plt.figure(figsize=(10, 6))
    plt.barh(performance_df['Model'], performance_df['Accuracy'], color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Comparison of Model Accuracies')
    plt.xlim([0, 1])
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_path, 'model_accuracy_comparison.png'))
    plt.close()
