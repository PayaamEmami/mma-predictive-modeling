# utils.py

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

def plot_confusion_matrix(model_name, y_test, y_pred, label_encoder, output_path):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot()
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(os.path.join(output_path, f'confusion_matrix_{model_name.replace(" ", "_")}.png'))
    plt.close()

def plot_model_accuracies(performance_df, output_path):
    performance_df = performance_df.sort_values(by='Accuracy', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(performance_df['Model'], performance_df['Accuracy'], color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Comparison of Model Accuracies')
    plt.xlim([0, 1])
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_path, 'model_accuracy_comparison.png'))
    plt.close()
