# evaluation.py

import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
)
import pandas as pd
from config import OUTPUT_PATH

def evaluate_models(models, X_test, y_test, label_encoder, device):
    model_performances = {}

    for name, model in models.items():
        if name == 'Neural Network':
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test.astype(np.float32)).to(device)
                outputs = model(X_test_tensor)
                _, y_pred_tensor = torch.max(outputs.data, 1)
                y_pred = y_pred_tensor.cpu().numpy()
                accuracy = (y_pred == y_test).sum() / y_test.size
        else:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

        model_performances[name] = accuracy
        print(f"Classification Report for {name}:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        disp.plot()
        plt.title(f'Confusion Matrix for {name}')
        plt.savefig(os.path.join(OUTPUT_PATH, f'confusion_matrix_{name.replace(" ", "_")}.png'))
        plt.close()

    # output model performances
    performance_df = pd.DataFrame(list(model_performances.items()), columns=['Model', 'Accuracy'])
    performance_df.to_csv(os.path.join(OUTPUT_PATH, 'model_performances.csv'), index=False)
    plt.figure(figsize=(10, 6))
    plt.barh(performance_df['Model'], performance_df['Accuracy'], color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Comparison of Model Accuracies')
    plt.xlim([0, 1])
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(OUTPUT_PATH, 'model_accuracy_comparison.png'))
    plt.show()
    print(performance_df)
    print(f"All tasks completed. Results and plots saved in {OUTPUT_PATH}.")
