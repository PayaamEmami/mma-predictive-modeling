"""
Model inference module for predicting upcoming fight outcomes.
Loads trained models and generates predictions for upcoming fights.
"""

import pickle
import json
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any
import boto3
from io import StringIO
from models import FCNN, Transformer
from config import DEVICE
from preprocessing import preprocess_features


class ModelInference:
    """Handles model loading and inference for upcoming fights."""

    def __init__(self, s3_bucket: str, models_prefix: str = "models/"):
        self.s3_bucket = s3_bucket
        self.models_prefix = models_prefix
        self.s3_client = boto3.client("s3")
        self.models = {}
        self.label_encoder = None

    def load_models(self) -> bool:
        """Load all trained models from S3."""
        try:
            # Load scikit-learn models
            sklearn_models = [
                "Random_Forest",
                "Gradient_Boosting",
                "SVM",
                "Logistic_Regression",
                "KNN",
                "Naive_Bayes",
                "Decision_Tree",
            ]

            for model_name in sklearn_models:
                try:
                    model_key = f"{self.models_prefix}{model_name}.pkl"
                    obj = self.s3_client.get_object(
                        Bucket=self.s3_bucket, Key=model_key
                    )
                    model = pickle.loads(obj["Body"].read())
                    self.models[model_name.replace("_", " ")] = model
                    print(f"Loaded {model_name} model")
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")

            # Load PyTorch models (FCNN and Transformer)
            pytorch_models = ["FCNN", "Transformer"]
            for model_name in pytorch_models:
                try:
                    model_key = f"{self.models_prefix}{model_name}.pth"
                    obj = self.s3_client.get_object(
                        Bucket=self.s3_bucket, Key=model_key
                    )
                    model_data = torch.load(
                        StringIO(obj["Body"].read().decode()), map_location=DEVICE
                    )

                    if model_name == "FCNN":
                        model = FCNN(
                            input_size=model_data["input_size"],
                            hidden_size=model_data["hidden_size"],
                        ).to(DEVICE)
                    elif model_name == "Transformer":
                        model = Transformer(
                            input_size=model_data["input_size"],
                            embedding_dim=model_data["embedding_dim"],
                            num_layers=model_data["num_layers"],
                            nhead=model_data["nhead"],
                        ).to(DEVICE)

                    model.load_state_dict(model_data["state_dict"])
                    model.eval()
                    self.models[model_name] = model
                    print(f"Loaded {model_name} model")
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")

            # Load label encoder
            try:
                encoder_key = f"{self.models_prefix}label_encoder.pkl"
                obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=encoder_key)
                self.label_encoder = pickle.loads(obj["Body"].read())
                print("Loaded label encoder")
            except Exception as e:
                print(f"Failed to load label encoder: {e}")

            print(f"Successfully loaded {len(self.models)} models")
            return len(self.models) > 0

        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def predict_with_features(self, fight_features: np.ndarray) -> Dict[str, Any]:
        """Generate predictions for preprocessed fight features using all models."""
        predictions = {}

        for model_name, model in self.models.items():
            try:
                if model_name in ["FCNN", "Transformer"]:
                    # PyTorch models
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(fight_features).to(DEVICE)
                        outputs = model(features_tensor)
                        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                        prediction = int(np.argmax(probabilities))
                        confidence = float(np.max(probabilities))
                else:
                    # Scikit-learn models
                    prediction = model.predict(fight_features)[0]
                    try:
                        probabilities = model.predict_proba(fight_features)[0]
                        confidence = float(np.max(probabilities))
                    except:
                        confidence = (
                            0.6  # Default confidence for models without predict_proba
                        )

                predictions[model_name] = {
                    "prediction": int(prediction),
                    "confidence": confidence,
                    "winner": "Fighter 1" if prediction == 1 else "Fighter 2",
                }

            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                predictions[model_name] = {
                    "prediction": 0,
                    "confidence": 0.5,
                    "winner": "Unknown",
                    "error": str(e),
                }

        return predictions


def main(
    s3_bucket: str,
    upcoming_fights_key: str,
    historical_data_key: str,
    predictions_key: str,
):
    """Main function for generating fight predictions."""

    # Initialize inference engine
    inference = ModelInference(s3_bucket)

    # Load models
    if not inference.load_models():
        print("Failed to load models. Exiting.")
        return

    # Load upcoming fights data
    try:
        s3_client = boto3.client("s3")
        obj = s3_client.get_object(Bucket=s3_bucket, Key=upcoming_fights_key)
        upcoming_fights_data = json.loads(obj["Body"].read().decode())
        print(f"Loaded upcoming fights data: {len(upcoming_fights_data)} events")
    except Exception as e:
        print(f"Failed to load upcoming fights data: {e}")
        return

    # Use preprocess_features to get properly normalized features
    # This replicates the exact same preprocessing pipeline used in training
    try:
        print("Preprocessing features using training pipeline...")
        processed_features = preprocess_features(
            upcoming_fights_data, historical_data_key
        )
        print(f"Generated features with shape: {processed_features.shape}")
    except Exception as e:
        print(f"Failed to preprocess features: {e}")
        import traceback

        traceback.print_exc()
        return

    # Generate predictions using processed features
    predictions = {"generated_at": pd.Timestamp.now().isoformat(), "fights": []}

    # Process each event and fight
    for event_idx, event in enumerate(upcoming_fights_data):
        for fight_idx, fight in enumerate(event.get("main_card", [])):
            print(
                f"Generating predictions for: {fight['fighter_a']} vs {fight['fighter_b']}"
            )

            # Get the corresponding preprocessed features for this fight
            # Each fight should have one row in processed_features
            if event_idx < len(processed_features):
                fight_features = processed_features[event_idx].reshape(1, -1)

                # Generate predictions using all models
                fight_predictions = inference.predict_with_features(fight_features)

                # Calculate aggregate predictions
                fighter1_votes = sum(
                    1
                    for pred in fight_predictions.values()
                    if isinstance(pred, dict) and pred.get("prediction") == 1
                )
                fighter2_votes = sum(
                    1
                    for pred in fight_predictions.values()
                    if isinstance(pred, dict) and pred.get("prediction") == 0
                )
                total_models = len(
                    [
                        pred
                        for pred in fight_predictions.values()
                        if isinstance(pred, dict)
                    ]
                )

                avg_confidence = np.mean(
                    [
                        pred.get("confidence", 0.5)
                        for pred in fight_predictions.values()
                        if isinstance(pred, dict)
                    ]
                )

                fight_result = {
                    "fighter1_name": fight["fighter_a"],
                    "fighter2_name": fight["fighter_b"],
                    "weight_class": fight.get("weight_class", ""),
                    "model_predictions": fight_predictions,
                    "aggregate": {
                        "fighter1_votes": fighter1_votes,
                        "fighter2_votes": fighter2_votes,
                        "total_models": total_models,
                        "fighter1_percentage": (
                            round((fighter1_votes / total_models) * 100, 1)
                            if total_models > 0
                            else 0
                        ),
                        "fighter2_percentage": (
                            round((fighter2_votes / total_models) * 100, 1)
                            if total_models > 0
                            else 0
                        ),
                        "predicted_winner": (
                            fight["fighter_a"]
                            if fighter1_votes > fighter2_votes
                            else fight["fighter_b"]
                        ),
                        "average_confidence": round(avg_confidence, 3),
                    },
                }

                predictions["fights"].append(fight_result)

    # Save predictions to S3
    try:
        predictions_json = json.dumps(predictions, indent=2)
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=predictions_key,
            Body=predictions_json,
            ContentType="application/json",
        )
        print(f"Predictions saved to s3://{s3_bucket}/{predictions_key}")
    except Exception as e:
        print(f"Failed to save predictions: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MMA Fight Prediction Inference")
    parser.add_argument("--s3_bucket", required=True)
    parser.add_argument("--upcoming_fights_key", default="upcoming_fights.json")
    parser.add_argument("--historical_data_key", default="fight_events.csv")
    parser.add_argument(
        "--predictions_key", default="predictions/latest_predictions.json"
    )

    args = parser.parse_args()
    main(
        args.s3_bucket,
        args.upcoming_fights_key,
        args.historical_data_key,
        args.predictions_key,
    )
