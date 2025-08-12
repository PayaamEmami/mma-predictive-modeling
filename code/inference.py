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

# Debug preprocessing module import
print("=== DEBUGGING PREPROCESSING IMPORT ===")
import sys
import os

print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

# Check if preprocessing.py exists
if os.path.exists("preprocessing.py"):
    print("✓ preprocessing.py file exists")
    with open("preprocessing.py", "r") as f:
        content = f.read()
        if "def preprocess_features" in content:
            print("✓ preprocess_features function found in file")
        else:
            print("✗ preprocess_features function NOT found in file")
            print(
                "Functions found:",
                [
                    line.strip()
                    for line in content.split("\n")
                    if line.strip().startswith("def ")
                ],
            )
else:
    print("✗ preprocessing.py file does NOT exist")

try:
    import preprocessing

    print("✓ Successfully imported preprocessing module")
    print(
        f"Preprocessing module attributes: {[attr for attr in dir(preprocessing) if not attr.startswith('_')]}"
    )

    if hasattr(preprocessing, "preprocess_features"):
        print("✓ preprocess_features attribute exists in module")
    else:
        print("✗ preprocess_features attribute does NOT exist in module")

except Exception as e:
    print(f"✗ Failed to import preprocessing module: {e}")
    import traceback

    traceback.print_exc()

try:
    from preprocessing import preprocess_features

    print("✓ Successfully imported preprocess_features function")
except Exception as e:
    print(f"✗ Failed to import preprocess_features function: {e}")
    import traceback

    traceback.print_exc()

print("=== END DEBUGGING ===")

from config import DEVICE


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

    def get_fighter_features(
        self, fighter_url: str, historical_data: pd.DataFrame
    ) -> np.ndarray:
        """Extract features for a specific fighter from historical data."""
        try:
            # Filter data for this fighter (either as Fighter1 or Fighter2)
            fighter_fights = historical_data[
                (historical_data["Fighter1_ID"] == fighter_url)
                | (historical_data["Fighter2_ID"] == fighter_url)
            ].copy()

            if len(fighter_fights) == 0:
                print(f"No historical data found for fighter: {fighter_url}")
                return np.zeros(30)  # Return zero features if no data

            # Aggregate fighter statistics
            features = []

            # Process fights where fighter was Fighter1
            f1_fights = fighter_fights[fighter_fights["Fighter1_ID"] == fighter_url]
            # Process fights where fighter was Fighter2
            f2_fights = fighter_fights[fighter_fights["Fighter2_ID"] == fighter_url]

            # Combine stats from both perspectives
            all_stats = []

            for _, fight in f1_fights.iterrows():
                stats = [
                    self._safe_float(fight.get("Fighter1_Takedowns", "0")),
                    self._safe_float(fight.get("Fighter1_Submission_Attempts", "0")),
                    self._safe_float(fight.get("Fighter1_Reversals", "0")),
                    self._safe_float(fight.get("Fighter1_Knockdowns", "0")),
                    self._safe_float(fight.get("Fighter1_Significant_Strikes", "0")),
                    self._safe_float(fight.get("Fighter1_Head_Strikes", "0")),
                    self._safe_float(fight.get("Fighter1_Body_Strikes", "0")),
                    self._safe_float(fight.get("Fighter1_Leg_Strikes", "0")),
                    self._safe_float(fight.get("Fighter1_Distance_Strikes", "0")),
                    self._safe_float(fight.get("Fighter1_Clinch_Strikes", "0")),
                    self._safe_float(fight.get("Fighter1_Ground_Strikes", "0")),
                ]
                all_stats.append(stats)

            for _, fight in f2_fights.iterrows():
                stats = [
                    self._safe_float(fight.get("Fighter2_Takedowns", "0")),
                    self._safe_float(fight.get("Fighter2_Submission_Attempts", "0")),
                    self._safe_float(fight.get("Fighter2_Reversals", "0")),
                    self._safe_float(fight.get("Fighter2_Knockdowns", "0")),
                    self._safe_float(fight.get("Fighter2_Significant_Strikes", "0")),
                    self._safe_float(fight.get("Fighter2_Head_Strikes", "0")),
                    self._safe_float(fight.get("Fighter2_Body_Strikes", "0")),
                    self._safe_float(fight.get("Fighter2_Leg_Strikes", "0")),
                    self._safe_float(fight.get("Fighter2_Distance_Strikes", "0")),
                    self._safe_float(fight.get("Fighter2_Clinch_Strikes", "0")),
                    self._safe_float(fight.get("Fighter2_Ground_Strikes", "0")),
                ]
                all_stats.append(stats)

            if all_stats:
                # Calculate averages and additional derived features
                stats_array = np.array(all_stats)
                avg_stats = np.mean(stats_array, axis=0)
                max_stats = np.max(stats_array, axis=0)
                total_fights = len(all_stats)

                # Win rate calculation
                wins = 0
                for _, fight in f1_fights.iterrows():
                    if fight.get("Winner") == "1":
                        wins += 1
                for _, fight in f2_fights.iterrows():
                    if fight.get("Winner") == "2":
                        wins += 1

                win_rate = wins / total_fights if total_fights > 0 else 0

                # Combine all features
                features = np.concatenate(
                    [
                        avg_stats,  # 11 features
                        max_stats[:5],  # 5 features (max of first 5 stats)
                        [win_rate],  # 1 feature
                        [total_fights],  # 1 feature
                        [len(f1_fights)],  # 1 feature (fights as fighter1)
                        [len(f2_fights)],  # 1 feature (fights as fighter2)
                    ]
                )

                # Pad or trim to expected size
                if len(features) < 30:
                    features = np.pad(features, (0, 30 - len(features)))
                elif len(features) > 30:
                    features = features[:30]

                return features
            else:
                return np.zeros(30)

        except Exception as e:
            print(f"Error extracting features for fighter {fighter_url}: {e}")
            return np.zeros(30)

    def _safe_float(self, value: str) -> float:
        """Safely convert string to float, handling fractions and empty values."""
        try:
            if not value or value == "--" or value == "":
                return 0.0

            # Handle fractions like "5 of 10"
            if " of " in str(value):
                parts = str(value).split(" of ")
                if len(parts) == 2:
                    numerator = float(parts[0])
                    denominator = float(parts[1])
                    return numerator / denominator if denominator != 0 else 0.0

            return float(value)
        except:
            return 0.0

    def predict_fight(
        self, fighter1_features: np.ndarray, fighter2_features: np.ndarray
    ) -> Dict[str, Any]:
        """Generate predictions for a single fight using all models."""
        predictions = {}

        # Combine features for the fight
        fight_features = np.concatenate([fighter1_features, fighter2_features]).reshape(
            1, -1
        )

        # Preprocess features (normalize)
        fight_features = preprocess_features(fight_features)

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

    def generate_predictions(
        self, upcoming_fights_data: Dict, historical_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate predictions for all upcoming fights."""
        event_predictions = {
            "event_name": upcoming_fights_data.get("EventName", ""),
            "event_date": upcoming_fights_data.get("EventDate", ""),
            "event_location": upcoming_fights_data.get("EventLocation", ""),
            "generated_at": pd.Timestamp.now().isoformat(),
            "fights": [],
        }

        for fight in upcoming_fights_data.get("Fights", []):
            print(
                f"Generating predictions for: {fight['Fighter1Name']} vs {fight['Fighter2Name']}"
            )

            # Extract features for both fighters
            fighter1_features = self.get_fighter_features(
                fight["Fighter1Url"], historical_data
            )
            fighter2_features = self.get_fighter_features(
                fight["Fighter2Url"], historical_data
            )

            # Generate predictions
            fight_predictions = self.predict_fight(fighter1_features, fighter2_features)

            # Calculate aggregate predictions
            fighter1_votes = sum(
                1
                for pred in fight_predictions.values()
                if isinstance(pred, dict) and pred.get("prediction") == 1
            )
            fighter2_votes = sum(
                1
                for pred in fight_predictions.values()
                if isinstance(pred, dict) and pred.get("prediction") == 2
            )
            total_models = len(
                [pred for pred in fight_predictions.values() if isinstance(pred, dict)]
            )

            avg_confidence = np.mean(
                [
                    pred.get("confidence", 0.5)
                    for pred in fight_predictions.values()
                    if isinstance(pred, dict)
                ]
            )

            fight_result = {
                "fighter1_name": fight["Fighter1Name"],
                "fighter2_name": fight["Fighter2Name"],
                "weight_class": fight.get("WeightClass", ""),
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
                        fight["Fighter1Name"]
                        if fighter1_votes > fighter2_votes
                        else fight["Fighter2Name"]
                    ),
                    "average_confidence": round(avg_confidence, 3),
                },
            }

            event_predictions["fights"].append(fight_result)

        return event_predictions


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
        print(
            f"Loaded upcoming fights data: {len(upcoming_fights_data.get('Fights', []))} fights"
        )
    except Exception as e:
        print(f"Failed to load upcoming fights data: {e}")
        return

    # Load historical data
    try:
        obj = s3_client.get_object(Bucket=s3_bucket, Key=historical_data_key)
        historical_data = pd.read_csv(StringIO(obj["Body"].read().decode()))
        print(f"Loaded historical data: {len(historical_data)} fights")
    except Exception as e:
        print(f"Failed to load historical data: {e}")
        return

    # Generate predictions
    predictions = inference.generate_predictions(upcoming_fights_data, historical_data)

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
