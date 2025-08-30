"""
AWS Lambda Function: Model Leaderboard API
==========================================

This Lambda function pre-calculates and serves model performance statistics
for the MMA prediction system. It processes all historical predictions to
generate accuracy metrics for each model, eliminating the need for frontend
to make 100+ API calls.

API Endpoints:
- GET /model-leaderboard - Get pre-calculated model performance statistics
- POST /model-leaderboard/refresh - Trigger recalculation of leaderboard (internal)

Features:
- Pre-calculates model accuracy statistics from all past predictions
- Caches results in S3 for fast retrieval
- Automatically updates when new predictions are archived
- Reduces frontend API calls from 100+ to 1

Dependencies:
- S3 bucket with archived-predictions/ and data/fight_events.csv
- Environment variables: BUCKET_NAME, FIGHT_EVENTS_KEY
"""

import json
import boto3
import csv
import os
from io import StringIO
from datetime import datetime
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")

# Get configuration from environment variables
BUCKET_NAME = os.environ.get("BUCKET_NAME")
FIGHT_EVENTS_KEY = os.environ.get("FIGHT_EVENTS_KEY", "data/fight_events.csv")
LEADERBOARD_CACHE_KEY = "cache/model_leaderboard.json"


def lambda_handler(event, context):
    """
    Lambda function to handle model leaderboard API endpoints:

    Endpoints:
    - GET /model-leaderboard - Get pre-calculated model performance statistics
    - POST /model-leaderboard/refresh - Trigger recalculation of leaderboard

    Returns:
    - Model performance statistics with accuracy rankings
    - Cached results for fast retrieval
    """

    # Enable CORS
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Content-Type": "application/json",
    }

    try:
        http_method = event.get("httpMethod", "GET")
        path = event.get("path", "")

        if http_method == "OPTIONS":
            return {
                "statusCode": 200,
                "headers": headers,
                "body": "",
            }

        if http_method == "GET":
            return handle_get_leaderboard(headers)
        elif http_method == "POST" and "refresh" in path:
            return handle_refresh_leaderboard(headers)
        else:
            return {
                "statusCode": 405,
                "headers": headers,
                "body": json.dumps({"error": "Method not allowed"}),
            }

    except Exception as e:
        logger.error(f"Error in model leaderboard handler: {str(e)}")
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps(
                {
                    "error": "Internal server error",
                    "message": "Failed to process model leaderboard request",
                }
            ),
        }


def handle_get_leaderboard(headers):
    """Get pre-calculated model leaderboard from cache or calculate if needed."""
    try:
        # Try to get cached leaderboard first
        try:
            response = s3_client.get_object(
                Bucket=BUCKET_NAME, Key=LEADERBOARD_CACHE_KEY
            )
            cached_data = json.loads(response["Body"].read().decode("utf-8"))

            # Check if cache is recent (less than 24 hours old)
            cache_time = datetime.fromisoformat(cached_data.get("generated_at", ""))
            if (datetime.now() - cache_time).total_seconds() < 24 * 3600:
                logger.info("Returning cached model leaderboard")
                return {
                    "statusCode": 200,
                    "headers": headers,
                    "body": json.dumps(cached_data),
                }
        except s3_client.exceptions.NoSuchKey:
            logger.info("No cached leaderboard found, calculating...")
        except Exception as e:
            logger.warning(f"Error reading cached leaderboard: {e}")

        # Calculate fresh leaderboard
        leaderboard_data = calculate_model_leaderboard()

        # Cache the results
        try:
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=LEADERBOARD_CACHE_KEY,
                Body=json.dumps(leaderboard_data),
                ContentType="application/json",
            )
            logger.info("Cached new model leaderboard")
        except Exception as e:
            logger.warning(f"Failed to cache leaderboard: {e}")

        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps(leaderboard_data),
        }

    except Exception as e:
        logger.error(f"Error getting model leaderboard: {str(e)}")
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps(
                {
                    "error": "Failed to get model leaderboard",
                    "message": str(e),
                }
            ),
        }


def handle_refresh_leaderboard(headers):
    """Force recalculation and caching of model leaderboard."""
    try:
        logger.info("Refreshing model leaderboard...")
        leaderboard_data = calculate_model_leaderboard()

        # Cache the results
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=LEADERBOARD_CACHE_KEY,
            Body=json.dumps(leaderboard_data),
            ContentType="application/json",
        )

        logger.info("Model leaderboard refreshed and cached")

        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps(
                {
                    "message": "Model leaderboard refreshed successfully",
                    "generated_at": leaderboard_data["generated_at"],
                    "total_models": len(leaderboard_data["models"]),
                    "total_events_processed": leaderboard_data["metadata"][
                        "total_events_processed"
                    ],
                }
            ),
        }

    except Exception as e:
        logger.error(f"Error refreshing model leaderboard: {str(e)}")
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps(
                {
                    "error": "Failed to refresh model leaderboard",
                    "message": str(e),
                }
            ),
        }


def calculate_model_leaderboard():
    """Calculate model performance statistics from all past predictions."""
    logger.info("Starting model leaderboard calculation...")

    # Load fight events data for comparison
    fight_events = load_fight_events()
    if not fight_events:
        raise Exception("Failed to load fight events data")

    # Get all past prediction files
    response = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME, Prefix="predictions/archive/"
    )

    if "Contents" not in response:
        raise Exception("No archived predictions found")

    # Initialize model statistics tracking
    model_stats = {}
    events_processed = 0
    total_fights_processed = 0

    # Process each prediction file
    for obj in response["Contents"]:
        key = obj["Key"]
        if not key.endswith(".json") or key == "predictions/archive/":
            continue

        try:
            # Get the prediction file content
            file_response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
            content = file_response["Body"].read().decode("utf-8")
            prediction_data = json.loads(content)

            event_name = prediction_data.get("event_name", "")

            # Only process events that exist in fight_events.csv
            if not event_exists_in_fight_data(fight_events, event_name):
                logger.debug(f"Skipping '{event_name}' - no fight data available")
                continue

            logger.debug(f"Processing event: {event_name}")
            events_processed += 1

            # Get accuracy data for this event
            accuracy_data = get_event_accuracy(prediction_data, fight_events)
            if not accuracy_data:
                continue

            # Process each fight to calculate model-level accuracy
            for fight in prediction_data.get("fights", []):
                total_fights_processed += 1

                # Find the actual winner for this fight
                fight_result = None
                for result in accuracy_data.get("fight_results", []):
                    if result.get("fighter1_name") == fight.get(
                        "fighter1_name"
                    ) and result.get("fighter2_name") == fight.get("fighter2_name"):
                        fight_result = result
                        break

                if not fight_result or not fight_result.get("actual_winner"):
                    continue

                # Check each model's prediction for this fight
                for model_name, prediction in fight.get(
                    "model_predictions", {}
                ).items():
                    if model_name not in model_stats:
                        model_stats[model_name] = {"correct": 0, "total": 0}

                    model_stats[model_name]["total"] += 1

                    # Map prediction winner to actual fighter names
                    predicted_winner = prediction.get("winner", "")
                    if predicted_winner == "Fighter 1":
                        predicted_winner = fight.get("fighter1_name", "")
                    elif predicted_winner == "Fighter 2":
                        predicted_winner = fight.get("fighter2_name", "")

                    if predicted_winner == fight_result["actual_winner"]:
                        model_stats[model_name]["correct"] += 1

        except Exception as e:
            logger.warning(f"Error processing prediction file {key}: {e}")
            continue

    # Convert to leaderboard format and sort
    models = []
    for model_name, stats in model_stats.items():
        accuracy = (
            (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        )
        models.append(
            {
                "modelName": model_name,
                "correctPredictions": stats["correct"],
                "totalPredictions": stats["total"],
                "accuracy": round(accuracy, 1),
            }
        )

    # Sort by accuracy (descending), then by total predictions (descending)
    models.sort(key=lambda x: (-x["accuracy"], -x["totalPredictions"]))

    leaderboard_data = {
        "generated_at": datetime.now().isoformat(),
        "models": models,
        "metadata": {
            "total_events_processed": events_processed,
            "total_fights_processed": total_fights_processed,
            "total_models": len(models),
        },
    }

    logger.info(
        f"Model leaderboard calculated: {len(models)} models, {events_processed} events, {total_fights_processed} fights"
    )
    return leaderboard_data


def load_fight_events():
    """Load fight events data from S3."""
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=FIGHT_EVENTS_KEY)
        content = response["Body"].read().decode("utf-8")

        # Parse CSV content
        csv_reader = csv.DictReader(StringIO(content))
        fight_events = list(csv_reader)

        logger.info(f"Loaded {len(fight_events)} fight events from CSV")
        return fight_events
    except Exception as e:
        logger.error(f"Failed to load fight events: {e}")
        return None


def event_exists_in_fight_data(fight_events, event_name):
    """Check if an event exists in the fight events data."""
    if not fight_events or not event_name:
        return False

    for fight in fight_events:
        if fight.get("EventName", "").strip() == event_name.strip():
            return True
    return False


def get_event_accuracy(prediction_data, fight_events):
    """Calculate accuracy metrics for a specific event."""
    try:
        if not prediction_data or not fight_events:
            return None

        event_name = prediction_data.get("event_name", "")
        fights = prediction_data.get("fights", [])

        fight_results = []
        correct_predictions = 0
        total_fights = 0

        for fight in fights:
            fighter1_name = fight.get("fighter1_name", "")
            fighter2_name = fight.get("fighter2_name", "")
            predicted_winner = fight.get("aggregate", {}).get("predicted_winner", "")

            # Find actual winner from fight events
            actual_winner = None
            for event_fight in fight_events:
                if event_fight.get("EventName", "").strip() == event_name.strip() and (
                    (
                        event_fight.get("Fighter1_Name", "").strip()
                        == fighter1_name.strip()
                        and event_fight.get("Fighter2_Name", "").strip()
                        == fighter2_name.strip()
                    )
                    or (
                        event_fight.get("Fighter1_Name", "").strip()
                        == fighter2_name.strip()
                        and event_fight.get("Fighter2_Name", "").strip()
                        == fighter1_name.strip()
                    )
                ):
                    winner_num = event_fight.get("Winner", "").strip()
                    # Convert numeric winner to actual fighter name
                    if winner_num == "1":
                        actual_winner = event_fight.get("Fighter1_Name", "").strip()
                    elif winner_num == "2":
                        actual_winner = event_fight.get("Fighter2_Name", "").strip()
                    break

            is_correct = actual_winner == predicted_winner if actual_winner else False
            if actual_winner:
                total_fights += 1
                if is_correct:
                    correct_predictions += 1

            fight_results.append(
                {
                    "fighter1_name": fighter1_name,
                    "fighter2_name": fighter2_name,
                    "predicted_winner": predicted_winner,
                    "actual_winner": actual_winner,
                    "is_correct": is_correct,
                    "confidence": fight.get("aggregate", {}).get(
                        "average_confidence", 0
                    ),
                }
            )

        overall_accuracy = (
            (correct_predictions / total_fights * 100) if total_fights > 0 else 0
        )

        return {
            "overall_accuracy": round(overall_accuracy, 1),
            "correct_predictions": correct_predictions,
            "total_fights": total_fights,
            "fight_results": fight_results,
        }

    except Exception as e:
        logger.error(f"Error calculating event accuracy: {e}")
        return None
