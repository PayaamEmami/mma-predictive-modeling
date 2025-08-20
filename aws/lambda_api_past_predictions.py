"""
AWS Lambda Function: Past Predictions and Accuracy API
======================================================

This Lambda function serves historical prediction data and accuracy metrics
via API Gateway. It calculates prediction accuracy by comparing archived
predictions with actual fight outcomes from scraped data.

API Endpoints:
- GET /past-predictions - List all past prediction events with pagination and accuracy metrics
- GET /past-predictions/{event-filename} - Get specific past prediction details
- GET /past-predictions/{event-filename}/accuracy - Get detailed accuracy metrics for specific event

Key Features:
- Only shows events that have actual fight outcome data in fight_events.csv
- Calculates fighter-level and event-level accuracy statistics
- Supports pagination and filtering

Dependencies:
- S3 bucket with archived-predictions/ and data/fight_events.csv
- Environment variables: BUCKET_NAME, FIGHT_EVENTS_KEY
"""

import json
import boto3
import csv
import os
from io import StringIO
from urllib.parse import unquote_plus

s3_client = boto3.client("s3")

# Get configuration from environment variables
BUCKET_NAME = os.environ.get("BUCKET_NAME")
FIGHT_EVENTS_KEY = os.environ.get("FIGHT_EVENTS_KEY")

if not BUCKET_NAME:
    raise ValueError("BUCKET_NAME environment variable is required")


def lambda_handler(event, context):
    """
    Lambda function to handle past predictions API endpoints:

    Endpoints:
    - GET /past-predictions - List all past prediction events with pagination and accuracy metrics
    - GET /past-predictions/{event-filename} - Get specific past prediction details
    - GET /past-predictions/{event-filename}/accuracy - Get detailed accuracy metrics for specific event

    Returns:
    - Past prediction data with calculated accuracy metrics
    - Only shows events that have actual fight outcome data in fight_events.csv
    - Includes fighter-level and event-level accuracy statistics
    """

    # Enable CORS
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Content-Type": "application/json",
    }

    try:
        http_method = event.get("httpMethod", "GET")
        path = event.get("path", "")
        query_params = event.get("queryStringParameters") or {}

        if http_method == "OPTIONS":
            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps({"message": "CORS preflight"}),
            }

        # Route to appropriate handler
        if path == "/past-predictions":
            return handle_list_past_predictions(query_params, headers)
        elif path.startswith("/past-predictions/") and path.endswith("/accuracy"):
            event_filename = extract_event_filename_from_path(path, "/accuracy")
            return handle_event_accuracy(event_filename, headers)
        elif path.startswith("/past-predictions/"):
            event_filename = extract_event_filename_from_path(path)
            return handle_get_past_prediction(event_filename, headers)
        else:
            return {
                "statusCode": 404,
                "headers": headers,
                "body": json.dumps({"error": "Endpoint not found"}),
            }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": "Internal server error", "details": str(e)}),
        }


def handle_list_past_predictions(query_params, headers):
    """
    List all past prediction events with pagination.
    Query params: page (default 1), limit (default 10)
    """
    try:
        page = int(query_params.get("page", 1))
        limit = int(query_params.get("limit", 10))

        # List all archived prediction files
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME, Prefix="predictions/archive/", Delimiter="/"
        )

        if "Contents" not in response:
            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps(
                    {
                        "events": [],
                        "total": 0,
                        "page": page,
                        "limit": limit,
                        "total_pages": 0,
                    }
                ),
            }

        # Process each archived file to get metadata
        # Load fight events data once for all checks
        fight_events = load_fight_events()
        print(
            f"Loaded fight events data: {len(fight_events) if fight_events is not None else 0} total fights"
        )

        if fight_events is not None and not fight_events.empty:
            unique_events = fight_events["Event"].unique()
            print(
                f"Available events in fight_events.csv: {len(unique_events)} unique events"
            )

        events = []
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

                # Only include events that exist in fight_events.csv
                # This ensures we have actual fight outcomes to compare against
                if not event_exists_in_fight_data(fight_events, event_name):
                    print(f"Skipping '{event_name}' - no fight data available yet")
                    continue

                print(f"Including '{event_name}' - fight data confirmed available")

                # Get accuracy if available
                filename = key.split("/")[-1].replace(".json", "")
                accuracy = get_event_accuracy(prediction_data)

                event_info = {
                    "filename": filename,
                    "event_name": event_name,
                    "event_date": prediction_data.get("event_date", ""),
                    "event_location": prediction_data.get("event_location", ""),
                    "generated_at": prediction_data.get("generated_at", ""),
                    "total_fights": len(prediction_data.get("fights", [])),
                    "last_modified": obj["LastModified"].isoformat(),
                    "accuracy": accuracy,
                }
                events.append(event_info)

            except Exception as e:
                print(f"Error processing {key}: {str(e)}")
                continue

        # Sort by event date (most recent first)
        events.sort(key=lambda x: x["event_date"], reverse=True)

        # Apply pagination
        total = len(events)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_events = events[start_idx:end_idx]

        total_pages = (total + limit - 1) // limit

        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps(
                {
                    "events": paginated_events,
                    "total": total,
                    "page": page,
                    "limit": limit,
                    "total_pages": total_pages,
                }
            ),
        }

    except Exception as e:
        print(f"Error listing past predictions: {str(e)}")
        raise


def handle_get_past_prediction(event_filename, headers):
    """
    Get a specific past prediction by filename.
    """
    try:
        key = f"predictions/archive/{event_filename}.json"

        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        content = response["Body"].read().decode("utf-8")
        prediction_data = json.loads(content)

        # Add accuracy information
        accuracy = get_event_accuracy(prediction_data)
        prediction_data["accuracy"] = accuracy

        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps(prediction_data),
        }

    except s3_client.exceptions.NoSuchKey:
        return {
            "statusCode": 404,
            "headers": headers,
            "body": json.dumps({"error": "Prediction event not found"}),
        }
    except Exception as e:
        print(f"Error getting past prediction: {str(e)}")
        raise


def handle_event_accuracy(event_filename, headers):
    """
    Get accuracy metrics for a specific event.
    """
    try:
        key = f"predictions/archive/{event_filename}.json"

        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        content = response["Body"].read().decode("utf-8")
        prediction_data = json.loads(content)

        accuracy = get_event_accuracy(prediction_data)

        return {"statusCode": 200, "headers": headers, "body": json.dumps(accuracy)}

    except s3_client.exceptions.NoSuchKey:
        return {
            "statusCode": 404,
            "headers": headers,
            "body": json.dumps({"error": "Prediction event not found"}),
        }
    except Exception as e:
        print(f"Error getting event accuracy: {str(e)}")
        raise


def get_event_accuracy(prediction_data):
    """
    Calculate accuracy metrics for an event by comparing with fight_events.csv.
    """
    try:
        # Load fight events data
        fight_events = load_fight_events()

        event_name = prediction_data.get("event_name", "")
        fights = prediction_data.get("fights", [])

        if not fights:
            return {
                "overall_accuracy": 0,
                "correct_predictions": 0,
                "total_fights": 0,
                "fight_results": [],
            }

        correct_predictions = 0
        fight_results = []

        for fight in fights:
            fighter1_name = fight["fighter1_name"]
            fighter2_name = fight["fighter2_name"]
            predicted_winner = fight["aggregate"]["predicted_winner"]

            # Find the actual result from fight_events.csv
            actual_result = find_fight_result(
                fight_events, event_name, fighter1_name, fighter2_name
            )

            is_correct = False
            if actual_result:
                actual_winner = actual_result["winner_name"]
                is_correct = predicted_winner == actual_winner
                if is_correct:
                    correct_predictions += 1

            fight_results.append(
                {
                    "fighter1_name": fighter1_name,
                    "fighter2_name": fighter2_name,
                    "predicted_winner": predicted_winner,
                    "actual_winner": (
                        actual_result["winner_name"] if actual_result else None
                    ),
                    "is_correct": is_correct,
                    "confidence": fight["aggregate"]["average_confidence"],
                }
            )

        overall_accuracy = (correct_predictions / len(fights)) * 100 if fights else 0

        return {
            "overall_accuracy": round(overall_accuracy, 1),
            "correct_predictions": correct_predictions,
            "total_fights": len(fights),
            "fight_results": fight_results,
        }

    except Exception as e:
        print(f"Error calculating accuracy: {str(e)}")
        return {
            "overall_accuracy": 0,
            "correct_predictions": 0,
            "total_fights": len(prediction_data.get("fights", [])),
            "fight_results": [],
            "error": str(e),
        }


def event_exists_in_fight_data(fight_events_df, event_name):
    """
    Check if an event exists in the fight events data.
    This ensures we only show past predictions for events where we have actual outcomes.
    """
    if fight_events_df is None or fight_events_df.empty:
        return False

    # Clean and normalize event names for comparison
    event_name_clean = event_name.strip().lower()

    # Check if any fight in the CSV matches this event
    matching_fights = fight_events_df[
        fight_events_df["Event"].str.strip().str.lower() == event_name_clean
    ]

    return len(matching_fights) > 0


def load_fight_events():
    """
    Load and parse the fight_events.csv file from S3.
    """
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=FIGHT_EVENTS_KEY)
        content = response["Body"].read().decode("utf-8")

        csv_reader = csv.DictReader(StringIO(content))
        return list(csv_reader)

    except Exception as e:
        print(f"Error loading fight events: {str(e)}")
        return []


def find_fight_result(fight_events, event_name, fighter1_name, fighter2_name):
    """
    Find the actual fight result from the fight_events data.
    """
    for event in fight_events:
        if event["EventName"] == event_name:
            # Check if this is the right fight (either fighter order)
            if (
                event["Fighter1_Name"] == fighter1_name
                and event["Fighter2_Name"] == fighter2_name
            ) or (
                event["Fighter1_Name"] == fighter2_name
                and event["Fighter2_Name"] == fighter1_name
            ):

                winner_num = event["Winner"]
                if winner_num == "1":
                    winner_name = event["Fighter1_Name"]
                elif winner_num == "2":
                    winner_name = event["Fighter2_Name"]
                else:
                    winner_name = None

                return {
                    "winner_name": winner_name,
                    "method": event["Method"],
                    "round": event["Round"],
                    "time": event["Time"],
                }

    return None


def extract_event_filename_from_path(path, suffix_to_remove=""):
    """
    Extract event filename from API path.
    """
    # Remove /past-predictions/ prefix and optional suffix
    filename = path.replace("/past-predictions/", "")
    if suffix_to_remove:
        filename = filename.replace(suffix_to_remove, "")
    return filename
