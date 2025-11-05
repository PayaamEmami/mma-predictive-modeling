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

s3_client = boto3.client("s3")

# Get configuration from environment variables
BUCKET_NAME = os.environ.get("BUCKET_NAME")
FIGHT_EVENTS_KEY = os.environ.get("FIGHT_EVENTS_KEY")

if not BUCKET_NAME:
    raise ValueError("BUCKET_NAME environment variable is required")

# Global cache for Lambda container reuse (persists across invocations)
_fight_events_cache = None
_event_names_cache = None


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

        if fight_events is not None and len(fight_events) > 0:
            unique_events = list(
                set(fight.get("EventName", "") for fight in fight_events)
            )
            print(
                f"Available events in fight_events.csv: {len(unique_events)} unique events"
            )

        events = []
        total_fights_processed = 0

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

                total_fights_processed += accuracy.get("total_fights", 0)

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
                    "metadata": {
                        "total_events_processed": total,
                        "total_fights_processed": total_fights_processed,
                    }
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

            # Determine fight status and correctness
            fight_changed = actual_result is None
            result_type = actual_result.get("result_type") if actual_result else None
            is_correct = False

            if actual_result:
                actual_winner = actual_result["winner_name"]
                # Only calculate correctness for fights with actual winners (not NC/Draw)
                if actual_winner and result_type is None:
                    is_correct = predicted_winner == actual_winner
                    if is_correct:
                        correct_predictions += 1

            fight_result_entry = {
                "fighter1_name": fighter1_name,
                "fighter2_name": fighter2_name,
                "predicted_winner": predicted_winner,
                "actual_winner": (
                    actual_result["winner_name"] if actual_result else None
                ),
                "is_correct": is_correct,
                "fight_changed": fight_changed,
                "confidence": fight["aggregate"]["average_confidence"],
            }

            # Add result_type only for exceptional cases
            if result_type:
                fight_result_entry["result_type"] = result_type
            elif fight_changed:
                fight_result_entry["result_type"] = "changed"

            fight_results.append(fight_result_entry)

        # Calculate accuracy only for fights that actually occurred (not changed/cancelled)
        actual_fights = [
            fr for fr in fight_results
            if not fr.get("fight_changed", False)
            and not fr.get("result_type")
            and fr.get("actual_winner")  # Exclude fights with no actual winner (data issues)
        ]
        overall_accuracy = (
            (correct_predictions / len(actual_fights)) * 100 if actual_fights else 0
        )

        return {
            "overall_accuracy": round(overall_accuracy, 1),
            "correct_predictions": correct_predictions,
            "total_fights": len(
                actual_fights
            ),  # Only count fights that actually occurred
            "total_predicted_fights": len(fights),  # Total originally predicted
            "changed_fights": len(fights)
            - len(actual_fights),  # Number of changed fights
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


def load_fight_events():
    """
    Load and parse the fight_events.csv file from S3 with caching.
    Uses global cache to persist data across Lambda invocations in the same container.
    """
    global _fight_events_cache, _event_names_cache

    # Return cached data if available
    if _fight_events_cache is not None:
        print(f"Using cached fight events data ({len(_fight_events_cache)} fights)")
        return _fight_events_cache

    try:
        print("Loading fight events from S3...")
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=FIGHT_EVENTS_KEY)
        content = response["Body"].read().decode("utf-8")

        csv_reader = csv.DictReader(StringIO(content))
        _fight_events_cache = list(csv_reader)

        # Pre-compute set of unique event names for fast lookup
        _event_names_cache = set(
            fight.get("EventName", "").strip()
            for fight in _fight_events_cache
            if fight.get("EventName", "").strip()
        )

        print(f"Cached {len(_fight_events_cache)} fights, {len(_event_names_cache)} unique events")
        return _fight_events_cache

    except Exception as e:
        print(f"Error loading fight events: {str(e)}")
        return []


def event_exists_in_fight_data(fight_events, event_name):
    """
    Check if an event exists in the fight events data.
    Returns True if the event has actual fight outcome data.
    Uses cached set for O(1) lookup instead of O(n) iteration.
    """
    global _event_names_cache

    if fight_events is None or len(fight_events) == 0:
        return False

    # Use pre-computed set for fast lookup
    if _event_names_cache is not None:
        return event_name.strip() in _event_names_cache

    # Fallback to linear search if cache not available (shouldn't happen)
    for fight in fight_events:
        if fight.get("EventName", "").strip() == event_name.strip():
            return True

    return False


def find_fight_result(fight_events, event_name, fighter1_name, fighter2_name):
    """
    Find the actual fight result from the fight_events data.
    Returns result with winner_name and optional result_type for exceptional cases.
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
                result = {
                    "method": event["Method"],
                    "round": event["Round"],
                    "time": event["Time"],
                }

                if winner_num == "1":
                    result["winner_name"] = event["Fighter1_Name"]
                elif winner_num == "2":
                    result["winner_name"] = event["Fighter2_Name"]
                elif winner_num == "NC":
                    result["winner_name"] = "No Contest"
                    result["result_type"] = "no_contest"
                elif winner_num == "D":
                    result["winner_name"] = "Draw"
                    result["result_type"] = "draw"
                else:
                    result["winner_name"] = None

                return result

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
