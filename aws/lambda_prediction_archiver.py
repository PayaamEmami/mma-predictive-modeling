import json
import boto3
import re
import os
from urllib.parse import unquote_plus
from datetime import datetime

s3_client = boto3.client("s3")

ARCHIVE_PREFIX = os.environ.get("ARCHIVE_PREFIX")
PREDICTIONS_PREFIX = os.environ.get("PREDICTIONS_PREFIX")
PREDICTIONS_FILENAME = os.environ.get("PREDICTIONS_FILENAME")


def lambda_handler(event, context):
    """
    Lambda function to archive prediction files when latest_predictions.json is uploaded.
    Creates a copy of the file with a name based on the event name from the JSON content.
    """

    try:
        # Get the bucket and object key from the S3 event
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = unquote_plus(event["Records"][0]["s3"]["object"]["key"])

        print(f"Processing file: {key} in bucket: {bucket}")

        # Only process latest_predictions.json files in the predictions/ folder
        if not (
            key.startswith(PREDICTIONS_PREFIX) and key.endswith(PREDICTIONS_FILENAME)
        ):
            print(
                f"Skipping file {key} - not a {PREDICTIONS_FILENAME} in {PREDICTIONS_PREFIX} folder"
            )
            return {
                "statusCode": 200,
                "body": json.dumps(f"File skipped - not a {PREDICTIONS_FILENAME}"),
            }

        # Download the file content
        response = s3_client.get_object(Bucket=bucket, Key=key)
        file_content = response["Body"].read().decode("utf-8")

        # Parse the JSON to get the event name
        prediction_data = json.loads(file_content)
        event_name = prediction_data.get("event_name", "Unknown Event")
        event_date = prediction_data.get("event_date", "")

        print(f"Event name: {event_name}")
        print(f"Event date: {event_date}")

        # Convert event name to filename format
        # Example: "UFC 319: Du Plessis vs. Chimaev" -> "ufc-319-du-plessis-vs-chimaev"
        filename = convert_event_name_to_filename(event_name)

        # Create the new file path in the archive folder
        archive_key = f"{ARCHIVE_PREFIX}{filename}.json"

        print(f"Creating archive file: {archive_key}")

        # Copy the file to the archive location
        s3_client.put_object(
            Bucket=bucket,
            Key=archive_key,
            Body=file_content,
            ContentType="application/json",
            Metadata={
                "original-key": key,
                "event-name": event_name,
                "event-date": event_date,
                "archived-at": datetime.utcnow().isoformat(),
            },
        )

        print(f"Successfully archived {key} as {archive_key}")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Prediction file archived successfully",
                    "original_file": key,
                    "archived_file": archive_key,
                    "event_name": event_name,
                }
            ),
        }

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"error": "Error archiving prediction file", "details": str(e)}
            ),
        }


def convert_event_name_to_filename(event_name):
    """
    Convert event name to a filename-safe format.
    """
    # Convert to lowercase
    filename = event_name.lower()

    # Replace common separators with hyphens
    filename = re.sub(r"[:\-–—]+", "-", filename)

    # Replace spaces and other non-alphanumeric characters with hyphens
    filename = re.sub(r"[^a-z0-9]+", "-", filename)

    # Remove leading/trailing hyphens and multiple consecutive hyphens
    filename = re.sub(r"^-+|-+$", "", filename)
    filename = re.sub(r"-+", "-", filename)

    return filename


def create_prediction_metadata(prediction_data):
    """
    Create metadata for the prediction file to make querying easier.
    """
    fights = prediction_data.get("fights", [])

    metadata = {
        "event_name": prediction_data.get("event_name", ""),
        "event_date": prediction_data.get("event_date", ""),
        "event_location": prediction_data.get("event_location", ""),
        "total_fights": len(fights),
        "generated_at": prediction_data.get("generated_at", ""),
        "weight_classes": list(set(fight.get("weight_class", "") for fight in fights)),
    }

    return metadata
