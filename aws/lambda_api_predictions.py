import json
import boto3
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")
BUCKET_NAME = os.environ.get("S3_BUCKET")
PREDICTIONS_KEY = "predictions/latest_predictions.json"


def lambda_handler(event, context):
    """
    Lambda function to serve latest predictions via API Gateway
    """
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    try:
        # Handle CORS preflight requests
        if event.get("httpMethod") == "OPTIONS":
            return {"statusCode": 200, "headers": headers, "body": ""}

        # Only allow GET requests
        if event.get("httpMethod") != "GET":
            return {
                "statusCode": 405,
                "headers": headers,
                "body": json.dumps({"error": "Method not allowed"}),
            }

        # Try to fetch the latest predictions from S3
        try:
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=PREDICTIONS_KEY)
            predictions_data = response["Body"].read().decode("utf-8")
            predictions = json.loads(predictions_data)

            logger.info(
                f"Successfully retrieved predictions for event: {predictions.get('event_name', 'Unknown')}"
            )

            return {"statusCode": 200, "headers": headers, "body": predictions_data}

        except s3_client.exceptions.NoSuchKey:
            logger.warning(f"No predictions file found at {PREDICTIONS_KEY}")
            return {
                "statusCode": 404,
                "headers": headers,
                "body": json.dumps(
                    {
                        "error": "No predictions available",
                        "message": "No upcoming fight predictions found",
                    }
                ),
            }

    except Exception as e:
        logger.error(f"Error serving predictions: {str(e)}")
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps(
                {
                    "error": "Internal server error",
                    "message": "Failed to retrieve predictions",
                }
            ),
        }
