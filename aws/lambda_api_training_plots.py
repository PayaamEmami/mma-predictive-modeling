"""
AWS Lambda Function: Training Plots and Metrics API
===================================================

This Lambda function serves training curve plots and model performance metrics
via API Gateway. It retrieves visualization data and performance metrics from S3.

API Endpoints:
- GET /training/overview - Get training overview with model performance metrics
- GET /training/plot?name={plot_name} - Get specific training plot as base64-encoded image
- GET /training/plots - List all available training plots

Dependencies:
- S3 bucket with results/ folder containing training artifacts
- Environment variable: S3_BUCKET
"""

import json
import boto3
import logging
import os
import base64
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")
BUCKET_NAME = os.environ.get("S3_BUCKET")
RESULTS_PREFIX = "results/"


def lambda_handler(event, context):
    """
    Lambda function to handle training plots and metrics API endpoints:

    Endpoints:
    - GET /training/overview - Get training overview with model performance metrics
    - GET /training/plot?name={plot_name} - Get specific training plot as base64-encoded image
    - GET /training/plots - List all available training plots

    Returns:
    - Training metrics, model performance data, and visualization plots
    - Base64-encoded PNG images for plot endpoints
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

        # Get path parameter to determine what to return
        path_params = event.get("pathParameters", {})
        resource_type = path_params.get("type", "overview")

        if resource_type == "overview":
            return get_training_overview()
        elif resource_type == "plot":
            plot_name = event.get("queryStringParameters", {}).get("name")
            if not plot_name:
                return {
                    "statusCode": 400,
                    "headers": headers,
                    "body": json.dumps({"error": "Plot name parameter required"}),
                }
            return get_plot_image(plot_name)
        else:
            return {
                "statusCode": 404,
                "headers": headers,
                "body": json.dumps({"error": "Resource not found"}),
            }

    except Exception as e:
        logger.error(f"Error serving training data: {str(e)}")
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps(
                {
                    "error": "Internal server error",
                    "message": "Failed to retrieve training data",
                }
            ),
        }


def get_training_overview():
    """Get overview of available training curves and metrics"""
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    try:
        # List all objects in the results folder
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=RESULTS_PREFIX)

        if "Contents" not in response:
            return {
                "statusCode": 404,
                "headers": headers,
                "body": json.dumps(
                    {
                        "error": "No training results found",
                        "message": "No model training results available",
                    }
                ),
            }

        # Organize files by type - only process PNG files for display
        learning_curves = []
        comparison_plots = []

        # Filter to exclude automation files and only include PNG files
        png_files = [
            obj
            for obj in response["Contents"]
            if obj["Key"].endswith(".png") and not obj["Key"].endswith("/done.json")
        ]

        for obj in png_files:
            key = obj["Key"]
            filename = key.replace(RESULTS_PREFIX, "")

            if filename.startswith("learning_curve_") and filename.endswith(".png"):
                model_name = filename.replace("learning_curve_", "").replace(".png", "")
                learning_curves.append(
                    {
                        "model_name": model_name,
                        "filename": filename,
                        "last_modified": obj["LastModified"].isoformat(),
                        "size": obj["Size"],
                    }
                )
            elif (
                filename.endswith("comparison.png")
                or filename.endswith("_report.png")
                or "accuracy" in filename.lower()
                or "performance" in filename.lower()
            ):
                comparison_plots.append(
                    {
                        "name": filename.replace(".png", ""),
                        "filename": filename,
                        "last_modified": obj["LastModified"].isoformat(),
                        "size": obj["Size"],
                    }
                )

        overview_data = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_models": len(learning_curves),
            "learning_curves": sorted(learning_curves, key=lambda x: x["model_name"]),
            "comparison_plots": sorted(comparison_plots, key=lambda x: x["name"]),
            "api_endpoints": {
                "plot_url": "?type=plot&name={filename}",
                "example": "?type=plot&name=learning_curve_Random_Forest.png",
            },
        }

        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps(overview_data, indent=2),
        }

    except Exception as e:
        logger.error(f"Error getting training overview: {str(e)}")
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps(
                {"error": "Failed to get training overview", "message": str(e)}
            ),
        }


def get_plot_image(plot_name):
    """Get a specific plot image as a direct S3 URL or fallback to base64"""
    headers_image = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    try:
        # Sanitize and validate the plot name
        if not plot_name.endswith(".png"):
            plot_name += ".png"

        # Security check: ensure it's a valid PNG filename and doesn't contain path traversal
        if (
            not plot_name.endswith(".png")
            or ".." in plot_name
            or "/" in plot_name
            or "\\" in plot_name
            or plot_name in ["done.json", "report.txt", "model_performances.csv"]
        ):
            return {
                "statusCode": 400,
                "headers": headers_image,
                "body": json.dumps(
                    {
                        "error": "Invalid plot name",
                        "message": "Only PNG image files are allowed",
                    }
                ),
            }

        object_key = f"{RESULTS_PREFIX}{plot_name}"

        # Check if object exists first
        try:
            head_response = s3_client.head_object(Bucket=BUCKET_NAME, Key=object_key)
        except s3_client.exceptions.NoSuchKey:
            return {
                "statusCode": 404,
                "headers": headers_image,
                "body": json.dumps(
                    {
                        "error": "Plot not found",
                        "message": f"Plot '{plot_name}' does not exist",
                    }
                ),
            }

        # Generate a presigned URL for direct access (expires in 1 hour)
        try:
            presigned_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": BUCKET_NAME, "Key": object_key},
                ExpiresIn=3600,  # 1 hour
            )

            return {
                "statusCode": 200,
                "headers": headers_image,
                "body": json.dumps(
                    {
                        "filename": plot_name,
                        "content_type": "image/png",
                        "url": presigned_url,  # Direct S3 URL
                        "size": head_response.get("ContentLength", 0),
                        "last_modified": (
                            head_response.get("LastModified", "").isoformat()
                            if head_response.get("LastModified")
                            else None
                        ),
                        "expires_at": (datetime.utcnow().timestamp() + 3600)
                        * 1000,  # JavaScript timestamp
                        "delivery_method": "presigned_url",
                    }
                ),
            }
        except Exception as e:
            logger.warning(
                f"Failed to generate presigned URL, falling back to base64: {e}"
            )

            # Fallback to base64 if presigned URL fails
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=object_key)
            image_data = response["Body"].read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            return {
                "statusCode": 200,
                "headers": headers_image,
                "body": json.dumps(
                    {
                        "filename": plot_name,
                        "content_type": "image/png",
                        "data": image_base64,  # Base64 fallback
                        "size": len(image_data),
                        "last_modified": (
                            response.get("LastModified", "").isoformat()
                            if response.get("LastModified")
                            else None
                        ),
                        "delivery_method": "base64_fallback",
                    }
                ),
            }

    except Exception as e:
        logger.error(f"Error getting plot image: {str(e)}")
        return {
            "statusCode": 500,
            "headers": headers_image,
            "body": json.dumps(
                {"error": "Failed to get plot image", "message": str(e)}
            ),
        }
