"""
Configuration constants for MMA data scraper.

All configuration values, URLs, file paths, and constants used throughout
the scraper are centralized here for easy maintenance.

Sensitive values (S3 configuration) are loaded from environment variables.
"""

import os

# Load environment variables from .env file (optional - Lambda uses native env vars)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not available (e.g., in Lambda) - use native environment variables
    pass

# File paths
# Use /tmp in Lambda environment, ./data locally
# Lambda sets LAMBDA_TASK_ROOT, AWS_LAMBDA_FUNCTION_NAME, or AWS_EXECUTION_ENV
_is_lambda = (
    os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None or
    os.environ.get('LAMBDA_TASK_ROOT') is not None or
    os.environ.get('AWS_EXECUTION_ENV') is not None
)
_data_dir = "/tmp" if _is_lambda else "./data"
CSV_FILE_PATH = f"{_data_dir}/fight_events.csv"
JSON_FILE_PATH = f"{_data_dir}/upcoming_fights.json"

# S3 configuration (loaded from environment variables)
# Lambda uses S3_BUCKET, local uses AWS_S3_BUCKET for consistency
S3_BUCKET = os.getenv("S3_BUCKET") or os.getenv("AWS_S3_BUCKET")
S3_REGION = os.getenv("AWS_REGION", "us-west-1")
S3_CSV_KEY = os.getenv("S3_CSV_KEY")
S3_JSON_KEY = os.getenv("S3_JSON_KEY")

# Scraping URLs
EVENTS_URL = os.getenv("EVENTS_URL", "")

# Request settings
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.5  # Seconds between requests for polite scraping
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# CSV headers - all 39 columns in exact order
CSV_HEADERS = [
    "EventName",
    "EventDate",
    "EventLocation",
    "Fighter1_ID",
    "Fighter1_Name",
    "Fighter1_DOB",
    "Fighter1_Height",
    "Fighter1_Reach",
    "Fighter1_Stance",
    "Fighter1_Takedowns",
    "Fighter1_Submission_Attempts",
    "Fighter1_Reversals",
    "Fighter1_Control_Time",
    "Fighter1_Knockdowns",
    "Fighter1_Significant_Strikes",
    "Fighter1_Head_Strikes",
    "Fighter1_Body_Strikes",
    "Fighter1_Leg_Strikes",
    "Fighter1_Distance_Strikes",
    "Fighter1_Clinch_Strikes",
    "Fighter1_Ground_Strikes",
    "Fighter2_ID",
    "Fighter2_Name",
    "Fighter2_DOB",
    "Fighter2_Height",
    "Fighter2_Reach",
    "Fighter2_Stance",
    "Fighter2_Takedowns",
    "Fighter2_Submission_Attempts",
    "Fighter2_Reversals",
    "Fighter2_Control_Time",
    "Fighter2_Knockdowns",
    "Fighter2_Significant_Strikes",
    "Fighter2_Head_Strikes",
    "Fighter2_Body_Strikes",
    "Fighter2_Leg_Strikes",
    "Fighter2_Distance_Strikes",
    "Fighter2_Clinch_Strikes",
    "Fighter2_Ground_Strikes",
    "WeightClass",
    "Method",
    "Round",
    "Time",
    "Winner"
]

