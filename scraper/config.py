"""
Configuration constants for MMA data scraper.

All configuration values, URLs, file paths, and constants used throughout
the scraper are centralized here for easy maintenance.

Sensitive values (S3 configuration) are loaded from environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# File paths
CSV_FILE_PATH = "./data/fight_events.csv"
JSON_FILE_PATH = "./data/upcoming_fights.json"

# S3 configuration (loaded from environment variables)
# Uses existing AWS_S3_BUCKET and AWS_REGION from project .env
S3_BUCKET = os.getenv("AWS_S3_BUCKET")
S3_REGION = os.getenv("AWS_REGION")
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

