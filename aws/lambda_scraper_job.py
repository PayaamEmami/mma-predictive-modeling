"""
AWS Lambda Function: Data Scraper Job Trigger
==============================================

This Lambda function triggers data scraping jobs for collecting MMA fight data.
Supports both historical data collection and upcoming fights extraction.

Triggered by:
- EventBridge scheduled rules (cron)
- Manual invocation for ad-hoc scraping
- S3 events for re-scraping (optional)

Key Features:
- Dual-mode operation: historical vs upcoming fights
- Downloads scraper code from S3 (deployed via GitHub workflow)
- Automatic cleanup of downloaded artifacts
- Comprehensive error handling and logging

Scraping Modes:
- Historical: Scrapes all completed events and updates fight_events.csv
- Upcoming: Scrapes next upcoming event for predictions (upcoming_fights.json)

Dependencies:
- Environment variables: S3_BUCKET, CODE_S3_KEY, EVENTS_URL
- Scraper code package deployed to S3
- Appropriate S3 permissions for data read/write

Code Deployment:
- GitHub workflow zips repo and uploads to S3
- Lambda downloads zip, extracts, and imports scraper module
- Scraper uploads results directly to S3
"""

import boto3
import json
import os
import sys
import tarfile
from pathlib import Path


def get_config_from_env():
    """
    Get configuration from environment variables.

    Returns:
        Dictionary with configuration values
    """
    config = {
        's3_bucket': os.environ.get('S3_BUCKET'),
        'code_s3_key': os.environ.get('CODE_S3_KEY', 'code.tar.gz'),
        'events_url': os.environ.get('EVENTS_URL'),
        'csv_key': os.environ.get('S3_CSV_KEY', 'data/fight_events.csv'),
        'json_key': os.environ.get('S3_JSON_KEY', 'data/upcoming_fights.json'),
    }

    return config


def lambda_handler(event, context):
    """
    AWS Lambda handler for scraper invocation.

    Args:
        event: Lambda event object
            {
                "mode": "historical" | "upcoming"  # Optional, defaults to historical
            }
        context: Lambda context object

    Returns:
        Response with status and message
    """
    # Determine scraping mode
    mode = event.get('mode', 'historical')
    print(f"Starting data scraper in {mode} mode")

    # Load configuration from environment
    config = get_config_from_env()
    s3_bucket = config['s3_bucket']
    code_s3_key = config['code_s3_key']

    if not s3_bucket:
        error_msg = "Missing required environment variable: S3_BUCKET"
        print(f"Error: {error_msg}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': error_msg})
        }

    try:
        # Download scraper code from S3
        print(f"Downloading scraper code from s3://{s3_bucket}/{code_s3_key}")
        download_code_from_s3(s3_bucket, code_s3_key)

        # Import and run scraper
        print("Importing scraper module...")
        from scraper.main import scrape_historical_data, scrape_upcoming_fights

        # Execute appropriate scraping mode
        if mode == 'upcoming':
            print("Executing upcoming fights scraper...")
            scrape_upcoming_fights()
            message = "Successfully scraped upcoming fights"
        else:
            print("Executing historical data scraper...")
            scrape_historical_data()
            message = "Successfully scraped historical data"

        print(f"✓ {message}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': message,
                'mode': mode
            })
        }

    except Exception as e:
        error_msg = f"Error in scraper: {str(e)}"
        print(f"✗ {error_msg}")
        import traceback
        traceback.print_exc()

        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': error_msg,
                'mode': mode
            })
        }

    finally:
        # Cleanup downloaded artifacts
        cleanup_temp_files()


def download_code_from_s3(bucket: str, key: str) -> None:
    """
    Download and extract scraper code from S3.

    Args:
        bucket: S3 bucket name
        key: S3 object key for the code tar.gz file
    """
    s3_client = boto3.client('s3')

    # Create temp directory for code
    temp_dir = Path('/tmp/scraper-code')
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Download tar.gz file
    tar_path = temp_dir / 'code.tar.gz'
    s3_client.download_file(bucket, key, str(tar_path))
    print(f"Downloaded {key} to {tar_path}")

    # Extract tar.gz
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(temp_dir)
    print(f"Extracted code to {temp_dir}")

    # Add to Python path so we can import
    sys.path.insert(0, str(temp_dir))
    print(f"Added {temp_dir} to Python path")


def cleanup_temp_files() -> None:
    """
    Clean up temporary files and directories.

    Removes downloaded code and extracted files from /tmp.
    """
    import shutil

    temp_dir = Path('/tmp/scraper-code')
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Failed to cleanup temp files: {e}")

