"""
Main orchestration and CLI entry point for MMA data scraper.

Coordinates all scraping operations by composing functions from other modules.
Provides both historical data scraping and upcoming fights scraping modes.
"""

import sys
import json
from datetime import datetime
from typing import Optional

from . import config
from . import http_client
from . import parsers
from . import data_transforms
from . import csv_operations
from . import s3_operations


def scrape_historical_data() -> None:
    """
    Main function to scrape all historical MMA events.

    This is the default mode that:
    1. Downloads existing CSV from S3
    2. Identifies unprocessed events
    3. Scrapes new events
    4. Updates and uploads CSV back to S3
    """
    print("=" * 80)
    print("MMA Historical Data Scraper")
    print("=" * 80)
    print()

    # Step 1: Setup CSV
    print("Setting up CSV file...")
    csv_operations.ensure_csv_exists(config.CSV_FILE_PATH, config.CSV_HEADERS)

    # Step 2: Download existing CSV from S3
    print("\nDownloading existing data from S3...")
    try:
        s3_operations.download_from_s3(
            bucket=config.S3_BUCKET,
            s3_key=config.S3_CSV_KEY,
            local_path=config.CSV_FILE_PATH,
            region=config.S3_REGION
        )
    except Exception as e:
        print(f"Note: Could not download from S3 (will start fresh): {e}")

    # Step 3: Load already processed events
    print("\nLoading processed events...")
    processed_dates = csv_operations.load_processed_event_dates(config.CSV_FILE_PATH)
    print(f"Already processed: {len(processed_dates)} event dates")

    # Step 4: Fetch and parse events listing page
    print("\nFetching events listing...")
    try:
        events_html = http_client.fetch_html(config.EVENTS_URL)
        events = parsers.parse_events_page(events_html)
        print(f"Found {len(events)} total events on listing page")
    except Exception as e:
        print(f"Error fetching events page: {e}")
        return

    # Step 5: Filter events to process
    print("\nFiltering events to process...")
    events_to_process = []
    today = datetime.now().strftime("%Y-%m-%d")

    for event_url, event_name, event_date_str in events:
        # Parse date
        event_date = data_transforms.parse_date(event_date_str)
        if not event_date:
            continue

        # Skip future events
        if event_date > today:
            continue

        # Skip already processed
        if event_date in processed_dates:
            continue

        events_to_process.append((event_url, event_name, event_date))

    print(f"Will process {len(events_to_process)} new events")

    if not events_to_process:
        print("\nNo new events to process!")
        print("All events are up to date.")
        return

    # Step 6: Process each event
    print("\nProcessing events...")
    print("-" * 80)

    for i, (event_url, event_name, event_date) in enumerate(events_to_process, 1):
        print(f"\n[{i}/{len(events_to_process)}] {event_name} ({event_date})")
        try:
            process_single_event(event_url, event_name, event_date, config.CSV_FILE_PATH)
        except Exception as e:
            print(f"  ✗ Error processing event: {e}")
            continue

    # Step 7: Upload updated CSV to S3
    print("\n" + "=" * 80)
    print("Uploading updated data to S3...")
    try:
        s3_operations.upload_to_s3(
            local_path=config.CSV_FILE_PATH,
            bucket=config.S3_BUCKET,
            s3_key=config.S3_CSV_KEY,
            region=config.S3_REGION
        )
    except Exception as e:
        print(f"Error uploading to S3: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("Historical scraping complete!")
    total_fights = csv_operations.count_csv_rows(config.CSV_FILE_PATH)
    print(f"Total fights in database: {total_fights}")
    print("=" * 80)


def process_single_event(
    event_url: str,
    event_name: str,
    event_date: str,
    csv_path: str
) -> None:
    """
    Process a single MMA event and append all fights to CSV.

    Args:
        event_url: URL of the event page
        event_name: Name of the event
        event_date: Date of the event (yyyy-MM-dd format)
        csv_path: Path to CSV file
    """
    # Fetch event page
    try:
        event_html = http_client.fetch_html(event_url)
        parsed_event_name, event_location, fight_urls = parsers.parse_event_page(event_html)
    except Exception as e:
        raise Exception(f"Failed to fetch event page: {e}")

    # Use parsed name if available
    if parsed_event_name:
        event_name = parsed_event_name

    print(f"  Location: {event_location}")
    print(f"  Fights: {len(fight_urls)}")

    if not fight_urls:
        print("  ✗ No fights found!")
        return

    # Track winner for swapping logic
    last_winner = None
    fights_processed = 0

    # Process each fight
    for fight_url in fight_urls:
        try:
            fight = process_single_fight(
                fight_url=fight_url,
                event_name=event_name,
                event_date=event_date,
                event_location=event_location
            )

            # Apply fighter swapping logic
            should_swap, last_winner = data_transforms.should_swap_fighters(
                fight['Winner'],
                last_winner
            )

            if should_swap:
                fight = data_transforms.swap_fighter_data(fight)

            # Append to CSV
            csv_operations.append_fight_to_csv(fight, csv_path, config.CSV_HEADERS)
            print(f"    ✓ {fight['Fighter1_Name']} vs {fight['Fighter2_Name']}")
            fights_processed += 1

        except Exception as e:
            print(f"    ✗ Error processing fight: {e}")
            continue

    print(f"  Processed: {fights_processed}/{len(fight_urls)} fights")


def process_single_fight(
    fight_url: str,
    event_name: str,
    event_date: str,
    event_location: str
) -> dict:
    """
    Process a single fight - fetch details and both fighter profiles.

    Args:
        fight_url: URL of the fight details page
        event_name: Name of the event
        event_date: Date of the event (yyyy-MM-dd)
        event_location: Location of the event

    Returns:
        Complete fight dictionary with all fields

    Raises:
        Exception: If critical data cannot be fetched
    """
    # Fetch and parse fight details
    fight_html = http_client.fetch_html(fight_url)
    fight_details = parsers.parse_fight_details(fight_html)

    # Get fighter profile URLs
    fighter1_url = fight_details.get('Fighter1_ID', '')
    fighter2_url = fight_details.get('Fighter2_ID', '')

    # Fetch fighter profiles
    fighter1_profile = {}
    if fighter1_url:
        try:
            fighter1_html = http_client.fetch_html(fighter1_url)
            fighter1_profile = parsers.parse_fighter_profile(fighter1_html)
        except Exception as e:
            print(f"      Warning: Could not fetch Fighter1 profile: {e}")

    fighter2_profile = {}
    if fighter2_url:
        try:
            fighter2_html = http_client.fetch_html(fighter2_url)
            fighter2_profile = parsers.parse_fighter_profile(fighter2_html)
        except Exception as e:
            print(f"      Warning: Could not fetch Fighter2 profile: {e}")

    # Merge all data
    complete_fight = data_transforms.merge_fight_data(
        fight_details,
        fighter1_profile,
        fighter2_profile,
        event_name,
        event_date,
        event_location
    )

    return complete_fight


def scrape_upcoming_fights() -> None:
    """
    Scrape the next upcoming MMA event for prediction purposes.

    This mode:
    1. Finds the next future event
    2. Extracts fighter matchups
    3. Saves to JSON
    4. Uploads to S3
    """
    print("=" * 80)
    print("MMA Upcoming Fights Scraper")
    print("=" * 80)
    print()

    # Step 1: Fetch events listing
    print("Fetching events listing...")
    try:
        events_html = http_client.fetch_html(config.EVENTS_URL)
        events = parsers.parse_events_page(events_html)
        print(f"Found {len(events)} total events")
    except Exception as e:
        print(f"Error fetching events page: {e}")
        return

    # Step 2: Find first future event
    print("\nLooking for upcoming event...")
    upcoming_event = None
    today = datetime.now().strftime("%Y-%m-%d")

    for event_url, event_name, event_date_str in events:
        event_date = data_transforms.parse_date(event_date_str)
        if event_date and event_date > today:
            upcoming_event = (event_url, event_name, event_date)
            break

    if not upcoming_event:
        print("No upcoming events found!")
        return

    event_url, event_name, event_date = upcoming_event
    print(f"Found: {event_name} ({event_date})")

    # Step 3: Fetch and parse upcoming event
    print("\nFetching event details...")
    try:
        event_html = http_client.fetch_html(event_url)
        event_data = parsers.parse_upcoming_event(event_html)
    except Exception as e:
        print(f"Error parsing event: {e}")
        return

    # Add metadata
    event_data['EventName'] = event_name
    event_data['EventDate'] = event_date
    event_data['ScrapedAt'] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    num_fights = len(event_data['Fights'])
    print(f"Found {num_fights} upcoming fights:")
    for fight in event_data['Fights']:
        print(f"  • {fight['Fighter1Name']} vs {fight['Fighter2Name']} ({fight['WeightClass']})")

    # Step 4: Save to JSON
    print(f"\nSaving to {config.JSON_FILE_PATH}...")
    try:
        with open(config.JSON_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(event_data, f, indent=2, ensure_ascii=False)
        print("✓ Saved successfully")
    except Exception as e:
        print(f"Error saving JSON: {e}")
        return

    # Step 5: Upload to S3
    print("\nUploading to S3...")
    try:
        s3_operations.upload_to_s3(
            local_path=config.JSON_FILE_PATH,
            bucket=config.S3_BUCKET,
            s3_key=config.S3_JSON_KEY,
            region=config.S3_REGION
        )
    except Exception as e:
        print(f"Error uploading to S3: {e}")

    print("\n" + "=" * 80)
    print("Upcoming fights scraping complete!")
    print("=" * 80)


def main():
    """
    CLI entry point.

    Usage:
        python -m scraper.main              # Scrape historical data
        python -m scraper.main --upcoming   # Scrape upcoming fights
    """
    if len(sys.argv) > 1 and sys.argv[1] == '--upcoming':
        scrape_upcoming_fights()
    else:
        scrape_historical_data()


if __name__ == '__main__':
    main()

