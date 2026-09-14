"""Unit tests for scraper CSV event deduplication."""

import csv
from pathlib import Path

from scraper.csv_operations import load_processed_events


def test_load_processed_events_keys_by_name_and_date(tmp_path: Path):
    csv_path = tmp_path / "fight_events.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["EventName", "EventDate", "Fighter1_Name"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "EventName": "UFC Fight Night",
                "EventDate": "2024-06-01",
                "Fighter1_Name": "A",
            }
        )
        writer.writerow(
            {
                "EventName": "DWCS",
                "EventDate": "2024-06-01",
                "Fighter1_Name": "B",
            }
        )

    processed = load_processed_events(str(csv_path))

    assert ("UFC Fight Night", "2024-06-01") in processed
    assert ("DWCS", "2024-06-01") in processed
    assert len(processed) == 2
    # A different event on the same date is not considered processed
    assert ("Other Card", "2024-06-01") not in processed
