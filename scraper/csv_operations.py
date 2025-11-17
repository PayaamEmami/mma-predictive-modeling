"""
CSV file operations.

Functions for creating, reading, and writing CSV files containing fight data.
Uses Python's csv module for proper handling of quotes and special characters.
"""

import csv
from pathlib import Path
from typing import Optional


def ensure_csv_exists(file_path: str, headers: list[str]) -> None:
    """
    Create CSV file with headers if it doesn't exist.
    
    Args:
        file_path: Path to the CSV file
        headers: List of column headers
    """
    path = Path(file_path)
    
    if not path.exists():
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create CSV with headers
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(headers)
        
        print(f"Created new CSV: {file_path}")
    else:
        print(f"CSV already exists: {file_path}")


def load_processed_event_dates(file_path: str) -> set[str]:
    """
    Load all event dates that have been processed from the CSV.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        Set of event dates in "yyyy-MM-dd" format
    """
    path = Path(file_path)
    if not path.exists():
        return set()
    
    dates = set()
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                event_date = row.get('EventDate', '').strip()
                if event_date:
                    dates.add(event_date)
    except Exception as e:
        print(f"Warning: Error reading CSV dates: {e}")
        return set()
    
    return dates


def append_fight_to_csv(fight: dict, file_path: str, headers: list[str]) -> None:
    """
    Append a single fight record to the CSV file.
    
    Args:
        fight: Dictionary containing fight data
        file_path: Path to the CSV file
        headers: List of column headers (defines field order)
    """
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_ALL)
        writer.writerow(fight)


def append_fights_to_csv(fights: list[dict], file_path: str, headers: list[str]) -> None:
    """
    Append multiple fight records to the CSV file.
    
    Args:
        fights: List of fight dictionaries
        file_path: Path to the CSV file
        headers: List of column headers (defines field order)
    """
    if not fights:
        return
    
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_ALL)
        writer.writerows(fights)


def read_csv_as_dicts(file_path: str) -> list[dict]:
    """
    Read entire CSV file as list of dictionaries.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        List of dictionaries, one per row
    """
    path = Path(file_path)
    if not path.exists():
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def count_csv_rows(file_path: str) -> int:
    """
    Count the number of data rows in the CSV (excluding header).
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        Number of data rows
    """
    path = Path(file_path)
    if not path.exists():
        return 0
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Skip header
        next(reader, None)
        return sum(1 for _ in reader)

