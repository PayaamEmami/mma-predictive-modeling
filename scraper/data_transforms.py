"""
Data transformation functions.

Pure functions for cleaning, formatting, and transforming scraped data.
Includes date parsing and measurement formatting helpers.
"""

from datetime import datetime


def parse_date(date_str: str, input_format: str = "%B %d, %Y") -> str:
    """
    Parse date string and convert to yyyy-MM-dd format.

    Args:
        date_str: Date string like "January 15, 2024" or "Sep 10, 1996"
        input_format: strptime format string

    Returns:
        Date in "yyyy-MM-dd" format or empty string if parse fails

    Examples:
        >>> parse_date("January 15, 2024")
        "2024-01-15"
        >>> parse_date("Sep 10, 1996")
        "1996-09-10"
        >>> parse_date("invalid")
        ""
    """
    # Try provided format first
    try:
        dt = datetime.strptime(date_str.strip(), input_format)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        pass

    # Try abbreviated month format (e.g., "Sep 10, 1996")
    try:
        dt = datetime.strptime(date_str.strip(), "%b %d, %Y")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        pass

    # Try full month format without comma (e.g., "September 10 1996")
    try:
        dt = datetime.strptime(date_str.strip(), "%B %d %Y")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        return ''


def format_height(height_str: str) -> str:
    """
    Convert height from 5'11" to 5ft11in format.

    Args:
        height_str: Height string with feet and inches

    Returns:
        Formatted height string

    Examples:
        >>> format_height('5\'11"')
        "5ft11in"
        >>> format_height("")
        ""
    """
    if not height_str:
        return ''
    return height_str.replace('"', 'in').replace("'", 'ft')


def format_reach(reach_str: str) -> str:
    """
    Convert reach from 72" to 72in format.

    Args:
        reach_str: Reach string with inches

    Returns:
        Formatted reach string

    Examples:
        >>> format_reach('72"')
        "72in"
    """
    if not reach_str:
        return ''
    return reach_str.replace('"', 'in')


def clean_strike_stat(stat_str: str) -> str:
    """
    Clean strike statistics string.

    Args:
        stat_str: Strike stat like "15 of 30"

    Returns:
        Cleaned string
    """
    return stat_str.strip() if stat_str else ''


def merge_fight_data(
    fight_details: dict,
    fighter1_profile: dict,
    fighter2_profile: dict,
    event_name: str,
    event_date: str,
    event_location: str
) -> dict:
    """
    Merge fight details with fighter profiles and event info into complete record.

    Args:
        fight_details: Dictionary with fight statistics and basic info
        fighter1_profile: Dictionary with Fighter1 profile data
        fighter2_profile: Dictionary with Fighter2 profile data
        event_name: Name of the event
        event_date: Date of the event (yyyy-MM-dd format)
        event_location: Location of the event

    Returns:
        Complete fight dictionary with all fields populated
    """
    fight = fight_details.copy()

    # Add event info
    fight['EventName'] = event_name
    fight['EventDate'] = event_date
    fight['EventLocation'] = event_location

    # Add Fighter1 profile data with formatting
    fight['Fighter1_DOB'] = fighter1_profile.get('DOB', '')
    fight['Fighter1_Height'] = format_height(fighter1_profile.get('Height', ''))
    fight['Fighter1_Reach'] = format_reach(fighter1_profile.get('Reach', ''))
    fight['Fighter1_Stance'] = fighter1_profile.get('Stance', '')

    # Add Fighter2 profile data with formatting
    fight['Fighter2_DOB'] = fighter2_profile.get('DOB', '')
    fight['Fighter2_Height'] = format_height(fighter2_profile.get('Height', ''))
    fight['Fighter2_Reach'] = format_reach(fighter2_profile.get('Reach', ''))
    fight['Fighter2_Stance'] = fighter2_profile.get('Stance', '')

    return fight


def initialize_empty_fight_dict() -> dict:
    """
    Create a fight dictionary with all fields initialized to empty strings.

    Returns:
        Dictionary with all CSV_HEADERS as keys, values set to empty strings
    """
    from . import config
    return {header: '' for header in config.CSV_HEADERS}
