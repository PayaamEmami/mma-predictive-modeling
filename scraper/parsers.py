"""
HTML parsing functions.

Functions to extract MMA fight data from event pages.
Uses BeautifulSoup for robust HTML parsing.
"""

from bs4 import BeautifulSoup
from typing import Optional
from . import data_transforms


def parse_events_page(html: str) -> list[tuple[str, str, str]]:
    """
    Parse the MMA events listing page.

    Args:
        html: HTML content of events page

    Returns:
        List of (event_url, event_name, event_date_str) tuples
    """
    soup = BeautifulSoup(html, 'html.parser')
    events = []

    # Find all event rows in the table
    rows = soup.select('tbody tr.b-statistics__table-row_type_first, tbody tr.b-statistics__table-row')

    for row in rows:
        link_elem = row.select_one('a')
        date_elem = row.select_one('.b-statistics__date')

        if link_elem and date_elem:
            event_url = link_elem.get('href', '')
            event_name = link_elem.get_text(strip=True)
            event_date_str = date_elem.get_text(strip=True)

            if event_url and event_name and event_date_str:
                events.append((event_url, event_name, event_date_str))

    return events


def parse_event_page(html: str) -> tuple[str, str, list[str]]:
    """
    Parse a single MMA event page to extract event details and fight URLs.

    Args:
        html: HTML content of event page

    Returns:
        Tuple of (event_name, event_location, list_of_fight_detail_urls)
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Extract event name
    name_elem = soup.select_one('.b-content__title-highlight')
    event_name = name_elem.get_text(strip=True) if name_elem else ''

    # Extract location - find the li that contains "Location:" in its title
    event_location = ''
    location_items = soup.select('ul.b-list__box-list li.b-list__box-list-item')
    for item in location_items:
        title_elem = item.select_one('i.b-list__box-item-title')
        if title_elem and 'Location:' in title_elem.get_text():
            # Get all text from the li, then remove the "Location:" label
            location_text = item.get_text(separator=' ', strip=True)
            event_location = location_text.replace('Location:', '').strip()
            break

    # Extract fight detail URLs from data-link attributes
    fight_rows = soup.select('tbody.b-fight-details__table-body tr')
    fight_urls = []
    for row in fight_rows:
        fight_url = row.get('data-link', '')
        if fight_url:
            fight_urls.append(fight_url)

    return event_name, event_location, fight_urls


def parse_fight_details(html: str) -> dict:
    """
    Parse fight details page to extract all fight statistics.

    Args:
        html: HTML content of fight details page

    Returns:
        Dictionary with fight statistics and fighter info
    """
    soup = BeautifulSoup(html, 'html.parser')
    fight_data = data_transforms.initialize_empty_fight_dict()

    # Extract fighter names and profile URLs
    fighter_anchors = soup.select('td.b-fight-details__table-col.l-page_align_left p.b-fight-details__table-text a.b-link')

    if len(fighter_anchors) >= 2:
        fight_data['Fighter1_Name'] = fighter_anchors[0].get_text(strip=True)
        fight_data['Fighter1_ID'] = fighter_anchors[0].get('href', '')
        fight_data['Fighter2_Name'] = fighter_anchors[1].get_text(strip=True)
        fight_data['Fighter2_ID'] = fighter_anchors[1].get('href', '')

    # Determine winner from outcome indicators
    outcome_elems = soup.select('i.b-fight-details__person-status')
    if len(outcome_elems) >= 2:
        f1_outcome = outcome_elems[0].get_text(strip=True)
        f2_outcome = outcome_elems[1].get_text(strip=True)

        if f1_outcome == 'W':
            fight_data['Winner'] = '1'
        elif f2_outcome == 'W':
            fight_data['Winner'] = '2'
        elif 'NC' in (f1_outcome, f2_outcome):
            fight_data['Winner'] = 'NC'
        elif f1_outcome == 'D' and f2_outcome == 'D':
            fight_data['Winner'] = 'D'
        else:
            fight_data['Winner'] = '0'

    # Extract weight class
    weight_elem = soup.select_one('div.b-fight-details__fight-head i.b-fight-details__fight-title')
    if weight_elem:
        fight_data['WeightClass'] = weight_elem.get_text(strip=True)

    # Extract method, round, and time
    fight_content = soup.select('p.b-fight-details__text')
    if fight_content:
        detail_text = fight_content[0]

        # Method
        method_elem = detail_text.select_one('i[style*="font-style: normal"]')
        if method_elem:
            fight_data['Method'] = method_elem.get_text(strip=True)

        # Round - find the label and get next sibling text
        round_label = detail_text.select_one('i.b-fight-details__text-item:nth-of-type(2) i.b-fight-details__label')
        if round_label and round_label.next_sibling:
            round_text = str(round_label.next_sibling).strip()
            fight_data['Round'] = round_text

        # Time - similar approach
        time_label = detail_text.select_one('i.b-fight-details__text-item:nth-of-type(3) i.b-fight-details__label')
        if time_label and time_label.next_sibling:
            time_text = str(time_label.next_sibling).strip()
            fight_data['Time'] = time_text

    # Extract fight statistics from tables
    # Table 0: Overall totals
    # Table 1: Round-specific data (skip)
    # Table 2: Significant strikes breakdown
    # Table 3: Round-specific strikes (skip)
    fight_tables = soup.select('tbody.b-fight-details__table-body')

    if len(fight_tables) >= 3:  # Need at least tables 0 and 2
        totals_table = fight_tables[0]
        significant_strikes_table = fight_tables[2]

        # Extract from totals table
        totals_rows = totals_table.select('tr')
        if totals_rows:
            totals_row = totals_rows[0]

            # Helper function to extract fighter stats from a column
            def get_fighter_stats(row, col_index):
                td = row.select_one(f'td:nth-child({col_index})')
                if not td:
                    return '', ''
                p_elems = td.select('p.b-fight-details__table-text')
                if len(p_elems) >= 2:
                    f1_stat = p_elems[0].get_text(strip=True)
                    f2_stat = p_elems[1].get_text(strip=True)
                    return f1_stat, f2_stat
                return '', ''

            # Column 2: Knockdowns
            f1_kd, f2_kd = get_fighter_stats(totals_row, 2)
            fight_data['Fighter1_Knockdowns'] = f1_kd
            fight_data['Fighter2_Knockdowns'] = f2_kd

            # Column 3: Significant Strikes
            f1_sig, f2_sig = get_fighter_stats(totals_row, 3)
            fight_data['Fighter1_Significant_Strikes'] = f1_sig
            fight_data['Fighter2_Significant_Strikes'] = f2_sig

            # Column 6: Takedowns
            f1_td, f2_td = get_fighter_stats(totals_row, 6)
            fight_data['Fighter1_Takedowns'] = f1_td
            fight_data['Fighter2_Takedowns'] = f2_td

            # Column 8: Submission Attempts
            f1_sub, f2_sub = get_fighter_stats(totals_row, 8)
            fight_data['Fighter1_Submission_Attempts'] = f1_sub
            fight_data['Fighter2_Submission_Attempts'] = f2_sub

            # Column 9: Reversals
            f1_rev, f2_rev = get_fighter_stats(totals_row, 9)
            fight_data['Fighter1_Reversals'] = f1_rev
            fight_data['Fighter2_Reversals'] = f2_rev

            # Column 10: Control Time
            f1_ctrl, f2_ctrl = get_fighter_stats(totals_row, 10)
            fight_data['Fighter1_Control_Time'] = f1_ctrl
            fight_data['Fighter2_Control_Time'] = f2_ctrl

        # Extract from significant strikes breakdown table
        sig_strikes_rows = significant_strikes_table.select('tr')
        if sig_strikes_rows:
            sig_row = sig_strikes_rows[0]

            # Column 4: Head Strikes
            f1_head, f2_head = get_fighter_stats(sig_row, 4)
            fight_data['Fighter1_Head_Strikes'] = f1_head
            fight_data['Fighter2_Head_Strikes'] = f2_head

            # Column 5: Body Strikes
            f1_body, f2_body = get_fighter_stats(sig_row, 5)
            fight_data['Fighter1_Body_Strikes'] = f1_body
            fight_data['Fighter2_Body_Strikes'] = f2_body

            # Column 6: Leg Strikes
            f1_leg, f2_leg = get_fighter_stats(sig_row, 6)
            fight_data['Fighter1_Leg_Strikes'] = f1_leg
            fight_data['Fighter2_Leg_Strikes'] = f2_leg

            # Column 7: Distance Strikes
            f1_dist, f2_dist = get_fighter_stats(sig_row, 7)
            fight_data['Fighter1_Distance_Strikes'] = f1_dist
            fight_data['Fighter2_Distance_Strikes'] = f2_dist

            # Column 8: Clinch Strikes
            f1_clinch, f2_clinch = get_fighter_stats(sig_row, 8)
            fight_data['Fighter1_Clinch_Strikes'] = f1_clinch
            fight_data['Fighter2_Clinch_Strikes'] = f2_clinch

            # Column 9: Ground Strikes
            f1_ground, f2_ground = get_fighter_stats(sig_row, 9)
            fight_data['Fighter1_Ground_Strikes'] = f1_ground
            fight_data['Fighter2_Ground_Strikes'] = f2_ground

    return fight_data


def parse_fighter_profile(html: str) -> dict:
    """
    Parse fighter profile page to extract biographical information.

    Args:
        html: HTML content of fighter profile page

    Returns:
        Dictionary with DOB, Height, Reach, Stance
    """
    soup = BeautifulSoup(html, 'html.parser')
    profile = {}

    # Find all list items in the profile
    list_items = soup.select('li.b-list__box-list-item')

    for item in list_items:
        text = item.get_text()

        if 'DOB:' in text:
            dob_text = text.split('DOB:')[-1].strip()
            # Parse and format the date
            profile['DOB'] = data_transforms.parse_date(dob_text)

        elif 'Height:' in text:
            profile['Height'] = text.split('Height:')[-1].strip()

        elif 'Reach:' in text:
            profile['Reach'] = text.split('Reach:')[-1].strip()

        elif 'STANCE:' in text.upper():
            profile['Stance'] = text.split(':')[-1].strip()

    # Set defaults for missing fields
    profile.setdefault('DOB', '')
    profile.setdefault('Height', '')
    profile.setdefault('Reach', '')
    profile.setdefault('Stance', '')

    return profile


def parse_upcoming_event(html: str) -> dict:
    """
    Parse upcoming event page to extract matchup information.

    Args:
        html: HTML content of upcoming event page

    Returns:
        Dictionary with event details and list of upcoming fights
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Extract event name
    name_elem = soup.select_one('.b-content__title-highlight')
    event_name = name_elem.get_text(strip=True) if name_elem else ''

    # Extract location - find the li that contains "Location:" in its title
    event_location = ''
    location_items = soup.select('ul.b-list__box-list li.b-list__box-list-item')
    for item in location_items:
        title_elem = item.select_one('i.b-list__box-item-title')
        if title_elem and 'Location:' in title_elem.get_text():
            # Get all text from the li, then remove the "Location:" label
            location_text = item.get_text(separator=' ', strip=True)
            event_location = location_text.replace('Location:', '').strip()
            break

    event_data = {
        'EventName': event_name,
        'EventLocation': event_location,
        'Fights': []
    }

    # Extract fight rows
    fight_rows = soup.select('tbody.b-fight-details__table-body tr')

    for row in fight_rows:
        # Get fighter cell (2nd column)
        fighter_cell = row.select_one('td:nth-child(2)')
        if not fighter_cell:
            continue

        # Find fighter links
        fighter_anchors = fighter_cell.select('p.b-fight-details__table-text a.b-link')

        if len(fighter_anchors) < 2:
            continue

        fighter1_name = fighter_anchors[0].get_text(strip=True)
        fighter1_url = fighter_anchors[0].get('href', '')
        fighter2_name = fighter_anchors[1].get_text(strip=True)
        fighter2_url = fighter_anchors[1].get('href', '')

        # Skip if missing essential data
        if not all([fighter1_name, fighter1_url, fighter2_name, fighter2_url]):
            continue

        # Extract weight class (7th column)
        weight_cell = row.select_one('td:nth-child(7)')
        weight_class = ''
        if weight_cell:
            weight_text = weight_cell.get_text(strip=True)
            # Clean up weight class text
            weight_class = weight_text.replace('\n', ' ').replace('\r', '')
            # Remove any HTML artifacts
            if '<' in weight_class:
                weight_class = weight_class[:weight_class.index('<')].strip()

        fight = {
            'Fighter1Name': fighter1_name,
            'Fighter1Url': fighter1_url,
            'Fighter2Name': fighter2_name,
            'Fighter2Url': fighter2_url,
            'WeightClass': weight_class
        }

        event_data['Fights'].append(fight)

    return event_data

