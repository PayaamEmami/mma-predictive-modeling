import unittest
import pandas as pd
import numpy as np
from code.data import (
    parse_height,
    parse_reach,
    parse_strike,
    parse_control_time,
    parse_time,
    calculate_fight_time,
    process_landed_attempted,
    is_finish,
    compute_historical_stats,
)


class TestPreprocessing(unittest.TestCase):
    def test_parse_height_valid(self):
        self.assertAlmostEqual(parse_height("5ft 7in"), (5 * 12 + 7) * 2.54, places=2)

    def test_parse_height_only_inches(self):
        self.assertAlmostEqual(parse_height("69in"), 69 * 2.54, places=2)

    def test_parse_height_null_and_invalid(self):
        for val in (None, np.nan):
            self.assertEqual(parse_height(val), 177)
        self.assertEqual(parse_height("invalid"), 177)

    def test_parse_height_zero(self):
        self.assertEqual(parse_height("0ft 0in"), 0)

    def test_parse_height_feet_only(self):
        self.assertEqual(parse_height("6ft"), 177)

    def test_parse_height_whitespace(self):
        self.assertAlmostEqual(parse_height(" 5ft 7in "), (5 * 12 + 7) * 2.54, places=2)
        self.assertAlmostEqual(parse_height(" 69in "), 69 * 2.54, places=2)

    def test_parse_reach_valid(self):
        self.assertAlmostEqual(parse_reach("70in"), 70 * 2.54, places=2)

    def test_parse_reach_null_and_invalid(self):
        for val in (None, np.nan):
            self.assertEqual(parse_reach(val), 183)
        self.assertEqual(parse_reach("bad"), 183)

    def test_parse_reach_whitespace(self):
        self.assertAlmostEqual(parse_reach(" 70in "), 70 * 2.54, places=2)

    def test_parse_reach_invalid_format(self):
        self.assertEqual(parse_reach("72 inches"), 183)

    def test_parse_strike_valid(self):
        self.assertEqual(parse_strike("5 of 20"), [5, 20])
        self.assertEqual(parse_strike(" 7   of   8"), [7, 8])

    def test_parse_strike_null_and_invalid(self):
        for val in (None, np.nan):
            self.assertEqual(parse_strike(val), [0, 0])
        self.assertEqual(parse_strike("invalid"), [0, 0])

    def test_parse_strike_empty_and_no_match(self):
        self.assertEqual(parse_strike(""), [0, 0])
        self.assertEqual(parse_strike("5of10"), [0, 0])

    def test_parse_strike_trailing_whitespace(self):
        self.assertEqual(parse_strike("5 of 20 "), [5, 20])

    def test_parse_control_time_valid(self):
        self.assertEqual(parse_control_time("5:46"), 5 * 60 + 46)

    def test_parse_control_time_null_and_invalid(self):
        for val in (None, np.nan):
            self.assertEqual(parse_control_time(val), 0)
        self.assertEqual(parse_control_time("invalid"), 0)

    def test_parse_control_time_zero(self):
        self.assertEqual(parse_control_time("00:00"), 0)

    def test_parse_control_time_whitespace(self):
        self.assertEqual(parse_control_time(" 5:46 "), 5 * 60 + 46)

    def test_parse_time_valid(self):
        self.assertEqual(parse_time("2:30"), 2 * 60 + 30)

    def test_parse_time_null_and_invalid(self):
        for val in (None, np.nan):
            self.assertEqual(parse_time(val), 0)
        self.assertEqual(parse_time("bad"), 0)

    def test_parse_time_zero(self):
        self.assertEqual(parse_time("00:00"), 0)

    def test_parse_time_whitespace(self):
        self.assertEqual(parse_time(" 2:30 "), 2 * 60 + 30)

    def test_calculate_fight_time_valid(self):
        self.assertEqual(calculate_fight_time(2, "1:30"), 300 + 90)
        self.assertEqual(calculate_fight_time(1, "0:45"), 0 + 45)

    def test_calculate_fight_time_null(self):
        for round_val, time_val in (
            (None, "1:30"),
            (2, None),
            (np.nan, "1:30"),
            (2, np.nan),
        ):
            self.assertEqual(calculate_fight_time(round_val, time_val), 0)

    def test_calculate_fight_time_invalid_time(self):
        self.assertEqual(calculate_fight_time(2, "invalid"), 300)

    def test_process_landed_attempted(self):
        df = pd.DataFrame({"Strikes": ["5 of 10", "invalid", None, "0 of 0"]})
        process_landed_attempted(df, "Strikes")
        self.assertIn("Strikes_Landed", df)
        self.assertIn("Strikes_Attempted", df)
        self.assertEqual(df["Strikes_Landed"].tolist(), [5, 0, 0, 0])
        self.assertEqual(df["Strikes_Attempted"].tolist(), [10, 0, 0, 0])

    def test_process_landed_attempted_empty_str(self):
        df = pd.DataFrame({"Strikes": ["", "5 of 5 "]})
        process_landed_attempted(df, "Strikes")
        self.assertEqual(df["Strikes_Landed"].tolist(), [0, 5])
        self.assertEqual(df["Strikes_Attempted"].tolist(), [0, 5])

    def test_is_finish(self):
        self.assertTrue(is_finish("Submission"))
        self.assertTrue(is_finish("KO/TKO"))
        self.assertFalse(is_finish("Decision - Unanimous"))
        for val in (None, np.nan):
            self.assertFalse(is_finish(val))

    def test_compute_stats_single_fight(self):
        df = pd.DataFrame(
            {
                "EventDate": ["2020-01-01"],
                "Fight_Time_sec": [100],
                "Fighter1_ID": ["A"],
                "Fighter2_ID": ["B"],
                "Fighter1_Control_Time_sec": [30],
                "Fighter2_Control_Time_sec": [20],
                "Fighter1_Submission_Attempts": [1],
                "Fighter2_Submission_Attempts": [0],
                "Fighter1_Leg_Strikes_Landed": [5],
                "Fighter2_Leg_Strikes_Landed": [2],
                "Fighter1_Clinch_Strikes_Landed": [3],
                "Fighter2_Clinch_Strikes_Landed": [1],
                "Fighter1_Significant_Strikes_Landed": [8],
                "Fighter1_Significant_Strikes_Attempted": [10],
                "Fighter2_Significant_Strikes_Landed": [4],
                "Fighter2_Significant_Strikes_Attempted": [5],
                "Fighter1_Takedowns_Landed": [0],
                "Fighter1_Takedowns_Attempted": [0],
                "Fighter2_Takedowns_Landed": [0],
                "Fighter2_Takedowns_Attempted": [0],
                "Fighter1_Reversals": [0],
                "Fighter2_Reversals": [0],
                "Winner": ["1"],
                "Method": ["KO/TKO"],
            }
        )
        df["EventDate"] = pd.to_datetime(df["EventDate"])
        result = compute_historical_stats(df.copy())
        self.assertEqual(result.loc[0, "Fighter1_AvgFightTime"], 0)
        self.assertEqual(result.loc[0, "Fighter1_TimeSinceLastFight"], 0)
        self.assertEqual(result.loc[0, "Fighter1_FinishRate"], 0)
        self.assertEqual(result.loc[0, "Fighter1_StrikeAccuracy"], 0)
        self.assertEqual(result.loc[0, "Fighter1_Wins"], 0)
        self.assertEqual(result.loc[0, "Fighter1_Losses"], 0)
        self.assertEqual(result.loc[0, "Fighter2_AvgFightTime"], 0)
        self.assertEqual(result.loc[0, "Fighter2_Wins"], 0)

    def test_compute_stats_two_fights(self):
        df = pd.DataFrame(
            {
                "EventDate": ["2020-01-01", "2020-01-08"],
                "Fight_Time_sec": [100, 200],
                "Fighter1_ID": ["A", "A"],
                "Fighter2_ID": ["B", "B"],
                "Fighter1_Control_Time_sec": [30, 50],
                "Fighter2_Control_Time_sec": [20, 40],
                "Fighter1_Submission_Attempts": [1, 2],
                "Fighter2_Submission_Attempts": [0, 1],
                "Fighter1_Leg_Strikes_Landed": [5, 10],
                "Fighter2_Leg_Strikes_Landed": [2, 4],
                "Fighter1_Clinch_Strikes_Landed": [3, 6],
                "Fighter2_Clinch_Strikes_Landed": [1, 2],
                "Fighter1_Significant_Strikes_Landed": [8, 16],
                "Fighter1_Significant_Strikes_Attempted": [10, 20],
                "Fighter2_Significant_Strikes_Landed": [4, 8],
                "Fighter2_Significant_Strikes_Attempted": [5, 10],
                "Fighter1_Takedowns_Landed": [0, 1],
                "Fighter1_Takedowns_Attempted": [0, 2],
                "Fighter2_Takedowns_Landed": [0, 0],
                "Fighter2_Takedowns_Attempted": [0, 0],
                "Fighter1_Reversals": [0, 1],
                "Fighter2_Reversals": [0, 0],
                "Winner": ["1", "2"],
                "Method": ["KO/TKO", "Decision - Unanimous"],
            }
        )
        df["EventDate"] = pd.to_datetime(df["EventDate"])
        result = compute_historical_stats(df.copy())
        idx = 1
        self.assertEqual(result.loc[idx, "Fighter1_AvgFightTime"], 100)
        self.assertEqual(result.loc[idx, "Fighter1_TimeSinceLastFight"], 7)
        self.assertEqual(result.loc[idx, "Fighter1_FinishRate"], 1.0)
        self.assertAlmostEqual(
            result.loc[idx, "Fighter1_StrikeAccuracy"], 0.8, places=2
        )
        self.assertEqual(result.loc[idx, "Fighter1_Wins"], 1)
        self.assertEqual(result.loc[idx, "Fighter2_Wins"], 0)
        self.assertEqual(result.loc[idx, "Fighter2_Losses"], 1)
        self.assertEqual(result.loc[idx, "Fighter2_AvgFightTime"], 100)
        self.assertAlmostEqual(
            result.loc[idx, "Fighter2_StrikeAccuracy"], 4 / 5, places=2
        )


if __name__ == "__main__":
    unittest.main()
