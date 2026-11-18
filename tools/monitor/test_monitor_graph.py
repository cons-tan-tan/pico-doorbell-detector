"""Checks for score parsing, time spacing, bounded history and USB gaps."""

import unittest
from datetime import datetime, timedelta

from monitor_graph import Scores, render


def at(second):
    return datetime(2026, 11, 18, 12) + timedelta(seconds=second)


def record(second, background=90, entrance=7, door=3):
    return (
        f"[{at(second).isoformat(timespec='milliseconds')}] "
        f"Scores background={background}%; entrance={entrance}%; door={door}%\r\n"
    )


class ScoreTests(unittest.TestCase):
    def test_partial_lines_and_delayed_scores(self):
        state = Scores()
        line = record(0.1)
        state.feed(line[:35])
        self.assertFalse(state.points)
        state.feed(line[35:] + "[2026-11-18T12:00:10.200] [WARN] Audio gap\r\n")
        self.assertEqual(state.points[-1][1], (90, 7, 3))
        self.assertEqual(len(state.points), 1)
        self.assertIn("last sample 12.0s ago", render(state, 80, 10, at(12.1)))

    def test_time_axis_preserves_sample_spacing_and_ages_out(self):
        state = Scores()
        state.feed(record(0) + record(0.25, 70, 20, 10))
        points = state.curves(at(1.25))[0]
        self.assertEqual([point[0] for point in points], [-1.25, -1.0])
        self.assertEqual(points[-1][1], (70, 20, 10))
        self.assertEqual(state.curves(at(61)), [])

    def test_request_marks_use_send_times_on_the_score_axis(self):
        state = Scores()
        state.feed(
            record(0)
            + "[2026-11-18T12:00:00.250] Detected Entrance\r\n"
            + "[2026-11-18T12:00:00.300] [WARN] Notification queue full\r\n"
            + "[2026-11-18T12:00:01.000] Notification delivered\r\n"
        )
        self.assertEqual(list(state.request_offsets(at(1))), [])
        state.feed("[2026-11-18T12:00:02.000] Notification reque")
        self.assertEqual(list(state.request_offsets(at(2))), [])
        state.feed(
            "st sent\r\n"
            "[2026-11-18T12:00:03.000] [WARN] Notification attempt 1 failed: timeout\r\n"
            "[2026-11-18T12:00:05.000] Notification request sent\r\n"
            "[2026-11-18T12:00:05.250] Notification delivered\r\n"
        )
        self.assertEqual(list(state.request_offsets(at(5.25))), [-3.25, -0.25])
        self.assertEqual(state.curves(at(5.25))[0][0][0], -5.25)
        self.assertEqual(list(state.request_offsets(at(63))), [-58])
        self.assertEqual(list(state.request_offsets(at(66))), [])

    def test_usb_disconnect_and_long_silence_break_the_line(self):
        state = Scores()
        state.feed(record(0))
        state.feed("\r[2026-11-18T12:00:00.100] Disconnected\r\n")
        self.assertFalse(state.connected)
        state.feed("\r[2026-11-18T12:00:00.200] Connected to /dev/ttyACM0\r\n")
        state.feed(record(0.3) + record(10))
        self.assertTrue(state.connected)
        self.assertEqual(len(state.curves(at(10))), 3)

    def test_invalid_scores_are_ignored(self):
        state = Scores()
        for value in (-1, 101, "nan", "inf", "1.2.3"):
            state.feed(record(0, value))
        self.assertFalse(state.points)
        state.feed(record(0, 100, 0, 0))
        self.assertEqual(state.points[-1][1], (100, 0, 0))

    def test_burst_history_remains_bounded(self):
        state = Scores()
        state.feed(record(0) * 3000)
        self.assertEqual(len(state.points), 2048)
        state.feed("[2026-11-18T12:00:00.100] Notification request sent\n" * 3000)
        self.assertEqual(len(state.request_times), 2048)

    def test_clock_correction_starts_a_new_timeline(self):
        state = Scores()
        state.feed(record(40))
        state.feed("[2026-11-18T12:00:40.100] Notification request sent\n")
        state.feed(record(10, 70, 20, 10))
        self.assertEqual(len(state.points), 1)
        self.assertEqual(state.curves(at(10))[0][0][0], 0)
        self.assertEqual(list(state.request_offsets(at(10))), [])


if __name__ == "__main__":
    unittest.main()
