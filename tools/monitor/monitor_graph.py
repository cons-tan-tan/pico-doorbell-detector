"""Live terminal chart of the scores in a timestamped Pico serial log."""

import argparse
import os
import re
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import plotext as plt

WINDOW_SECONDS = 60
RECORD = re.compile(r"^\r?\[([^]]+)\] (.*)$")
SCORES = re.compile(
    r"Scores background=([\d.+-]+)%; entrance=([\d.+-]+)%; door=([\d.+-]+)%$"
)
SERIES = (("background", "cyan"), ("entrance", "green"), ("door", "magenta"))
REQUEST_COLOR = 11  # ANSI bright yellow


class Scores:
    def __init__(self):
        self.points = deque(maxlen=2048)
        self.request_times = deque(maxlen=2048)
        self.last_stamp = None
        self.connected = False
        self.segment = 0
        self.pending = ""

    def feed(self, data):
        lines = (self.pending + data).split("\n")
        self.pending = lines.pop()[-8192:]
        for line in lines:
            match = RECORD.fullmatch(line.rstrip("\r"))
            if not match:
                continue
            stamp, body = match.groups()
            try:
                stamp = datetime.fromisoformat(stamp)
            except ValueError:
                continue
            if self.last_stamp is not None and stamp < self.last_stamp:
                # A host clock correction starts a new timeline for both layers.
                self.points.clear()
                self.request_times.clear()
            self.last_stamp = stamp
            while (
                self.request_times
                and (stamp - self.request_times[0]).total_seconds() > WINDOW_SECONDS
            ):
                self.request_times.popleft()
            values = SCORES.fullmatch(body)
            if values:
                try:
                    values = tuple(float(value) for value in values.groups())
                except ValueError:
                    continue
                if not all(0 <= value <= 100 for value in values):
                    continue
                self.points.append((stamp, values, self.segment))
                self.connected = True
                while (
                    self.points
                    and (stamp - self.points[0][0]).total_seconds() > WINDOW_SECONDS
                ):
                    self.points.popleft()
            elif body == "Notification request sent":
                self.request_times.append(stamp)
            elif body.startswith(("Connected to ", "Disconnected")):
                self.connected = body.startswith("Connected to ")
                self.segment += 1

    def request_offsets(self, now):
        for stamp in self.request_times:
            x = (stamp - now).total_seconds()
            if -WINDOW_SECONDS <= x <= 0:
                yield x

    def curves(self, now):
        """Keep real time spacing; don't join points across USB outages."""
        groups = []
        previous = None
        for stamp, values, segment in self.points:
            age = (stamp - now).total_seconds()
            if not -WINDOW_SECONDS <= age <= 0:
                continue
            if (
                previous is None
                or segment != previous[1]
                or (stamp - previous[0]).total_seconds() > 3
            ):
                groups.append([])
            groups[-1].append((age, values))
            previous = (stamp, segment)
        return groups


def render(state, width, height, now):
    status = "waiting for scores"
    latest = None
    if state.points:
        stamp, latest, _ = state.points[-1]
        age = max(0, (now - stamp).total_seconds())
        status = f"last sample {age:.1f}s ago"
        if not state.connected:
            status = "USB disconnected; " + status
    header = f"Pico live scores | {status} | q: quit"[:width]
    current = []
    for index, (name, color) in enumerate(SERIES):
        value = "--" if latest is None else f"{latest[index]:.1f}%"
        current.append(plt.colorize(f"{name}: {value}", color))
    percentages = "  ".join(current)
    if width < 50 or height < 10:
        hint = "Enlarge the chart pane to at least 50 x 10."
        return "\n".join((header, plt.uncolorize(percentages)[:width], hint[:width]))

    plt.clear_figure()
    plt.theme("clear")
    plt.plotsize(width, height - 2)
    plt.ylim(0, 100)
    plt.yticks([0, 50, 100] if height < 14 else [0, 25, 50, 75, 100])
    plt.xlim(-WINDOW_SECONDS, 0)
    plt.xticks([-60, -45, -30, -15, 0], ["-60s", "-45s", "-30s", "-15s", "now"])
    plt.ylabel("score %")
    plt.xlabel("N = notification request sent")
    for points in state.curves(now):
        x = [point[0] for point in points]
        for index, (_, color) in enumerate(SERIES):
            plt.plot(
                x, [point[1][index] for point in points], color=color, marker="braille"
            )
    for x in state.request_offsets(now):
        plt.vertical_line(x, color=REQUEST_COLOR)
        plt.text("N", x, 100, color=REQUEST_COLOR, style="bold")
    return "\n".join((header, percentages, plt.build().rstrip("\n")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path)
    args = parser.parse_args()
    if not sys.stdout.isatty():
        raise SystemExit("The graph viewer requires a terminal.")
    state = Scores()
    with args.log_path.open(encoding="utf-8", errors="replace", newline="") as source:
        while True:
            state.feed(source.read(65536))
            width, height = os.get_terminal_size()
            # tio timestamps every line with the host's local clock.
            frame = render(state, width, height, datetime.now())
            # Erase the previous frame first when a score or the pane shrinks.
            print(f"\x1b[H\x1b[J{frame}", end="", flush=True)
            time.sleep(0.2)


if __name__ == "__main__":
    main()
