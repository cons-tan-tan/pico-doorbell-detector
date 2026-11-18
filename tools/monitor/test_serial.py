"""Exercise the packaged runners against a pseudo serial device and real lnav."""

import os
import pty
import re
import select
import signal
import socket
import subprocess
import sys
import tempfile
import termios
import time
import unittest
from pathlib import Path

MONITOR, BOOTSEL = sys.argv[1:3]
del sys.argv[1:3]


class SerialRunners(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pico-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = dict(
            os.environ,
            TERM="xterm-256color",
            XDG_RUNTIME_DIR=str(self.root),
            XDG_CONFIG_HOME=str(self.root / "config"),
            XDG_STATE_HOME=str(self.root / "state"),
        )
        self.serial, slave = pty.openpty()
        self.device = os.ttyname(slave)
        self.addCleanup(os.close, self.serial)
        self.addCleanup(os.close, slave)
        self.ui, self.ui_slave = pty.openpty()
        termios.tcsetwinsize(self.ui_slave, (32, 180))
        self.addCleanup(os.close, self.ui)
        self.addCleanup(os.close, self.ui_slave)
        self.screen = b""
        self.terminal_queries = {}

    def wait_for(self, condition, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if select.select([self.ui], [], [], 0.05)[0]:
                self.screen += os.read(self.ui, 65536)
                # notcurses queries terminal capabilities before drawing. A PTY
                # alone has no emulator to answer cursor/geometry/device queries.
                for query, response in {
                    b"\x1b[6n": b"\x1b[1;1R",
                    b"\x1b[>c": b"\x1b[>0;370;0c",
                    b"\x1b[18t": b"\x1b[8;32;180t",
                    b"\x1b[c": b"\x1b[?1;2c",
                }.items():
                    count = self.screen.count(query)
                    previous = self.terminal_queries.get(query, 0)
                    if count > previous:
                        os.write(self.ui, response * (count - previous))
                        self.terminal_queries[query] = count
            if condition():
                return
        self.fail(f"Timed out; terminal output: {self.screen[-2000:]!r}")

    @property
    def text(self):
        positioned = re.sub(rb"\x1b\[[\d;]*[GH]", b" ", self.screen)
        return re.sub(rb"\x1b\[[0-?]*[ -/]*[@-~]", b"", positioned)

    def start_monitor(self, *options):
        process = subprocess.Popen(
            [MONITOR, *options, self.device],
            stdin=self.ui_slave,
            stdout=self.ui_slave,
            stderr=self.ui_slave,
            env=self.env,
            cwd=self.root,
            start_new_session=True,
        )

        def cleanup():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)

        self.addCleanup(cleanup)
        self.wait_for(lambda: bool(list(self.root.rglob("*.sock"))))
        self.log = next((self.root / "artifacts/monitor").glob("*.log"))
        if "--raw" in options:
            self.wait_for(lambda: b"Connected to" in self.screen)
        else:
            self.wait_for(lambda: "Connected to" in self.log.read_text())
            if "--all" in options:
                self.wait_for(lambda: b"Press" in self.text)
            elif "--logs" in options:
                self.wait_for(lambda: b"filter activated" in self.text)
            else:
                self.wait_for(
                    lambda: b"Pico live scores" in self.text and b"LOG" in self.text
                )
        return process

    def capture_pane(self, index):
        sockets = list(self.root.rglob("*.tmux"))
        if not sockets:
            return ""
        result = subprocess.run(
            [
                "tmux",
                "-S",
                str(sockets[0]),
                "capture-pane",
                "-p",
                "-t",
                f"doorbell:0.{index}",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
        return result.stdout

    def send_logs(self):
        os.write(
            self.serial,
            b"Scores background=99%; entrance=0%; door=1%\r\n"
            b"Detected entrance\r\n"
            b"[WARN] Audio gap\r\n",
        )
        self.wait_for(lambda: "Audio gap" in self.log.read_text())

    def request_bootsel(self):
        result = subprocess.run(
            [BOOTSEL, self.device],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        received = b""
        deadline = time.monotonic() + 3
        while b"\nBOOTSEL\n" not in received and time.monotonic() < deadline:
            if select.select([self.serial], [], [], 0.1)[0]:
                received += os.read(self.serial, 4096)
        self.assertIn(b"\nBOOTSEL\n", received)
        return result.stdout

    def test_event_filter_full_capture_and_bootsel_while_monitoring(self):
        process = self.start_monitor("--logs")
        self.send_logs()
        self.wait_for(lambda: b"Detected entrance" in self.text)
        self.assertNotIn(b"Scores background=99%", self.text)
        self.assertIn("Scores background=99%", self.log.read_text())

        # The viewer's keys must not reach the board. Ctrl-F reveals the details.
        os.write(self.ui, b"\x06")
        self.wait_for(lambda: b"Scores background=99%" in self.text)
        self.assertFalse(select.select([self.serial], [], [], 0.1)[0])

        duplicate = subprocess.run(
            [MONITOR, self.device],
            env=self.env,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn("already using", duplicate.stderr)
        self.assertIn("through the running monitor", self.request_bootsel())

        os.write(self.ui, b"q")
        self.wait_for(lambda: process.poll() is not None)
        self.assertEqual(process.returncode, 0)
        self.assertFalse(list(self.root.rglob("*.sock")))
        # Exiting must release both the receiver and serial lock.
        self.assertIn("Requested BOOTSEL on", self.request_bootsel())

    def test_all_mode_shows_periodic_metrics(self):
        process = self.start_monitor("--all")
        self.send_logs()
        self.wait_for(lambda: b"Scores background=99%" in self.text)
        os.write(self.ui, b"q")
        self.wait_for(lambda: process.poll() is not None)
        self.assertEqual(process.returncode, 0)

    def test_bootsel_keeps_reading_while_the_monitor_forwards_the_command(self):
        self.request_bootsel()
        address = next(self.root.rglob("*.lock")).with_suffix(".sock")
        with socket.socket(socket.AF_UNIX) as server:
            server.bind(str(address))
            server.listen(1)
            server.settimeout(3)
            with subprocess.Popen(
                [BOOTSEL, self.device], env=self.env, stdout=subprocess.DEVNULL
            ) as request:
                connection, _ = server.accept()
                with connection, connection.makefile("rb") as command:
                    connection.settimeout(3)
                    self.assertEqual(command.read(), b"\nBOOTSEL\n")
                    # tio can forward a serial log before finishing the USB write.
                    connection.sendall(b"Scores background=99%; entrance=0%; door=1%\r\n")
                self.assertEqual(request.wait(timeout=3), 0)

    def test_raw_mode_accepts_keyboard_and_control_socket(self):
        process = self.start_monitor("--raw")
        self.send_logs()
        self.wait_for(lambda: b"Scores background=99%" in self.text)
        os.write(self.ui, b"WIFI INFO\n")
        self.wait_for(lambda: bool(select.select([self.serial], [], [], 0)[0]))
        self.assertIn(b"WIFI INFO", os.read(self.serial, 4096))
        self.assertIn("through the running monitor", self.request_bootsel())
        os.write(self.ui, b"\x14q")
        self.wait_for(lambda: process.poll() is not None)
        self.assertEqual(process.returncode, 0)

    def test_default_view_keeps_graph_live_while_browsing_logs(self):
        process = self.start_monitor()
        self.send_logs()
        self.wait_for(
            lambda: (
                "background: 99.0%" in self.capture_pane(0)
                and "Scores background=99%" in self.capture_pane(1)
                and "Audio gap" in self.capture_pane(1)
            )
        )
        # The legend contains N too; check the marker inside the plotted area.
        request_mark = re.compile(r"^100┤[^\n]*N", re.MULTILINE)
        self.assertNotRegex(self.capture_pane(0), request_mark)
        os.write(self.serial, b"Notification request sent\r\n")
        self.wait_for(lambda: request_mark.search(self.capture_pane(0)))
        # Focus starts in lnav. Its filtering doesn't hide the graph's data.
        os.write(self.ui, b"\x06")
        self.wait_for(lambda: "Scores background=" not in self.capture_pane(1))
        self.assertIn("background: 99.0%", self.capture_pane(0))
        os.write(self.ui, b"\x06")
        self.wait_for(lambda: "Scores background=99%" in self.capture_pane(1))

        # Scroll the log back, then keep receiving new measurements above it.
        for index in range(40):
            os.write(self.serial, f"Event {index:02d}\r\n".encode())
        self.wait_for(lambda: "Event 39" in self.log.read_text())
        os.write(self.ui, b"G")
        self.wait_for(lambda: "Event 39" in self.capture_pane(1))
        os.write(self.ui, b"\x1b[5~")
        self.wait_for(lambda: "Event 39" not in self.capture_pane(1))
        earlier = re.findall(r"Event \d+", self.capture_pane(1))
        self.assertTrue(earlier)
        os.write(self.serial, b"Scores background=21%; entrance=70%; door=9%\r\n")
        self.wait_for(
            lambda: (
                "entrance: 70.0%" in self.capture_pane(0)
                and "door: 9.0%" in self.capture_pane(0)
            )
        )
        self.assertEqual(re.findall(r"Event \d+", self.capture_pane(1)), earlier)
        self.assertIn("entrance=70%", self.log.read_text())
        os.write(self.serial, b"Scores background=91%; entrance=6%; door=3%\r\n")
        self.wait_for(lambda: "background: 91.0%" in self.capture_pane(0))
        self.assertEqual(
            self.capture_pane(0).splitlines()[1].rstrip(),
            "background: 91.0%  entrance: 6.0%  door: 3.0%",
        )
        self.assertIn("through the running monitor", self.request_bootsel())
        socket = next(self.root.rglob("*.tmux"))
        subprocess.run(
            ["tmux", "-S", str(socket), "select-pane", "-t", "doorbell:0.0"], check=True
        )
        os.write(self.ui, b"q")
        self.wait_for(lambda: process.poll() is not None)
        self.assertEqual(process.returncode, 0)
        self.assertFalse(list(self.root.rglob("*.sock")))
        self.assertFalse(list(self.root.rglob("*.tmux")))


if __name__ == "__main__":
    unittest.main()
