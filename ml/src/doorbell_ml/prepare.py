import csv
import hashlib
import http.client
import io
import json
import os
import shutil
import struct
import tempfile
import time
import urllib.request
import zipfile
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data import file_digest, read_manifests


def fetch_range(url: str, start: int, end: int, total: int) -> bytes:
    for attempt in range(4):
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if (
                    response.status != 206
                    or response.headers.get("Content-Range")
                    != f"bytes {start}-{end}/{total}"
                ):
                    raise ValueError("Server did not honor the requested byte range")
                content = response.read(end - start + 2)
                if len(content) != end - start + 1:
                    raise ValueError("Incomplete or oversized range response")
                return content
        except (OSError, http.client.IncompleteRead):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("Unreachable")


class MultipartZipReader(io.RawIOBase):
    def __init__(self, parts):
        self.parts = parts
        self.starts = [0]
        for part in parts:
            self.starts.append(self.starts[-1] + part["size"])
        self.position = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        origin = {
            os.SEEK_SET: 0,
            os.SEEK_CUR: self.position,
            os.SEEK_END: self.starts[-1],
        }
        position = origin[whence] + offset
        if position < 0:
            raise ValueError("Negative ZIP offset")
        self.position = position
        return position

    def read(self, size=-1):
        remaining = max(0, self.starts[-1] - self.position)
        size = remaining if size < 0 else min(size, remaining)
        chunks = []
        while size:
            volume = bisect_right(self.starts, self.position) - 1
            part = self.parts[volume]
            start = self.position - self.starts[volume]
            count = min(size, part["size"] - start)
            chunks.append(
                fetch_range(part["url"], start, start + count - 1, part["size"])
            )
            self.position += count
            size -= count
        return b"".join(chunks)


def fsd_directory(parts):
    remote = MultipartZipReader(parts)
    remote.seek(-65536, os.SEEK_END)
    tail = remote.read()
    offset = tail.rfind(b"PK\x06\x06")
    if offset < 0:
        raise ValueError("Missing FSD50K ZIP64 directory")
    end = struct.unpack_from("<4sQ2H2L4Q", tail, offset)
    volume, count, size, start = end[5], end[7], end[8], end[9]
    remote.seek(remote.starts[volume] + start)
    central = remote.read(size)
    return central + struct.pack(
        "<4s4H2LH", b"PK\x05\x06", 0, 0, count, count, size, 0, 0
    )


def store_audio(path: Path, content: bytes, expected: str):
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Downloaded audio checksum mismatch: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as staged:
        staged_path = Path(staged.name)
        staged.write(content)
    try:
        staged_path.replace(path)
    finally:
        staged_path.unlink(missing_ok=True)


def fetch_fsd(parts, missing):
    directory = fsd_directory(parts)

    def acquire(item):
        path, expected = item
        with zipfile.ZipFile(io.BytesIO(directory)) as archive:
            info = archive.getinfo(f"FSD50K.eval_audio/{path.name}")
            remote = MultipartZipReader(parts)
            info.header_offset += remote.starts[info.volume]
            info._end_offset = None
            archive.fp = remote
            store_audio(path, archive.read(info), expected)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, _ in enumerate(pool.map(acquire, missing), 1):
            if index % 100 == 0 or index == len(missing):
                print(f"FSD50K: downloaded {index}/{len(missing)}", flush=True)


def fetch_esc(source, missing):
    with tempfile.TemporaryFile() as temporary:
        with urllib.request.urlopen(source["archive"], timeout=60) as response:
            shutil.copyfileobj(response, temporary)
        temporary.seek(0)
        with zipfile.ZipFile(temporary) as archive:
            for path, expected in missing:
                content = archive.read(source["prefix"] + path.name)
                store_audio(path, content, expected)


def prepare_data(root: Path, *, offline: bool = False):
    root = root.resolve()
    sources = json.loads((root / "public/sources.json").read_text())
    manifests = [root / "recordings.csv", root / "synthetic/manifest.csv"]
    read_manifests(manifests)
    pending = {}
    for name in ("esc50", "fsd50k"):
        manifest = root / f"public/{name}.csv"
        manifests.append(manifest)
        missing = []
        with manifest.open(newline="") as handle:
            for row in csv.DictReader(handle):
                path = (manifest.parent / row["path"]).resolve()
                if not path.is_relative_to(root / "cache" / name):
                    raise ValueError(f"Public clip must stay in its cache: {path}")
                expected = row["sha256"]
                if len(expected) != 64 or any(
                    c not in "0123456789abcdef" for c in expected
                ):
                    raise ValueError(f"Missing or invalid pinned checksum: {path}")
                if path.exists():
                    if file_digest(path) != expected:
                        raise ValueError(
                            f"Audio checksum mismatch; remove and re-fetch: {path}"
                        )
                else:
                    missing.append((path, expected))
        pending[name] = missing
    if offline and any(pending.values()):
        counts = {name: len(rows) for name, rows in pending.items()}
        raise ValueError(f"Public audio missing in offline mode: {counts}")
    for name, missing in pending.items():
        if missing:
            print(f"Fetching {len(missing)} pinned {name} clips", flush=True)
            if name == "esc50":
                fetch_esc(sources[name], missing)
            else:
                fetch_fsd(sources[name]["parts"], missing)
    records = read_manifests(manifests)
    print(f"Verified {len(records)} recordings across {len(manifests)} manifests")
    return manifests
