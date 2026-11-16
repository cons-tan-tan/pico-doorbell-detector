# /// script
# requires-python = ">=3.13"
# dependencies = ["pynacl>=1.6,<2"]
# ///

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

from nacl.public import PublicKey, SealedBox

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "doorbell-wifi-sealed-v1"
DEFAULT_OUTPUT = ROOT / "firmware/settings/wifi.json"


def encrypt(public_key: str, ssid: str, password: str) -> dict[str, str]:
    key = bytes.fromhex(public_key)
    if len(key) != 32:
        raise ValueError("Invalid public key")
    ssid_bytes = ssid.encode("utf-8")
    password_bytes = password.encode("utf-8")
    valid_password = 8 <= len(password_bytes) <= 63 or (
        len(password_bytes) == 64
        and all(byte in b"0123456789abcdefABCDEF" for byte in password_bytes)
    )
    if not 1 <= len(ssid_bytes) <= 32 or not valid_password:
        raise ValueError("Invalid Wi-Fi credential lengths")
    payload = (
        bytes((1, len(ssid_bytes), len(password_bytes), 0))
        + ssid_bytes.ljust(32, b"\x00")
        + password_bytes.ljust(64, b"\x00")
    )
    ciphertext = SealedBox(PublicKey(key)).encrypt(payload)
    return {"format": FORMAT, "recipient": key.hex(), "ciphertext": ciphertext.hex()}


def main() -> int:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(
        description="Encrypt Wi-Fi credentials from 1Password for one Pico."
    )
    parser.add_argument("--public-key", help="64 hex digits returned by WIFI INFO")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seal", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.seal:
            result = encrypt(
                args.public_key,
                os.environ.pop("WIFI_SSID"),
                os.environ.pop("WIFI_PASS"),
            )
            print(json.dumps(result))
            return 0

        public_key = args.public_key
        if public_key is None:
            previous = json.loads(args.output.read_text())
            if previous["format"] != FORMAT:
                raise ValueError("Unknown settings format")
            public_key = previous["recipient"]
        key = bytes.fromhex(public_key)
        if len(key) != 32:
            raise ValueError("Invalid public key")

        # Resolve dependencies before SecretSpec supplies secrets to the short-lived child.
        completed = subprocess.run(
            [
                "secretspec",
                "run",
                "--file",
                str(ROOT / "secretspec.toml"),
                "--profile",
                "default",
                "--provider",
                "onepassword",
                "--reason",
                "Encrypt Pico Wi-Fi settings",
                "--",
                sys.executable,
                str(Path(__file__).resolve()),
                "--seal",
                "--public-key",
                key.hex(),
            ],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError("Credential retrieval or encryption failed")
        result = json.loads(completed.stdout)
        if (
            set(result) != {"format", "recipient", "ciphertext"}
            or result["format"] != FORMAT
            or result["recipient"] != key.hex()
            or len(bytes.fromhex(result["ciphertext"])) != 148
        ):
            raise ValueError("Invalid encryption result")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=args.output.parent, prefix=".wifi-", delete=False
            ) as file:
                temporary = Path(file.name)
                json.dump(result, file, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(args.output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"Encrypted Wi-Fi settings saved to {args.output}")
        return 0
    except Exception:  # noqa: BLE001 - Provider exceptions can contain credentials.
        print(
            "Wi-Fi encryption failed. Check the public key, credential lengths, "
            "1Password access, and output path.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
