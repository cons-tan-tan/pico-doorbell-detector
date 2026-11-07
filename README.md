# Pico Doorbell Detector

TinyGo firmware for the Raspberry Pi Pico WH that detects "ding-dong" doorbell
chimes through a MAX9814 microphone and sends notifications over Wi-Fi using
[ntfy](https://docs.ntfy.sh/).

The detector recognizes two chime patterns using 850 Hz and 662 Hz tones; other
chimes may need tuning. In current use, chimes are rarely missed, though
occasional false positives occur.

## Hardware and requirements

- **Raspberry Pi Pico WH**, built with TinyGo's `pico-w` target.
- **MAX9814 analog microphone module**, with its output connected to `ADC0`.
- A WPA2 Wi-Fi network and an ntfy server reachable from the Pico over HTTP.

## Development

With Nix flakes enabled, enter the development shell (`x86_64-linux`) from the
repository root:

```sh
nix develop
```

Before building, adjust the static IP, gateway, DNS, subnet, and ntfy endpoint in
[network.go](network.go) for your environment. Notifications use plain HTTP
without authentication.

```sh
export WIFI_SSID='your-ssid'
export WIFI_PASS='your-password'

just build       # Development mode: topic "debug"
just build prod  # Production mode: topic "doorbell"
just check       # go vet and tests via nix flake check
just --list      # List all available commands
```

Both builds write `firmware.uf2` with the Wi-Fi credentials embedded. Subscribe
to the matching topic on your configured server with an
[ntfy client](https://docs.ntfy.sh/subscribe/phone/) to receive notifications.

## Flashing and logs

Connect the Pico over USB while holding **BOOTSEL**, then copy `firmware.uf2` to
the **RPI-RP2** volume. The board reboots after the copy completes.

The following helpers assume WSL with PowerShell and usbipd-win on Windows.
Adjust the drive letter (`D:`) and serial device path in [justfile](justfile)
before using them:

```sh
just flash prod   # Build and copy production firmware to D:
just auto-attach  # Attach the Pico's serial USB device to WSL
just monitor      # Read serial logs with tio
```
