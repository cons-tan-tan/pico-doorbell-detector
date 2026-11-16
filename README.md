# Pico Doorbell Detector

Recognize doorbell chimes through a MAX9814 microphone on a Raspberry Pi Pico 2 W
and send Wi-Fi notifications via [ntfy](https://docs.ntfy.sh/). Detection runs on
the board using Rust and TFLite Micro; Python tools handle training on a PC.

The included model targets the two chimes recorded for this installation.
Capture, inference, and notifications have been checked on the device; accuracy
in the installed room and long-term reliability still need evaluation.

## Start developing

Use Nix with flakes enabled on `x86_64-linux`. On NixOS, enable `nix-ld` for the
Python runtime downloaded by uv. Run the commands below from the repository root:

```sh
nix develop
nix run .#check
```

The check command runs Rust and Python tests, lint checks, and a Pico firmware
build. Builds and checks work without a board or 1Password access; any included
Wi-Fi settings are already encrypted. Use `nix run .#build` for a firmware build alone.

## Run on a Pico

The supplied flashing and USB attachment commands target NixOS-WSL with Windows
PowerShell, `op.exe`, and usbipd-win available.

### Flash

Before the first flash:

- Connect the MAX9814 microphone's analog output to GPIO26 on a Pico 2 W.
- Set your static network configuration in [network.rs](firmware/src/network.rs)
  and ntfy host in [notification.rs](firmware/src/notification.rs).

Hold **BOOTSEL** while connecting the board over USB, then release it. Once the
**RP2350** volume appears in Windows, run:

```sh
nix run .#flash
```

This builds and writes the firmware, then reboots the board. Set up Wi-Fi once
using the steps below; later flashes reuse the encrypted settings and device key.
Subscribe to the `debug` topic on your configured ntfy server
to receive alerts. Use `nix run .#flash -- prod` for the `doorbell` topic.
Without Wi-Fi settings, capture and detection run with notifications disabled.

### Monitor

For the first connection, enable `wsl.usbip.enable` and add your user to `dialout`
in the NixOS system configuration, then apply it. These are system settings;
entering the development shell does not configure USB access.

After the board reboots into normal mode, share it from an **elevated Windows
terminal**. Find the Pico's bus ID, then bind that device:

```powershell
usbipd list
usbipd bind --busid <bus-id>
```

Run USB attachment in one WSL terminal:

```sh
nix run .#auto-attach
```

Read the serial logs in another WSL terminal, also at the repository root:

```sh
nix run .#monitor
```

### Set up Wi-Fi

After the first flash, type `WIFI INFO` and press Enter in the monitor. Copy the
64-digit public key it returns. Configure the item references in
[secretspec.toml](secretspec.toml), enable Windows 1Password CLI integration, and run:

```sh
nix develop
uv run --script --locked firmware/tools/encrypt_wifi.py --public-key <public-key>
```

The script retrieves credentials through SecretSpec and writes only the public
key and encrypted settings to `firmware/settings/wifi.json`. Keep this file in
Git, then flash again. Builds and subsequent flashes need no vault access.

To update credentials, rerun the script without `--public-key`; it reuses the key
in the saved file. A replacement board or a full flash erase requires a new key
and re-encryption. The [credential design](docs/design-doc/rust-firmware.md#credential-handling)
describes the protection boundary.

## Train a model

The included model is ready to build and flash. To train a new candidate, enter
`nix develop` at the repository root and run:

```sh
uv run --project ml --locked -m doorbell_ml prepare-data
uv run --project ml --locked -m doorbell_ml retrain --output artifacts/runs/my-run
```

Preparation verifies the retained training audio and downloads missing public
backgrounds. [training.toml](ml/training.toml) defines the CPU training recipe;
CUDA, phone originals, and previous models are not required. Choose a new output
directory for each run.

The candidate appears in `artifacts/runs/my-run/candidate`. Follow the
[evaluation policy](docs/design-doc/learned-doorbell-detection.md#model-selection-and-evaluation)
before replacing the deployed model. Training leaves `firmware/model/` unchanged.
For evaluation and export commands, use
`uv run --project ml --locked -m doorbell_ml --help`.

## Repository and design

| Location | Purpose |
| --- | --- |
| [firmware/](firmware/) | Rust firmware, C++ inference glue, and the deployed model bundle. |
| [ml/](ml/) | Python training and evaluation, tests, dependency lock, and training settings. |
| [data/](data/) | Prepared recordings, synthetic audio, public-source manifests, and download cache. |
| [nix/](nix/) | Shared development environment and build, flash, and monitor commands. |
| `artifacts/` | Local experiment outputs and measurements; ignored by Git. |

Design docs explain the choices and trade-offs:

- [Learned doorbell detection](docs/design-doc/learned-doorbell-detection.md):
  recognition, training data, model selection, and evidence limits.
- [Rust firmware](docs/design-doc/rust-firmware.md): hardware, runtime architecture,
  model updates, and credential handling.
