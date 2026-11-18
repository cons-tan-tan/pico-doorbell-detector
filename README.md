# Pico Doorbell Detector

A Raspberry Pi Pico 2 W listens for doorbell chimes through a MAX9814 microphone
and sends notifications over Wi-Fi using [ntfy](https://docs.ntfy.sh/). Its Rust
firmware runs a TFLite Micro model trained for this installation's two chimes.
Accuracy in the installed room still needs evaluation.

## Start developing

Use Nix flakes on `x86_64-linux` and run commands from the repository root.
On NixOS, enable `nix-ld` for uv's Python runtime.

```sh
nix develop
nix run .#check
```

`check` runs hardware-independent tests, lint checks, and a firmware build.
Use `nix run .#build` for the build alone.

## Run on a Pico

The USB commands use NixOS-WSL, Windows PowerShell, and usbipd-win.
Connect the MAX9814 analog output to GPIO26. For notifications, configure the
static IP, gateway, DNS, and ntfy host for your network. [Network settings](firmware/src/network.rs) · [ntfy host](firmware/src/notification.rs)

### Flash

For the first flash, hold **BOOTSEL** while connecting USB, then release it.
Run this command in WSL once the **RP2350** drive appears in Windows.

```sh
nix run .#flash
```

Later flashes enter BOOTSEL automatically, even with the monitor open. Repeat
the button procedure if the firmware stops responding. With the Pico attached
to WSL, `nix run .#bootsel` enters the bootloader on demand.

### Monitor

Enable `wsl.usbip.enable`, add your user to `dialout`, and apply the NixOS
configuration. After the first flash, open **PowerShell as administrator** and
bind the Pico once, using its normal-mode bus ID in place of `BUS_ID`.

```powershell
usbipd list
usbipd bind --busid BUS_ID
```

Leave `nix run .#auto-attach` running in one WSL terminal so USB reconnects
after flashing. In a second WSL terminal, start the monitor.

```sh
nix run .#monitor
```

The top pane shows the last 60 seconds of `background`, `entrance`, and `door`
scores as percentages. A yellow line marked `N` shows when the Pico sent a
notification request, including retries. In the log pane below, `Ctrl+F` toggles
periodic metrics, `/` searches, and `q` quits. To reopen a saved log in the
development shell, run `lnav -I tools/monitor/lnav artifacts/monitor/FILE.log`.

### Set up Wi-Fi

Detection works offline. For notifications, close the monitor and run
`nix run .#monitor -- --raw`. Enter `WIFI INFO`, copy the 64-digit hex public
key, then press `Ctrl+T` followed by `q` to exit.

Enable Windows 1Password CLI integration (`op.exe`) and configure the item
references in [secretspec.toml](secretspec.toml). In the development shell,
substitute the Pico's public key and flash the encrypted settings.

```sh
uv run --script --locked firmware/tools/encrypt_wifi.py --public-key YOUR_PUBLIC_KEY
nix run .#flash
```

Keep the encrypted `firmware/settings/wifi.json` in Git. To update credentials
on the same board, omit `--public-key`. Repeat setup after replacing the board
or fully erasing flash. [Credential handling](docs/design-doc/rust-firmware.md#credential-handling)

Subscribe to the `debug` topic on your ntfy server. Use `nix run .#flash -- prod`
for the `doorbell` topic.

## Train a model

To train a new candidate, run these commands in the development shell with a
fresh output directory.

```sh
uv run --project ml --locked -m doorbell_ml prepare-data
uv run --project ml --locked -m doorbell_ml retrain --output artifacts/runs/my-run
```

Evaluate `artifacts/runs/my-run/candidate` before replacing the deployed model.
[Training recipe](ml/training.toml) · [Evaluation criteria](docs/design-doc/learned-doorbell-detection.md#model-selection-and-evaluation)
