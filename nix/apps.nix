{
  perSystem =
    {
      pkgs,
      firmwareEnv,
      firmwareTools,
      pythonLibraries,
      ...
    }:
    let
      buildFirmware = pkgs.writeShellApplication {
        name = "doorbell-build-firmware";
        runtimeInputs = firmwareTools;
        runtimeEnv = firmwareEnv;
        text = ''
          umask 077
          unset WIFI_SSID WIFI_PASS
          : "''${CARGO_TARGET_DIR:?CARGO_TARGET_DIR is required}"
          export DOORBELL_MODE="''${1:-dev}"
          cargo build --manifest-path firmware/Cargo.toml --locked --release --features device --target thumbv8m.main-none-eabihf
          picotool uf2 convert "$CARGO_TARGET_DIR/thumbv8m.main-none-eabihf/release/doorbell" -t elf "$CARGO_TARGET_DIR/firmware.uf2" --family 0xe48bff59 --platform rp2350
          arm-none-eabi-size "$CARGO_TARGET_DIR/thumbv8m.main-none-eabihf/release/doorbell"
        '';
      };
      flashScript = pkgs.writeText "flash-pico.ps1" ''
        param([Parameter(Mandatory=$true)][string]$Source)
        $ErrorActionPreference = 'Stop'
        $targets = @(Get-Volume | Where-Object { $_.FileSystemLabel -eq 'RP2350' -and $_.DriveLetter })
        if ($targets.Count -ne 1) {
          throw "Connect exactly one Pico 2 W in BOOTSEL mode (found $($targets.Count))."
        }
        Copy-Item -LiteralPath $Source -Destination ($targets[0].DriveLetter + ':\')
      '';
      runners = {
        build = pkgs.writeShellApplication {
          name = "doorbell-build";
          runtimeInputs = [ buildFirmware ];
          text = ''
            if [[ $# -gt 1 || ( "''${1:-dev}" != dev && "''${1:-dev}" != prod ) ]]; then
              printf 'Usage: nix run .#build -- [dev|prod]\n' >&2
              exit 64
            fi
            export CARGO_TARGET_DIR="$PWD/firmware/target/build"
            exec doorbell-build-firmware "$@"
          '';
        };
        flash = pkgs.writeShellApplication {
          name = "doorbell-flash";
          runtimeInputs = [
            pkgs.coreutils
            buildFirmware
          ];
          text = ''
            if [[ $# -gt 1 || ( "''${1:-dev}" != dev && "''${1:-dev}" != prod ) ]]; then
              printf 'Usage: nix run .#flash -- [dev|prod]\n' >&2
              exit 64
            fi
            command -v powershell.exe >/dev/null
            command -v wslpath >/dev/null
            export CARGO_TARGET_DIR="$PWD/firmware/target/build"
            doorbell-build-firmware "$@"
            install -m 600 "$CARGO_TARGET_DIR/firmware.uf2" firmware.uf2
            script=$(wslpath -w '${flashScript}')
            source=$(wslpath -w "$PWD/firmware.uf2")
            powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$script" -Source "$source"
          '';
        };
        auto-attach = pkgs.writeShellApplication {
          name = "doorbell-auto-attach";
          text = ''
            if [[ $# -ne 0 ]]; then
              printf 'Usage: nix run .#auto-attach\n' >&2
              exit 64
            fi
            exec usbipd.exe attach --wsl --hardware-id 2e8a:000a --auto-attach
          '';
        };
        monitor = pkgs.writeShellApplication {
          name = "doorbell-monitor";
          runtimeInputs = [
            pkgs.coreutils
            pkgs.tio
          ];
          text = ''
            if [[ $# -gt 1 ]]; then
              printf 'Usage: nix run .#monitor -- [serial-device]\n' >&2
              exit 64
            fi
            if [[ $# -eq 1 ]]; then
              device=$1
            else
              shopt -s nullglob
              ports=(/dev/serial/by-id/usb-Raspberry_Pi_Pico_*-if00)
              if [[ ''${#ports[@]} -ne 1 ]]; then
                printf 'Specify a serial device; found %s matching Pico devices.\n' "''${#ports[@]}" >&2
                exit 1
              fi
              device=''${ports[0]}
            fi
            if [[ ! -r "$device" || ! -w "$device" ]] \
              && [[ " $(id -Gn) " != *" dialout "* ]] \
              && [[ " $(id -Gn "$(id -un)") " == *" dialout "* ]]; then
              export DOORBELL_MONITOR_DEVICE="$device"
              # A Nix-store sg lacks the host's setuid wrapper needed to change groups.
              exec sg dialout -c "exec ${pkgs.lib.getExe pkgs.tio} \"\$DOORBELL_MONITOR_DEVICE\""
            fi
            exec tio "$device"
          '';
        };
        check = pkgs.writeShellApplication {
          name = "doorbell-check";
          runtimeInputs = firmwareTools ++ [
            pkgs.uv
            pkgs.ruff
          ];
          runtimeEnv = firmwareEnv;
          text = ''
            export LD_LIBRARY_PATH="${pythonLibraries}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export CARGO_TARGET_DIR="$PWD/firmware/target/verification"
            unset WIFI_SSID WIFI_PASS
            cargo fmt --manifest-path firmware/Cargo.toml --check
            cargo clippy --manifest-path firmware/Cargo.toml --locked --lib --tests --examples -- -D warnings
            cargo test --manifest-path firmware/Cargo.toml --locked --lib
            export DOORBELL_MODE=dev
            cargo clippy --manifest-path firmware/Cargo.toml --locked --features device --target thumbv8m.main-none-eabihf -- -D warnings
            ${pkgs.lib.getExe buildFirmware} dev
            ruff check firmware/tools
            requirements=$(uv export --script firmware/tools/encrypt_wifi.py --locked --format requirements-txt)
            uv run --no-project --with-requirements <(printf '%s\n' "$requirements") firmware/tools/test_encrypt_wifi.py
            cd ml
            ruff check src tests
            uv run --locked -m unittest discover -s tests -v
          '';
        };
      };
    in
    {
      _module.args = { inherit runners; };
      apps = pkgs.lib.mapAttrs (name: runner: {
        type = "app";
        program = pkgs.lib.getExe runner;
        meta.description = "Pico doorbell ${name}";
      }) runners;
    };
}
