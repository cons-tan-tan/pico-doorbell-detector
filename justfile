# https://just.systems

default:
    @just -l

check:
    nix flake check

build mode="dev":
    tinygo build -target=pico-w -scheduler=tasks \
      -ldflags "-X main.wifiSSID=${WIFI_SSID:-} -X main.wifiPass=${WIFI_PASS:-} -X main.mode={{ mode }}" \
      -o firmware.uf2 .

flash mode="dev": (build mode)
    powershell.exe Copy-Item "\\\\wsl.localhost\\${WSL_DISTRO_NAME}$(pwd)/firmware.uf2" D:\\

auto-attach:
    usbipd.exe attach --wsl --hardware-id 2e8a:000a --auto-attach

monitor:
    tio /dev/serial/by-id/usb-Raspberry_Pi_Pico_e6613008e3436a34-if00
