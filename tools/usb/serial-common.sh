# Shared by the monitor and BOOTSEL runners (packaged with writeShellApplication).
select_serial() {
  shopt -s nullglob
  local port ports=(/dev/serial/by-id/usb-Raspberry_Pi_Pico_*-if00)
  if [[ $# -eq 1 ]]; then
    device=$(realpath -m -- "$1")
    # Keep the same receiver and control socket when Linux renumbers ttyACM after a flash.
    for port in "${ports[@]}"; do
      if [[ "$port" -ef "$device" ]]; then
        device=$port
        break
      fi
    done
  else
    if [[ ${#ports[@]} -ne 1 ]]; then
      printf 'Specify a serial device; found %s matching Pico devices.\n' "${#ports[@]}" >&2
      return 1
    fi
    device=${ports[0]}
  fi
  if [[ ! -c "$device" ]]; then
    printf 'Serial device not found: %s\n' "$device" >&2
    return 1
  fi
}

serial_access() {
  if [[ ! -r "$device" || ! -w "$device" ]]; then
    if [[ " $(id -Gn) " != *" dialout "* ]] \
      && [[ " $(id -Gn "$(id -un)") " == *" dialout "* ]]; then
      export DOORBELL_RUNNER="$0" DOORBELL_SERIAL_DEVICE="$device"
      # Use the host's setuid wrapper, not a Nix-store sg.
      # Expand the variables in the new group session.
      # shellcheck disable=SC2016
      exec sg dialout -c 'exec "$DOORBELL_RUNNER" "$DOORBELL_SERIAL_DEVICE"'
    fi
    printf 'Cannot access %s. Enable dialout membership and reconnect your session.\n' "$device" >&2
    return 1
  fi
}

serial_runtime() {
  local key
  key=$(printf '%s\n' "$device" | sha256sum)
  runtime="${XDG_RUNTIME_DIR:-/tmp}/doorbell-$(id -u)"
  if [[ -L "$runtime" || ( -e "$runtime" && ! -O "$runtime" ) ]]; then
    printf 'Unsafe runtime directory: %s\n' "$runtime" >&2
    return 1
  fi
  umask 077
  mkdir -p "$runtime"
  chmod 700 "$runtime"
  socket="$runtime/${key:0:16}.sock"
  serial_lock="$runtime/${key:0:16}.lock"
}
