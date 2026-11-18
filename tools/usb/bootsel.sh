if [[ $# -gt 1 || ${1:-} == --* ]]; then
  printf 'Usage: nix run .#bootsel -- [serial-device]\n' >&2
  exit 64
fi
select_serial "$@"
serial_access
serial_runtime

# Reuse the receiver's connection so flashing also works while monitoring.
if [[ -S "$socket" ]] && printf '\nBOOTSEL\n' | timeout 3 socat -u - "UNIX-CONNECT:$socket"; then
  printf 'Requested BOOTSEL through the running monitor.\n'
  exit 0
fi

exec {serial_fd}>"$serial_lock"
if ! flock -n "$serial_fd"; then
  printf 'The monitor is starting or its control socket is unavailable. Retry shortly.\n' >&2
  exit 1
fi
# The child shell receives the device as $1.
# shellcheck disable=SC2016
timeout 3 bash -c '
  exec 3<> "$1"
  if ! flock -n 3; then
    printf "Serial device is busy: %s\n" "$1" >&2
    exit 1
  fi
  stty -F "$1" 115200 raw -echo -hupcl
  printf "\nBOOTSEL\n" >&3
  sleep 0.2
' bash "$device"
printf 'Requested BOOTSEL on %s.\n' "$device"
