mode=${DOORBELL_MONITOR_MODE:---graph}
serial_device=()
for argument in "$@"; do
  case "$argument" in
    --raw|--all|--graph|--logs) mode=$argument ;;
    --*) printf 'Usage: nix run .#monitor -- [--logs|--all|--raw|--graph] [serial-device]\n' >&2; exit 64 ;;
    *) serial_device+=("$argument") ;;
  esac
done
if [[ ${#serial_device[@]} -gt 1 ]]; then
  printf 'Usage: nix run .#monitor -- [--logs|--all|--raw|--graph] [serial-device]\n' >&2
  exit 64
fi
export DOORBELL_MONITOR_MODE="$mode"
select_serial "${serial_device[@]}"
serial_access
serial_runtime
exec {serial_fd}>"$serial_lock"
if ! flock -n "$serial_fd"; then
  printf 'A monitor is already using %s. Close it before opening another.\n' "$device" >&2
  exit 1
fi
# A crashed receiver can leave its socket behind; the lock proves it is unused.
unlink "$socket" 2>/dev/null || true

log_directory="$PWD/artifacts/monitor"
mkdir -p "$log_directory"
log_file=$(mktemp "$log_directory/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")
printf 'Full log: %s\n' "$log_file" >&2
tio_options=(--timestamp --timestamp-format iso8601 --socket "unix:$socket")
if [[ "$mode" == --raw ]]; then
  exec tio "${tio_options[@]}" --log --log-file "$log_file" --log-strip --log-append "$device"
fi
if [[ ! -t 0 || ! -t 1 ]]; then
  printf 'The log viewer requires a terminal. Use --raw for plain serial output.\n' >&2
  exit 1
fi

receiver_pid=
viewer_socket=
cleanup() {
  if [[ -n "$viewer_socket" ]]; then
    tmux -S "$viewer_socket" kill-server 2>/dev/null || true
    unlink "$viewer_socket" 2>/dev/null || true
  fi
  if [[ -n "$receiver_pid" ]]; then
    kill "$receiver_pid" 2>/dev/null || true
    wait "$receiver_pid" 2>/dev/null || true
  fi
  unlink "$socket" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# tio needs a terminal to keep receiving; the coprocess keeps its input separate from lnav.
printf -v receiver_command '%q ' tio "${tio_options[@]}" --color none "$device"
coproc {
  SHELL=bash exec script --quiet --return --command "exec $receiver_command" /dev/null >"$log_file" 2>&1
}
receiver_pid=$!
# Surface startup failures rather than leaving the viewer on an empty file.
connected=false
for ((attempt = 0; attempt < 50; attempt++)); do
  if ! kill -0 "$receiver_pid" 2>/dev/null; then
    cat "$log_file" >&2
    exit 1
  fi
  if [[ -S "$socket" && $(<"$log_file") == *"Connected to "* ]]; then
    connected=true
    break
  fi
  sleep 0.1
done
if [[ "$connected" != true ]]; then
  printf 'Timed out starting the serial receiver. See %s\n' "$log_file" >&2
  exit 1
fi
# Always apply filters to the log view, including when no firmware data has
# arrived yet and the file contains only the receiver's connection messages.
viewer_options=(-I "$DOORBELL_LNAV_CONFIG" -c ':switch-to-view log')
if [[ "$mode" != --all ]]; then
  viewer_options+=(-c ':filter-out ^\[[^]]+\] (Processing frontend=|Audio samples=|Scores background=|ADC rate=|Gate t=|PCM )')
fi
if [[ "$mode" == --graph ]]; then
  # Both views read the same saved log. Only tio owns the USB connection.
  # Keep the regular log stream visible; Ctrl-F enables the periodic filter.
  viewer_options+=(-c ':toggle-filtering')
  read -r terminal_rows terminal_columns < <(stty size)
  chart_height=$((terminal_rows / 3))
  ((chart_height < 10)) && chart_height=10
  ((chart_height > 16)) && chart_height=16
  log_height=$((terminal_rows - chart_height - 2))
  if ((log_height < 6 || terminal_columns < 50)); then
    printf 'The combined view needs a terminal of at least 50 columns x 18 rows. Use --logs for a smaller terminal.\n' >&2
    exit 1
  fi
  viewer_socket="$socket.tmux"
  # This private server has its own config and never changes existing tmux sessions.
  tmux -S "$viewer_socket" -f "$DOORBELL_TMUX_CONFIG" new-session -d -s doorbell \
    -x "$terminal_columns" -y "$terminal_rows" doorbell-graph "$log_file"
  tmux -S "$viewer_socket" split-window -v -l "$log_height" -t doorbell:0.0 \
    lnav "${viewer_options[@]}" "$log_file"
  tmux -S "$viewer_socket" select-pane -t doorbell:0.1
  env -u TMUX tmux -S "$viewer_socket" attach-session -t doorbell
  exit 0
fi
lnav "${viewer_options[@]}" "$log_file"
