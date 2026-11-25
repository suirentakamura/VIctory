#!/usr/bin/env bash
# linux/play.sh [--auto] [--name <name>] [--seconds <n>] [--stock-wineserver]
#
# Plays the title with the settings the test runs use. Close the window to
# quit. Logs go to $VICTORY_LOG_DIR/<name>/ (default: play-<date>).
#   --auto               press Enter at 10 s and 11 s: through the start
#                        screens into the first mission's intro cinematic
#   --seconds <n>        stop after n seconds
#   --stock-wineserver   the system wineserver instead of the patched one
here=$(dirname "$0")
name="play-$(date +%Y%m%d-%H%M%S)" secs=0
while [ $# -gt 0 ]; do
  case "$1" in
    --auto) export VICTORY_KEYS=${VICTORY_KEYS:-10:Enter,11:Enter} ;;
    --name) name=$2; shift ;;
    --seconds) secs=$2; shift ;;
    --stock-wineserver) export VICTORY_WINESERVER=stock ;;
    -h|--help) sed -n 2,11p "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
# Bounds the loops of a few translated programs that otherwise run away on
# the GPU in the cinematic (docs/linux-wine.md, "Open problems").
export VICTORY_SHADER_LOOP_LIMIT=${VICTORY_SHADER_LOOP_LIMIT:-4096}
exec "$here/run.sh" "$name" "$secs"
