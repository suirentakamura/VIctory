# shellcheck shell=bash
# Shared by the scripts in linux/: settings, paths and the Wine environment.
# Not run by itself. Settings come from the environment or from
# ${XDG_CONFIG_HOME:-~/.config}/victory/linux.sh (see linux/config.example.sh).

set -euo pipefail

linux_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(dirname "$linux_dir")

config=${VICTORY_LINUX_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/victory/linux.sh}
# shellcheck disable=SC1090
[ -f "$config" ] && . "$config"

: "${VICTORY_HOME:=$HOME/.local/share/victory}"
: "${VICTORY_RUN_DIR:=$VICTORY_HOME/run}"
: "${VICTORY_LOG_DIR:=$VICTORY_HOME/logs}"
: "${VICTORY_DATA_DIR:=$VICTORY_HOME/data}"
: "${VICTORY_XWIN:=$HOME/.local/share/xwin-sdk}"
: "${VICTORY_TARGET_DIR:=$repo_dir/target}"
: "${VICTORY_PROFILE:=diag}"
: "${WINEPREFIX:=$VICTORY_HOME/prefix}"
export WINEPREFIX

die() { echo "${0##*/}: $*" >&2; exit 1; }

# A Linux path as Wine's Z: drive sees it.
winpath() {
  local p
  p=$(realpath -m "$1")
  printf 'Z:%s' "${p//\//\\}"
}

# The executable a build is run as: victory.exe, or victory-<name>.exe with
# VICTORY_BUILD=<name> (several builds side by side).
victory_exe() { echo "$VICTORY_RUN_DIR/victory${VICTORY_BUILD:+-$VICTORY_BUILD}.exe"; }

# The title's executable in VICTORY_GAME_DIR (VICTORY_GAME_EXE overrides).
game_exe() {
  [ -n "${VICTORY_GAME_DIR:-}" ] || die "set VICTORY_GAME_DIR to the folder with your game files"
  [ -d "$VICTORY_GAME_DIR" ] || die "VICTORY_GAME_DIR=$VICTORY_GAME_DIR is not a folder"
  if [ -n "${VICTORY_GAME_EXE:-}" ]; then echo "$VICTORY_GAME_DIR/$VICTORY_GAME_EXE"; return; fi
  local found
  found=$(find "$VICTORY_GAME_DIR/" -maxdepth 1 -iname 'game_*.exe' | sort | head -n 1)
  [ -n "$found" ] || die "no game_*.exe in $VICTORY_GAME_DIR; set VICTORY_GAME_EXE"
  echo "$found"
}

# The wineserver to use: the patched one from linux/wine/build-wineserver.sh
# when it is built, the system one otherwise or with VICTORY_WINESERVER=stock.
wineserver_bin() {
  local own=${VICTORY_WINESERVER:-$VICTORY_HOME/wine-build/server/wineserver}
  if [ "$own" != stock ] && [ -x "$own" ]; then echo "$own"; else command -v wineserver; fi
}

# Exports everything wine needs to run VIctory with vkd3d-proton and DXVK's
# DXGI, and starts the wineserver. $1: the folder for vkd3d-proton's log.
wine_env() {
  local log_dir=$1
  [ -e "$(victory_exe)" ] || die "no build at $(victory_exe); run linux/build.sh"
  [ -f "$VICTORY_RUN_DIR/d3d12.dll" ] || die "no d3d12.dll in $VICTORY_RUN_DIR; run linux/setup.sh"
  export WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=n;winemenubuilder.exe=d"
  export WINEDEBUG=${WINEDEBUG:--all,err+module}
  if [ -n "${VICTORY_GPU:-}" ]; then
    export VKD3D_FILTER_DEVICE_NAME=$VICTORY_GPU DXVK_FILTER_DEVICE_NAME=$VICTORY_GPU
  fi
  export VKD3D_LOG_FILE="$log_dir/vkd3d.log" VKD3D_DEBUG=${VKD3D_DEBUG:-warn}
  export VICTORY_CONFIG
  VICTORY_CONFIG=$(winpath "$VICTORY_RUN_DIR/victory.toml")
  export VICTORY_LOG=${VICTORY_LOG:-info}
  if [ -n "${VICTORY_DXC_DLL:-}" ]; then
    export VICTORY_DXC
    VICTORY_DXC=$(winpath "$VICTORY_DXC_DLL")
  fi
  if [ -n "${VICTORY_FIDELITYFX_DIR:-}" ]; then
    VICTORY_FIDELITYFX_DIR=$(winpath "$VICTORY_FIDELITYFX_DIR")
    export VICTORY_FIDELITYFX_DIR
  fi
  # Wine's loader starts the wineserver installed next to wine and ignores
  # $WINESERVER, so start the chosen one ourselves first.
  WINESERVER=$(wineserver_bin)
  export WINESERVER
  "$WINESERVER" -k 2>/dev/null || true
  wineserver -k 2>/dev/null || true
  sleep 1
  "$WINESERVER" -p
}

wine_stop() { "${WINESERVER:-wineserver}" -k 2>/dev/null || true; }
