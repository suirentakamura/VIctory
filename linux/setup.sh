#!/usr/bin/env bash
# linux/setup.sh <vkd3d-proton dir> <dxvk dir>
#
# Prepares the run folder ($VICTORY_RUN_DIR): vkd3d-proton's d3d12.dll and
# d3d12core.dll, DXVK's dxgi.dll, a victory.toml pointing at your game files
# and VIctory's cache and data folders, and the Wine prefix.
#
# Each dir is a folder holding the x86_64 Windows DLLs, e.g. from GE-Proton:
#   linux/setup.sh GE-Proton11-14/files/lib/wine/vkd3d-proton/x86_64-windows \
#                  GE-Proton10-32/files/lib/wine/dxvk/x86_64-windows
. "$(dirname "$0")/common.sh"

[ $# -eq 2 ] || { sed -n 2,11p "$0"; exit 2; }
vkd3d=$1 dxvk=$2
for f in "$vkd3d/d3d12.dll" "$vkd3d/d3d12core.dll" "$dxvk/dxgi.dll"; do
  [ -f "$f" ] || die "missing $f"
done
game_exe >/dev/null

mkdir -p "$VICTORY_RUN_DIR" "$VICTORY_LOG_DIR" "$VICTORY_DATA_DIR/cache" "$VICTORY_DATA_DIR/data"
install -m 755 "$vkd3d/d3d12.dll" "$vkd3d/d3d12core.dll" "$dxvk/dxgi.dll" "$VICTORY_RUN_DIR/"

toml=$VICTORY_RUN_DIR/victory.toml
if [ -e "$toml" ]; then
  echo "keeping $toml"
else
  cat >"$toml" <<TOML
# Written by linux/setup.sh; paths as Wine sees them.
[paths]
game_dir = '$(winpath "$VICTORY_GAME_DIR")'
cache_dir = '$(winpath "$VICTORY_DATA_DIR/cache")'
data_dir = '$(winpath "$VICTORY_DATA_DIR/data")'
test_titles = []
TOML
  echo "wrote $toml"
fi

[ -d "$WINEPREFIX" ] || { echo "creating the Wine prefix $WINEPREFIX"; WINEDEBUG=-all wineboot -i >/dev/null 2>&1; }
echo "ready: build with linux/build.sh, then linux/translate.sh once, then linux/play.sh"
