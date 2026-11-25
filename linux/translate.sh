#!/usr/bin/env bash
# linux/translate.sh [extra translate args...]
#
# Translates every shader the title can ask for into the cache and builds the
# pipeline library (victory translate --cache --pipelines), under the same
# Wine and vkd3d-proton as the runs, so the library matches the driver. Run
# it once after setup and again after a change to the shader translator.
# About 20 minutes on 16 threads the first time; seconds when nothing changed.
. "$(dirname "$0")/common.sh"

out=$VICTORY_LOG_DIR/translate${VICTORY_BUILD:+-$VICTORY_BUILD}
mkdir -p "$out"
exe=$(game_exe)
wine_env "$out"
g=$VICTORY_GAME_DIR
inputs=("$g/shaders.rpf" "$exe")
for dll in XboxBvhBuilder.dll amd_fidelityfx_upscaler_xbox.dll; do
  [ -f "$g/$dll" ] && inputs+=("$g/$dll")
done
args=()
for f in "${inputs[@]}"; do args+=("$(winpath "$f")"); done

cd "$VICTORY_RUN_DIR"
rc=0
wine "$(basename "$(victory_exe)")" translate --cache --pipelines "${args[@]}" "$@" >"$out/log.txt" 2>&1 || rc=$?
wine_stop
echo "exit=$rc" >>"$out/log.txt"
tail -n 5 "$out/log.txt"
