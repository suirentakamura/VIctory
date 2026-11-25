#!/usr/bin/env bash
# linux/run.sh <name> <seconds> [victory run args...]
#
# One logged run of the title: log.txt, vkd3d.log and captured frames go to
# $VICTORY_LOG_DIR/<name>/. <seconds> > 0 kills the run after that long
# (0: until the window is closed). Useful variables (see docs/linux-wine.md):
#   VICTORY_KEYS=10:Enter,11:Enter   press keys at those seconds
#   VICTORY_GPU_TIMING=1             per-program GPU times in the log
#   VICTORY_SYNC_SUBMIT=1            submit and wait on the title's thread
#   VICTORY_SHADER_LOOP_LIMIT=<n>    bound every translated loop
. "$(dirname "$0")/common.sh"

[ $# -ge 2 ] || { sed -n 2,12p "$0"; exit 2; }
name=$1 secs=$2; shift 2
out=$VICTORY_LOG_DIR/$name
mkdir -p "$out/frames"
exe=$(game_exe)
wine_env "$out"
export VICTORY_CAPTURE=${VICTORY_CAPTURE-$(winpath "$out/frames")}

cd "$VICTORY_RUN_DIR"
limit=()
[ "$secs" != 0 ] && limit=(timeout -s KILL "$secs")
start=$(date +%s)
rc=0
"${limit[@]}" wine "$(basename "$(victory_exe)")" run "$(winpath "$exe")" --execute "$@" >"$out/log.txt" 2>&1 || rc=$?
echo "exit=$rc elapsed=$(( $(date +%s) - start ))" >>"$out/log.txt"
wine_stop
tail -n 5 "$out/log.txt"
