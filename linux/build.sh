#!/usr/bin/env bash
# linux/build.sh [cargo args...]
#
# Cross-builds victory.exe for x86_64-pc-windows-msvc with clang-cl, lld-link
# and the Windows SDK from xwin, and links it into the run folder as
# victory.exe (victory-<name>.exe with VICTORY_BUILD=<name>).
# VICTORY_PROFILE=diag (default: optimised, with the diagnostics feature that
# the logs, VICTORY_KEYS and the GPU timers need) or release.
. "$(dirname "$0")/common.sh"
. "$linux_dir/env.sh"

features=()
[ "$VICTORY_PROFILE" = diag ] && features=(--features diagnostics)
cd "$repo_dir"
CARGO_TARGET_DIR=$VICTORY_TARGET_DIR cargo build --profile "$VICTORY_PROFILE" -p victory \
  "${features[@]}" --target x86_64-pc-windows-msvc "$@"

exe=$VICTORY_TARGET_DIR/x86_64-pc-windows-msvc/$VICTORY_PROFILE/victory.exe
mkdir -p "$VICTORY_RUN_DIR"
ln -sfn "$exe" "$(victory_exe)"
echo "$(victory_exe) -> $exe"
