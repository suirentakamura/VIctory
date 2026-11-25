#!/usr/bin/env bash
# linux/cmd.sh <victory args...>
#
# Any other victory.exe subcommand under the same Wine setup, e.g.
#   linux/cmd.sh config     check the paths in victory.toml
. "$(dirname "$0")/common.sh"

[ $# -ge 1 ] || { sed -n 2,6p "$0"; exit 2; }
mkdir -p "$VICTORY_LOG_DIR"
WINEDEBUG=${WINEDEBUG:--all}
wine_env "$VICTORY_LOG_DIR"
cd "$VICTORY_RUN_DIR"
rc=0
wine "$(basename "$(victory_exe)")" "$@" || rc=$?
wine_stop
exit $rc
