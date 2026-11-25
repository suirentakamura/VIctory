# shellcheck shell=bash
# Source this for a cargo build or test for x86_64-pc-windows-msvc on Linux:
# clang-cl compiles C, lld-link links, against the MSVC CRT and Windows SDK
# that xwin downloaded into $VICTORY_XWIN with
#   xwin --accept-license splat --use-winsysroot-style --output $VICTORY_XWIN
: "${VICTORY_XWIN:=$HOME/.local/share/xwin-sdk}"
if [ ! -d "$VICTORY_XWIN/VC/Tools/MSVC" ]; then
  echo "env.sh: no xwin SDK (winsysroot layout) in $VICTORY_XWIN" >&2
  return 1 2>/dev/null || exit 1
fi
# The flags below are split on spaces, so reach "Windows Kits/10" by a link.
[ -e "$VICTORY_XWIN/sdk10" ] || ln -s "Windows Kits/10" "$VICTORY_XWIN/sdk10"
_msvc=$(ls -d "$VICTORY_XWIN"/VC/Tools/MSVC/* | sort -V | tail -n 1)
_sdk=$VICTORY_XWIN/sdk10
_v=$(ls "$_sdk/Lib" | sort -V | tail -n 1)
export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=lld-link
export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS="-Lnative=$_msvc/lib/x64 -Lnative=$_sdk/Lib/$_v/um/x64 -Lnative=$_sdk/Lib/$_v/ucrt/x64"
export CC_x86_64_pc_windows_msvc=clang-cl CXX_x86_64_pc_windows_msvc=clang-cl
export AR_x86_64_pc_windows_msvc=llvm-lib
export CFLAGS_x86_64_pc_windows_msvc="--target=x86_64-pc-windows-msvc /imsvc$_msvc/include /imsvc$_sdk/Include/$_v/ucrt /imsvc$_sdk/Include/$_v/um /imsvc$_sdk/Include/$_v/shared"
export CXXFLAGS_x86_64_pc_windows_msvc="$CFLAGS_x86_64_pc_windows_msvc"
# Test binaries run under Wine: cargo test --target x86_64-pc-windows-msvc.
export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER=wine
unset _msvc _sdk _v
