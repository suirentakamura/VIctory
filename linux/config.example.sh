# shellcheck shell=bash
# Settings for the scripts in linux/. Copy to ~/.config/victory/linux.sh and
# edit; every value can also be given in the environment for one command.

# Folder with your game files (the title's executable, shaders.rpf, ...).
# Keep it outside the repository.
VICTORY_GAME_DIR=$HOME/games/gta6-xbox
# The title's executable inside it, if there is more than one game_*.exe.
#VICTORY_GAME_EXE=

# Where builds are linked, logs written and VIctory keeps its cache and data
# (shader cache about 1.3 GB, pipeline library a few MB).
#VICTORY_HOME=$HOME/.local/share/victory
#WINEPREFIX=$VICTORY_HOME/prefix

# Only the GPU whose name contains this (vkd3d-proton and DXVK), when there
# is more than one, e.g. an integrated GPU next to the discrete one.
#VICTORY_GPU="RX 9060"

# The Windows SDK and MSVC libraries fetched by xwin (linux/README section 2).
#VICTORY_XWIN=$HOME/.local/share/xwin-sdk

# Build profile: diag (release speed with diagnostics, what was tested) or
# release.
#VICTORY_PROFILE=diag

# Microsoft's dxcompiler.dll (Windows build of DXC) for the 195 geometry
# programs VIctory compiles from HLSL at run time; Wine has no DXC of its own.
#VICTORY_DXC_DLL=$HOME/tools/dxc-win/bin/x64/dxcompiler.dll

# A folder with AMD's PC build of FidelityFX (amd_fidelityfx_upscaler_dx12.dll
# and its loader): the title's upscaling then runs on the host's own FSR
# instead of the console's DLL translated. Not tried with every build; see
# docs/linux-wine.md.
#VICTORY_FIDELITYFX_DIR=$HOME/tools/fidelityfx

# The patched wineserver (linux/wine/build-wineserver.sh builds it into
# $VICTORY_HOME/wine-build). "stock" uses the system wineserver.
#VICTORY_WINESERVER=stock
