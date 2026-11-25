#!/usr/bin/env bash
# linux/wine/build-wineserver.sh [wine source dir]
#
# Builds a wineserver with linux/wine/wineserver-view-index.patch for the
# installed Wine, into $VICTORY_HOME/wine-build/server/wineserver, where the
# other scripts find it. Only the wineserver is replaced; the rest of Wine
# stays the system's.
#
# The patch keeps each process's mapped views in a tree by address. Stock
# wineserver looks a view up by walking a list; the title maps tens of
# thousands of views and asks about them constantly (write watches, queries),
# which kept the wineserver at about 60% of a core and stalled every caller.
#
# Without an argument it clones the tag of the installed version (wine
# --version) from gitlab.winehq.org. The wineserver must match the client's
# protocol, so rebuild after every Wine update.
. "$(dirname "$0")/../common.sh"

ver=$(wine --version | sed 's/^wine-//; s/ .*//')
src=${1:-$VICTORY_HOME/wine-src}
build=$VICTORY_HOME/wine-build
patch=$linux_dir/wine/wineserver-view-index.patch

if [ ! -d "$src" ]; then
  git clone --depth 1 --branch "wine-$ver" https://gitlab.winehq.org/wine/wine.git "$src"
fi
if git -C "$src" apply --reverse --check "$patch" 2>/dev/null; then
  echo "patch already applied in $src"
else
  git -C "$src" apply "$patch"
fi
mkdir -p "$build"
cd "$build"
[ -f Makefile ] || "$src/configure" --enable-win64 --without-mingw --without-x \
  --without-freetype --without-wayland --disable-tests >configure.log
make -j"$(nproc)" server/wineserver
echo "built $build/server/wineserver for wine $ver"
