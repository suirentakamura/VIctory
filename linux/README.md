# Building and running on Arch Linux

VIctory is a Windows program, so on Linux it is cross-compiled for Windows and
run under Wine. Direct3D 12 goes to Vulkan through vkd3d-proton, and DXGI
through DXVK. These steps were tested on Arch Linux with the system in the
fork's README. For what works and what doesn't, see
[../docs/linux-wine.md](../docs/linux-wine.md).

You supply the game files. Keep them outside the repository, and never commit
anything from them (see [../docs/asset-policy.md](../docs/asset-policy.md)).

## 1. Packages

```sh
sudo pacman -S --needed git base-devel flex bison rustup clang lld llvm \
    wine vulkan-radeon vulkan-icd-loader
rustup default stable        # the repository pins its own toolchain
```

Get the source:

```sh
git clone https://github.com/suirentakamura/victory-suiren.git
cd victory-suiren
```

`vulkan-radeon` (RADV) is the AMD driver. Use your GPU's Vulkan driver
instead. Tested versions: Wine 11.22, Mesa 26.3.1, clang/lld/llvm 23.1.5, and
the Rust toolchain pinned in `rust-toolchain.toml` (1.98.1, with the
`x86_64-pc-windows-msvc` target; rustup installs it on the first `cargo` run).

## 2. Windows SDK and CRT (xwin)

The build links against Microsoft's import libraries and C runtime.
[xwin](https://github.com/Jake-Shadle/xwin) downloads them; running it means
you accept Microsoft's licence.

```sh
cargo install xwin --locked
xwin --accept-license splat --use-winsysroot-style --output ~/.local/share/xwin-sdk
```

`linux/env.sh` sets up clang-cl, lld-link and llvm-lib against that folder.
Another location goes in `VICTORY_XWIN`.

## 3. vkd3d-proton and DXVK

VIctory needs vkd3d-proton 3.2 or newer. Earlier builds crash RADV on some of
the translated shaders. The tested DLLs come from
[GE-Proton](https://github.com/GloriousEggroll/proton-ge-custom/releases):

| DLL | Taken from |
|-----|------------|
| `d3d12.dll`, `d3d12core.dll` | GE-Proton11-14, `files/lib/wine/vkd3d-proton/x86_64-windows/` (vkd3d-proton `v3.1-83-gcae8353ed`) |
| `dxgi.dll` | GE-Proton10-32, `files/lib/wine/dxvk/x86_64-windows/` (DXVK 2.7.1) |

Unpack both releases anywhere. Only those three DLLs are used.

## 4. Settings

```sh
mkdir -p ~/.config/victory
cp linux/config.example.sh ~/.config/victory/linux.sh
$EDITOR ~/.config/victory/linux.sh     # at least VICTORY_GAME_DIR
```

Every setting can also be given in the environment for a single command.

## 5. The patched wineserver (recommended)

```sh
linux/wine/build-wineserver.sh
```

This clones the source of the installed Wine version and applies
[`wine/wineserver-view-index.patch`](wine/wineserver-view-index.patch). It
then builds only the wineserver, into `~/.local/share/victory/wine-build`,
where the scripts pick it up. The rest of Wine stays the system's.

The patch looks up mapped views in a tree instead of walking a list. The
title maps tens of thousands of views, and stock Wine 11.22 spends about 60%
of a core on those lookups in the world. A new Wine version needs a rebuild,
because the server must speak the client's protocol. Run it again after every
`wine` upgrade. Set `VICTORY_WINESERVER=stock` to go back to the system
wineserver.

## 6. Build, set up, pre-build shaders

```sh
linux/build.sh                 # cross-build victory.exe (diag profile) and link it into the run folder
linux/setup.sh <GE-Proton11-14>/files/lib/wine/vkd3d-proton/x86_64-windows \
               <GE-Proton10-32>/files/lib/wine/dxvk/x86_64-windows
linux/cmd.sh config            # check the paths VIctory sees
linux/translate.sh             # translate every shader and build the pipeline library
```

`translate.sh` runs once, and again after any change to the shader translator.
The first run takes about 20 minutes on 16 threads and fills about 1.3 GB of
shader cache. Later runs skip what is already there. Without it, shaders are
translated during play and their draws wait.

The `diag` profile builds an optimised binary with the `diagnostics` feature,
which provides the frame-time log, `VICTORY_KEYS` and the GPU timers. Every
test was run with it. `VICTORY_PROFILE=release` builds without diagnostics.

## 7. Play

```sh
linux/play.sh                  # play; close the window to quit
linux/play.sh --auto           # press Enter at 10 s and 11 s: on into the intro cinematic
linux/play.sh --auto --seconds 150 --name test1
```

Each run writes `log.txt`, `vkd3d.log` and a few captured frames to
`~/.local/share/victory/logs/<name>/`. The log prints a frame-time line every
second:

```
2.4 fps, 422.2 ms/frame (190.7 preparing: ...; 216.9 on the GPU; 55.3 copying back ...; 212.3 waiting for the submission before)
```

- **preparing**: CPU time spent turning the title's command buffers into
  D3D12 work.
- **on the GPU**: the GPU's own time.
- **copying back**: GPU-written memory copied into the title's memory.
- **waiting**: time the title's thread waited for the previous submission.

`linux/run.sh <name> <seconds> [args]` is the same without the defaults.
`linux/cmd.sh <subcommand>` runs any other `victory.exe` subcommand under the
same setup.

If the GPU hangs, the title is killed by the timeout or by closing the
window. A Wine process left behind is stopped with
`WINEPREFIX=~/.local/share/victory/prefix wineserver -k`.

## Variables

| Variable | Effect |
|----------|--------|
| `VICTORY_KEYS=10:Enter,11:Enter` | press keys at those seconds after launch (diag builds) |
| `VICTORY_SHADER_LOOP_LIMIT=<n>` | bound every translated loop to n iterations; `play.sh` sets 4096 |
| `VICTORY_SHADER_LOOP_REPORT=1` | diag builds: log the loops that run 64 rounds or more, their highest count seen and the lanes the limit stopped (re-translates every shader) |
| `VICTORY_SYNC_SUBMIT=1` | submit and wait for the GPU on the title's thread (the behaviour before asynchronous submission) |
| `VICTORY_GPU_TIMING=1` | GPU time per program in the log (submissions then wait for the GPU) |
| `VICTORY_D3D12_DRED=1` | D3D12 device-removed breadcrumbs |
| `VICTORY_GPU=<name>` | use only the GPU whose name contains this |
| `VICTORY_DXC_DLL=<path>` | Windows `dxcompiler.dll` for the geometry stages compiled at run time |
| `VICTORY_BUILD=<name>` | run `victory-<name>.exe` (several builds side by side) |

## Tests

```sh
. linux/env.sh
cargo test -p victory-shader                                     # natively
cargo test -p victory-gpu --target x86_64-pc-windows-msvc        # under Wine
```

GPU tests need vkd3d-proton's `d3d12.dll`, `d3d12core.dll` and DXVK's
`dxgi.dll` next to the test binary, in
`target/x86_64-pc-windows-msvc/debug/deps/`. The tests that are expected to
fail under Wine are listed in [../docs/linux-wine.md](../docs/linux-wine.md).
Delete the `vkd3d-proton.cache*` files the tests leave behind, and never
commit them.
