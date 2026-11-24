# Running under Wine / Proton on Linux

Status of this fork's `linux/shaders` branch, 2026-11-25. It is upstream VIctory
plus the open pull requests listed in the git history, plus fixes found while
running the title under Wine with vkd3d-proton 3.2 on an AMD RX 9060 XT (RADV).
Nothing here needs game data in the repository; supply your own files as usual.

## State

| Stage | State |
|-------|-------|
| Boot, menus, loading screen | works |
| Intro cinematic | runs; the picture is still mostly black (partial G-buffer, no scene lighting), about 1 fps |
| Shaders | the corpus translates 24,757 of 26,503 programs; 1,282 of 1,289 sampled newly translated programs pass `dxil.dll` validation (the 7 failures are water vertex/pixel programs with a signature-container mismatch) |

## Shader work on this branch

- **Descriptors read only as data.** Console waterfall loops that load each
  lane's image descriptor just to read its base level lower now (kind taken from
  a sibling load; the waterfall proof covers access-less loops).
- **Skipped descriptor copies.** A descriptor copied by only some lanes (the
  others skip with `s_cbranch_execz`) takes another lane's entry on the skip
  edge, so the waterfall after it lowers. The waterfall proof checks an
  access's conditions without the loop's reach first, so the BDD budget no
  longer runs out. This unlocks most of the mesh programs that were refused.
- **Implicit-derivative samples in compute.** `image_sample`, `_o` and `_b`
  outside pixel shaders take gradients from the lane's 2x2 quad
  (`quadReadLaneAt` at lanes 0, 1, 2) and emit `SampleGrad`, because the
  implicit-derivative operations are invalid DXIL outside pixel shaders.
  Cube samples are refused by name.
- **Rectangle lists.** CPU expansion, GPU expansion for indirect draws (#346)
  and a geometry stage for rectangles placed by vertex number (#1020).

## Frame rate: what made the world slow under Wine

- **The system wineserver.** Wine's loader starts the wineserver next to its
  own binary and only falls back to `$WINESERVER`, so the indexed view lookup
  (#340) was never in use: with ~117,000 mapped views and ~800 maps a second
  the stock server spent ~60 % of a CPU searching them, and every request
  waited for it. Start the patched server yourself before `wine`
  (`$WINESERVER -p`). In the opening cinematic this took preparation from
  1-3 s to ~120 ms a frame (play55 → play56).
- **Write-watch askings once per frame** (`VICTORY_WATCH_PER=submission`
  restores per submission): answers, including "not watched" and guarded
  view memory, are kept for the frame; ~1,500 askings a frame became ~550
  distinct ones.
- The title's own `VirtualAlloc` reservations are write-watched, like
  `XMemVirtualAlloc`'s.

What is left (play56, ~0.6-1 s a frame): GPU time (~330-600 ms) and reading
shader writes back (~150-220 ms).

## Unified memory experiment (`VICTORY_UMA`, off by default)

The world's frames take 1-2 s, most of it spent keeping host copies of title
memory current. `crates/victory-gpu/src/uma.rs` tries the console's model
instead (also what XCL's design describes): title memory opened as a host
heap (`OpenExistingHeapFromAddress1`) with a buffer placed over it, so the
GPU uses the title's bytes in place. Measured on Wine + vkd3d-proton 3.2 +
RADV (RX 9060 XT):

- Private committed memory opens (up to 4095 MiB, under 1 ms per 256 MiB),
  coherent both ways, read by the card at about 28 GB/s.
- Section views (the physical-page pool) do not: vkd3d-proton silently
  gives an unrelated allocation. Only `MEM_PRIVATE` memory is opened.
- Freeing memory under an open heap loses the device, so the title's
  `VirtualFree` of such memory is deferred until the heap is released.
- Shader writes in place corrupted the title's CPU data (play43), and
  shader-heavy submissions on system memory hung the card (play44): buffers
  shaders write always stay copies in card memory.
- Opening a whole 1-2 GiB run made every write-watch reset in it re-pin the
  heap at the next submission: 10 s of GPU time per frame (play46). Heaps
  now cover only the buffer's own pages.
- With buffers up to 4 MiB in place (`VICTORY_UMA_MAX_KB`), GPU time is as
  before but frames are no faster (play47): the cost is in big buffers and
  in buffers shaders write, which must stay copies. A profile shows the
  submitting thread busy only about a fifth of the time.

`VICTORY_UMA=1` turns it on; `VICTORY_UMA_LOOSE=1` also opens memory outside
buffer objects.

## Variables used for this work

| Variable | Effect |
|----------|--------|
| `VICTORY_DXC=<path to dxcompiler.dll>` | compiler for the rectangle geometry stage (#1020); under Wine give a `Z:\` path |
| `VICTORY_DXIL=<path to dxil.dll>` | (tests) validate translated shaders with Microsoft's validator |
| `VICTORY_CAPTURE_BUSY=<draws>,<n>` | (diagnostics) capture the first `n` frames with at least that many draws |
| `VICTORY_SHADER_LOOP_LIMIT=<n>` | see the Debugging table in the README |

Run under Wine from a fresh `wineserver` (`wineserver -k; wineserver -w`) or the
game can exit within a second of starting.

## Tests

The shader unit tests run natively. Tests that need a GPU are built for
`x86_64-pc-windows-msvc` and run under Wine with vkd3d-proton's `d3d12`,
`d3d12core` and `dxgi` DLLs beside the test binary; delete the
`vkd3d-proton.cache*` files they leave behind and never commit them.
