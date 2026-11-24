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
