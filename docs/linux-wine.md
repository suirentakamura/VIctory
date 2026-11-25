# Running under Wine / Proton on Linux

Status of this fork's `linux/upstream` branch, 2026-11-25. It is upstream
VIctory (`main` at 940d32f) plus fixes found while running the title under
Wine with vkd3d-proton 3.2 on an AMD RX 9060 XT (RADV). Nothing here needs
game data in the repository; supply your own files as usual. The older
`linux/shaders` and `linux/uma` branches hold earlier work on an older
upstream (the unified-memory experiment and the wineserver notes are there).

## State

| Stage | State |
|-------|-------|
| Boot, menus, loading screen | works |
| Intro cinematic | draws: world geometry, sky, volumetric clouds, sea, overlaid credits. Characters and near buildings are still unlit; cut-in credit text is missing; 1.6-2 s a frame |
| Shaders | 25,756 of 26,177 programs translate; every emitted vertex and pixel program (14,255) passes `dxil.dll` validation |

## Why the picture was black under Wine

The indirect argument clamp and the depth conversion are small compute
shaders compiled at run time by the system's `D3DCompile`. Wine's compiler
(vkd3d-shader HLSL) has no `Load` on writable byte-address buffers and
ignores `[RootSignature]`, so neither built and every indirect draw was
dropped. They now use `RWStructuredBuffer<uint>` by word, with the root
signature built in code (`crates/victory-gpu/src/helper.rs`). vkd3d-proton
also refuses pipelines that blend on an integer render target; blending is
switched off there (the hardware ignores it too).

## Shader work on this branch

- **Descriptor words.** Each heap slot's entry in the view address table holds
  the hardware's eight descriptor dwords (buffers: address, stride, records,
  swizzle; images: GFX10 size, levels, array base, type), so programs that
  read a descriptor only as data (size and level queries) translate.
- **Lane masks.** A data join with a mask edge takes the mask's ballot number;
  a mask join with an untracked edge, and `v_cndmask`/carry-in on an untracked
  mask, take the lane's bit of the number; a read SCC of a mask operation is
  "any lane".
- **Barycentrics read as data** become an `SV_Barycentrics` input.
- **`v_cvt_f32_f64`**: exact round-to-nearest-even in 32-bit integer steps.
- **`v_permlanex16_b32`** through `waveReadLaneAt`.
- **Descriptor joins** whose other edge holds only stale input registers take
  the defined edge's descriptor.

Still refused (native survey): 198 geometry-shader-only containers, 10 mask
joins of BVH addresses (ray-traced shadows), 8 masks spilled through VGPR
lanes, 8 f64 wave sums, and a handful of single cases. 750 compute programs
fail validation only because the container records validator 1.6 while using
compute derivatives (SM 6.6); vkd3d-proton does not check it.

## Variables used for this work

| Variable | Effect |
|----------|--------|
| `VICTORY_DXC=<path to dxcompiler.dll>` | compiler for the rectangle geometry stage; under Wine give a `Z:\` path |
| `VICTORY_DXIL=<path to dxil.dll>` | `translate --validate` and tests: Microsoft's validator |
| `VICTORY_SHADER_LOOP_LIMIT=<n>` | see the Debugging table in the README |

After a translator change, run `victory translate --cache --pipelines <files>`
once (the shader cache is keyed on the translator's source). Start the
patched wineserver yourself before `wine` (`$WINESERVER -p`): Wine's loader
otherwise starts its own and ignores `$WINESERVER`.

## Tests

The shader unit tests run natively. Tests that need a GPU are built for
`x86_64-pc-windows-msvc` and run under Wine with vkd3d-proton's `d3d12`,
`d3d12core` and `dxgi` DLLs beside the test binary. Two of victory-shader's
always fail there because they check host properties
(`hosts_that_do_not_fuse_are_named`, `stub_pipeline_is_created`). Eight of
victory-gpu's fail because their test shaders need what Wine's HLSL compiler
lacks (`[RootSignature]`, unbounded resource arrays) or a pipeline library
vkd3d-proton does not reject. Delete the `vkd3d-proton.cache*` files the
tests leave behind and never commit them.
