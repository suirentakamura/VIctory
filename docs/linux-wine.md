# Running under Wine on Linux (victory-suiren)

This is the state of the `main` branch of [victory-suiren](https://github.com/suirentakamura/victory-suiren)
as of 2026-11-25. victory-suiren is upstream VIctory (`main` at 940d32f) plus:

- open pull requests #332 (partial-channel image stores), #1924 (geometry
  stages) and #1065 (validator 1.8 for compute derivatives);
- the fixes and speed-ups below, found while running the title under Wine with
  vkd3d-proton on an AMD RX 9060 XT (RADV).

How to build and run it on Arch Linux:
[../linux/README.md](../linux/README.md). Nothing here needs game data in the
repository.

The older `linux/shaders` and `linux/uma` branches hold earlier work on an
older upstream, including the unified-memory experiment.

## State

| Stage | State (runs play65-play88) |
|-------|-------|
| Boot, first-boot screens, intro video, start screen | works, 30 fps |
| Loading into the world | works, about 1-2 minutes from launch |
| Intro cinematic | Drawn: buildings, lit windows and street lights, sky, volumetric clouds, sea, overlaid credits, depth of field. Characters are dark, unlit silhouettes with spiked geometry. Some distant geometry breaks into stray triangles. |
| Cinematic frame rate | 2-2.7 fps, with dips to 1 fps (it was 0.6-3 before the work below) |
| Stability | Runs play72-play88 ended at the test timeout with no device removal. 1 of 7 earlier runs crashed in the title (access violation). |
| Shaders | 25,956 programs translate (of 26,177, plus 195 geometry programs). Every emitted program passes `dxil.dll` and SPIR-V validation, and the pipeline pre-build completes. |

## Where a cinematic frame goes

Measured per frame in play85, with `VICTORY_SHADER_LOOP_LIMIT=4096`:

| Part | Time | Notes |
|------|------|-------|
| Preparing (CPU) | 180-220 ms | turning about 52 title submissions into D3D12 work; now overlaps the GPU |
| GPU | 150-250 ms | about half is one program, see below |
| Copying back | about 50 ms | GPU-written pages copied into title memory after each submission |

The frame is now bound by the GPU plus copy-back. With one submission in
flight, about 270 ms a frame (3.7 fps) is the limit until those shrink.

## Frame-time work on this branch

- **TSC clock (`victory-gpu/src/clock.rs`).** In Wine 11.22,
  `QueryPerformanceCounter` is a system call every time, with a full register
  save. The copy and write-watch timers called it hundreds of thousands of
  times a frame, about 30% of the submitting thread. The clock now reads the
  invariant TSC, calibrated once against `Instant`, and counts nanoseconds.
- **Fast hashing (`fast.rs`).** A multiply-rotate hasher replaces SipHash on
  the maps keyed by page, address and descriptor. SipHash was 22% of the
  thread.
- **Fewer checks.** Shadow updates return early when no tracked page changed
  (one check per range instead of about 300,000 a frame). Host descriptor-heap
  slots whose view is unchanged are skipped.
- **Asynchronous submission.** `ExecuteCommandLists` hands each submission to
  a completion thread. That thread waits for the GPU, copies GPU writes back
  into title memory, runs bottom-of-pipe writes and signals deferred fences,
  while the title's thread prepares the next submission.
  - Snapshots of shader-written memory are kept per submission.
  - A copy, fill or host step waits only if it touches memory the submission
    in flight writes.
  - Lists wait for their last submission before they are reset or released.
  - Blocking fence waits and present let the submission in flight finish
    first.
  - `VICTORY_SYNC_SUBMIT=1` brings back the synchronous path; tests use it.
- **Thread placement (`victory-hle/src/modules/cpus.rs`).** The title pins
  its threads with masks for the console's 7 cores (14 logical processors).
  Passed through unchanged, they kept the title on the host's first 14
  logical processors. Console core c now maps to host cores
  [c·n/7, (c+1)·n/7) with all their SMT siblings. On a 12-core, two-CCD
  Ryzen, each four-core console cluster gets one CCD, and the whole-console
  mask gets all 24 threads.
- **Patched wineserver** ([../linux/wine/](../linux/wine/)). Mapped views are
  looked up by a tree instead of a list walk, which took about 60% of a core
  in the world with stock Wine.

Steps, measured per frame:

| Run | Change | Preparing | fps |
|-----|--------|-----------|-----|
| play72-play74 | before | 280-380 ms | 1.5-2.3 |
| play76 | TSC clock | 220-270 ms | |
| play79 | fast hasher, early-outs | 160-200 ms | 2.1-2.6 |
| play81-play85 | asynchronous submission, overlap gating | 180-220 ms, overlapped | 2.0-2.7 |

## Why the picture was black under Wine

Two small compute shaders, the indirect argument clamp and the depth
conversion, were compiled at run time by the system's `D3DCompile`. Wine's
compiler (vkd3d-shader HLSL) has no `Load` on writable byte-address buffers
and ignores `[RootSignature]`. Neither shader built, so every indirect draw
was dropped. They now use `RWStructuredBuffer<uint>` by word, with the root
signature built in code (`crates/victory-gpu/src/helper.rs`).

vkd3d-proton also refuses pipelines that blend on an integer render target.
Blending is switched off there; the hardware ignores it too.

## Shader work on this branch

- **Descriptor words.** Each heap slot's entry in the view address table holds
  the hardware's eight descriptor dwords. For buffers that is address, stride,
  records and swizzle; for images, the GFX10 size, levels, array base and
  type. Programs that read a descriptor only as data (size and level queries)
  now translate.
- **Lane masks.**
  - A data join with a mask edge takes the mask's ballot number.
  - A mask join with an untracked edge takes the lane's bit of the number.
    So do `v_cndmask` and carry-in on an untracked mask.
  - The SCC read after a mask operation means "any lane".
- **Barycentrics read as data** become an `SV_Barycentrics` input.
- **`v_cvt_f32_f64`**: exact round-to-nearest-even in 32-bit integer steps.
- **`v_permlanex16_b32`** through `waveReadLaneAt`.
- **Descriptor joins.** Where the other edge holds only stale input
  registers, the join takes the defined edge's descriptor. A descriptor
  chosen by control flow may also come from an earlier join's phi.
- **SCC after an EXEC branch.** An EXEC write's SCC ("EXEC not empty") is
  known in the successors of the `s_cbranch_execnz`/`_execz` after it. The
  ped blend-shape skinning reads it after a waterfall loop.
- **UAVs at every stage.** Vertex and geometry programs that bind a UAV (the
  write-mark bitmap) declare it.
- **One handle per annotation.** A block can store floats through a texture
  view and also run an integer atomic on it. That now gets two handles.
  Before, the atomic reused the float handle and dxil-spirv emitted a float
  atomic: invalid SPIR-V in 713 programs.
- **Scalars parked in VGPR lanes.** When the console compiler runs out of
  SGPRs, it saves scalars in lanes of a spare VGPR (`v_writelane` and
  `v_readlane` with a constant lane). That VGPR is now read as extra scalar
  registers. Parked descriptors trace back to their loads, and no cross-lane
  operation is left for a wave split in two on the host.
- **One latch per loop: the pre-build hang.**
  - A GFX10 loop with several branches back to its header became a DXIL loop
    with several back edges.
  - dxil-spirv (inside vkd3d-proton) recursed without end on such a header,
    at any stack size.
  - Wine ends a thread whose stack overflows without unwinding it, so
    `translate --pipelines` lost its workers on 24 ray-tracing programs and
    hung.
  - The emitter now gives each such header one latch block, as LLVM's loop
    simplification does for DXC's loops.

Still refused, from a survey with a native dxil-spirv:

- 10 mask joins of BVH addresses (ray-traced shadows);
- 8 f64 wave sums;
- 4 ped decal pixel programs that join a stale lane mask with data;
- 3 text shaders shipped as PC DXIL only;
- 4 single cases.

## Open problems

1. **Ped skinning runs to the loop limit.** `MS_VS_PedTransformPreSkinD` is a
   mesh program, run through an indirect dispatch. Each dispatch takes 40-80
   ms, which is most of the frame's GPU time. That time halves when
   `VICTORY_SHADER_LOOP_LIMIT` drops from 4096 to 256 (play80 against
   play86). So a loop in its translation never meets its exit and stops only
   at the limit. That costs time and is a likely cause of the spiked
   characters.
2. **Runaway loops without the limit.** Since the VGPR-lane port, a few
   programs loop on the GPU for about 800 ms a frame unless
   `VICTORY_SHADER_LOOP_LIMIT` is set, which is why `play.sh` sets 4096. It is
   probably the same fault as problem 1.
3. **Unlit characters.** Characters are drawn as dark silhouettes; the
   scenery is lit.
4. **Blocky, low-resolution cut-ins.** Some shots render at a fraction of the
   resolution. The title's dynamic resolution is suspected.
5. **Occasional title crashes**: an access violation reading
   `0xffffffffffffffff`.

## Plans and next steps

In order, toward at least 10 fps in the cinematic:

1. **Fix the skinning loop** (open problems 1 and 2). Disassemble the program
   and compare its loop exit with the console code. It is the largest single
   GPU cost and probably the character geometry bug.
2. **Batch the copy-back.** About 50 ms a frame goes on copying GPU-written
   pages back one at a time; gather them into one pass per submission.
3. **More than one submission in flight.** The CPU now overlaps the GPU, but
   only by one submission. A deeper queue needs the write-mark bitmap and the
   per-copy state split per submission.
4. **Spread the preparation over threads.** About 200 ms of single-threaded
   CPU work a frame. Pages can be compared and copied in parallel. The title's
   own worker threads mostly wait on the GPU chain today, so the extra cores
   pay off once the GPU side shrinks.
5. **Lighting for characters.** Then the cut-in resolution, and the crash.

## Variables used for this work

See [../linux/README.md](../linux/README.md#variables) for the full list.

| Variable | Effect |
|----------|--------|
| `VICTORY_DXC=<path to dxcompiler.dll>` | compiler for the rectangle geometry stage; under Wine give a `Z:\` path (`linux/` scripts set it from `VICTORY_DXC_DLL`) |
| `VICTORY_DXIL=<path to dxil.dll>` | Microsoft's validator, for `translate --validate` and the tests |
| `VICTORY_SHADER_LOOP_LIMIT=<n>` | bound every translated loop |
| `VICTORY_SYNC_SUBMIT=1` | synchronous submission |

After a translator change, run `linux/translate.sh` once; the shader cache is
keyed on the translator's source.

The scripts start the patched wineserver themselves. Wine's loader otherwise
starts the one installed next to `wine` and ignores `$WINESERVER`.

## Tests

The shader unit tests run natively. Tests that need a GPU are built for
`x86_64-pc-windows-msvc` and run under Wine, with vkd3d-proton's `d3d12` and
`d3d12core` DLLs and DXVK's `dxgi` beside the test binary.

Tests expected to fail under Wine:

- **victory-shader, 2 tests**, because they check host properties:
  `hosts_that_do_not_fuse_are_named` and `stub_pipeline_is_created`.
- **victory-gpu, 8 tests.**
  - Most of them need what Wine's HLSL compiler lacks: `[RootSignature]` and
    unbounded resource arrays.
  - The rest need a pipeline library to be rejected, which vkd3d-proton does
    not do.

Delete the `vkd3d-proton.cache*` files the tests leave behind, and never
commit them.
