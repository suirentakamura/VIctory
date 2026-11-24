# Integration branch

This branch is VIctory `main` plus open pull requests and fixes from other branches, each one tested together, aiming for a build that boots and plays as far as possible. Every PR is merged as is, so its commits keep their authors. Credit goes to the people below; please review and merge their PRs upstream.

Maintained by @pablogonz12. Tested on Linux (RX 9070 XT, RADV, GE-Proton / vkd3d-proton master).

## Merged pull requests

| PR | Author | Title | Why |
|---|---|---|---|
| [#177](https://github.com/Marvisak/VIctory/pull/177) | @Ortomanic | Shaders: key heap handles by element type (atomic on a float-read UAV) | valid SPIR-V for atomics on float-read images (Linux) |
| [#190](https://github.com/Marvisak/VIctory/pull/190) | @LaunchDoesntCode | GPU: build the indirect-argument clamp with our own writer (no shader compiler at run time) | indirect-argument clamp without a shader compiler (Wine) |
| [#439](https://github.com/Marvisak/VIctory/pull/439) | @r4tedx | Shaders: one console wave sharing LDS runs in the console flow on 32-lane hosts | +3 world shaders |
| [#453](https://github.com/Marvisak/VIctory/pull/453) | @AntoninMichelq | Shaders: pixel barycentrics through SV_Barycentrics (values and computed interpolation) | barycentrics: +2 world pixel shaders |
| [#454](https://github.com/Marvisak/VIctory/pull/454) | @ruvenesh | Loader: opt-in VICTORY_NO_WATCHDOG hook for the title's watchdog (build 1.0.2.1) | VICTORY_NO_WATCHDOG for long world runs |
| [#457](https://github.com/Marvisak/VIctory/pull/457) | @jeremygold02 | fix(shader): accept overwritten VGPR inputs in counted indexed writes | +1 world shader |
| [#493](https://github.com/Marvisak/VIctory/pull/493) | @hophaver | GPU: bind root SRVs as one-view tables bounded to their host copy | root SRVs as bounded tables (world GPU faults, #752) |
| [#539](https://github.com/Marvisak/VIctory/pull/539) | @SAitken91 | Shaders: 64-bit image atomics | 64-bit image atomics: +4 world pixel shaders |
| [#571](https://github.com/Marvisak/VIctory/pull/571) | @SAitken91 | Shaders: LDS byte and half accesses | LDS bytes and halves: +1 world shader |
| [#573](https://github.com/Marvisak/VIctory/pull/573) | @banbanzhige | Shaders: lower ds_bpermute_b32 | ds_bpermute: +2 world shaders |
| [#579](https://github.com/Marvisak/VIctory/pull/579) | @wbkepler | GPU: VICTORY_SKIP_WAVE64 diagnostics switch for wave64 programs | VICTORY_SKIP_WAVE64 diagnostics |
| [#638](https://github.com/Marvisak/VIctory/pull/638) | @claude-GTA6 | GPU diagnostics: find what the world hang waits for, and look under a black frame | world-hang diagnostics |

## Our own fixes

| Fix | Upstream |
|---|---|
| GPU: no blending on integer render targets in host pipelines | [#806](https://github.com/Marvisak/VIctory/issues/806) |

## Measured

751 shaders dumped from a run into the world, re-translated offline: `main` 666 translate; this branch 677, all valid SPIR-V under vkd3d-proton master.
