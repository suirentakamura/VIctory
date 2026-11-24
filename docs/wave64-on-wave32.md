# Console wave64 compute on 32-lane hosts

Status: plan. Step 1 (translating for the host's wave sizes) is in.

## The problem

The console runs most of the title's compute programs in waves of 64
lanes. Per-lane code does not care how many lanes a host wave has, so most
of them translate without a wave size. A program that depends on the lane
layout declares the console's (`WaveSize(64)`, see `emit::lower`, where
`wave_size` is set):

- **console lanes**: it reads other lanes (`v_readlane`, `v_writelane`,
  DPP, `ds_swizzle`, `ds_(b)permute`, `v_permlane*`), counts with
  `v_mbcnt`, reads a lane mask as a number, or tests its own bit of a mask
  held as a number;
- **one wave sharing LDS**: the group is a single console wave and uses
  LDS, where the console compiler leaves out barriers because one wave runs
  in order.

D3D12 reports NVIDIA's wave range as 32..32, so these programs cannot run
there as they are. Since step 1 they stop at translation with the reason
(`emit::Target`) instead of being refused at pipeline creation.
`victory translate --host-lanes 32-32` counts them.

## The form

One console wave of 64 runs as two host waves of 32 lanes ("halves") of the
same thread group. The shader declares `WaveSize(32)`; console lane
= console thread % 64, half = bit 5 of it.

**Lane assignment.** The relabelling in `Lower::thread_id` assumes each
host wave holds a wave-aligned run of flat thread ids. That holds for 1-D
groups on the hosts seen so far, but not for all 3-D groups on NVIDIA. The
split form takes console threads from a groupshared counter instead: one
lane of each host wave adds 32 to it when the wave starts and shares the
result, and the wave's lanes take the next 32 console threads in lane order.

**What stays inside a half** needs nothing: DPP16 and DPP8 (rows of 16,
groups of 8), `v_permlane16`, `v_permlanex16` (the two rows of a 32-lane
half), `ds_swizzle` (groups of 32 or 4). The lowering must compute their
source lanes from the host lane (0..31), not from `wave() - 1` masks.

**What crosses halves** goes through an exchange in groupshared memory:

| operation | exchanged |
|---|---|
| a lane mask read whole (ballots: scalar compares, `s_bcnt1`, `s_ff1`, mask tests) | each half's 32-bit ballot |
| `v_mbcnt_lo` in the upper half | the lower half's ballot |
| `v_readlane`, `v_readfirstlane` (uniform lane) | the value of lane `l & 31` from the half that owns `l` |
| `v_writelane` to the other half | the value and lane |
| `ds_bpermute`, `ds_permute` | open: check whether GFX10 wave64 crosses halves for these, then 32 words per half if it does |

A mask read only lane by lane (a `v_cndmask` selector, a carry-in) needs
just the lane's own half. Exchanging those anyway is correct but slow.

## Synchronisation: the hard part

Both halves must reach every exchange together. On the console they do,
because a wave runs every instruction whatever EXEC holds. Our
structurisation turns EXEC into per-lane branches, so a half with no lane
on a path skips that path and never reaches an exchange there. Bathyal can
rely on uniform control flow per console wave (its recompiler runs every
lane through every instruction). We cannot, and need:

- **Uniform regions.** A region that contains an exchange (directly or in
  a nested region) is entered by both halves whenever either half has a lane
  for it. The branch into it is the console-wave OR of the lanes' conditions,
  which is itself an exchange, taken where both halves still are. Inside,
  the lanes' own conditions become guards (the existing guard machinery). A
  loop containing an exchange runs while any lane of the console wave
  continues.
- **The exchange itself.** For a group of one console wave (64 threads or
  fewer), use a slot per half and `GroupMemoryBarrierWithGroupSync`. That is
  legal because both halves are in the same uniform region, so every thread
  of the group reaches it. For groups of several console waves, a group
  barrier would also wait for other console waves, which may be elsewhere.
  Pairs meet instead through per-console-wave arrival counters, with
  double-buffered slots and a bounded wait that gives up rather than hanging
  the GPU.
- **One wave sharing LDS** needs barriers where the console relied on
  in-order execution. These are allowed only in uniform regions, so this
  comes after them.

## Testing

The differential tests need the console SDK's compiler, which we do not
assume. These tests are hand-assembled GFX10 programs instead (as in
`emit::tests`), run twice:

- in the native form on an adapter with 64-lane waves (AMD RDNA, such as
  the integrated GPU of Ryzen 7000 desktop CPUs), as the reference;
- in the split form on a 32-lane adapter (NVIDIA),

then compared buffer for buffer. This needs adapter selection in
`emit::host::Host`.

## Steps

1. ~~Translate for the host's wave range; cache by it; `--host-lanes`.~~
2. Adapter selection for tests and tools.
3. Measure the title's programs by reason (`--host-lanes 32-32`), to
   order steps 4 to 6.
4. In-half programs: console-lane programs whose cross-lane operations all
   stay inside a half, with no mask read whole. They need the lane
   assignment and nothing else.
5. Uniform regions and the one-wave exchange (ballots, readlane, mbcnt).
6. Multi-wave groups (counters), then one-wave LDS sharing.

## Credit

The approach of pairing two 32-lane waves through groupshared memory, and
the finding that NVIDIA forms waves from non-consecutive thread ids in some
3-D groups, come from XCL's design notes (`docs/wave64-on-wave32.md` in
github.com/ivall/xcl and its fork Bathyal). XCL is GPL-2.0-only. Nothing
here is copied from its code.
