//! One-to-one SSA form of a native program (decision 0003, step 1).
//!
//! Every machine instruction becomes one [`Inst`], except scheduling-only
//! ones (`s_nop`, `s_waitcnt*`, `s_inst_prefetch`...), which are dropped.
//! Each register dword is a variable: a read names the [`Value`] it sees, a
//! write defines a new one, and [`Phi`]s merge values where control flow
//! joins. Nothing is raised: an instruction keeps its opcode and decoded
//! fields, so its exact semantics stay with it.
//!
//! Implicit operands are explicit: SCC, M0, and EXEC, which every vector
//! instruction reads. A write to a VGPR also reads the register's old value,
//! because lanes disabled in EXEC keep it.
//!
//! M0-relative VGPR access (`v_movrels/d/sd_b32`, `v_swaprel_b32`) names its
//! operand's whole possible range: from the encoded register up to the
//! program's VGPR count, all read (and all written, for a destination).

use crate::ShaderError;
use crate::gfx10::{self, Fields, Flow, Instruction, Operand};

/// One register dword. Scalar codes as in [`Operand::Scalar`]: 0-105 SGPRs,
/// 106/107 VCC, 108-123 trap temporaries, 124 M0, 126/127 EXEC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reg {
    S(u8),
    Scc,
    V(u8),
}

pub const VCC: u8 = 106;
pub const M0: u8 = 124;
pub const EXEC: u8 = 126;

/// An SSA value: the content of one register dword (SCC: one bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Value(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub wave64: bool,
    /// VGPRs v0 up to this exist; bounds M0-relative access.
    pub vgprs: u16,
    /// In code order, then copies made to give every loop one entry; block 0
    /// is the entry.
    pub blocks: Vec<Block>,
    /// Registers read before any write, with their value on entry: what the
    /// hardware or a called fetch routine set up (user SGPRs, system values,
    /// vertex inputs).
    pub inputs: Vec<(Reg, Value)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// Byte offset of the block's first machine instruction (shared by a
    /// block and its copies).
    pub offset: usize,
    pub preds: Vec<usize>,
    pub phis: Vec<Phi>,
    pub insts: Vec<Inst>,
    pub exit: Exit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    Jump(usize),
    /// `s_cbranch_*`, the block's last instruction.
    Branch {
        taken: usize,
        not_taken: usize,
    },
    /// `s_endpgm` and its variants.
    End,
    /// `s_setpc_b64` to an address held in registers, e.g. the handover to
    /// the next half of a merged hardware stage.
    Indirect,
}

impl Exit {
    pub fn successors(&self) -> Vec<usize> {
        match *self {
            Exit::Jump(b) => vec![b],
            Exit::Branch { taken, not_taken } => vec![taken, not_taken],
            Exit::End | Exit::Indirect => vec![],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phi {
    pub reg: Reg,
    pub value: Value,
    /// One per predecessor, in [`Block::preds`] order.
    pub args: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inst {
    pub machine: Instruction,
    pub fields: Fields,
    /// Every register dword read, with the value it holds.
    pub reads: Vec<(Reg, Value)>,
    /// Every register dword written, with the value it receives.
    pub writes: Vec<(Reg, Value)>,
}

impl Inst {
    pub fn name(&self) -> &'static str {
        self.machine.name().unwrap_or("")
    }

    /// The value `reg` holds when this instruction reads it.
    pub fn read(&self, reg: Reg) -> Option<Value> {
        self.reads.iter().find(|r| r.0 == reg).map(|r| r.1)
    }

    /// The value this instruction writes to `reg`.
    pub fn write(&self, reg: Reg) -> Option<Value> {
        self.writes.iter().find(|w| w.0 == reg).map(|w| w.1)
    }
}

/// The first user SGPR that the spill table of root arguments stands for.
const SPILL_FIRST: u8 = 16;

/// Rewrites loads from the spill table (`s[0:1]`, the high dword masked to a
/// 40-bit address, usually through copies in `s2`/`s3`) into `s_mov_b32` from
/// the user SGPRs the table's dwords stand for. Only loads with a constant
/// offset, in code no branch joins after the pointer is set up, are rewritten.
///
/// The table holds the arguments as they were on entry, but the program may
/// have written the SGPRs those dwords stand for by the time it loads them
/// (a load into `s[8:23]` overwrites what stands for table dwords 0..7). So
/// the program first copies the user SGPRs it reads into a block of SGPRs it
/// never uses, and the loads read that block. Without such a block, or when a
/// branch can return to the entry, nothing is rewritten.
fn spill_loads(
    list: Vec<Instruction>,
    wave64: bool,
    vgprs: u16,
    code_len: usize,
) -> Result<Vec<Instruction>, ShaderError> {
    use gfx10::Fields::{Smem, Sop, Valu};
    let mut targets = std::collections::BTreeSet::new();
    for inst in list.iter() {
        let flow = gfx10::control(inst, code_len, |at| {
            list[..list.partition_point(|i| i.offset < at)].last()
        })?;
        if let Flow::Jump(t) | Flow::Branch(t) = flow {
            targets.insert(t);
        }
    }
    let Some(entry) = list.first().map(|i| i.offset) else {
        return Ok(list);
    };
    if targets.contains(&entry) {
        return Ok(list);
    }
    // Every SGPR the program reads or writes, to find a block it never uses.
    let mut used = [false; 128];
    for inst in list.iter() {
        let Some(name) = inst.name() else {
            return Ok(list);
        };
        let Ok(effects) = effects(
            inst,
            &inst.fields(wave64),
            name,
            1 + u8::from(wave64),
            vgprs,
        ) else {
            return Ok(list);
        };
        for reg in effects
            .into_iter()
            .flat_map(|(r, w)| r.into_iter().chain(w))
        {
            if let Reg::S(c) = reg {
                used[usize::from(c).min(127)] = true;
            }
        }
    }
    let original = list.clone();
    // Where each rewritten copy is in `out`, and the user SGPR it stands for.
    let mut copies: Vec<(usize, u8)> = Vec::new();
    // 1: a copy of s0, 2: a copy of s1 (masked to the address's high bits).
    let mut tag = [0u8; 128];
    tag[0] = 1;
    tag[1] = 2;
    let mut out = Vec::with_capacity(list.len());
    let scalar = |o: &Operand| match *o {
        Operand::Scalar { code, dwords } if code < 106 => Some((code, dwords.max(1))),
        _ => None,
    };
    for inst in list {
        if targets.contains(&inst.offset) {
            tag = [0; 128];
        }
        let f = inst.fields(wave64);
        let name = inst.name().unwrap_or("");
        // The load of the table.
        if let Smem {
            data: Some(ref d),
            ref base,
            offset,
            ref soffset,
            ..
        } = f
            && name.starts_with("s_load_dword")
            && let Some((a, 2)) = scalar(base)
            && tag[a as usize] == 1
            && tag[a as usize + 1] == 2
            && offset >= 0
            && offset % 4 == 0
            && matches!(*soffset, Operand::Scalar { code: 125, .. })
            && let Some((dst, n)) = scalar(d)
        {
            let first = u32::from(SPILL_FIRST) + (offset as u32) / 4;
            if first + u32::from(n) <= 105 {
                for k in 0..n {
                    let src = (first + u32::from(k)) as u8;
                    copies.push((out.len(), src));
                    out.push(s_mov(dst + k, src, inst.offset)?);
                }
                for k in 0..n {
                    tag[(dst + k) as usize] = 0;
                }
                continue;
            }
        }
        // What this instruction does to the tracked registers.
        let mut written: Vec<(u8, u8)> = Vec::new();
        match &f {
            Sop {
                dst: Some(d), src, ..
            } => {
                if let Some((dc, dn)) = scalar(d) {
                    let new = match (name, src.as_slice()) {
                        ("s_mov_b32", [x]) => match scalar(x) {
                            Some((c, _)) if tag[c as usize] != 0 => tag[c as usize],
                            _ => 0,
                        },
                        ("s_and_b32", [x, y]) => {
                            // `s1 & mask` with a small mask, either way round.
                            // The high dword's eight low bits are address bits
                            // (40-bit addresses): a mask clearing any of them
                            // can point elsewhere, so it drops the tag. No
                            // inline constant (1..=64) keeps them all.
                            let masked = |reg: &Operand, mask: &Operand| match (scalar(reg), mask) {
                                (Some((c, _)), Operand::Literal(m)) => {
                                    tag[c as usize] == 2 && *m <= 0xFFFF && *m & 0xFF == 0xFF
                                }
                                _ => false,
                            };
                            if masked(x, y) || masked(y, x) { 2 } else { 0 }
                        }
                        _ => 0,
                    };
                    if dn == 1 {
                        tag[dc as usize] = new;
                    } else {
                        written.push((dc, dn));
                    }
                }
            }
            Smem { data: Some(d), .. } => written.extend(scalar(d)),
            Valu(v) => {
                written.extend(v.dst.as_ref().and_then(scalar));
                written.extend(v.carry.as_ref().and_then(scalar));
            }
            _ => {}
        }
        for (c, n) in written {
            for k in c..c.saturating_add(n).min(128) {
                tag[k as usize] = 0;
            }
        }
        out.push(inst);
    }
    let Some(last) = copies.iter().map(|c| c.1).max() else {
        return Ok(out);
    };
    // A block of SGPRs past the user SGPRs the table stands for that the
    // program never touches, below VCC.
    let size = usize::from(last - SPILL_FIRST) + 1;
    let Some(block) = (usize::from(last) + 1..=usize::from(VCC) - size)
        .find(|&b| used[b..b + size].iter().all(|u| !u))
    else {
        return Ok(original);
    };
    let moved = |src: u8| (block + usize::from(src - SPILL_FIRST)) as u8;
    let mut read: Vec<u8> = copies.iter().map(|c| c.1).collect();
    read.sort_unstable();
    read.dedup();
    for &(at, src) in &copies {
        let Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            ..
        } = out[at].fields(wave64)
        else {
            unreachable!("a copy is an s_mov_b32");
        };
        out[at] = s_mov(code, moved(src), out[at].offset)?;
    }
    let mut entries = Vec::with_capacity(read.len() + out.len());
    for &src in &read {
        entries.push(s_mov(moved(src), src, entry)?);
    }
    entries.extend(out);
    Ok(entries)
}

/// `s_mov_b32 s<dst>, s<src>`, placed at `offset`.
fn s_mov(dst: u8, src: u8, offset: usize) -> Result<Instruction, ShaderError> {
    let word = 0xBE80_0000 | u32::from(dst) << 16 | 3 << 8 | u32::from(src);
    let mut bytes = word.to_le_bytes().to_vec();
    bytes.extend(0xBF81_0000u32.to_le_bytes()); // s_endpgm
    let mut one = gfx10::decode(&bytes)?.remove(0);
    one.offset = offset;
    Ok(one)
}

/// Decodes `code` and builds its SSA form; `vgprs` is the program's VGPR
/// count. Fails on instructions the IR cannot express yet
/// ([`ShaderError::Unsupported`]).
pub fn build(code: &[u8], wave64: bool, vgprs: u16) -> Result<Program, ShaderError> {
    build_with(code, wave64, vgprs, false)
}

/// [`build`]; with `spilled`, loads from the root-argument spill table that
/// `s[0:1]` points to read the user SGPRs the table stands for (table dword
/// `k` is user SGPR `16 + k`), as plain register copies.
pub fn build_with(
    code: &[u8],
    wave64: bool,
    vgprs: u16,
    spilled: bool,
) -> Result<Program, ShaderError> {
    let mut list = gfx10::decode(code)?;
    if spilled {
        list = spill_loads(list, wave64, vgprs, code.len())?;
    }
    let lanes = 1 + u8::from(wave64);
    let isa = |offset, reason| ShaderError::Isa { offset, reason };

    let mut flows = Vec::with_capacity(list.len());
    let mut leaders = vec![0];
    for (i, inst) in list.iter().enumerate() {
        let flow = gfx10::control(inst, code.len(), |at| {
            list[..list.partition_point(|i| i.offset < at)].last()
        })?;
        if let Flow::Jump(t) | Flow::Branch(t) = flow {
            leaders.push(t);
        }
        if flow != Flow::Next
            && let Some(next) = list.get(i + 1)
        {
            leaders.push(next.offset);
        }
        flows.push(flow);
    }
    leaders.sort_unstable();
    leaders.dedup();
    let block_at = |offset: usize| {
        leaders
            .binary_search(&offset)
            .map_err(|_| isa(offset, "jump into a block"))
    };

    let mut blocks: Vec<Block> = leaders
        .iter()
        .map(|&offset| Block {
            offset,
            preds: Vec::new(),
            phis: Vec::new(),
            insts: Vec::new(),
            exit: Exit::End,
        })
        .collect();
    let next_offsets: Vec<Option<usize>> = (0..list.len())
        .map(|i| list.get(i + 1).map(|n| n.offset))
        .collect();
    for ((machine, flow), next) in list.into_iter().zip(flows).zip(next_offsets) {
        let b = leaders.partition_point(|&l| l <= machine.offset) - 1;
        let name = machine.name().ok_or(ShaderError::Unsupported {
            offset: machine.offset,
            what: "unknown opcode",
        })?;
        let fields = machine.fields(wave64);
        let fallthrough = || block_at(next.ok_or(isa(machine.offset, "falls off the code"))?);
        let ends = flow != Flow::Next || next.is_none_or(|n| leaders.binary_search(&n).is_ok());
        if ends {
            blocks[b].exit = match flow {
                Flow::Next => Exit::Jump(fallthrough()?),
                Flow::Jump(t) => Exit::Jump(block_at(t)?),
                Flow::Branch(t) => Exit::Branch {
                    taken: block_at(t)?,
                    not_taken: fallthrough()?,
                },
                Flow::Stop if name == "s_setpc_b64" => Exit::Indirect,
                Flow::Stop => Exit::End,
            };
        }
        if let Some((reads, writes)) = effects(&machine, &fields, name, lanes, vgprs)? {
            let unset = |regs: Vec<Reg>| regs.into_iter().map(|r| (r, NONE)).collect();
            blocks[b].insts.push(Inst {
                machine,
                fields,
                reads: unset(reads),
                writes: unset(writes),
            });
        }
    }
    link(&mut blocks);
    if !blocks[0].preds.is_empty() {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "a branch to the program entry",
        });
    }
    split_irreducible(&mut blocks)?;

    let order = reverse_postorder(&blocks);
    let (live_in, live_out) = liveness(&blocks, &order);
    let mut values = 0u32;
    let mut new_value = || {
        values += 1;
        Value(values - 1)
    };
    let mut inputs = Vec::new();
    let mut outs: Vec<Vec<(Reg, Value)>> = vec![Vec::new(); blocks.len()];
    let mut current = [NONE; REGS];
    for &b in &order {
        current.fill(NONE);
        let block = &mut blocks[b];
        for reg in members(&live_in[b]) {
            current[index(reg)] = match block.preds[..] {
                [] => {
                    let v = new_value();
                    inputs.push((reg, v));
                    v
                }
                // A single predecessor comes earlier in reverse postorder.
                [p] => lookup(&outs[p], reg),
                _ => {
                    let value = new_value();
                    block.phis.push(Phi {
                        reg,
                        value,
                        args: Vec::new(),
                    });
                    value
                }
            };
        }
        for inst in &mut block.insts {
            for (reg, v) in &mut inst.reads {
                *v = current[index(*reg)];
                debug_assert_ne!(*v, NONE, "{reg:?} read at {:#x}", inst.machine.offset);
            }
            for (reg, v) in &mut inst.writes {
                *v = new_value();
                current[index(*reg)] = *v;
            }
        }
        outs[b] = members(&live_out[b])
            .map(|reg| (reg, current[index(reg)]))
            .collect();
    }
    for block in &mut blocks {
        for phi in &mut block.phis {
            phi.args = block
                .preds
                .iter()
                .map(|&p| lookup(&outs[p], phi.reg))
                .collect();
            debug_assert!(!phi.args.contains(&NONE), "{phi:?}");
        }
    }
    remove_trivial_phis(&mut blocks, values);
    Ok(Program {
        wave64,
        vgprs,
        blocks,
        inputs,
    })
}

/// Placeholder before renaming.
const NONE: Value = Value(u32::MAX);
/// SGPR codes 0-127, SCC, VGPRs 0-255.
const REGS: usize = 128 + 1 + 256;
type RegSet = [u64; REGS.div_ceil(64)];

fn index(reg: Reg) -> usize {
    match reg {
        Reg::S(c) => usize::from(c),
        Reg::Scc => 128,
        Reg::V(r) => 129 + usize::from(r),
    }
}

fn reg_at(i: usize) -> Reg {
    match i {
        0..128 => Reg::S(i as u8),
        128 => Reg::Scc,
        _ => Reg::V((i - 129) as u8),
    }
}

fn insert(set: &mut RegSet, reg: Reg) {
    let i = index(reg);
    set[i / 64] |= 1 << (i % 64);
}

fn contains(set: &RegSet, reg: Reg) -> bool {
    let i = index(reg);
    set[i / 64] >> (i % 64) & 1 != 0
}

/// In [`Reg`] order, which is also index order.
fn members(set: &RegSet) -> impl Iterator<Item = Reg> + '_ {
    (0..REGS)
        .filter(|&i| set[i / 64] >> (i % 64) & 1 != 0)
        .map(reg_at)
}

/// `outs` is sorted by register, and holds every register live out.
fn lookup(outs: &[(Reg, Value)], reg: Reg) -> Value {
    outs.binary_search_by_key(&reg, |&(r, _)| r)
        .map_or(NONE, |i| outs[i].1)
}

pub(crate) fn reverse_postorder(blocks: &[Block]) -> Vec<usize> {
    let mut seen = vec![false; blocks.len()];
    let mut post = Vec::with_capacity(blocks.len());
    let mut stack = vec![(0, 0)];
    seen[0] = true;
    while let Some(top) = stack.last_mut() {
        let (b, i) = *top;
        top.1 += 1;
        match blocks[b].exit.successors().get(i) {
            Some(&s) if !seen[s] => {
                seen[s] = true;
                stack.push((s, 0));
            }
            Some(_) => {}
            None => {
                post.push(b);
                stack.pop();
            }
        }
    }
    post.reverse();
    post
}

/// Recomputes every block's predecessors from the exits.
fn link(blocks: &mut [Block]) {
    for block in blocks.iter_mut() {
        block.preds.clear();
    }
    for b in 0..blocks.len() {
        for s in blocks[b].exit.successors() {
            blocks[s].preds.push(b);
        }
    }
}

/// An edge to an earlier block in reverse postorder whose target does not
/// dominate its source, if any: there the control flow is irreducible.
/// `index` is each block's position in `order`.
pub(crate) fn irreducible_edge(
    blocks: &[Block],
    order: &[usize],
    index: &[usize],
) -> Option<(usize, usize)> {
    let &entry = order.first()?;
    let idom = idoms(blocks, order, index);
    for &b in order {
        for s in blocks[b].exit.successors() {
            if index[s] > index[b] {
                continue;
            }
            let mut d = b;
            while d != s && d != entry {
                d = idom[d];
            }
            if d != s {
                return Some((b, s));
            }
        }
    }
    None
}

/// Each block's immediate dominator (the entry's is itself, unreachable
/// blocks' `usize::MAX`); `order` is reverse postorder, `index` each
/// block's position in it.
pub(crate) fn idoms(blocks: &[Block], order: &[usize], index: &[usize]) -> Vec<usize> {
    let mut idom = vec![usize::MAX; blocks.len()];
    let Some(&entry) = order.first() else {
        return idom;
    };
    idom[entry] = entry;
    let intersect = |idom: &[usize], mut a: usize, mut b: usize| {
        while a != b {
            while index[a] > index[b] {
                a = idom[a];
            }
            while index[b] > index[a] {
                b = idom[b];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &order[1..] {
            let mut new = usize::MAX;
            for &q in blocks[b].preds.iter().filter(|&&q| idom[q] != usize::MAX) {
                new = if new == usize::MAX {
                    q
                } else {
                    intersect(&idom, q, new)
                };
            }
            if idom[b] != new {
                idom[b] = new;
                changed = true;
            }
        }
    }
    idom
}

/// Node splitting, before SSA. While a loop is entered at several blocks
/// (a compiler-rotated loop whose peeled first test jumps into its body),
/// the entry first in reverse postorder stays the header, and another
/// entry's outside predecessors move to a copy of everything that entry
/// reaches in the loop before the header. The copies run the same
/// instructions, so behaviour is unchanged.
fn split_irreducible(blocks: &mut Vec<Block>) -> Result<(), ShaderError> {
    let limit = 4 * blocks.len() + 64;
    loop {
        let order = reverse_postorder(blocks);
        let mut index = vec![usize::MAX; blocks.len()];
        for (i, &b) in order.iter().enumerate() {
            index[b] = i;
        }
        let Some((from, _)) = irreducible_edge(blocks, &order, &index) else {
            return Ok(());
        };
        let found = multi_entry_loop(blocks, &vec![true; blocks.len()]);
        let Some((region, mut entries)) = found.filter(|_| blocks.len() < limit) else {
            return Err(ShaderError::Unsupported {
                offset: blocks[from].offset,
                what: "irreducible control flow",
            });
        };
        entries.sort_by_key(|&e| index[e]);
        let (header, entry) = (entries[0], entries[1]);

        // What `entry` reaches in the loop before the header, and its copies.
        let mut copy = vec![usize::MAX; blocks.len()];
        let mut copied = Vec::new();
        let mut stack = vec![entry];
        while let Some(b) = stack.pop() {
            if copy[b] != usize::MAX {
                continue;
            }
            copy[b] = blocks.len() + copied.len();
            copied.push(b);
            for s in blocks[b].exit.successors() {
                if region[s] && s != header {
                    stack.push(s);
                }
            }
        }
        let retarget = |exit: &mut Exit, to: &dyn Fn(usize) -> usize| match exit {
            Exit::Jump(s) => *s = to(*s),
            Exit::Branch { taken, not_taken } => {
                *taken = to(*taken);
                *not_taken = to(*not_taken);
            }
            Exit::End | Exit::Indirect => {}
        };
        for &b in &copied {
            let mut block = blocks[b].clone();
            retarget(&mut block.exit, &|s| {
                if copy[s] == usize::MAX { s } else { copy[s] }
            });
            blocks.push(block);
        }
        let outside: Vec<usize> = blocks[entry]
            .preds
            .iter()
            .copied()
            .filter(|&p| !region[p])
            .collect();
        for p in outside {
            retarget(&mut blocks[p].exit, &|s| {
                if s == entry { copy[entry] } else { s }
            });
        }
        link(blocks);
    }
}

/// A loop among the `inside` blocks entered at more than one block, with
/// its entries. Loops nested in a single-entry loop are searched with its
/// header removed, so one is found whenever the control flow is irreducible.
fn multi_entry_loop(blocks: &[Block], inside: &[bool]) -> Option<(Vec<bool>, Vec<usize>)> {
    let reach = |from: usize, forward: bool| {
        let mut seen = vec![false; blocks.len()];
        let mut stack = vec![from];
        while let Some(b) = stack.pop() {
            let next = if forward {
                blocks[b].exit.successors()
            } else {
                blocks[b].preds.clone()
            };
            for n in next {
                if inside[n] && !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        seen
    };
    let mut done = vec![false; blocks.len()];
    for v in (0..blocks.len()).filter(|&v| inside[v]) {
        if done[v] {
            continue;
        }
        // Blocks on a cycle through `v`; `v` itself only if there is one.
        let (ahead, behind) = (reach(v, true), reach(v, false));
        let region: Vec<bool> = ahead.iter().zip(&behind).map(|(a, b)| a & b).collect();
        if !region[v] {
            continue;
        }
        let members = (0..blocks.len()).filter(|&b| region[b]);
        for b in members.clone() {
            done[b] = true;
        }
        let entries: Vec<usize> = members
            .filter(|&b| blocks[b].preds.iter().any(|&p| !region[p]))
            .collect();
        match entries[..] {
            [] => {}
            [header] => {
                let mut nested = region;
                nested[header] = false;
                if let Some(found) = multi_entry_loop(blocks, &nested) {
                    return Some(found);
                }
            }
            _ => return Some((region, entries)),
        }
    }
    None
}

/// Registers live into and out of each block.
fn liveness(blocks: &[Block], order: &[usize]) -> (Vec<RegSet>, Vec<RegSet>) {
    let empty = [0; REGS.div_ceil(64)];
    let mut uses = vec![empty; blocks.len()];
    let mut defs = vec![empty; blocks.len()];
    for (b, block) in blocks.iter().enumerate() {
        for inst in &block.insts {
            for &(reg, _) in &inst.reads {
                if !contains(&defs[b], reg) {
                    insert(&mut uses[b], reg);
                }
            }
            for &(reg, _) in &inst.writes {
                insert(&mut defs[b], reg);
            }
        }
    }
    let mut live_in = vec![empty; blocks.len()];
    let mut live_out = vec![empty; blocks.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().rev() {
            let mut out = empty;
            for s in blocks[b].exit.successors() {
                for (o, i) in out.iter_mut().zip(&live_in[s]) {
                    *o |= i;
                }
            }
            let mut inn = empty;
            for w in 0..inn.len() {
                inn[w] = uses[b][w] | out[w] & !defs[b][w];
            }
            if inn != live_in[b] || out != live_out[b] {
                (live_in[b], live_out[b]) = (inn, out);
                changed = true;
            }
        }
    }
    (live_in, live_out)
}

/// Phis are placed for every live register at every join; those whose
/// arguments are all one value (or the phi itself) are replaced by it.
fn remove_trivial_phis(blocks: &mut [Block], values: u32) {
    let mut replaced: Vec<Value> = (0..values).map(Value).collect();
    let find = |replaced: &[Value], mut v: Value| {
        while replaced[v.0 as usize] != v {
            v = replaced[v.0 as usize];
        }
        v
    };
    let mut changed = true;
    while changed {
        changed = false;
        for block in blocks.iter_mut() {
            block.phis.retain(|phi| {
                let mut only = None;
                for &a in &phi.args {
                    let a = find(&replaced, a);
                    if a == phi.value || Some(a) == only {
                        continue;
                    }
                    if only.is_some() {
                        return true;
                    }
                    only = Some(a);
                }
                let Some(v) = only else { return true };
                replaced[phi.value.0 as usize] = v;
                changed = true;
                false
            });
        }
    }
    for block in blocks {
        for phi in &mut block.phis {
            for a in &mut phi.args {
                *a = find(&replaced, *a);
            }
        }
        for inst in &mut block.insts {
            for (_, v) in &mut inst.reads {
                *v = find(&replaced, *v);
            }
        }
    }
}

/// Scheduling hints, performance counters and tracing: no effect on results.
const DROPPED: &[&str] = &[
    "s_nop",
    "s_waitcnt",
    "s_waitcnt_vscnt",
    "s_waitcnt_vmcnt",
    "s_waitcnt_expcnt",
    "s_waitcnt_lgkmcnt",
    "s_waitcnt_depctr",
    "s_wait_idle",
    "s_inst_prefetch",
    "s_clause",
    "s_setprio",
    "s_sleep",
    "s_ttracedata",
    "s_ttracedata_imm",
    "s_incperflevel",
    "s_decperflevel",
    "v_nop",
    "v_pipeflush",
    "v_clrexcp",
];

/// Registers read, registers written.
type Effects = (Vec<Reg>, Vec<Reg>);

/// Registers an instruction reads and writes, one per dword, or `None` if
/// the IR drops it.
fn effects(
    inst: &Instruction,
    fields: &Fields,
    name: &'static str,
    lanes: u8,
    vgprs: u16,
) -> Result<Option<Effects>, ShaderError> {
    if DROPPED.contains(&name) {
        return Ok(None);
    }
    let unsupported = || ShaderError::Unsupported {
        offset: inst.offset,
        what: name,
    };
    let mask = |code: u8| (0..lanes).map(move |i| Reg::S(code + i));
    let (mut r, mut w) = (Vec::new(), Vec::new());
    let regs = |op: &Operand, out: &mut Vec<Reg>| {
        add_regs(op, lanes, out).map_err(|_| ShaderError::Isa {
            offset: inst.offset,
            reason: "register range out of bounds",
        })
    };
    let vector = !matches!(
        fields,
        Fields::Sop { .. } | Fields::Sopp { .. } | Fields::Smem { .. }
    );
    match fields {
        Fields::Sop { dst, src, .. } => {
            if name.starts_with("s_movrel")
                || name.starts_with("s_subvector_loop")
                || matches!(name, "s_call_b64" | "s_rfe_b64")
            {
                return Err(unsupported());
            }
            for s in src {
                regs(s, &mut r)?;
            }
            if let Some(d) = dst {
                regs(d, &mut w)?;
                // These change (or may keep) the destination's old value.
                if name.starts_with("s_bitset")
                    || name.starts_with("s_cmov")
                    || matches!(name, "s_addk_i32" | "s_mulk_i32")
                {
                    regs(d, &mut r)?;
                }
            }
            if matches!(name, "s_addc_u32" | "s_subb_u32" | "s_cmovk_i32")
                || name.starts_with("s_cselect_")
                || name.starts_with("s_cmov_")
            {
                r.push(Reg::Scc);
            }
            if name.contains("saveexec") || name.contains("wrexec") {
                r.extend(mask(EXEC));
                w.extend(mask(EXEC));
            }
            if writes_scc(inst.encoding, name) {
                w.push(Reg::Scc);
            }
        }
        Fields::Sopp { .. } => match name {
            "s_cbranch_scc0" | "s_cbranch_scc1" => r.push(Reg::Scc),
            "s_cbranch_vccz" | "s_cbranch_vccnz" => r.extend(mask(VCC)),
            "s_cbranch_execz" | "s_cbranch_execnz" => r.extend(mask(EXEC)),
            "s_sendmsg" | "s_sendmsghalt" => r.push(Reg::S(M0)),
            "s_endpgm"
            | "s_endpgm_saved"
            | "s_endpgm_ordered_ps_done"
            | "s_code_end"
            | "s_branch"
            | "s_barrier"
            | "s_icache_inv"
            | "s_wakeup"
            | "s_round_mode"
            | "s_denorm_mode" => {}
            _ => return Err(unsupported()),
        },
        Fields::Smem {
            data,
            base,
            soffset,
            ..
        } => match name {
            "s_memtime" | "s_memrealtime" | "s_get_waveid_in_workgroup" => {
                regs(data.as_ref().ok_or_else(unsupported)?, &mut w)?;
            }
            "s_gl1_inv" | "s_dcache_inv" | "s_dcache_wb" => {}
            _ if name.contains("_load_") || name.contains("_store_") => {
                regs(base, &mut r)?;
                regs(soffset, &mut r)?;
                let data = data.as_ref().ok_or_else(unsupported)?;
                regs(
                    data,
                    if name.contains("_load_") {
                        &mut w
                    } else {
                        &mut r
                    },
                )?;
            }
            _ => return Err(unsupported()),
        },
        Fields::Valu(v) => {
            let (src_indexed, dst_indexed) = match name {
                "v_movrels_b32" => (true, false),
                "v_movreld_b32" => (false, true),
                "v_movrelsd_b32" | "v_movrelsd_2_b32" | "v_swaprel_b32" => (true, true),
                _ => (false, false),
            };
            // v[reg + M0] for any M0 that stays inside the program's VGPRs.
            let indexed = |op: &Operand, out: &mut Vec<Reg>| match *op {
                Operand::Vector { reg, .. } if u16::from(reg) < vgprs => {
                    out.extend((u16::from(reg)..vgprs).map(|r| Reg::V(r as u8)));
                    Ok(())
                }
                _ => Err(ShaderError::Isa {
                    offset: inst.offset,
                    reason: "relative VGPR base outside the program's VGPRs",
                }),
            };
            for (i, s) in v.src.iter().enumerate() {
                if i == 0 && src_indexed {
                    indexed(s, &mut r)?;
                } else {
                    regs(s, &mut r)?;
                }
            }
            // GFX10 compares into EXEC write nothing else.
            if name.starts_with("v_cmpx") {
                w.extend(mask(EXEC));
            } else if let Some(d) = &v.dst {
                if dst_indexed {
                    indexed(d, &mut w)?;
                } else {
                    regs(d, &mut w)?;
                }
            }
            if let Some(c) = &v.carry {
                regs(c, &mut w)?;
            }
            match name {
                "v_swap_b32" => {
                    regs(&v.src[0], &mut w)?;
                    if let Some(d) = &v.dst {
                        regs(d, &mut r)?;
                    }
                }
                "v_swaprel_b32" => {
                    indexed(&v.src[0], &mut w)?;
                    if let Some(d) = &v.dst {
                        indexed(d, &mut r)?;
                    }
                }
                _ => {}
            }
            if name.starts_with("v_interp_") || src_indexed || dst_indexed {
                r.push(Reg::S(M0));
            }
        }
        Fields::Vintrp { dst, src, .. } => {
            regs(src, &mut r)?;
            r.push(Reg::S(M0));
            regs(dst, &mut w)?;
        }
        Fields::Ds {
            dst,
            addr,
            data,
            gds,
            ..
        } => {
            for o in addr.iter().chain(data) {
                regs(o, &mut r)?;
            }
            if let Some(d) = dst {
                regs(d, &mut w)?;
            }
            if *gds
                || name.starts_with("ds_gws")
                || name.contains("addtid")
                || matches!(name, "ds_ordered_count" | "ds_append" | "ds_consume")
            {
                r.push(Reg::S(M0));
            }
        }
        Fields::Buffer {
            data,
            addr,
            rsrc,
            soffset,
            offen,
            idxen,
            lds,
            cache,
            ..
        } => {
            if *offen || *idxen {
                regs(addr, &mut r)?;
            }
            regs(rsrc, &mut r)?;
            regs(soffset, &mut r)?;
            if *lds {
                r.push(Reg::S(M0));
            }
            if let Some(d) = data {
                if name.contains("_load_") {
                    if !lds {
                        regs(d, &mut w)?;
                    }
                } else if name.contains("_store_") {
                    regs(d, &mut r)?;
                } else if name.contains("_atomic_") {
                    regs(d, &mut r)?;
                    if cache.glc {
                        regs(&atomic_result(d, name), &mut w)?;
                    }
                } else {
                    return Err(unsupported());
                }
            }
        }
        Fields::Image {
            data,
            addr,
            rsrc,
            sampler,
            cache,
            ..
        } => {
            for a in addr.iter().chain([rsrc]).chain(sampler) {
                regs(a, &mut r)?;
            }
            if name.contains("_store") {
                regs(data, &mut r)?;
            } else if name.contains("_atomic_") {
                regs(data, &mut r)?;
                if cache.glc {
                    regs(&atomic_result(data, name), &mut w)?;
                }
            } else {
                regs(data, &mut w)?;
            }
        }
        Fields::Flat {
            dst,
            addr,
            data,
            saddr,
            ..
        } => {
            for o in [addr, data, saddr].into_iter().flatten() {
                regs(o, &mut r)?;
            }
            if let Some(d) = dst {
                regs(d, &mut w)?;
            }
        }
        Fields::Exp { src, .. } => {
            for s in src.iter().flatten() {
                regs(s, &mut r)?;
            }
        }
    }
    if vector {
        r.extend(mask(EXEC));
        // Lanes disabled in EXEC keep the old value.
        r.extend(w.iter().filter(|w| matches!(w, Reg::V(_))));
    }
    let mut seen = Vec::with_capacity(r.len());
    r.retain(|reg| {
        let new = !seen.contains(reg);
        seen.push(*reg);
        new
    });
    Ok(Some((r, w)))
}

/// Scalar ALU opcodes that set SCC (carry, overflow, comparison, or "result
/// is not zero").
fn writes_scc(encoding: gfx10::Encoding, name: &str) -> bool {
    use gfx10::Encoding::*;
    match encoding {
        Sopc => true,
        Sopk => name.starts_with("s_cmpk_") || name == "s_addk_i32",
        _ => {
            let stem = name.split('_').nth(1).unwrap_or("");
            name.contains("saveexec")
                || name.contains("wrexec")
                || matches!(
                    stem,
                    "add"
                        | "sub"
                        | "addc"
                        | "subb"
                        | "min"
                        | "max"
                        | "and"
                        | "or"
                        | "xor"
                        | "andn2"
                        | "orn2"
                        | "nand"
                        | "nor"
                        | "xnor"
                        | "lshl"
                        | "lshr"
                        | "ashr"
                        | "bfe"
                        | "absdiff"
                        | "lshl1"
                        | "lshl2"
                        | "lshl3"
                        | "lshl4"
                        | "not"
                        | "wqm"
                        | "bcnt0"
                        | "bcnt1"
                        | "quadmask"
                        | "abs"
                )
        }
    }
}

/// What an atomic with `glc` returns: the old value (for compare-and-swap,
/// half the data operand).
fn atomic_result(data: &Operand, name: &str) -> Operand {
    match *data {
        Operand::Vector { reg, dwords } if name.contains("cmpswap") => Operand::Vector {
            reg,
            dwords: dwords / 2,
        },
        op => op,
    }
}

/// Appends the register dwords an operand names; reading `vccz`, `execz`,
/// `scc` or `lds_direct` reads the register behind it.
fn add_regs(op: &Operand, lanes: u8, out: &mut Vec<Reg>) -> Result<(), ()> {
    match *op {
        Operand::Scalar { code, .. } if code == gfx10::NULL => {}
        Operand::Scalar { code, dwords } => {
            let end = u16::from(code) + u16::from(dwords);
            if end > 128 {
                return Err(());
            }
            out.extend((code..end as u8).map(Reg::S));
        }
        Operand::Vector { reg, dwords } => {
            if u16::from(reg) + u16::from(dwords) > 256 {
                return Err(());
            }
            out.extend((0..dwords).map(|i| Reg::V(reg + i)));
        }
        Operand::Const(251) => out.extend((0..lanes).map(|i| Reg::S(VCC + i))),
        Operand::Const(252) => out.extend((0..lanes).map(|i| Reg::S(EXEC + i))),
        Operand::Const(253) => out.push(Reg::Scc),
        Operand::Const(254) => out.push(Reg::S(M0)),
        Operand::Const(_) | Operand::Literal(_) | Operand::Attr { .. } => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn build_words(words: &[u32], wave64: bool) -> Program {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        build(&code, wave64, 256).unwrap()
    }

    fn input(p: &Program, reg: Reg) -> Value {
        p.inputs.iter().find(|i| i.0 == reg).expect("input").1
    }

    /// A mask on the table pointer's high dword that clears one of its
    /// address bits points elsewhere: the load through it stays a load
    /// (found by NexForever on #489).
    #[test]
    fn spill_table_masks_keep_every_address_bit() {
        let loads_left = |mask: u32, inline: bool| {
            let mut words = vec![0xBE82_0300u32]; // s_mov_b32 s2, s0
            if inline {
                words.push(0x8703_0001 | (128 + mask) << 8); // s_and_b32 s3, s1, mask
            } else {
                words.extend([0x8703_FF01, mask]); // s_and_b32 s3, s1, lit
            }
            words.extend([0xF408_0201, 0xFA00_0010, S_ENDPGM]); // s_load_dwordx4 s[8:11], s[2:3], 0x10
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let p = build_with(&code, false, 256, true).unwrap();
            p.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .filter(|i| i.name() == "s_load_dwordx4")
                .count()
        };
        for mask in 0..=0xFF {
            assert_eq!(
                loads_left(mask, false),
                usize::from(mask != 0xFF),
                "{mask:#x}"
            );
        }
        for mask in 1..=64 {
            assert_eq!(loads_left(mask, true), 1, "inline {mask}");
        }
        assert_eq!(loads_left(0x1FF, false), 0);
        assert_eq!(loads_left(0xFFFF, false), 0);
        assert_eq!(loads_left(0xFFFE, false), 1);
        assert_eq!(loads_left(0x1_00FF, false), 1);
    }

    /// A compute program whose root arguments are in the spill table `s[0:1]`
    /// points to: the loads become copies of the user SGPRs the dwords stand for.
    #[test]
    fn spill_table_loads_read_user_sgprs() {
        let code: Vec<u8> = [
            0xBE82_0300u32, // s_mov_b32 s2, s0
            0x8703_FF01,    // s_and_b32 s3, s1, lit
            0x0000_00FF,    //   0xff
            0xF408_0201,    // s_load_dwordx4 s[8:11], s[2:3], 0x10
            0xFA00_0010,
            0xBE86_0308, // s_mov_b32 s6, s8
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let plain = build_with(&code, false, 256, false).unwrap();
        assert!(
            plain
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| i.name() == "s_load_dwordx4")
        );
        let p = build_with(&code, false, 256, true).unwrap();
        let insts: Vec<_> = p.blocks.iter().flat_map(|b| &b.insts).collect();
        assert!(!insts.iter().any(|i| i.name().starts_with("s_load")));
        // Table dwords 4..8 are user SGPRs 20..24.
        for k in 0..4 {
            assert!(
                p.inputs.iter().any(|i| i.0 == Reg::S(20 + k)),
                "s{} read",
                20 + k
            );
        }
        // s2, the four entry copies of s20..s23, the four loaded dwords, s6.
        assert_eq!(
            insts.iter().filter(|i| i.name() == "s_mov_b32").count(),
            1 + 4 + 4 + 1
        );
    }

    /// A later table load still reads the arguments as they were on entry,
    /// although an earlier load overwrote the SGPRs its dwords stand for.
    #[test]
    fn spill_table_loads_read_entry_values() {
        let code: Vec<u8> = [
            0xBE82_0300u32, // s_mov_b32 s2, s0
            0x8703_FF01,    // s_and_b32 s3, s1, lit
            0x0000_00FF,    //   0xff
            0xF40C_0301,    // s_load_dwordx8 s[12:19], s[2:3], 0x10 (writes s16..s19)
            0xFA00_0010,
            0xF404_0101, // s_load_dwordx2 s[4:5], s[2:3], 0 (table dwords 0..1)
            0xFA00_0000,
            0xBE86_0304, // s_mov_b32 s6, s4
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p = build_with(&code, false, 256, true).unwrap();
        let insts: Vec<_> = p.blocks.iter().flat_map(|b| &b.insts).collect();
        assert!(!insts.iter().any(|i| i.name().starts_with("s_load")));
        // Table dwords 0..1 and 4..11 are user SGPRs 16..17 and 20..27, all
        // read on entry.
        for reg in (16..18).chain(20..28) {
            assert!(
                p.inputs.iter().any(|i| i.0 == Reg::S(reg)),
                "s{reg} read on entry"
            );
        }
        // s6 holds what s16 held on entry, not what the first load put there.
        let entry = input(&p, Reg::S(16));
        let copied = |v: Value| {
            insts
                .iter()
                .find(|i| i.name() == "s_mov_b32" && i.writes.iter().any(|w| w.1 == v))
                .and_then(|i| i.reads.first().map(|r| r.1))
        };
        let mut value = insts
            .iter()
            .rev()
            .find_map(|i| i.write(Reg::S(6)))
            .expect("s6 written");
        while value != entry {
            value = copied(value).expect("a chain of copies back to the entry value");
        }
    }

    /// Loads from a pointer that is not the spill table stay loads.
    #[test]
    fn spill_table_leaves_other_loads() {
        let code: Vec<u8> = [
            0xF408_0A00u32, // s_load_dwordx4 s[8:11], s[0:1], 0 (s0:1 not yet copied: tagged)
            0xFA00_0000,
            0xBE82_0380, // s_mov_b32 s2, 0
            0xF408_0201, // s_load_dwordx4 s[8:11], s[2:3], 0 (s2 overwritten)
            0xFA00_0000,
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p = build_with(&code, false, 256, true).unwrap();
        let loads = p
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| i.name().starts_with("s_load"))
            .count();
        // The first load is from s[0:1] itself, so it is rewritten; the second
        // reads a pointer the program overwrote, so it stays.
        assert_eq!(loads, 1);
    }

    #[test]
    fn joins_get_phis_only_where_values_differ() {
        let code = [
            0xBF06_0100, // s_cmp_eq_u32 s0, s1
            0xBF85_0001, // s_cbranch_scc1 -> 12
            0xBE82_0380, // s_mov_b32 s2, 0
            0x7E00_0202, // 12: v_mov_b32 v0, s2
            0x7E02_0200, // v_mov_b32 v1, s0
            S_ENDPGM,
        ];
        for wave64 in [false, true] {
            let p = build_words(&code, wave64);
            let [b0, b1, b2] = &p.blocks[..] else {
                panic!("{:?}", p.blocks)
            };
            assert_eq!(
                b0.exit,
                Exit::Branch {
                    taken: 2,
                    not_taken: 1
                }
            );
            assert_eq!(b1.exit, Exit::Jump(2));
            assert_eq!(b2.preds, [0, 1]);
            assert_eq!(b0.insts[1].read(Reg::Scc), b0.insts[0].write(Reg::Scc));

            // s2 differs per path; s0, EXEC, v0 and v1 do not.
            let [phi] = &b2.phis[..] else {
                panic!("{:?}", b2.phis)
            };
            assert_eq!(phi.reg, Reg::S(2));
            assert_eq!(
                phi.args,
                [input(&p, Reg::S(2)), b1.insts[0].write(Reg::S(2)).unwrap()]
            );
            let [mov0, mov1, end] = &b2.insts[..] else {
                panic!()
            };
            assert_eq!(mov0.read(Reg::S(2)), Some(phi.value));
            assert_eq!(mov1.read(Reg::S(0)), Some(input(&p, Reg::S(0))));
            // The old v1 is read: lanes outside EXEC keep it.
            assert_eq!(mov1.read(Reg::V(1)), Some(input(&p, Reg::V(1))));
            assert_eq!(mov1.read(Reg::S(EXEC)), Some(input(&p, Reg::S(EXEC))));
            assert_eq!(mov1.read(Reg::S(EXEC + 1)).is_some(), wave64);
            assert_eq!((end.name(), b2.exit), ("s_endpgm", Exit::End));
        }
    }

    #[test]
    fn in_place_scalar_ops_read_their_destination() {
        let p = build_words(
            &[
                0xBE8D_1D94, // s_bitset1_b32 s13, 20
                0xB78E_0010, // s_addk_i32 s14, 0x10
                0xBE8F_0380, // s_mov_b32 s15, 0
                S_ENDPGM,
            ],
            false,
        );
        let [bitset, addk, mov, _] = &p.blocks[0].insts[..] else {
            panic!("{:?}", p.blocks[0].insts)
        };
        assert_eq!(bitset.read(Reg::S(13)), Some(input(&p, Reg::S(13))));
        assert_eq!(addk.read(Reg::S(14)), Some(input(&p, Reg::S(14))));
        assert_eq!(mov.read(Reg::S(15)), None);
    }

    #[test]
    fn loops_carry_values_through_phis() {
        let p = build_words(
            &[
                0xBE80_0380, // s_mov_b32 s0, 0
                0x8000_8100, // 4: s_add_u32 s0, s0, 1
                0xBF07_8400, // s_cmp_lg_u32 s0, 4
                0xBF85_FFFD, // s_cbranch_scc1 -> 4
                S_ENDPGM,
            ],
            true,
        );
        assert!(p.inputs.is_empty(), "{:?}", p.inputs);
        let [b0, b1, b2] = &p.blocks[..] else {
            panic!()
        };
        assert_eq!(b1.preds, [0, 1]);
        assert_eq!(
            b1.exit,
            Exit::Branch {
                taken: 1,
                not_taken: 2
            }
        );
        let [phi] = &b1.phis[..] else {
            panic!("{:?}", b1.phis)
        };
        let add = &b1.insts[0];
        assert_eq!(phi.reg, Reg::S(0));
        assert_eq!(
            phi.args,
            [
                b0.insts[0].write(Reg::S(0)).unwrap(),
                add.write(Reg::S(0)).unwrap()
            ]
        );
        assert_eq!(add.read(Reg::S(0)), Some(phi.value));
        assert_eq!(b1.insts[1].read(Reg::S(0)), add.write(Reg::S(0)));
        assert_eq!(b2.insts[0].name(), "s_endpgm");
    }

    #[test]
    fn drops_scheduling_and_follows_far_jumps() {
        let p = build_words(
            &[
                0xBF80_0000, // s_nop 0
                0xBE88_1F00, // s_getpc_b64 s[8:9]
                0x8008_FF08,
                20,          // s_add_u32 s8, s8, 20  -> 8 + 20 = 28
                0x8209_8009, // s_addc_u32 s9, s9, 0
                0xBE80_2008, // s_setpc_b64 s[8:9]
                0xFFFF_FFFF, // not code
                S_ENDPGM,
            ],
            false,
        );
        let names: Vec<_> = p.blocks[0].insts.iter().map(Inst::name).collect();
        assert_eq!(
            names,
            ["s_getpc_b64", "s_add_u32", "s_addc_u32", "s_setpc_b64"]
        );
        assert_eq!(p.blocks[0].exit, Exit::Jump(1));
        assert_eq!(p.blocks[1].offset, 28);
        // s_addc_u32 reads the carry s_add_u32 left in SCC.
        let i = &p.blocks[0].insts;
        assert_eq!(i[2].read(Reg::Scc), i[1].write(Reg::Scc));
    }

    #[test]
    fn reports_what_it_cannot_express() {
        let code: Vec<u8> = [0xBE80_2E01_u32, S_ENDPGM] // s_movrels_b32 s0, s1
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        assert_eq!(
            build(&code, false, 256),
            Err(ShaderError::Unsupported {
                offset: 0,
                what: "s_movrels_b32"
            })
        );
    }

    #[test]
    fn relative_vgpr_access_spans_up_to_the_vgpr_count() {
        let code: Vec<u8> = [
            0x7E04_8501_u32, // v_movreld_b32 v2, v1: v[2 + m0] = v1
            0x7E00_8704,     // v_movrels_b32 v0, v4: v0 = v[4 + m0]
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p = build(&code, false, 6).unwrap();
        let [movreld, movrels, _] = &p.blocks[0].insts[..] else {
            panic!()
        };
        let vgprs = |regs: &[(Reg, Value)]| -> Vec<u8> {
            regs.iter()
                .filter_map(|r| match r.0 {
                    Reg::V(v) => Some(v),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(vgprs(&movreld.writes), [2, 3, 4, 5]);
        // v1 is the source; v2-v5 are read for the lanes and indices not written.
        assert_eq!(vgprs(&movreld.reads), [1, 2, 3, 4, 5]);
        assert!(movreld.read(Reg::S(M0)).is_some());
        assert_eq!(vgprs(&movrels.writes), [0]);
        assert_eq!(vgprs(&movrels.reads), [4, 5, 0]);
        assert_eq!(movrels.read(Reg::V(4)), movreld.write(Reg::V(4)));
        assert!(build(&code, false, 4).is_err(), "v4 is past 4 VGPRs");
    }
}
