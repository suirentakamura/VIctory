//! Structured control flow for `main`, as the last step before bitcode.
//!
//! The lowering keeps the console program's branches: jumps from different
//! depths into shared code, exits from nested loops to blocks outside every
//! loop. D3D12 drivers take such DXIL, but translators that need structured
//! control flow (vkd3d-proton's DXIL-to-SPIR-V, the only D3D12 on Linux)
//! must find a structure first and often get it wrong: the Vulkan driver
//! then rejects the SPIR-V (Mesa's RADV crashes on it).
//!
//! Here the function is rebuilt from two shapes only, which every such
//! translator handles: a guarded block (`if (next == id) { block }`) and a
//! loop with one back edge and one exit (`loop { ... } while (next ==
//! header)`). `next` names the block control goes to: one variable, as one
//! thread is at one block at a time. A region (the function, or the body of
//! a loop with its inner loops as single nodes) runs its nodes in reverse
//! postorder, a topological order of its forward edges, so every block runs
//! after the blocks that can branch to it; a block's branch sets `next`
//! instead. A return sets `next` to [`RET`], which no node matches, and the
//! function returns at its end.
//!
//! Values then reach uses their definitions no longer dominate, and phis
//! their predecessors no longer branch to: phis become variables assigned
//! at the end of each predecessor, and SSA form is rebuilt (Cytron et al.:
//! phis at the iterated dominance frontiers of the assignments, reads
//! resolved up the dominator tree) for those, `next`, and values used out of
//! their dominance.
//!
//! The control flow must be reducible (the IR splits irreducible loops);
//! otherwise the function is left as it is.

use std::collections::{HashMap, HashSet};

use super::bitcode::{Module, Op, Term, Ty, V, pred};

/// `next` once the function returns.
const RET: u32 = 0x7fff_ffff;

fn succs(m: &Module, b: usize) -> Vec<usize> {
    match m.blocks[b].term {
        Some(Term::Br(t)) => vec![t],
        Some(Term::CondBr(_, t, f)) => vec![t, f],
        _ => vec![],
    }
}

/// Reverse postorder from block 0, and each block's position in it
/// (`usize::MAX` if unreachable).
fn rpo(m: &Module) -> (Vec<usize>, Vec<usize>) {
    let n = m.blocks.len();
    let mut seen = vec![false; n];
    let mut post = Vec::new();
    let mut stack = vec![(0usize, 0usize)];
    seen[0] = true;
    while let Some(top) = stack.last_mut() {
        let (b, k) = *top;
        top.1 += 1;
        match succs(m, b).get(k) {
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
    let mut pos = vec![usize::MAX; n];
    for (i, &b) in post.iter().enumerate() {
        pos[b] = i;
    }
    (post, pos)
}

/// Predecessors among reachable blocks.
fn preds(m: &Module, order: &[usize]) -> Vec<Vec<usize>> {
    let mut p = vec![Vec::new(); m.blocks.len()];
    for &b in order {
        for s in succs(m, b) {
            if !p[s].contains(&b) {
                p[s].push(b);
            }
        }
    }
    p
}

/// Immediate dominators (Cooper, Harvey and Kennedy); the entry's is
/// itself, unreachable blocks' `usize::MAX`.
fn idoms(order: &[usize], pos: &[usize], preds: &[Vec<usize>]) -> Vec<usize> {
    let mut idom = vec![usize::MAX; pos.len()];
    idom[order[0]] = order[0];
    let intersect = |idom: &[usize], mut a: usize, mut b: usize| {
        while a != b {
            while pos[a] > pos[b] {
                a = idom[a];
            }
            while pos[b] > pos[a] {
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
            for &q in preds[b].iter().filter(|&&q| idom[q] != usize::MAX) {
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

/// Pre and post numbers in the dominator tree: `a` dominates `b` when
/// `pre[a] <= pre[b] && post[b] <= post[a]`.
fn dom_numbers(order: &[usize], idom: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let n = idom.len();
    let mut children = vec![Vec::new(); n];
    for &b in &order[1..] {
        children[idom[b]].push(b);
    }
    let (mut pre, mut post) = (vec![0; n], vec![0; n]);
    let mut clock = 0;
    let mut stack = vec![(order[0], 0usize)];
    pre[order[0]] = clock;
    while let Some(top) = stack.last_mut() {
        let (b, k) = *top;
        top.1 += 1;
        match children[b].get(k) {
            Some(&c) => {
                clock += 1;
                pre[c] = clock;
                stack.push((c, 0));
            }
            None => {
                clock += 1;
                post[b] = clock;
                stack.pop();
            }
        }
    }
    (pre, post)
}

fn operands_mut(op: &mut Op) -> Vec<&mut V> {
    match op {
        Op::Bin(_, a, b) | Op::Cmp(_, a, b) | Op::Store(a, b) | Op::AtomicRmw(_, a, b) => {
            vec![a, b]
        }
        Op::Cast(_, v) | Op::Extract(v, _) | Op::Load(v) => vec![v],
        Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => vec![a, b, c],
        Op::Call(_, args) => args.iter_mut().collect(),
        Op::Phi(inc) => inc.iter_mut().map(|x| &mut x.0).collect(),
        Op::Gep(_, p, idx) => std::iter::once(p).chain(idx.iter_mut()).collect(),
    }
}

/// A region node: a block, or a loop (its header) run as one.
#[derive(Clone, Copy)]
enum Node {
    Block(usize),
    Loop(usize),
}

/// Variables the SSA rebuild tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Var {
    Next,
    /// An original phi (instruction), assigned by its predecessors.
    Edge(u32),
    /// An original value (instruction) read out of its dominance.
    Value(u32),
}

/// How a block assigns a variable at its end.
#[derive(Clone, Copy)]
enum Def {
    Fixed(V),
    /// An original value, resolved where it is assigned.
    Orig(V),
    /// `select(cond, if_true, if_false)` of original values, where `None`
    /// keeps the variable's value.
    Select(V, Option<V>, Option<V>),
}

struct Ssa {
    /// New-CFG predecessors and dominance.
    preds: Vec<Vec<usize>>,
    idom: Vec<usize>,
    pre: Vec<usize>,
    post: Vec<usize>,
    df: Vec<Vec<usize>>,
    defs: HashMap<Var, HashMap<usize, Def>>,
    phi_at: HashMap<Var, HashSet<usize>>,
    entry: HashMap<(Var, usize), V>,
    exit: HashMap<(Var, usize), V>,
    pending: Vec<(Var, usize, V)>,
    /// Original values' blocks, and what an original phi reads as.
    def_block: HashMap<u32, usize>,
    phi_value: HashMap<u32, V>,
    /// Addresses computed again in a block, and how many such
    /// instructions open each block.
    remat: HashMap<(u32, usize), V>,
    front: HashMap<usize, usize>,
    i32: Ty,
}

impl Ssa {
    fn dominates(&self, a: usize, b: usize) -> bool {
        self.pre[a] <= self.pre[b] && self.post[b] <= self.post[a]
    }

    fn ty(&self, m: &Module, var: Var) -> Ty {
        match var {
            Var::Next => self.i32,
            Var::Edge(i) | Var::Value(i) => m.type_of(V::Inst(i)),
        }
    }

    fn place_phis(&mut self, var: Var) {
        if self.phi_at.contains_key(&var) {
            return;
        }
        let mut placed = HashSet::new();
        let mut work: Vec<usize> = self
            .defs
            .get(&var)
            .map(|d| d.keys().copied().collect())
            .unwrap_or_default();
        let mut queued: HashSet<usize> = work.iter().copied().collect();
        while let Some(b) = work.pop() {
            for &f in &self.df[b] {
                if placed.insert(f) && queued.insert(f) {
                    work.push(f);
                }
            }
        }
        self.phi_at.insert(var, placed);
    }

    /// `var` on entry to `b`.
    fn entry_value(&mut self, m: &mut Module, var: Var, b: usize) -> V {
        self.place_phis(var);
        let mut chain = Vec::new();
        let mut at = b;
        let found = loop {
            if let Some(&v) = self.entry.get(&(var, at)) {
                break v;
            }
            if self.phi_at[&var].contains(&at) {
                let ty = self.ty(m, var);
                let phi = m.phi(at, ty);
                self.entry.insert((var, at), phi);
                self.pending.push((var, at, phi));
                break phi;
            }
            if at == 0 {
                let ty = self.ty(m, var);
                break m.undef(ty);
            }
            chain.push(at);
            let up = self.idom[at];
            if let Some(&v) = self.exit.get(&(var, up)) {
                break v;
            }
            if self.defs.get(&var).is_some_and(|d| d.contains_key(&up)) {
                break self.exit_value(m, var, up);
            }
            at = up;
        };
        for c in chain {
            self.entry.insert((var, c), found);
        }
        found
    }

    /// `var` on exit from `b`.
    fn exit_value(&mut self, m: &mut Module, var: Var, b: usize) -> V {
        if let Some(&v) = self.exit.get(&(var, b)) {
            return v;
        }
        let def = self.defs.get(&var).and_then(|d| d.get(&b)).copied();
        let v = match def {
            None => return self.entry_value(m, var, b),
            Some(Def::Fixed(v)) => v,
            Some(Def::Orig(v)) => self.resolve(m, v, b),
            Some(Def::Select(c, t, f)) => {
                let c = self.resolve(m, c, b);
                let side = |s: &mut Self, m: &mut Module, x: Option<V>| match x {
                    Some(x) => s.resolve(m, x, b),
                    None => s.entry_value(m, var, b),
                };
                let t = side(self, m, t);
                let f = side(self, m, f);
                m.select(b, c, t, f)
            }
        };
        self.exit.insert((var, b), v);
        v
    }

    /// Original value `v` as read in (new) block `at`.
    fn resolve(&mut self, m: &mut Module, v: V, at: usize) -> V {
        let V::Inst(i) = v else { return v };
        let Some(&d) = self.def_block.get(&i) else {
            return v;
        };
        let here = |s: &mut Self, m: &mut Module| match s.phi_value.get(&i) {
            Some(&p) => p,
            None if s.def_block.contains_key(&i) && s.is_phi(i) => {
                let p = s.entry_value(m, Var::Edge(i), d);
                s.phi_value.insert(i, p);
                p
            }
            None => v,
        };
        if self.dominates(d, at) {
            return here(self, m);
        }
        // Addresses and resource handles are made again where they are
        // used: a phi of a pointer has no SPIR-V storage class, and one of
        // a handle (with an undefined incoming value) no resource;
        // vkd3d-proton gives up on both.
        let handle = m.callee(i).is_some_and(|f| {
            f.starts_with("dx.op.createHandle") || f.starts_with("dx.op.annotateHandle")
        });
        if handle || matches!(m.insts[i as usize].op, Op::Gep(..)) {
            if let Some(&c) = self.remat.get(&(i, at)) {
                return c;
            }
            let mut op = m.insts[i as usize].op.clone();
            for x in operands_mut(&mut op) {
                *x = self.resolve(m, *x, at);
            }
            let slot = self.front.entry(at).or_insert(0);
            let pos = *slot;
            *slot += 1;
            let rty = m.insts[i as usize].ty;
            let c = m.insert(at, pos, rty, op);
            self.remat.insert((i, at), c);
            return c;
        }
        let var = Var::Value(i);
        if !self.defs.get(&var).is_some_and(|x| x.contains_key(&d)) {
            let value = here(self, m);
            self.defs
                .entry(var)
                .or_default()
                .insert(d, Def::Fixed(value));
        }
        self.entry_value(m, var, at)
    }

    fn is_phi(&self, i: u32) -> bool {
        self.defs.contains_key(&Var::Edge(i))
    }

    /// Phi operands, until no phi is left without them.
    fn finish(&mut self, m: &mut Module) {
        while let Some((var, b, phi)) = self.pending.pop() {
            let preds = self.preds[b].clone();
            let mut incoming = Vec::with_capacity(preds.len());
            for p in preds {
                incoming.push((self.exit_value(m, var, p), p));
            }
            m.set_phi(phi, incoming);
        }
    }
}

/// Rebuilds `main` from guarded blocks and single-exit loops (see the
/// module documentation). Returns whether it did.
pub(super) fn structurize(m: &mut Module) -> bool {
    let (order, pos) = rpo(m);
    if order.len() < 2 {
        return false;
    }
    let p = preds(m, &order);
    let idom = idoms(&order, &pos, &p);

    // Natural loops; a retreating edge to a block that does not dominate
    // its source makes the control flow irreducible.
    let dominates_orig = |a: usize, mut b: usize| loop {
        if a == b {
            return true;
        }
        if idom[b] == b {
            return false;
        }
        b = idom[b];
    };
    let mut loops: HashMap<usize, HashSet<usize>> = HashMap::new();
    for &u in &order {
        for h in succs(m, u) {
            if pos[h] <= pos[u] {
                if !dominates_orig(h, u) {
                    return false;
                }
                let body = loops.entry(h).or_insert_with(|| HashSet::from([h]));
                let mut work = vec![u];
                while let Some(x) = work.pop() {
                    if body.insert(x) {
                        work.extend(p[x].iter().copied());
                    }
                }
            }
        }
    }
    // Innermost loop of each block, and each loop's parent.
    let mut by_size: Vec<usize> = loops.keys().copied().collect();
    by_size.sort_by_key(|h| std::cmp::Reverse(loops[h].len()));
    let mut inner: Vec<Option<usize>> = vec![None; m.blocks.len()];
    for &h in &by_size {
        for &b in &loops[&h] {
            inner[b] = Some(h);
        }
    }
    let parent = |h: usize| -> Option<usize> {
        by_size
            .iter()
            .copied()
            .filter(|&o| o != h && loops[&o].contains(&h))
            .min_by_key(|o| loops[o].len())
    };
    let mut nodes: HashMap<Option<usize>, Vec<Node>> = HashMap::new();
    for &b in &order {
        let region = inner[b];
        nodes.entry(region).or_default().push(Node::Block(b));
        if loops.contains_key(&b) {
            // The header is the loop's first block, and the loop a node of
            // the region around it.
            nodes.entry(parent(b)).or_default().push(Node::Loop(b));
        }
    }
    for list in nodes.values_mut() {
        list.sort_by_key(|n| match *n {
            Node::Block(b) => (pos[b], 1),
            Node::Loop(h) => (pos[h], 0),
        });
    }

    // The entry moves to a block of its own; block 0 becomes the new entry.
    let old_entry = m.new_block();
    m.blocks.swap(0, old_entry);
    let remap = |b: usize| if b == 0 { old_entry } else { b };
    for blk in &mut m.blocks {
        blk.term = match blk.term {
            Some(Term::Br(t)) => Some(Term::Br(remap(t))),
            Some(Term::CondBr(c, t, f)) => Some(Term::CondBr(c, remap(t), remap(f))),
            t => t,
        };
    }
    for i in 0..m.insts.len() {
        if let Op::Phi(inc) = &mut m.insts[i].op {
            for x in inc.iter_mut() {
                x.1 = remap(x.1);
            }
        }
    }
    let fix = |b: usize| if b == 0 { old_entry } else { b };
    let order: Vec<usize> = order.iter().map(|&b| fix(b)).collect();
    let nodes: HashMap<Option<usize>, Vec<Node>> = nodes
        .into_iter()
        .map(|(r, list)| {
            let list = list
                .into_iter()
                .map(|n| match n {
                    Node::Block(b) => Node::Block(fix(b)),
                    Node::Loop(h) => Node::Loop(fix(h)),
                })
                .collect();
            (r.map(fix), list)
        })
        .collect();

    // Original values and phis.
    let mut def_block = HashMap::new();
    let mut phis: Vec<(u32, usize)> = Vec::new();
    for &b in &order {
        for &i in &m.blocks[b].phis {
            def_block.insert(i, b);
            phis.push((i, b));
        }
        for &i in &m.blocks[b].body {
            def_block.insert(i, b);
        }
    }
    let original_terms: HashMap<usize, Term> = order
        .iter()
        .map(|&b| (b, m.blocks[b].term.expect("terminated block")))
        .collect();
    let original_body: Vec<(usize, Vec<u32>)> = order
        .iter()
        .map(|&b| (b, m.blocks[b].body.clone()))
        .collect();

    let i32t = m.int(32);
    let mut defs: HashMap<Var, HashMap<usize, Def>> = HashMap::new();
    // Phis become edge variables their predecessors assign.
    for &(i, s) in &phis {
        let Op::Phi(inc) = m.insts[i as usize].op.clone() else {
            unreachable!()
        };
        let entry = defs.entry(Var::Edge(i)).or_default();
        for (v, pb) in inc {
            let Some(&term) = original_terms.get(&pb) else {
                continue;
            };
            let def = match term {
                Term::CondBr(c, t, f) if t == s && f != s => Def::Select(c, Some(v), None),
                Term::CondBr(c, t, f) if f == s && t != s => Def::Select(c, None, Some(v)),
                _ => Def::Orig(v),
            };
            entry.insert(pb, def);
        }
        m.blocks[s].phis.retain(|&x| x != i);
    }
    // `next`: the entry starts at the old entry; each block names its
    // successor.
    let label = |m: &mut Module, b: usize| m.i32(b as u32);
    let mut next_defs = HashMap::new();
    let start = label(m, old_entry);
    next_defs.insert(0usize, Def::Fixed(start));
    for (&b, &term) in &original_terms {
        let def = match term {
            Term::Br(t) => Def::Fixed(label(m, t)),
            Term::CondBr(c, t, f) => {
                let (t, f) = (label(m, t), label(m, f));
                Def::Select(c, Some(t), Some(f))
            }
            Term::Ret => Def::Fixed(m.i32(RET)),
        };
        next_defs.insert(b, def);
    }
    defs.insert(Var::Next, next_defs);

    // The new control flow. Checks read `next` (filled in once SSA is
    // rebuilt).
    let mut checks: Vec<(usize, V)> = Vec::new();
    fn sequence(
        m: &mut Module,
        nodes: &HashMap<Option<usize>, Vec<Node>>,
        region: Option<usize>,
        mut cur: usize,
        checks: &mut Vec<(usize, V)>,
    ) -> usize {
        for &node in nodes.get(&region).map(Vec::as_slice).unwrap_or(&[]) {
            let after = m.new_block();
            match node {
                Node::Block(b) => {
                    let id = m.i32(b as u32);
                    checks.push((cur, id));
                    // The condition is set once `next` is known.
                    m.terminate(cur, Term::CondBr(V::Const(u32::MAX), b, after));
                    m.terminate(b, Term::Br(after));
                }
                Node::Loop(h) => {
                    let id = m.i32(h as u32);
                    let head = m.new_block();
                    checks.push((cur, id));
                    m.terminate(cur, Term::CondBr(V::Const(u32::MAX), head, after));
                    let latch = sequence(m, nodes, Some(h), head, checks);
                    checks.push((latch, id));
                    let exit = m.new_block();
                    m.terminate(latch, Term::CondBr(V::Const(u32::MAX), head, exit));
                    m.terminate(exit, Term::Br(after));
                }
            }
            cur = after;
        }
        cur
    }
    let first = m.new_block();
    m.terminate(0, Term::Br(first));
    let end = sequence(m, &nodes, None, first, &mut checks);
    m.terminate(end, Term::Ret);

    // Dominance in the new control flow.
    let (order2, pos2) = rpo(m);
    let p2 = preds(m, &order2);
    let idom2 = idoms(&order2, &pos2, &p2);
    let (pre, post) = dom_numbers(&order2, &idom2);
    let mut df = vec![Vec::new(); m.blocks.len()];
    for &b in &order2 {
        if p2[b].len() >= 2 {
            for &q in &p2[b] {
                let mut r = q;
                while r != idom2[b] {
                    if !df[r].contains(&b) {
                        df[r].push(b);
                    }
                    r = idom2[r];
                }
            }
        }
    }
    let mut ssa = Ssa {
        preds: p2,
        idom: idom2,
        pre,
        post,
        df,
        defs,
        phi_at: HashMap::new(),
        entry: HashMap::new(),
        exit: HashMap::new(),
        pending: Vec::new(),
        def_block,
        phi_value: HashMap::new(),
        remat: HashMap::new(),
        front: HashMap::new(),
        i32: i32t,
    };

    // Phis first, in reverse postorder: a chain of phis passing a value on
    // then resolves one link at a time, not by recursion through all of
    // them (deep enough to overflow a thread's stack).
    for &(i, s) in &phis {
        let _ = ssa.resolve(m, V::Inst(i), s);
    }
    // Original instructions read their operands where they now are.
    for (b, body) in original_body {
        for i in body {
            let mut op = m.insts[i as usize].op.clone();
            for v in operands_mut(&mut op) {
                *v = ssa.resolve(m, *v, b);
            }
            m.insts[i as usize].op = op;
        }
    }
    // Checks and latches compare `next`.
    for (block, id) in checks {
        let next = ssa.entry_value(m, Var::Next, block);
        let cond = m.cmp(block, pred::EQ, next, id);
        if let Some(Term::CondBr(_, t, f)) = m.blocks[block].term {
            m.blocks[block].term = Some(Term::CondBr(cond, t, f));
        }
    }
    ssa.finish(m);
    true
}
