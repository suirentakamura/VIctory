//! The compute shader that bounds indirect arguments before the host GPU
//! reads them (`victory-gpu`'s `indirect` says why and when it runs),
//! built with our own writer so that no system shader compiler is needed.
//!
//! One thread per record. The record's counts are clamped in place, their
//! cost is added to the list's budget, and a record past the budget is
//! emptied.

use super::bitcode::{Attr, Bin, Term, V, pred};
use super::shader::Shader;

/// The root parameter of the argument buffer (`u1`, a raw buffer at the
/// first record).
pub const ARGUMENTS: u8 = 1;
/// The root parameter of the list's budget counter (`u2`, a raw buffer).
pub const BUDGET: u8 = 2;
/// Threads of one group; the dispatch covers the records in groups of it.
pub const GROUP: u32 = 64;

/// A draw's vertex or index count is cut to this.
const VERTICES: u32 = 1 << 24;
/// A draw's vertices times instances are cut to this.
const VERTEX_INSTANCES: u32 = 1 << 26;
/// A dispatch's groups per dimension are cut to this.
const DIMENSION: u32 = 65535;
/// A dispatch of more groups than this runs empty.
const GROUPS: u32 = 1 << 20;
/// The work of one list (vertex instances, threads): records past it run
/// empty.
const LIST: u32 = 1 << 27;

/// The clamp as a DXIL container. Its user data is four dwords: the kind
/// of record (0 and 1 draws, 2 dispatches, else mesh dispatches), the
/// records' stride in bytes, their count, and one unused.
pub fn argument_clamp() -> Vec<u8> {
    let mut s = Shader::new(false);
    s.threads = Some([GROUP, 1, 1]);
    s.user_data = 4;
    let i32t = s.t.i32;
    let entry = s.entry;
    let zero = s.m.i32(0);

    let user = s.user_data_handle();
    let row = s.op(
        entry,
        "cbufferLoadLegacy.i32",
        Attr::ReadOnly,
        s.t.cbuf_ret,
        59,
        &[user, zero],
    );
    let kind = s.m.extract(entry, row, 0);
    let stride = s.m.extract(entry, row, 1);
    let records = s.m.extract(entry, row, 2);
    let args = s.root_uav_handle(ARGUMENTS);
    let budget = s.root_uav_handle(BUDGET);
    let id = s.op(entry, "threadId.i32", Attr::ReadNone, i32t, 93, &[zero]);

    let [body, draw, other, dispatch, mesh, join, empty, exit] =
        std::array::from_fn(|_| s.m.new_block());
    let live = s.m.cmp(entry, pred::IULT, id, records);
    s.m.terminate(entry, Term::CondBr(live, body, exit));

    let at = s.m.bin(body, Bin::Mul, id, stride);
    let one = s.m.i32(1);
    let is_draw = s.m.cmp(body, pred::IULE, kind, one);
    s.m.terminate(body, Term::CondBr(is_draw, draw, other));

    // Draws: vertex or index count, then instances.
    let draw_cost = {
        let b = draw;
        let count = load(&mut s, b, args, at, 0);
        let count = umin(&mut s, b, count, VERTICES);
        let divisor = at_least_one(&mut s, b, count);
        let most = s.m.i32(VERTEX_INSTANCES);
        let most = s.m.bin(b, Bin::UDiv, most, divisor);
        let instances = load(&mut s, b, args, at, 4);
        let fits = s.m.cmp(b, pred::IULT, instances, most);
        let instances = s.m.select(b, fits, instances, most);
        store(&mut s, b, args, at, 0, count);
        store(&mut s, b, args, at, 4, instances);
        s.m.terminate(b, Term::Br(join));
        s.m.bin(b, Bin::Mul, count, instances)
    };

    let two = s.m.i32(2);
    let is_dispatch = s.m.cmp(other, pred::EQ, kind, two);
    s.m.terminate(other, Term::CondBr(is_dispatch, dispatch, mesh));

    // Dispatches: three group counts; x * y fits 32 bits, the whole count
    // may not.
    let dispatch_cost = {
        let b = dispatch;
        let mut g = [0, 4, 8].map(|offset| {
            let x = load(&mut s, b, args, at, offset);
            umin(&mut s, b, x, DIMENSION)
        });
        let xy = s.m.bin(b, Bin::Mul, g[0], g[1]);
        let divisor = at_least_one(&mut s, b, g[2]);
        let most = s.m.i32(GROUPS);
        let most = s.m.bin(b, Bin::UDiv, most, divisor);
        let deep = s.m.cmp(b, pred::NE, g[2], zero);
        let many = s.m.cmp(b, pred::IUGT, xy, most);
        let over = s.m.bin(b, Bin::And, deep, many);
        for x in &mut g {
            *x = s.m.select(b, over, zero, *x);
        }
        let cost = groups(&mut s, b, g, 64);
        for (x, offset) in g.into_iter().zip([0, 4, 8]) {
            store(&mut s, b, args, at, offset, x);
        }
        s.m.terminate(b, Term::Br(join));
        cost
    };

    // Mesh dispatches: the console's are one-dimensional.
    let mesh_cost = {
        let b = mesh;
        let g = [(0, DIMENSION), (4, 1), (8, 1)].map(|(offset, most)| {
            let x = load(&mut s, b, args, at, offset);
            umin(&mut s, b, x, most)
        });
        let cost = groups(&mut s, b, g, 128);
        for (x, offset) in g.into_iter().zip([0, 4, 8]) {
            store(&mut s, b, args, at, offset, x);
        }
        s.m.terminate(b, Term::Br(join));
        cost
    };

    let cost = s.m.phi(join, i32t);
    s.m.set_phi(
        cost,
        vec![
            (draw_cost, draw),
            (dispatch_cost, dispatch),
            (mesh_cost, mesh),
        ],
    );
    let undef = s.m.undef(i32t);
    // Adds the cost and returns the budget's value before it.
    let before = s.op(
        join,
        "atomicBinOp.i32",
        Attr::NoUnwind,
        i32t,
        78,
        &[budget, zero, zero, undef, undef, cost],
    );
    let list = s.m.i32(LIST);
    let spent = s.m.cmp(join, pred::IUGE, before, list);
    let after = s.m.bin(join, Bin::Add, before, cost);
    let past = s.m.cmp(join, pred::IUGT, after, list);
    let over = s.m.bin(join, Bin::Or, spent, past);
    s.m.terminate(join, Term::CondBr(over, empty, exit));

    // Draws keep their layout with no vertices; groups become zero.
    store(&mut s, empty, args, at, 0, zero);
    s.m.terminate(empty, Term::Br(exit));

    s.m.terminate(exit, Term::Ret);
    s.finish()
}

/// The dword `offset` bytes into the record at `at`.
fn load(s: &mut Shader, block: usize, buffer: V, at: V, offset: u32) -> V {
    let i32t = s.t.i32;
    let ret =
        s.m.named_struct("dx.types.ResRet.i32", vec![i32t, i32t, i32t, i32t, i32t]);
    let at = address(s, block, at, offset);
    let (undef, mask, align) = (s.m.undef(i32t), s.m.i8(1), s.m.i32(4));
    let r = s.op(
        block,
        "rawBufferLoad.i32",
        Attr::ReadOnly,
        ret,
        139,
        &[buffer, at, undef, mask, align],
    );
    s.m.extract(block, r, 0)
}

/// Writes the dword `offset` bytes into the record at `at`.
fn store(s: &mut Shader, block: usize, buffer: V, at: V, offset: u32, value: V) {
    let (i32t, void) = (s.t.i32, s.t.void);
    let at = address(s, block, at, offset);
    let (undef, mask, align) = (s.m.undef(i32t), s.m.i8(1), s.m.i32(4));
    s.op(
        block,
        "rawBufferStore.i32",
        Attr::NoUnwind,
        void,
        140,
        &[buffer, at, undef, value, undef, undef, undef, mask, align],
    );
}

fn address(s: &mut Shader, block: usize, at: V, offset: u32) -> V {
    if offset == 0 {
        return at;
    }
    let offset = s.m.i32(offset);
    s.m.bin(block, Bin::Add, at, offset)
}

/// The smaller of `x` and the constant `most`, unsigned.
fn umin(s: &mut Shader, block: usize, x: V, most: u32) -> V {
    let most = s.m.i32(most);
    let fits = s.m.cmp(block, pred::IULT, x, most);
    s.m.select(block, fits, x, most)
}

/// `x`, or 1 for 0: a divisor.
fn at_least_one(s: &mut Shader, block: usize, x: V) -> V {
    let (zero, one) = (s.m.i32(0), s.m.i32(1));
    let none = s.m.cmp(block, pred::EQ, x, zero);
    s.m.select(block, none, one, x)
}

/// The threads of `g` groups of `threads` each.
fn groups(s: &mut Shader, block: usize, g: [V; 3], threads: u32) -> V {
    let xy = s.m.bin(block, Bin::Mul, g[0], g[1]);
    let xyz = s.m.bin(block, Bin::Mul, xy, g[2]);
    let threads = s.m.i32(threads);
    s.m.bin(block, Bin::Mul, xyz, threads)
}

#[cfg(all(test, windows))]
mod tests {
    use windows::Win32::Graphics::Direct3D12::*;

    use super::{
        ARGUMENTS, BUDGET, DIMENSION, GROUP, GROUPS, LIST, VERTEX_INSTANCES, VERTICES,
        argument_clamp,
    };
    use crate::emit::host::{Dispatch, Host, Root};
    use crate::emit::{ROOT_CBV_SPACE, USER_DATA_SPACE};

    /// Dwords of a test record: three counts and one the clamp must leave.
    const STRIDE: usize = 4;
    const MARK: u32 = 0xC0FF_EE00;

    /// What the clamp does to one record, and the record's cost.
    fn clamped(kind: u32, r: [u32; 3]) -> ([u32; 3], u32) {
        match kind {
            0 | 1 => {
                let count = r[0].min(VERTICES);
                let instances = r[1].min(VERTEX_INSTANCES / count.max(1));
                ([count, instances, r[2]], count * instances)
            }
            2 => {
                let mut g = r.map(|x| x.min(DIMENSION));
                if g[2] != 0 && g[0] * g[1] > GROUPS / g[2] {
                    g = [0; 3];
                }
                (g, g[0] * g[1] * g[2] * 64)
            }
            _ => {
                let g = [r[0].min(DIMENSION), r[1].min(1), r[2].min(1)];
                (g, g[0] * g[1] * g[2] * 128)
            }
        }
    }

    /// Runs the clamp over `records` (and one more it must not touch) with
    /// the budget at `spent`: the buffer and the budget afterwards.
    fn run(host: &Host, kind: u32, records: &[[u32; 3]], spent: u32) -> (Vec<u32>, u32) {
        let uav = |slot: u8| D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: u32::from(slot),
                    RegisterSpace: ROOT_CBV_SPACE,
                    Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let params = [
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: USER_DATA_SPACE,
                        Num32BitValues: 4,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            uav(ARGUMENTS),
            uav(BUDGET),
        ];
        let root = host.root_signature_of(&params, &[]).unwrap();
        let pso = host
            .compute_pipeline_with(&root, &argument_clamp())
            .unwrap();
        let mut data: Vec<u32> = records
            .iter()
            .flat_map(|r| [r[0], r[1], r[2], MARK])
            .collect();
        // Past the record count: left alone.
        data.extend([u32::MAX, u32::MAX, u32::MAX, MARK]);
        let count = records.len() as u32;
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (
                            0,
                            Root::Constants(vec![kind, (STRIDE * 4) as u32, count, 0]),
                        ),
                        (u32::from(ARGUMENTS), Root::Uav(data)),
                        (u32::from(BUDGET), Root::Uav(vec![spent])),
                    ],
                    views: &[],
                    samplers: &[],
                    groups: [count.div_ceil(GROUP).max(1), 1, 1],
                },
            )
            .unwrap();
        (out[0].clone(), out[1][0])
    }

    #[test]
    fn records_are_clamped_in_place() {
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let cases: [(u32, &[[u32; 3]]); 4] = [
            // Draws and indexed draws: count, instances (the third dword
            // is not theirs to clamp).
            (
                0,
                &[
                    [100, 3, 7],
                    [u32::MAX, u32::MAX, 7],
                    [0, 9, 7],
                    [1000, 1 << 20, 7],
                ],
            ),
            (1, &[[36, 1, 0], [VERTICES + 1, 2, 0]]),
            // Dispatches: one dimension cut, too many groups, no depth.
            (
                2,
                &[[70000, 2, 3], [65535, 65535, 2], [5, 5, 0], [1024, 1024, 1]],
            ),
            // Mesh dispatches: one-dimensional.
            (3, &[[70000, 9, 0], [10, 5, 7], [3, 1, 1]]),
        ];
        for (kind, records) in cases {
            // The cases stay inside the budget, so no record is emptied
            // whatever order the threads add their costs in.
            let total: u32 = records.iter().map(|&r| clamped(kind, r).1).sum();
            assert!(total <= LIST, "kind {kind}: the case is past the budget");
            let (data, budget) = run(&host, kind, records, 0);
            for (n, &r) in records.iter().enumerate() {
                let (want, _) = clamped(kind, r);
                assert_eq!(
                    data[n * STRIDE..n * STRIDE + 3],
                    want,
                    "kind {kind}, record {n} {r:?}"
                );
                assert_eq!(data[n * STRIDE + 3], MARK, "kind {kind}, record {n}");
            }
            assert_eq!(budget, total, "kind {kind}: the budget");
            let past = records.len() * STRIDE;
            assert_eq!(
                data[past..],
                [u32::MAX, u32::MAX, u32::MAX, MARK],
                "kind {kind}: a record past the count"
            );
        }
    }

    #[test]
    fn records_past_the_budget_run_empty() {
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        // Fits exactly; crosses the budget; the budget already spent.
        for (spent, emptied) in [(LIST - 300, false), (LIST - 299, true), (LIST, true)] {
            let (data, budget) = run(&host, 0, &[[100, 3, 7]], spent);
            let first = if emptied { 0 } else { 100 };
            assert_eq!(data[..4], [first, 3, 7, MARK], "budget at {spent:#x}");
            // The cost is added either way.
            assert_eq!(budget, spent + 300, "budget at {spent:#x}");
        }
        // A dispatch past the budget loses its first dimension.
        let (data, _) = run(&host, 2, &[[4, 4, 4]], LIST);
        assert_eq!(data[..4], [0, 4, 4, MARK]);
    }
}
