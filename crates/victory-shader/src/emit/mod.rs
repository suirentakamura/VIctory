//! DXIL emission (decision 0003, step 3): a console vertex, pixel, compute
//! or mesh shader as a PC DXIL container (SM 6.6) that host D3D12 accepts.
//! Compute shaders keep the console's thread-group size; one that depends
//! on the lane layout (cross-lane operations) declares the console's wave
//! size (`WaveSize`), so the host device must support it: [`Target`] says
//! which wave sizes the host runs, and a shader that needs another one does
//! not translate for it (rather than being refused at pipeline creation).
//!
//! What the host binds for a translated shader ([`HostShader`]):
//!
//! * **User data** at `b0, space 0` (32-bit root constants or a constant
//!   buffer, per stage): dword `index` holds a root parameter's value, as
//!   [`UserData`] lists. A root-constant dword is the constant. A
//!   descriptor table is the byte offset of its first descriptor in its
//!   heap: the descriptor index times 32 for CBV/SRV/UAV tables, times 16
//!   for sampler tables (the console's descriptor sizes; the code adds
//!   offsets in those units). The heap bases of dynamic resources are 0
//!   and not listed.
//! * **Descriptors** through SM 6.6 `ResourceDescriptorHeap[]` and
//!   `SamplerDescriptorHeap[]`, at the index the console address gives, so
//!   the host heaps keep the title's descriptor order. The root signature
//!   needs `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` and
//!   `SAMPLER_HEAP_DIRECTLY_INDEXED`.
//! * **Immediate samplers** as static samplers `s<n>, space 0`, `n` the
//!   position in [`HostShader::samplers`] (the console sampler words) and
//!   [`HostShader::static_samplers`] (the same decoded into a D3D12 static
//!   sampler description). Comparison sampling (added 2026-11-23): an
//!   immediate sampler it uses has a comparison filter there (bit 0x80);
//!   a heap sampler it uses must be a comparison sampler on the host.
//! * **Root CBVs and root SRVs** (raw buffers) as root descriptors
//!   `b<i>` / `t<i>` in [`ROOT_CBV_SPACE`], `i` the title's root parameter
//!   index ([`HostShader::root_cbvs`], [`HostShader::root_srvs`]); the
//!   address the title sets is the buffer's. An address the title passes
//!   in two root constants is a root view too, at a register that names
//!   those constants ([`RootView`]).
//! * **Vertex fetch** (added 2026-11-23). Grand Theft Auto VI's mesh and
//!   vertex programs fetch vertices themselves with typed-format loads
//!   (`tbuffer_load_format_*`): the format (8_8_8_8 unorm, 16_16 snorm,
//!   2_10_10_10 unorm, 8_8_8_8 and 16_16_16_16 uint...) is in the
//!   instruction, so the translation reads raw bytes and converts them
//!   itself. Their descriptors are built around a root SRV (stride 1, no
//!   record limit): the host binds the root SRV `t<slot>` above with the
//!   address the title sets, nothing more. The vertex index, the stride
//!   and the attribute offset are the shader's own arithmetic on data in
//!   that buffer (per-stream headers), not descriptor fields.
//! * **Buffer views in the heap** are declared by how the shader reads
//!   them: raw (ByteAddressBuffer) for dword loads and stores, atomics and
//!   typed-format loads without an index; structured with the used-resource
//!   entry's stride when indexed; typed with four 32-bit float channels for
//!   `buffer_load_format_*` and `buffer_store_format_*`. The host's view at
//!   that index must be of that kind and start at the console descriptor's
//!   base address (raw: `R32_TYPELESS` with the raw flag), whatever kind the
//!   title's view description says. Typed-format loads from heap
//!   descriptors (1,024 world compute programs, ray-tracing material
//!   resolves) read raw views this way.
//!
//! * **Texture atomics** (added 2026-11-23): a texture UAV the shader uses
//!   atomically is declared a 32-bit unsigned single-channel view; the
//!   host's view at that index must be an R32 integer view (64-bit atomics
//!   stop the translation). Global loads from a root SRV read it raw.
//!
//! Entry points: [`translate`] (container bytes as the title passes them
//! at pipeline creation → [`HostShader`]), [`emit`] (the same with the
//! earlier stages already run). The root constants of each stage must
//! cover [`HostShader::user_data_dwords`]. Stability: these types and the
//! binding scheme are what pipeline creation builds on; changes are
//! announced.
//! * **Signatures** as the console container has them (same semantics,
//!   registers and masks), so the input layout and stage linking are the
//!   title's; pixel outputs are plain 32-bit values.
//! * **Mesh shaders** ([`HwStage::Ms`], `ms_6_6`): user data and
//!   descriptors as above (the root constants visible to the mesh stage);
//!   the pipeline is a mesh pipeline (pipeline-state stream, no input
//!   layout) with the title's pixel shader. The thread group is the
//!   console's in lanes (`[numthreads]` in the DXIL), the vertex and
//!   primitive maxima its own, triangles only, no wave size required. The
//!   console numbers groups in x only: the draw path passes the title's
//!   count as `DispatchMesh(x, 1, 1)`. Per-primitive outputs (`PSG1`) are
//!   the console's, without `SV_PrimitiveID` and `SV_CullPrimitive` (the
//!   console never exports them).

mod bitcode;
pub mod clamp;
mod container;
#[cfg(all(test, windows))]
mod diff_tests;
#[cfg(all(windows, any(test, feature = "host")))]
pub mod host;
#[cfg(all(test, windows))]
mod image_tests;
#[cfg(all(test, windows))]
mod lane_read_loop_tests;
mod lower;
mod sampler;
mod shader;
#[cfg(all(test, windows))]
mod split_tests;
#[cfg(windows)]
pub mod validator;
#[cfg(all(test, windows))]
mod valu_tests;
#[cfg(all(test, windows))]
mod waterfall_tests;
mod wave;

use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io, ir, resource, structure};

pub use sampler::StaticSampler;
pub use shader::{
    ROOT_CBV_SPACE, RawWindow, SAMPLER_SPACE, STRIDE_SPACE, StrideEntry, USER_DATA_SPACE,
};

/// The loop limit without an override: 2^20.
pub const DEFAULT_LOOP_LIMIT: u32 = lower::LOOP_LIMIT;

/// The iteration count after which a translated loop is left, per lane and
/// per entry into the loop: [`DEFAULT_LOOP_LIMIT`].
///
/// With the `diagnostics` feature, `VICTORY_SHADER_LOOP_LIMIT` (decimal or
/// `0x` hex, at least 1) sets it instead. A loop that hits the limit in
/// every lane of a full-screen pass still runs for seconds (2^20 times a
/// 4K screen is about 10^13 iterations); a much lower limit shows whether
/// such loops are what makes a frame slow. The value is part of the
/// translation, so the shader cache must key on it (`victory-gpu`'s does).
pub fn loop_limit() -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        if cfg!(feature = "diagnostics") {
            parse_loop_limit(std::env::var("VICTORY_SHADER_LOOP_LIMIT").ok().as_deref())
        } else {
            lower::LOOP_LIMIT
        }
    })
}

/// `VICTORY_SHADER_LOOP_LIMIT`'s value; the default when it is unset,
/// unreadable or 0. Pure, for testing.
fn parse_loop_limit(value: Option<&str>) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    match parsed {
        Some(n) if n > 0 => n,
        _ => lower::LOOP_LIMIT,
    }
}

/// A translated shader and what the host binds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostShader {
    pub stage: HwStage,
    /// The DXIL container, hashed as D3D12 expects.
    pub dxil: Vec<u8>,
    /// The user-data dwords the shader reads.
    pub user_data: Vec<UserData>,
    /// Immediate samplers: the four sampler words of `s<n>`.
    pub samplers: Vec<[u32; 4]>,
    /// The same samplers decoded, `s<n>` at position n: the static sampler
    /// descriptions the host's root signature declares.
    pub static_samplers: Vec<StaticSampler>,
    /// Root CBVs the shader reads, each bound as `b<register>` in
    /// [`ROOT_CBV_SPACE`] (see [`RootView`] for the register).
    pub root_cbvs: Vec<u16>,
    /// Root SRVs (raw buffers) the shader reads: `t<register>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_srvs: Vec<u16>,
    /// Root UAVs (raw buffers) the shader reads and writes: `u<register>`
    /// in [`ROOT_CBV_SPACE`].
    pub root_uavs: Vec<u16>,
    /// True when the shader reads the runtime stride table (`t0` in
    /// [`STRIDE_SPACE`]).
    pub stride_table: bool,
}

/// Where the address of a root view comes from, and the register it is
/// bound at: the title's root parameter index for a root descriptor;
/// `(dword + 1) << 8 | slot` for an address in two root constants, low
/// half at `dword` (#614).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootView {
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// The root constant holding the address's low half (the high half is
    /// the next one); none for a root descriptor. Below 255.
    pub dword: Option<u8>,
}

impl RootView {
    pub fn register(self) -> u16 {
        u16::from(self.dword.map_or(0, |d| d + 1)) << 8 | u16::from(self.slot)
    }

    pub fn of(register: u16) -> RootView {
        RootView {
            slot: register as u8,
            dword: (register >> 8).checked_sub(1).map(|d| d as u8),
        }
    }
}

impl HostShader {
    /// Size of the user-data buffer the shader declares, in dwords (a
    /// multiple of 4; 0 when it reads none). Root constants must cover it.
    pub fn user_data_dwords(&self) -> u32 {
        let n = self
            .user_data
            .iter()
            .map(|u| u.index + 1)
            .max()
            .unwrap_or(0);
        n.next_multiple_of(4)
    }
}

/// One dword of user data and the root parameter that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserData {
    /// Dword in the user-data buffer (the console's user-data index).
    pub index: u32,
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// `Constants` or `Table` (see the module notes for the value).
    pub root: RootType,
    /// Dword within the parameter.
    pub dword: u32,
}

/// The wave sizes (lanes per wave) the host GPU runs, as D3D12 reports them
/// (`D3D12_FEATURE_DATA_D3D12_OPTIONS1::WaveLaneCountMin` / `Max`). A
/// compute shader that declares the console's wave size translates only
/// for a host whose range holds it: NVIDIA's range is 32..32, so a console
/// wave64 program that reads other lanes needs 64-lane waves it cannot get.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Target {
    pub lanes_min: u32,
    pub lanes_max: u32,
}

impl Target {
    /// No limit known: every wave size is taken to run (the host refuses
    /// a pipeline it cannot run).
    pub const ANY: Target = Target {
        lanes_min: 4,
        lanes_max: 128,
    };

    /// Whether the host runs waves of `lanes`.
    pub fn runs(self, lanes: u32) -> bool {
        (self.lanes_min..=self.lanes_max).contains(&lanes)
    }

    /// The range as bytes, for cache keys.
    pub fn key(self) -> [u8; 8] {
        let mut k = [0; 8];
        k[..4].copy_from_slice(&self.lanes_min.to_le_bytes());
        k[4..].copy_from_slice(&self.lanes_max.to_le_bytes());
        k
    }
}

impl Default for Target {
    fn default() -> Self {
        Target::ANY
    }
}

/// A diagnostic: what the module this thread emitted last declares and
/// calls (thread-group size, wave size, `dx.op` operations), for reports of
/// a module the validator refuses. Nothing of the console program is in it.
pub fn last_module_summary() -> String {
    shader::last_summary()
}

/// Translates the vertex, pixel, compute or mesh program of a console
/// shader container, for any host ([`Target::ANY`]).
pub fn translate(container: &[u8]) -> Result<HostShader, ShaderError> {
    translate_for(container, Target::ANY)
}

/// [`translate`] for a host that runs the wave sizes of `target`.
pub fn translate_for(container: &[u8], target: Target) -> Result<HostShader, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c
        .part(FourCC::DXIL)
        .ok_or(ShaderError::Malformed("no DXIL part"))?;
    let native = crate::native_programs(dxil.data)?
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms
            )
        })
        .ok_or(ShaderError::Unsupported {
            offset: 0,
            what: "a container without a vertex, pixel, compute or mesh program",
        })?;
    let spilled = native.spills_root_args();
    let p = ir::build_with(&native.patched_code(), native.wave64, native.vgprs, spilled)?;
    // Structurisation first: tracing per-lane descriptors needs it (its
    // failure is reported after the earlier stages').
    let st = structure::structure(&p);
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io = io::raise(&p, &native)?;
    let st = st?;
    emit(&c, &native, &p, &st, &io, &res, target)
}

/// Emits a program whose earlier stages already ran, for `target`.
pub fn emit(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    target: Target,
) -> Result<HostShader, ShaderError> {
    let sig = |fourcc| -> Result<Vec<container::SigElement>, ShaderError> {
        match c.part(fourcc) {
            Some(part) => signature(part.data),
            None => Ok(Vec::new()),
        }
    };
    let pixel = native.stage == HwStage::Ps;
    let mut inputs = sig(FourCC::ISG1)?;
    let mut outputs = sig(FourCC::OSG1)?;
    // Interpolation as DXC declares it by default; pixel inputs that are
    // read get the mode their barycentrics say.
    let interpolated = if pixel { &mut inputs } else { &mut outputs };
    for e in interpolated {
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            // Constant, as DXC declares it (the AMD compiler crashed on a
            // front face without a mode).
            (13, _) => 1,
            // Until a read says which mode (`lower`).
            (28, _) => 2,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
    }
    // With a conservative depth output (SV_DepthLessEqual, GreaterEqual) DXIL
    // wants the position centroid (or per sample), as DXC declares it then.
    if pixel && outputs.iter().any(|e| matches!(e.kind, 18 | 19)) {
        for e in inputs.iter_mut().filter(|e| e.kind == 3) {
            e.interp = 5;
        }
    }
    // Per-primitive outputs are constant. The console never exports a
    // primitive id or cull flag a mesh shader writes (it culls with the
    // primitive export), so those rows are left out.
    let mut prims = sig(FourCC::PSG1)?;
    prims.retain(|e| !matches!(e.kind, 10 | 30));
    for e in &mut prims {
        e.interp = 1;
    }
    // A mesh group's id arrives multiplied by max(threads, vertices,
    // primitives) (the code divides by that); the thread count is in PSV0
    // from runtime info version 2 on.
    // shortcut: older containers fall back to the group's lane count, right
    // when the HLSL group fills its waves (Grand Theft Auto VI's do).
    let mesh_stride = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            threads.map_or(ms.threads, |t| {
                t.max(ms.max_vertices).max(ms.max_primitives)
            })
        }
        _ => 0,
    };
    let mut parts = lower::Parts {
        native,
        p,
        st,
        io,
        res,
        inputs,
        outputs,
        prims,
        mesh_stride,
        target,
        console_cf: false,
        per_lane_only: false,
    };
    let l = match lower::lower(&parts) {
        Err(ShaderError::Unsupported { what, .. }) if what == lower::CONSOLE_FLOW => {
            let console = structure::structure_console(p);
            let lowered = match &console {
                Ok(console) => {
                    parts.st = console;
                    parts.console_cf = true;
                    lower::lower(&parts)
                }
                Err(e) => Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match e {
                        ShaderError::Unsupported { what, .. } => what,
                        _ => "a program the console structure cannot hold",
                    },
                }),
            };
            match lowered {
                Ok(l) => l,
                // What the console flow cannot lower keeps the per-lane form
                // where that translates; else the console flow's reason.
                Err(e) => {
                    parts.st = st;
                    parts.console_cf = false;
                    parts.per_lane_only = true;
                    lower::lower(&parts).map_err(|_| e)?
                }
            }
        }
        l => l?,
    };
    Ok(HostShader {
        stage: native.stage,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: l
            .samplers
            .iter()
            .enumerate()
            .map(|(n, &w)| {
                let mut s = StaticSampler::from_words(w);
                if l.cmp_samplers >> n & 1 != 0 {
                    s.filter |= 0x80;
                }
                s
            })
            .collect(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        stride_table: l.stride_table,
    })
}

/// The rows of an `ISG1`/`OSG1` part.
fn signature(part: &[u8]) -> Result<Vec<container::SigElement>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        let mask = *part
            .get(at + 24)
            .ok_or(ShaderError::Malformed("truncated"))?;
        if u32_at(part, at + 28)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a minimum-precision signature element",
            });
        }
        if u32_at(part, at)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a signature element of another stream",
            });
        }
        let kind = match u32_at(part, at + 12)? {
            0 => 0,
            1 => 3,
            2 => 6,
            3 => 7,
            4 => 4,
            5 => 5,
            6 => 1,
            7 => 10,
            8 => 2,
            9 => 13,
            10 => 12,
            64 => 16,
            65 => 17,
            66 => 14,
            67 => 19,
            68 => 18,
            69 => 20,
            70 => 15,
            23 => 28, // barycentrics
            25 => 30, // cull primitive
            _ => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "a system value in the signature",
                });
            }
        };
        // 16-bit elements are declared 32-bit; reading or writing one
        // converts (16-bit values travel packed on the console).
        let raw = u32_at(part, at + 16)?;
        let comp = match (kind, raw) {
            // DXIL declares the front face a uint (the validator insists).
            (13, _) => 5,
            (_, 1 | 4) => 5,
            (_, 2 | 5) => 4,
            (_, 3 | 6) => 9,
            (_, c) => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match c {
                        0 => "a signature element of unknown type",
                        _ => "a 64-bit signature element",
                    },
                });
            }
        };
        let register = u32_at(part, at + 20)?;
        out.push(container::SigElement {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            kind,
            comp,
            interp: 0,
            row: (register != u32::MAX).then_some(register),
            col: mask.trailing_zeros().min(3) as u8,
            cols: mask.count_ones().max(1) as u8,
            usage: 0,
            half: (4..=6).contains(&raw),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::Target;
    use super::bitcode::Attr;
    use super::lower::LOOP_LIMIT;
    use super::parse_loop_limit;

    #[test]
    fn the_loop_limit_switch_parses_or_falls_back() {
        assert_eq!(parse_loop_limit(None), LOOP_LIMIT);
        assert_eq!(parse_loop_limit(Some("4096")), 4096);
        assert_eq!(parse_loop_limit(Some(" 0x1000 ")), 4096);
        assert_eq!(parse_loop_limit(Some("1")), 1);
        for bad in ["", "0", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_limit(Some(bad)), LOOP_LIMIT, "{bad:?}");
        }
    }
    use super::container::SigElement;
    use super::shader::Shader;
    use super::*;
    use crate::native::{ComputeIo, StageIo};

    /// A pixel shader writing SV_StencilRef (channel y of `exp mrtz`).
    /// DXC sets ShaderFlags.StencilRef (bit 11) for a pixel shader whose
    /// output signature has SV_StencilRef, and the feature bit
    /// D3D_SHADER_FEATURE_STENCIL_REF (0x200) in SFI0; the validator checks
    /// both against the module.
    #[test]
    fn a_stencil_export_declares_the_stencil_ref_feature() {
        let words = [
            0x7E00_02F2u32, // v_mov_b32 v0, 1.0
            0x7E02_0285,    // v_mov_b32 v1, 5
            0xF800_1883,
            0x0000_0100, // exp mrtz v0, v1, off, off done vm
            0xBF81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, kind, comp| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp: 0,
            row: None,
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let osg1 =
            container::signature(&[el("SV_Depth", 17, 9), el("SV_StencilRef", 20, 5)], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res, Target::ANY).unwrap();
        // dxil.dll, where there is one, checks the flags against the module.
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
        let host = Container::parse(&out.dxil).unwrap();
        let sfi0 = host.part(FourCC::SFI0).unwrap().data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_ne!(features & 0x200, 0, "SFI0 {features:#x}: no StencilRef");
    }

    /// A wave64 compute program that writes only EXEC's high dword
    /// (`s_mov_b32 exec_hi, 0`) before a vector instruction. The new mask is
    /// opaque and keyed by the unchanged low dword, so building its lane bit
    /// needed the mask's own value: lowering recursed until the stack
    /// overflowed, which takes the whole process down (found by
    /// random-input testing). Now the instruction fails like any other that
    /// cannot be expressed: its result is missing, and only a side effect
    /// that uses it would stop the translation.
    #[test]
    fn an_exec_high_write_does_not_overflow_the_stack() {
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, v0; s_endpgm
        let words = [0xBEFF_0380u32, 0x7E02_0300, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        if let Err(e) = emit(&c, &native, &p, &st, &io, &res, Target::ANY) {
            assert!(e.to_string().contains("its own value"), "{e}");
        }
    }

    #[test]
    fn indexed_move_initializes_an_input_before_its_first_use() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Case {
            Complete,
            NonzeroBase,
            WrongStart,
            SkippedSlot,
            Decrement,
            Wrap,
            UnknownStart,
            ShortLoop,
            ClobberedScc,
            EarlyExit,
            DivergentExit,
            MaskedWrite,
            SourceIsOld,
            ReadBeforeWrite,
            ReadOnBackedge,
        }
        use Case::*;

        // A two-block do-while loop writes v[M0] from v2, then exports v1.
        // On the first two passes M0 selects v0 and v1 respectively. Both
        // registers initially contain raw perspective-center barycentrics.
        let words = |case| {
            let init = match case {
                WrongStart => 0xbe80_0382,   // s_mov_b32 s0, 2
                Wrap => 0xbe80_03c1,         // s_mov_b32 s0, -1
                UnknownStart => 0x8000_8080, // s_add_u32 s0, 0, 0
                _ => 0xbe80_0380,            // s_mov_b32 s0, 0
            };
            let mut w = vec![init, 0x7e04_0281, 0xbf82_0000]; // v_mov_b32 v2, 1; s_branch header
            let latch = w.len();
            if case == EarlyExit {
                w.extend([0xbf06_8000, 0xbf85_0000]); // s_cmp_eq_u32 s0, 0; s_cbranch_scc1 exit
            }
            w.push(0xbe80_0302); // s_mov_b32 s0, s2
            let header = w.len();
            w[2] |= (header - 3) as u32;
            w.push(0xbefc_0300); // s_mov_b32 m0, s0
            w.push(match case {
                SkippedSlot => 0x8102_8200, // s_add_i32 s2, s0, 2
                Decrement => 0x8102_c100,   // s_add_i32 s2, s0, -1
                _ => 0x8102_8100,           // s_add_i32 s2, s0, 1
            });
            if case == ReadBeforeWrite {
                w.push(0x7e06_0301);
            } // v_mov_b32 v3, v1
            if case == MaskedWrite {
                w.extend([0x7d84_0480, 0xbe84_3c6a]); // v_cmp_eq_u32 vcc, 0, v2; s_and_saveexec_b32 s4, vcc_lo
            }
            w.push(match case {
                SourceIsOld => 0x7e00_8501, // v_movreld_b32 v0, v1
                NonzeroBase => 0x7e02_8502, // v_movreld_b32 v1, v2
                _ => 0x7e00_8502,           // v_movreld_b32 v0, v2
            });
            if case == MaskedWrite {
                w.push(0x887e_047e);
            } // s_or_b32 exec_lo, exec_lo, s4
            if case == ReadOnBackedge {
                w.push(0x7e06_0301);
            } // v_mov_b32 v3, v1
            let branch = if case == DivergentExit {
                w.push(0x7d84_0480); // v_cmp_eq_u32 vcc, 0, v2
                0xbf87_0000 // s_cbranch_vccnz latch
            } else {
                w.push(if matches!(case, ShortLoop | NonzeroBase) {
                    0xbf06_8102
                } else {
                    0xbf06_8402
                }); // s_cmp_eq_u32 s2, 1/4
                if case == ClobberedScc {
                    w.push(0xbf06_8000); // s_cmp_eq_u32 s0, 0 exits on the first pass
                }
                0xbf84_0000 // s_cbranch_scc0 latch
            };
            let rel = latch as i32 - (w.len() as i32 + 1);
            w.push(branch | u32::from(rel as u16));
            let exit = w.len();
            if case == EarlyExit {
                w[latch + 1] |= (exit - latch - 2) as u32;
            }
            let output = if matches!(case, ReadBeforeWrite | ReadOnBackedge) {
                3
            } else {
                1
            };
            w.extend([0xf800_1801, output, 0xbf81_0000]); // exp mrt0.x done vm; s_endpgm
            w
        };
        let target = SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let sig = super::container::signature(&[target], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &sig)]);
        let c = Container::parse(&bytes).unwrap();
        for case in [
            Complete,
            NonzeroBase,
            WrongStart,
            SkippedSlot,
            Decrement,
            Wrap,
            UnknownStart,
            ShortLoop,
            ClobberedScc,
            EarlyExit,
            DivergentExit,
            MaskedWrite,
            SourceIsOld,
            ReadBeforeWrite,
            ReadOnBackedge,
        ] {
            let native = NativeProgram {
                stage: HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: StageIo::Ps {
                    inputs: vec![],
                    input_addr: 2,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words(case).iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = structure::structure(&p).unwrap();
            let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let shader = emit(&c, &native, &p, &st, &io, &res, Target::ANY);
            if matches!(case, Complete | NonzeroBase) {
                let shader = shader.unwrap();
                assert!(
                    Container::parse(&shader.dxil)
                        .unwrap()
                        .part(FourCC::DXIL)
                        .is_some()
                );
                #[cfg(windows)]
                if let Ok(mut validator) =
                    validator::Validator::load(validator::Validator::default_path().as_deref())
                {
                    let mut dxil = shader.dxil;
                    validator.validate(&mut dxil).unwrap();
                    eprintln!("synthetic {case:?} SDK valid bytes={}", dxil.len());
                }
            } else {
                eprintln!("synthetic {case:?} {:?}", shader.as_ref().err());
                assert!(
                    matches!(
                        shader,
                        Err(ShaderError::Unsupported {
                            what: "barycentrics read directly",
                            ..
                        })
                    ),
                    "{case:?}"
                );
            }
        }
    }

    /// A pixel shader writing a constant colour.
    #[cfg(windows)]
    fn constant_color() -> Vec<u8> {
        let mut s = Shader::new(true);
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        });
        let b = s.entry;
        for (c, v) in [1.0f32, 0.5, 0.25, 1.0].into_iter().enumerate() {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(c as u8),
                s.m.f32(v.to_bits()),
            ];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, super::bitcode::Term::Ret);
        s.finish()
    }

    #[test]
    #[cfg(windows)]
    fn stub_pipeline_is_created() {
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let ps = super::HostShader {
            stage: crate::HwStage::Ps,
            dxil: constant_color(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            stride_table: false,
        };
        let file = |var: &str, stage| {
            std::env::var(var).ok().map(|p| super::HostShader {
                stage,
                dxil: std::fs::read(p).unwrap(),
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                stride_table: false,
            })
        };
        let ps = file("VICTORY_REF_PS", crate::HwStage::Ps).unwrap_or(ps);
        let vs = file("VICTORY_REF_VS", crate::HwStage::Vs)
            .unwrap_or_else(|| super::host::stub_vs(&ps).unwrap());
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil.clone()).unwrap();
        }
        host.pipeline(&vs, &ps).unwrap();
        // The runtime checks the container hash: no hash or a wrong one is
        // refused. (Recent runtimes also take sixteen 0x01 bytes, a bypass
        // value; ours works with older ones too.)
        let mut wrong = ps.dxil[4..20].to_vec();
        wrong[0] ^= 1;
        for digest in [&[0u8; 16][..], &wrong] {
            let mut other = ps.clone();
            other.dxil[4..20].copy_from_slice(digest);
            assert!(
                host.pipeline(&vs, &other).is_err(),
                "digest {digest:02x?} accepted"
            );
        }
    }

    #[test]
    #[cfg(windows)]
    fn constant_color_validates() {
        let mut c = constant_color();
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        let ours = c[4..20].to_vec();
        v.validate(&mut c).unwrap();
        assert_eq!(c[4..20], ours[..], "our hash differs from the validator's");
    }

    /// A pixel shader interpolating at sample positions (PERSP_SAMPLE
    /// barycentrics, `SPI_PS_INPUT_ADDR` bit 0) runs at sample frequency:
    /// DXC's PSV0 says so in PSInfo.SampleFrequency (runtime info byte 1)
    /// whenever an input is interpolated per sample.
    #[test]
    fn sample_interpolated_inputs_set_psv_sample_frequency() {
        let words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ];
        let psv_of = |input_addr: u32| {
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: false,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half: false,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res =
                crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let out = super::emit(&c, &native, &p, &st, &io, &res, super::Target::ANY).unwrap();
            // dxil.dll, where there is one, checks PSV0 against the module.
            #[cfg(windows)]
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut out.dxil.clone()).unwrap();
            }
            let host = super::Container::parse(&out.dxil).unwrap();
            host.part(super::FourCC::PSV0).unwrap().data.to_vec()
        };
        // Runtime info starts after its size; PSInfo is {DepthOutput, SampleFrequency}.
        let center = psv_of(1 << 1);
        assert_eq!(center[4 + 1], 0, "center interpolation is per pixel");
        let sample = psv_of(1 << 0);
        assert_eq!(sample[4 + 1], 1, "PSInfo.SampleFrequency");
    }

    /// A pixel shader reading channel z of an input whose element starts at
    /// w. Looking the channel up subtracted the element's first channel
    /// before checking the channel lies in it, so in a build with overflow
    /// checks (tests, debug) any read below an element that starts at z or
    /// w panicked, whichever row it was in (found by random-input testing).
    #[test]
    fn reading_a_channel_below_an_input_element_does_not_underflow() {
        // v_interp_mov_f32 v3, p0, attr0.z; s_endpgm
        let words = [0xC80E_0202u32, 0xBF81_0000];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 0,
                    mask: 0b1000,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let w_only = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 3,
            cols: 1,
            usage: 0,
            half: false,
        };
        let isg1 = super::container::signature(&[w_only], true);
        let bytes = crate::dxbc::build(&[(super::FourCC::ISG1, &isg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        // Translated or refused with a reason; it must not panic.
        let _ = super::emit(&c, &native, &p, &st, &io, &res, Target::ANY);
    }

    /// A compute program of 64 threads in one group: `words` as GFX10 code.
    pub(super) fn compute(words: &[u32], wave64: bool) -> crate::NativeProgram {
        compute_of(words, wave64, 64)
    }

    /// [`compute`] with `threads` threads in the group.
    pub(super) fn compute_of(words: &[u32], wave64: bool, threads: u32) -> crate::NativeProgram {
        crate::NativeProgram {
            stage: crate::HwStage::Cs,
            wave64,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Cs(crate::native::ComputeIo {
                rsrc2: 0,
                threads: [threads, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    pub(super) fn emit_for(
        native: &crate::NativeProgram,
        target: Target,
    ) -> Result<super::HostShader, crate::ShaderError> {
        let c = crate::Container {
            digest: [0; 16],
            version: (1, 0),
            parts: vec![],
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs)?;
        let st = crate::structure::structure(&p);
        let res = crate::resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
        let io = crate::io::raise(&p, native)?;
        let st = st?;
        super::emit(&c, native, &p, &st, &io, &res, target)
    }

    const NVIDIA: Target = Target {
        lanes_min: 32,
        lanes_max: 32,
    };
    const RDNA: Target = Target {
        lanes_min: 32,
        lanes_max: 64,
    };

    #[test]
    fn console_lanes_translate_only_for_hosts_with_that_wave_size() {
        let words = [
            0xD760_0000,
            0x0001_0B00, // v_readlane_b32 s0, v0, 5
            0x7E02_0200, // v_mov_b32 v1, s0
            0xBF81_0000, // s_endpgm
        ];
        // One console wave per group: in the console flow on NVIDIA.
        let wave64 = compute(&words, true);
        assert!(emit_for(&wave64, Target::ANY).is_ok());
        assert!(emit_for(&wave64, RDNA).is_ok());
        assert!(emit_for(&wave64, NVIDIA).is_ok());
        // Two per group: the halves cannot meet at a group barrier yet.
        let two = compute_of(&words, true, 128);
        assert!(emit_for(&two, RDNA).is_ok());
        assert_eq!(
            emit_for(&two, NVIDIA),
            Err(crate::ShaderError::Unsupported {
                offset: 0,
                what: "console wave64 lanes in a group of more than 64 threads on a host without 64-lane waves",
            })
        );
        // The same code in waves of 32 runs on either.
        let wave32 = compute(&words, false);
        assert!(emit_for(&wave32, NVIDIA).is_ok());
        assert!(emit_for(&wave32, RDNA).is_ok());
    }

    #[test]
    #[cfg(windows)]
    fn what_translates_for_the_host_creates_a_pipeline() {
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let target = host.target();
        eprintln!("host waves {}-{}", target.lanes_min, target.lanes_max);
        let words = [0xD760_0000, 0x0001_0B00, 0x7E02_0200, 0xBF81_0000];
        for wave64 in [false, true] {
            match emit_for(&compute_of(&words, wave64, 128), target) {
                Ok(cs) => {
                    if let Err(e) = host.compute_pipeline(&cs) {
                        panic!("wave64 {wave64}: translated for the host, refused by it: {e}");
                    }
                }
                Err(e) => {
                    // The host has no waves of the program's size. (The
                    // software adapter of a machine without a GPU has
                    // 4-lane waves: not the wave32 program either.)
                    let wave = if wave64 { 64 } else { 32 };
                    assert!(!target.runs(wave), "{e}");
                    // What it replaces: the host refuses the console's
                    // wave size at pipeline creation.
                    let any = emit_for(&compute_of(&words, wave64, 128), Target::ANY).unwrap();
                    assert!(host.compute_pipeline(&any).is_err());
                }
            }
        }
    }

    /// On every hardware adapter: a wave64 program that reads other lanes
    /// translates in groups of one console wave (natively or in the console
    /// flow) and the adapter takes it; in groups of two it translates
    /// exactly when the adapter runs 64-lane waves. On a machine with an AMD
    /// and an NVIDIA GPU this checks both answers.
    #[test]
    #[cfg(windows)]
    fn each_adapter_gets_what_it_runs() {
        let Ok(names) = super::host::adapters() else {
            eprintln!("no DXGI; skipped");
            return;
        };
        let words = [0xD760_0000, 0x0001_0B00, 0x7E02_0200, 0xBF81_0000];
        for name in names {
            let Ok(host) = super::host::Host::on(&name) else {
                eprintln!("{name}: no SM 6.6 device; skipped");
                continue;
            };
            let target = host.target();
            eprintln!("{name}: waves {}-{}", target.lanes_min, target.lanes_max);
            let one =
                emit_for(&compute(&words, true), target).unwrap_or_else(|e| panic!("{name}: {e}"));
            host.compute_pipeline(&one)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            match emit_for(&compute_of(&words, true, 128), target) {
                Ok(cs) => {
                    assert!(target.runs(64), "{name}");
                    host.compute_pipeline(&cs)
                        .unwrap_or_else(|e| panic!("{name}: {e}"));
                }
                Err(_) => assert!(!target.runs(64), "{name}"),
            }
        }
    }

    #[test]
    fn programs_reading_no_other_lanes_translate_for_any_wave_size() {
        let words = [
            0x7E02_0300, // v_mov_b32 v1, v0
            0xBF81_0000, // s_endpgm
        ];
        assert!(emit_for(&compute(&words, true), NVIDIA).is_ok());
    }
}

#[cfg(test)]
mod barycentric_tests {
    use super::container::SigElement;
    use super::*;
    use crate::native::{PsInput, StageIo, Usage};

    /// `SPI_PS_INPUT_ADDR` bits: PERSP_CENTER, PERSP_CENTROID,
    /// LINEAR_CENTER.
    const PERSP_CENTER: u32 = 1 << 1;
    const PERSP_CENTROID: u32 = 1 << 2;
    const LINEAR_CENTER: u32 = 1 << 5;

    /// `exp mrt0, v<r>, v<r>, v<r>, v<r> done vm`.
    fn exp_mrt0(r: u32) -> [u32; 2] {
        [0xF800_180F, r * 0x0101_0101]
    }

    /// VOP2 `v_add_f32 v<dst>, v<a>, v<b>`.
    fn v_add_f32(dst: u32, a: u32, b: u32) -> u32 {
        3 << 25 | dst << 17 | b << 9 | (256 + a)
    }

    fn element(name: &str, kind: u8, row: u32) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp: 0,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }
    }

    /// Translates a pixel program with the input VGPRs `input_addr`
    /// enables and one float4 TEXCOORD0 attribute (attr 0, row 0).
    fn pixel(words: &[u32], input_addr: u32) -> Result<HostShader, ShaderError> {
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 0,
                    mask: 0b1111,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let mut target = element("SV_Target", 16, 0);
        target.usage = 15;
        let isg1 = container::signature(&[element("TEXCOORD", 0, 0)], true);
        let osg1 = container::signature(&[target], false);
        let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
        let st = structure::structure(&p)?;
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native)?;
        emit(&c, &native, &p, &st, &io, &res, Target::ANY)
    }

    /// The translation's `SFI0` features and `ISG1` rows.
    fn parts(shader: &HostShader) -> (u64, Vec<SigElement>) {
        let c = Container::parse(&shader.dxil).unwrap();
        let sfi0 = c.part(FourCC::SFI0).unwrap().data.try_into().unwrap();
        (
            u64::from_le_bytes(sfi0),
            signature(c.part(FourCC::ISG1).unwrap().data).unwrap(),
        )
    }

    fn validate(shader: &HostShader) {
        #[cfg(windows)]
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut shader.dxil.clone()).unwrap();
        }
        let _ = shader;
    }

    /// Custom interpolation reads I and J (v0, v1 with PERSP_CENTER) as
    /// plain floats: they are channels y and z of `SV_Barycentrics`, the
    /// weights of vertices 1 and 2.
    #[test]
    fn barycentrics_read_as_values_are_sv_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [v_add_f32(2, 0, 1), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        let bary: Vec<_> = inputs.iter().filter(|e| e.kind == 28).collect();
        assert_eq!(bary.len(), 1, "{inputs:?}");
        assert_eq!((bary[0].index, bary[0].col, bary[0].cols), (0, 0, 3));
        // Not packed: the validator wants packing location -1.
        assert_eq!(bary[0].row, None);
        // Only y and z (I and J) are read.
        let isg1 = Container::parse(&shader.dxil).unwrap();
        let isg1 = isg1.part(FourCC::ISG1).unwrap().data;
        let rw: Vec<u8> = (0..2).map(|k| isg1[8 + 32 * k + 25]).collect();
        assert!(rw.contains(&0b110), "{rw:?}");
        validate(&shader);
    }

    /// Two kinds of barycentrics read directly: one element per mode.
    #[test]
    fn two_barycentric_modes_get_two_elements() {
        // v0-v1 perspective i, j; v2-v3 linear i, j.
        let [e0, e1] = exp_mrt0(4);
        let words = [v_add_f32(4, 0, 3), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER | LINEAR_CENTER).unwrap();
        let (_, inputs) = parts(&shader);
        let mut indices: Vec<u32> = inputs
            .iter()
            .filter(|e| e.kind == 28)
            .map(|e| e.index)
            .collect();
        indices.sort();
        assert_eq!(indices, [0, 1], "{inputs:?}");
        validate(&shader);
    }

    /// Center and centroid perspective I/J both read as values: DXIL
    /// allows one perspective `SV_Barycentrics` element (the validator:
    /// "There can only be up to two input attributes of SV_Barycentrics
    /// with different perspective interpolation mode"), so the program is
    /// refused rather than given two, or center weights for centroid ones.
    #[test]
    fn center_and_centroid_barycentrics_are_refused() {
        // v0-v1 perspective center i, j; v2-v3 perspective centroid i, j.
        let [e0, e1] = exp_mrt0(4);
        let words = [v_add_f32(4, 0, 3), e0, e1, 0xBF81_0000];
        match pixel(&words, PERSP_CENTER | PERSP_CENTROID) {
            Err(ShaderError::Unsupported { what, .. }) => {
                assert!(what.contains("two perspective modes"), "{what}")
            }
            Err(e) => panic!("{e:?}"),
            Ok(shader) => panic!("translated: {:?}", parts(&shader).1),
        }
        // Centroid alone (v2, v3) still translates, with one element.
        let words = [v_add_f32(4, 2, 3), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER | PERSP_CENTROID).unwrap();
        let (_, inputs) = parts(&shader);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        validate(&shader);
    }

    /// I and J only fed to `v_interp_*` stay host interpolation: no
    /// barycentrics are declared.
    #[test]
    fn interpolation_alone_declares_no_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_eq!(features & 0x2_0000, 0);
        assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
        validate(&shader);
    }

    /// Both: the attribute is interpolated and J is used as a value.
    #[test]
    fn interpolation_and_a_barycentric_value_together() {
        let [e0, e1] = exp_mrt0(3);
        let words = [
            0xC808_0000,        // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,        // v_interp_p2_f32 v2, v1, attr0.x
            v_add_f32(3, 2, 1), // v3 = attribute + j
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        validate(&shader);
    }
    /// The `PSV0` interpolation mode of TEXCOORD0 (row 0, x-w, float).
    fn texcoord_mode(shader: &HostShader) -> u8 {
        let c = Container::parse(&shader.dxil).unwrap();
        let psv = c.part(FourCC::PSV0).unwrap().data;
        let row = [1, 0, 0x44, 0, 3];
        let at = psv.windows(5).position(|w| w == row).unwrap();
        psv[at + 5]
    }

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// An interpolation whose i is computed (`v_mul_f32 v3, 0.5, v0`) reads
    /// the attribute's vertices: TEXCOORD0 becomes constant, read with
    /// `attributeAtVertex`; i and j are `SV_Barycentrics` values.
    #[test]
    fn computed_barycentrics_read_the_vertices() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0x1006_00F0, // v_mul_f32 v3, 0.5, v0
            0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// Weights from constants alone: the vertices, no `SV_Barycentrics`.
    #[test]
    fn constant_weights_need_no_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0x7E06_02F0, // v_mov_b32 v3, 0.5
            0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
            0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// The same attribute interpolated with the loaded i and j and with a
    /// computed j: both from its vertices, the first weighed with
    /// `SV_Barycentrics`.
    #[test]
    fn computed_and_loaded_barycentrics_on_one_attribute() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC810_0100, // v_interp_p1_f32 v4, v0, attr0.y
            0xC811_0101, // v_interp_p2_f32 v4, v1, attr0.y
            0x1006_02F0, // v_mul_f32 v3, 0.5, v1
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
            v_add_f32(2, 2, 4),
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (_, inputs) = parts(&shader);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// P10 of an attribute the program interpolates is P1 - P0.
    #[test]
    fn p10_of_an_interpolated_attribute() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0xC812_0000, // v_interp_mov_f32 v4, p10, attr0.x
            v_add_f32(2, 2, 4),
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
        // Alone, what the hardware holds is not known.
        let [e0, e1] = exp_mrt0(4);
        let words = [0xC812_0000, e0, e1, S_ENDPGM];
        assert!(pixel(&words, PERSP_CENTER).is_err());
    }

    /// A compute program over a table at `s4` (user data dwords `s0..s3`
    /// before it): a shader view at table dword 0 (and whatever follows up
    /// to `uav`), an unordered access view at table dword `uav`, neither
    /// with a stride known here.
    fn unknown_stride_program(
        words: &[u32],
        uav: u32,
        compute: crate::native::ComputeIo,
    ) -> super::lower::Lowered {
        let used = |kind, table_offset_dwords| crate::UsedResource {
            kind,
            root: crate::RootType::Table,
            table_offset_dwords,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        };
        let native = super::NativeProgram {
            stage: super::HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![
                crate::RootParam {
                    root: crate::RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 4,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                crate::RootParam {
                    root: crate::RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 4,
                    slot: 1,
                },
            ],
            used: vec![
                used(crate::ResourceType::Srv, 0),
                used(crate::ResourceType::Uav, uav),
            ],
            io: crate::native::StageIo::Cs(compute),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let mut l = super::lower::lower(&super::lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            target: super::Target::ANY,
            console_cf: false,
            per_lane_only: false,
        })
        .unwrap();
        assert!(l.stride_table);
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut l.dxil).unwrap();
        }
        l
    }

    /// One thread, user SGPRs `s0..s4`.
    const ONE_THREAD: crate::native::ComputeIo = crate::native::ComputeIo {
        rsrc2: 5 << 1,
        threads: [1, 1, 1],
    };

    #[cfg(windows)]
    fn compute_shader(l: super::lower::Lowered) -> super::HostShader {
        super::HostShader {
            stage: super::HwStage::Cs,
            dxil: l.dxil,
            user_data: l.user_data,
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: l.root_cbvs,
            root_srvs: l.root_srvs,
            root_uavs: l.root_uavs,
            stride_table: l.stride_table,
        }
    }

    /// The hardware adapter and WARP, those there are.
    #[cfg(windows)]
    fn hosts() -> Vec<(&'static str, super::host::Host)> {
        [("the default adapter", false), ("WARP", true)]
            .into_iter()
            .filter_map(|(name, warp)| Some((name, super::host::Host::new(warp).ok()?)))
            .collect()
    }

    /// What the GPU side makes for a buffer view of `count` elements of
    /// `stride` bytes from element `first` of the buffer in heap slot `of`:
    /// the raw view of those bytes (to go in heap slot `raw_slot`) and the
    /// slot's runtime stride table entry.
    #[cfg(windows)]
    fn raw_view_of(
        of: usize,
        raw_slot: u32,
        (first, count, stride): (u32, u32, u32),
    ) -> (super::host::View<'static>, super::StrideEntry) {
        let window = super::RawWindow::covering(
            u64::from(first * stride),
            u64::from(count * stride),
        )
        .unwrap();
        (
            super::host::View::RawAlias {
                of,
                first: window.first as u32,
                dwords: window.dwords,
            },
            super::StrideEntry {
                stride,
                raw_slot: Some(raw_slot),
                bias: window.bias,
            },
        )
    }

    /// A buffer read without a stride known at translation reads the bytes
    /// the console reads, whatever view the title put in the slot: raw,
    /// structured of any stride, or typed. The shader reads the raw view the
    /// runtime stride table names for the slot (the slot itself for a raw
    /// one), never the slot's own view through a handle of another kind,
    /// which the RTX 5070 answers with other dwords than WARP and a Radeon
    /// (121 for the 107 of the structured view of 12 bytes below).
    ///
    /// Checked for every kind on `host`: a load, a store and atomics.
    #[cfg(windows)]
    fn unknown_stride_views_read_their_bytes(name: &str, host: &super::host::Host) {
        use super::StrideEntry;
        use super::host::{Dispatch, Root, Stride, View};
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT;

        let words: &[u32] = &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
            0xF408_0302,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
            0x7E00_0282, // v_mov_b32 v0, 2
            0x7E02_0284, // v_mov_b32 v1, 4
            0xE030_3000,
            0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
            0x7E00_0281, // v_mov_b32 v0, 1
            0x7E02_0288, // v_mov_b32 v1, 8
            0xE070_3000,
            0x8003_0200, // buffer_store_dword v2, v[0:1], s[12:15], 0 idxen offen
            0x7E00_0282, // v_mov_b32 v0, 2
            0x7E02_0280, // v_mov_b32 v1, 0
            0x7E06_028A, // v_mov_b32 v3, 10
            0xE0C8_3000,
            0x8003_0300, // buffer_atomic_add v3, v[0:1], s[12:15], 0 idxen offen
            0xE0C8_3000,
            0x8003_0300, // buffer_atomic_add v3, v[0:1], s[12:15], 0 idxen offen
            0xBF81_0000, // s_endpgm
        ];
        let cs = compute_shader(unknown_stride_program(words, 8, ONE_THREAD));
        let (root, pso) = host.compute_pipeline(&cs).unwrap();
        let data: Vec<u32> = (100..164).collect();
        // The view in the slot, its stride, and the elements it has of the
        // buffer (first, count); none: a raw view, read in its own slot.
        let kinds = [
            ("raw", Stride::Raw, 4, None),
            // (The console's descriptor carries a stride for raw views too.)
            ("raw, stride 12", Stride::Raw, 12, None),
            ("structured 12", Stride::Structured(12), 12, Some((0, 21))),
            ("structured 16", Stride::Structured(16), 16, Some((0, 16))),
            ("structured 20", Stride::Structured(20), 20, Some((0, 12))),
            ("typed RGBA32F", Stride::Typed, 16, Some((0, 16))),
            (
                "typed R32_UINT",
                Stride::Format(DXGI_FORMAT_R32_UINT, 4),
                4,
                Some((0, 64)),
            ),
            // Views starting 4 and 12 bytes into their raw views.
            (
                "structured 12 from element 3",
                Stride::Structured(12),
                12,
                Some((3, 10)),
            ),
            (
                "structured 20 from element 3",
                Stride::Structured(20),
                20,
                Some((3, 6)),
            ),
        ];
        for (kind, view, stride, elements) in kinds {
            // Heap: the shader view, the unordered one, their raw views.
            let mut views = vec![
                View::Buffer {
                    data: &data,
                    stride: view,
                },
                View::BufferUav(64, view),
            ];
            let mut table = Vec::new();
            for slot in 0..2 {
                let entry = match elements {
                    Some((first, count)) => {
                        let (raw, entry) = raw_view_of(slot, 2 + slot as u32, (first, count, stride));
                        views.push(raw);
                        entry
                    }
                    None => StrideEntry {
                        stride,
                        ..StrideEntry::default()
                    },
                };
                table.extend(entry.words());
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(vec![0, 0, 0, 0, 0])),
                            (
                                1,
                                Root::View {
                                    srv: true,
                                    data: table,
                                },
                            ),
                        ],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            let first = elements.map_or(0, |e| e.0);
            let at = |index: u32, off: u32| (((first + index) * stride + off) / 4) as usize;
            // The load of [2, 4], stored at [1, 8]; 10 added twice at [2, 0].
            let mut want = vec![0; 64];
            want[at(1, 8)] = data[at(2, 4)];
            want[at(2, 0)] = 20;
            assert_eq!(out[0], want, "{kind} on {name}");
        }
    }

    #[test]
    fn structured_buffer_unknown_stride_uses_runtime_stride_table() {
        // Translated and validated on every host; run where there is a GPU.
        let _ = unknown_stride_program(
            &[
                0xBE85_0380, // s_mov_b32 s5, 0
                0xF408_0202,
                0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
                0x7E00_0282, // v_mov_b32 v0, 2
                0x7E02_0284, // v_mov_b32 v1, 4
                0xE030_3000,
                0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
                0xBF81_0000, // s_endpgm
            ],
            8,
            ONE_THREAD,
        );
        #[cfg(windows)]
        for (name, host) in hosts() {
            unknown_stride_views_read_their_bytes(name, &host);
        }
    }

    /// The heap slot of an unknown-stride read loaded from a buffer at run
    /// time, per thread group: each group's lanes read the bytes of the
    /// view in the slot their group chose (a structured view of 12 bytes in
    /// one, of 16 in the other), since the table entry is looked up by the
    /// same computed slot the handle is made from. A lane reading past the
    /// chosen view's elements reads zero, not the buffer's next bytes: the
    /// raw view ends where the view does.
    #[test]
    fn unknown_stride_views_chosen_by_a_loaded_heap_index() {
        let words: &[u32] = &[
            0xBE94_0304, // s_mov_b32 s20, s4
            0xBE95_0380, // s_mov_b32 s21, 0
            0xF408_040A,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[20:21], 0x0
            0x8F07_8205, // s_lshl_b32 s7, s5, 2
            0xF420_0188,
            0x0E00_0000, // s_buffer_load_dword s6, s[16:19], s7
            0x8F06_8506, // s_lshl_b32 s6, s6, 5
            0xF408_020A,
            0x0C00_0000, // s_load_dwordx4 s[8:11], s[20:21], s6
            0xF408_030A,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[20:21], 0x20
            0x7E02_0284, // v_mov_b32 v1, 4
            0xE030_3000,
            0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
            0x8F07_8405, // s_lshl_b32 s7, s5, 4
            0x7E02_0207, // v_mov_b32 v1, s7
            0xE070_3000,
            0x8003_0200, // buffer_store_dword v2, v[0:1], s[12:15], 0 idxen offen
            0xBF81_0000, // s_endpgm
        ];
        // Four threads (v0: the thread's id), the group's id in s5 after
        // the user SGPRs s0..s4.
        let l = unknown_stride_program(
            words,
            8,
            crate::native::ComputeIo {
                rsrc2: 5 << 1 | 1 << 7,
                threads: [4, 1, 1],
            },
        );
        #[cfg(not(windows))]
        let _ = l;
        #[cfg(windows)]
        {
            use super::StrideEntry;
            use super::host::{Dispatch, Root, Stride, View};
            let cs = compute_shader(l);
            let (a, b): (Vec<u32>, Vec<u32>) = ((100..164).collect(), (200..264).collect());
            // Slot 2: three elements of 12 bytes; slot 3: four of 16 bytes
            // from element 1; their raw views in slots 4 and 5.
            let (raw_a, entry_a) = raw_view_of(2, 4, (0, 3, 12));
            let (raw_b, entry_b) = raw_view_of(3, 5, (1, 4, 16));
            let raw = StrideEntry {
                stride: 4,
                ..StrideEntry::default()
            };
            let table: Vec<u32> = [raw, raw, entry_a, entry_b]
                .into_iter()
                .flat_map(StrideEntry::words)
                .collect();
            // Lane `l` reads [l, 4]: of the first view dwords 1, 4, 7 and
            // (past its three elements) nothing, of the second 5, 9, 13, 17.
            let of_a = [101, 104, 107, 0];
            let of_b = [205, 209, 213, 217];
            for (name, host) in hosts() {
                let (root, pso) = host.compute_pipeline(&cs).unwrap();
                // The slot each of the two groups reads, in the buffer in
                // slot 0.
                for (slots, want) in [([2, 3], [of_a, of_b]), ([3, 2], [of_b, of_a])] {
                    let out = host
                        .dispatch(
                            &root,
                            &pso,
                            &Dispatch {
                                roots: vec![
                                    (0, Root::Constants(vec![0, 0, 0, 0, 0])),
                                    (
                                        1,
                                        Root::View {
                                            srv: true,
                                            data: table.clone(),
                                        },
                                    ),
                                ],
                                views: &[
                                    View::Buffer {
                                        data: &slots,
                                        stride: Stride::Raw,
                                    },
                                    View::RawUav(8),
                                    View::Buffer {
                                        data: &a,
                                        stride: Stride::Structured(12),
                                    },
                                    View::Buffer {
                                        data: &b,
                                        stride: Stride::Structured(16),
                                    },
                                    raw_a,
                                    raw_b,
                                ],
                                samplers: &[],
                                groups: [2, 1, 1],
                            },
                        )
                        .unwrap();
                    assert_eq!(out[0], want.concat(), "slots {slots:?} on {name}");
                }
            }
        }
    }

    /// The descriptor of an unknown-stride read chosen by control flow: the
    /// table entry read is the chosen slot's.
    #[test]
    fn unknown_stride_descriptor_chosen_by_control_flow() {
        let words: &[u32] = &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBF06_8000, // s_cmp_eq_u32 s0, 0
            0xBF85_0003, // s_cbranch_scc1 -> 0x18
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
            0xBF82_0002, // s_branch -> 0x20
            0xF408_0202,
            0xFA00_0020, // 0x18: s_load_dwordx4 s[8:11], s[4:5], 0x20
            0xF408_0302,
            0xFA00_0040, // 0x20: s_load_dwordx4 s[12:15], s[4:5], 0x40
            0x7E00_0282, // v_mov_b32 v0, 2
            0x7E02_0284, // v_mov_b32 v1, 4
            0xE030_3000,
            0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
            0x7E00_0281, // v_mov_b32 v0, 1
            0x7E02_0288, // v_mov_b32 v1, 8
            0xE070_3000,
            0x8003_0200, // buffer_store_dword v2, v[0:1], s[12:15], 0 idxen offen
            0xBF81_0000, // s_endpgm
        ];
        let l = unknown_stride_program(words, 16, ONE_THREAD);
        #[cfg(not(windows))]
        let _ = l;
        #[cfg(windows)]
        {
            use super::StrideEntry;
            use super::host::{Dispatch, Root, Stride, View};
            let cs = compute_shader(l);
            let (a, b): (Vec<u32>, Vec<u32>) = ((100..164).collect(), (200..264).collect());
            // Slots 0 and 1: structured views of 12 and 16 bytes, their raw
            // views in slots 3 and 4; slot 2: a raw unordered view.
            let (raw_a, entry_a) = raw_view_of(0, 3, (0, 21, 12));
            let (raw_b, entry_b) = raw_view_of(1, 4, (0, 16, 16));
            let raw = StrideEntry {
                stride: 4,
                ..StrideEntry::default()
            };
            let table: Vec<u32> = [entry_a, entry_b, raw]
                .into_iter()
                .flat_map(StrideEntry::words)
                .collect();
            for (name, host) in hosts() {
                let (root, pso) = host.compute_pipeline(&cs).unwrap();
                // s0: 1 takes the first load, 0 the second.
                for (s0, want) in [(1, 107), (0, 209)] {
                    let out = host
                        .dispatch(
                            &root,
                            &pso,
                            &Dispatch {
                                roots: vec![
                                    (0, Root::Constants(vec![s0, 0, 0, 0, 0])),
                                    (
                                        1,
                                        Root::View {
                                            srv: true,
                                            data: table.clone(),
                                        },
                                    ),
                                ],
                                views: &[
                                    View::Buffer {
                                        data: &a,
                                        stride: Stride::Structured(12),
                                    },
                                    View::Buffer {
                                        data: &b,
                                        stride: Stride::Structured(16),
                                    },
                                    View::RawUav(16),
                                    raw_a,
                                    raw_b,
                                ],
                                samplers: &[],
                                groups: [1, 1, 1],
                            },
                        )
                        .unwrap();
                    // Stored at [1, 8] of the raw view: stride 4, dword 3.
                    assert_eq!(out[0][3], want, "s0 = {s0} on {name}");
                }
            }
        }
    }

    #[test]
    fn stride_entries_name_the_raw_view_and_its_bias() {
        use super::{RawWindow, StrideEntry};
        // No raw slot: the slot's own view, whatever the table held before.
        assert_eq!(StrideEntry::default().words(), [0, 0]);
        let own = StrideEntry {
            stride: 4,
            raw_slot: None,
            bias: 0,
        };
        assert_eq!(own.words(), [4, 0]);
        let aliased = StrideEntry {
            stride: 12,
            raw_slot: Some(0),
            bias: 12,
        };
        assert_eq!(aliased.words(), [12, 1 | 12 << 28]);
        // A view on 16 bytes starts its raw view; one 4 to 12 bytes past
        // starts it early, and the raw view covers the bytes before too.
        let window = |first, dwords, bias| {
            Some(RawWindow {
                first,
                dwords,
                bias,
            })
        };
        assert_eq!(RawWindow::covering(32, 24), window(8, 6, 0));
        assert_eq!(RawWindow::covering(36, 24), window(8, 7, 4));
        assert_eq!(RawWindow::covering(44, 22), window(8, 9, 12));
        assert_eq!(RawWindow::covering(0, 0), window(0, 1, 0));
        // Off a dword: no raw view reads it.
        assert_eq!(RawWindow::covering(34, 24), None);
        assert_eq!(RawWindow::covering(16, 4 << 32), None);
    }
}
