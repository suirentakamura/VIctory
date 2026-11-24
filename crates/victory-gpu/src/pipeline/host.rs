//! Host pipelines for title pipelines (decision 0007, step 3): the
//! title's graphics state with the translated shaders, under a root
//! signature of our own that the translated shaders' bindings define
//! (`victory_shader::emit`): per stage one block of root constants for the
//! user data, root views for the title's root views, static samplers for
//! the immediate samplers, and directly indexed heaps.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use victory_shader::emit::{ROOT_CBV_SPACE, SAMPLER_SPACE, STRIDE_SPACE, USER_DATA_SPACE};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use super::{Shader, Translated};
use crate::host::host;

/// A stream's graphics state, owned (the title's stream lives only for the
/// creation call). Subobjects the stream leaves out keep D3D12's defaults.
pub(crate) struct GraphicsState {
    blend: D3D12_BLEND_DESC,
    sample_mask: u32,
    rasterizer: D3D12_RASTERIZER_DESC,
    depth: D3D12_DEPTH_STENCIL_DESC,
    /// Input elements, their names pointing into `names`.
    inputs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    names: Vec<CString>,
    strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
    pub(crate) topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    pub(crate) targets: D3D12_RT_FORMAT_ARRAY,
    depth_format: DXGI_FORMAT,
    samples: DXGI_SAMPLE_DESC,
}

impl Default for GraphicsState {
    fn default() -> Self {
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let keep = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        GraphicsState {
            blend: D3D12_BLEND_DESC {
                RenderTarget: [target; 8],
                ..Default::default()
            },
            sample_mask: u32::MAX,
            rasterizer: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            depth: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                StencilReadMask: 0xff,
                StencilWriteMask: 0xff,
                FrontFace: keep,
                BackFace: keep,
                ..Default::default()
            },
            inputs: Vec::new(),
            names: Vec::new(),
            strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
            targets: D3D12_RT_FORMAT_ARRAY::default(),
            depth_format: DXGI_FORMAT_UNKNOWN,
            samples: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
        }
    }
}

/// Integer render target formats: D3D12 cannot blend into them.
fn integer_format(f: DXGI_FORMAT) -> bool {
    matches!(
        f,
        DXGI_FORMAT_R32G32B32A32_UINT
            | DXGI_FORMAT_R32G32B32A32_SINT
            | DXGI_FORMAT_R32G32B32_UINT
            | DXGI_FORMAT_R32G32B32_SINT
            | DXGI_FORMAT_R16G16B16A16_UINT
            | DXGI_FORMAT_R16G16B16A16_SINT
            | DXGI_FORMAT_R32G32_UINT
            | DXGI_FORMAT_R32G32_SINT
            | DXGI_FORMAT_R10G10B10A2_UINT
            | DXGI_FORMAT_R8G8B8A8_UINT
            | DXGI_FORMAT_R8G8B8A8_SINT
            | DXGI_FORMAT_R16G16_UINT
            | DXGI_FORMAT_R16G16_SINT
            | DXGI_FORMAT_R32_UINT
            | DXGI_FORMAT_R32_SINT
            | DXGI_FORMAT_R8G8_UINT
            | DXGI_FORMAT_R8G8_SINT
            | DXGI_FORMAT_R16_UINT
            | DXGI_FORMAT_R16_SINT
            | DXGI_FORMAT_R8_UINT
            | DXGI_FORMAT_R8_SINT
    )
}

/// The title's blend state as the host takes it: blending off on integer
/// render targets. The console ignores blending there, and titles leave it
/// on (the world sets it on an `R8_UINT` target), but D3D12 does not allow
/// it: vkd3d-proton refuses such a pipeline with `E_INVALIDARG` ("using
/// integer format is not supported"), and its draws are lost. With one
/// description for every target, each target gets its own copy so the
/// others keep theirs.
fn host_blend(blend: &D3D12_BLEND_DESC, targets: &D3D12_RT_FORMAT_ARRAY) -> D3D12_BLEND_DESC {
    let count = (targets.NumRenderTargets as usize).min(8);
    let per_target = blend.IndependentBlendEnable.as_bool();
    let blends = |i: usize| {
        blend.RenderTarget[if per_target { i } else { 0 }]
            .BlendEnable
            .as_bool()
    };
    if !(0..count).any(|i| integer_format(targets.RTFormats[i]) && blends(i)) {
        return *blend;
    }
    let mut host = *blend;
    if !per_target {
        host.IndependentBlendEnable = true.into();
        host.RenderTarget = [blend.RenderTarget[0]; 8];
    }
    for i in 0..count {
        if integer_format(targets.RTFormats[i]) {
            host.RenderTarget[i].BlendEnable = false.into();
        }
    }
    host
}

impl GraphicsState {
    /// Takes in one subobject of a stream: its type, then its data at +4
    /// (+8 for those holding pointers).
    ///
    /// # Safety
    /// `at` points at a whole subobject of type `kind`.
    pub(crate) unsafe fn read(&mut self, kind: u32, at: *const u8) {
        // SAFETY: the caller passes a whole subobject.
        unsafe {
            let data = at.add(4);
            match kind {
                8 => self.blend = data.cast::<D3D12_BLEND_DESC>().read_unaligned(),
                9 => self.sample_mask = data.cast::<u32>().read_unaligned(),
                10 => self.rasterizer = data.cast::<D3D12_RASTERIZER_DESC>().read_unaligned(),
                11 => self.depth = data.cast::<D3D12_DEPTH_STENCIL_DESC>().read_unaligned(),
                21 => {
                    // shortcut: the depth bounds test is dropped (the host
                    // pipeline uses the plain description); bring it back
                    // with a stream if a title enables it.
                    let d = data.cast::<D3D12_DEPTH_STENCIL_DESC1>().read_unaligned();
                    self.depth = D3D12_DEPTH_STENCIL_DESC {
                        DepthEnable: d.DepthEnable,
                        DepthWriteMask: d.DepthWriteMask,
                        DepthFunc: d.DepthFunc,
                        StencilEnable: d.StencilEnable,
                        StencilReadMask: d.StencilReadMask,
                        StencilWriteMask: d.StencilWriteMask,
                        FrontFace: d.FrontFace,
                        BackFace: d.BackFace,
                    };
                }
                12 => {
                    let layout = at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned();
                    if layout.pInputElementDescs.is_null() {
                        return;
                    }
                    let elements = std::slice::from_raw_parts(
                        layout.pInputElementDescs,
                        layout.NumElements as usize,
                    );
                    self.names = elements
                        .iter()
                        .map(|e| CStr::from_ptr(e.SemanticName.0.cast()).to_owned())
                        .collect();
                    self.inputs = elements
                        .iter()
                        .zip(&self.names)
                        .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                            SemanticName: PCSTR(name.as_ptr().cast()),
                            Format: DXGI_FORMAT(
                                crate::resource::pc_format(e.Format.0 as u32) as i32
                            ),
                            ..*e
                        })
                        .collect();
                }
                13 => {
                    self.strip_cut =
                        D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(data.cast::<i32>().read_unaligned())
                }
                14 => {
                    self.topology =
                        D3D12_PRIMITIVE_TOPOLOGY_TYPE(data.cast::<i32>().read_unaligned())
                }
                15 => {
                    self.targets = data.cast::<D3D12_RT_FORMAT_ARRAY>().read_unaligned();
                    for f in &mut self.targets.RTFormats {
                        f.0 = crate::resource::pc_format(f.0 as u32) as i32;
                    }
                }
                16 => {
                    let f = data.cast::<u32>().read_unaligned();
                    self.depth_format = DXGI_FORMAT(crate::resource::pc_format(f) as i32);
                }
                17 => self.samples = data.cast::<DXGI_SAMPLE_DESC>().read_unaligned(),
                _ => {}
            }
        }
    }

    /// Everything the state holds, as text: part of a pipeline's name in
    /// the library and of the key a title pipeline is started early by.
    pub(crate) fn describe(&self) -> String {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .zip(&self.names)
            .map(|(e, name)| {
                format!(
                    "{name:?} {} {} {} {} {} {}",
                    e.SemanticIndex,
                    e.Format.0,
                    e.InputSlot,
                    e.AlignedByteOffset,
                    e.InputSlotClass.0,
                    e.InstanceDataStepRate
                )
            })
            .collect();
        format!(
            "{:?} {} {:?} {:?} {inputs:?} {} {} {:?} {} {:?}",
            self.blend,
            self.sample_mask,
            self.rasterizer,
            self.depth,
            self.strip_cut.0,
            self.topology.0,
            self.targets,
            self.depth_format.0,
            self.samples
        )
    }

    /// The vertex buffer slots the input layout reads.
    fn slots(&self) -> u32 {
        self.inputs
            .iter()
            .fold(0, |m, e| m | 1 << e.InputSlot.min(31))
    }

    /// The input elements with their byte offsets worked out.
    fn elements(&self) -> Vec<Input> {
        let mut next = [0u32; 32];
        self.inputs
            .iter()
            .map(|e| {
                let slot = e.InputSlot.min(31) as usize;
                let format = e.Format.0 as u32;
                let offset = if e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT {
                    next[slot]
                } else {
                    e.AlignedByteOffset
                };
                let bytes = crate::resource::element(format).map_or(0, |(b, _)| b as u32);
                next[slot] = offset + bytes;
                Input {
                    slot: slot as u32,
                    offset,
                    format,
                    per_instance: e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                }
            })
            .collect()
    }
}

/// One input element: vertex buffer slot, byte offset, format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Input {
    pub(crate) slot: u32,
    pub(crate) offset: u32,
    pub(crate) format: u32,
    pub(crate) per_instance: bool,
}

/// The console's own primitive topology types.
const RECTANGLES: i32 = 19;
const QUADS: i32 = 32;

/// Whether root SRVs are bound as one-view tables of the buffer's host
/// copy (reads past its end return zeros) rather than root descriptors
/// (unbounded: a mesh shader's vertex fetch, which has no record limit,
/// read past the copy into memory the host had freed). Set
/// `VICTORY_ROOT_SRV_DIRECT=1` to go back.
pub(crate) fn bounded_root_srvs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ROOT_SRV_DIRECT").is_none())
}

/// Whether draws and dispatches that write root UAVs run. Off unless
/// `VICTORY_ROOT_UAVS=1`: the title's BVH builder writes its nodes through
/// root UAVs in a chain of dispatches, and while some of its stages do not
/// translate (their dispatches dropped), the later ones read node lists
/// nobody wrote and store nodes at offsets taken from them, over title
/// memory past their buffers.
pub(crate) fn root_uavs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ROOT_UAVS").is_some_and(|v| v == "1"))
}

/// One stage of a host pipeline and where its bindings go.
pub(crate) struct Stage {
    pub(crate) shader: Arc<Shader>,
    /// The root parameter of its user data, and its size in dwords.
    pub(crate) user_data: Option<(u32, u32)>,
    /// Root parameters of its root views: (register, ours). The register
    /// names where the title sets the address
    /// ([`victory_shader::emit::RootView`]).
    pub(crate) root_cbvs: Vec<(u16, u32)>,
    pub(crate) root_srvs: Vec<(u16, u32)>,
    /// Root UAVs (raw buffers the shader writes), as `root_srvs`.
    pub(crate) root_uavs: Vec<(u16, u32)>,
    /// Root parameter of the runtime stride table (`t0`, [`STRIDE_SPACE`]).
    pub(crate) stride_table: Option<u32>,
}

impl Stage {
    pub(crate) fn translated(&self) -> &Translated {
        self.shader
            .translated()
            .expect("host pipelines are made from translated shaders")
    }
}

pub(crate) struct HostPipeline {
    /// A compute pipeline (else graphics).
    pub(crate) compute: bool,
    /// A mesh pipeline (graphics without vertex input).
    pub(crate) mesh: bool,
    pub(crate) pso: ID3D12PipelineState,
    pub(crate) root: ID3D12RootSignature,
    /// Vertex, then pixel stage, or the compute stage.
    pub(crate) stages: Vec<Stage>,
    /// The vertex buffer slots it reads (bit per slot).
    pub(crate) slots: u32,
    pub(crate) inputs: Vec<Input>,
    /// Its shaders, for reports of slow submissions.
    pub(crate) name: String,
    /// A rectangle-list pipeline whose geometry stage (ours) completes each
    /// rectangle from its three vertices: its draws run as triangle lists
    /// with the title's own vertex counts.
    pub(crate) rect_gs: bool,
}

/// Logs a pipeline that gets no host pipeline, once per reason.
fn refuse(reason: &'static str, shaders: &[Arc<Shader>]) -> Option<HostPipeline> {
    static SEEN: std::sync::Mutex<Vec<&str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&reason) {
        seen.push(reason);
        tracing::warn!(?shaders, "no host pipeline: {reason} (logged once)");
    }
    None
}

impl HostPipeline {
    /// A host pipeline for vertex and pixel shaders with the graphics
    /// state `g`, or for one compute shader.
    pub(crate) fn new(shaders: &[Arc<Shader>], g: Option<&GraphicsState>) -> Option<HostPipeline> {
        let find = |stage| shaders.iter().find(|s| s.stage == stage);
        let compute = match shaders {
            [cs] if cs.stage == "compute" => Some(cs),
            _ => None,
        };
        if compute.is_none()
            && shaders
                .iter()
                .any(|s| !matches!(s.stage, "vertex" | "pixel" | "mesh"))
        {
            return refuse(
                "only vertex, mesh, pixel and compute stages are translated",
                shaders,
            );
        }
        let mesh = find("mesh");
        let stages = if let Some(cs) = compute {
            vec![(cs, D3D12_SHADER_VISIBILITY_ALL)]
        } else {
            let first = match (mesh, find("vertex")) {
                (Some(ms), _) => (ms, D3D12_SHADER_VISIBILITY_MESH),
                (None, Some(vs)) => (vs, D3D12_SHADER_VISIBILITY_VERTEX),
                (None, None) => return refuse("no vertex or mesh shader", shaders),
            };
            let mut stages = vec![first];
            stages.extend(find("pixel").map(|ps| {
                // A refused pixel shader would drop the geometry with it:
                // draw it with the stand-in instead (Linux branch, on by
                // default; VICTORY_NO_STAND_IN_PIXEL=1 turns it off).
                if crate::pipeline::stand_in_pixels() && ps.translated().is_none() {
                    tracing::info!(pixel = %ps.hash, "a refused pixel shader is drawn with the stand-in");
                    (crate::pipeline::stand_in_pixel(), D3D12_SHADER_VISIBILITY_PIXEL)
                } else {
                    (ps, D3D12_SHADER_VISIBILITY_PIXEL)
                }
            }));
            stages
        };
        if mesh.is_some() && !host()?.mesh_shaders() {
            return refuse(
                "the host GPU has no mesh shaders (or no shader model 6.6)",
                shaders,
            );
        }
        for (s, _) in &stages {
            s.translated()?;
        }
        let host = host()?;

        // The root signature the translated shaders' bindings define.
        let mut params: Vec<D3D12_ROOT_PARAMETER1> = Vec::new();
        // Ranges the table parameters point at, alive until the root
        // signature is made.
        let mut ranges: Vec<Box<D3D12_DESCRIPTOR_RANGE1>> = Vec::new();
        let mut samplers = Vec::new();
        let mut bound = Vec::new();
        for &(shader, visibility) in &stages {
            let t = &shader.translated()?.host;
            let mut stage = Stage {
                shader: shader.clone(),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                stride_table: None,
            };
            let dwords = t.user_data_dwords();
            if dwords > 0 {
                stage.user_data = Some((params.len() as u32, dwords));
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: USER_DATA_SPACE,
                            Num32BitValues: dwords,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (slots, kind, out) in [
                (
                    &t.root_cbvs,
                    D3D12_ROOT_PARAMETER_TYPE_CBV,
                    &mut stage.root_cbvs,
                ),
                (
                    &t.root_srvs,
                    D3D12_ROOT_PARAMETER_TYPE_SRV,
                    &mut stage.root_srvs,
                ),
                (
                    &t.root_uavs,
                    D3D12_ROOT_PARAMETER_TYPE_UAV,
                    &mut stage.root_uavs,
                ),
            ] {
                for &slot in slots {
                    if kind != D3D12_ROOT_PARAMETER_TYPE_CBV && bounded_root_srvs() {
                        // A one-view table: the view's size bounds the
                        // accesses, which a root descriptor leaves unbounded.
                        ranges.push(Box::new(D3D12_DESCRIPTOR_RANGE1 {
                            RangeType: if kind == D3D12_ROOT_PARAMETER_TYPE_UAV {
                                D3D12_DESCRIPTOR_RANGE_TYPE_UAV
                            } else {
                                D3D12_DESCRIPTOR_RANGE_TYPE_SRV
                            },
                            NumDescriptors: 1,
                            BaseShaderRegister: u32::from(slot),
                            RegisterSpace: ROOT_CBV_SPACE,
                            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE
                                | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
                            OffsetInDescriptorsFromTableStart: 0,
                        }));
                        let range: *const D3D12_DESCRIPTOR_RANGE1 = &**ranges.last()?;
                        out.push((slot, params.len() as u32));
                        params.push(D3D12_ROOT_PARAMETER1 {
                            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                                    NumDescriptorRanges: 1,
                                    pDescriptorRanges: range,
                                },
                            },
                            ShaderVisibility: visibility,
                        });
                        continue;
                    }
                    out.push((slot, params.len() as u32));
                    params.push(D3D12_ROOT_PARAMETER1 {
                        ParameterType: kind,
                        Anonymous: D3D12_ROOT_PARAMETER1_0 {
                            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                                ShaderRegister: u32::from(slot),
                                RegisterSpace: ROOT_CBV_SPACE,
                                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                            },
                        },
                        ShaderVisibility: visibility,
                    });
                }
            }
            if t.stride_table {
                stage.stride_table = Some(params.len() as u32);
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: 0,
                            RegisterSpace: STRIDE_SPACE,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (n, s) in t.static_samplers.iter().enumerate() {
                samplers.push(D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER(s.filter as i32),
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE(s.address[0] as i32),
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE(s.address[1] as i32),
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE(s.address[2] as i32),
                    MipLODBias: s.mip_lod_bias,
                    MaxAnisotropy: s.max_anisotropy,
                    ComparisonFunc: D3D12_COMPARISON_FUNC(s.comparison as i32),
                    BorderColor: D3D12_STATIC_BORDER_COLOR(s.border as i32),
                    MinLOD: s.min_lod,
                    MaxLOD: s.max_lod,
                    ShaderRegister: n as u32,
                    RegisterSpace: SAMPLER_SPACE,
                    ShaderVisibility: visibility,
                });
            }
            bound.push(stage);
        }
        let (root, root_blob) =
            match root_signature(&host.device, &params, &samplers, compute.is_some()) {
                Ok(root) => root,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host root signature refused");
                    return None;
                }
            };
        let code = |i: usize| {
            stages.get(i).and_then(|(s, _)| s.translated()).map_or(
                D3D12_SHADER_BYTECODE::default(),
                |t| D3D12_SHADER_BYTECODE {
                    pShaderBytecode: t.host.dxil.as_ptr().cast(),
                    BytecodeLength: t.host.dxil.len(),
                },
            )
        };
        if compute.is_some() {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: a borrowed pointer the description never releases.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                CS: code(0),
                ..Default::default()
            };
            let name = super::library::name("compute", &root_blob, &[&code_bytes(&code(0))], "");
            let made = super::library::pipeline(
                &name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateComputePipelineState(&desc) },
            );
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: true,
                mesh: false,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
                rect_gs: false,
            });
        }
        let Some(g) = g else {
            return refuse("graphics state not kept (plain description)", shaders);
        };
        let state = g.describe();
        if mesh.is_some() {
            let name = super::library::name(
                "mesh",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1))],
                &state,
            );
            let pso = match mesh_pipeline(&host.device, &root, code(0), code(1), g, &name) {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host mesh pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host mesh pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: false,
                mesh: true,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
                rect_gs: false,
            });
        }

        // A rectangle list whose vertex stage reads no vertex buffer (it
        // places corners by vertex number): the fourth corner cannot come
        // from expanded buffers, so a geometry stage of ours adds it.
        let gs = (g.topology.0 == RECTANGLES && g.elements().iter().all(|e| e.per_instance))
            .then(|| rect_gs(&code_bytes(&code(0))))
            .flatten();
        let mut raster = g.rasterizer;
        if gs.is_some() {
            // Which way a rectangle's three vertices wind means nothing.
            raster.CullMode = D3D12_CULL_MODE_NONE;
        }
        let gs_code = gs.as_ref().map_or(D3D12_SHADER_BYTECODE::default(), |b| {
            D3D12_SHADER_BYTECODE {
                pShaderBytecode: b.as_ptr().cast(),
                BytecodeLength: b.len(),
            }
        });
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: code(0),
            GS: gs_code,
            PS: code(1),
            BlendState: host_blend(&g.blend, &g.targets),
            SampleMask: g.sample_mask,
            RasterizerState: raster,
            DepthStencilState: g.depth,
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: g.inputs.as_ptr(),
                NumElements: g.inputs.len() as u32,
            },
            IBStripCutValue: g.strip_cut,
            // The console's rectangle and quad types draw as triangles on
            // the host (see `draw`).
            PrimitiveTopologyType: match g.topology.0 {
                RECTANGLES | QUADS => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                _ => g.topology,
            },
            NumRenderTargets: g.targets.NumRenderTargets,
            RTVFormats: g.targets.RTFormats,
            DSVFormat: g.depth_format,
            SampleDesc: g.samples,
            ..Default::default()
        };
        let name = super::library::name(
            "graphics",
            &root_blob,
            &[
                &code_bytes(&code(0)),
                &code_bytes(&code(1)),
                gs.as_deref().unwrap_or(&[]),
            ],
            &state,
        );
        let made = super::library::pipeline(
            &name,
            // SAFETY: the description points at live data.
            |library, name| unsafe { library.LoadGraphicsPipeline(name, &desc) },
            // SAFETY: as above.
            || unsafe { host.device.CreateGraphicsPipelineState(&desc) },
        );
        let pso = match made {
            Ok(pso) => pso,
            Err(e) => {
                tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                host.log_messages();
                return None;
            }
        };
        tracing::info!(?shaders, "host pipeline");
        crate::host::name(&pso, || format!("{shaders:?}"));
        tracing::debug!(
            ?shaders,
            depth = ?g.depth,
            dsv = g.depth_format.0,
            targets = ?&g.targets.RTFormats[..g.targets.NumRenderTargets.min(8) as usize],
            blend = ?g.blend.RenderTarget[0],
            cull = g.rasterizer.CullMode.0,
            "host graphics state"
        );
        Some(HostPipeline {
            compute: false,
            mesh: false,
            pso,
            root,
            stages: bound,
            name: format!("{shaders:?}"),
            slots: g.slots(),
            inputs: g.elements(),
            rect_gs: gs.is_some(),
        })
    }
}

/// A mesh pipeline from the title's graphics state: the pipeline-state
/// stream form (no input layout or topology).
fn mesh_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    ms: D3D12_SHADER_BYTECODE,
    ps: D3D12_SHADER_BYTECODE,
    g: &GraphicsState,
    name: &str,
) -> windows::core::Result<ID3D12PipelineState> {
    use windows::core::Interface;
    /// A stream subobject: its type, then its description, pointer-aligned.
    #[repr(C, align(8))]
    struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);
    #[repr(C)]
    struct Stream {
        root: Sub<*mut std::ffi::c_void>,
        ms: Sub<D3D12_SHADER_BYTECODE>,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        blend: Sub<D3D12_BLEND_DESC>,
        mask: Sub<u32>,
        raster: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
        dsv: Sub<DXGI_FORMAT>,
        samples: Sub<DXGI_SAMPLE_DESC>,
    }
    let mut stream = Stream {
        root: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
            root.as_raw(),
        ),
        ms: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms),
        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
        blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, host_blend(&g.blend, &g.targets)),
        mask: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, g.sample_mask),
        raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer),
        depth: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, g.depth),
        targets: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
            g.targets,
        ),
        dsv: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
            g.depth_format,
        ),
        samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples),
    };
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: size_of::<Stream>(),
        pPipelineStateSubobjectStream: (&raw mut stream).cast(),
    };
    let device = device.cast::<ID3D12Device2>()?;
    super::library::pipeline(
        name,
        // SAFETY: the stream holds whole subobjects pointing at live data.
        |library, name| unsafe { library.LoadPipeline(name, &desc) },
        // SAFETY: as above.
        || unsafe { device.CreatePipelineState(&desc) },
    )
}

/// The bytes of shader code (none for an absent stage).
fn code_bytes(code: &D3D12_SHADER_BYTECODE) -> Vec<u8> {
    if code.pShaderBytecode.is_null() {
        return Vec::new();
    }
    // SAFETY: a translation's own bytes, alive while its shader is.
    unsafe { std::slice::from_raw_parts(code.pShaderBytecode.cast::<u8>(), code.BytecodeLength) }
        .to_vec()
}

pub(crate) fn root_signature(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER1],
    samplers: &[D3D12_STATIC_SAMPLER_DESC],
    compute: bool,
) -> windows::core::Result<(ID3D12RootSignature, Vec<u8>)> {
    let input = if compute {
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    } else {
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    };
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: windows::Win32::Graphics::Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: input
                    | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                    | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
            },
        },
    };
    // SAFETY: the description points at live arrays; the blob is read in
    // full before it is released.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        if let Err(e) = D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)) {
            if let Some(error) = error {
                let text = std::slice::from_raw_parts(
                    error.GetBufferPointer().cast::<u8>(),
                    error.GetBufferSize(),
                );
                tracing::warn!("root signature: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        Ok((device.CreateRootSignature(0, bytes)?, bytes.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn targets(formats: &[DXGI_FORMAT]) -> D3D12_RT_FORMAT_ARRAY {
        let mut t = D3D12_RT_FORMAT_ARRAY {
            NumRenderTargets: formats.len() as u32,
            ..Default::default()
        };
        t.RTFormats[..formats.len()].copy_from_slice(formats);
        t
    }

    fn blending(independent: bool) -> D3D12_BLEND_DESC {
        let mut b = GraphicsState::default().blend;
        b.IndependentBlendEnable = independent.into();
        for t in &mut b.RenderTarget {
            t.BlendEnable = true.into();
            t.SrcBlend = D3D12_BLEND_SRC_ALPHA;
            t.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        }
        b
    }

    #[test]
    fn integer_targets_do_not_blend() {
        let t = targets(&[
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            DXGI_FORMAT_R8_UINT,
        ]);
        for independent in [false, true] {
            let host = host_blend(&blending(independent), &t);
            assert!(host.IndependentBlendEnable.as_bool());
            assert!(host.RenderTarget[0].BlendEnable.as_bool());
            assert!(host.RenderTarget[1].BlendEnable.as_bool());
            assert!(!host.RenderTarget[2].BlendEnable.as_bool());
            // The rest of the target's state is the title's.
            assert_eq!(host.RenderTarget[2].SrcBlend, D3D12_BLEND_SRC_ALPHA);
            assert_eq!(host.RenderTarget[0].DestBlend, D3D12_BLEND_INV_SRC_ALPHA);
        }
    }

    #[test]
    fn blend_states_without_integer_blending_are_kept() {
        // No integer target.
        let b = blending(false);
        let t = targets(&[DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R11G11B10_FLOAT]);
        assert_eq!(host_blend(&b, &t), b);
        // An integer target that does not blend.
        let mut b = blending(true);
        b.RenderTarget[1].BlendEnable = false.into();
        let t = targets(&[DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32_UINT]);
        assert_eq!(host_blend(&b, &t), b);
        // Integer formats past the targets in use.
        let mut t = targets(&[DXGI_FORMAT_R8G8B8A8_UNORM]);
        t.RTFormats[1] = DXGI_FORMAT_R8_UINT;
        assert_eq!(host_blend(&blending(false), &t), blending(false));
    }
}

/// The geometry stage completing rectangles for a vertex stage with output
/// signature as in `vs` (a translated shader's container): three vertices
/// in, the rectangle they span out as a strip. `None` if the signature has
/// no position, an element our source cannot name, or the compiler lays the
/// elements out in other registers than the vertex stage's.
///
/// The console infers a rectangle's fourth corner from three. Which of the
/// three sits between the other two differs by convention, so the corner
/// opposite the longest side is taken as the middle one: the fourth is the
/// other two's sum less it, for every interpolated value.
fn rect_gs(vs: &[u8]) -> Option<Vec<u8>> {
    static MADE: std::sync::Mutex<Option<std::collections::HashMap<String, Option<Vec<u8>>>>> =
        std::sync::Mutex::new(None);
    let outputs = signature(vs, b"OSG1")?;
    let mut fields = String::new();
    let mut fourth = String::new();
    let mut position = None;
    for (n, e) in outputs.iter().enumerate() {
        let comps = e.mask.count_ones();
        // Elements start at x and are contiguous in what we emit.
        if comps == 0 || u32::from(e.mask) != (1u32 << comps) - 1 {
            return None;
        }
        let ty = match e.kind {
            1 => "uint",
            2 => "int",
            3 => "float",
            _ => return None,
        };
        if !e.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return None;
        }
        let system = e.name.to_ascii_uppercase().starts_with("SV_");
        let semantic = if system && e.index == 0 {
            e.name.clone()
        } else {
            format!("{}{}", e.name, e.index)
        };
        let interp = if e.kind == 3 { "" } else { "nointerpolation " };
        fields.push_str(&format!("    {interp}{ty}{comps} e{n} : {semantic};\n"));
        if e.kind == 3 {
            fourth.push_str(&format!("    d.e{n} = i[s].e{n} + i[t].e{n} - i[r].e{n};\n"));
        } else {
            fourth.push_str(&format!("    d.e{n} = i[r].e{n};\n"));
        }
        if e.name.eq_ignore_ascii_case("SV_Position") && comps == 4 {
            position = Some(n);
        }
    }
    let p = position?;
    let source = format!(
        "struct V {{\n{fields}}};\n\
         [maxvertexcount(4)]\n\
         void main(triangle V i[3], inout TriangleStream<V> o) {{\n\
         \x20   float2 a = i[0].e{p}.xy / i[0].e{p}.w;\n\
         \x20   float2 b = i[1].e{p}.xy / i[1].e{p}.w;\n\
         \x20   float2 c = i[2].e{p}.xy / i[2].e{p}.w;\n\
         \x20   float d01 = dot(a - b, a - b), d02 = dot(a - c, a - c), d12 = dot(b - c, b - c);\n\
         \x20   uint r = (d12 >= d01 && d12 >= d02) ? 0 : ((d02 >= d01) ? 1 : 2);\n\
         \x20   uint s = (r + 1) % 3, t = (r + 2) % 3;\n\
         \x20   V d;\n{fourth}\
         \x20   o.Append(i[r]); o.Append(i[s]); o.Append(i[t]); o.Append(d);\n\
         }}\n"
    );
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    let made = made.get_or_insert_with(std::collections::HashMap::new);
    if let Some(done) = made.get(&source) {
        return done.clone();
    }
    // The host takes no pipeline mixing shader models below 6 with ours,
    // so the stage is compiled to DXIL.
    let bytes = match dxc_compile(&source, "gs_6_0") {
        Ok(bytes) => {
            // The stage must hand the pixel stage what the vertex stage did,
            // in the same registers.
            let same = signature(&bytes, b"OSG1").is_some_and(|o| {
                o.len() == outputs.len()
                    && o.iter().zip(&outputs).all(|(x, y)| {
                        (x.register, x.mask, x.index) == (y.register, y.mask, y.index)
                            && x.name.eq_ignore_ascii_case(&y.name)
                    })
            });
            if !same {
                tracing::warn!("rectangle geometry stage: registers differ from the vertex stage's");
            }
            same.then_some(bytes)
        }
        Err(e) => {
            tracing::warn!(error = %e, "rectangle geometry stage not compiled");
            None
        }
    };
    if bytes.is_some() {
        tracing::info!(elements = outputs.len(), "a rectangle geometry stage made");
    }
    made.insert(source, bytes.clone());
    bytes
}

/// One element of a shader container's signature part.
struct Element {
    name: String,
    index: u32,
    kind: u32,
    register: u32,
    mask: u8,
}

/// The elements of signature part `part` of shader container `code`: the
/// 32-byte records of `ISG1`/`OSG1`, the 24-byte ones of `ISGN`/`OSGN`, the
/// 28-byte ones of `OSG5`.
fn signature(code: &[u8], part: &[u8; 4]) -> Option<Vec<Element>> {
    let word = |b: &[u8], at: usize| {
        b.get(at..at + 4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as usize)
    };
    if code.get(..4)? != b"DXBC" {
        return None;
    }
    let parts = word(code, 28)?;
    for n in 0..parts {
        let at = word(code, 32 + n * 4)?;
        if code.get(at..at + 4)? != part {
            continue;
        }
        let size = word(code, at + 4)?;
        let d = code.get(at + 8..at + 8 + size)?;
        let count = word(d, 0)?;
        // Records: with a stream first (`OSG5`, `xSG1`), and a minimum
        // precision last (`xSG1`).
        let (record, skip) = match part {
            b"ISG1" | b"OSG1" => (32, 4),
            b"OSG5" => (28, 4),
            _ => (24, 0),
        };
        let mut out = Vec::new();
        for k in 0..count {
            let e = 8 + k * record + skip;
            let name_at = word(d, e)?;
            let end = d.get(name_at..)?.iter().position(|&b| b == 0)?;
            out.push(Element {
                name: String::from_utf8_lossy(&d[name_at..name_at + end]).into_owned(),
                index: word(d, e + 4)? as u32,
                kind: word(d, e + 12)? as u32,
                register: word(d, e + 16)? as u32,
                mask: *d.get(e + 20)?,
            });
        }
        return Some(out);
    }
    None
}

/// Compiles HLSL `source` (entry `main`) for `target` with the Windows
/// SDK's `dxcompiler.dll` (`VICTORY_DXC` names another), signed by the
/// `dxil.dll` beside it.
///
/// shortcut: needs the Windows SDK on the machine; without it rectangle
/// lists drawn by vertex number stay unexpanded.
fn dxc_compile(source: &str, target: &str) -> Result<Vec<u8>, String> {
    use windows::Win32::Graphics::Direct3D::Dxc::{
        CLSID_DxcCompiler, DXC_CP_UTF8, DxcBuffer, IDxcBlob, IDxcBlobEncoding, IDxcCompiler3,
        IDxcResult,
    };
    use windows::core::{GUID, HRESULT, Interface, PCWSTR};
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
    type Create = unsafe extern "system" fn(
        *const GUID,
        *const GUID,
        *mut *mut std::ffi::c_void,
    ) -> HRESULT;
    static CREATE: std::sync::OnceLock<Result<usize, String>> = std::sync::OnceLock::new();
    let create = CREATE
        .get_or_init(|| {
            let path = match std::env::var_os("VICTORY_DXC") {
                Some(p) => std::path::PathBuf::from(p),
                None => {
                    let kits = std::path::PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10\bin");
                    let mut versions: Vec<std::path::PathBuf> = std::fs::read_dir(&kits)
                        .map_err(|e| format!("no Windows SDK: {e}"))?
                        .filter_map(Result::ok)
                        .map(|e| e.path().join(r"x64\dxcompiler.dll"))
                        .filter(|p| p.exists())
                        .collect();
                    versions.sort();
                    versions.pop().ok_or("no dxcompiler.dll in the Windows SDK")?
                }
            };
            let wide = |p: &std::path::Path| -> Vec<u16> {
                p.as_os_str().to_string_lossy().encode_utf16().chain([0]).collect()
            };
            // The signer first, by path: the compiler then finds it loaded.
            let signer = wide(&path.with_file_name("dxil.dll"));
            let compiler = wide(&path);
            // SAFETY: plain Win32 calls with NUL-terminated names.
            unsafe {
                LoadLibraryW(signer.as_ptr());
                let module = LoadLibraryW(compiler.as_ptr());
                if module.is_null() {
                    return Err(format!("cannot load {}", path.display()));
                }
                GetProcAddress(module, c"DxcCreateInstance".as_ptr().cast())
                    .map(|f| f as usize)
                    .ok_or_else(|| "dxcompiler.dll lacks DxcCreateInstance".to_string())
            }
        })
        .clone()?;
    // SAFETY: the export has this signature.
    let create: Create = unsafe { std::mem::transmute(create) };
    let mut raw = std::ptr::null_mut();
    // SAFETY: a valid class and interface; the out pointer is a local.
    unsafe { create(&CLSID_DxcCompiler, &IDxcCompiler3::IID, &mut raw) }
        .ok()
        .map_err(|e| e.to_string())?;
    // SAFETY: the object `create` returned, owned from here.
    let compiler = unsafe { IDxcCompiler3::from_raw(raw) };
    let buffer = DxcBuffer {
        Ptr: source.as_ptr().cast(),
        Size: source.len(),
        Encoding: DXC_CP_UTF8.0,
    };
    let wide = |s: &str| -> Vec<u16> { s.encode_utf16().chain([0]).collect() };
    let words = [wide("-E"), wide("main"), wide("-T"), wide(target)];
    let args: Vec<PCWSTR> = words.iter().map(|w| PCWSTR(w.as_ptr())).collect();
    // SAFETY: the buffer and arguments live through the call.
    let result: IDxcResult =
        unsafe { compiler.Compile(&buffer, Some(&args), None) }.map_err(|e| e.to_string())?;
    // SAFETY: a live result.
    let status = unsafe { result.GetStatus() }.map_err(|e| e.to_string())?;
    if status.is_err() {
        // SAFETY: a live result; the blob's own bytes.
        let text = unsafe {
            result.GetErrorBuffer().ok().map(|b: IDxcBlobEncoding| {
                String::from_utf8_lossy(std::slice::from_raw_parts(
                    b.GetBufferPointer().cast::<u8>(),
                    b.GetBufferSize(),
                ))
                .into_owned()
            })
        };
        return Err(format!("{status:?}: {}", text.unwrap_or_default()));
    }
    // SAFETY: a live result; the blob's own bytes.
    unsafe {
        let blob: IDxcBlob = result.GetResult().map_err(|e| e.to_string())?;
        Ok(std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()).to_vec())
    }
}


#[cfg(test)]
mod rect_tests {
    use super::*;
    use windows::core::Interface;
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
    };

    /// Vertex stages placing a rectangle's corners by vertex number, as
    /// the title's full-screen and tiled passes do. In the first the second
    /// vertex sits between the other two (top left, bottom left, bottom
    /// right); in the second the first vertex does. The stage does not
    /// depend on which.
    const VS: [&str; 2] = [
        "float4 main(uint id : SV_VertexID) : SV_Position {
            float x = (id % 3) == 2 ? 1.0 : -1.0;
            float y = (id % 3) == 0 ? 1.0 : -1.0;
            return float4(x, y, 0.5, 1.0);
        }",
        "float4 main(uint id : SV_VertexID) : SV_Position {
            float x = (id % 3) == 2 ? 1.0 : -1.0;
            float y = (id % 3) == 1 ? -1.0 : 1.0;
            return float4(x, y, 0.5, 1.0);
        }",
    ];
    const PS: &str = "float4 main() : SV_Target { return float4(1, 1, 1, 1); }";
    const SIDE: u32 = 8;

    /// The signature parser reads what the compiler wrote.
    #[test]
    fn signature_of_a_compiled_vertex_stage() {
        let Ok(vs) = dxc_compile(VS[0], "vs_6_0") else {
            return; // no compiler on this machine
        };
        let outputs = signature(&vs, b"OSG1").unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].name.eq_ignore_ascii_case("SV_Position"));
        assert_eq!((outputs[0].register, outputs[0].mask, outputs[0].kind), (0, 0xf, 3));
        let inputs = signature(&vs, b"ISG1").unwrap();
        assert!(inputs[0].name.eq_ignore_ascii_case("SV_VertexID"));
    }

    /// Three vertices placed by number draw the whole rectangle through the
    /// geometry stage, and only the triangle they span without it.
    #[test]
    fn rectangles_placed_by_vertex_number_are_completed() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        for source in VS {
            let (Ok(vs), Ok(ps)) = (dxc_compile(source, "vs_6_0"), dxc_compile(PS, "ps_6_0"))
            else {
                return; // no compiler on this machine
            };
            let gs = rect_gs(&vs).expect("a geometry stage for a position-only vertex stage");
            let with = covered(&host.device, &vs, Some(&gs), &ps);
            let without = covered(&host.device, &vs, None, &ps);
            assert_eq!(with, SIDE * SIDE, "the rectangle covers the target");
            assert!(
                without < SIDE * SIDE && without > 0,
                "three vertices alone span a triangle: {without} pixels"
            );
        }
    }

    /// How many pixels of a `SIDE`-square target one draw of three vertices
    /// covers with these stages.
    fn covered(device: &ID3D12Device, vs: &[u8], gs: Option<&[u8]>, ps: &[u8]) -> u32 {
        let code = |b: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: b.as_ptr().cast(),
            BytecodeLength: b.len(),
        };
        // SAFETY: plain D3D12 calls on objects made here; every pointer
        // names live local data.
        unsafe {
            let mut blob = None;
            D3D12SerializeVersionedRootSignature(
                &D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
                    Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
                    Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                        Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1::default(),
                    },
                },
                &mut blob,
                None,
            )
            .unwrap();
            let blob: ID3DBlob = blob.unwrap();
            let root: ID3D12RootSignature = device
                .CreateRootSignature(
                    0,
                    std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()),
                )
                .unwrap();
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(vs),
                GS: gs.map_or(D3D12_SHADER_BYTECODE::default(), code),
                PS: code(ps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                DSVFormat: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xf;
            let pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc).unwrap();

            let texture = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(SIDE),
                Height: SIDE,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let mut target: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &texture,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut target,
                )
                .unwrap();
            let target = target.unwrap();
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut total = 0;
            device.GetCopyableFootprints(&texture, 0, 1, 0, Some(&mut footprint), None, None, Some(&mut total));
            let mut readback: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_READBACK,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: total,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
            let readback = readback.unwrap();
            let views: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .unwrap();
            let view = views.GetCPUDescriptorHandleForHeapStart();
            device.CreateRenderTargetView(&target, None, view);

            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, &pso)
                .unwrap();
            list.SetGraphicsRootSignature(&root);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: SIDE as f32,
                Height: SIDE as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: SIDE as i32,
                bottom: SIDE as i32,
            }]);
            list.OMSetRenderTargets(1, Some(&view), false, None);
            list.ClearRenderTargetView(view, &[0.0; 4], None);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.DrawInstanced(3, 1, 0, 0);
            list.ResourceBarrier(&[crate::host::transition(
                &target,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.CopyTextureRegion(
                &D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&readback),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: footprint,
                    },
                },
                0,
                0,
                0,
                &D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&target),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                },
                None,
            );
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            let started = std::time::Instant::now();
            while fence.GetCompletedValue() < 1 {
                assert!(started.elapsed().as_secs() < 30, "the draw did not finish");
                std::thread::yield_now();
            }
            let mut at = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut at)).unwrap();
            let pitch = footprint.Footprint.RowPitch as usize;
            let mut covered = 0;
            for y in 0..SIDE as usize {
                for x in 0..SIDE as usize {
                    if *at.cast::<u8>().add(y * pitch + x * 4) == 255 {
                        covered += 1;
                    }
                }
            }
            readback.Unmap(0, None);
            covered
        }
    }
}
