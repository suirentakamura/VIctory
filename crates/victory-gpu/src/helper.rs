//! Our own small compute shaders (the indirect argument clamp, the depth
//! conversion), compiled at run time by the system's HLSL compiler.
//!
//! Their sources keep to what every system compiler accepts: Wine's
//! (vkd3d-shader) reads writable buffers only by index, so they use
//! `RWStructuredBuffer<uint>` rather than `RWByteAddressBuffer.Load`, and
//! it ignores `[RootSignature]`, so the root signature is built here
//! rather than taken from the shader.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::PCSTR;

use crate::host::host;

pub(crate) struct Compute {
    pub(crate) root: ID3D12RootSignature,
    pub(crate) pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Compute {}
// SAFETY: as above.
unsafe impl Sync for Compute {}

fn text(blob: &ID3DBlob) -> String {
    // SAFETY: the blob's own bytes.
    let bytes = unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()) };
    String::from_utf8_lossy(bytes).into_owned()
}

/// A root view after the constants: a read-only (`Srv`) or writable (`Uav`)
/// buffer at the given shader register.
#[derive(Clone, Copy)]
pub(crate) enum Slot {
    Srv(u32),
    Uav(u32),
}

/// The `main` of `source` as a 5.1 compute shader, with root parameter 0
/// `constants` 32-bit constants at `b0` and parameters 1.. root unordered
/// views `u0`.. (`uavs` of them).
pub(crate) fn compute(name: PCSTR, source: &str, constants: u32, uavs: u32) -> windows::core::Result<Compute> {
    let slots: Vec<Slot> = (0..uavs).map(Slot::Uav).collect();
    compute_with(name, source, constants, &slots)
}

/// As [`compute`], with root parameters 1.. the root views `slots` name, in
/// that order.
pub(crate) fn compute_with(name: PCSTR, source: &str, constants: u32, slots: &[Slot]) -> windows::core::Result<Compute> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            name,
            None,
            None,
            windows::core::s!("main"),
            windows::core::s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: a static C string.
            let name = unsafe { name.to_string() }.unwrap_or_default();
            tracing::warn!("{name}: {}", text(&errors));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    let mut params = vec![D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: constants },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }];
    params.extend(slots.iter().map(|slot| {
        let (kind, register) = match *slot {
            Slot::Srv(r) => (D3D12_ROOT_PARAMETER_TYPE_SRV, r),
            Slot::Uav(r) => (D3D12_ROOT_PARAMETER_TYPE_UAV, r),
        };
        D3D12_ROOT_PARAMETER {
            ParameterType: kind,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: register, RegisterSpace: 0 },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        }
    }));
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        ..Default::default()
    };
    // SAFETY: the description points at live arrays; blobs are read in full
    // before they are released.
    unsafe {
        let mut signature: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        if let Err(e) = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut signature, Some(&mut errors)) {
            if let Some(errors) = errors {
                tracing::warn!("root signature: {}", text(&errors));
            }
            return Err(e);
        }
        let signature = signature.ok_or_else(windows::core::Error::empty)?;
        let signature = std::slice::from_raw_parts(signature.GetBufferPointer().cast::<u8>(), signature.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, signature)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.GetBufferPointer(), BytecodeLength: code.GetBufferSize() },
            ..Default::default()
        })?;
        Ok(Compute { root, pso })
    }
}
