//! End-to-end tests of synthetic GFX10 image samples lowered to DXIL,
//! validated by `dxil.dll` when present, and run on the host GPU and WARP:
//! address layouts (a16 packing, gradients, clamps) and texel offsets.

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// Inline constant 0 as an operand.
const ZERO: u16 = 128;
/// A literal dword follows the instruction.
const LITERAL: u16 = 255;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u16) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32 & 0xFF
}

/// SMEM `s_load_dword{x4,x8} s[dst:], s[base:base+1], offset` (`op` 2 or 3).
const fn s_load(op: u32, dst: u8, base: u8, offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | op << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25 | offset,
    ]
}

/// VOP1 `v_mov_b32 vdst, src0` (`src0`: 0..105 SGPR, 128.. inline, 256.. VGPR).
const fn v_mov(vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | 0x01 << 9 | (src0 as u32 & 0x1FF)
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

/// MIMG `op v[vdata], v[vaddr:], s[16:23], s[24:27] dmask:0x1 dim:2D`.
const fn mimg(op: u32, vdata: u8, vaddr: u8, a16: bool) -> [u32; 2] {
    [
        0x3C << 26 | op << 18 | 1 << 8 | 1 << 3,
        vaddr as u32 | (vdata as u32) << 8 | (16 >> 2) << 16 | (24 >> 2) << 21 | (a16 as u32) << 30,
    ]
}

const IMAGE_SAMPLE_D_CL: u32 = 0x23;
const IMAGE_SAMPLE_L_O: u32 = 0x34;
const IMAGE_SAMPLE_LZ_O: u32 = 0x37;

/// A 1-thread compute program: root constants in `s0..s3`, the view table
/// address in `s4`, the sampler table's in `s5`. It loads the UAV (view
/// 0), the 2D texture (view 1) and sampler 0, sets the VGPRs `setup` lists
/// ((VGPR, literal)), runs `sample` (its result in `v0`) and stores `v0`.
fn build(setup: &[(u8, u32)], sample: [u32; 2]) -> HostShader {
    let mut words = vec![
        s_mov_b32(6, 4),
        s_mov_b32(7, ZERO),
        s_mov_b32(10, 5),
        s_mov_b32(11, ZERO),
    ];
    words.extend(s_load(2, 12, 6, 0));
    words.extend(s_load(3, 16, 6, 32));
    words.extend(s_load(2, 24, 10, 0));
    for &(v, x) in setup {
        words.extend([v_mov(v, LITERAL), x]);
    }
    words.extend(sample);
    words.extend(buffer_store_dword(0, 12, 0));
    words.push(S_ENDPGM);

    let table = |user_data, slot| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot,
    };
    let used = |kind, table_offset_dwords, dwords, user_data, slot| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            table(4, 1),
            table(5, 2),
        ],
        used: vec![
            used(ResourceType::Uav, 0, 4, 4, 1),
            used(ResourceType::Srv, 8, 8, 4, 1),
            used(ResourceType::Sampler, 0, 4, 5, 2),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 6 << 1, // 6 user SGPRs: s0..s5
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    shader
}

/// A 4×4 texture whose texel (x, y) is `10 y + x` in every channel.
fn texels() -> Vec<[f32; 4]> {
    (0..16)
        .map(|i| [(10 * (i / 4) + i % 4) as f32; 4])
        .collect()
}

/// Runs `shader` on each SM 6.6 device, point sampling with clamping;
/// returns the stored value per device.
fn run(shader: &HostShader) -> Vec<(&'static str, f32)> {
    let texels = texels();
    let sampler = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Constants, 0) => 0,
                // Both tables start at their heap's first entry.
                (RootType::Table, 1 | 2) => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let result = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1), View::Texture(4, 4, &texels)],
                    samplers: &[sampler],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, f32::from_bits(result[0][0])));
    }
    assert!(!out.is_empty(), "no SM 6.6 device available");
    out
}

/// The centre of texel `n` of 4 in normalized coordinates.
fn centre(n: u32) -> f32 {
    (n as f32 + 0.5) / 4.0
}

/// `x` as f16 bits (normal values with at most 10 fraction bits).
fn f16(x: f32) -> u32 {
    if x == 0.0 {
        return 0;
    }
    let bits = x.to_bits();
    let exp = (bits >> 23 & 0xFF) + 15 - 127;
    exp << 10 | (bits >> 13 & 0x3FF)
}

// Compute programs here take explicit levels: implicit derivatives
// (image_sample, _b, _o) are pixel-only in the translation.

#[test]
fn a_level_zero_sample_with_a_constant_offset_reads_the_offset_texel() {
    // image_sample_lz_o v0, v[3:5]: offset word (x +1, y +2), u, v.
    let shader = build(
        &[
            (3, 1 | 2 << 8),
            (4, centre(1).to_bits()),
            (5, centre(0).to_bits()),
        ],
        mimg(IMAGE_SAMPLE_LZ_O, 0, 3, false),
    );
    for (backend, got) in run(&shader) {
        assert_eq!(got, 22.0, "{backend}: texel (1, 0) offset by (1, 2)");
    }
}

#[test]
fn a_level_sample_with_an_offset_reads_its_level_after_the_coordinates() {
    // image_sample_l_o v0, v[3:6]: offset word (x -1), u, v, lod 0.
    let shader = build(
        &[
            (3, 0x3F),
            (4, centre(2).to_bits()),
            (5, centre(3).to_bits()),
            (6, 0),
        ],
        mimg(IMAGE_SAMPLE_L_O, 0, 3, false),
    );
    for (backend, got) in run(&shader) {
        assert_eq!(got, 31.0, "{backend}: texel (2, 3) offset by (-1, 0)");
    }
}

#[test]
fn an_a16_sample_reads_its_coordinates_after_the_extra_word() {
    // image_sample_l_o v0, v[3:5] a16: the offset word (y -1) alone in v3,
    // (u, v) as f16 halves of v4, the level (0) in v5's low half.
    let shader = build(
        &[
            (3, 0x3F << 8),
            (4, f16(centre(3)) | f16(centre(2)) << 16),
            (5, 0),
        ],
        mimg(IMAGE_SAMPLE_L_O, 0, 3, true),
    );
    for (backend, got) in run(&shader) {
        assert_eq!(got, 13.0, "{backend}: texel (3, 2) offset by (0, -1)");
    }
}

#[test]
fn a_gradient_sample_with_a_clamp_reads_the_coordinates_after_the_gradients() {
    // image_sample_d_cl v0, v[3:9]: dx (2), dy (2), u, v, clamp.
    let shader = build(
        &[
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, centre(1).to_bits()),
            (8, centre(2).to_bits()),
            (9, 0),
        ],
        mimg(IMAGE_SAMPLE_D_CL, 0, 3, false),
    );
    for (backend, got) in run(&shader) {
        assert_eq!(got, 21.0, "{backend}: texel (1, 2)");
    }
}
