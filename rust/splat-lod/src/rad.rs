
use std::io::{Read, Write};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use half::f16;

pub struct RadFile {
    pub magic: u32,
    pub meta_size: u32,
    pub meta_bytes: Vec<u8>,
    pub payloads: Vec<Vec<u8>>,
}

impl RadFile {
    const RAD_MAGIC: u32 = 0x30444152; // 'RAD0'
    const JSON_PRETTY: bool = true;

    #[allow(unused)]
    pub fn new<M: Serialize>(meta: &M, payloads: Vec<Vec<u8>>) -> anyhow::Result<Self> {
        let meta_bytes = if Self::JSON_PRETTY {
            serde_json::to_vec_pretty(meta)?
        } else {
            serde_json::to_vec(meta)?
        };
        println!("Meta bytes size: {}", meta_bytes.len());
        Ok(Self {
            magic: Self::RAD_MAGIC,
            meta_size: meta_bytes.len() as u32,
            meta_bytes,
            payloads,
        })
    }

    #[allow(unused)]
    pub fn write_to<W: Write>(&self, writer: &mut W) -> anyhow::Result<()> {
        writer.write_all(&self.magic.to_le_bytes())?;

        let write_pad = |writer: &mut W, size: usize| -> anyhow::Result<()> {
            let pad = (8 - (size & 7)) & 7;
            if pad != 0 {
                let zero_pad = [0u8; 8];
                writer.write_all(&zero_pad[..pad as usize])?;
            }
            Ok(())
        };

        writer.write_all(&self.meta_size.to_le_bytes())?;
        writer.write_all(&self.meta_bytes)?;
        write_pad(writer, self.meta_size as usize)?;

        // Write payloads: [u64 size][bytes][padding to 8]
        for payload in &self.payloads {
            let size_u64 = payload.len() as u64;
            writer.write_all(&size_u64.to_le_bytes())?;
            writer.write_all(payload)?;
            write_pad(writer, size_u64 as usize)?;
        }

        // Final pseudo-payload signaling end
        writer.write_all(&0u64.to_le_bytes())?;
        Ok(())
    }

    #[allow(unused)]
    pub fn read_from<R: Read>(reader: &mut R) -> anyhow::Result<Self> {
        let mut magic_bytes = [0u8; 4];
        reader.read_exact(&mut magic_bytes)?;
        let magic = u32::from_le_bytes(magic_bytes);
        if magic != Self::RAD_MAGIC {
            anyhow::bail!("invalid RAD magic: expected 0x{:08x}, got 0x{:08x}", Self::RAD_MAGIC, magic);
        }

        let mut size_bytes = [0u8; 4];
        reader.read_exact(&mut size_bytes)?;
        let meta_size = u32::from_le_bytes(size_bytes);

        let mut meta_bytes = vec![0u8; meta_size as usize];
        reader.read_exact(&mut meta_bytes)?;

        // Skip padding to align to next 8-byte boundary
        let pad = (8 - (meta_size & 7)) & 7;
        if pad != 0 {
            let mut discard = [0u8; 8];
            reader.read_exact(&mut discard[..pad as usize])?;
        }

        // Read payloads until a final 0-sized pseudo-payload
        let mut payloads: Vec<Vec<u8>> = Vec::new();
        loop {
            let mut size_buf = [0u8; 8];
            reader.read_exact(&mut size_buf)?;
            let size_u64 = u64::from_le_bytes(size_buf);
            if size_u64 == 0 {
                break;
            }
            let mut payload = vec![0u8; size_u64 as usize];
            reader.read_exact(&mut payload)?;

            // Skip padding to 8-byte boundary
            let pad = (8 - (size_u64 & 7)) & 7;
            if pad != 0 {
                let mut discard = [0u8; 8];
                reader.read_exact(&mut discard[..pad as usize])?;
            }

            payloads.push(payload);
        }

        Ok(Self { magic, meta_size, meta_bytes, payloads })
    }

    #[allow(unused)]
    pub fn meta_as<M: DeserializeOwned>(&self) -> anyhow::Result<M> {
        let meta: M = serde_json::from_slice(&self.meta_bytes)?;
        Ok(meta)
    }

    #[allow(unused)]
    pub fn new_from_gsplats<R: RadGsplatReader>(meta: &RadMeta, reader: &R, rgb_range: Option<RgbMinMax>, ln_range: Option<LnScaleMinMax>) -> anyhow::Result<Self> {
        if !matches!(meta.ty, RadType::Gsplat) { anyhow::bail!("Unsupported RAD type"); }

        let count = meta.count as usize;
        let rgb_range = rgb_range.unwrap_or_default();
        let ln_scale_range = ln_range.unwrap_or_default();
        let payloads = [
            RadPayload::new_center(0, count, |i| reader.center(i)),
            RadPayload::new_alpha(0, count, |i| reader.alpha(i)),
            RadPayload::new_rgb_with_range(0, count, |i| reader.rgb(i), rgb_range),
            RadPayload::new_scales_with_range(0, count, |i| reader.scales(i), ln_scale_range),
            RadPayload::new_orientation(0, count, |i| reader.orientation(i)),
            RadPayload::new_lod_scales_with_range(0, count, |i| reader.lod_scales(i), ln_scale_range),
        ];

        let (payload_metas, payload_bytes) = payloads.into_iter().unzip();
        let out_meta = RadMeta {
            version: 1,
            ty: RadType::Gsplat,
            count: meta.count,
            antialias: meta.antialias,
            payloads: payload_metas,
            lodMeta: meta.lodMeta.clone(),
        };
        Self::new(&out_meta, payload_bytes)
    }
}

#[derive(Serialize, Deserialize, Default, Clone)]
#[allow(non_snake_case)]
pub struct RadMeta {
    pub version: u32,
    #[serde(rename = "type")]
    pub ty: RadType,
    pub count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub antialias: Option<bool>,
    pub payloads: Vec<RadPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lodMeta: Option<RadLodMeta>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
pub enum RadType {
    #[default]
    #[serde(rename = "gsplat")]
    Gsplat,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
pub enum RadProperty {
    #[serde(rename = "center")]
    #[default]
    Center,
    #[serde(rename = "alpha")]
    Alpha,
    #[serde(rename = "rgb")]
    Rgb,
    #[serde(rename = "scales")]
    Scales,
    #[serde(rename = "orientation")]
    Orientation,
    #[serde(rename = "lodScales")]
    LodScales,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum RadEncoding {
    #[serde(rename = "float16ByteOrder")]
    Float16ByteOrder,
    #[serde(rename = "uint8")]
    Uint8,
    #[serde(rename = "logUint8")]
    LogUint8,
    #[serde(rename = "oct88R8")]
    Oct88R8,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[allow(non_snake_case)]
pub struct RadLodMeta {
    pub pixelSizes: Vec<(f32, u32)>,
    pub boundCenter: [f32; 3],
    pub boundRadius: f32,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum RadCompression {
    #[serde(rename = "gzip")]
    Gzip,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct RadPayload {
    pub bytes: u64,
    pub property: RadProperty,
    pub encoding: RadEncoding,
    pub base: u32,
    pub count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub components: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compression: Option<RadCompression>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mins: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maxs: Option<Vec<f32>>,
}

#[derive(Clone, Copy, Debug)]
pub struct RgbMinMax {
    pub min: f32,
    pub max: f32,
}

impl Default for RgbMinMax {
    fn default() -> Self { Self { min: 0.0, max: 1.0 } }
}

#[derive(Clone, Copy, Debug)]
pub struct LnScaleMinMax {
    pub min: f32,
    pub max: f32,
}

impl Default for LnScaleMinMax {
    fn default() -> Self { Self { min: -12.0, max: 9.0 } }
}

impl Default for RadPayload {
    fn default() -> Self {
        Self {
            bytes: 0,
            property: RadProperty::Alpha,
            encoding: RadEncoding::Uint8,
            base: 0,
            count: 0,
            components: None,
            compression: None,
            mins: None,
            maxs: None,
        }
    }
}

fn u16_byte_order(items: &[u16]) -> Vec<u8> {
    let mut buffer = vec![0u8; items.len() * 2];
    for (i, item) in items.iter().enumerate() {
        let bytes = item.to_le_bytes();
        buffer[i] = bytes[0];
        buffer[i + items.len()] = bytes[1];
    }
    buffer
}

fn quat_xyzw_to_oct88r8(quat: [f32; 4]) -> [u8; 3] {
    let axis_norm2 = quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2];
    let quat_norm = (axis_norm2 + quat[3]*quat[3]).sqrt();
    let negate = quat[3] < 0.0;
    let quat_w = (if negate { -quat[3] } else { quat[3] }) / quat_norm;
    let theta = 2.0 * quat_w.acos();

    let axis_norm = axis_norm2.sqrt();
    let zero_axis = axis_norm < 1e-6;
    let axis_norm = if negate { -axis_norm } else { axis_norm };
    let [axis_x, axis_y, axis_z] = if zero_axis {
        [1.0, 0.0, 0.0]
    } else {
        [quat[0] / axis_norm, quat[1] / axis_norm, quat[2] / axis_norm]
    };

    let sum = axis_x.abs() + axis_y.abs() + axis_z.abs();
    let [mut p_x, mut p_y] = [axis_x / sum, axis_y / sum];
    if axis_z < 0.0 {
        [p_x, p_y] = [
            (1.0 - p_y.abs()) * (if p_x >= 0.0 { 1.0 } else { -1.0 }),
            (1.0 - p_x.abs()) * (if p_y >= 0.0 { 1.0 } else { -1.0 }),
        ];
    }

    let u = ((0.5 + p_x * 0.5) * 255.0).clamp(0.0, 255.0).round();
    let v = ((0.5 + p_y * 0.5) * 255.0).clamp(0.0, 255.0).round();
    let r = (theta * 255.0 / std::f32::consts::PI).clamp(0.0, 255.0).round();
    [u as u8, v as u8, r as u8]
}

fn oct88r8_to_quat_xyzw(u: u8, v: u8, r: u8) -> [f32; 4] {
    let mut fx = (u as f32 / 255.0 - 0.5) * 2.0;
    let mut fy = (v as f32 / 255.0 - 0.5) * 2.0;
    let half_theta = 0.5 * (r as f32 / 255.0) * std::f32::consts::PI;

    let fz = 1.0 - (fx.abs() + fy.abs());
    let t = (-fz).max(0.0);
    fx += if fx >= 0.0 { -t } else { t };
    fy += if fy >= 0.0 { -t } else { t };
    let axis_len = (fx * fx + fy * fy + fz * fz).sqrt();
    let zero_axis = axis_len < 1e-6;
    let axis_x = if zero_axis { 1.0 } else { fx / axis_len };
    let axis_y = if zero_axis { 0.0 } else { fy / axis_len };
    let axis_z = if zero_axis { 0.0 } else { fz / axis_len };
    let s = half_theta.sin();
    let w = half_theta.cos();
    [axis_x * s, axis_y * s, axis_z * s, w]
}

impl RadPayload {
    pub fn into_gzipped(self, bytes: Vec<u8>) -> (Self, Vec<u8>) {
        assert!(self.compression.is_none());
        let compressed = gzip_compress_bytes(&bytes).unwrap();
        let meta = Self {
            bytes: compressed.len() as u64,
            compression: Some(RadCompression::Gzip),
            ..self
        };
        (meta, compressed)
    }

    pub fn new_center<F: Fn(usize) -> [f32; 3]>(base: usize, count: usize, get: F) -> (Self, Vec<u8>) {
        let mut buffer = vec![f16::from_f32(0.0).to_bits(); 3 * count];
        for i in 0..count {
            let [x, y, z] = get(base + i);
            buffer[i] = f16::from_f32(x).to_bits();
            buffer[i + count] = f16::from_f32(y).to_bits();
            buffer[i + 2 * count] = f16::from_f32(z).to_bits();
        }
        let buffer = u16_byte_order(&buffer);

        let meta = Self {
            bytes: buffer.len() as u64,
            property: RadProperty::Center,
            encoding: RadEncoding::Float16ByteOrder,
            base: base as u32,
            count: count as u32,
            components: Some(3),
            ..Default::default()
        };
        meta.into_gzipped(buffer)
    }

    pub fn new_alpha<F: Fn(usize) -> f32>(base: usize, count: usize, get: F) -> (Self, Vec<u8>) {
        let mut elements = vec![f16::from_f32(0.0).to_bits(); count];
        for i in 0..count {
            elements[i] = f16::from_f32(get(base + i)).to_bits();
        }
        let buffer = u16_byte_order(&elements);

        let meta = Self {
            bytes: buffer.len() as u64,
            property: RadProperty::Alpha,
            encoding: RadEncoding::Float16ByteOrder,
            base: base as u32,
            count: count as u32,
            components: Some(1),
            ..Default::default()
        };
        meta.into_gzipped(buffer)
    }

    pub fn new_rgb_with_range<F: Fn(usize) -> [f32; 3]>(base: usize, count: usize, get: F, range: RgbMinMax) -> (Self, Vec<u8>) {
        let min = range.min;
        let max = range.max;
        let mut buffer = vec![0u8; 3 * count];
        for i in 0..count {
            let rgb = get(base + i);
            let rgb = rgb.map(|c| (c - min) / (max - min));
            let rgb = rgb.map(|c| (c * 255.0).clamp(0.0, 255.0).round() as u8);
            buffer[i] = rgb[0];
            buffer[i + count] = rgb[1];
            buffer[i + 2 * count] = rgb[2];
        }

        let meta = Self {
            bytes: buffer.len() as u64,
            property: RadProperty::Rgb,
            encoding: RadEncoding::Uint8,
            base: base as u32,
            count: count as u32,
            components: Some(3),
            mins: Some(vec![min; 1]),
            maxs: Some(vec![max; 1]),
            ..Default::default()
        };
        meta.into_gzipped(buffer)
    }

    #[allow(unused)]
    pub fn new_rgb<F: Fn(usize) -> [f32; 3]>(base: usize, count: usize, get: F) -> (Self, Vec<u8>) {
        Self::new_rgb_with_range(base, count, get, RgbMinMax::default())
    }

    const MIN_LOG_SCALE: f32 = -50.0;

    pub fn new_scales_with_range<F: Fn(usize) -> [f32; 3]>(base: usize, count: usize, get: F, range: LnScaleMinMax) -> (Self, Vec<u8>) {
        let min = range.min;
        let max = range.max;
        let min_scale: f32 = Self::MIN_LOG_SCALE.exp();
        let mut buffer = vec![0u8; 3 * count];
        for i in 0..count {
            let scales = get(base + i);
            let scales = scales.map(|s| {
                let s = s.abs();
                if s < min_scale {
                    0
                } else {
                    ((s.ln() - min) / (max - min) * 254.0).clamp(0.0, 254.0).round() as u8 + 1
                }
            });
            buffer[i] = scales[0];
            buffer[i + count] = scales[1];
            buffer[i + 2 * count] = scales[2];
        }

        let meta = Self {
            bytes: buffer.len() as u64,
            property: RadProperty::Scales,
            encoding: RadEncoding::LogUint8,
            base: base as u32,
            count: count as u32,
            components: Some(3),
            mins: Some(vec![min; 1]),
            maxs: Some(vec![max; 1]),
            ..Default::default()
        };
        meta.into_gzipped(buffer)
    }

    #[allow(unused)]
    pub fn new_scales<F: Fn(usize) -> [f32; 3]>(base: usize, count: usize, get: F) -> (Self, Vec<u8>) {
        Self::new_scales_with_range(base, count, get, LnScaleMinMax::default())
    }

    pub fn new_lod_scales_with_range<F: Fn(usize) -> [f32; 4]>(base: usize, count: usize, get: F, range: LnScaleMinMax) -> (Self, Vec<u8>) {
        let min = range.min;
        let max = range.max;
        let min_scale: f32 = Self::MIN_LOG_SCALE.exp();
        let mut buffer = vec![0u8; 4 * count];
        for i in 0..count {
            let scales = get(base + i);
            let scales = scales.map(|s| {
                let s = s.abs();
                if s < min_scale {
                    0
                } else {
                    ((s.ln() - min) / (max - min) * 254.0).clamp(0.0, 254.0).round() as u8 + 1
                }
            });
            buffer[i] = scales[0];
            buffer[i + count] = scales[1];
            buffer[i + 2 * count] = scales[2];
            buffer[i + 3 * count] = scales[3];
        }

        let meta = Self {
            bytes: buffer.len() as u64,
            property: RadProperty::LodScales,
            encoding: RadEncoding::LogUint8,
            base: base as u32,
            count: count as u32,
            components: Some(4),
            mins: Some(vec![min; 1]),
            maxs: Some(vec![max; 1]),
            ..Default::default()
        };
        meta.into_gzipped(buffer)
    }

    #[allow(unused)]
    pub fn new_lod_scales<F: Fn(usize) -> [f32; 4]>(base: usize, count: usize, get: F) -> (Self, Vec<u8>) {
        Self::new_lod_scales_with_range(base, count, get, LnScaleMinMax::default())
    }

    pub fn new_orientation<F: Fn(usize) -> [f32; 4]>(base: usize, count: usize, get: F) -> (Self, Vec<u8>) {
        let mut buffer = vec![0u8; 3 * count];
        for i in 0..count {
            let orientation = get(base + i);
            let [u, v, r] = quat_xyzw_to_oct88r8(orientation);
            let i3 = i * 3;
            buffer[i3] = u;
            buffer[i3 + 1] = v;
            buffer[i3 + 2] = r;
        }

        let meta = Self {
            bytes: buffer.len() as u64,
            property: RadProperty::Orientation,
            encoding: RadEncoding::Oct88R8,
            base: base as u32,
            count: count as u32,
            ..Default::default()
        };
        meta.into_gzipped(buffer)
    }
}

pub fn morton_coord16_to_index([x, y, z]: [u16; 3]) -> u64 {
    fn expand3(x: u16) -> u64 {
        let mut x = x as u64;
        x = (x | x << 32) & 0x1f00000000ffff;
        x = (x | x << 16) & 0x1f0000ff0000ff;
        x = (x | x << 8) & 0x100f00f00f00f00f;
        x = (x | x << 4) & 0x10c30c30c30c30c3;
        x = (x | x << 2) & 0x1249249249249249;
        x
    }

    (expand3(x) << 0) | (expand3(y) << 1) | (expand3(z) << 2)
}

pub fn morton_coord_to_index(xyz: [f32; 3]) -> u64 {
    let coord16 = xyz.map(|x| f16::from_f32(x).to_bits());
    morton_coord16_to_index(coord16)
}

#[allow(unused)]
pub trait RadGsplatReader {
    fn center(&self, i: usize) -> [f32; 3] { let _ = i; [0.0, 0.0, 0.0] }
    fn alpha(&self, i: usize) -> f32 { let _ = i; 0.0 }
    fn rgb(&self, i: usize) -> [f32; 3] { let _ = i; [0.0, 0.0, 0.0] }
    fn scales(&self, i: usize) -> [f32; 3] { let _ = i; [0.0, 0.0, 0.0] }
    fn lod_scales(&self, i: usize) -> [f32; 4] { let _ = i; [0.0, 0.0, 0.0, 0.0] }
    fn orientation(&self, i: usize) -> [f32; 4] { let _ = i; [0.0, 0.0, 0.0, 1.0] }
}

#[allow(unused)]
pub trait RadGsplatWriter {
    fn begin(&mut self, _meta: &RadMeta) -> anyhow::Result<()> { Ok(()) }
    fn set_encoding(&mut self, _params: RadEncodingParams) {}
    fn write_center(&mut self, _i: usize, _x: f32, _y: f32, _z: f32) {}
    fn write_alpha(&mut self, _i: usize, _a: f32) {}
    fn write_rgb(&mut self, _i: usize, _r: f32, _g: f32, _b: f32) {}
    fn write_scales(&mut self, _i: usize, _sx: f32, _sy: f32, _sz: f32) {}
    fn write_lod_scales(&mut self, _i: usize, _lmin: f32, _llow: f32, _lhigh: f32, _lmax: f32) {}
    fn write_orientation(&mut self, _i: usize, _x: f32, _y: f32, _z: f32, _w: f32) {}
    fn end(&mut self) -> anyhow::Result<()> { Ok(()) }
}

#[allow(unused)]
pub struct RadEncodingParams {
    pub rgb_min: Option<f32>,
    pub rgb_max: Option<f32>,
    pub ln_scale_min: Option<f32>,
    pub ln_scale_max: Option<f32>,
    pub sh1_min: Option<f32>,
    pub sh1_max: Option<f32>,
    pub sh2_min: Option<f32>,
    pub sh2_max: Option<f32>,
    pub sh3_min: Option<f32>,
    pub sh3_max: Option<f32>,
}

fn gunzip_bytes(input: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut decoder = GzDecoder::new(input);
    let mut bytes = Vec::new();
    decoder.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn gzip_compress_bytes(input: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(input)?;
    let compressed = encoder.finish()?;
    Ok(compressed)
}


impl RadFile {
    #[allow(unused)]
    pub fn parse_splats(&self, meta: &RadMeta, writer: &mut impl RadGsplatWriter) -> anyhow::Result<()> {
        if meta.version != 1 {
            anyhow::bail!("Unsupported RAD version: {}", meta.version);
        }
        if (meta.ty != RadType::Gsplat) {
            anyhow::bail!("Unsupported RAD type: {:?}", meta.ty);
        }

        if meta.payloads.len() != self.payloads.len() {
            anyhow::bail!(
                "Payload count mismatch: meta {} vs file {}",
                meta.payloads.len(),
                self.payloads.len()
            );
        }

        writer.begin(meta)?;
        for (i, pmeta) in meta.payloads.iter().enumerate() {
            let mut bytes = self.payloads[i].clone();
            if let Some(RadCompression::Gzip) = pmeta.compression {
                bytes = gunzip_bytes(&bytes)?;
            }

            let components = pmeta.components.unwrap_or(1) as usize;

            let mut mins = pmeta.mins.clone().unwrap_or_default();
            if mins.is_empty() { mins.push(0.0); }
            while mins.len() < components { mins.push(mins[0]); }

            let mut maxs = pmeta.maxs.clone().unwrap_or_default();
            if maxs.is_empty() { maxs.push(1.0); }
            while maxs.len() < components { maxs.push(maxs[0]); }

            match pmeta.encoding {
                RadEncoding::Float16ByteOrder => {
                    let expected = (2usize) * (pmeta.count as usize) * components;
                    if bytes.len() != expected {
                        anyhow::bail!("Mismatched RAD encoding: {} != {}", bytes.len(), expected);
                    }

                    let plane_size = (pmeta.count as usize) * components;
                    for idx in 0..(pmeta.count as usize) {
                        let mut args: [f32; 4] = [0.0; 4];
                        for d in 0..components {
                            let off = d * (pmeta.count as usize) + idx;
                            let lo = bytes[off] as u16;
                            let hi = bytes[plane_size + off] as u16;
                            let u = lo | (hi << 8);
                            args[1 + d] = f16::from_bits(u).to_f32();
                        }
                        let base_i = pmeta.base as usize + idx;
                        match pmeta.property {
                            RadProperty::Center => writer.write_center(base_i, args[1], args[2], args[3]),
                            RadProperty::Alpha => writer.write_alpha(base_i, args[1]),
                            _ => {}
                        }
                    }
                }
                RadEncoding::Uint8 => {
                    let expected = (pmeta.count as usize) * components;
                    if bytes.len() != expected {
                        anyhow::bail!("Mismatched RAD encoding: {} != {}", bytes.len(), expected);
                    }

                    if let RadProperty::Rgb = pmeta.property {
                        writer.set_encoding(RadEncodingParams { rgb_min: Some(mins[0]), rgb_max: Some(maxs[0]), ln_scale_min: None, ln_scale_max: None, sh1_min: None, sh1_max: None, sh2_min: None, sh2_max: None, sh3_min: None, sh3_max: None });
                    }

                    for idx in 0..(pmeta.count as usize) {
                        let mut args: [f32; 4] = [0.0; 4];
                        for d in 0..components {
                            let u = bytes[d * (pmeta.count as usize) + idx] as f32 / 255.0;
                            args[1 + d] = u * (maxs[d] - mins[d]) + mins[d];
                        }
                        let base_i = pmeta.base as usize + idx;
                        match pmeta.property {
                            RadProperty::Rgb => writer.write_rgb(base_i, args[1], args[2], args[3]),
                            _ => {}
                        }
                    }
                }
                RadEncoding::LogUint8 => {
                    let expected = (pmeta.count as usize) * components;
                    if bytes.len() != expected {
                        anyhow::bail!("Mismatched RAD encoding: {} != {}", bytes.len(), expected);
                    }

                    if components == 3 && matches!(pmeta.property, RadProperty::Scales) {
                        writer.set_encoding(RadEncodingParams { rgb_min: None, rgb_max: None, ln_scale_min: Some(mins[0]), ln_scale_max: Some(maxs[0]), sh1_min: None, sh1_max: None, sh2_min: None, sh2_max: None, sh3_min: None, sh3_max: None });
                    }

                    for idx in 0..(pmeta.count as usize) {
                        let mut args: [f32; 5] = [0.0; 5];
                        for d in 0..components {
                            let u8v = bytes[d * (pmeta.count as usize) + idx];
                            args[1 + d] = if u8v == 0 { 0.0 } else { (((u8v as f32) - 1.0) / 254.0 * (maxs[d] - mins[d]) + mins[d]).exp() };
                        }
                        let base_i = pmeta.base as usize + idx;
                        match pmeta.property {
                            RadProperty::Scales => writer.write_scales(base_i, args[1], args[2], args[3]),
                            RadProperty::LodScales => writer.write_lod_scales(base_i, args[1], args[2], args[3], args[4]),
                            _ => {}
                        }
                    }
                }
                RadEncoding::Oct88R8 => {
                    let expected = 3 * (pmeta.count as usize);
                    if bytes.len() != expected {
                        anyhow::bail!("Mismatched RAD encoding: {} != {}", bytes.len(), expected);
                    }

                    for idx in 0..(pmeta.count as usize) {
                        let i3 = idx * 3;
                        let q = oct88r8_to_quat_xyzw(bytes[i3], bytes[i3 + 1], bytes[i3 + 2]);
                        let base_i = pmeta.base as usize + idx;
                        writer.write_orientation(base_i, q[0], q[1], q[2], q[3]);
                    }
                }
            }
        }

        writer.end()?;
        Ok(())
    }
}
