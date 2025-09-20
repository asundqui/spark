
use std::io::{Read, Write};

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
}

#[derive(Serialize, Deserialize)]
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

#[derive(Serialize, Deserialize)]
pub enum RadType {
    #[serde(rename = "gsplat")]
    Gsplat,
}

#[derive(Serialize, Deserialize)]
pub enum RadProperty {
    #[serde(rename = "center")]
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

#[derive(Serialize, Deserialize)]
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

#[derive(Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct RadLodMeta {
    pub pixelSizes: Vec<(f32, u32)>,
    pub boundCenter: [f32; 3],
    pub boundRadius: f32,
}

#[derive(Serialize, Deserialize)]
pub enum RadCompression {
    #[serde(rename = "gzip")]
    Gzip,
}

#[derive(Serialize, Deserialize)]
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

fn gzip_compress_bytes(input: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(input)?;
    let compressed = encoder.finish()?;
    Ok(compressed)
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

    pub fn new_rgb<F: Fn(usize) -> [f32; 3]>(base: usize, count: usize, get: F) -> (Self, Vec<u8>) {
        let min = 0.0;
        let max = 1.0;
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

    const MIN_LOG_SCALE: f32 = -50.0;

    pub fn new_scales<F: Fn(usize) -> [f32; 3]>(base: usize, count: usize, get: F) -> (Self, Vec<u8>) {
        let min = -12.0;
        let max = 9.0;
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

    pub fn new_lod_scales<F: Fn(usize) -> [f32; 4]>(base: usize, count: usize, get: F) -> (Self, Vec<u8>) {
        let min = -12.0;
        let max = 9.0;
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
