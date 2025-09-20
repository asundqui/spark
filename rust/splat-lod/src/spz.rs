use anyhow::anyhow;
use flate2::read::GzDecoder;
use std::io::Read;

use crate::lod::Gsplat;

const SPZ_MAGIC: u32 = 0x5053_474e; // 'NGSP'
const SH_C0: f32 = 0.28209479177387814;

#[allow(unused)]
pub struct SpzReader {
    data: Vec<u8>,
    offset: usize,

    pub version: u32,
    pub num_splats: usize,
    pub sh_degree: u8,
    pub fractional_bits: u8,
    pub flags: u8,
    pub flag_antialias: bool,
}

impl SpzReader {
    pub fn new_from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        // SPZ is a gzip-compressed stream. Decompress whole stream for simple, fast indexed reads.
        let mut decoder = GzDecoder::new(bytes);
        let mut data = Vec::new();
        decoder.read_to_end(&mut data)?;

        if data.len() < 16 {
            return Err(anyhow!("Invalid SPZ: too short"));
        }

        let mut offset = 0;
        let read_u32_le = |buf: &[u8], off: &mut usize| -> u32 {
            let v = u32::from_le_bytes(buf[*off..*off + 4].try_into().unwrap());
            *off += 4;
            v
        };

        let magic = read_u32_le(&data, &mut offset);
        if magic != SPZ_MAGIC {
            return Err(anyhow!("Invalid SPZ magic: 0x{:08x}", magic));
        }
        let version = read_u32_le(&data, &mut offset);
        if version < 1 || version > 2 {
            return Err(anyhow!("Unsupported SPZ version: {}", version));
        }
        let num_splats_u32 = read_u32_le(&data, &mut offset);
        if data.len() < 16 {
            return Err(anyhow!("Invalid SPZ: header incomplete"));
        }
        let sh_degree = data[offset];
        let fractional_bits = data[offset + 1];
        let flags = data[offset + 2];
        let _reserved = data[offset + 3];
        offset += 4;

        let flag_antialias = (flags & 0x01) != 0;

        Ok(Self {
            data,
            offset,
            version,
            num_splats: num_splats_u32 as usize,
            sh_degree,
            fractional_bits,
            flags,
            flag_antialias,
        })
    }

    fn read(&mut self, len: usize) -> anyhow::Result<&[u8]> {
        if self.offset + len > self.data.len() {
            return Err(anyhow!(
                "SPZ truncated: need {} bytes at {}, total {}",
                len,
                self.offset,
                self.data.len()
            ));
        }
        let start = self.offset;
        self.offset += len;
        Ok(&self.data[start..start + len])
    }

    pub fn into_gsplats(mut self) -> anyhow::Result<Vec<Gsplat>> {
        let ns = self.num_splats;
        let mut splats = vec![Gsplat::default(); ns];

        // Centers
        match self.version {
            1 => {
                let bytes = self.read(ns * 3 * 2)?; // 3 x f16 per splat
                for i in 0..ns {
                    let base = i * 6;
                    let x = read_f16_le(&bytes[base..base + 2]);
                    let y = read_f16_le(&bytes[base + 2..base + 4]);
                    let z = read_f16_le(&bytes[base + 4..base + 6]);
                    splats[i].center = [x, y, z];
                }
            }
            2 => {
                let fixed = 1_i32 << self.fractional_bits;
                let bytes = self.read(ns * 9)?; // 3 x i24 per splat
                for i in 0..ns {
                    let base = i * 9;
                    let x = read_i24_le(&bytes[base..base + 3]) as f32 / fixed as f32;
                    let y = read_i24_le(&bytes[base + 3..base + 6]) as f32 / fixed as f32;
                    let z = read_i24_le(&bytes[base + 6..base + 9]) as f32 / fixed as f32;
                    splats[i].center = [x, y, z];
                }
            }
            _ => return Err(anyhow!("Unreachable version")),
        }

        // Alpha (opacity 0..1)
        {
            let bytes = self.read(ns)?;
            for i in 0..ns {
                splats[i].opacity = bytes[i] as f32 / 255.0;
            }
        }

        // RGB, scaled around 0.5 using SH_C0/0.15 factor
        {
            let bytes = self.read(ns * 3)?;
            let scale = SH_C0 / 0.15;
            for i in 0..ns {
                let base = i * 3;
                let r = (bytes[base] as f32 / 255.0 - 0.5) * scale + 0.5;
                let g = (bytes[base + 1] as f32 / 255.0 - 0.5) * scale + 0.5;
                let b = (bytes[base + 2] as f32 / 255.0 - 0.5) * scale + 0.5;
                splats[i].color = [r, g, b];
            }
        }

        // Scales: exp(byte/16 - 10)
        {
            let bytes = self.read(ns * 3)?;
            for i in 0..ns {
                let base = i * 3;
                let sx = ((bytes[base] as f32) / 16.0 - 10.0).exp();
                let sy = ((bytes[base + 1] as f32) / 16.0 - 10.0).exp();
                let sz = ((bytes[base + 2] as f32) / 16.0 - 10.0).exp();
                splats[i].scales = [sx, sy, sz];
            }
        }

        // Quaternion: x,y,z in [-1,1], w reconstructed as positive sqrt
        {
            let bytes = self.read(ns * 3)?;
            for i in 0..ns {
                let base = i * 3;
                let qx = bytes[base] as f32 / 127.5 - 1.0;
                let qy = bytes[base + 1] as f32 / 127.5 - 1.0;
                let qz = bytes[base + 2] as f32 / 127.5 - 1.0;
                let qw_sq = 1.0 - (qx * qx + qy * qy + qz * qz);
                let qw = if qw_sq > 0.0 { qw_sq.sqrt() } else { 0.0 };
                splats[i].quaternion = [qx, qy, qz, qw];
            }
        }

        // SH coefficients (optional) — advance cursor and ignore
        if self.sh_degree >= 1 {
            let sh_vecs = match self.sh_degree {
                1 => 3,
                2 => 8,
                3 => 15,
                _ => 0,
            };
            if sh_vecs > 0 {
                let bytes_needed = ns * sh_vecs as usize * 3; // 3 color channels
                let _ = self.read(bytes_needed)?;
            }
        }

        Ok(splats)
    }
}

#[inline]
fn read_f16_le(two: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([two[0], two[1]]);
    half::f16::from_bits(bits).to_f32()
}

#[inline]
fn read_i24_le(three: &[u8]) -> i32 {
    // Sign-extend 24-bit little-endian to i32
    let v = (three[2] as u32) << 16 | (three[1] as u32) << 8 | (three[0] as u32);
    if (v & 0x0080_0000) != 0 {
        (v | 0xFF00_0000) as i32
    } else {
        v as i32
    }
}

pub fn read_spz(filename: &str) -> anyhow::Result<Vec<Gsplat>> {
    let mut file = std::fs::File::open(filename)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let reader = SpzReader::new_from_bytes(&bytes)?;
    reader.into_gsplats()
}