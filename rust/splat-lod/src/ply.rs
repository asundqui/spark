use anyhow::anyhow;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub enum PlyPropertyType {
    Float,
    Uchar,
}

impl PlyPropertyType {
    pub fn size(&self) -> usize {
        match self {
            PlyPropertyType::Float => 4,
            PlyPropertyType::Uchar => 1,
        }
    }

    pub fn get_f32(&self, data: &[u8], offset: usize) -> f32 {
        match self {
            PlyPropertyType::Float => {
                let u8_4: [u8; 4] = data[offset..offset + 4].try_into().unwrap();
                f32::from_le_bytes(u8_4)
            },
            PlyPropertyType::Uchar => {
                data[offset] as f32 / 255.0
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PlyProperty {
    pub ty: PlyPropertyType,
    pub offset: usize,
}

impl PlyProperty {
    pub fn get_f32(&self, data: &[u8], array_offset: usize) -> f32 {
        self.ty.get_f32(data, array_offset + self.offset)
    }
}

#[allow(dead_code)]
pub struct PlyFile {
    pub num_splats: usize,
    pub record_size: usize,
    pub properties: HashMap<String, PlyProperty>,
    pub header: String,
    pub data: Vec<u8>,
    pub xyz: [PlyProperty; 3],
    pub scale: [PlyProperty; 3],
    pub rot: [PlyProperty; 4],
    pub opacity_logi: PlyProperty,
    pub f_dc: [PlyProperty; 3],
}

impl PlyFile {
    pub fn new_from_bytes(bytes: Vec<u8>) -> anyhow::Result<Self> {
        // Find the end of the header
        const TERMINATOR: &[u8] = b"end_header\n";
        let header_end = bytes.windows(TERMINATOR.len()).position(|window| window == TERMINATOR);
        let Some(header_end) = header_end else {
            return Err(anyhow!("Could not find end of header in PLY file"));
        };

        // Split the bytes into the header and the binary data
        let header = std::str::from_utf8(&bytes[..header_end])?;
        let data = &bytes[header_end + TERMINATOR.len()..];

        let mut num_splats: Option<usize> = None;
        let mut properties: HashMap<String, PlyProperty> = HashMap::new();
        let mut record_size: usize = 0;

        for (line_index, line) in header.lines().enumerate() {
            let line = line.trim();
            if line_index == 0 {
                if line != "ply" {
                    return Err(anyhow!("Invalid PLY header"));
                }
                continue;
            }
            if line.is_empty() {
                continue;
            }

            let fields: Vec<_> = line.split_whitespace().collect();
            match (fields[0], fields.len()) {
                ("format", 3) => {
                    if fields[1] != "binary_little_endian" {
                        return Err(anyhow!("Unsupported PLY format: {}", fields[1]));
                    }
                    if fields[2] != "1.0" {
                        return Err(anyhow!("Unsupported PLY version: {}", fields[2]));
                    }
                },
                ("element", 3) => {
                    if fields[1] != "vertex" {
                        return Err(anyhow!("Unsupported PLY element: {}", fields[1]));
                    }
                    num_splats = Some(fields[2].parse()?);
                },
                ("property", 3) => {
                    let property_type = match fields[1] {
                        "float" => PlyPropertyType::Float,
                        "uchar" => PlyPropertyType::Uchar,
                        _ => return Err(anyhow!("Unsupported PLY property type: {}", fields[1])),
                    };
                    properties.insert(fields[2].to_string(), PlyProperty {
                        ty: property_type,
                        offset: record_size,
                    });
                    record_size += property_type.size();
                },
                ("comment", _) => {
                    // Ignore comments
                },
                _ => {
                    return Err(anyhow!("Unsupported PLY header line: {}", line));
                },
            }
        }

        let Some(num_splats) = num_splats else {
            return Err(anyhow!("Could not find number of splats in PLY file"));
        };
        let expected_size = num_splats * record_size;
        if data.len() != expected_size {
            return Err(anyhow!("Invalid PLY data size: {} != {}", data.len(), expected_size));
        }

        let xyz = [
            *properties.get("x").ok_or(anyhow!("Missing x property"))?,
            *properties.get("y").ok_or(anyhow!("Missing y property"))?,
            *properties.get("z").ok_or(anyhow!("Missing z property"))?,
        ];
        let scale = [
            *properties.get("scale_0").ok_or(anyhow!("Missing scale_0 property"))?,
            *properties.get("scale_1").ok_or(anyhow!("Missing scale_1 property"))?,
            *properties.get("scale_2").ok_or(anyhow!("Missing scale_2 property"))?,
        ];
        let rot = [
            *properties.get("rot_1").ok_or(anyhow!("Missing rot_0 property"))?,
            *properties.get("rot_2").ok_or(anyhow!("Missing rot_1 property"))?,
            *properties.get("rot_3").ok_or(anyhow!("Missing rot_2 property"))?,
            *properties.get("rot_0").ok_or(anyhow!("Missing rot_3 property"))?,
        ];
        let opacity_logi = *properties.get("opacity").ok_or(anyhow!("Missing opacity property"))?;
        let f_dc = [
            *properties.get("f_dc_0").ok_or(anyhow!("Missing f_dc_0 property"))?,
            *properties.get("f_dc_1").ok_or(anyhow!("Missing f_dc_1 property"))?,
            *properties.get("f_dc_2").ok_or(anyhow!("Missing f_dc_2 property"))?,
        ];
        Ok(Self {
            num_splats,
            record_size,
            properties,
            header: header.to_string(),
            data: data.to_vec(),
            xyz,
            scale,
            rot,
            opacity_logi,
            f_dc,
        })
    }
}
