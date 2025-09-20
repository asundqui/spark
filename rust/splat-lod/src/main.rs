use std::io::{BufWriter, Read, Write};

mod ply;
mod spz;
mod lod;
mod octlod;
mod rad;

use ahash::AHashMap;
use lod::Gsplat;
use serde::Serialize;
const SH_C0: f32 = 0.28209479177387814;

const INPUT_FILENAME: &str = "../../examples/assets/splats/butterfly-ai.ply";
// const INPUT_FILENAME: &str = "../../examples/assets/splats/toad.ply";
// const INPUT_FILENAME: &str = "/Users/asundqui/Downloads/combined_sparse_5m_cropped.ply";
// const INPUT_FILENAME: &str = "sunken.spz";
// const INPUT_FILENAME: &str = "../../examples/assets/splats/test_input.ply";
// const INPUT_FILENAME: &str = "/Users/asundqui/Downloads/hog_20m.ply";
// const INPUT_FILENAME: &str = "/Users/asundqui/Downloads/furry_mcmc_250K_SH2.ply";
// const INPUT_FILENAME: &str = "../../examples/assets/splats/32e9c3f3b1f97154-flux-pro.spz";
const OUTPUT_FILENAME: &str = "../../examples/assets/splats/test.rad";
const META_FILENAME: &str = "../../examples/assets/splats/test.meta.json";

fn write_f32_slice(file: &mut BufWriter<std::fs::File>, values: &[f32]) -> std::io::Result<()> {
    for &v in values {
        file.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn write_ply(splats: &[Gsplat], filename: &str) {
    println!("Writing {} splats to {}", splats.len(), filename);
    // Write utf8 header and then binary bytes
    let file = std::fs::File::create(filename).unwrap();
    let mut file = BufWriter::new(file);
    // Create string of entire header
    let header = format!(
        concat!(
            "ply\n",
            "format binary_little_endian 1.0\n",
            "element vertex {}\n",
            "property float x\n",
            "property float y\n",
            "property float z\n",
            "property float scale_0\n",
            "property float scale_1\n",
            "property float scale_2\n",
            "property float rot_0\n",
            "property float rot_1\n",
            "property float rot_2\n",
            "property float rot_3\n",
            "property float alpha\n",
            "property float f_dc_0\n",
            "property float f_dc_1\n",
            "property float f_dc_2\n",
            "property float lod_min\n",
            "property float lod_low\n",
            "property float lod_high\n",
            "property float lod_max\n",
            "end_header\n"
        ),
        splats.len(),
    );
    file.write_all(header.as_bytes()).unwrap();

    for splat in splats {
        write_f32_slice(&mut file, &splat.center).unwrap();

        // let opacity = splat.opacity.min(1.0);
        // let scales = splat.adjusted_scales();

        let opacity = splat.opacity;
        let scales = splat.scales;

        let ln_scales = scales.map(|s| s.ln());
        write_f32_slice(&mut file, &ln_scales).unwrap();
        let quat = [splat.quaternion[3], splat.quaternion[0], splat.quaternion[1], splat.quaternion[2]];
        write_f32_slice(&mut file, &quat).unwrap();
        // let op_logistic = (splat.opacity / (1.0 - splat.opacity)).ln();
        // file.write_all(&op_logistic.to_le_bytes()).unwrap();
        // file.write_all(&splat.opacity.min(1.0).to_le_bytes()).unwrap();
        file.write_all(&opacity.to_le_bytes()).unwrap();
        let f_dc = splat.color.map(|c| (c - 0.5) / SH_C0);
        write_f32_slice(&mut file, &f_dc).unwrap();
        file.write_all(&splat.lod_min.to_le_bytes()).unwrap();
        file.write_all(&splat.lod_low.to_le_bytes()).unwrap();
        file.write_all(&splat.lod_high.to_le_bytes()).unwrap();
        file.write_all(&splat.lod_max.to_le_bytes()).unwrap();
    }
}

fn read_ply(filename: &str) -> Vec<Gsplat> {
    let mut input_file = std::fs::File::open(filename).unwrap();
    let mut input_bytes = Vec::new();
    input_file.read_to_end(&mut input_bytes).unwrap();
    let ply = ply::PlyFile::new_from_bytes(input_bytes).unwrap();

    let splats: Vec<Gsplat> = (0..ply.num_splats).map(|i| {
        use std::array::from_fn;
        let offset = i * ply.record_size;
        let center = from_fn(|d| ply.xyz[d].get_f32(&ply.data, offset));
        let scales = from_fn(|d| ply.scale[d].get_f32(&ply.data, offset).exp());

        let quat: [f32; 4] = from_fn(|d| ply.rot[d].get_f32(&ply.data, offset));
        let quat_magnitude = quat.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let quaternion = from_fn(|d| quat[d] / quat_magnitude);

        let op_logistic = ply.opacity_logi.get_f32(&ply.data, offset);
        let opacity = 1.0 / (1.0 + (-op_logistic).exp());
        let color = from_fn(|d| ply.f_dc[d].get_f32(&ply.data, offset) * SH_C0 + 0.5);

        Gsplat {
            center,
            scales,
            quaternion,
            opacity,
            color,
            // cov: lod::SymMat3::new_covariance(scales, quaternion),
            ..Default::default()
        }
    }).collect();
    splats
}

fn write_txt(content: &str, filename: &str) {
    let mut file = std::fs::File::create(filename).unwrap();
    file.write_all(content.as_bytes()).unwrap();
}

#[allow(unused)]
fn create_splats() -> Vec<Gsplat> {
    let mut splats = Vec::new();
    let step = 0.02;
    let [x_min, x_max] = [-100, 100];
    let [y_min, y_max] = [-100, 100];
    let scale_scale = 0.3;
    
    for x in x_min..=x_max {
        for y in y_min..=y_max {
            let center = [x as f32 * step, y as f32 * step, 0.0];
            let scales = [scale_scale * step, scale_scale * step, scale_scale * step];
            let quaternion = [0.0, 0.0, 0.0, 1.0];
            let color = [
                (x - x_min) as f32 / (x_max - x_min) as f32,
                (y - y_min) as f32 / (y_max - y_min) as f32,
                0.0,
            ];
            let cov = lod::SymMat3::new_covariance(scales, quaternion);
            let splat = Gsplat {
                center,
                scales,
                quaternion,
                opacity: 1.0,
                color,
                cov,
                ..Default::default()
            };
            // println!("splat: {:?}", splat);
            splats.push(splat);
        }
    }

    splats
}

#[allow(unused)]
fn partition_splats(splats: &[Gsplat], grid_size: f32) -> AHashMap<[i64; 3], Vec<Gsplat>> {
    let mut grid = AHashMap::new();
    for splat in splats {
        let cell = [
            (splat.center[0] / grid_size).floor() as i64,
            (splat.center[1] / grid_size).floor() as i64,
            (splat.center[2] / grid_size).floor() as i64,
        ];
        grid.entry(cell).or_insert(Vec::new()).push(splat.clone());
    }
    grid
}

#[derive(Debug, Serialize)]
struct LodMeta {
    cuts: Vec<(f32, i32)>,
    bound_center: [f32; 3],
    bound_radius: f32,
}

impl rad::RadGsplatReader for Vec<Gsplat> {
    fn center(&self, i: usize) -> [f32; 3] {
        self[i].center
    }
    fn alpha(&self, i: usize) -> f32 {
        self[i].opacity
    }
    fn rgb(&self, i: usize) -> [f32; 3] {
        self[i].color
    }
    fn scales(&self, i: usize) -> [f32; 3] {
        self[i].scales
    }
    fn lod_scales(&self, i: usize) -> [f32; 4] {
        [self[i].lod_min, self[i].lod_low, self[i].lod_high, self[i].lod_max]
    }
    fn orientation(&self, i: usize) -> [f32; 4] {
        self[i].quaternion
    }
}

fn convert(input_filename: &str, output_filename: &str, meta_filename: &str) {
    let mut splats = if input_filename.ends_with(".ply") {
        read_ply(input_filename)
    } else {
        spz::read_spz(input_filename).unwrap()
    };

    let meta = octlod::create_octree_lod(&mut splats);

    if output_filename.ends_with(".ply") {
        write_ply(&splats, output_filename);

        // Encode meta to JSON
        let meta_json = serde_json::to_string(&meta).unwrap();
        write_txt(&meta_json, meta_filename);
    }

    if output_filename.ends_with(".rad") {
        println!("Sorting in Morton order");
        let start_time = std::time::Instant::now();
        let mut start = 0;
        for &(_, end) in meta.cuts.iter() {
            splats[start..end as usize].sort_by_key(|splat| rad::morton_coord_to_index(splat.center));
            start = end as usize;
        }
        // splats.sort_by_key(|splat| rad::morton_coord_to_index(splat.center));
        let sort_duration = start_time.elapsed();
        println!("Sorting took {:?}", sort_duration);

        println!("Encoding RAD");
        let rad_meta = rad::RadMeta {
            ty: rad::RadType::Gsplat,
            count: splats.len() as u32,
            antialias: Some(true),
            lodMeta: Some(rad::RadLodMeta {
                pixelSizes: meta.cuts.iter().map(|(px, n)| (*px, *n as u32)).collect(),
                boundCenter: meta.bound_center,
                boundRadius: meta.bound_radius,
            }),
            ..Default::default()
        };
        let rad_file = rad::RadFile::new_from_gsplats(&rad_meta, &splats, None, None).unwrap();

        println!("Writing RAD");
        let rad_filename = output_filename.replace(".ply", ".rad");
        let mut writer = BufWriter::new(std::fs::File::create(rad_filename).unwrap());
        rad_file.write_to(&mut writer).unwrap();
        println!("RAD written");
    }
}

#[allow(unused)]
fn convert_dir(input_dir: &str, output_dir: &str) {
    let files = std::fs::read_dir(input_dir).unwrap();
    for file in files {
        let file = file.unwrap();
        let path = file.path();
        if path.extension().unwrap() == "spz" || path.extension().unwrap() == "ply" {
            // Get filename without extension
            let name = path.file_stem().unwrap().to_str().unwrap();
            // Create output filename with PLY
            let output_filename = format!("{}/{}.rad", output_dir, name);
            let meta_filename = format!("{}/{}.meta.json", output_dir, name);
            println!("--------------------------------");
            println!("Converting {} to {} and {}", path.to_str().unwrap(), output_filename, meta_filename);
            convert(&path.to_str().unwrap(), &output_filename, &meta_filename);
            println!("--------------------------------");
        }
    }
}

fn main() {
    // // Read through directory and find all *.spz files: in /Users/asundqui/tasty/tasty_spz/
    // let source = "/Users/asundqui/tasty/tasty_spz/";
    // convert_dir(source, "../../examples/assets/splats/tasty/");

    // // Read through directory and find all *.ply files: in /Users/asundqui/spark/samples/
    // let source = "/Users/asundqui/spark/samples/";
    // convert_dir(source, "../../examples/assets/splats/samples/");

    convert(INPUT_FILENAME, OUTPUT_FILENAME, META_FILENAME);
}
