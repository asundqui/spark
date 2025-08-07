use half::f16;

pub fn raycast_spheres(
    buffer: &[u32], buffer2: Option<&[u32]>, distances: &mut Vec<f32>, 
    origin: [f32; 3], dir: [f32; 3], near: f32, far: f32,
    ln_scale_min: f32, ln_scale_max: f32, min_opacity: f32,
) {
    let quad_a = vec3_dot(dir, dir);

    if let Some(buffer2) = buffer2 {
        let chunks = buffer.len().min(buffer2.len()) / 4;
        for i in 0..chunks {
            let i4 = i * 4;
            let packed = &buffer[i4..i4 + 4];
            if extract_opacity_ext(packed) < min_opacity {
                continue;
            }
            let packed2 = &buffer2[i4..i4 + 4];
            let origin = vec3_sub(origin, extract_center_ext(packed));
            let scale = extract_scale_ext(packed2, ln_scale_min, ln_scale_max);
            if let Some(t) = compute_sphere_t(origin, scale, dir, quad_a) {
                if t >= near && t <= far {
                    distances.push(t);
                }
            }
        }
    } else {
        for packed in buffer.chunks(4) {
            if extract_opacity(packed) < min_opacity {
                continue;
            }

            let origin = vec3_sub(origin, extract_center(packed));
            let scale = extract_scale(packed, ln_scale_min, ln_scale_max);            
            if let Some(t) = compute_sphere_t(origin, scale, dir, quad_a) {
                if t >= near && t <= far {
                    distances.push(t);
                }
            }
        }
    }
}

pub fn raycast_ellipsoids(
    buffer: &[u32], buffer2: Option<&[u32]>, distances: &mut Vec<f32>, 
    origin: [f32; 3], dir: [f32; 3], near: f32, far: f32,
    ln_scale_min: f32, ln_scale_max: f32, min_opacity: f32,
) {
    if let Some(buffer2) = buffer2 {
        let chunks = buffer.len().min(buffer2.len()) / 4;
        for i in 0..chunks {
            let i4 = i * 4;
            let packed = &buffer[i4..i4 + 4];
            if extract_opacity_ext(packed) < min_opacity {
                continue;
            }
            let packed2 = &buffer2[i4..i4 + 4];
            let origin = vec3_sub(origin, extract_center_ext(packed));
            let scale = extract_scale_ext(packed2, ln_scale_min, ln_scale_max);
            let quat = extract_quat_ext(packed2);
            if let Some(t) = compute_ellipsoid_t(origin, scale, quat, dir) {
                if t >= near && t <= far {
                    distances.push(t);
                }
            }
        }
    } else {
        for packed in buffer.chunks(4) {
            if extract_opacity(packed) < min_opacity {
                continue;
            }
        
            let origin = vec3_sub(origin, extract_center(packed));
            let scale = extract_scale(packed, ln_scale_min, ln_scale_max);
            let quat = extract_quat(packed);
            if let Some(t) = compute_ellipsoid_t(origin, scale, quat, dir) {
                if t >= near && t <= far {
                    distances.push(t);
                }
            }
        }
    }
}

fn compute_sphere_t(origin: [f32; 3], scale: [f32; 3], dir: [f32; 3], quad_a: f32) -> Option<f32> {
    // Model the Gsplat as a sphere for faster approximate raycasting
    let radius = (scale[0] + scale[1] + scale[2]) / 3.0;
    let quad_b = vec3_dot(dir, origin);
    let quad_c = vec3_dot(origin, origin) - radius * radius;
    let discriminant = quad_b * quad_b - quad_a * quad_c;
    if discriminant < 0.0 {
        None
    } else {
        Some((-quad_b - discriminant.sqrt()) / quad_a)
    }
}

fn compute_ellipsoid_t(origin: [f32; 3], scale: [f32; 3], quat: [f32; 4], dir: [f32; 3]) -> Option<f32> {
    let inv_quat = [-quat[0], -quat[1], -quat[2], quat[3]];

    // Model the Gsplat as an ellipsoid for higher quality raycasting
    let local_origin = quat_vec(inv_quat, origin);
    let local_dir = quat_vec(inv_quat, dir);

    let min_scale = scale[0].max(scale[1]).max(scale[2]) * 0.01;
    let t = if scale[2] < min_scale {
        // Treat it as a flat elliptical disk
        if local_dir[2].abs() < 1e-6 {
            return None;
        }
        let t = -local_origin[2] / local_dir[2];
        let p_x = local_origin[0] + t * local_dir[0];
        let p_y = local_origin[1] + t * local_dir[1];
        if sqr(p_x / scale[0]) + sqr(p_y / scale[1]) > 1.0 {
            return None;
        }
        t
    } else if scale[1] < min_scale {
        // Treat it as a flat elliptical disk
        if local_dir[1].abs() < 1e-6 {
            return None;
        }
        let t = -local_origin[1] / local_dir[1];
        let p_x = local_origin[0] + t * local_dir[0];
        let p_z = local_origin[2] + t * local_dir[2];
        if sqr(p_x / scale[0]) + sqr(p_z / scale[2]) > 1.0 {
            return None
        }
        t
    } else if scale[0] < min_scale {
        // Treat it as a flat elliptical disk
        if local_dir[0].abs() < 1e-6 {
            return None;
        }
        let t = -local_origin[0] / local_dir[0];
        let p_y = local_origin[1] + t * local_dir[1];
        let p_z = local_origin[2] + t * local_dir[2];
        if sqr(p_y / scale[1]) + sqr(p_z / scale[2]) > 1.0 {
            return None;
        }
        t
    } else {
        let inv_scale = [1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2]];
        let local_origin = vec3_mul(local_origin, inv_scale);
        let local_dir = vec3_mul(local_dir, inv_scale);

        let a = vec3_dot(local_dir, local_dir);
        let b = vec3_dot(local_origin, local_dir);
        let c = vec3_dot(local_origin, local_origin) - 1.0;
        let discriminant = b * b - a * c;
        if discriminant < 0.0 {
            return None;
        }
        (-b - discriminant.sqrt()) / a
    };
    Some(t)
}

pub fn decode_scales(scales: [u8; 3], ln_scale_min: f32, ln_scale_max: f32) -> [f32; 3] {
    let ln_scale_scale = (ln_scale_max - ln_scale_min) / 254.0;
    [
        if scales[0] == 0 { 0.0 } else { (ln_scale_min + (scales[0] - 1) as f32 * ln_scale_scale).exp() },
        if scales[1] == 0 { 0.0 } else { (ln_scale_min + (scales[1] - 1) as f32 * ln_scale_scale).exp() },
        if scales[2] == 0 { 0.0 } else { (ln_scale_min + (scales[2] - 1) as f32 * ln_scale_scale).exp() },
    ]
}

pub fn decode_scales_ext(scales: [u32; 3], ln_scale_min: f32, ln_scale_max: f32) -> [f32; 3] {
    let ln_scale_scale = (ln_scale_max - ln_scale_min) / 510.0;
    [
        if scales[0] == 0 { 0.0 } else { (ln_scale_min + (scales[0] - 1) as f32 * ln_scale_scale).exp() },
        if scales[1] == 0 { 0.0 } else { (ln_scale_min + (scales[1] - 1) as f32 * ln_scale_scale).exp() },
        if scales[2] == 0 { 0.0 } else { (ln_scale_min + (scales[2] - 1) as f32 * ln_scale_scale).exp() },
    ]
}

fn extract_opacity(packed: &[u32]) -> f32 {
    ((packed[0] >> 24) as u8) as f32 / 255.0
}

fn extract_center(packed: &[u32]) -> [f32; 3] {
    let x = f16::from_bits(packed[1] as u16).to_f32();
    let y = f16::from_bits((packed[1] >> 16) as u16).to_f32();
    let z = f16::from_bits(packed[2] as u16).to_f32();
    [x, y, z]
}

fn extract_scale(packed: &[u32], ln_scale_min: f32, ln_scale_max: f32) -> [f32; 3] {
    let unpacked = [packed[3] as u8, (packed[3] >> 8) as u8, (packed[3] >> 16) as u8];
    decode_scales(unpacked, ln_scale_min, ln_scale_max)
}

fn extract_quat(packed: &[u32]) -> [f32; 4] {
    let quant_u = ((packed[2] >> 16) as u8) & 0xff;
    let quant_v = ((packed[2] >> 24) as u8) & 0xff;
    let angle_int = ((packed[3] >> 24) as u8) & 0xff;

    let u_f = quant_u as f32 / 255.0;
    let v_f = quant_v as f32 / 255.0;
    let mut f_x = (u_f - 0.5) * 2.0;
    let mut f_y = (v_f - 0.5) * 2.0;

    let f_z = 1.0 - (f_x.abs() + f_y.abs());
    let t = (-f_z).max(0.0);
    f_x += if f_x >= 0.0 { -t } else { t };
    f_y += if f_y >= 0.0 { -t } else { t };

    let axis_len = (sqr(f_x) + sqr(f_y) + sqr(f_z)).sqrt();
    let (axis_x, axis_y, axis_z) = if axis_len < 1.0e-6 {
        (0.0, 0.0, 0.0)
    } else {
        (f_x / axis_len, f_y / axis_len, f_z / axis_len)
    };

    let theta = angle_int as f32 / 255.0 * std::f32::consts::PI;
    let half_theta = theta * 0.5;
    let (s, w) = half_theta.sin_cos();
    [axis_x * s, axis_y * s, axis_z * s, w]
}

fn extract_opacity_ext(packed: &[u32]) -> f32 {
    f16::from_bits(packed[3] as u16).to_f32()
}

fn extract_center_ext(packed: &[u32]) -> [f32; 3] {
    let x = f32::from_bits(packed[0]);
    let y = f32::from_bits(packed[1]);
    let z = f32::from_bits(packed[2]);
    [x, y, z]
}

fn extract_scale_ext(packed2: &[u32], ln_scale_min: f32, ln_scale_max: f32) -> [f32; 3] {
    let unpacked = [packed2[2] & 0x1ff, (packed2[2] >> 9) & 0x1ff, (packed2[2] >> 18) & 0x1ff];
    decode_scales_ext(unpacked, ln_scale_min, ln_scale_max)
}

fn extract_quat_ext(packed2: &[u32]) -> [f32; 4] {
    let quant_u = packed2[1] & 0x3ff;
    let quant_v = (packed2[1] >> 10) & 0x3ff;
    let angle_int = (packed2[1] >> 20) & 0xfff;

    let u_f = quant_u as f32 / 1023.0;
    let v_f = quant_v as f32 / 1023.0;
    let mut f_x = (u_f - 0.5) * 2.0;
    let mut f_y = (v_f - 0.5) * 2.0;

    let f_z = 1.0 - (f_x.abs() + f_y.abs());
    let t = (-f_z).max(0.0);
    f_x += if f_x >= 0.0 { -t } else { t };
    f_y += if f_y >= 0.0 { -t } else { t };

    let axis_len = (sqr(f_x) + sqr(f_y) + sqr(f_z)).sqrt();
    let (axis_x, axis_y, axis_z) = if axis_len < 1.0e-6 {
        (0.0, 0.0, 0.0)
    } else {
        (f_x / axis_len, f_y / axis_len, f_z / axis_len)
    };

    let theta = angle_int as f32 / 4095.0 * std::f32::consts::PI;
    let half_theta = theta * 0.5;
    let (s, w) = half_theta.sin_cos();
    [axis_x * s, axis_y * s, axis_z * s, w]
}

fn sqr(x: f32) -> f32 {
    x * x
}

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_mul(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn quat_vec(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let q_vec = [q[0], q[1], q[2]];
    let uv = vec3_cross(q_vec, v);
    let uuv = vec3_cross(q_vec, uv);
    [
        v[0] + 2.0 * (q[3] * uv[0] + uuv[0]),
        v[1] + 2.0 * (q[3] * uv[1] + uuv[1]),
        v[2] + 2.0 * (q[3] * uv[2] + uuv[2]),
    ]
}
