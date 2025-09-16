use std::cmp::Reverse;
use ahash::{AHashMap, AHashSet};
use smallvec::SmallVec;
use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;
use std::array;

const MIN_FACTOR: f32 = 4.0;
const DISAPPEAR_FACTOR: f32 = 2.0;

pub type Vec3 = [f32; 3];
pub type Quat = [f32; 4];

pub struct Mat3([[f32; 3]; 3]);

impl Mat3 {
    fn new_identity() -> Self {
        Self([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    }

    fn new_from_symmat3(symmat: &SymMat3) -> Self {
        Self([
            [symmat.0[0], symmat.0[3], symmat.0[4]],
            [symmat.0[3], symmat.0[1], symmat.0[5]],
            [symmat.0[4], symmat.0[5], symmat.0[2]],
        ])
    }

    fn new_scale_quaternion(scale: Vec3, quaternion: Quat) -> Self {
        Self([
            [
                scale[0]
                    * (1.0
                        - 2.0
                            * (quaternion[1] * quaternion[1]
                                + quaternion[2] * quaternion[2])),
                scale[0]
                    * (2.0
                        * (quaternion[0] * quaternion[1]
                            + quaternion[2] * quaternion[3])),
                scale[0]
                    * (2.0
                        * (quaternion[0] * quaternion[2]
                            - quaternion[1] * quaternion[3])),
            ],
            [
                scale[1]
                    * (2.0
                        * (quaternion[0] * quaternion[1]
                            - quaternion[2] * quaternion[3])),
                scale[1]
                    * (1.0
                        - 2.0
                            * (quaternion[0] * quaternion[0]
                                + quaternion[2] * quaternion[2])),
                scale[1]
                    * (2.0
                        * (quaternion[1] * quaternion[2]
                            + quaternion[0] * quaternion[3])),
            ],
            [
                scale[2]
                    * (2.0
                        * (quaternion[0] * quaternion[2]
                            + quaternion[1] * quaternion[3])),
                scale[2]
                    * (2.0
                        * (quaternion[1] * quaternion[2]
                            - quaternion[0] * quaternion[3])),
                scale[2]
                    * (1.0
                        - 2.0
                            * (quaternion[0] * quaternion[0]
                                + quaternion[1] * quaternion[1])),
            ],
        ])
    }

    fn transpose(other: &Self) -> Self {
        Self([
            [other.0[0][0], other.0[1][0], other.0[2][0]],
            [other.0[0][1], other.0[1][1], other.0[2][1]],
            [other.0[0][2], other.0[1][2], other.0[2][2]],
        ])
    }

    fn mul(a: &Self, b: &Self) -> Self {
        Self([
            [
                a.0[0][0] * b.0[0][0] + a.0[0][1] * b.0[1][0] + a.0[0][2] * b.0[2][0],
                a.0[0][0] * b.0[0][1] + a.0[0][1] * b.0[1][1] + a.0[0][2] * b.0[2][1],
                a.0[0][0] * b.0[0][2] + a.0[0][1] * b.0[1][2] + a.0[0][2] * b.0[2][2],
            ],
            [
                a.0[1][0] * b.0[0][0] + a.0[1][1] * b.0[1][0] + a.0[1][2] * b.0[2][0],
                a.0[1][0] * b.0[0][1] + a.0[1][1] * b.0[1][1] + a.0[1][2] * b.0[2][1],
                a.0[1][0] * b.0[0][2] + a.0[1][1] * b.0[1][2] + a.0[1][2] * b.0[2][2],
            ],
            [
                a.0[2][0] * b.0[0][0] + a.0[2][1] * b.0[1][0] + a.0[2][2] * b.0[2][0],
                a.0[2][0] * b.0[0][1] + a.0[2][1] * b.0[1][1] + a.0[2][2] * b.0[2][1],
                a.0[2][0] * b.0[0][2] + a.0[2][1] * b.0[1][2] + a.0[2][2] * b.0[2][2],
            ],
        ])
    }

    fn new_covariance(scale: Vec3, quaternion: Quat) -> Self {
        let rs_t = Self::new_scale_quaternion(scale, quaternion);
        let rs = Self::transpose(&rs_t);
        Self::mul(&rs, &rs_t)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SymMat3([f32; 6]);

impl SymMat3 {
    fn new_zeros() -> Self {
        Self([0.0; 6])
    }

    fn new_from_mat3(mat: &Mat3) -> Self {
        Self([
            mat.0[0][0],
            mat.0[1][1],
            mat.0[2][2],
            // upper triangular off-diagonals: m01, m02, m12
            mat.0[0][1],
            mat.0[0][2],
            mat.0[1][2],
        ])
    }
    
    pub fn new_covariance(scale: Vec3, quaternion: Quat) -> Self {
        Self::new_from_mat3(&Mat3::new_covariance(scale, quaternion))
    }

    fn add_weighted(&mut self, other: &Self, weight: f32) {
        self.0[0] += other.0[0] * weight;
        self.0[1] += other.0[1] * weight;
        self.0[2] += other.0[2] * weight;
        self.0[3] += other.0[3] * weight;
        self.0[4] += other.0[4] * weight;
        self.0[5] += other.0[5] * weight;
    }

    fn new_average(a: &Self, b: &Self) -> Self {
        Self([
            0.5 * (a.0[0] + b.0[0]),
            0.5 * (a.0[1] + b.0[1]),
            0.5 * (a.0[2] + b.0[2]),
            0.5 * (a.0[3] + b.0[3]),
            0.5 * (a.0[4] + b.0[4]),
            0.5 * (a.0[5] + b.0[5]),
        ])
    }

    fn determinant(&self) -> f32 {
        let m00 = self.0[0];
        let m11 = self.0[1];
        let m22 = self.0[2];
        let m01 = self.0[3];
        let m02 = self.0[4];
        let m12 = self.0[5];

        m00 * (m11 * m22 - m12 * m12) -
        m01 * (m01 * m22 - m12 * m02) +
        m02 * (m01 * m12 - m11 * m02)
    }

    fn inverse(&self) -> Option<Self> {
        let m00 = self.0[0];
        let m11 = self.0[1];
        let m22 = self.0[2];
        let m01 = self.0[3];
        let m02 = self.0[4];
        let m12 = self.0[5];

        let det = self.determinant();
        // Use a relative tolerance based on matrix scale (diagonal magnitudes)
        let diag_max = self.0[0].abs().max(self.0[1].abs()).max(self.0[2].abs());
        let rel_tol = 1e-9_f32 * (diag_max * diag_max * diag_max).max(1e-30_f32);
        if det.abs() < rel_tol {
            // println!("Matrix: {:?}", self);
            // panic!("Matrix is singular or too close to singular");
            return None;
        }
        let inv_det = 1.0 / det;
        
        let inv_xx = (m11 * m22 - m12 * m12) * inv_det;
        let inv_yy = (m00 * m22 - m02 * m02) * inv_det;
        let inv_zz = (m00 * m11 - m01 * m01) * inv_det;
        let inv_xy = (m02 * m12 - m01 * m22) * inv_det;
        let inv_xz = (m01 * m12 - m02 * m11) * inv_det;
        let inv_yz = (m01 * m02 - m00 * m12) * inv_det;
        
        Some(Self([inv_xx, inv_yy, inv_zz, inv_xy, inv_xz, inv_yz]))
    }

    pub fn eigens(&self) -> ([f32; 3], [Vec3; 3]) {
        const MAX_ITERS: usize = 32;
        // Relative tolerance based on matrix scale
        let eps: f32 = {
            let s = self.0[0].abs() + self.0[1].abs() + self.0[2].abs();
            1e-6_f32 * s.max(1.0)
        };
        
        let mut a: Mat3 = Mat3::new_from_symmat3(self);
    
        // Accumulate eigenvectors (columns)
        let mut v: Mat3 = Mat3::new_identity();
    
        #[inline]
        fn off_diag_norm2(a: &Mat3) -> f32 {
            let a01 = a.0[0][1];
            let a02 = a.0[0][2];
            let a12 = a.0[1][2];
            a01*a01 + a02*a02 + a12*a12
        }
    
        let mut k = 0;
        while k < MAX_ITERS && off_diag_norm2(&a) > (eps*eps) {
            // choose largest |off-diagonal|
            let mut p = 0usize;
            let mut q = 1usize;
            let mut max_val = a.0[0][1].abs();
            let cand = [(0,2, a.0[0][2].abs()), (1,2, a.0[1][2].abs())];
            for (i,j,val) in cand {
                if val > max_val {
                    max_val = val; p = i; q = j;
                }
            }
    
            let apq = a.0[p][q];
            if apq.abs() > eps {
                let app = a.0[p][p];
                let aqq = a.0[q][q];
                let tau = aqq - app;
                let phi = 0.5 * (2.0 * apq).atan2(tau);
                let (c, s) = (phi.cos(), phi.sin());
    
                // A = J^T A J  (symmetric update)
                for r in 0..3 {
                    let arp = a.0[r][p];
                    let arq = a.0[r][q];
                    a.0[r][p] = c*arp - s*arq;
                    a.0[r][q] = s*arp + c*arq;
                }
                for r in 0..3 {
                    let apr = a.0[p][r];
                    let aqr = a.0[q][r];
                    a.0[p][r] = c*apr - s*aqr;
                    a.0[q][r] = s*apr + c*aqr;
                }
                a.0[p][q] = 0.0;
                a.0[q][p] = 0.0;
    
                // V = V J
                for r in 0..3 {
                    let vrp = v.0[r][p];
                    let vrq = v.0[r][q];
                    v.0[r][p] = c*vrp - s*vrq;
                    v.0[r][q] = s*vrp + c*vrq;
                }
            }
            k += 1;
        }
    
        // Eigenvalues on diagonal; columns of V are eigenvectors
        let vals = [a.0[0][0], a.0[1][1], a.0[2][2]];
        let mut vecs: [Vec3; 3] = [
            [v.0[0][0], v.0[1][0], v.0[2][0]],
            [v.0[0][1], v.0[1][1], v.0[2][1]],
            [v.0[0][2], v.0[1][2], v.0[2][2]],
        ];
    
        // Normalize vectors
        for j in 0..3 {
            let n = (vecs[j][0]*vecs[j][0] + vecs[j][1]*vecs[j][1] + vecs[j][2]*vecs[j][2]).sqrt();
            if n > 0.0 {
                vecs[j][0] /= n; vecs[j][1] /= n; vecs[j][2] /= n;
            }
        }
    
        // Sort by descending eigenvalue, keeping alignment
        let mut idx = [0usize, 1, 2];
        idx.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]));
        let sorted_vals = [vals[idx[0]], vals[idx[1]], vals[idx[2]]];
        let sorted_vecs = [vecs[idx[0]], vecs[idx[1]], vecs[idx[2]]];
    
        (sorted_vals, sorted_vecs)
    }

    pub fn positive_eigens(&self) -> ([f32; 3], [Vec3; 3]) {
        let (vals, mut vecs_cols) = self.eigens();

        // Ensure right-handed basis (determinant > 0)
        let det =
            vecs_cols[0][0] * (vecs_cols[1][1] * vecs_cols[2][2] - vecs_cols[1][2] * vecs_cols[2][1]) -
            vecs_cols[0][1] * (vecs_cols[1][0] * vecs_cols[2][2] - vecs_cols[1][2] * vecs_cols[2][0]) +
            vecs_cols[0][2] * (vecs_cols[1][0] * vecs_cols[2][1] - vecs_cols[1][1] * vecs_cols[2][0]);

        if det < 0.0 {
            vecs_cols[2] = [-vecs_cols[2][0], -vecs_cols[2][1], -vecs_cols[2][2]];
        }
        (vals, vecs_cols)
    }
}

pub fn quat_from_eigvecs([ex, ey, ez]: [Vec3; 3]) -> Quat {
    // Rotation matrix with columns = ex, ey, ez
    let r00 = ex[0]; let r01 = ey[0]; let r02 = ez[0];
    let r10 = ex[1]; let r11 = ey[1]; let r12 = ez[1];
    let r20 = ex[2]; let r21 = ey[2]; let r22 = ez[2];

    // Robust matrix -> quaternion
    let trace = r00 + r11 + r22;
    let (x, y, z, w);
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0; // 4w
        w = 0.25 * s;
        x = (r21 - r12) / s;
        y = (r02 - r20) / s;
        z = (r10 - r01) / s;
    } else if r00 > r11 && r00 > r22 {
        let s = (1.0 + r00 - r11 - r22).sqrt() * 2.0; // 4x
        x = 0.25 * s;
        y = (r01 + r10) / s;
        z = (r02 + r20) / s;
        w = (r21 - r12) / s;
    } else if r11 > r22 {
        let s = (1.0 + r11 - r00 - r22).sqrt() * 2.0; // 4y
        x = (r01 + r10) / s;
        y = 0.25 * s;
        z = (r12 + r21) / s;
        w = (r02 - r20) / s;
    } else {
        let s = (1.0 + r22 - r00 - r11).sqrt() * 2.0; // 4z
        x = (r02 + r20) / s;
        y = (r12 + r21) / s;
        z = 0.25 * s;
        w = (r10 - r01) / s;
    }

    // Normalize and enforce w >= 0 for a consistent sign
    let mut q = [x, y, z, w];
    let n = (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]).sqrt();
    if n > 0.0 {
        q[0] /= n; q[1] /= n; q[2] /= n; q[3] /= n;
    }
    if q[3] < 0.0 { q[0] = -q[0]; q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3]; }
    q
}

pub struct GridHash<T: PartialEq + Copy> {
    step: f32,
    cells: AHashMap<[i64; 3], SmallVec<[T; 4]>>,
}

impl<T: PartialEq + Copy> GridHash<T> {
    pub fn new(step: f32) -> Self {
        Self {
            step,
            cells: AHashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.cells.clear();
    }

    pub fn count(&self) -> usize {
        self.cells.len()
    }

    pub fn get_cell(&self, pos: Vec3) -> [i64; 3] {
        pos.map(|p| (p / self.step).floor() as i64)
    }

    pub fn add(&mut self, pos: Vec3, value: T) {
        let key = self.get_cell(pos);
        self.cells
            .entry(key)
            .or_default()
            .push(value);
    }

    pub fn get(&self, pos: Vec3) -> Option<&[T]> {
        self.cells.get(&self.get_cell(pos)).map(|v| v.as_slice())
    }

    pub fn remove(&mut self, pos: Vec3, value: T) {
        self.cells
            .get_mut(&self.get_cell(pos))
            .unwrap()
            .retain(|v| *v != value);
    }

    pub fn get_around(&self, pos: Vec3) -> SmallVec<[T; 8]> {
        let mut result = SmallVec::new();
        let cell = self.get_cell(pos);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbor = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                    if let Some(splats) = self.cells.get(&neighbor) {
                        result.extend_from_slice(splats);
                    }
                }
            }
        }
        result
    }
}

fn ellipsoid_area(scales: [f32; 3]) -> f32 {
    const P: f32 = 1.6075;
    let numerator = (scales[0] * scales[1]).powf(P) + (scales[0] * scales[2]).powf(P) + (scales[1] * scales[2]).powf(P);
    4.0 * std::f32::consts::PI * (numerator / 3.0).powf(1.0 / P)
}

#[derive(Debug, Clone, Default)]
pub struct Gsplat {
    pub center: Vec3,
    pub scales: Vec3,
    pub quaternion: Quat,
    pub opacity: f32,
    pub color: Vec3,
    pub cov: SymMat3,
    pub lod_min: f32,
    pub lod_low: f32,
    pub lod_high: f32,
    pub lod_max: f32,
    pub output: bool,
    pub children: [i32; 2],
}

impl Gsplat {
    fn get_scale(scales: &Vec3) -> f32 {
        scales[0].max(scales[1]).max(scales[2])
    }

    pub fn scale(&self) -> f32 {
        Self::get_scale(&self.scales)
    }

    pub fn adjustment(&self) -> f32 {
        if self.opacity > 1.0 {
            (1.0 + 2.0 * self.opacity.ln()).sqrt()
        } else {
            1.0
        }
    }

    pub fn adjusted_scales(&self) -> Vec3 {
        let adjustment = self.adjustment();
        self.scales.map(|s| s * adjustment)
    }

    pub fn size(&self) -> f32 {
        self.scale() * self.adjustment() * 2.0
    }

    pub fn distance(&self, other: &Gsplat) -> f32 {
        let dist2 = (self.center[0] - other.center[0]).powi(2) +
            (self.center[1] - other.center[1]).powi(2) +
            (self.center[2] - other.center[2]).powi(2);
        dist2.sqrt()
    }

    pub fn bhattacharyya_distance(&self, other: &Gsplat) -> f32 {
        // println!("Bhattacharyya distance between {:?}", self);
        // println!("  and {:?}", other);
        // Compute the average covariance
        let sigma = SymMat3::new_average(&self.cov, &other.cov);
        // println!("Sigma: {:?}", sigma);
        let Some(sigma_inv) = sigma.inverse() else {
            return f32::INFINITY;
        };

        // First term: (1/8) * diff^T * Sigma_inv * diff for symmetric 3x3 stored as [xx, yy, zz, xy, xz, yz]
        let diff: Vec3 = array::from_fn(|i| other.center[i] - self.center[i]);
        let dx = diff[0];
        let dy = diff[1];
        let dz = diff[2];
        let inv = &sigma_inv.0;
        let quad = inv[0] * dx * dx
            + inv[1] * dy * dy
            + inv[2] * dz * dz
            + 2.0 * inv[3] * dx * dy
            + 2.0 * inv[4] * dx * dz
            + 2.0 * inv[5] * dy * dz;
        let term1_val = 0.125 * quad;

        // Second term: (1/2) * ln(det(Sigma) / sqrt(det(Sigma_A)*det(Sigma_B)))
        let det_sigma = sigma.determinant();
        let det_a = self.cov.determinant();
        let det_b = other.cov.determinant();
        let term2 = 0.5 * (det_sigma / (det_a * det_b).sqrt()).ln();

        // Bhattacharyya distance
        term1_val + term2
    }

    pub fn bhattacharyya_coeff(&self, other: &Gsplat) -> f32 {
        (-self.bhattacharyya_distance(other)).exp()
        // let center_delta2 = (self.center[0] - other.center[0]).powi(2) +
        //     (self.center[1] - other.center[1]).powi(2) +
        //     (self.center[2] - other.center[2]).powi(2);
        // (-center_delta2).exp()
    }

    pub fn similarity_metric(&self, other: &Gsplat) -> f32 {
        let color_delta2 = (self.color[0] - other.color[0]).powi(2) +
            (self.color[1] - other.color[1]).powi(2) +
            (self.color[2] - other.color[2]).powi(2);
        self.bhattacharyya_coeff(other) * (-color_delta2).exp()
    }

    pub fn area(&self) -> f32 {
        ellipsoid_area(self.scales)
    }
        
    pub fn new_merged(a: &Self, b: &Self) -> Self {
        let [weight_a, weight_b] = [a.area() * a.opacity, b.area() * b.opacity];
        let total_weight = weight_a + weight_b;
        let [weight_a, weight_b] = if total_weight == 0.0 {
            [0.5, 0.5]
        } else {
            [weight_a / total_weight, weight_b / total_weight]
        };

        let new_center: Vec3 = array::from_fn(|i| weight_a * a.center[i] + weight_b * b.center[i]);
        let new_color: Vec3 = array::from_fn(|i| weight_a * a.color[i] + weight_b * b.color[i]);

        let mut covariance = SymMat3::new_zeros();
        for (weight, center, cov) in [
            (weight_a, a.center, &a.cov),
            (weight_b, b.center, &b.cov),
        ] {
            let center_delta: Vec3 = array::from_fn(|i| center[i] - new_center[i]);
            covariance.add_weighted(&SymMat3([
                center_delta[0] * center_delta[0] + cov.0[0],
                center_delta[1] * center_delta[1] + cov.0[1],
                center_delta[2] * center_delta[2] + cov.0[2],
                center_delta[0] * center_delta[1] + cov.0[3],
                center_delta[0] * center_delta[2] + cov.0[4],
                center_delta[1] * center_delta[2] + cov.0[5],
            ]), weight);
        }

        let (vals, vecs) = covariance.positive_eigens();
        let new_quaternion = quat_from_eigvecs(vecs);
        let new_scales = [vals[0].max(0.0).sqrt(), vals[1].max(0.0).sqrt(), vals[2].max(0.0).sqrt()];
        let new_opacity = total_weight / ellipsoid_area(new_scales);

        if new_scales[0].is_nan() || new_scales[1].is_nan() || new_scales[2].is_nan() {
            println!("area_a: {:?}", a.area());
            println!("area_b: {:?}", b.area());
            println!("a.opacity: {:?}", a.opacity);
            println!("b.opacity: {:?}", b.opacity);
            println!("weight_a: {:?}", weight_a);
            println!("weight_b: {:?}", weight_b);
            println!("total_weight: {:?}", total_weight);
            println!("new_center: {:?}", new_center);
            println!("new_color: {:?}", new_color);
            println!("covariance: {:?}", covariance);
            println!("vals: {:?}", vals);
            println!("vecs: {:?}", vecs);
            println!("new_quaternion: {:?}", new_quaternion);
            println!("new_scales: {:?}", new_scales);
            println!("total_weight: {:?}", total_weight);
            println!("ellipsoid_area: {:?}", ellipsoid_area(new_scales));
            println!("new_opacity: {:?}", new_opacity);
        }

        for i in 0..3 {
            assert!(!new_scales[i].is_nan());
        }

        Self {
            center: new_center,
            scales: new_scales,
            quaternion: new_quaternion,
            opacity: new_opacity,
            color: new_color,
            cov: covariance,
            ..Default::default()
        }
    }
}


#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Priority(Reverse<OrderedFloat<f32>>);

impl Priority {
    fn new(value: f32) -> Self {
        Self(Reverse(OrderedFloat(value)))
    }
}


pub fn create_lod(splats: &mut Vec<Gsplat>) {
    let initial_splats = splats.len();

    for splat in splats.iter_mut() {
        splat.cov = SymMat3::new_covariance(splat.scales, splat.quaternion);
        splat.children = [-1, -1];
        splat.output = true;
    }

    // Find minimum of maximum scales of all splats
    let min_size = splats.iter().fold(f32::INFINITY, |min, s| min.min(s.size()));
    let mut level = min_size.log2().floor().exp2();
    let mut active: AHashSet<i32> = (0..splats.len() as i32).collect();

    while active.len() > 1 {
        println!("*** Iteration level={}", level);
        println!("# active={}", active.len());

        let mut grid = GridHash::new(level);
        let mut priority = BinaryHeap::<(Priority, i32, i32)>::new();
        let mut splat_count = 0;
        
        for &index in active.iter() {
            let splat = &splats[index as usize];
            if splat.size() < level {
                grid.add(splat.center, index);
                splat_count += 1;
            }
        }

        println!("# splats in level: {}", splat_count);
        println!("Finding neighbors in {} grids", grid.count());

        for &index in active.iter() {
            let splat = &splats[index as usize];
            if splat.size() < level {
                for neighbor in grid.get_around(splat.center) {
                    if neighbor > index && active.contains(&neighbor) {
                        let neighbor_splat = &splats[neighbor as usize];
                        if active.contains(&neighbor) && splat.distance(neighbor_splat) < level {
                            let metric = splat.similarity_metric(neighbor_splat);
                            priority.push((Priority::new(metric), index, neighbor));
                        }
                    }
                }
            }
        }

        println!("Merging from queue of size {}", priority.len());

        while let Some((Priority(_metric), i, j)) = priority.pop() {
            if !active.contains(&i) || !active.contains(&j) {
                continue;
            }
            let splat_i = &splats[i as usize];
            let splat_j = &splats[j as usize];
            active.remove(&i);
            active.remove(&j);
            grid.remove(splat_i.center, i);
            grid.remove(splat_j.center, j);

            let mut merged = Gsplat::new_merged(splat_i, splat_j);
            merged.children = [i, j];
            let new_index = splats.len() as i32;

            if merged.size() < level {
                for neighbor in grid.get_around(merged.center) {
                    if active.contains(&neighbor) {
                        let neighbor_splat = &splats[neighbor as usize];
                        if neighbor_splat.size() < level && merged.distance(neighbor_splat) < level {
                            let metric = merged.similarity_metric(neighbor_splat);
                            priority.push((Priority::new(metric), neighbor, new_index));
                        }
                    }
                }
                grid.add(merged.center, new_index);
            }

            splats.push(merged);
            active.insert(new_index);

            if (new_index % 10000) == 0 {
                println!("new_index={}", new_index);
            }
        }

        for &index in &active {
            splats[index as usize].output = true;
        }

        level *= 2.0;
    }

    println!("Initial # splats: {}", initial_splats);
    println!("Total splats before consolidating: {}", splats.len());

    let mut children = Vec::new();
    let mut stack = Vec::new();

    for i in 0..splats.len() {
        if !splats[i].output {
            continue;
        }
        if splats[i].children[0] == -1 {
            let size = splats[i].size();
            splats[i].lod_min = 0.0;
            splats[i].lod_low = 0.0;
            splats[i].lod_high = size * MIN_FACTOR;
            splats[i].lod_max = size * MIN_FACTOR * DISAPPEAR_FACTOR;
            continue;
        }

        children.clear();
        stack.clear();
        stack.push(splats[i].children[0]);
        stack.push(splats[i].children[1]);
        while let Some(child) = stack.pop() {
            let child = child as usize;
            if !splats[child].output {
                stack.push(splats[child].children[0]);
                stack.push(splats[child].children[1]);
            } else {
                children.push(child);
            }
        }

        let lod_min = children.iter().fold(f32::NEG_INFINITY, |lod, &c| lod.max(splats[c].size()));
        let size = splats[i].size();
        splats[i].lod_min = lod_min;
        splats[i].lod_low = size;
        splats[i].lod_high = size * MIN_FACTOR;
        splats[i].lod_max = size * MIN_FACTOR * DISAPPEAR_FACTOR;

        for &child in &children {
            splats[child].lod_max = splats[child].lod_max.min(size);
            splats[child].lod_high = splats[child].lod_high.min(lod_min).min(splats[child].lod_max);
            splats[child].lod_low = splats[child].lod_low.min(splats[child].lod_high);
            splats[child].lod_min = splats[child].lod_min.min(splats[child].lod_low);
        }
    }

    splats.retain(|s| s.output);
    splats.sort_by_key(|s| OrderedFloat(s.lod_max));

    println!("Total splats after consolidating: {}", splats.len());
}
