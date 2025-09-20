use std::io::Write;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use ahash::{AHashMap, AHashSet};
use ordered_float::OrderedFloat;
use smallvec::SmallVec;

use crate::{lod::{Gsplat, SymMat3}, LodMeta};

const MIN_FACTOR: f32 = 2.0;
const DISAPPEAR_FACTOR: f32 = 2.0;
const CHUNK_SPLATS: usize = 100000;
const MAX_CHUNK_SPLATS: usize = 150000;
const CHUNK_LEVELS: usize = 1;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct MergePriority(Reverse<OrderedFloat<f32>>);

impl MergePriority {
    fn new(value: f32) -> Self {
        Self(Reverse(OrderedFloat(value)))
    }
}

type Octree<T> = AHashMap<i32, AHashMap<[i64; 3], T>>;

fn create_splat_tree(splats: &mut Vec<Gsplat>) {
    let mut octree: Octree<SmallVec<[i32; 4]>> = Octree::new();
    let mut min_size = f32::INFINITY;

    for (index, splat) in splats.iter_mut().enumerate() {
        splat.cov = SymMat3::new_covariance(splat.scales, splat.quaternion);
        splat.children = [-1, -1];
        splat.output = true;

        let size = splat.size();
        min_size = min_size.min(size);

        let lod = size.log2().ceil() as i32;
        let lod_size = (lod as f32).exp2();
        let cell = splat.center.map(|x| (x / lod_size).floor() as i64);
        octree.entry(lod).or_default().entry(cell).or_default().push(index as i32);
    }

    let mut lod = min_size.log2().ceil() as i32;
    let mut active: AHashSet<i32> = (0..splats.len() as i32).collect();
    let mut merge_queue: BinaryHeap<(MergePriority, i32, i32)> = BinaryHeap::new();

    while active.len() > 1 {
        let lod_size = (lod as f32).exp2();
        println!("Merging LOD {} (size={}): #active={}", lod, lod_size, active.len());

        let mut lod_count = 0;
        for &index in active.iter() {
            let splat = &splats[index as usize];
            if splat.size() <= lod_size {
                lod_count += 1;
                let cell = splat.center.map(|x| (x / lod_size).floor() as i64);
                for z in (cell[2] - 1)..=(cell[2] + 1) {
                    for y in (cell[1] - 1)..=(cell[1] + 1) {
                        for x in (cell[0] - 1)..=(cell[0] + 1) {
                            if let Some(neighbors) = octree.get(&lod).and_then(|level| level.get(&[x, y, z])) {
                                for &neighbor in neighbors.iter() {
                                    if neighbor > index && active.contains(&neighbor) {
                                        let neighbor_splat = &splats[neighbor as usize];
                                        if splat.distance(neighbor_splat) < lod_size {
                                            let metric = splat.similarity_metric(neighbor_splat);
                                            merge_queue.push((MergePriority::new(metric), index, neighbor));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("* {} splats within LOD", lod_count);

        println!("- Initial merge queue size: {}", merge_queue.len());
        while let Some((MergePriority(_metric), index_i, index_j)) = merge_queue.pop() {
            if (merge_queue.len() % 1000000) == 0 {
                print!(".");
                std::io::stdout().flush().unwrap();
            }
            if !active.contains(&index_i) || !active.contains(&index_j) {
                continue;
            }

            let splat_i = &splats[index_i as usize];
            let splat_j = &splats[index_j as usize];
            for (center, index) in [(splat_i.center, index_i), (splat_j.center, index_j)] {
                let cell = center.map(|x| (x / lod_size).floor() as i64);
                let level = octree.get_mut(&lod).unwrap();
                let octree_cell = level.get_mut(&cell).unwrap();
                octree_cell.retain(|x| *x != index);
                if octree_cell.is_empty() {
                    level.remove(&cell);
                }
            }
            active.remove(&index_i);
            active.remove(&index_j);

            let mut merged = Gsplat::new_merged(splat_i, splat_j);
            let merged_size = merged.size();
            merged.children = [index_i, index_j];
            let new_index = splats.len() as i32;

            if merged_size <= lod_size {
                let cell = merged.center.map(|x| (x / lod_size).floor() as i64);
                for z in (cell[2] - 1)..=(cell[2] + 1) {
                    for y in (cell[1] - 1)..=(cell[1] + 1) {
                        for x in (cell[0] - 1)..=(cell[0] + 1) {
                            if let Some(neighbors) = octree.get(&lod).and_then(|level| level.get(&[x, y, z])) {
                                for &neighbor in neighbors.iter() {
                                    if active.contains(&neighbor) {
                                        let neighbor_splat = &splats[neighbor as usize];
                                        if merged.distance(neighbor_splat) < lod_size {
                                            let metric = merged.similarity_metric(neighbor_splat);
                                            merge_queue.push((MergePriority::new(metric), neighbor, new_index));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let merged_lod = if merged_size <= lod_size { lod } else { (merged_size.log2().ceil() as i32).max(lod + 1) };
            let merged_lod_size = (merged_lod as f32).exp2();
            let merged_cell = merged.center.map(|x| (x / merged_lod_size).floor() as i64);
            octree.entry(merged_lod).or_default().entry(merged_cell).or_default().push(new_index);

            splats.push(merged);
            active.insert(new_index);

            if (new_index % 10000) == 0 {
                print!("+");
                std::io::stdout().flush().unwrap();
            }
        }

        let next_lod = lod + 1;
        let next_lod_size = (next_lod as f32).exp2();

        for &index in active.iter() {
            let splat = &splats[index as usize];
            if splat.size() <= lod_size {
                let cell = splat.center.map(|x| (x / lod_size).floor() as i64);
                let level = octree.get_mut(&lod).unwrap();
                let octree_cell = level.get_mut(&cell);
                // if octree_cell.is_none() {
                //     println!("\noctree_cell is none: index={}, size={}, lod_size={}", index, splat.size(), lod_size);
                // }
                let octree_cell = octree_cell.unwrap();
                octree_cell.retain(|x| *x != index);
                if octree_cell.is_empty() {
                    level.remove(&cell);
                }

                let new_cell = splat.center.map(|x| (x / next_lod_size).floor() as i64);
                octree.entry(next_lod).or_default().entry(new_cell).or_default().push(index);
            } else {
                splats[index as usize].output = true;
            }
        }

        // let lod_level_count = octree.get(&lod).unwrap().len();
        // println!("\nlod_level_count={}", lod_level_count);
        // if lod_level_count > 0 {
        //     for (cell, splat_indices) in octree.get(&lod).unwrap().iter() {
        //         println!("cell={:?}, splats={:?}", cell, splat_indices);
        //         for &splat in splat_indices.iter() {
        //             println!("splat={:?}", splats[splat as usize]);
        //             let cell = splats[splat as usize].center.map(|x| (x / lod_size).floor() as i64);
        //             let size = splats[splat as usize].size();
        //             println!("- size={}, size_log2={}, size_log2_ceil={}, cell={:?}", size, size.log2(), size.log2().ceil(), cell);
        //         }
        //     }
        // }

        lod = next_lod;
        println!();
    }

    drop(merge_queue);
    drop(octree);

    println!("Consolidating and computing LoD ranges");
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
    splats.sort_by_key(|s| Reverse(OrderedFloat(s.lod_max)));
    println!("Total splats after consolidating: {}", splats.len());
}

pub fn create_octree_lod(splats:  &mut Vec<Gsplat>) -> LodMeta {
    // let min_scale = (-16.0f32).exp();
    // for splat in splats.iter_mut() {
    //     splat.scales = splat.scales.map(|s| s.max(min_scale));
    // }

    let (bmin, bmax) = splats.iter().fold(([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]), |(bmin, bmax), s| (
        [bmin[0].min(s.center[0]), bmin[1].min(s.center[1]), bmin[2].min(s.center[2])],
        [bmax[0].max(s.center[0]), bmax[1].max(s.center[1]), bmax[2].max(s.center[2])]
    ));
    let bcenter = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1]), 0.5 * (bmin[2] + bmax[2])];
    let bradius = splats.iter().fold(0.0f32, |bradius, s| {
        let r2 = (s.center[0] - bcenter[0]).powi(2) + (s.center[1] - bcenter[1]).powi(2) + (s.center[2] - bcenter[2]).powi(2);
        bradius.max(r2.sqrt())
    });
    println!("Bounding box: {:?} - {:?}", bmin, bmax);
    println!("Bounding sphere: {:?} - {}", bcenter, bradius);

    create_splat_tree(splats);

    let max_lod = splats[0].lod_max.log2().ceil() as i32;
    let min_lod = splats.last().unwrap().lod_max.log2().ceil() as i32;

    let range = (bmax[0] - bmin[0]).max(bmax[1] - bmin[1]).max(bmax[2] - bmin[2]);
    let max_lod = max_lod.max(range.log2().ceil() as i32);
    let max_lod_size = (max_lod as f32).exp2();
    let cmin = bmin.map(|x| (x / max_lod_size).floor() as i64);
    let cmax = bmax.map(|x| (x / max_lod_size).floor() as i64);
    println!("Max LOD: {} => {}", max_lod, (max_lod as f32).exp2());

    // let mut hexes: Octree<u32> = Octree::new();
    // for splat in splats.iter() {
    //     let lod = splat.lod_max.log2().ceil() as i32;
    //     let lod_size = (lod as f32).exp2();
    //     let cell = splat.center.map(|x| (x / 16.0 / lod_size).floor() as i64);
    //     *hexes.entry(lod).or_default().entry(cell).or_default() += 1;
    // }

    // for lod in min_lod..=max_lod {
    //     if let Some(level) = hexes.get(&lod) {
    //         let mut cell_counts: Vec<_> = level.values().copied().collect();
    //         cell_counts.sort();
    //         let total_splats = cell_counts.iter().sum();
    //         let mut remaining_splats: u32 = total_splats;
    //         let total_multi: u32 = cell_counts.iter().filter(|&c| *c > 1).copied().sum();
    //         let multi_cells = cell_counts.iter().filter(|&c| *c > 1).count();
    //         let max_count = *cell_counts.last().unwrap();

    //         let mut c10 = 0;
    //         let mut c100 = 0;
    //         let mut c1000 = 0;
    //         let mut c10000 = 0;
    //         let mut c100000 = 0;
    //         let mut c1000000 = 0;
    //         let mut c10000000 = 0;
    //         let mut num_cells = 0;
    //         while let Some(count) = cell_counts.pop() {
    //             num_cells += 1;

    //             remaining_splats -= count;
    //             if c10 == 0 && remaining_splats <= 10 {
    //                 c10 = num_cells;
    //             }
    //             if c100 == 0 && remaining_splats <= 100 {
    //                 c100 = num_cells;
    //             }
    //             if c1000 == 0 && remaining_splats <= 1000 {
    //                 c1000 = num_cells;
    //             }
    //             if c10000 == 0 && remaining_splats <= 10000 {
    //                 c10000 = num_cells;
    //             }
    //             if c100000 == 0 && remaining_splats <= 100000 {
    //                 c100000 = num_cells;
    //             }
    //             if c1000000 == 0 && remaining_splats <= 1000000 {
    //                 c1000000 = num_cells;
    //             }
    //             if c10000000 == 0 && remaining_splats <= 10000000 {
    //                 c10000000 = num_cells;
    //             }
    //         }
            
    //         println!("Hexes: lod={}: {} cells, {} splats, max_count={}, multi_cells={}, total_multi={}", lod, level.len(), total_splats, max_count, multi_cells, total_multi);
    //         println!("- c10={}, c100={}, c1000={}, c10000={}, c100000={}, c1000000={}, c10000000={}", c10, c100, c1000, c10000, c100000, c1000000, c10000000);
    //     }
    // }
    // drop(hexes);

    #[derive(Default)]
    struct Node {
        subtree_count: usize,
        splats: SmallVec<[i32; 4]>,
    }

    let mut octree: Octree<Node> = Octree::new();
    for (index, splat) in splats.iter().enumerate() {
        let mut lod = splat.lod_max.log2().ceil() as i32;
        let lod_size = (lod as f32).exp2();
        let mut cell = splat.center.map(|x| (x / lod_size).floor() as i64);
        octree.entry(lod).or_default().entry(cell).or_default().splats.push(index as i32);

        while lod < max_lod {
            lod += 1;
            cell = cell.map(|x| x >> 1);
            octree.entry(lod).or_default().entry(cell).or_default().subtree_count += 1;
        }
    }

    for lod in min_lod..=max_lod {
        let level = octree.get(&lod).unwrap();
        let nonzero_count = level.iter().fold(0, |count, (_, node)| count + if node.splats.is_empty() { 0 } else { 1 });
        let level_splats = level.iter().fold(0, |count, (_, node)| count + node.splats.len());
        println!("lod={}: {} cells ({} nonzero), {} splats", lod, level.len(), nonzero_count, level_splats);
    }

    //

    fn recurse_chunk(octree: &mut Octree<Node>, lod: i32, cell: [i64; 3], max_splats_per_chunk: usize) {
        let Some(node_splats) = octree.get_mut(&lod).and_then(|level| level.get_mut(&cell).map(|node| node.splats.len())) else {
            return;
        };

        let mut child_splats = Vec::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let child_cell = [2 * cell[0] + x, 2 * cell[1] + y, 2 * cell[2] + z];
                    if octree.get(&(lod - 1)).and_then(|level| level.get(&child_cell)).is_some() {
                        recurse_chunk(octree, lod - 1, child_cell, max_splats_per_chunk);
                        let child = octree.get_mut(&(lod - 1)).unwrap().get_mut(&child_cell).unwrap();
                        child_splats.push((child_cell, child.splats.len()));
                    }
                }
            }
        }

        let mut total_splats: usize = child_splats.iter().map(|(_, count)| count).sum();
        if (total_splats + node_splats) <= max_splats_per_chunk {
            for z in 0..2 {
                for y in 0..2 {
                    for x in 0..2 {
                        let child_cell = [2 * cell[0] + x, 2 * cell[1] + y, 2 * cell[2] + z];
                        if let Some(child) = octree.get_mut(&(lod - 1)).and_then(|level| level.get_mut(&child_cell).map(|node| &mut node.splats)) {
                            let temp: Vec<_> = child.drain(..).collect();
                            octree.get_mut(&lod).unwrap().get_mut(&cell).unwrap().splats.extend(temp);
                        }
                    }
                }
            }
        } else {
            // println!("Exceeded capacity: total_splats={}, node_splats={}", total_splats, node_splats);
            child_splats.sort_by_key(|(_, count)| *count);
            total_splats = 0;
            let mut child_cells: SmallVec<[[i64; 3]; 8]> = SmallVec::new();
            for (index, (child_cell, count)) in child_splats.into_iter().enumerate() {
                if (total_splats + count) > max_splats_per_chunk {
                    println!("Partial child chunk lod={}: index={}, total_splats={}, child_cells={:?}", lod - 1, index, total_splats, child_cells.drain(..));
                    total_splats = 0;
                }
                total_splats += count;
                child_cells.push(child_cell);
            }
            if total_splats > 0 {
                println!("Final child chunk lod={}: total_splats={}, child_cells={:?}", lod - 1, total_splats, child_cells.drain(..));
            }
        }
    }

    for z in cmin[2]..=cmax[2] {
        for y in cmin[1]..=cmax[1] {
            for x in cmin[0]..=cmax[0] {
                recurse_chunk(&mut octree, max_lod, [x, y, z], CHUNK_SPLATS);
                let node = octree.get(&max_lod).unwrap().get(&[x, y, z]).unwrap();
                println!("Final root chunk {:?}: lod={}, {} splats", [x, y, z], max_lod, node.splats.len());
            }
        }
    }


    // fn compute_sub_counts(octree: &Octree<Node>, lod: i32, cell: [i64; 3], start_lod: i32, counts: &mut Vec<usize>) {
    //     if let Some(node) = octree.get(&lod).and_then(|level| level.get(&cell)) {
    //         if lod <= start_lod {
    //             let count_index = (start_lod - lod) as usize;
    //             if count_index >= counts.len() {
    //                 counts.resize(count_index + 1, 0);
    //             }
    //             counts[count_index] += node.splats.len();
    //         }
    //         for z in 0..2 {
    //             for y in 0..2 {
    //                 for x in 0..2 {
    //                     compute_sub_counts(octree, lod - 1, [2 * cell[0] + x, 2 * cell[1] + y, 2 * cell[2] + z], start_lod, counts);
    //                 }
    //             }
    //         }
    //     }
    // }

    // fn compute_subspace_counts(octree: &Octree<Node>, subspace: (i32, Vec<[i64; 3]>), start_lod: i32, counts: &mut Vec<usize>) {
    //     for &cell in subspace.1.iter() {
    //         compute_sub_counts(octree, subspace.0, cell, start_lod, counts);
    //     }
    // }

    // fn compute_cumulative_counts(counts: &mut Vec<usize>) {
    //     for i in 1..counts.len() {
    //         counts[i] += counts[i - 1];
    //     }
    // }

    // fn collect_sub_splats(octree: &Octree<Node>, lod: i32, cell: [i64; 3], start_lod: i32, num_lods: i32, splats: &mut Vec<i32>) {
    //     if let Some(node) = octree.get(&lod).and_then(|level| level.get(&cell)) {
    //         if lod <= start_lod {
    //             splats.extend(node.splats.iter().copied());
    //         }

    //         if lod > (start_lod - num_lods + 1) {
    //             for z in 0..2 {
    //                 for y in 0..2 {
    //                     for x in 0..2 {
    //                         collect_sub_splats(octree, lod - 1, [2 * cell[0] + x, 2 * cell[1] + y, 2 * cell[2] + z], start_lod, num_lods, splats);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    // fn collect_subspace_splats(octree: &Octree<Node>, subspace: (i32, Vec<[i64; 3]>), start_lod: i32, num_lods: i32, splats: &mut Vec<i32>) {
    //     for &cell in subspace.1.iter() {
    //         collect_sub_splats(octree, subspace.0, cell, start_lod, num_lods, splats);
    //     }
    // }

    // fn create_chunk(octree: &Octree<Node>, subspace: (i32, Vec<[i64; 3]>), start_lod: i32) {
    //     println!("create_chunk: {:?}, start_lod={}", subspace, start_lod);

    //     let (mut lod, mut active) = subspace.clone();
    //     let mut chunk_splats = Vec::new();

    //     active.retain(|cell| octree.get(&lod).is_some_and(|level| level.contains_key(cell)));

    //     while !active.is_empty() && lod > start_lod {
    //         let mut next_active = Vec::new();
    //         for &cell in active.iter() {
    //             if octree.get(&lod).is_some_and(|level| level.contains_key(&cell)) {
    //                 for z in 0..2 {
    //                     for y in 0..2 {
    //                         for x in 0..2 {
    //                             let cell = [2 * cell[0] + x, 2 * cell[1] + y, 2 * cell[2] + z];
    //                             if octree.get(&(lod - 1)).is_some_and(|level| level.contains_key(&cell)) {
    //                                 next_active.push(cell);
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }

    //         lod -= 1;
    //         active = next_active;
    //     }

    //     let total_count = active.iter().fold(0, |count, cell| {
    //         count + octree.get(&lod).and_then(|level| level.get(cell)).map_or(0, |node| node.splats.len() + node.subtree_count)
    //     });

    //     fn recursive_add(octree: &Octree<Node>, lod: i32, cell: [i64; 3], splats: &mut Vec<i32>) {
    //         if let Some(node) = octree.get(&lod).and_then(|level| level.get(&cell)) {
    //             splats.extend(node.splats.iter().copied());
    //             for z in 0..2 {
    //                 for y in 0..2 {
    //                     for x in 0..2 {
    //                         recursive_add(octree, lod - 1, [2 * cell[0] + x, 2 * cell[1] + y, 2 * cell[2] + z], splats);
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     if total_count == 0 {
    //         return;
    //     }
    //     if total_count <= MAX_CHUNK_SPLATS {
    //         for &cell in active.iter() {
    //             recursive_add(octree, lod, cell, &mut chunk_splats);
    //         }
    //         println!("Complete chunk: subspace={:?}, start_lod={}, {} splats", subspace, start_lod, chunk_splats.len());
    //         return;
    //     }

    //     while !active.is_empty() {
    //         let mut level_splats = Vec::new();
    //         let mut next_active = Vec::new();

    //         for &cell in active.iter() {
    //             if let Some(node) = octree.get(&lod).and_then(|level| level.get(&cell)) {
    //                 level_splats.extend(node.splats.iter().copied());
    //                 for z in 0..2 {
    //                     for y in 0..2 {
    //                         for x in 0..2 {
    //                             let next_cell = [2 * cell[0] + x, 2 * cell[1] + y, 2 * cell[2] + z];
    //                             if octree.get(&(lod - 1)).is_some_and(|level| level.contains_key(&next_cell)) {
    //                                 next_active.push(next_cell);
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }

    //         if (chunk_splats.len() + level_splats.len()) <= CHUNK_SPLATS {
    //             chunk_splats.extend(level_splats);
    //         } else {
    //             println!("Exceeded capacity: chunk_splats.len={}, level_splats.len={}", chunk_splats.len(), level_splats.len());
    //             break;
    //         }

    //         lod -= 1;
    //         active = next_active;
    //     }

    //     if !chunk_splats.is_empty() {
    //         println!("Chunk: lod={}, cells={:?}, start_lod={}, end_lod={}, {} splats", subspace.0, subspace.1, start_lod, lod, chunk_splats.len());
    //     }

    //     if !active.is_empty() {
    //         if subspace.1.len() > 1 {
    //             for &cell in subspace.1.iter() {
    //                 create_chunk(octree, (subspace.0, vec![cell]), lod);
    //             }
    //         } else {
    //             if lod == start_lod {
    //                 let cell = subspace.1[0];
    //                 for z in 0..2 {
    //                     for y in 0..2 {
    //                         for x in 0..2 {
    //                             create_chunk(octree, (subspace.0 - 1, vec![[2 * cell[0] + x, 2 * cell[1] + y, 2 * cell[2] + z]]), lod);
    //                         }
    //                     }
    //                 }
    //             } else {
    //                 create_chunk(octree, subspace, lod);
    //             }
    //         }
    //     }
    // }

    // let mut subspace = (max_lod, vec![]);
    // for z in cmin[2]..=cmax[2] {
    //     for y in cmin[1]..=cmax[1] {
    //         for x in cmin[0]..=cmax[0] {
    //             subspace.1.push([x, y, z]);
    //         }
    //     }
    // }

    // // create_chunk(&octree, subspace, max_lod);

    // let mut counts = Vec::new();
    // compute_subspace_counts(&octree, subspace.clone(), max_lod, &mut counts);
    // compute_cumulative_counts(&mut counts);
    // println!("counts: {:?}", counts);

    // let mut root_lods = 1;
    // while root_lods < counts.len() {
    //     if counts[root_lods] > CHUNK_SPLATS {
    //         break;
    //     }
    //     root_lods += 1;
    // }

    // let mut collected_splats = Vec::new();

    // let mut root_splats = Vec::new();
    // collect_subspace_splats(&octree, subspace.clone(), max_lod, root_lods as i32, &mut root_splats);
    // println!("root_splats: {:?}, #lods={}, #splats={}", subspace, root_lods, root_splats.len());
    // collected_splats.push(root_splats);

    // fn recurse_chunk(octree: &Octree<Node>, subspace: (i32, [i64; 3]), start_lod: i32, collected_splats: &mut Vec<Vec<i32>>) {
    //     let mut counts = Vec::new();
    //     compute_sub_counts(octree, subspace.0, subspace.1, start_lod, &mut counts);
    //     if counts.is_empty() {
    //         return;
    //     }

    //     compute_cumulative_counts(&mut counts);

    //     let mut num_lods = 0;
    //     while num_lods < counts.len() && counts[num_lods] <= CHUNK_SPLATS {
    //         num_lods += 1;
    //     }
    //     // println!("recurse_chunk {:?} start_lod={}: {:?}, num_lods={}", subspace, start_lod, counts, num_lods);

    //     if num_lods == counts.len() {
    //         let mut splats = Vec::new();
    //         collect_sub_splats(octree, subspace.0, subspace.1, start_lod, counts.len() as i32, &mut splats);
    //         println!("* Complete chunk: {:?}, start_lod={}, {} splats", subspace, start_lod, splats.len());
    //         collected_splats.push(splats);
    //         return;
    //     }

    //     let mut new_start_lod = start_lod;

    //     if num_lods >= CHUNK_LEVELS {
    //         let mut splats = Vec::new();
    //         collect_sub_splats(octree, subspace.0, subspace.1, start_lod, num_lods as i32, &mut splats);
    //         println!("* Limited chunk: {:?}, start_lod={}, #lods={}, {} splats", subspace, start_lod, num_lods, splats.len());
    //         collected_splats.push(splats);

    //         new_start_lod = start_lod - num_lods as i32;
    //         // recurse_chunk(octree, subspace, new_start_lod, collected_splats);
    //         // return;
    //     }

    //     for z in 0..2 {
    //         for y in 0..2 {
    //             for x in 0..2 {
    //                 let cell = [2 * subspace.1[0] + x, 2 * subspace.1[1] + y, 2 * subspace.1[2] + z];
    //                 recurse_chunk(octree, (subspace.0 - 1, cell), new_start_lod, collected_splats);
    //             }
    //         }
    //     }
    // }

    // for &cell in subspace.1.iter() {
    //     recurse_chunk(&octree, (max_lod, cell), max_lod - root_lods as i32, &mut collected_splats);
    // }

    // println!("Collected splats: {:?}", collected_splats.iter().map(|splats| splats.len()).collect::<Vec<_>>());
    // println!("Total collected splats: [{}] {}", collected_splats.len(), collected_splats.iter().map(|splats| splats.len()).sum::<usize>());
    // println!("Expected splats: {}", splats.len());
    
    let mut cut = splats[0].lod_max;
    let mut cuts = Vec::new();
    for (i, splat) in splats.iter().enumerate() {
        while cut >= splat.lod_max {
            println!(" {}: {}", cut, i);
            cuts.push((cut, i as i32));
            cut /= 2.0;
        }
    }
    cuts.push((0.0, splats.len() as i32));

    LodMeta {
        cuts,
        bound_center: bcenter,
        bound_radius: bradius,
    }
}
