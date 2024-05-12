use bevy::{prelude::*, render::{render_asset::RenderAsset, render_resource::ShaderType}};


pub const VOXEL_DIM: usize = 4;
pub const VOXEL_TREE_DEPTH: usize = 6;
pub const VOXEL_MASK_LEN: usize = 2;

pub const VOXEL_COUNT: usize = VOXEL_DIM * VOXEL_DIM * VOXEL_DIM;

pub fn pos_to_idx(ipos: IVec3) -> i32 {
    let dim = VOXEL_DIM as i32;
    ipos.x * dim * dim + ipos.y * dim + ipos.z
}

#[derive(Reflect, Clone, Copy, Default, ShaderType, Debug)]
pub struct Voxel {
    pub color: Vec3,
}

#[derive(Reflect, Clone, ShaderType, Debug)]
pub struct VoxelLeaf {
    pub mask: [u32; VOXEL_MASK_LEN],
    pub voxels: [Voxel; VOXEL_COUNT],
}

impl Default for VoxelLeaf {
    fn default() -> Self {
        Self {
            mask: [0; VOXEL_MASK_LEN],
            voxels: [Voxel { color: Vec3::ONE }; VOXEL_COUNT],
        }
    }
}

#[derive(Reflect, Clone, ShaderType, Debug)]
pub struct VoxelNode {
    pub mask: [u32; VOXEL_MASK_LEN],
    pub indices: [u32; VOXEL_COUNT],
}

impl VoxelNode {
//    pub fn debug_print(&self, self_idx: usize, depth: usize, tree: &VoxelTree) {
//        let mask: u64 = ((self.mask[1] as u64) << 32) | (self.mask[0] as u64);
//
//        error!("{:indent$}Node: {}", "", self_idx, indent = depth * 2);
//        error!(
//            "{:indent$}Indices: {:?}",
//            "",
//            self.indices,
//            indent = (depth + 1) * 2
//        );
//        if depth == 1 {
//            return;
//        }
//
//        for i in 0..64 {
//            if mask & (1u64 << i) == 0 {
//                continue;
//            }
//            error!("{:indent$}Idx: {}", "", i, indent = (depth + 1) * 2);
//
//            let nidx = self.indices[i as usize] as usize;
//
//            tree.nodes[nidx].debug_print(nidx, depth + 1, tree);
//        }
//    }
}

impl Default for VoxelNode {
    fn default() -> Self {
        Self {
            mask: [0; VOXEL_MASK_LEN],
            indices: [0; VOXEL_COUNT],
        }
    }
}

#[derive(Asset, Reflect, Clone, Default, Debug)]
pub struct VoxelTree {
    pub depth: u8,
    pub leafs: Vec<VoxelLeaf>,
    pub nodes: Vec<VoxelNode>,
}

impl VoxelTree {
    pub fn new(depth: u8) -> Self {
        let root = VoxelNode {
            mask: [0; VOXEL_MASK_LEN],
            indices: [0u32; VOXEL_COUNT],
        };

        Self {
            depth,
            leafs: Vec::new(),
            nodes: vec![root],
        }
    }

//    pub fn debug_print(&self) {
//        error!("Num of nodes: {}", self.nodes.len());
//
//        self.nodes[0].debug_print(0, 0, self);
//    }

    pub fn set_or_create_node(&mut self, parent_idx: u32, pos: IVec3) -> u32 {
        let nodes_len = self.nodes.len();
        let parent = &mut self.nodes[parent_idx as usize];
        let idx = pos_to_idx(pos);

        if get_mask(&parent.mask, idx) {
            parent.indices[idx as usize]
        } else {
            let res = nodes_len as u32;
            parent.indices[idx as usize] = res;
            set_mask(&mut parent.mask, idx);

            self.nodes.push(VoxelNode {
                mask: [0; VOXEL_MASK_LEN],
                indices: [0; VOXEL_COUNT],
            });

            res
        }
    }

    pub fn set_or_create_leaf(&mut self, parent_idx: u32, pos: IVec3) -> u32 {
        assert!(pos.x >= 0);
        assert!(pos.y >= 0);
        assert!(pos.z >= 0);

        let parent = &mut self.nodes[parent_idx as usize];
        let idx = pos_to_idx(pos);

        if get_mask(&parent.mask, idx) {
            parent.indices[idx as usize]
        } else {
            let res = self.leafs.len() as u32;
            parent.indices[idx as usize] = res;
            set_mask(&mut parent.mask, idx);

            self.leafs.push(VoxelLeaf {
                mask: [0; VOXEL_MASK_LEN],
                voxels: [Voxel { color: Vec3::ONE }; VOXEL_COUNT],
            });

            res
        }
    }

    pub fn set_voxel(&mut self, pos: IVec3, voxel: Voxel) {
        assert_ne!(self.depth, 0);

        let max = (VOXEL_DIM as i32).pow(self.depth as u32);

        if pos.x < 0 || pos.x >= max {
            return;
        }

        if pos.y < 0 || pos.y >= max {
            return;
        }

        if pos.z < 0 || pos.z >= max {
            return;
        }

        // TODO: assert pos vs tree size

        let mut parent_idx = 0;

        for depth in (1..self.depth).rev() {
            let local_pos = pos / (VOXEL_DIM as i32).pow(depth as u32) % (VOXEL_DIM as i32);

            if depth == 1 {
                let idx = self.set_or_create_leaf(parent_idx, local_pos);

                let leaf = &mut self.leafs[idx as usize];
                let local_pos = pos % (VOXEL_DIM as i32);
                let idx = pos_to_idx(local_pos);
                set_mask(&mut leaf.mask, idx);
                leaf.voxels[idx as usize] = voxel;
            } else {
                parent_idx = self.set_or_create_node(parent_idx, local_pos);
            }
        }
    }
}

pub fn set_mask(mask: &mut [u32; VOXEL_MASK_LEN], idx: i32) {
    let a = idx >> 5;
    let b = a * 32;
    mask[a as usize] |= 1u32 << (idx - b);
}

pub fn get_mask(mask: &[u32; VOXEL_MASK_LEN], idx: i32) -> bool {
    let a = idx >> 5;
    let b = a * 32;
    return (mask[a as usize] & 1u32 << (idx - b)) > 0
}

pub fn gen_voxel_leaf(offset: IVec3, f: &impl Fn(IVec3) -> bool) -> Option<VoxelLeaf> {
    let mut mask = [0u32; VOXEL_MASK_LEN];
    let mut add = false;
    for x in 0..VOXEL_DIM {
        for y in 0..VOXEL_DIM {
            for z in 0..VOXEL_DIM {
                let v = IVec3 {
                    x: x as i32,
                    y: y as i32,
                    z: z as i32,
                };

                if f(offset + v) {
                    set_mask(&mut mask, pos_to_idx(v));
                    add = true;
                }
            }
        }
    }

    if add {
        Some(VoxelLeaf {
            mask,
            voxels: [Voxel {
                color: Vec3::splat(1.),
            }; VOXEL_COUNT],
        })
    } else {
        None
    }
}

pub fn gen_voxel_node(
    tree: &mut VoxelTree,
    offset: IVec3,
    depth: u8,
    leaf_depth: u8,
    f: &impl Fn(IVec3) -> bool,
) -> Option<u32> {
    let idx_cur = tree.nodes.len();
    tree.nodes.push(Default::default());

    let mut mask = [0u32; VOXEL_MASK_LEN];
    let mut add = false;
    for x in 0..VOXEL_DIM {
        for y in 0..VOXEL_DIM {
            for z in 0..VOXEL_DIM {
                let v = IVec3 {
                    x: x as i32,
                    y: y as i32,
                    z: z as i32,
                };
                let index = pos_to_idx(v);
                let offset = (offset + v) * IVec3::splat(VOXEL_DIM as i32);

                if depth == leaf_depth - 1 {
                    if let Some(leaf) = gen_voxel_leaf(offset, f) {
                        tree.nodes[idx_cur].indices[index as usize] = tree.leafs.len() as u32;
                        tree.leafs.push(leaf);
                        
                        add = true;

                        set_mask(&mut mask, index);
                    }
                } else {
                    if let Some(idx) = gen_voxel_node(tree, offset, depth + 1, leaf_depth, f) {
                        tree.nodes[idx_cur].indices[index as usize] = idx;

                        add = true;

                        set_mask(&mut mask, index);
                    }
                }
            }
        }
    }

    if add {
        tree.nodes[idx_cur].mask = mask;

        return Some(idx_cur as u32);
    } else {
        assert_eq!(idx_cur, tree.nodes.len() - 1);
        tree.nodes.pop();

        return None;
    }
}

pub fn gen_voxel_tree(depth: u8, f: &impl Fn(IVec3) -> bool) -> VoxelTree {
    let mut res = VoxelTree::default();
    gen_voxel_node(&mut res, IVec3::ZERO, 0, depth - 1, f);

    res
}

pub fn gen_test_scene(voxel_tree: &mut VoxelTree, size: i32, color: Vec3) {
    let voxel = Voxel { color };

    for x in 0..size {
        for z in 0..size {
            voxel_tree.set_voxel(IVec3::new(x, 0, z), voxel);
        }
    }

    for x in 0..size {
        for y in 0..size {
            voxel_tree.set_voxel(IVec3::new(x, y, size - 1), voxel);
        }
    }

    let r = size / 4;
    for x in 0..(r * 2) {
        for y in 0..(r * 2) {
            for z in 0..(r * 2) {
                let v = Vec3::new(x as f32, y as f32, z as f32) - Vec3::splat(r as f32);
                let len = v.length();

                let offset = (IVec3::splat(size) - IVec3::splat(r * 2)) / 2;

                if len > (r as f32 - 5.) && len < (r as f32 + 5.) {
                    voxel_tree.set_voxel(IVec3::new(x, y, z) + offset, voxel);
                }
            }
        }
    }
}
