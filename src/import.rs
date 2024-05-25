use bevy::prelude::*;
use dot_vox::DotVoxData;

use crate::{math::IMat4, Voxel, VoxelTree};


pub fn rot_to_mat(rot: u8) -> IMat4 {
    let mut res = IMat4::ZERO;

    let index_nz1 = rot & 0b11;
    let index_nz2 = (rot >> 2) & 0b11;
    let index_nz3 = 3 - index_nz1 - index_nz2;

    let row_1_sign: i32 = if rot & (1 << 4) == 0 { 1 } else { -1 };
    let row_2_sign: i32 = if rot & (1 << 5) == 0 { 1 } else { -1 };
    let row_3_sign: i32 = if rot & (1 << 6) == 0 { 1 } else { -1 };

    res.col_mut(index_nz1 as usize)[0] = row_1_sign;
    res.col_mut(index_nz2 as usize)[1] = row_2_sign;
    res.col_mut(index_nz3 as usize)[2] = row_3_sign;
    res.col_mut(3)[3] = 1;

    res
}

pub fn place_vox_model(tree: &mut VoxelTree, vox: &DotVoxData, model_id: u32, tr: &IMat4) {
    let model = &vox.models[model_id as usize];
    
    let translate = -IVec3::new(model.size.x as i32, model.size.y as i32, model.size.z as i32) / 2;
   //  let translate_inv = -translate;
    
    let rotated = IVec4::new(1, 1, 1, 0);
    let rotated = tr.mul_vec4(rotated);
    let rotated = IVec4::splat(0).max(rotated);

    // info!("{:#?}", model.size);

    // let trs = IMat4::from_translation(translate_inv).mul_mat4(&tr.mul_mat4(&IMat4::from_translation(translate)));
    let trs = *tr * IMat4::from_translation(translate);

    for &dot_vox::Voxel { x, y, z, i } in &model.voxels {
        let mut pos = trs * IVec4::new(x as i32, y as i32, z as i32, 1);
        pos += rotated;
        // error!("{} {} {} -> {} {} {}", x, y, z, pos.x, pos.y, pos.z);
        
        assert_eq!(pos.w, 1);
        
        let color = vox.palette[i as usize];
        // if color.a != 255 {
        //     continue;
        // }
        let color = Vec3::new(color.r as f32 / 255., color.g as f32 / 255., color.b as f32 / 255.);

        let color = (color * 255.);
        let color = IVec3::new(color.x as i32, color.y as i32, color.z as i32);
        let color = (color.z << 16) | (color.y << 8) | (color.x << 0);
        let voxel = Voxel { color: color as u32 };

        tree.set_voxel(pos.xyz(), voxel);
    }
}

pub fn place_vox_scene_node(tree: &mut VoxelTree, vox: &DotVoxData, node_idx: u32, tr: &IMat4) {
    let node = &vox.scenes[node_idx as usize];
    match &node {
        dot_vox::SceneNode::Transform { attributes: _, frames, child, layer_id: _ } => {
            let attr = &frames[0].attributes;

            let t = attr.get("_t").map_or(IVec3::ZERO, |s| {
                let v: Vec<&str> = s.split(' ').collect();
                let x = v[0].parse().unwrap();
                let y = v[1].parse().unwrap();
                let z = v[2].parse().unwrap();
                IVec3::new(x, y, z)
            });

            let r =  attr.get("_r").map_or(0b100, |s| s.parse().unwrap());
            
            let trn = IMat4::from_translation(t) * rot_to_mat(r);

            place_vox_scene_node(tree, vox, *child, &(*tr * trn))
        }
        dot_vox::SceneNode::Group { attributes: _, children } => {
            for child in children {
                place_vox_scene_node(tree, vox, *child, tr)
            }
        } 
        dot_vox::SceneNode::Shape { attributes: _, models } => {
            for dot_vox::ShapeModel { model_id, attributes: _ }  in models {
                place_vox_model(tree, vox, *model_id, tr);
            }
        }
    }
}

pub fn place_vox(tree: &mut VoxelTree, vox: &DotVoxData, offset: IVec3) {
    let t = IMat4::from_translation(offset);
    let r = IMat4::from_cols(
        IVec4::new(1, 0, 0, 0),
        IVec4::new(0, 0, 1, 0),
        IVec4::new(0, 1, 0, 0),
        IVec4::new(0, 0, 0, 1),
    );
    
    let tr = t * r;

    place_vox_scene_node(tree, vox, 0, &tr);
}