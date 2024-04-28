use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    ecs::{
        system::{lifetimeless::SRes, SystemParamItem},
        world::error,
    },
    input::mouse::*,
    prelude::*,
    reflect::DynamicTypePath,
    render::{
        self,
        camera::CameraProjection,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        primitives::Frustum,
        render_asset::{
            PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssetUsages, RenderAssets,
        },
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            binding_types::{
                storage_buffer, storage_buffer_read_only, texture_storage_2d, uniform_buffer,
            },
            encase::ArrayLength,
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::ImageSampler,
        Extract, Render, RenderApp, RenderSet,
    },
    window::WindowPlugin,
};
use dot_vox::DotVoxData;
use std::borrow::Cow;

const SIZE: (u32, u32, u32) = (4, 4, 4);
const VOXEL_COUNT: usize = 4 * 4 * 4;
const VOXEL_DIM: usize = 4;
const SCREEN_SIZE: (u32, u32) = (1280, 720);
const WORKGROUP_SIZE: u32 = 8;

fn main() {
    color_backtrace::install();

    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    // uncomment for unthrottled FPS
                    // present_mode: bevy::window::PresentMode::AutoNoVsync,
                    ..default()
                }),
                ..default()
            }),
            GameOfLifeComputePlugin,
        ))
        .add_systems(Startup, setup)
        //.add_systems(Update, update_camera)
        .add_systems(Update, update_game_camera)
        .run();
}

fn _update_camera(
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion_evr: EventReader<MouseMotion>,
    mut wheel_evr: EventReader<MouseWheel>,
    mut q: Query<(&mut Transform, &mut OrthographicProjection), With<Camera>>,
) {
    let (mut transform, mut projection) = q.single_mut();

    for ev in wheel_evr.read() {
        projection.scale -= ev.y * 0.01f32;
    }

    if buttons.pressed(MouseButton::Right) {
        for ev in motion_evr.read() {
            transform.translation -=
                Vec3::new(ev.delta.x, -ev.delta.y, 0f32) * 0.625f32 * projection.scale;
        }
    }
}

fn update_game_camera(
    time: Res<Time>,
    input: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion_evr: EventReader<MouseMotion>,
    mut q: Query<(&mut Transform, &mut Projection), With<GameCamera>>,
) {
    let (mut transform, mut projection) = q.single_mut();

    if let Projection::Perspective(ref mut p) = *projection {
        p.aspect_ratio = 1280. / 720.;
    }

    let speed = if input.pressed(KeyCode::ShiftLeft) {
        500.
    } else {
        100.
    };

    let mut v = Vec3::ZERO;

    let forward = -Vec3::from(transform.local_z());
    let right = Vec3::from(transform.local_x());
    let up = Vec3::from(transform.local_y());

    if input.pressed(KeyCode::KeyW) {
        v += forward;
    }

    if input.pressed(KeyCode::KeyS) {
        v -= forward;
    }

    if input.pressed(KeyCode::KeyA) {
        v -= right;
    }

    if input.pressed(KeyCode::KeyD) {
        v += right;
    }

    if input.pressed(KeyCode::KeyQ) {
        v -= up;
    }

    if input.pressed(KeyCode::KeyE) {
        v += up;
    }

    let mut x = 0.;
    let mut y = 0.;

    if buttons.pressed(MouseButton::Right) {
        for ev in motion_evr.read() {
            x += ev.delta.x;
            y += ev.delta.y;
        }
    }

    let factor = 0.01;

    transform.translation += v.normalize_or_zero() * time.delta_seconds() * speed;

    let (yaw, pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);

    let yaw = yaw - x * factor;
    let pitch = pitch - y * factor;

    transform.rotation = Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch);
}

#[derive(Component, Clone, ExtractComponent)]
struct GameCamera;

#[derive(Bundle)]
struct GameCameraBundle {
    marker: GameCamera,
    projection: Projection,
    frustum: Frustum,
    transform: Transform,
    global_transform: GlobalTransform,
}

impl Default for GameCameraBundle {
    fn default() -> Self {
        Self {
            marker: GameCamera,
            projection: Default::default(),
            frustum: Default::default(),
            transform: Default::default(),
            global_transform: Default::default(),
        }
    }
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut voxel_trees: ResMut<Assets<VoxelTree>>,
) {
    let mut image = Image::new_fill(
        Extent3d {
            width: SCREEN_SIZE.0,
            height: SCREEN_SIZE.1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    image.sampler = ImageSampler::nearest();

    let image = images.add(image);

    commands.spawn(SpriteBundle {
        sprite: Sprite {
            custom_size: Some(Vec2::new(SCREEN_SIZE.0 as f32, SCREEN_SIZE.1 as f32)),
            ..default()
        },
        texture: image.clone(),
        ..default()
    });
    commands.spawn(Camera2dBundle::default());

    commands.spawn(GameCameraBundle {
        transform: Transform::from_xyz(-10.0, -5.0, -5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..Default::default()
    });

    const DEPTH: u8 = 4;

    let mut voxel_tree = VoxelTree::new(DEPTH);
    
    // let model_path = "assets/Church_Of_St_Sophia.vox";

    // let vox_model = dot_vox::load(model_path).expect("Failed to load");
    // place_vox(&mut voxel_tree, &vox_model);
    
    let size = 200;
    for x in 0..size {
        for z in 0..size {
            voxel_tree.set_voxel(IVec3::new(x, 0, z), Voxel { value: 1});
        }
    }

    for x in 0..size {
        for y in 0..size {
            voxel_tree.set_voxel(IVec3::new(x, y, size - 20), Voxel { value: 1});
        }
    }
    
    let r = 80;
    for x in 0..(r * 2) {
        for y in 0..(r * 2) {
            for z in 0..(r * 2) {
                let v = Vec3::new(x as f32, y as f32, z as f32) - Vec3::splat(r as f32);
                let len = v.length();
                
                let offset = (IVec3::splat(size) - IVec3::splat(r * 2)) / 2;
                
                if len > (r as f32 - 5.) && len < (r as f32 + 5.) {
                    voxel_tree.set_voxel(IVec3::new(x, y, z) + offset, Voxel { value: 1});
                }
            }
        }
    }

    // error!("materials: {:#?}", vox_model.materials);
    // error!("palette: {:#?}", vox_model.palette);
    // error!("models: {:#?}", vox_model.models);
    // error!("scenes: {:#?}", vox_model.scenes);
    //error!("layers: {:#?}", vox_model.layers);

    // let voxel_tree = gen_voxel_tree(DEPTH, &|v| {
    //     //v.x >= 63
    //     let v = Vec3::new(v.x as f32, v.y as f32, v.z as f32) / ((VOXEL_DIM.pow(DEPTH as u32) - 1) as f32);
    //     let v = v * 2f32 - 1f32;

    //     v.length() <= 1.
    // });

    // TODO: zero bug
    // voxel_tree.set_voxel(IVec3::new(0, 0, 0), Voxel { value: 1 });
    // voxel_tree.set_voxel(IVec3::new(1, 1, 1), Voxel { value: 1 });
    // voxel_tree.set_voxel(IVec3::new(2, 2, 2), Voxel { value: 1 });
    // voxel_tree.set_voxel(IVec3::new(3, 3, 3), Voxel { value: 1 });

    // TODO: cube tearing
    // let size = (VOXEL_DIM as u32).pow(DEPTH as u32) as i32;
    // for x in 0..size {
    //     for y in 0..size {
    //         for z in 0..size {
    //             let mut place = false;
    //             place = place || x == 3 || x == size - 4;
    //             place = place || y == 3 || y == size - 4;
    //             place = place || z == 3 || z == size - 4;
    //
    //             if place {
    //                 voxel_tree.set_voxel(IVec3::new(x, y, z), Voxel { value: 1});
    //             }
    //         }
    //     }
    // }

    // error!("voxel tree: {:#?}", voxel_tree);
    // voxel_tree.debug_print();

    let voxel_tree = voxel_trees.add(voxel_tree);

    commands.insert_resource(GameOfLifeImage {
        texture: image,
        view: Default::default(),
        voxel_tree: voxel_tree,
    });
}


#[derive(Clone, Copy)]
#[repr(C)]
pub struct IMat4 {
    pub x_axis: IVec4,
    pub y_axis: IVec4,
    pub z_axis: IVec4,
    pub w_axis: IVec4,
}

impl IMat4 {
    pub const ZERO: Self = Self::from_cols(IVec4::ZERO, IVec4::ZERO, IVec4::ZERO, IVec4::ZERO);
    pub const IDENTITY: Self = Self::from_cols(IVec4::X, IVec4::Y, IVec4::Z, IVec4::W);

    pub const fn from_cols(x_axis: IVec4, y_axis: IVec4, z_axis: IVec4, w_axis: IVec4) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
            w_axis,
        }
    }

    pub fn mul_vec4(&self, rhs: IVec4) -> IVec4 {
        let mut res = self.x_axis * rhs.xxxx();
        res = res + self.y_axis * rhs.yyyy();
        res = res + self.z_axis * rhs.zzzz();
        res = res + self.w_axis * rhs.wwww();
        res
    }
    
    pub fn mul_mat4(&self, rhs: &Self) -> Self {
        Self::from_cols(
            self.mul_vec4(rhs.x_axis),
            self.mul_vec4(rhs.y_axis),
            self.mul_vec4(rhs.z_axis),
            self.mul_vec4(rhs.w_axis),
        )
    }

    pub fn col_mut(&mut self, index: usize) -> &mut IVec4 {
        match index {
            0 => &mut self.x_axis,
            1 => &mut self.y_axis,
            2 => &mut self.z_axis,
            3 => &mut self.w_axis,
            _ => panic!("index out of bounds"),
        }
    }

    pub fn from_translation(translation: IVec3) -> Self {
        Self::from_cols(
            IVec4::X,
            IVec4::Y,
            IVec4::Z,
            IVec4::new(translation.x, translation.y, translation.z, 1),
        )
    }
}


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


fn place_vox_model(tree: &mut VoxelTree, vox: &DotVoxData, model_id: u32, tr: &IMat4) {
    let model = &vox.models[model_id as usize];
    for &dot_vox::Voxel { x, y, z, i } in &model.voxels {
        let pos = tr.mul_vec4(IVec4::new(x as i32, y as i32, z as i32, 1));
        
        assert_eq!(pos.w, 1);
        
        let color = vox.palette[i as usize];
        // if color.a != 255 {
        //     continue;
        // }

        tree.set_voxel(pos.xyz(), Voxel { value: 1 });
    }
}

fn place_vox_scene_node(tree: &mut VoxelTree, vox: &DotVoxData, node_idx: u32, tr: &IMat4) {
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
            
            let trn = IMat4::from_translation(t).mul_mat4(&rot_to_mat(r));

            place_vox_scene_node(tree, vox, *child, &tr.mul_mat4(&trn))
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

fn place_vox(tree: &mut VoxelTree, vox: &DotVoxData) {
    let t = IMat4::from_translation(IVec3::new(2000, 50, 2000));
    let r = IMat4::from_cols(
        IVec4::new(1, 0, 0, 0),
        IVec4::new(0, 0, 1, 0),
        IVec4::new(0, 1, 0, 0),
        IVec4::new(0, 0, 0, 1),
    );
    
    let tr = t.mul_mat4(&r);

    place_vox_scene_node(tree, vox, 0, &tr);
}

fn pos_to_idx(ipos: IVec3) -> i32 {
    ipos.x * 4 * 4 + ipos.y * 4 + ipos.z
}

fn gen_voxel_leaf(offset: IVec3, f: &impl Fn(IVec3) -> bool) -> Option<VoxelLeaf> {
    let mut mask: u64 = 0;
    for x in 0..VOXEL_DIM {
        for y in 0..VOXEL_DIM {
            for z in 0..VOXEL_DIM {
                let v = IVec3 {
                    x: x as i32,
                    y: y as i32,
                    z: z as i32,
                };

                if f(offset + v) {
                    mask |= 1u64 << pos_to_idx(v);
                }
            }
        }
    }

    if mask != 0 {
        Some(VoxelLeaf {
            mask: [mask as u32, (mask >> 32) as u32],
            voxels: [Voxel { value: 1 }; VOXEL_COUNT],
        })
    } else {
        None
    }
}

fn gen_voxel_node(
    tree: &mut VoxelTree,
    offset: IVec3,
    depth: u8,
    leaf_depth: u8,
    f: &impl Fn(IVec3) -> bool,
) -> Option<u32> {
    let idx_cur = tree.nodes.len();
    tree.nodes.push(Default::default());

    let mut mask: u64 = 0;
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

                        mask |= 1u64 << index
                    }
                } else {
                    if let Some(idx) = gen_voxel_node(tree, offset, depth + 1, leaf_depth, f) {
                        tree.nodes[idx_cur].indices[index as usize] = idx;

                        mask |= 1u64 << index
                    }
                }
            }
        }
    }

    if mask != 0 {
        tree.nodes[idx_cur].mask = [mask as u32, (mask >> 32) as u32];

        return Some(idx_cur as u32);
    } else {
        assert_eq!(idx_cur, tree.nodes.len() - 1);
        tree.nodes.pop();

        return None;
    }
}

fn gen_voxel_tree(depth: u8, f: &impl Fn(IVec3) -> bool) -> VoxelTree {
    let mut res = VoxelTree::default();
    gen_voxel_node(&mut res, IVec3::ZERO, 0, depth - 1, f);

    res
}

struct GameOfLifeComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct GameOfLifeLabel;

impl Plugin for GameOfLifeComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugins(ExtractResourcePlugin::<GameOfLifeImage>::default());
        app.add_plugins(RenderAssetPlugin::<VoxelTree>::default());
        app.add_plugins(ExtractComponentPlugin::<GameCamera>::default());
        // app.add_plugins(FrameTimeDiagnosticsPlugin::default());
        // app.add_plugins(LogDiagnosticsPlugin::default());
        app.init_asset::<VoxelTree>();
        let render_app = app.sub_app_mut(RenderApp);

        render_app.add_systems(Render, prepare_view.in_set(RenderSet::PrepareResources));

        render_app.add_systems(
            Render,
            prepare_bind_group.in_set(RenderSet::PrepareBindGroups),
        );

        render_app.add_systems(ExtractSchedule, extract_view);

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node(GameOfLifeLabel, GameOfLifeNode::default());
        render_graph.add_node_edge(GameOfLifeLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<GameOfLifePipeline>();
        render_app.init_resource::<ViewUniforms>();
    }
}

#[derive(Clone, ShaderType, Default)]
struct ViewUniform {
    view_proj: Mat4,
    inverse_view_proj: Mat4,
    view: Mat4,
    inverse_view: Mat4,
    projection: Mat4,
    inverse_projection: Mat4,
    // viewport(x_origin, y_origin, width, height)
    //    viewport: Vec4,
    //    frustum: [Vec4; 6]
}

#[derive(Resource, Default)]
struct ViewUniforms {
    uniforms: UniformBuffer<ViewUniform>,
}

#[derive(Reflect, Clone, Copy, Default, ShaderType, Debug)]
struct Voxel {
    value: u32,
}

#[derive(Reflect, Clone, ShaderType, Debug)]
struct VoxelLeaf {
    mask: [u32; 2],
    voxels: [Voxel; VOXEL_COUNT],
}

impl Default for VoxelLeaf {
    fn default() -> Self {
        Self {
            mask: [0; 2],
            voxels: [Voxel { value: 0 }; VOXEL_COUNT],
        }
    }
}

#[derive(Reflect, Clone, ShaderType, Debug)]
struct VoxelNode {
    mask: [u32; 2],
    indices: [u32; VOXEL_COUNT],
}

impl VoxelNode {
    fn debug_print(&self, self_idx: usize, depth: usize, tree: &VoxelTree) {
        let mask: u64 = ((self.mask[1] as u64) << 32) | (self.mask[0] as u64);

        error!("{:indent$}Node: {}", "", self_idx, indent = depth * 2);
        error!(
            "{:indent$}Indices: {:?}",
            "",
            self.indices,
            indent = (depth + 1) * 2
        );
        if depth == 1 {
            return;
        }

        for i in 0..64 {
            if mask & (1u64 << i) == 0 {
                continue;
            }
            error!("{:indent$}Idx: {}", "", i, indent = (depth + 1) * 2);

            let nidx = self.indices[i as usize] as usize;

            tree.nodes[nidx].debug_print(nidx, depth + 1, tree);
        }
    }
}

impl Default for VoxelNode {
    fn default() -> Self {
        Self {
            mask: [0; 2],
            indices: [0; VOXEL_COUNT],
        }
    }
}

#[derive(Asset, Reflect, Clone, Default, Debug)]
struct VoxelTree {
    depth: u8,
    leafs: Vec<VoxelLeaf>,
    nodes: Vec<VoxelNode>,
}

fn set_mask(mask: &mut [u32; 2], idx: u32) {
    let mut mask64: u64 = ((mask[1] as u64) << 32) | (mask[0] as u64);
    mask64 |= 1u64 << idx;

    mask[0] = mask64 as u32;
    mask[1] = (mask64 >> 32) as u32;
}

impl VoxelTree {
    fn new(depth: u8) -> Self {
        let root = VoxelNode {
            mask: [0u32, 0u32],
            indices: [0u32; VOXEL_COUNT],
        };

        Self {
            depth,
            leafs: Vec::new(),
            nodes: vec![root],
        }
    }

    fn debug_print(&self) {
        error!("Num of nodes: {}", self.nodes.len());

        self.nodes[0].debug_print(0, 0, self);
    }

    fn set_or_create_node(&mut self, parent_idx: u32, pos: IVec3) -> u32 {
        let nodes_len = self.nodes.len();
        let parent = &mut self.nodes[parent_idx as usize];
        let idx = pos_to_idx(pos);
        let mask: u64 = ((parent.mask[1] as u64) << 32) | (parent.mask[0] as u64);

        if mask & (1u64 << idx) != 0 {
            parent.indices[idx as usize]
        } else {
            let res = nodes_len as u32;
            parent.indices[idx as usize] = res;
            set_mask(&mut parent.mask, idx as u32);

            self.nodes.push(VoxelNode {
                mask: [0, 0],
                indices: [0; VOXEL_COUNT],
            });

            res
        }
    }

    fn set_or_create_leaf(&mut self, parent_idx: u32, pos: IVec3) -> u32 {
        assert!(pos.x >= 0);
        assert!(pos.y >= 0);
        assert!(pos.z >= 0);

        let parent = &mut self.nodes[parent_idx as usize];
        let idx = pos_to_idx(pos);
        let mask: u64 = ((parent.mask[1] as u64) << 32) | (parent.mask[0] as u64);

        if mask & (1u64 << idx) != 0 {
            parent.indices[idx as usize]
        } else {
            let res = self.leafs.len() as u32;
            parent.indices[idx as usize] = res;
            set_mask(&mut parent.mask, idx as u32);

            self.leafs.push(VoxelLeaf {
                mask: [0, 0],
                voxels: [Voxel { value: 0 }; VOXEL_COUNT],
            });

            res
        }
    }

    fn set_voxel(&mut self, pos: IVec3, voxel: Voxel) {
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
                set_mask(&mut leaf.mask, idx as u32);
                leaf.voxels[idx as usize] = voxel;
            } else {
                parent_idx = self.set_or_create_node(parent_idx, local_pos);
            }
        }
    }
}

#[derive(Resource, Default)]
struct GpuVoxelTree {
    nodes: StorageBuffer<Vec<VoxelNode>>,
    leafs: StorageBuffer<Vec<VoxelLeaf>>,
}

impl RenderAsset for VoxelTree {
    type PreparedAsset = GpuVoxelTree;
    type Param = (SRes<RenderDevice>, SRes<RenderQueue>);

    fn asset_usage(&self) -> RenderAssetUsages {
        RenderAssetUsages::RENDER_WORLD
    }

    /// Converts the extracted image into a [`GpuImage`].
    fn prepare_asset(
        self,
        (device, queue): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self>> {
        let mut nodes = StorageBuffer::default();
        nodes.set(self.nodes.clone());
        nodes.write_buffer(device, queue);

        let mut leafs = StorageBuffer::default();
        leafs.set(self.leafs.clone());
        leafs.write_buffer(device, queue);

        Ok(GpuVoxelTree { nodes, leafs })
    }
}

#[derive(Resource, Clone, ExtractResource)]
struct GameOfLifeImage {
    texture: Handle<Image>,
    view: ViewUniform,
    voxel_tree: Handle<VoxelTree>,
}

impl GameOfLifeImage {
    fn as_bind_group(
        &self,
        layout: &BindGroupLayout,
        render_device: &RenderDevice,
        images: &RenderAssets<Image>,
        voxel_trees: &RenderAssets<VoxelTree>,
        view_uniforms: &ViewUniforms,
    ) -> Result<BindGroup, AsBindGroupError> {
        let texture = images
            .get(self.texture.clone())
            .ok_or(AsBindGroupError::RetryNextUpdate)?;

        let view = view_uniforms
            .uniforms
            .binding()
            .ok_or(AsBindGroupError::RetryNextUpdate)?;

        let voxel_tree = voxel_trees
            .get(self.voxel_tree.clone())
            .ok_or(AsBindGroupError::RetryNextUpdate)?;

        let voxel_nodes = voxel_tree
            .nodes
            .binding()
            .ok_or(AsBindGroupError::RetryNextUpdate)?;

        let voxel_leafs = voxel_tree
            .leafs
            .binding()
            .ok_or(AsBindGroupError::RetryNextUpdate)?;

        let res = render_device.create_bind_group(
            "voxel_tree_group",
            layout,
            &BindGroupEntries::sequential((&texture.texture_view, view, voxel_nodes, voxel_leafs)),
        );

        Ok(res)
    }

    fn bind_group_layout(render_device: &RenderDevice) -> BindGroupLayout {
        render_device.create_bind_group_layout(
            "voxel_tree_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::ReadWrite),
                    uniform_buffer::<ViewUniform>(false),
                    storage_buffer_read_only::<Vec<VoxelNode>>(false),
                    storage_buffer_read_only::<Vec<VoxelLeaf>>(false),
                ),
            ),
        )
    }
}

#[derive(Resource)]
struct GameOfLifeImageBindGroup(BindGroup);

fn extract_view(
    mut commands: Commands,
    query: Extract<Query<(Entity, &Projection, &Transform), With<GameCamera>>>,
) {
    let (entity, proj, trs) = query.single();
    let mut commands = commands.get_or_spawn(entity);

    commands.insert(proj.clone());
    commands.insert(trs.clone());
}

fn prepare_view(
    mut view_uniforms: ResMut<ViewUniforms>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    q: Query<(&Projection, &Transform), With<GameCamera>>,
) {
    let (proj, trs) = q.single();

    let inverse_view = trs.compute_matrix();
    let view = inverse_view.inverse();

    let projection = proj.get_projection_matrix();
    let view_proj = projection * view;

    let view = ViewUniform {
        view_proj,
        inverse_view_proj: view_proj.inverse(),
        view,
        inverse_view,
        projection: projection,
        inverse_projection: projection.inverse(),
    };

    view_uniforms.uniforms.set(view);

    view_uniforms
        .uniforms
        .write_buffer(&render_device, &render_queue);
}

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<GameOfLifePipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    gpu_voxel_tress: Res<RenderAssets<VoxelTree>>,
    game_of_life_image: Res<GameOfLifeImage>,
    view_uniforms: Res<ViewUniforms>,
    render_device: Res<RenderDevice>,
) {
    // let view = gpu_images.get(&game_of_life_image.texture).unwrap();
    // let u = view_uniforms.uniforms.binding().unwrap();
    // let voxel_tree = gpu_voxel_tress.get(&game_of_life_image.cube).unwrap();

    // let bind_group =  render_device.create_bind_group(
    //     None,
    //     &pipeline.texture_bind_group_layout,
    //     &BindGroupEntries::sequential((&view.texture_view, u, &cube.texture_view)),
    // );
    let bind_group = game_of_life_image
        .as_bind_group(
            &pipeline.texture_bind_group_layout,
            &render_device,
            &gpu_images,
            &gpu_voxel_tress,
            &view_uniforms,
        )
        .unwrap();

    commands.insert_resource(GameOfLifeImageBindGroup(bind_group));
}

#[derive(Resource)]
struct GameOfLifePipeline {
    texture_bind_group_layout: BindGroupLayout,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
}

impl FromWorld for GameOfLifePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let texture_bind_group_layout = GameOfLifeImage::bind_group_layout(render_device);
        let shader = world.resource::<AssetServer>().load("shaders/test.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![texture_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("init"),
        });
        let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![texture_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("update"),
        });

        GameOfLifePipeline {
            texture_bind_group_layout,
            init_pipeline,
            update_pipeline,
        }
    }
}

#[derive(Debug)]
enum GameOfLifeState {
    Loading,
    Init,
    Update,
}

struct GameOfLifeNode {
    state: GameOfLifeState,
}

impl Default for GameOfLifeNode {
    fn default() -> Self {
        Self {
            state: GameOfLifeState::Loading,
        }
    }
}

impl render_graph::Node for GameOfLifeNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<GameOfLifePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            GameOfLifeState::Loading => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline)
                {
                    self.state = GameOfLifeState::Init;
                }
            }
            GameOfLifeState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
                {
                    self.state = GameOfLifeState::Update;
                }
            }
            GameOfLifeState::Update => {
                if pipeline_cache
                    .get_compute_pipeline(pipeline.update_pipeline)
                    .is_none()
                {
                    self.state = GameOfLifeState::Loading;
                }
            }
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let texture_bind_group = &world.resource::<GameOfLifeImageBindGroup>().0;
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<GameOfLifePipeline>();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, texture_bind_group, &[]);

        // select the pipeline based on the current state
        match self.state {
            GameOfLifeState::Loading => {}
            GameOfLifeState::Init => {
                let init_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.init_pipeline)
                    .unwrap();
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(
                    (SIZE.0 / WORKGROUP_SIZE).max(1),
                    (SIZE.1 / WORKGROUP_SIZE).max(1),
                    (SIZE.2 / WORKGROUP_SIZE).max(1),
                );
            }
            GameOfLifeState::Update => {
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.update_pipeline)
                    .unwrap();
                pass.set_pipeline(update_pipeline);
                pass.dispatch_workgroups(
                    SCREEN_SIZE.0 / WORKGROUP_SIZE,
                    SCREEN_SIZE.1 / WORKGROUP_SIZE,
                    1,
                );
            }
        }

        Ok(())
    }
}
