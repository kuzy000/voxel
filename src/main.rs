use bevy::{
    core_pipeline::{
        fxaa::Fxaa,
        prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass},
    },
    diagnostic::{EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin},
    ecs::system::{lifetimeless::SRes, SystemParamItem},
    pbr::{DefaultOpaqueRendererMethod, DirectionalLightShadowMap},
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_asset::{
            PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssetUsages, RenderAssets,
        },
        render_graph::{self, RenderLabel},
        render_resource::{
            binding_types::{storage_buffer_read_only, texture_storage_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::ImageSampler, RenderApp,
    },
    window::WindowPlugin,
};
use camera::*;
use import::*;
use material::*;
use render::*;
use std::borrow::Cow;
use voxel_tree::*;

mod camera;
mod import;
mod material;
mod math;
mod render;
mod ui;
mod voxel_tree;

fn main() {
    color_backtrace::install();

    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(Msaa::Off)
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(DirectionalLightShadowMap { size: 4096 })
        .init_resource::<GpuVoxelTree>()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    // uncomment for unthrottled FPS
                    // present_mode: bevy::window::PresentMode::AutoNoVsync,
                    ..default()
                }),
                ..default()
            }),
            MaterialPlugin::<VoxelTreeMaterial> {
                prepass_enabled: true,
                ..default()
            },
            VoxelTracerPlugin,
            ui::GameUiPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, update_game_camera)
        .run();
}

fn _setup_voxel_tracer(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut voxel_trees: ResMut<Assets<VoxelTree>>,
    asset_server: Res<AssetServer>,
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

    commands.spawn((
        GameCamera,
        DepthPrepass,
        NormalPrepass,
        MotionVectorPrepass,
        DeferredPrepass,
        Fxaa::default(),
        Camera3dBundle {
            transform: Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        EnvironmentMapLight {
            diffuse_map: asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2"),
            specular_map: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            intensity: 250.0,
        },
    ));

    // commands.spawn(GameCameraBundle {
    //     transform: Transform::from_xyz(-10.0, -5.0, -5.0).looking_at(Vec3::ZERO, Vec3::Y),
    //     ..Default::default()
    // });

    const DEPTH: u8 = 6;

    let mut voxel_tree = VoxelTree::new(DEPTH);

    // let model_path = "assets/Church_Of_St_Sophia.vox";
    let model_path = "assets/monu2.vox";

    let vox_model = dot_vox::load(model_path).expect("Failed to load");
    place_vox(&mut voxel_tree, &vox_model);

    let voxel_tree = voxel_trees.add(voxel_tree);

    commands.insert_resource(VoxelTracer {
        texture: image,
        voxel_tree: voxel_tree,
    });
}

fn setup(
    mut commands: Commands,
    //mut images: ResMut<Assets<Image>>,
    //mut voxel_trees: ResMut<Assets<VoxelTree>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut vt_materials: ResMut<Assets<VoxelTreeMaterial>>,
    mut gpu_voxel_tree: ResMut<GpuVoxelTree>,
    mut voxel_trees: ResMut<Assets<VoxelTree>>,
    asset_server: Res<AssetServer>,
    render_device: ResMut<RenderDevice>,
    render_queue: ResMut<RenderQueue>,
) {
    const DEPTH: u8 = 6;

    let mut voxel_tree = VoxelTree::new(DEPTH);

    let model_path = "assets/Church_Of_St_Sophia.vox";
    // let model_path = "assets/monu2.vox";

    let vox_model = dot_vox::load(model_path).expect("Failed to load");
    place_vox(&mut voxel_tree, &vox_model);

    {
        gpu_voxel_tree.nodes.set(voxel_tree.nodes.clone());
        gpu_voxel_tree
            .nodes
            .write_buffer(&render_device, &render_queue);

        gpu_voxel_tree.leafs.set(voxel_tree.leafs.clone());
        gpu_voxel_tree
            .leafs
            .write_buffer(&render_device, &render_queue);
    }

    voxel_trees.add(voxel_tree);

    let size = 4f32.powf(6.0f32);

    commands.spawn(MaterialMeshBundle::<VoxelTreeMaterial> {
        mesh: meshes.add(Cuboid::new(size, size, size)),
        material: vt_materials.add(VoxelTreeMaterial {
            base: StandardMaterial {
                cull_mode: Some(Face::Front),
                opaque_render_method: bevy::pbr::OpaqueRendererMethod::Deferred,
                ..default()
            },
            extension: VoxelTreeMaterialExtension {
                voxel_nodes: gpu_voxel_tree.nodes.buffer().unwrap().clone(),
                voxel_leafs: gpu_voxel_tree.leafs.buffer().unwrap().clone(),
            },
        }),
        transform: Transform::from_translation(Vec3::splat(size * 0.5)),
        ..default()
    });

    // gltf
    commands.spawn(SceneBundle {
        scene: asset_server.load("models/FlightHelmet/FlightHelmet.gltf#Scene0"),
        ..default()
    });

    // light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    // camera
    commands.spawn((
        GameCamera,
        DepthPrepass,
        NormalPrepass,
        MotionVectorPrepass,
        DeferredPrepass,
        Fxaa::default(),
        Camera3dBundle {
            transform: Transform::from_xyz(-10., -10., -10.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        EnvironmentMapLight {
            diffuse_map: asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2"),
            specular_map: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            intensity: 250.0,
        },
    ));
}

struct VoxelTracerPlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct VoxelTracerLabel;

impl Plugin for VoxelTracerPlugin {
    fn build(&self, app: &mut App) {
        // app.add_plugins(ExtractResourcePlugin::<VoxelTracer>::default());
        app.add_plugins(RenderAssetPlugin::<VoxelTree>::default());
        app.add_plugins(FrameTimeDiagnosticsPlugin::default());
        app.add_plugins(EntityCountDiagnosticsPlugin::default());
        app.add_plugins(SystemInformationDiagnosticsPlugin::default());
        app.init_asset::<VoxelTree>();
        let render_app = app.sub_app_mut(RenderApp);

        // render_app.add_systems(
        //     Render,
        //     //prepare_voxel_tracer_bind_groups.in_set(RenderSet::PrepareBindGroups),
        //     prepare_voxel_tree_bind_groups.in_set(RenderSet::PrepareBindGroups),
        // );

        // let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        // render_graph.add_node(VoxelTracerLabel, VoxelTracerRenderNode::default());
        // render_graph.add_node_edge(VoxelTracerLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        // let render_app = app.sub_app_mut(RenderApp);
        // render_app.init_resource::<VoxelTracerPipelines>();
    }
}
