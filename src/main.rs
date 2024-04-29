use bevy::{
    ecs::system::{lifetimeless::SRes, SystemParamItem},
    prelude::*,
    render::{
        camera::CameraProjection,
        extract_component::ExtractComponentPlugin,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::{
            PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssetUsages, RenderAssets,
        },
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            binding_types::{storage_buffer_read_only, texture_storage_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::ImageSampler,
        Extract, Render, RenderApp, RenderSet,
    },
    window::WindowPlugin,
};
use camera::*;
use import::*;
use render::*;
use std::borrow::Cow;
use voxel_tree::*;

mod camera;
mod import;
mod math;
mod render;
mod voxel_tree;

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
            VoxelTracerPlugin,
        ))
        .add_systems(Startup, setup)
        //.add_systems(Update, update_camera)
        .add_systems(Update, update_game_camera)
        .run();
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

    const DEPTH: u8 = 6;

    let mut voxel_tree = VoxelTree::new(DEPTH);

    let model_path = "assets/Church_Of_St_Sophia.vox";

    let vox_model = dot_vox::load(model_path).expect("Failed to load");
    place_vox(&mut voxel_tree, &vox_model);

    let voxel_tree = voxel_trees.add(voxel_tree);

    commands.insert_resource(VoxelTracer {
        texture: image,
        voxel_tree: voxel_tree,
    });
}

struct VoxelTracerPlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct VoxelTracerLabel;

impl Plugin for VoxelTracerPlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugins(ExtractResourcePlugin::<VoxelTracer>::default());
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
        render_graph.add_node(VoxelTracerLabel, VoxelTracerRenderNode::default());
        render_graph.add_node_edge(VoxelTracerLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<VoxelTracerPipeline>();
        render_app.init_resource::<ViewUniforms>();
    }
}
