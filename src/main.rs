// TODO: remove it at some point
#![allow(unused)]

use bevy::{
    color::palettes::css::GREEN,
    core_pipeline::{
        core_3d::graph::{Core3d, Node3d},
        fxaa::Fxaa,
        prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass},
    },
    diagnostic::{
        EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin,
        SystemInformationDiagnosticsPlugin,
    },
    ecs::system::{lifetimeless::SRes, SystemParamItem},
    pbr::{DefaultOpaqueRendererMethod, DirectionalLightShadowMap},
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        graph::CameraDriverLabel,
        render_asset::{
            PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssetUsages, RenderAssets,
        },
        render_graph::{self, RenderGraph, RenderGraphApp, RenderLabel, ViewNodeRunner},
        render_resource::{
            binding_types::{storage_buffer_read_only, texture_storage_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        settings::{Backends, InstanceFlags, RenderCreation, WgpuSettings},
        texture::ImageSampler,
        Render, RenderApp, RenderPlugin, RenderSet,
    },
    window::WindowPlugin,
};
use std::{borrow::Cow, fs};

use camera::*;
use import::*;
use render::*;
use voxel_tree::*;

mod camera;
mod import;
mod math;
mod render;
mod ui;
mod voxel_tree;

fn main() {
    color_backtrace::install();
    let mut app = App::new();
    app.insert_resource(ClearColor(Color::BLACK))
        .insert_resource(Msaa::Off)
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(DirectionalLightShadowMap { size: 4096 })
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        // uncomment for unthrottled FPS
                        present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(WgpuSettings {
                        instance_flags: InstanceFlags::default().with_env() | InstanceFlags::DEBUG,
                        // constrained_limits: WgpuLimits {
                        //     max_storage_buffer_binding_size: u32::MAX, // 4GiB
                        //     max_buffer_size: (u32::MAX as u64),
                        //     ..default()
                        // }.into(),
                        ..default()
                    }),
                    ..default()
                }),
            VoxelTracerPlugin,
            ui::GameUiPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, update_game_camera)
        .add_systems(Update, update_gizmos);

    let render_graph = bevy_mod_debugdump::render_graph_dot(&app, &default());
    fs::write("render_graph.graph", render_graph);

    let render_schedule =
        bevy_mod_debugdump::schedule_graph_dot(app.sub_app_mut(RenderApp), Render, &default());
    fs::write("render_schedule.graph", render_schedule);

    app.run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut voxel_trees: ResMut<Assets<VoxelTree>>,
    asset_server: Res<AssetServer>,
) {
    // TODO: is there better way?
    std::mem::forget(asset_server.load::<Shader>("shaders/common.wgsl"));
    std::mem::forget(asset_server.load::<Shader>("shaders/sdf.wgsl"));
    std::mem::forget(asset_server.load::<Shader>("shaders/voxel_common.wgsl"));
    std::mem::forget(asset_server.load::<Shader>("shaders/voxel_read.wgsl"));
    std::mem::forget(asset_server.load::<Shader>("shaders/voxel_write.wgsl"));
    std::mem::forget(asset_server.load::<Shader>("shaders/draw.wgsl"));

    let mut voxel_tree = VoxelTree::new(VOXEL_TREE_DEPTH as u8);
    //gen_test_scene(&mut voxel_tree, 4i32.pow(DEPTH as u32), Vec3::new(1., 0.5, 1.));

    // let model_path = "assets/Church_Of_St_Sophia.vox";
    let model_path = "assets/monu2.vox";

    let vox_model = dot_vox::load(model_path).expect("Failed to load");
    // place_vox(&mut voxel_tree, &vox_model, IVec3::new(2000, 50, 2000));
    place_vox(&mut voxel_tree, &vox_model, IVec3::new(200, 50, 200));

    std::mem::forget(voxel_trees.add(voxel_tree));

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
            transform: Transform::from_xyz(-5., -5., -5.).looking_at(Vec3::ZERO, Vec3::Y),
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
        app.add_plugins(RenderAssetPlugin::<GpuVoxelTree>::default());
        app.add_plugins(FrameTimeDiagnosticsPlugin::default());
        app.add_plugins(EntityCountDiagnosticsPlugin::default());
        app.add_plugins(SystemInformationDiagnosticsPlugin::default());
        app.add_plugins(CameraDiagnosticsPlugin::default());
        app.init_asset::<VoxelTree>();
        let render_app = app.sub_app_mut(RenderApp);

        render_app.add_systems(
            Render,
            (
                prepare_voxel_bind_groups.in_set(RenderSet::PrepareBindGroups),
                prepare_voxel_view_bind_groups
                    .in_set(RenderSet::PrepareBindGroups)
                    .after(prepare_voxel_bind_groups),
            ),
        );

        render_app
            .add_render_graph_node::<ViewNodeRunner<VoxelWorldPepassNode>>(
                Core3d,
                VoxelWorldPepassNodeLabel,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::DeferredPrepass,
                    VoxelWorldPepassNodeLabel,
                    Node3d::CopyDeferredLightingId,
                ),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(VoxelDrawNodeLabel, VoxelDrawNode::default());
        render_graph.add_node_edge(VoxelDrawNodeLabel, CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<VoxelGpuScene>();
        render_app.init_resource::<VoxelPipelines>();
    }
}

#[derive(Component)]
struct LoadingCell {
    pos: IVec3,
}

pub fn update_gizmos(mut gizmos: Gizmos) {
    gizmos.cuboid(Transform::from_scale(Vec3::splat(10.)), GREEN);
}
