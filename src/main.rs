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
use import::*;
use std::borrow::Cow;
use voxel_tree::*;
use camera::*;

mod camera;
mod import;
mod math;
mod voxel_tree;

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

    commands.insert_resource(GameOfLifeImage {
        texture: image,
        voxel_tree: voxel_tree,
    });
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

#[derive(Resource, Default)]
pub struct GpuVoxelTree {
    pub nodes: StorageBuffer<Vec<VoxelNode>>,
    pub leafs: StorageBuffer<Vec<VoxelLeaf>>,
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
