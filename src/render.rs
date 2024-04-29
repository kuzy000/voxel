use bevy::{prelude::*, render::render_resource::{StorageBuffer, UniformBuffer}};

use crate::*;

pub const SCREEN_SIZE: (u32, u32) = (1280, 720);
pub const WORKGROUP_SIZE: u32 = 8;

#[derive(Clone, ShaderType, Default)]
pub struct ViewUniform {
    pub view_proj: Mat4,
    pub inverse_view_proj: Mat4,
    pub view: Mat4,
    pub inverse_view: Mat4,
    pub projection: Mat4,
    pub inverse_projection: Mat4,
    // viewport(x_origin, y_origin, width, height)
    //    viewport: Vec4,
    //    frustum: [Vec4; 6]
}

#[derive(Resource, Default)]
pub struct ViewUniforms {
    pub uniforms: UniformBuffer<ViewUniform>,
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

// TODO: rename me
#[derive(Resource, Clone, ExtractResource)]
pub struct VoxelTracer {
    pub texture: Handle<Image>,
    pub voxel_tree: Handle<VoxelTree>,
}

impl VoxelTracer {
    pub fn as_bind_group(
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

    pub fn bind_group_layout(render_device: &RenderDevice) -> BindGroupLayout {
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
pub struct VoxelTracerBindGroup(BindGroup);

pub fn extract_view(
    mut commands: Commands,
    query: Extract<Query<(Entity, &Projection, &Transform), With<GameCamera>>>,
) {
    let (entity, proj, trs) = query.single();
    let mut commands = commands.get_or_spawn(entity);

    commands.insert(proj.clone());
    commands.insert(trs.clone());
}

pub fn prepare_view(
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

pub fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<VoxelTracerPipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    gpu_voxel_trees: Res<RenderAssets<VoxelTree>>,
    voxel_tracer: Res<VoxelTracer>,
    view_uniforms: Res<ViewUniforms>,
    render_device: Res<RenderDevice>,
) {
    let bind_group = voxel_tracer
        .as_bind_group(
            &pipeline.texture_bind_group_layout,
            &render_device,
            &gpu_images,
            &gpu_voxel_trees,
            &view_uniforms,
        )
        .unwrap();

    commands.insert_resource(VoxelTracerBindGroup(bind_group));
}

#[derive(Resource)]
pub struct VoxelTracerPipeline {
    texture_bind_group_layout: BindGroupLayout,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
}

impl FromWorld for VoxelTracerPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let texture_bind_group_layout = VoxelTracer::bind_group_layout(render_device);
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

        VoxelTracerPipeline {
            texture_bind_group_layout,
            init_pipeline,
            update_pipeline,
        }
    }
}

#[derive(Debug)]
pub enum VoxelTracerState {
    Loading,
    Init,
    Update,
}

pub struct VoxelTracerRenderNode {
    pub state: VoxelTracerState,
}

impl Default for VoxelTracerRenderNode {
    fn default() -> Self {
        Self {
            state: VoxelTracerState::Loading,
        }
    }
}

impl render_graph::Node for VoxelTracerRenderNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<VoxelTracerPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            VoxelTracerState::Loading => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline)
                {
                    self.state = VoxelTracerState::Init;
                }
            }
            VoxelTracerState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
                {
                    self.state = VoxelTracerState::Update;
                }
            }
            VoxelTracerState::Update => {
                if pipeline_cache
                    .get_compute_pipeline(pipeline.update_pipeline)
                    .is_none()
                {
                    self.state = VoxelTracerState::Loading;
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
        let texture_bind_group = &world.resource::<VoxelTracerBindGroup>().0;
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<VoxelTracerPipeline>();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, texture_bind_group, &[]);

        // select the pipeline based on the current state
        match self.state {
            VoxelTracerState::Loading => {}
            VoxelTracerState::Init => {
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
            VoxelTracerState::Update => {
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
