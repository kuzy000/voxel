use bevy::{
    core_pipeline::prepass::ViewPrepassTextures,
    prelude::*,
    render::{
        render_resource::{binding_types::texture_2d, StorageBuffer, UniformBuffer},
        view::{ViewUniform, ViewUniforms},
    },
};

use crate::*;

pub const SCREEN_SIZE: (u32, u32) = (1280, 720);
pub const WORKGROUP_SIZE: u32 = 8;

#[derive(Default, Resource)]
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
pub struct VoxelTracerPipelines {
    view_bind_group_layout: BindGroupLayout,
    prepass_textures_bind_group_layout: BindGroupLayout,
    voxel_tree_bind_group_layout: BindGroupLayout,

    pipeline: CachedComputePipelineId,
}

impl FromWorld for VoxelTracerPipelines {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let texture_bind_group_layout = VoxelTracer::bind_group_layout(render_device);
        let shader = world.resource::<AssetServer>().load("shaders/test.wgsl");

        let view_bind_group_layout = render_device.create_bind_group_layout(
            "voxel_tracer_view_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (uniform_buffer::<ViewUniform>(true),),
            ),
        );

        let prepass_textures_bind_group_layout = render_device.create_bind_group_layout(
            "voxel_tracer_prepass_textures_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (texture_storage_2d(
                    TextureFormat::Rgba32Uint,
                    StorageTextureAccess::ReadWrite,
                ),),
            ),
        );

        let voxel_tree_bind_group_layout = render_device.create_bind_group_layout(
            "voxel_tracer_voxel_tree_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_read_only::<Vec<VoxelNode>>(false),
                    storage_buffer_read_only::<Vec<VoxelLeaf>>(false),
                ),
            ),
        );

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![
                view_bind_group_layout.clone(),
                prepass_textures_bind_group_layout.clone(),
                voxel_tree_bind_group_layout.clone(),
            ],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("update"),
        });

        VoxelTracerPipelines {
            view_bind_group_layout,
            prepass_textures_bind_group_layout,
            voxel_tree_bind_group_layout,
            pipeline,
        }
    }
}

#[derive(Resource)]
pub struct VoxelTracerBindGroups {
    view: BindGroup,
    prepass_textures: BindGroup,
    voxel_tree: BindGroup,
}

pub fn prepare_voxel_tracer_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    voxel_tracer: Res<VoxelTracer>,
    gpu_voxel_trees: Res<RenderAssets<VoxelTree>>,
    pipelines: Res<VoxelTracerPipelines>,
    view_uniforms: Res<ViewUniforms>,
    q: Query<&ViewPrepassTextures>,
) {
    let Some(view_uniforms) = view_uniforms.uniforms.binding() else {
        return;
    };

    let prepass_textures = q.single();

    let view = render_device.create_bind_group(
        "voxel_tracer_view_bind_group",
        &pipelines.view_bind_group_layout,
        &BindGroupEntries::single(view_uniforms.clone()),
    );

    let prepass_textures = render_device.create_bind_group(
        "voxel_tracer_prepass_textures_bind_group",
        &pipelines.prepass_textures_bind_group_layout,
        &BindGroupEntries::single(prepass_textures.deferred_view().unwrap()),
    );

    let voxel_tree = gpu_voxel_trees
        .get(voxel_tracer.voxel_tree.clone())
        .ok_or(AsBindGroupError::RetryNextUpdate)
        .unwrap();

    let voxel_nodes = voxel_tree
        .nodes
        .binding()
        .ok_or(AsBindGroupError::RetryNextUpdate)
        .unwrap();

    let voxel_leafs = voxel_tree
        .leafs
        .binding()
        .ok_or(AsBindGroupError::RetryNextUpdate)
        .unwrap();

    let voxel_tree = render_device.create_bind_group(
        "voxel_tracer_voxel_tree_bind_group",
        &pipelines.voxel_tree_bind_group_layout,
        &BindGroupEntries::sequential((voxel_nodes, voxel_leafs)),
    );

    commands.insert_resource(VoxelTracerBindGroups {
        view,
        prepass_textures,
        voxel_tree,
    });
}

#[derive(Debug)]
pub enum VoxelTracerState {
    Loading,
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
        let pipelines = world.resource::<VoxelTracerPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            VoxelTracerState::Loading => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipelines.pipeline)
                {
                    self.state = VoxelTracerState::Update;
                }
            }
            VoxelTracerState::Update => {
                if pipeline_cache
                    .get_compute_pipeline(pipelines.pipeline)
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
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<VoxelTracerPipelines>();

        let bind_groups = &world.resource::<VoxelTracerBindGroups>();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, &bind_groups.view, &[]);
        pass.set_bind_group(1, &bind_groups.prepass_textures, &[]);
        pass.set_bind_group(2, &bind_groups.voxel_tree, &[]);

        // select the pipeline based on the current state
        match self.state {
            VoxelTracerState::Loading => {}
            VoxelTracerState::Update => {
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.pipeline)
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
