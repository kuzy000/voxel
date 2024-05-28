use core::num;
use std::{io::Cursor, marker::PhantomData, num::NonZeroU64, sync::Arc};

use bevy::{
    core_pipeline::{
        core_3d::CORE_3D_DEPTH_FORMAT,
        deferred::{DEFERRED_LIGHTING_PASS_ID_FORMAT, DEFERRED_PREPASS_FORMAT},
        fullscreen_vertex_shader::fullscreen_shader_vertex_state,
        prepass::{ViewPrepassTextures, MOTION_VECTOR_PREPASS_FORMAT, NORMAL_PREPASS_FORMAT},
    },
    diagnostic::DiagnosticsStore,
    ecs::{query::QueryItem, system::lifetimeless::SResMut},
    prelude::*,
    render::{
        camera::ExtractedCamera,
        prelude::*,
        render_graph::{NodeRunError, RenderGraphContext, ViewNode},
        render_phase::TrackedRenderPass,
        render_resource::{
            binding_types::{storage_buffer, storage_buffer_read_only_sized, storage_buffer_sized},
            encase::internal::{BufferMut, WriteInto, Writer},
        },
        texture::GpuImage,
        view::{ViewDepthTexture, ViewUniform, ViewUniformOffset, ViewUniforms},
    },
    utils::info,
};

use crate::*;

const WORKING_GROUP: IVec3 = IVec3::new(8, 8, 8);
const DISPATCH_SIZE: IVec3 = IVec3::new(16, 16, 16);

pub struct GpuBufferAllocator<T>
where
    T: ShaderType + WriteInto,
{
    label: &'static str,
    size: u64,
    buffer: Buffer,
    _phantom: PhantomData<T>,
}

impl<T> GpuBufferAllocator<T>
where
    T: ShaderType + WriteInto,
{
    pub fn new(label: &'static str, size: u64, device: &RenderDevice) -> Self {
        Self {
            label,
            size,
            buffer: device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: (size * u64::from(T::min_size())).into(),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            _phantom: PhantomData,
        }
    }

    pub fn write(&mut self, offset: u64, data: &[T], device: &RenderDevice, queue: &RenderQueue) {
        assert!((offset + data.len() as u64) <= self.size);

        let offset = u64::from(T::min_size()) * offset;
        let size = u64::from(T::min_size()) * (data.len() as u64);

        let mut buffer_view = queue
            .write_buffer_with(&self.buffer, offset, size.try_into().unwrap())
            .unwrap();

        let mut offset = 0usize;
        for v in data {
            v.write_into(&mut Writer::new(&v, &mut *buffer_view, offset).unwrap());

            offset += u64::from(T::min_size()) as usize;
        }

        assert_eq!(offset, size as usize);
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn binding(&self) -> BindingResource<'_> {
        self.buffer.as_entire_binding()
    }

    pub fn size_bytes(&self) -> u64 {
        self.size * u64::from(T::min_size())
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}

#[derive(Debug, Clone, Copy, ShaderType)]
pub struct VoxelGpuSceneInfo {
    nodes_len: u32,
    nodes_cap: u32,

    leafs_len: u32,
    leafs_cap: u32,
}

#[derive(Resource)]
pub struct VoxelGpuScene {
    pub info: StorageBuffer<VoxelGpuSceneInfo>,
    pub nodes: GpuBufferAllocator<VoxelNode>,
    pub leafs: GpuBufferAllocator<VoxelLeaf>,

    pub bind_group_layout_view: BindGroupLayout,
    pub bind_group_layout_voxel: BindGroupLayout,
}

impl FromWorld for VoxelGpuScene {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();

        let bytes_nodes = 64 * 1024 * 1024; // 64MiB
        let bytes_leafs = 512 * 1024 * 1024; // 512MiB

        let num_nodes = bytes_nodes / std::mem::size_of::<VoxelNode>();
        let num_leafs = bytes_leafs / std::mem::size_of::<VoxelLeaf>();

        info!(
            "Allocating gpu voxel scene; bytes_nodes: {}, bytes_leafs: {}",
            bytes_nodes, bytes_leafs,
        );

        let nodes = GpuBufferAllocator::new("voxel_nodes_buffer", num_nodes as u64, device);
        let leafs = GpuBufferAllocator::new("voxel_leafs_buffer", num_leafs as u64, device);

        let nodes_size = nodes.size_bytes().try_into().unwrap();
        let leafs_size = leafs.size_bytes().try_into().unwrap();

        let info = VoxelGpuSceneInfo {
            nodes_len: 0,
            nodes_cap: nodes.size() as u32,
            leafs_len: 0,
            leafs_cap: leafs.size() as u32,
        };

        Self {
            info: info.into(),
            nodes,
            leafs,
            bind_group_layout_view: device.create_bind_group_layout(
                "voxel_view_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::FRAGMENT,
                    (uniform_buffer::<ViewUniform>(true),),
                ),
            ),
            bind_group_layout_voxel: device.create_bind_group_layout(
                "voxel_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    (
                        storage_buffer::<VoxelGpuSceneInfo>(false),
                        storage_buffer_sized(false, Some(nodes_size)),
                        storage_buffer_sized(false, Some(leafs_size)),
                    ),
                ),
            ),
        }
    }
}

#[derive(Resource)]
pub struct VoxelPipelines {
    prepass: CachedRenderPipelineId,
    draw: CachedComputePipelineId,
}

impl FromWorld for VoxelPipelines {
    fn from_world(world: &mut World) -> Self {
        let gpu_scene = world.resource::<VoxelGpuScene>();
        let shader_prepass = world.load_asset("shaders/voxel_prepass.wgsl");
        let shader_draw = world.load_asset("shaders/draw.wgsl");

        let view_layout = gpu_scene.bind_group_layout_view.clone();
        let voxel_layout = gpu_scene.bind_group_layout_voxel.clone();

        let pipeline_cache = world.resource_mut::<PipelineCache>();
        let shader_defs = vec![
            ShaderDefVal::Int("VOXEL_DIM".into(), VOXEL_DIM as i32),
            ShaderDefVal::Int("VOXEL_TREE_DEPTH".into(), VOXEL_TREE_DEPTH as i32),
            ShaderDefVal::UInt("VOXEL_IDX_EMPTY".into(), VOXEL_IDX_EMPTY as u32),
            // ShaderDefVal::Int("VOXEL_MASK_LEN".into(), VOXEL_MASK_LEN as i32),
        ];

        let shader_defs_compute = [
            shader_defs.as_slice(),
            &[
                ShaderDefVal::UInt("WG_X".into(), WORKING_GROUP.x as u32),
                ShaderDefVal::UInt("WG_Y".into(), WORKING_GROUP.y as u32),
                ShaderDefVal::UInt("WG_Z".into(), WORKING_GROUP.z as u32),
            ],
        ]
        .concat();

        Self {
            prepass: pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("voxel_prepass_pipeline".into()),
                layout: vec![voxel_layout.clone(), view_layout],
                push_constant_ranges: vec![],
                vertex: fullscreen_shader_vertex_state(),
                primitive: default(),
                depth_stencil: Some(DepthStencilState {
                    format: CORE_3D_DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: CompareFunction::GreaterEqual,
                    stencil: StencilState {
                        front: StencilFaceState::IGNORE,
                        back: StencilFaceState::IGNORE,
                        read_mask: 0,
                        write_mask: 0,
                    },
                    bias: DepthBiasState {
                        constant: 0,
                        slope_scale: 0.0,
                        clamp: 0.0,
                    },
                }),
                multisample: default(),
                fragment: Some(FragmentState {
                    shader: shader_prepass,
                    shader_defs: shader_defs.clone(),
                    entry_point: "fragment".into(),
                    targets: vec![
                        Some(ColorTargetState {
                            format: NORMAL_PREPASS_FORMAT,
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        }),
                        Some(ColorTargetState {
                            format: MOTION_VECTOR_PREPASS_FORMAT,
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        }),
                        Some(ColorTargetState {
                            format: DEFERRED_PREPASS_FORMAT,
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        }),
                        Some(ColorTargetState {
                            format: DEFERRED_LIGHTING_PASS_ID_FORMAT,
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        }),
                    ],
                }),
            }),
            draw: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("voxel_draw_pipeline".into()),
                layout: vec![voxel_layout.clone()],
                push_constant_ranges: Vec::new(),
                shader: shader_draw,
                shader_defs: shader_defs_compute.clone(),
                entry_point: "draw".into(),
            }),
        }
    }
}

#[derive(Component)]
pub struct VoxelViewBindGroups {
    view_bind_group: BindGroup,
}

#[derive(Resource)]
pub struct VoxelBindGroups(BindGroup);

pub fn prepare_voxel_bind_groups(
    mut commands: Commands,
    device: Res<RenderDevice>,
    gpu_scene: Res<VoxelGpuScene>,
) {
    let Some(info_uniforms) = gpu_scene.info.binding() else {
        return;
    };

    let nodes_binding = BufferBinding {
        buffer: gpu_scene.nodes.buffer(),
        offset: 0,
        size: Some(gpu_scene.nodes.size_bytes().try_into().unwrap()),
    };

    let leafs_binding = BufferBinding {
        buffer: gpu_scene.leafs.buffer(),
        offset: 0,
        size: Some(gpu_scene.leafs.size_bytes().try_into().unwrap()),
    };

    let bind_group = device.create_bind_group(
        "voxel_bind_group",
        &gpu_scene.bind_group_layout_voxel,
        &BindGroupEntries::sequential((
            info_uniforms.clone(),
            BindingResource::Buffer(nodes_binding.clone()),
            BindingResource::Buffer(leafs_binding.clone()),
        )),
    );

    commands.insert_resource(VoxelBindGroups(bind_group));
}

pub fn prepare_voxel_view_bind_groups(
    mut commands: Commands,
    device: Res<RenderDevice>,
    gpu_scene: Res<VoxelGpuScene>,
    voxel_bind_groups: Res<VoxelBindGroups>,
    view_uniforms: Res<ViewUniforms>,
    views: Query<Entity, With<ExtractedCamera>>,
) {
    let Some(view_uniforms) = view_uniforms.uniforms.binding() else {
        return;
    };

    let Some(info_uniforms) = gpu_scene.info.binding() else {
        return;
    };

    let nodes_binding = BufferBinding {
        buffer: gpu_scene.nodes.buffer(),
        offset: 0,
        size: Some(gpu_scene.nodes.size_bytes().try_into().unwrap()),
    };

    let leafs_binding = BufferBinding {
        buffer: gpu_scene.leafs.buffer(),
        offset: 0,
        size: Some(gpu_scene.leafs.size_bytes().try_into().unwrap()),
    };

    for view_entity in &views {
        let view_bind_group = device.create_bind_group(
            "voxel_view_bind_group",
            &gpu_scene.bind_group_layout_view,
            &BindGroupEntries::sequential((view_uniforms.clone(),)),
        );

        commands
            .entity(view_entity)
            .insert(VoxelViewBindGroups { view_bind_group });
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct VoxelWorldPepassNodeLabel;

#[derive(Default)]
pub struct VoxelWorldPepassNode;

impl ViewNode for VoxelWorldPepassNode {
    type ViewQuery = (
        &'static ViewUniformOffset,
        &'static VoxelViewBindGroups,
        &'static ViewDepthTexture,
        &'static ViewPrepassTextures,
    );

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (view_uniform_offset, bind_groups, view_depth_texture, view_prepass_textures): QueryItem<
            'w,
            Self::ViewQuery,
        >,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let voxel_pipelines = world.resource::<VoxelPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let voxel_scene = world.resource::<VoxelGpuScene>();
        let voxel_bind_group = world.resource::<VoxelBindGroups>();

        let Some(pipeline) = pipeline_cache.get_render_pipeline(voxel_pipelines.prepass) else {
            return Ok(());
        };

        let mut color_attachments = vec![];
        color_attachments.push(
            view_prepass_textures
                .normal
                .as_ref()
                .map(|normals_texture| normals_texture.get_attachment()),
        );
        color_attachments.push(
            view_prepass_textures
                .motion_vectors
                .as_ref()
                .map(|motion_vectors_texture| motion_vectors_texture.get_attachment()),
        );
        color_attachments.push(
            view_prepass_textures
                .deferred
                .as_ref()
                .map(|deferred_texture| deferred_texture.get_attachment()),
        );
        color_attachments.push(
            view_prepass_textures
                .deferred_lighting_pass_id
                .as_ref()
                .map(|deferred_lighting_pass_id| deferred_lighting_pass_id.get_attachment()),
        );

        let depth_stencil_attachment = Some(view_depth_texture.get_attachment(StoreOp::Store));

        let pass_descriptor = RenderPassDescriptor {
            label: Some("voxel_prepass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
        };

        let mut render_pass = render_context
            .command_encoder()
            .begin_render_pass(&pass_descriptor);

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &voxel_bind_group.0, &[]);
        render_pass.set_bind_group(
            1,
            &bind_groups.view_bind_group,
            &[view_uniform_offset.offset],
        );
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

#[derive(Default)]
pub struct GpuVoxelTree;

impl RenderAsset for GpuVoxelTree {
    type SourceAsset = VoxelTree;
    type Param = (
        SRes<RenderDevice>,
        SRes<RenderQueue>,
        SResMut<VoxelGpuScene>,
    );

    fn asset_usage(_source_asset: &Self::SourceAsset) -> RenderAssetUsages {
        RenderAssetUsages::RENDER_WORLD
    }

    fn prepare_asset(
        source_asset: Self::SourceAsset,
        (device, queue, gpu_scene): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        {
            let gn = gpu_scene.nodes.size();
            let gl = gpu_scene.leafs.size();

            let sn = source_asset.nodes.len() as u64;
            let sl = source_asset.leafs.len() as u64;

            assert!(gn >= sn, "nodes {} >= {}", gn, sn);
            assert!(gl >= sl, "leafs {} >= {}", gl, sl);
        }

        {
            let nodes = vec![VoxelNode::default(); gpu_scene.nodes.size() as usize];
            gpu_scene.nodes.write(0, nodes.as_slice(), device, queue);
        }
        {
            let leafs = vec![VoxelLeaf::default(); gpu_scene.leafs.size() as usize];
            gpu_scene.leafs.write(0, leafs.as_slice(), device, queue);
        }

        gpu_scene
            .nodes
            .write(0, source_asset.nodes.as_slice(), device, queue);
        gpu_scene
            .leafs
            .write(0, source_asset.leafs.as_slice(), device, queue);

        gpu_scene.info.get_mut().nodes_len = source_asset.nodes.len() as u32;
        gpu_scene.info.get_mut().leafs_len = source_asset.leafs.len() as u32;
        gpu_scene.info.write_buffer(device, queue);

        info!("VoxelTree extracted; info: {:?}", &gpu_scene.info.get());

        Ok(Self)
    }
}

pub enum VoxelDrawState {
    Loading,
    Run,
    Done,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct VoxelDrawNodeLabel;

pub struct VoxelDrawNode {
    state: VoxelDrawState,
}

impl Default for VoxelDrawNode {
    fn default() -> Self {
        Self {
            state: VoxelDrawState::Loading,
        }
    }
}

impl render_graph::Node for VoxelDrawNode {
    fn update(&mut self, world: &mut World) {
        let voxel_pipelines = world.resource::<VoxelPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();

        match self.state {
            VoxelDrawState::Loading => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(voxel_pipelines.draw)
                {
                    self.state = VoxelDrawState::Run;
                }
            }
            VoxelDrawState::Run => {
                self.state = VoxelDrawState::Done;
            }
            VoxelDrawState::Done => {
                match pipeline_cache.get_compute_pipeline_state(voxel_pipelines.draw) {
                    CachedPipelineState::Ok(_) => {
                        self.state = VoxelDrawState::Loading;
                    }
                    _ => (),
                }
            }
        }
    }

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let VoxelDrawState::Run = self.state else {
            return Ok(());
        };

        let voxel_pipelines = world.resource::<VoxelPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let voxel_scene = world.resource::<VoxelGpuScene>();
        let voxel_bind_group = world.resource::<VoxelBindGroups>();

        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: "voxel_draw".into(),
                    ..default()
                });

        pass.set_pipeline(
            pipeline_cache
                .get_compute_pipeline(voxel_pipelines.draw)
                .unwrap(),
        );

        pass.set_bind_group(0, &voxel_bind_group.0, &[]);

        let x = DISPATCH_SIZE.x.try_into().unwrap();
        let y = DISPATCH_SIZE.y.try_into().unwrap();
        let z = DISPATCH_SIZE.z.try_into().unwrap();

        pass.dispatch_workgroups(x, y, z);

        Ok(())
    }
}
