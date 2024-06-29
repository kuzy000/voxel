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
use gpu_buffer_allocator::{GpuBufferAllocator, GpuIdx};

use crate::*;

const WORKGROUP_SIZE: UVec3 = UVec3::new(8, 8, 8);

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
        let queue = world.resource::<RenderQueue>();

        let bytes_nodes = 256 * 1024 * 1024; // 256MiB
        let bytes_leafs = 2 * 1024 * 1024 * 1024; // 2GiB

        let num_nodes = bytes_nodes / std::mem::size_of::<VoxelNode>();
        let num_leafs = bytes_leafs / std::mem::size_of::<VoxelLeaf>();

        info!(
            "Allocating gpu voxel scene; bytes_nodes: {}, bytes_leafs: {}",
            bytes_nodes, bytes_leafs,
        );

        let nodes = GpuBufferAllocator::new(
            "voxel_nodes_buffer",
            num_nodes as GpuIdx,
            None,
            device,
            queue,
        );
        let leafs = GpuBufferAllocator::new(
            "voxel_leafs_buffer",
            num_leafs as GpuIdx,
            None,
            device,
            queue,
        );

        let nodes_size = nodes.size_bytes().try_into().unwrap();
        let leafs_size = leafs.size_bytes().try_into().unwrap();

        let mut info: StorageBuffer<_> = VoxelGpuSceneInfo {
            nodes_len: 1, // The first one is reserved for root node
            nodes_cap: nodes.size() as u32,
            leafs_len: 0,
            leafs_cap: leafs.size() as u32,
        }
        .into();

        info.write_buffer(device, queue);

        Self {
            info,
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
    clear_world: CachedComputePipelineId,
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
                ShaderDefVal::UInt("WG_X".into(), WORKGROUP_SIZE.x),
                ShaderDefVal::UInt("WG_Y".into(), WORKGROUP_SIZE.y),
                ShaderDefVal::UInt("WG_Z".into(), WORKGROUP_SIZE.z),
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
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..(4 * 4 * 5), // min: vec3i, max: vec3i, world_min: vec3i, world_max: vec3i, depth: u32
                }],
                shader: shader_draw.clone(),
                shader_defs: shader_defs_compute.clone(),
                entry_point: "draw".into(),
            }),
            clear_world: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("voxel_clear_world_pipeline".into()),
                layout: vec![voxel_layout.clone()],
                push_constant_ranges: vec![],
                shader: shader_draw.clone(),
                shader_defs: shader_defs_compute.clone(),
                entry_point: "clear_world".into(),
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

            let sn = source_asset.nodes.len() as GpuIdx;
            let sl = source_asset.leafs.len() as GpuIdx;

            assert!(gn >= sn, "nodes {} >= {}", gn, sn);
            assert!(gl >= sl, "leafs {} >= {}", gl, sl);
        }

        for node in &source_asset.nodes {
            let idx = gpu_scene.nodes.alloc();
            gpu_scene.nodes.write(idx, node, queue);
        }

        for leaf in &source_asset.leafs {
            let idx = gpu_scene.leafs.alloc();
            gpu_scene.leafs.write(idx, leaf, queue);
        }

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
                let mut ready = 0;

                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(voxel_pipelines.draw)
                {
                    ready += 1;
                }

                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(voxel_pipelines.clear_world)
                {
                    ready += 1;
                }

                if ready == 2 {
                    self.state = VoxelDrawState::Run;
                }
            }
            VoxelDrawState::Run => {
                self.state = VoxelDrawState::Done;
            }
            VoxelDrawState::Done => {
                let mut ready = 0;

                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(voxel_pipelines.draw)
                {
                    ready += 1;
                }

                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(voxel_pipelines.clear_world)
                {
                    ready += 1;
                }

                if ready != 2 {
                    self.state = VoxelDrawState::Loading;
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

        {
            let nodes_cap = voxel_scene.info.get().nodes_cap;
            let leafs_cap = voxel_scene.info.get().leafs_cap;
            let count = nodes_cap + leafs_cap;

            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: "voxel_clear_world".into(),
                        ..default()
                    });

            pass.set_pipeline(
                pipeline_cache
                    .get_compute_pipeline(voxel_pipelines.clear_world)
                    .unwrap(),
            );

            pass.set_bind_group(0, &voxel_bind_group.0, &[]);

            let wg = WORKGROUP_SIZE.x * WORKGROUP_SIZE.y * WORKGROUP_SIZE.z;
            let dispatch = (count + wg - 1) / wg;
            pass.dispatch_workgroups(dispatch, 1, 1);

            info!(
                "Clear; dispatch: {}, nodes_cap: {}, leafs_cap: {}",
                dispatch, nodes_cap, leafs_cap
            );
        }

        {
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

            let world_min = UVec3::new(0, 0, 0);
            let world_max = UVec3::new(512, 512, 512);

            for depth in 0..VOXEL_TREE_DEPTH {
                let size = (VOXEL_DIM as u32).pow((VOXEL_TREE_DEPTH - 1 - depth) as u32);
                let min = world_min / size;
                let max = (world_max - UVec3::ONE) / size + UVec3::ONE;

                pass.set_push_constants(4 * 0, &(min.x as i32).to_ne_bytes());
                pass.set_push_constants(4 * 1, &(min.y as i32).to_ne_bytes());
                pass.set_push_constants(4 * 2, &(min.z as i32).to_ne_bytes());
                pass.set_push_constants(4 * 3, &(0 as i32).to_ne_bytes());

                pass.set_push_constants(4 * 4, &(max.x as i32).to_ne_bytes());
                pass.set_push_constants(4 * 5, &(max.y as i32).to_ne_bytes());
                pass.set_push_constants(4 * 6, &(max.z as i32).to_ne_bytes());
                pass.set_push_constants(4 * 7, &(0 as i32).to_ne_bytes());

                pass.set_push_constants(4 * 8, &(world_min.x as i32).to_ne_bytes());
                pass.set_push_constants(4 * 9, &(world_min.y as i32).to_ne_bytes());
                pass.set_push_constants(4 * 10, &(world_min.z as i32).to_ne_bytes());
                pass.set_push_constants(4 * 11, &(0 as i32).to_ne_bytes());

                pass.set_push_constants(4 * 12, &(world_max.x as i32).to_ne_bytes());
                pass.set_push_constants(4 * 13, &(world_max.y as i32).to_ne_bytes());
                pass.set_push_constants(4 * 14, &(world_max.z as i32).to_ne_bytes());
                pass.set_push_constants(4 * 15, &(0 as i32).to_ne_bytes());

                pass.set_push_constants(4 * 16, &(depth as u32).to_ne_bytes());

                let dispatch_size = ((max - min + WORKGROUP_SIZE - UVec3::splat(1))
                    / WORKGROUP_SIZE)
                    .max(UVec3::ONE);

                info!(
                    "Draw; depth: {}, min: {}, max: {}, dispatch: {}",
                    depth, min, max, dispatch_size
                );
                pass.dispatch_workgroups(dispatch_size.x, dispatch_size.y, dispatch_size.z);
            }

            Ok(())
        }
    }
}
