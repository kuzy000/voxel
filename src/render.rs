use core::num;
use std::{num::NonZeroU64, sync::Arc};

use bevy::{
    core_pipeline::{
        core_3d::CORE_3D_DEPTH_FORMAT,
        deferred::{DEFERRED_LIGHTING_PASS_ID_FORMAT, DEFERRED_PREPASS_FORMAT},
        fullscreen_vertex_shader::fullscreen_shader_vertex_state,
        prepass::{ViewPrepassTextures, MOTION_VECTOR_PREPASS_FORMAT, NORMAL_PREPASS_FORMAT},
    },
    ecs::{query::QueryItem, system::lifetimeless::SResMut},
    prelude::*,
    render::{
        camera::ExtractedCamera,
        prelude::*,
        render_graph::{NodeRunError, RenderGraphContext, ViewNode},
        render_phase::TrackedRenderPass,
        render_resource::binding_types::storage_buffer_read_only_sized,
        texture::GpuImage,
        view::{ViewDepthTexture, ViewUniform, ViewUniformOffset, ViewUniforms},
    },
    utils::info,
};

use crate::*;

#[derive(Resource)]
pub struct VoxelGpuScene {
    pub nodes: BufferVec<VoxelNode>,
    pub leafs: BufferVec<VoxelLeaf>,

    pub bind_group_layout: BindGroupLayout,
}

impl FromWorld for VoxelGpuScene {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();

        let num_nodes = 1024 * 64;
        let num_leafs = 1024 * 1024;

        let bytes_nodes = num_nodes * std::mem::size_of::<VoxelNode>();
        let bytes_leafs = num_leafs * std::mem::size_of::<VoxelLeaf>();

        info!(
            "Allocating gpu voxel scene; bytes_nodes: {}, bytes_leafs: {}",
            bytes_nodes,
            bytes_leafs,
        );

        let mut res = Self {
            nodes: BufferVec::new(BufferUsages::STORAGE),
            leafs: BufferVec::new(BufferUsages::STORAGE),
            bind_group_layout: device.create_bind_group_layout(
                "voxel_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::FRAGMENT,
                    (
                        uniform_buffer::<ViewUniform>(true),
                        storage_buffer_read_only_sized(
                            false,
                            Some((bytes_nodes as u64).try_into().unwrap()),
                        ),
                        storage_buffer_read_only_sized(
                            false,
                            Some((bytes_leafs as u64).try_into().unwrap()),
                        ),
                    ),
                ),
            ),
        };

        res.nodes.reserve(num_nodes, &device);
        res.leafs.reserve(num_leafs, &device);

        res
    }
}

#[derive(Resource)]
pub struct VoxelPipelines {
    prepass: CachedRenderPipelineId,
}

impl FromWorld for VoxelPipelines {
    fn from_world(world: &mut World) -> Self {
        let gpu_scene = world.resource::<VoxelGpuScene>();
        let bind_group_layout = gpu_scene.bind_group_layout.clone();
        let shader = world.load_asset("shaders/voxel_prepass.wgsl");
        let pipeline_cache = world.resource_mut::<PipelineCache>();

        Self {
            prepass: pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("voxel_prepass_pipeline".into()),
                layout: vec![bind_group_layout],
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
                    shader: shader.clone(),
                    shader_defs: vec![],
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
        }
    }
}

#[derive(Component)]
pub struct VoxelViewBindGroups {
    bind_group: BindGroup,
}

pub fn prepare_voxel_view_bind_groups(
    mut commands: Commands,
    device: Res<RenderDevice>,
    gpu_scene: Res<VoxelGpuScene>,
    view_uniforms: Res<ViewUniforms>,
    views: Query<Entity, With<ExtractedCamera>>,
) {
    let Some(view_uniforms) = view_uniforms.uniforms.binding() else {
        return;
    };

    let Some(nodes) = gpu_scene.nodes.buffer() else {
        return;
    };

    let Some(leafs) = gpu_scene.leafs.buffer() else {
        return;
    };

    for view_entity in &views {
        let bind_group = device.create_bind_group(
            "voxel_bind_group",
            &gpu_scene.bind_group_layout,
            &BindGroupEntries::sequential((
                view_uniforms.clone(),
                nodes.as_entire_binding(),
                leafs.as_entire_binding(),
            )),
        );

        commands
            .entity(view_entity)
            .insert(VoxelViewBindGroups { bind_group });
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
        render_pass.set_bind_group(0, &bind_groups.bind_group, &[view_uniform_offset.offset]);
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
        let mut nodes = StorageBuffer::default();
        nodes.set(source_asset.nodes.clone());
        nodes.write_buffer(device, queue);

        let mut leafs = StorageBuffer::default();
        leafs.set(source_asset.leafs.clone());
        leafs.write_buffer(device, queue);

        {
            let gn = gpu_scene.nodes.capacity();
            let gl = gpu_scene.leafs.capacity();

            let sn = source_asset.nodes.capacity();
            let sl = source_asset.leafs.capacity();

            assert!(gn >= sn, "nodes {} >= {}", gn, sn);
            assert!(gl >= sl, "leafs {} >= {}", gl, sl);
        }

        for node in &source_asset.nodes {
            gpu_scene.nodes.push(node.clone());
        }

        for leaf in &source_asset.leafs {
            gpu_scene.leafs.push(leaf.clone());
        }

        gpu_scene.nodes.write_buffer(device, queue);
        gpu_scene.leafs.write_buffer(device, queue);

        info!(
            "VoxelTree extracted; nodes: {}, leafs: {}",
            source_asset.nodes.len(),
            source_asset.leafs.len()
        );

        Ok(Self)
    }
}
