use std::marker::PhantomData;

use bevy::{
    log::tracing_subscriber::filter::targets::Iter,
    prelude::*,
    render::{
        render_resource::{
            encase::internal::{BufferMut, WriteInto, Writer},
            BindingResource, Buffer, BufferDescriptor, BufferUsages, ShaderType,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use color_eyre::owo_colors::OwoColorize;

pub type GpuIdx = u32;

pub struct GpuBufferAllocator<T>
where
    T: ShaderType + WriteInto,
{
    label: &'static str,
    len: GpuIdx,
    cap: GpuIdx,
    buffer: Buffer,
    _phantom: PhantomData<T>,
}

impl<T> GpuBufferAllocator<T>
where
    T: ShaderType + WriteInto,
{
    pub fn new(
        label: &'static str,
        capacity: GpuIdx,
        fill: Option<&T>,
        device: &RenderDevice,
        queue: &RenderQueue,
    ) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: (capacity * u64::from(T::min_size()) as GpuIdx).into(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let size = u64::from(T::min_size()) as GpuIdx * capacity;

        if let Some(fill) = fill {
            let mut buffer_view = queue
                .write_buffer_with(&buffer, 0, (size as u64).try_into().unwrap())
                .unwrap();

            let mut offset = 0usize;
            for _ in 0..capacity {
                fill.write_into(&mut Writer::new(&fill, &mut *buffer_view, offset).unwrap());
                offset += u64::from(T::min_size()) as usize;
            }
            assert_eq!(offset, size as usize);
        }

        Self {
            label,
            len: 0,
            cap: capacity,
            buffer,
            _phantom: PhantomData,
        }
    }

    pub fn alloc(&mut self) -> GpuIdx {
        let res = self.len;
        self.len += 1;
        assert!(self.len <= self.cap);
        res
    }

    pub fn free(&mut self, idx: GpuIdx) {
        todo!("Make a freelist")
    }

    pub fn write(&mut self, idx: GpuIdx, data: &T, queue: &RenderQueue) {
        assert!(idx < self.len);

        let offset = u64::from(T::min_size()) * idx as u64;
        let size = u64::from(T::min_size());

        let mut buffer_view = queue
            .write_buffer_with(&self.buffer, offset, T::min_size())
            .unwrap();

        data.write_into(&mut Writer::new(&data, &mut *buffer_view, 0).unwrap());
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn binding(&self) -> BindingResource<'_> {
        self.buffer.as_entire_binding()
    }

    pub fn size_bytes(&self) -> u64 {
        self.cap as u64 * u64::from(T::min_size())
    }

    pub fn size(&self) -> GpuIdx {
        self.cap
    }
}
