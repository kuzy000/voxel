use std::marker::PhantomData;

use bevy::{
    prelude::*,
    render::{
        render_resource::{encase::{internal::{WriteInto, Writer}, StorageBuffer}, Buffer, BufferDescriptor, BufferUsages, ShaderType},
        renderer::{RenderDevice, RenderQueue},
    },
};

pub struct GpuRwBuffer<T>
where
    T: ShaderType + WriteInto,
{
    label: &'static str,
    scratch: StorageBuffer<Vec<u8>>,
    buffer: Buffer,
    _phantom: PhantomData<T>,
}

impl<T> GpuRwBuffer<T>
where
    T: ShaderType + WriteInto,
{
    pub fn new(label: &'static str, device: &RenderDevice) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: T::min_size().into(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            label,
            scratch: StorageBuffer::new(Vec::new()),
            buffer,
            _phantom: default(),
        }
    }
    
    pub fn write(&mut self, data: &T, queue: &RenderQueue) {
        self.scratch.write(data);
        queue.write_buffer(&self.buffer, 0, self.scratch.as_ref());
    }
    
    pub fn read(&self, queue: &RenderQueue) {

    }
}
