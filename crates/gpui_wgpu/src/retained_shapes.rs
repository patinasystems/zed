//! Retained shapes on wgpu: a GPU copy per [`gpui::RetainedShapes`] buffer (a
//! storage buffer, or on WebGL2 a uint texture read like the frame's instance
//! texture), updated with queue writes of the changed ranges only, and one
//! instanced draw per primitive with its uniforms at a dynamic offset.

use super::*;
use gpui::{RetainedShapesDraw, RetainedShapesUniforms, RetainedUpload, ShapeInstance};

/// GPU copies not drawn for this many frames are dropped.
const EVICT_AFTER_FRAMES: u64 = 600;
const UNIFORM_SIZE: u64 = std::mem::size_of::<RetainedShapesUniforms>() as u64;
const RECORD_SIZE: u64 = std::mem::size_of::<ShapeInstance>() as u64;

struct RetainedGpu {
    data: InstanceData,
    bind_group: wgpu::BindGroup,
    /// Bytes the storage holds.
    capacity: u64,
    /// Instances in the copy.
    len: usize,
    version: u64,
    last_frame: u64,
}

struct UniformRing {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    slots: u64,
}

/// Per-device retained shape state; dropped with the other resources on
/// device loss, so every buffer re-uploads.
#[derive(Default)]
pub(super) struct RetainedState {
    entries: FxHashMap<u64, RetainedGpu>,
    frame: u64,
    uniforms: Option<UniformRing>,
    /// Instance bytes uploaded so far.
    pub(super) uploaded_bytes: u64,
}

fn as_bytes<T>(values: &[T]) -> &[u8] {
    // SAFETY: `ShapeInstance` and `RetainedShapesUniforms` are `repr(C)` and
    // made of 4-byte scalars, so they have no padding.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

impl WgpuRendererCore {
    /// Upload what changed in the buffers this scene draws, and write every
    /// draw's uniforms. Returns the uniform slot stride.
    pub(super) fn prepare_retained_shapes(&mut self, scenes: &[(&Scene, [f32; 2])]) -> Result<u64> {
        let uses_texture = self.uses_webgl_instance_data;
        let max_storage = self.max_instance_data_size;
        let resources = &mut self.resources;
        let state = &mut resources.retained;
        state.frame += 1;
        let frame = state.frame;
        let draws = || scenes.iter().flat_map(|(scene, _)| &scene.retained_shapes);
        if draws().next().is_none() {
            return Ok(0);
        }
        let device = &resources.device;
        let queue = &resources.queue;
        let max_texture_dimension = device.limits().max_texture_dimension_2d;
        for draw in draws() {
            let id = draw.shapes.id();
            if state
                .entries
                .get(&id)
                .is_some_and(|e| e.last_frame == frame)
            {
                continue;
            }
            let have = state.entries.get(&id).map_or(0, |e| e.version);
            let mut result: Result<()> = Ok(());
            let mut uploaded = 0u64;
            let entries = &mut state.entries;
            let version = draw.shapes.sync(have, |upload| {
                let (data, ranges): (&[ShapeInstance], Vec<Range<usize>>) = match upload {
                    RetainedUpload::None => return,
                    RetainedUpload::All(data) => (data, vec![0..data.len()]),
                    RetainedUpload::Ranges(data, ranges) => (data, ranges.to_vec()),
                };
                let bytes = as_bytes(data);
                let needs_new = entries
                    .get(&id)
                    .is_none_or(|e| e.capacity < bytes.len() as u64);
                let ranges = if needs_new {
                    let wanted = (bytes.len() as u64 + bytes.len() as u64 / 2).max(256);
                    match create_storage(
                        device,
                        &resources.bind_group_layouts.instances,
                        uses_texture,
                        wanted,
                        max_storage,
                        max_texture_dimension,
                    ) {
                        Ok((data, bind_group, capacity)) => {
                            entries.insert(
                                id,
                                RetainedGpu {
                                    data,
                                    bind_group,
                                    capacity,
                                    len: 0,
                                    version: 0,
                                    last_frame: frame,
                                },
                            );
                        }
                        Err(error) => {
                            result = Err(error);
                            return;
                        }
                    }
                    vec![0..data.len()]
                } else {
                    ranges
                };
                let Some(entry) = entries.get_mut(&id) else {
                    return;
                };
                entry.len = data.len();
                for range in ranges {
                    let range = range.start.min(data.len())..range.end.min(data.len());
                    if range.is_empty() {
                        continue;
                    }
                    write_range(queue, &entry.data, bytes, range.clone());
                    uploaded += range.len() as u64 * RECORD_SIZE;
                }
            });
            result?;
            state.uploaded_bytes += uploaded;
            if let Some(entry) = state.entries.get_mut(&id) {
                entry.version = version;
                entry.last_frame = frame;
            }
        }
        state
            .entries
            .retain(|_, e| frame - e.last_frame < EVICT_AFTER_FRAMES);

        // Uniforms: one slot per draw, at dynamic offsets.
        let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment);
        let stride = UNIFORM_SIZE.next_multiple_of(alignment);
        let count = draws().count() as u64;
        if state
            .uniforms
            .as_ref()
            .is_none_or(|ring| ring.slots < count)
        {
            let slots = count.next_power_of_two().max(16);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("retained_shapes_uniforms"),
                size: slots * stride,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("retained_shapes_uniforms"),
                layout: &resources.bind_group_layouts.retained_uniforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffer,
                        offset: 0,
                        size: NonZeroU64::new(UNIFORM_SIZE),
                    }),
                }],
            });
            state.uniforms = Some(UniformRing {
                buffer,
                bind_group,
                slots,
            });
        }
        let mut bytes = vec![0u8; (count * stride) as usize];
        // Each draw's uniforms take its scene's target size.
        let with_viewports = scenes.iter().flat_map(|(scene, viewport)| {
            scene
                .retained_shapes
                .iter()
                .map(move |draw| (draw, *viewport))
        });
        for (i, (draw, viewport)) in with_viewports.enumerate() {
            let uniforms = draw.uniforms(viewport);
            let at = i * stride as usize;
            bytes[at..at + UNIFORM_SIZE as usize].copy_from_slice(as_bytes(&[uniforms]));
        }
        if let Some(ring) = &state.uniforms {
            queue.write_buffer(&ring.buffer, 0, &bytes);
        }
        Ok(stride)
    }

    /// Draw `draws`, which start at uniform slot `first` of the frame's retained shapes.
    pub(super) fn draw_retained_shapes(
        &self,
        draws: &[RetainedShapesDraw],
        first: usize,
        stride: u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        let resources = self.resources();
        let Some(ring) = &resources.retained.uniforms else {
            return;
        };
        pass.set_pipeline(&resources.pipelines.retained_shapes);
        pass.set_bind_group(0, &resources.globals_bind_group, &[]);
        for (i, draw) in draws.iter().enumerate() {
            let Some(entry) = resources.retained.entries.get(&draw.shapes.id()) else {
                continue;
            };
            let len = entry.len as u32;
            let range = draw.paint.range.start.min(len)..draw.paint.range.end.min(len);
            if range.is_empty() {
                continue;
            }
            pass.set_bind_group(1, &entry.bind_group, &[]);
            pass.set_bind_group(2, &ring.bind_group, &[((first + i) as u64 * stride) as u32]);
            pass.draw(0..4, range);
        }
    }
}

fn create_storage(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uses_texture: bool,
    wanted: u64,
    max_size: u64,
    max_texture_dimension: u32,
) -> Result<(InstanceData, wgpu::BindGroup, u64)> {
    anyhow::ensure!(
        wanted <= max_size,
        "retained shape buffer of {wanted} bytes exceeds the {max_size}-byte maximum"
    );
    let (data, capacity) = if uses_texture {
        WgpuRendererCore::create_instance_texture(device, wanted, max_texture_dimension)
    } else {
        let size = wanted.next_multiple_of(256);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("retained_shapes"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (InstanceData::Storage(buffer), size)
    };
    anyhow::ensure!(
        capacity >= wanted,
        "retained shape storage of {capacity} bytes is below the {wanted} needed"
    );
    let resource = match &data {
        InstanceData::Storage(buffer) => buffer.as_entire_binding(),
        InstanceData::Texture { view, .. } => wgpu::BindingResource::TextureView(view),
    };
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("retained_shapes"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource,
        }],
    });
    Ok((data, bind_group, capacity))
}

/// Copy `range` (instances) of `bytes` into the GPU copy.
fn write_range(queue: &wgpu::Queue, data: &InstanceData, bytes: &[u8], range: Range<usize>) {
    let start = range.start as u64 * RECORD_SIZE;
    let end = range.end as u64 * RECORD_SIZE;
    match data {
        InstanceData::Storage(buffer) => {
            queue.write_buffer(buffer, start, &bytes[start as usize..end as usize]);
        }
        InstanceData::Texture {
            texture,
            width,
            height,
            ..
        } => {
            // Records are 64 bytes and texels 16; a range of records still widens
            // to whole texels in case the record size changes.
            let texel = INSTANCE_TEXTURE_TEXEL_SIZE;
            let first = start / texel * texel;
            let last = end.next_multiple_of(texel).min(bytes.len() as u64);
            WgpuRendererCore::write_texture_bytes(
                queue,
                texture,
                *width,
                *height,
                first,
                &bytes[first as usize..last as usize],
            );
        }
    }
}
