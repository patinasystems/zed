//! Marks (`gpui::MarksShader`) on wgpu: the prelude and the program, validated
//! with Naga, compiled once per program, and drawn inside the frame's open pass
//! over the caller's `wgpu::Buffer`s. Each draw's uniforms go in one per-frame
//! ring buffer; its bind group is made while the frame is prepared.

use super::*;
use gpui::{
    MARKS_BUFFER_OFFSET_ALIGNMENT, MARKS_UNIFORM_BYTES, MAX_MARKS_BUFFERS, MAX_MARKS_TEXTURES,
    MarksDraw, MarksResource, MarksShader, MarksShaderId, MarksUniforms,
};

const MARKS_UNIFORMS_SIZE: u64 = std::mem::size_of::<MarksUniforms>() as u64;
/// Where a slot's program uniforms start, past the prelude's (aligned for any device).
const PROGRAM_UNIFORMS_AT: u64 = 512;
/// One draw's uniforms: the prelude's, then the program's.
const SLOT: u64 = PROGRAM_UNIFORMS_AT + MARKS_UNIFORM_BYTES as u64;

/// What a resource declared in a module is.
fn resource_kind(module: &naga::Module, variable: &naga::GlobalVariable) -> MarksResource {
    match variable.space {
        naga::AddressSpace::Uniform => MarksResource::Uniform,
        naga::AddressSpace::Storage { access } => {
            if access.contains(naga::StorageAccess::STORE) {
                MarksResource::StorageWrite
            } else {
                MarksResource::StorageRead
            }
        }
        naga::AddressSpace::Handle => match module.types[variable.ty].inner {
            naga::TypeInner::Image {
                dim: naga::ImageDimension::D2,
                arrayed: false,
                class:
                    naga::ImageClass::Sampled {
                        kind: naga::ScalarKind::Float,
                        multi: false,
                    },
            } => MarksResource::Texture2d,
            naga::TypeInner::Sampler { comparison: false } => MarksResource::Sampler,
            _ => MarksResource::Other,
        },
        _ => MarksResource::Other,
    }
}

/// Every storage buffer a program declares, and the fewest bytes it reads (a
/// fixed-size type's size, or one element of a runtime-sized array).
type StorageNeeds = Vec<(usize, u64)>;

/// Validates `shader` (with the prelude) as wgpu would compile it. Returns the
/// module's source and its storage needs.
pub(crate) fn validate(shader: &MarksShader) -> Result<(String, StorageNeeds)> {
    let source = shader.module_source();
    let module = naga::front::wgsl::parse_str(&source)
        .map_err(|error| anyhow::anyhow!(error.emit_to_string(&source)))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|error| anyhow::anyhow!(error.emit_to_string(&source)))?;
    let mut storage = Vec::new();
    for (_, variable) in module.global_variables.iter() {
        let Some(binding) = &variable.binding else {
            continue;
        };
        let kind = resource_kind(&module, variable);
        if !gpui::marks_binding_allowed(
            binding.group,
            binding.binding,
            variable.name.as_deref(),
            kind,
        ) {
            anyhow::bail!(gpui::marks_binding_error(binding.group, binding.binding));
        }
        let size = module.types[variable.ty]
            .inner
            .try_size(module.to_ctx())
            .ok_or_else(|| anyhow::anyhow!("@binding({}) is too large", binding.binding))?;
        if binding.binding == 1 && size as usize > MARKS_UNIFORM_BYTES {
            anyhow::bail!(
                "@binding(1) declares {size} bytes of uniforms; marks take at most {MARKS_UNIFORM_BYTES}"
            );
        }
        if kind == MarksResource::StorageRead {
            storage.push(((binding.binding - 2) as usize, u64::from(size)));
        }
    }
    Ok((source, storage))
}

/// Device objects every marks draw shares.
struct MarksGpu {
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
    /// Stands in for buffers a draw does not bind.
    blank_buffer: wgpu::Buffer,
    /// Stands in for textures a draw does not bind, or that are gone.
    blank_texture: wgpu::TextureView,
}

impl MarksGpu {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let stages = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: stages,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mut entries = vec![uniform(0), uniform(1)];
        for i in 0..MAX_MARKS_BUFFERS as u32 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 2 + i,
                visibility: stages,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        for i in 0..MAX_MARKS_TEXTURES as u32 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 6 + i,
                visibility: stages,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility: stages,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("marks_layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("marks_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("marks_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let blank_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("marks_blank_buffer"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let blank = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("marks_blank_texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            blank.as_image_copy(),
            &[0; 4],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        Self {
            layout,
            pipeline_layout,
            sampler,
            blank_buffer,
            blank_texture: blank.create_view(&wgpu::TextureViewDescriptor::default()),
        }
    }
}

/// Per-device marks state; dropped with the other resources on device loss.
#[derive(Default)]
pub(super) struct MarksState {
    gpu: Option<MarksGpu>,
    /// `None` for a program that failed to compile (logged once).
    pipelines: FxHashMap<
        (MarksShaderId, gpui::MarksTopology),
        Option<(wgpu::RenderPipeline, Rc<StorageNeeds>)>,
    >,
    /// The frame's uniform ring.
    uniforms: Option<(wgpu::Buffer, u64)>,
    /// The frame's draws, in the order `prepare_marks` saw them: what to bind,
    /// or `None` for a draw that is skipped.
    frame: Vec<Option<(wgpu::RenderPipeline, wgpu::BindGroup)>>,
    /// Whether "marks need storage buffers" was logged (WebGL2).
    warned_webgl: bool,
}

/// The draw's storage buffers, if every one is a `wgpu::Buffer` with `STORAGE`
/// usage at an aligned offset inside it.
fn wgpu_buffers(draw: &MarksDraw) -> Option<SmallVec<[(&wgpu::Buffer, u64); MAX_MARKS_BUFFERS]>> {
    draw.bindings
        .buffers
        .iter()
        .map(|binding| {
            let buffer = binding.downcast_ref::<wgpu::Buffer>()?;
            (buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                && binding.offset % MARKS_BUFFER_OFFSET_ALIGNMENT == 0
                && binding.offset < buffer.size())
            .then_some((buffer, binding.offset))
        })
        .collect()
}

impl WgpuRendererCore {
    /// Compiles the frame's marks programs, writes every draw's uniforms and makes
    /// its bind group, in the order of `scenes` and their draws.
    pub(super) fn prepare_marks(&mut self, scenes: &[(&Scene, [f32; 2])]) {
        let format = self.target_format;
        let webgl = self.uses_webgl_instance_data;
        let atlas = self.atlas.clone();
        let resources = &mut self.resources;
        let state = &mut resources.marks;
        state.frame.clear();
        let count: usize = scenes.iter().map(|(scene, _)| scene.marks.len()).sum();
        if count == 0 {
            return;
        }
        if webgl {
            if !state.warned_webgl {
                log::warn!("marks need storage buffers, which WebGL2 lacks: not drawn");
                state.warned_webgl = true;
            }
            state.frame.resize(count, None);
            return;
        }
        let device = &resources.device;
        let queue = &resources.queue;
        let max_binding = u64::from(device.limits().max_storage_buffer_binding_size);
        let gpu = state
            .gpu
            .get_or_insert_with(|| MarksGpu::new(device, queue));

        let needed = count as u64 * SLOT;
        if state
            .uniforms
            .as_ref()
            .is_none_or(|(_, size)| *size < needed)
        {
            let size = (count as u64).next_power_of_two().max(16) * SLOT;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("marks_uniforms"),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            state.uniforms = Some((buffer, size));
        }
        let Some((ring, _)) = &state.uniforms else {
            return;
        };
        let mut bytes = vec![0u8; needed as usize];
        let draws = scenes
            .iter()
            .flat_map(|(scene, viewport)| scene.marks.iter().map(move |draw| (draw, *viewport)));
        for (i, (draw, viewport)) in draws.enumerate() {
            let at = i as u64 * SLOT;
            let pipeline = state
                .pipelines
                .entry((draw.shader.id, draw.paint.topology))
                .or_insert_with(|| {
                    build_pipeline(device, gpu, format, &draw.shader, draw.paint.topology)
                })
                .clone();
            let Some((pipeline, needs)) = pipeline else {
                state.frame.push(None);
                continue;
            };
            let Some(buffers) = wgpu_buffers(draw) else {
                draw.shader.report_skipped_draw(
                    "a buffer is not a STORAGE wgpu::Buffer or its offset is not 256-aligned",
                );
                state.frame.push(None);
                continue;
            };
            // A storage binding takes at most the device's limit; the program sees the
            // rest of a larger buffer as zeros.
            let binding_size =
                |buffer: &wgpu::Buffer, offset: u64| (buffer.size() - offset).min(max_binding) & !3;
            // wgpu binds whole words: a buffer with under four bytes past its offset
            // cannot be bound at all.
            if buffers
                .iter()
                .any(|(buffer, offset)| binding_size(buffer, *offset) == 0)
            {
                draw.shader
                    .report_skipped_draw("a buffer has under 4 bytes past its offset");
                state.frame.push(None);
                continue;
            }
            let missing = needs.iter().find(|(index, min)| {
                buffers
                    .get(*index)
                    .is_none_or(|(buffer, offset)| binding_size(buffer, *offset) < *min)
            });
            if let Some((index, min)) = missing {
                draw.shader.report_skipped_draw(format!(
                    "storage buffer {index} is missing or under the {min} bytes its program reads"
                ));
                state.frame.push(None);
                continue;
            }
            let uniforms = draw.uniforms(viewport);
            bytes[at as usize..(at + MARKS_UNIFORMS_SIZE) as usize]
                .copy_from_slice(uniforms.as_bytes());
            let program = at + PROGRAM_UNIFORMS_AT;
            bytes[program as usize..(program as usize + MARKS_UNIFORM_BYTES)]
                .copy_from_slice(&draw.uniform_block());
            let views: SmallVec<[Option<wgpu::TextureView>; MAX_MARKS_TEXTURES]> = draw
                .bindings
                .textures
                .iter()
                .map(|id| {
                    let tile = gpui::PlatformAtlas::external_texture_tile(&*atlas, *id)?;
                    Some(atlas.get_texture_info(tile.texture_id)?.view)
                })
                .collect();
            let ring_binding = |offset, size| {
                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: ring,
                    offset,
                    size: NonZeroU64::new(size),
                })
            };
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ring_binding(at, MARKS_UNIFORMS_SIZE),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ring_binding(program, MARKS_UNIFORM_BYTES as u64),
                },
            ];
            for i in 0..MAX_MARKS_BUFFERS {
                let (buffer, offset) = buffers.get(i).copied().unwrap_or((&gpu.blank_buffer, 0));
                entries.push(wgpu::BindGroupEntry {
                    binding: 2 + i as u32,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset,
                        size: NonZeroU64::new(binding_size(buffer, offset)),
                    }),
                });
            }
            for i in 0..MAX_MARKS_TEXTURES {
                let view = views
                    .get(i)
                    .and_then(Option::as_ref)
                    .unwrap_or(&gpu.blank_texture);
                entries.push(wgpu::BindGroupEntry {
                    binding: 6 + i as u32,
                    resource: wgpu::BindingResource::TextureView(view),
                });
            }
            entries.push(wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::Sampler(&gpu.sampler),
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("marks"),
                layout: &gpu.layout,
                entries: &entries,
            });
            state.frame.push(Some((pipeline, bind_group)));
        }
        queue.write_buffer(ring, 0, &bytes);
    }

    /// Draws `draws`, which start at draw `first` of the frame's marks.
    pub(super) fn draw_marks(
        &self,
        draws: &[MarksDraw],
        first: usize,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        let frame = &self.resources().marks.frame;
        for (i, draw) in draws.iter().enumerate() {
            let Some(Some((pipeline, bind_group))) = frame.get(first + i) else {
                continue;
            };
            let Some((instances, vertices)) = draw.draw_range() else {
                continue;
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..vertices, instances);
        }
    }
}

fn build_pipeline(
    device: &wgpu::Device,
    gpu: &MarksGpu,
    format: wgpu::TextureFormat,
    shader: &MarksShader,
    topology: gpui::MarksTopology,
) -> Option<(wgpu::RenderPipeline, Rc<StorageNeeds>)> {
    let (source, needs) = validate(shader)
        .inspect_err(|error| shader.report_compile_error(format!("{error:#}")))
        .ok()?;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("marks"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    Some((
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("marks"),
            layout: Some(&gpu.pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("gpui_marks_vertex"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("gpui_marks_fragment"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: match topology {
                    gpui::MarksTopology::Triangles => wgpu::PrimitiveTopology::TriangleList,
                    gpui::MarksTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
                },
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        }),
        Rc::new(needs),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_program_validates() {
        let (_, storage) = validate(&MarksShader::new(gpui::marks_test_scene::RECTS)).unwrap();
        assert_eq!(storage, vec![(0, 32)]);
    }

    #[test]
    fn programs_take_at_most_256_bytes_of_uniforms() {
        let big = "struct Big { values: array<vec4<f32>, 17> }
@group(0) @binding(1) var<uniform> big: Big;
fn mark_vertex(vertex: u32, instance: u32) -> MarkVertex { return gpui_mark_vertex(big.values[0].xy); }
fn mark_fragment(input: MarkFragment) -> vec4<f32> { return vec4<f32>(1.0); }";
        let error = validate(&MarksShader::new(big)).unwrap_err().to_string();
        assert!(error.contains("272"), "{error}");
    }

    #[test]
    fn programs_may_bind_only_what_gpui_binds() {
        let write = "@group(0) @binding(2) var<storage, read_write> xs: array<f32>;
fn mark_vertex(vertex: u32, instance: u32) -> MarkVertex { xs[0] = 1.0; return gpui_mark_vertex(vec2<f32>(0.0)); }
fn mark_fragment(input: MarkFragment) -> vec4<f32> { return vec4<f32>(1.0); }";
        let error = validate(&MarksShader::new(write)).unwrap_err().to_string();
        assert!(error.contains("@binding(2)"), "{error}");
        let texture = "@group(0) @binding(6) var t: texture_2d<f32>;
fn mark_vertex(vertex: u32, instance: u32) -> MarkVertex { return gpui_mark_vertex(vec2<f32>(0.0)); }
fn mark_fragment(input: MarkFragment) -> vec4<f32> { return textureSample(t, gpui_marks_sampler, input.local.xy); }";
        validate(&MarksShader::new(texture)).unwrap();
    }
}
