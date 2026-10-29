//! Marks (`gpui::MarksShader`) on Metal: the prelude and the program are translated
//! to MSL with Naga and drawn inside the frame's open render encoder, over the
//! caller's `metal::Buffer`s.
use anyhow::{Context as _, Result, anyhow};
use gpui::{
    MARKS_BUFFER_OFFSET_ALIGNMENT, MARKS_UNIFORM_BYTES, MAX_MARKS_BUFFERS, MarksDraw,
    MarksResource, MarksShader,
};
use naga::back::msl;
use std::collections::BTreeMap;

/// Buffer slots: the prelude's uniforms, the program's, its storage buffers, and
/// the sizes of buffers that hold runtime-sized arrays (Naga's `_mslBufferSizes`).
const MARKS_UNIFORMS: u8 = 0;
const PROGRAM_UNIFORMS: u8 = 1;
const FIRST_STORAGE: u8 = 2;
const SIZES: u8 = 6;
const FIRST_TEXTURE: u8 = 0;
const SAMPLER: u8 = 0;

/// A compiled marks program.
pub(crate) struct MarksPipeline {
    pub state: metal::RenderPipelineState,
    /// For each `_mslBufferSizes` member, in order, the storage buffer whose
    /// bound length it carries.
    sized_buffers: Vec<usize>,
    /// Every storage buffer the program declares, and the fewest bytes it reads
    /// (a fixed-size type's size, or one element of a runtime-sized array).
    storage: Vec<(usize, u64)>,
}

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

/// Translates `shader` (with the prelude) to MSL. Returns the source, the MSL
/// names of the vertex and fragment entry points, and the sized buffers.
pub(crate) fn translate(
    shader: &MarksShader,
) -> Result<(String, String, String, Vec<usize>, Vec<(usize, u64)>)> {
    let source = shader.module_source();
    let module = naga::front::wgsl::parse_str(&source)
        .map_err(|error| anyhow!(error.emit_to_string(&source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|error| anyhow!(error.emit_to_string(&source)))?;
    let mut sized_buffers = Vec::new();
    let mut storage = Vec::new();
    for (_, variable) in module.global_variables.iter() {
        if let Some(binding) = &variable.binding {
            let kind = resource_kind(&module, variable);
            let size = module.types[variable.ty]
                .inner
                .try_size(module.to_ctx())
                .ok_or_else(|| anyhow!("@binding({}) is too large", binding.binding))?;
            if binding.binding == 1 && size as usize > MARKS_UNIFORM_BYTES {
                return Err(anyhow!(
                    "@binding(1) declares {size} bytes of uniforms; marks take at most {MARKS_UNIFORM_BYTES}"
                ));
            }
            if kind == MarksResource::StorageRead {
                storage.push(((binding.binding - 2) as usize, u64::from(size)));
            }
            if !gpui::marks_binding_allowed(
                binding.group,
                binding.binding,
                variable.name.as_deref(),
                kind,
            ) {
                return Err(anyhow!(gpui::marks_binding_error(
                    binding.group,
                    binding.binding
                )));
            }
        }
        // Naga's `_mslBufferSizes` has a member for every global holding a
        // runtime-sized array, in declaration order; only storage buffers can.
        if module.types[variable.ty]
            .inner
            .needs_host_buffer_byte_size(&module.types)
        {
            let binding = variable
                .binding
                .as_ref()
                .context("a runtime-sized array outside a binding")?;
            sized_buffers.push((binding.binding - 2) as usize);
        }
    }
    let binding = |binding| naga::ResourceBinding { group: 0, binding };
    let buffer = |slot| msl::BindTarget {
        buffer: Some(slot),
        ..Default::default()
    };
    let mut resources = BTreeMap::from([
        (binding(0), buffer(MARKS_UNIFORMS)),
        (binding(1), buffer(PROGRAM_UNIFORMS)),
        (
            binding(8),
            msl::BindTarget {
                sampler: Some(msl::BindSamplerTarget::Resource(SAMPLER)),
                ..Default::default()
            },
        ),
    ]);
    for i in 0..MAX_MARKS_BUFFERS as u8 {
        resources.insert(binding(2 + u32::from(i)), buffer(FIRST_STORAGE + i));
    }
    for i in 0..gpui::MAX_MARKS_TEXTURES as u8 {
        resources.insert(
            binding(6 + u32::from(i)),
            msl::BindTarget {
                texture: Some(FIRST_TEXTURE + i),
                ..Default::default()
            },
        );
    }
    let resources = msl::EntryPointResources {
        resources,
        sizes_buffer: Some(SIZES),
        ..Default::default()
    };
    // Reads past a buffer return zero, as on wgpu (robust buffer access).
    let checked = naga::proc::BoundsCheckPolicies {
        index: naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite,
        buffer: naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite,
        ..Default::default()
    };
    let options = msl::Options {
        lang_version: (2, 4),
        per_entry_point_map: BTreeMap::from([
            ("gpui_marks_vertex".to_string(), resources.clone()),
            ("gpui_marks_fragment".to_string(), resources),
        ]),
        fake_missing_bindings: false,
        bounds_check_policies: checked,
        ..Default::default()
    };
    let (msl_source, translation) =
        msl::write_string(&module, &info, &options, &msl::PipelineOptions::default())
            .context("translating the marks program to MSL")?;
    let name = |entry: &str| -> Result<String> {
        let index = module
            .entry_points
            .iter()
            .position(|point| point.name == entry)
            .with_context(|| format!("no `{entry}` entry point"))?;
        translation.entry_point_names[index]
            .clone()
            .map_err(|error| anyhow!("{entry}: {error}"))
    };
    Ok((
        msl_source,
        name("gpui_marks_vertex")?,
        name("gpui_marks_fragment")?,
        sized_buffers,
        storage,
    ))
}

/// A pipeline for `shader` that draws premultiplied color over the target.
pub(crate) fn build_pipeline(
    device: &metal::DeviceRef,
    shader: &MarksShader,
    pixel_format: metal::MTLPixelFormat,
) -> Result<MarksPipeline> {
    let (source, vertex_name, fragment_name, sized_buffers, storage) = translate(shader)?;
    let library = device
        .new_library_with_source(&source, &metal::CompileOptions::new())
        .map_err(|error| anyhow!("compiling the marks program's MSL: {error}"))?;
    let vertex_fn = library
        .get_function(&vertex_name, None)
        .map_err(|error| anyhow!(error))?;
    let fragment_fn = library
        .get_function(&fragment_name, None)
        .map_err(|error| anyhow!(error))?;
    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label("marks");
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    let state = device
        .new_render_pipeline_state(&descriptor)
        .map_err(|error| anyhow!("creating the marks pipeline: {error}"))?;
    Ok(MarksPipeline {
        state,
        sized_buffers,
        storage,
    })
}

/// The draw's storage buffers, if every one is a `metal::Buffer` at an aligned
/// offset inside it.
fn metal_buffers(draw: &MarksDraw) -> Option<Vec<(&metal::Buffer, u64)>> {
    draw.bindings
        .buffers
        .iter()
        .map(|binding| {
            let buffer = binding.downcast_ref::<metal::Buffer>()?;
            (binding.offset % MARKS_BUFFER_OFFSET_ALIGNMENT == 0
                && binding.offset < buffer.length())
            .then_some((buffer, binding.offset))
        })
        .collect()
}

/// Encodes `draw` into `encoder` (the frame's open pass). `textures` are its
/// textures, resolved; `blank` stands in for missing ones. Returns false when it
/// skipped the draw: a buffer it cannot bind, or a storage binding the program
/// declares that the draw leaves out or binds with fewer bytes than its type.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    encoder: &metal::RenderCommandEncoderRef,
    pipeline: &MarksPipeline,
    draw: &MarksDraw,
    viewport: [f32; 2],
    textures: &[Option<metal::Texture>],
    blank: &metal::TextureRef,
    sampler: &metal::SamplerStateRef,
    bound_pipeline: &mut Option<*const metal::RenderPipelineStateRef>,
) -> bool {
    let Some((instances, vertices)) = draw.draw_range() else {
        return true;
    };
    let Some(buffers) = metal_buffers(draw) else {
        draw.shader
            .report_skipped_draw("a buffer is not a metal::Buffer or its offset is not 256-aligned");
        return false;
    };
    for &(index, min) in &pipeline.storage {
        let fits = buffers
            .get(index)
            .is_some_and(|(buffer, offset)| buffer.length() - offset >= min);
        if !fits {
            draw.shader.report_skipped_draw(format!(
                "storage buffer {index} is missing or under the {min} bytes its program reads"
            ));
            return false;
        }
    }
    let mut sizes = [0u32; MAX_MARKS_BUFFERS];
    for (member, &index) in pipeline.sized_buffers.iter().enumerate() {
        let Some((buffer, offset)) = buffers.get(index) else {
            draw.shader.report_skipped_draw(format!(
                "the program reads buffer {index}, which the draw does not bind"
            ));
            return false;
        };
        sizes[member] = (buffer.length() - offset).min(u64::from(u32::MAX)) as u32;
    }
    let state: *const metal::RenderPipelineStateRef = &*pipeline.state;
    if *bound_pipeline != Some(state) {
        encoder.set_render_pipeline_state(&pipeline.state);
        *bound_pipeline = Some(state);
    }
    let uniforms = draw.uniforms(viewport);
    let block = draw.uniform_block();
    for stage in [Stage::Vertex, Stage::Fragment] {
        stage.bytes(encoder, MARKS_UNIFORMS, uniforms.as_bytes());
        stage.bytes(encoder, PROGRAM_UNIFORMS, &block[..MARKS_UNIFORM_BYTES]);
        for (i, (buffer, offset)) in buffers.iter().enumerate() {
            stage.buffer(encoder, FIRST_STORAGE + i as u8, buffer, *offset);
        }
        if !pipeline.sized_buffers.is_empty() {
            let len = pipeline.sized_buffers.len() * 4;
            stage.bytes(encoder, SIZES, &bytemuck_u32(&sizes)[..len]);
        }
    }
    for i in 0..gpui::MAX_MARKS_TEXTURES {
        let texture = textures.get(i).and_then(Option::as_ref);
        let texture: &metal::TextureRef = texture.map_or(blank, |t| t);
        encoder.set_vertex_texture(u64::from(FIRST_TEXTURE) + i as u64, Some(texture));
        encoder.set_fragment_texture(u64::from(FIRST_TEXTURE) + i as u64, Some(texture));
    }
    encoder.set_vertex_sampler_state(u64::from(SAMPLER), Some(sampler));
    encoder.set_fragment_sampler_state(u64::from(SAMPLER), Some(sampler));
    let primitive = match draw.paint.topology {
        gpui::MarksTopology::Triangles => metal::MTLPrimitiveType::Triangle,
        gpui::MarksTopology::TriangleStrip => metal::MTLPrimitiveType::TriangleStrip,
    };
    encoder.draw_primitives_instanced_base_instance(
        primitive,
        0,
        u64::from(vertices),
        u64::from(instances.end - instances.start),
        u64::from(instances.start),
    );
    true
}

fn bytemuck_u32(values: &[u32; MAX_MARKS_BUFFERS]) -> [u8; MAX_MARKS_BUFFERS * 4] {
    let mut out = [0u8; MAX_MARKS_BUFFERS * 4];
    for (i, v) in values.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
    }
    out
}

#[derive(Clone, Copy)]
enum Stage {
    Vertex,
    Fragment,
}

impl Stage {
    fn bytes(self, encoder: &metal::RenderCommandEncoderRef, slot: u8, bytes: &[u8]) {
        let (len, ptr) = (bytes.len() as u64, bytes.as_ptr().cast());
        match self {
            Stage::Vertex => encoder.set_vertex_bytes(u64::from(slot), len, ptr),
            Stage::Fragment => encoder.set_fragment_bytes(u64::from(slot), len, ptr),
        }
    }

    fn buffer(
        self,
        encoder: &metal::RenderCommandEncoderRef,
        slot: u8,
        buffer: &metal::BufferRef,
        offset: u64,
    ) {
        match self {
            Stage::Vertex => encoder.set_vertex_buffer(u64::from(slot), Some(buffer), offset),
            Stage::Fragment => encoder.set_fragment_buffer(u64::from(slot), Some(buffer), offset),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_program_translates_to_msl() {
        let (msl, vertex, fragment, sized, storage) =
            translate(&MarksShader::new(gpui::marks_test_scene::RECTS)).unwrap();
        assert!(msl.contains(&vertex) && msl.contains(&fragment));
        assert_eq!(sized, vec![0]);
        // One `Rect` (32 bytes) at least.
        assert_eq!(storage, vec![(0, 32)]);
    }

    #[test]
    fn programs_may_bind_only_what_gpui_binds() {
        let write = "@group(0) @binding(2) var<storage, read_write> xs: array<f32>;
fn mark_vertex(vertex: u32, instance: u32) -> MarkVertex { xs[0] = 1.0; return gpui_mark_vertex(vec2<f32>(0.0)); }
fn mark_fragment(input: MarkFragment) -> vec4<f32> { return vec4<f32>(1.0); }";
        let error = match translate(&MarksShader::new(write)) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a writable buffer translated"),
        };
        assert!(error.contains("@binding(2)"), "{error}");
        let extra = "@group(1) @binding(0) var<uniform> u: vec4<f32>;
fn mark_vertex(vertex: u32, instance: u32) -> MarkVertex { return gpui_mark_vertex(u.xy); }
fn mark_fragment(input: MarkFragment) -> vec4<f32> { return vec4<f32>(1.0); }";
        assert!(translate(&MarksShader::new(extra)).is_err());
        let big = "@group(0) @binding(1) var<uniform> u: array<vec4<f32>, 17>;
fn mark_vertex(vertex: u32, instance: u32) -> MarkVertex { return gpui_mark_vertex(u[16].xy); }
fn mark_fragment(input: MarkFragment) -> vec4<f32> { return vec4<f32>(1.0); }";
        let error = match translate(&MarksShader::new(big)) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("272 bytes of uniforms translated"),
        };
        assert!(error.contains("272"), "{error}");
    }

    #[test]
    fn invalid_programs_report_where() {
        let error = match translate(&MarksShader::new(
            "fn mark_vertex(vertex: u32, instance: u32) -> MarkVertex { return 1; }",
        )) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("an invalid program translated"),
        };
        assert!(error.contains("mark_vertex"), "{error}");
    }
}
