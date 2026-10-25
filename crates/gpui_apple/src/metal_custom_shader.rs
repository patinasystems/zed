//! Custom WGSL shaders (`gpui::Shader`) on Metal: the prelude and the shader's
//! source are translated to MSL with Naga, then compiled into a pipeline.
use anyhow::{Context as _, Result, anyhow};
use gpui::{BackdropFilter, Shader};
use naga::back::msl;
use std::collections::BTreeMap;

/// `GpuiShaderParams` in the prelude (`gpui/src/shader_prelude.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ShaderParams {
    pub bounds: [f32; 4],
    pub content_mask: [f32; 4],
    pub corner_radii: [f32; 4],
    pub backdrop_region: [f32; 4],
    pub viewport_size: [f32; 2],
    pub opacity: f32,
    pub scale_factor: f32,
}

impl ShaderParams {
    pub fn new(
        filter: &BackdropFilter,
        backdrop_region: [f32; 4],
        viewport_size: [f32; 2],
    ) -> Self {
        let b = filter.bounds;
        let m = filter.content_mask.bounds;
        let r = filter.corner_radii;
        Self {
            bounds: [b.origin.x.0, b.origin.y.0, b.size.width.0, b.size.height.0],
            content_mask: [m.origin.x.0, m.origin.y.0, m.size.width.0, m.size.height.0],
            corner_radii: [
                r.top_left.0,
                r.top_right.0,
                r.bottom_right.0,
                r.bottom_left.0,
            ],
            backdrop_region,
            viewport_size,
            opacity: filter.opacity,
            scale_factor: filter.shader.scale_factor,
        }
    }
}

/// The size of the uniform block every draw binds: the most `paint_shader` takes,
/// so a shader that declares more than it was given reads zeros.
pub(crate) const UNIFORM_BYTES: usize = 4096;

/// Buffer and texture slots of the prelude's bindings and the uniforms.
const PARAMS_BUFFER: u8 = 0;
const UNIFORMS_BUFFER: u8 = 1;
const BACKDROP_TEXTURE: u8 = 0;
const BACKDROP_SAMPLER: u8 = 0;

/// Translates `shader` (with the prelude) to MSL. Returns the source and the MSL
/// names of the vertex and fragment entry points.
pub(crate) fn translate(shader: &Shader) -> Result<(String, String, String)> {
    let source = shader.module_source();
    let module = naga::front::wgsl::parse_str(&source)
        .map_err(|error| anyhow!(error.emit_to_string(&source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|error| anyhow!(error.emit_to_string(&source)))?;
    for (_, variable) in module.global_variables.iter() {
        if let Some(binding) = &variable.binding
            && !gpui::shader_binding_allowed(
                binding.group,
                binding.binding,
                variable.name.as_deref(),
                variable.space == naga::AddressSpace::Uniform,
            )
        {
            return Err(anyhow!(
                "@group({}) @binding({}) is not bound by GPUI: uniforms are a `var<uniform>` at @group(0) @binding(3)",
                binding.group,
                binding.binding
            ));
        }
    }
    let binding = |binding| naga::ResourceBinding { group: 0, binding };
    let resources = msl::EntryPointResources {
        resources: BTreeMap::from([
            (
                binding(0),
                msl::BindTarget {
                    buffer: Some(PARAMS_BUFFER),
                    ..Default::default()
                },
            ),
            (
                binding(1),
                msl::BindTarget {
                    texture: Some(BACKDROP_TEXTURE),
                    ..Default::default()
                },
            ),
            (
                binding(2),
                msl::BindTarget {
                    sampler: Some(msl::BindSamplerTarget::Resource(BACKDROP_SAMPLER)),
                    ..Default::default()
                },
            ),
            (
                binding(3),
                msl::BindTarget {
                    buffer: Some(UNIFORMS_BUFFER),
                    ..Default::default()
                },
            ),
        ]),
        ..Default::default()
    };
    let options = msl::Options {
        lang_version: (2, 4),
        per_entry_point_map: BTreeMap::from([
            ("gpui_shader_vertex".to_string(), resources.clone()),
            ("gpui_shader_fragment".to_string(), resources),
        ]),
        fake_missing_bindings: false,
        ..Default::default()
    };
    let (msl_source, translation) =
        msl::write_string(&module, &info, &options, &msl::PipelineOptions::default())
            .context("translating the shader to MSL")?;
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
        name("gpui_shader_vertex")?,
        name("gpui_shader_fragment")?,
    ))
}

/// A pipeline for `shader` that draws premultiplied color over the target.
pub(crate) fn build_pipeline(
    device: &metal::DeviceRef,
    shader: &Shader,
    pixel_format: metal::MTLPixelFormat,
) -> Result<metal::RenderPipelineState> {
    let (source, vertex_name, fragment_name) = translate(shader)?;
    let library = device
        .new_library_with_source(&source, &metal::CompileOptions::new())
        .map_err(|error| anyhow!("compiling the shader's MSL: {error}"))?;
    let vertex_fn = library
        .get_function(&vertex_name, None)
        .map_err(|error| anyhow!(error))?;
    let fragment_fn = library
        .get_function(&fragment_name, None)
        .map_err(|error| anyhow!(error))?;
    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label("custom_shader");
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
    device
        .new_render_pipeline_state(&descriptor)
        .map_err(|error| anyhow!("creating the shader's pipeline: {error}"))
}

/// Draws `filter`'s shader with `backdrop` (the region it read, or a blank
/// texture) into the encoder's target.
pub(crate) fn draw(
    encoder: &metal::RenderCommandEncoderRef,
    pipeline: &metal::RenderPipelineStateRef,
    sampler: &metal::SamplerStateRef,
    params: &ShaderParams,
    uniforms: &[u8],
    backdrop: &metal::TextureRef,
) {
    let mut block = [0u8; UNIFORM_BYTES];
    let len = uniforms.len().min(UNIFORM_BYTES);
    block[..len].copy_from_slice(&uniforms[..len]);
    let params_size = std::mem::size_of::<ShaderParams>() as u64;
    let params_ptr = params as *const ShaderParams as *const _;
    encoder.set_render_pipeline_state(pipeline);
    encoder.set_vertex_bytes(PARAMS_BUFFER as u64, params_size, params_ptr);
    encoder.set_fragment_bytes(PARAMS_BUFFER as u64, params_size, params_ptr);
    encoder.set_vertex_bytes(
        UNIFORMS_BUFFER as u64,
        block.len() as u64,
        block.as_ptr() as *const _,
    );
    encoder.set_fragment_bytes(
        UNIFORMS_BUFFER as u64,
        block.len() as u64,
        block.as_ptr() as *const _,
    );
    encoder.set_fragment_texture(BACKDROP_TEXTURE as u64, Some(backdrop));
    encoder.set_fragment_sampler_state(BACKDROP_SAMPLER as u64, Some(sampler));
    encoder.draw_primitives(metal::MTLPrimitiveType::TriangleStrip, 0, 4);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_shaders_translate_to_msl() {
        use gpui::custom_shader_test_scene as t;
        for source in [t::SOLID, t::INVERT, t::GRADIENT] {
            let (msl, vertex, fragment) = translate(&Shader::new(source)).unwrap();
            assert!(msl.contains(&vertex) && msl.contains(&fragment));
        }
    }

    #[test]
    fn shaders_may_bind_only_what_gpui_binds() {
        let extra = "@group(0) @binding(4) var<uniform> extra: vec4<f32>;
fn shade(input: ShaderInput) -> vec4<f32> { return extra; }";
        let error = translate(&Shader::new(extra)).unwrap_err().to_string();
        assert!(error.contains("@binding(4)"), "{error}");
    }

    #[test]
    fn invalid_shaders_report_where() {
        let error = translate(&Shader::new(
            "fn shade(input: ShaderInput) -> vec4<f32> { return 1; }",
        ))
        .unwrap_err()
        .to_string();
        assert!(error.contains("shade"), "{error}");
    }
}
