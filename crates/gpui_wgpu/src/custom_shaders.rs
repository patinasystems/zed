//! Custom WGSL shaders (`gpui::Shader`): the prelude and the shader's source in
//! one module, validated with Naga before wgpu compiles it.
use anyhow::{Result, anyhow};
use collections::HashMap;
use gpui::{BackdropFilter, Shader, ShaderId};

/// `GpuiShaderParams` in the prelude (`gpui/src/shader_prelude.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
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

/// Validates `shader` (with the prelude) as wgpu would compile it.
pub(crate) fn validate(shader: &Shader) -> Result<String> {
    let source = shader.module_source();
    let module = naga::front::wgsl::parse_str(&source)
        .map_err(|error| anyhow!(error.emit_to_string(&source)))?;
    naga::valid::Validator::new(
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
    Ok(source)
}

/// Pipelines of the custom shaders a renderer has drawn, and what they bind.
pub(crate) struct CustomShaders {
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
    /// The backdrop of a shader that reads none.
    blank: wgpu::TextureView,
    /// `None` for a shader that failed to compile (logged once).
    pipelines: HashMap<ShaderId, Option<wgpu::RenderPipeline>>,
}

impl CustomShaders {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("custom_shader_layout"),
            entries: &[
                uniform(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                uniform(3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("custom_shader_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("custom_shader_backdrop"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let blank = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("custom_shader_blank"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Four bytes whatever the target's format: it is only ever transparent.
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
            blank: blank.create_view(&wgpu::TextureViewDescriptor::default()),
            pipelines: HashMap::default(),
        }
    }

    /// The pipeline for `shader`, compiled on first use.
    fn pipeline(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        shader: &Shader,
    ) -> Option<wgpu::RenderPipeline> {
        let layout = &self.pipeline_layout;
        self.pipelines
            .entry(shader.id)
            .or_insert_with(|| {
                let source = validate(shader)
                    .inspect_err(|error| log::error!("custom shader: {error:#}"))
                    .ok()?;
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("custom_shader"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
                Some(
                    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("custom_shader"),
                        layout: Some(layout),
                        vertex: wgpu::VertexState {
                            module: &module,
                            entry_point: Some("gpui_shader_vertex"),
                            buffers: &[],
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &module,
                            entry_point: Some("gpui_shader_fragment"),
                            targets: &[Some(wgpu::ColorTargetState {
                                format,
                                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleStrip,
                            ..Default::default()
                        },
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState::default(),
                        multiview_mask: None,
                        cache: None,
                    }),
                )
            })
            .clone()
    }

    /// Draws `filter`'s shader into `target` over what is there, with `backdrop`
    /// (the region it read and where it sits) when it reads one.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        shader: &Shader,
        filter: &BackdropFilter,
        uniforms: &[u8],
        backdrop: Option<(&wgpu::TextureView, [f32; 4])>,
        viewport_size: [f32; 2],
    ) {
        let Some(pipeline) = self.pipeline(device, format, shader) else {
            return;
        };
        let (backdrop, region) = backdrop.unwrap_or((&self.blank, [0., 0., 1., 1.]));
        let params = ShaderParams::new(filter, region, viewport_size);
        let buffer = |label, size: usize, bytes: &[u8]| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, bytes);
            buffer
        };
        let params = buffer(
            "custom_shader_params",
            size_of::<ShaderParams>(),
            bytemuck::bytes_of(&params),
        );
        let len = uniforms.len().min(UNIFORM_BYTES) & !3;
        let uniforms = buffer("custom_shader_uniforms", UNIFORM_BYTES, &uniforms[..len]);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("custom_shader"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(backdrop),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        });
        let mut pass = crate::wgpu_renderer::begin_load_pass(encoder, target, "custom_shader");
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..4, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_shaders_validate() {
        use gpui::custom_shader_test_scene as t;
        for source in [t::SOLID, t::INVERT, t::GRADIENT] {
            validate(&Shader::new(source)).unwrap();
        }
    }

    #[test]
    fn shaders_may_bind_only_what_gpui_binds() {
        let extra = "@group(0) @binding(4) var<uniform> extra: vec4<f32>;
fn shade(input: ShaderInput) -> vec4<f32> { return extra; }";
        let error = validate(&Shader::new(extra)).unwrap_err().to_string();
        assert!(error.contains("@binding(4)"), "{error}");
        let texture = "@group(0) @binding(3) var t: texture_2d<f32>;
fn shade(input: ShaderInput) -> vec4<f32> { return textureLoad(t, vec2<i32>(0), 0); }";
        assert!(validate(&Shader::new(texture)).is_err());
        let uniforms = "@group(0) @binding(3) var<uniform> u: vec4<f32>;
fn shade(input: ShaderInput) -> vec4<f32> { return u; }";
        validate(&Shader::new(uniforms)).unwrap();
    }

    #[test]
    fn invalid_shaders_report_where() {
        let error = validate(&Shader::new(
            "fn shade(input: ShaderInput) -> vec4<f32> { return 1; }",
        ))
        .unwrap_err()
        .to_string();
        assert!(error.contains("shade"), "{error}");
    }
}
