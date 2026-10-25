// The prelude of every `gpui::Shader`: its bindings, `ShaderInput`, `backdrop(uv)`
// and the entry points that call the shader's `shade`.

struct GpuiShaderParams {
    // The element's box: x, y, width, height in device pixels.
    bounds: vec4<f32>,
    // The content mask, as a box in device pixels.
    content_mask: vec4<f32>,
    // Top left, top right, bottom right, bottom left, in device pixels.
    corner_radii: vec4<f32>,
    // Where the backdrop texture sits in the target, in device pixels.
    backdrop_region: vec4<f32>,
    viewport_size: vec2<f32>,
    opacity: f32,
    // Device pixels per logical pixel.
    scale_factor: f32,
}

@group(0) @binding(0) var<uniform> gpui_shader: GpuiShaderParams;
@group(0) @binding(1) var gpui_backdrop_texture: texture_2d<f32>;
@group(0) @binding(2) var gpui_backdrop_sampler: sampler;

// What `shade` gets for each pixel.
struct ShaderInput {
    // The pixel's center in the element, in logical pixels from its top left.
    position: vec2<f32>,
    // `position / size`: 0 to 1 across the element.
    uv: vec2<f32>,
    // The element's size in logical pixels.
    size: vec2<f32>,
}

// What is drawn behind the element at `uv` (0 to 1 across the element), with
// straight alpha. Transparent black unless the shader reads the backdrop.
fn backdrop(uv: vec2<f32>) -> vec4<f32> {
    let position = gpui_shader.bounds.xy + uv * gpui_shader.bounds.zw;
    let region = gpui_shader.backdrop_region;
    let coord = (position - region.xy) / region.zw;
    let color = textureSampleLevel(gpui_backdrop_texture, gpui_backdrop_sampler, coord, 0.0);
    if color.a <= 0.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(color.rgb / color.a, color.a);
}

struct GpuiShaderVarying {
    @builtin(position) position: vec4<f32>,
}

// Four vertices as a triangle strip over the element's box.
@vertex
fn gpui_shader_vertex(@builtin(vertex_index) index: u32) -> GpuiShaderVarying {
    let corner = vec2<f32>(f32(index & 1u), f32((index >> 1u) & 1u));
    let position = gpui_shader.bounds.xy + corner * gpui_shader.bounds.zw;
    let ndc = position / gpui_shader.viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    var out: GpuiShaderVarying;
    out.position = vec4<f32>(ndc, 0.0, 1.0);
    return out;
}

// Coverage of `position` by the element's rounded box (antialiased edges).
fn gpui_shader_coverage(position: vec2<f32>) -> f32 {
    let bounds = gpui_shader.bounds;
    let half_size = bounds.zw * 0.5;
    let center_to_point = position - (bounds.xy + half_size);
    let radii = gpui_shader.corner_radii;
    var radius: f32;
    if center_to_point.y < 0.0 {
        radius = select(radii.x, radii.y, center_to_point.x >= 0.0);
    } else {
        radius = select(radii.w, radii.z, center_to_point.x >= 0.0);
    }
    let corner_to_point = abs(center_to_point) - half_size + vec2<f32>(radius);
    let distance = length(max(corner_to_point, vec2<f32>(0.0)))
        + min(max(corner_to_point.x, corner_to_point.y), 0.0)
        - radius;
    return clamp(0.5 - distance, 0.0, 1.0);
}

@fragment
fn gpui_shader_fragment(vertex: GpuiShaderVarying) -> @location(0) vec4<f32> {
    let position = vertex.position.xy;
    let mask = gpui_shader.content_mask;
    if position.x < mask.x || position.y < mask.y
        || position.x >= mask.x + mask.z || position.y >= mask.y + mask.w {
        discard;
    }
    let bounds = gpui_shader.bounds;
    let local = position - bounds.xy;
    let scale = gpui_shader.scale_factor;
    let color = shade(ShaderInput(local / scale, local / bounds.zw, bounds.zw / scale));
    let alpha = clamp(color.a, 0.0, 1.0) * gpui_shader_coverage(position) * gpui_shader.opacity;
    // Premultiplied, as renderers blend it.
    return vec4<f32>(clamp(color.rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * alpha, alpha);
}
