// Backdrop filters (CSS `backdrop-filter`): copy, blur and composite passes over a
// region read back from the frame. A standalone module: it binds its own texture and
// uniforms and reads no instance data.

@group(0) @binding(0) var t_source: texture_2d<f32>;
@group(0) @binding(1) var s_source: sampler;

struct FullscreenVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// One triangle that covers the target; `uv` runs 0..1 across it.
@vertex
fn vs_backdrop_fullscreen(@builtin(vertex_index) vertex_id: u32) -> FullscreenVarying {
    let uv = vec2<f32>(f32((vertex_id << 1u) & 2u), f32(vertex_id & 2u));
    var out: FullscreenVarying;
    out.position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_backdrop_copy(input: FullscreenVarying) -> @location(0) vec4<f32> {
    return textureSampleLevel(t_source, s_source, input.uv, 0.0);
}

struct BlurParams {
    // One texel along the blur axis, in uv.
    direction: vec2<f32>,
    // In texels.
    sigma: f32,
    // Taps on each side.
    radius: i32,
}

@group(1) @binding(0) var<uniform> blur: BlurParams;

// One axis of a gaussian blur.
@fragment
fn fs_backdrop_blur(input: FullscreenVarying) -> @location(0) vec4<f32> {
    var sum = vec4<f32>(0.0);
    var total = 0.0;
    for (var i = -blur.radius; i <= blur.radius; i += 1) {
        let weight = exp(-0.5 * f32(i * i) / (blur.sigma * blur.sigma));
        sum += textureSampleLevel(t_source, s_source, input.uv + blur.direction * f32(i), 0.0) * weight;
        total += weight;
    }
    return sum / total;
}

struct CompositeParams {
    // The filter's bounds and content mask, origin and size, device pixels.
    bounds: vec4<f32>,
    content_mask: vec4<f32>,
    // Top left, top right, bottom right, bottom left.
    corner_radii: vec4<f32>,
    // Where the filtered texture sits in the target: origin and size.
    region: vec4<f32>,
    viewport_size: vec2<f32>,
    saturate: f32,
    brightness: f32,
    contrast: f32,
    opacity: f32,
    pad1: f32,
    pad2: f32,
}

@group(1) @binding(0) var<uniform> composite: CompositeParams;

struct CompositeVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) clip_distances: vec4<f32>,
}

@vertex
fn vs_backdrop_composite(@builtin(vertex_index) vertex_id: u32) -> CompositeVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let position = unit_vertex * composite.bounds.zw + composite.bounds.xy;
    var out: CompositeVarying;
    let device = position / composite.viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    out.position = vec4<f32>(device, 0.0, 1.0);
    let tl = position - composite.content_mask.xy;
    let br = composite.content_mask.xy + composite.content_mask.zw - position;
    out.clip_distances = vec4<f32>(tl.x, br.x, tl.y, br.y);
    return out;
}

fn backdrop_corner_radius(center_to_point: vec2<f32>) -> f32 {
    if (center_to_point.x < 0.0) {
        if (center_to_point.y < 0.0) {
            return composite.corner_radii.x;
        }
        return composite.corner_radii.w;
    }
    if (center_to_point.y < 0.0) {
        return composite.corner_radii.y;
    }
    return composite.corner_radii.z;
}

@fragment
fn fs_backdrop_composite(input: CompositeVarying) -> @location(0) vec4<f32> {
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }
    let uv = (input.position.xy - composite.region.xy) / composite.region.zw;
    let color = textureSampleLevel(t_source, s_source, uv, 0.0);
    // Filter straight colors, then weight them by the backdrop's alpha again, so
    // transparent texels stay empty.
    var rgb = select(vec3<f32>(0.0), color.rgb / max(color.a, 1e-6), color.a > 0.0);
    // CSS saturate(), brightness(), contrast() (Filter Effects 1), in that order.
    let s = composite.saturate;
    let saturate_matrix = mat3x3<f32>(
        vec3<f32>(0.213 + 0.787 * s, 0.213 - 0.213 * s, 0.213 - 0.213 * s),
        vec3<f32>(0.715 - 0.715 * s, 0.715 + 0.285 * s, 0.715 - 0.715 * s),
        vec3<f32>(0.072 - 0.072 * s, 0.072 - 0.072 * s, 0.072 + 0.928 * s),
    );
    rgb = saturate(saturate_matrix * rgb);
    rgb = saturate(rgb * composite.brightness);
    rgb = saturate((rgb - 0.5) * composite.contrast + 0.5);
    // Rounded-rectangle coverage.
    let half_size = composite.bounds.zw / 2.0;
    let center_to_point = input.position.xy - (composite.bounds.xy + half_size);
    let radius = backdrop_corner_radius(center_to_point);
    let corner_to_point = abs(center_to_point) - half_size + radius;
    let distance = length(max(vec2<f32>(0.0), corner_to_point))
        + min(0.0, max(corner_to_point.x, corner_to_point.y)) - radius;
    let coverage = saturate(0.5 - distance) * composite.opacity;
    // Blended as color * 1 + target * (1 - coverage), keeping the target's alpha:
    // transparent backdrops stay transparent.
    return vec4<f32>(rgb * color.a * coverage, coverage);
}
