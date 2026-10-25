// Backdrop filters (CSS `backdrop-filter`) and compositing groups: copy, blur and
// composite passes over a region read back from the frame or a group's image. A
// standalone module: it binds its own textures and uniforms and reads no instance data.

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

struct Filtered {
    // Straight color and alpha.
    color: vec4<f32>,
    // Of the filter's rounded bounds, faded by its opacity.
    coverage: f32,
}

// The filtered backdrop at `position`.
fn backdrop_filtered(position: vec2<f32>) -> Filtered {
    let uv = (position - composite.region.xy) / composite.region.zw;
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
    let center_to_point = position - (composite.bounds.xy + half_size);
    let radius = backdrop_corner_radius(center_to_point);
    let corner_to_point = abs(center_to_point) - half_size + radius;
    let distance = length(max(vec2<f32>(0.0), corner_to_point))
        + min(0.0, max(corner_to_point.x, corner_to_point.y)) - radius;
    var out: Filtered;
    out.color = vec4<f32>(rgb, color.a);
    out.coverage = saturate(0.5 - distance) * composite.opacity;
    return out;
}

@fragment
fn fs_backdrop_composite(input: CompositeVarying) -> @location(0) vec4<f32> {
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }
    let filtered = backdrop_filtered(input.position.xy);
    // Blended as color * 1 + target * (1 - coverage), keeping the target's alpha:
    // transparent backdrops stay transparent.
    return vec4<f32>(filtered.color.rgb * filtered.color.a * filtered.coverage, filtered.coverage);
}

// The filtered backdrop that starts a group's image: premultiplied, drawn over the
// (transparent) image with its alpha.
@fragment
fn fs_backdrop_group(input: CompositeVarying) -> @location(0) vec4<f32> {
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }
    let filtered = backdrop_filtered(input.position.xy);
    return vec4<f32>(filtered.color.rgb, 1.0) * filtered.color.a * filtered.coverage;
}

// Compositing groups: a group's image (premultiplied, as targets hold colors) is
// color filtered, faded and blended onto the target.

struct GroupParams {
    // What to draw and the content mask: origin and size, device pixels.
    bounds: vec4<f32>,
    content_mask: vec4<f32>,
    // Where the group's image and the copy of what is behind it sit in the target.
    source_region: vec4<f32>,
    backdrop_region: vec4<f32>,
    // Color filter codes and amounts, four to a vector (`gpui::ColorFilter::gpu`).
    filter_kinds: array<vec4<u32>, 2>,
    filter_amounts: array<vec4<f32>, 2>,
    viewport_size: vec2<f32>,
    opacity: f32,
    // `gpui::BlendMode`.
    blend_mode: u32,
    filter_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(1) @binding(0) var<uniform> group_params: GroupParams;
@group(2) @binding(0) var t_group_backdrop: texture_2d<f32>;
@group(2) @binding(1) var s_group_backdrop: sampler;

struct GroupVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) clip_distances: vec4<f32>,
}

@vertex
fn vs_group_composite(@builtin(vertex_index) vertex_id: u32) -> GroupVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let position = unit_vertex * group_params.bounds.zw + group_params.bounds.xy;
    var out: GroupVarying;
    let device = position / group_params.viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    out.position = vec4<f32>(device, 0.0, 1.0);
    let tl = position - group_params.content_mask.xy;
    let br = group_params.content_mask.xy + group_params.content_mask.zw - position;
    out.clip_distances = vec4<f32>(tl.x, br.x, tl.y, br.y);
    return out;
}

fn group_matrix(
    rgb: vec3<f32>,
    r: vec3<f32>,
    g: vec3<f32>,
    b: vec3<f32>,
) -> vec3<f32> {
    return saturate(vec3<f32>(dot(r, rgb), dot(g, rgb), dot(b, rgb)));
}

// One CSS color filter function (Filter Effects 1) on a straight color.
fn group_color_filter(rgb: vec3<f32>, kind: u32, amount: f32) -> vec3<f32> {
    switch kind {
        // brightness
        case 1u: {
            return saturate(rgb * amount);
        }
        // contrast
        case 2u: {
            return saturate((rgb - 0.5) * amount + 0.5);
        }
        // grayscale
        case 3u: {
            let s = 1.0 - amount;
            return group_matrix(
                rgb,
                vec3<f32>(0.2126 + 0.7874 * s, 0.7152 - 0.7152 * s, 0.0722 - 0.0722 * s),
                vec3<f32>(0.2126 - 0.2126 * s, 0.7152 + 0.2848 * s, 0.0722 - 0.0722 * s),
                vec3<f32>(0.2126 - 0.2126 * s, 0.7152 - 0.7152 * s, 0.0722 + 0.9278 * s),
            );
        }
        // hue-rotate, in radians
        case 4u: {
            let c = cos(amount);
            let s = sin(amount);
            return group_matrix(
                rgb,
                vec3<f32>(0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928),
                vec3<f32>(0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283),
                vec3<f32>(0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072),
            );
        }
        // invert
        case 5u: {
            return saturate(mix(rgb, 1.0 - rgb, amount));
        }
        // saturate
        case 6u: {
            let s = amount;
            return group_matrix(
                rgb,
                vec3<f32>(0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s),
                vec3<f32>(0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s),
                vec3<f32>(0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s),
            );
        }
        // sepia
        case 7u: {
            let s = 1.0 - amount;
            return group_matrix(
                rgb,
                vec3<f32>(0.393 + 0.607 * s, 0.769 - 0.769 * s, 0.189 - 0.189 * s),
                vec3<f32>(0.349 - 0.349 * s, 0.686 + 0.314 * s, 0.168 - 0.168 * s),
                vec3<f32>(0.272 - 0.272 * s, 0.534 - 0.534 * s, 0.131 + 0.869 * s),
            );
        }
        default: {
            return rgb;
        }
    }
}

fn group_screen(b: vec3<f32>, s: vec3<f32>) -> vec3<f32> {
    return b + s - b * s;
}

fn group_hard_light(b: vec3<f32>, s: vec3<f32>) -> vec3<f32> {
    return select(group_screen(b, 2.0 * s - 1.0), b * 2.0 * s, s <= vec3<f32>(0.5));
}

fn group_lum(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.3, 0.59, 0.11));
}

fn group_set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    let d = c + (l - group_lum(c));
    let lum = group_lum(d);
    let n = min(d.r, min(d.g, d.b));
    let x = max(d.r, max(d.g, d.b));
    var out = d;
    if (n < 0.0) {
        out = lum + (out - lum) * lum / (lum - n);
    }
    if (x > 1.0) {
        out = lum + (out - lum) * (1.0 - lum) / (x - lum);
    }
    return out;
}

fn group_sat(c: vec3<f32>) -> f32 {
    return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
}

fn group_set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let n = min(c.r, min(c.g, c.b));
    let x = max(c.r, max(c.g, c.b));
    if (x > n) {
        return (c - n) * s / (x - n);
    }
    return vec3<f32>(0.0);
}

// The blend function B(backdrop, source) of a `gpui::BlendMode` (Compositing and
// Blending 1), on straight colors.
fn group_blend(b: vec3<f32>, s: vec3<f32>, mode: u32) -> vec3<f32> {
    switch mode {
        case 1u: {
            return b * s;
        }
        case 2u: {
            return group_screen(b, s);
        }
        case 3u: {
            return group_hard_light(s, b);
        }
        case 4u: {
            return min(b, s);
        }
        case 5u: {
            return max(b, s);
        }
        case 6u: {
            let dodge = min(vec3<f32>(1.0), b / max(1.0 - s, vec3<f32>(1e-6)));
            return select(select(dodge, vec3<f32>(1.0), s >= vec3<f32>(1.0)), vec3<f32>(0.0), b <= vec3<f32>(0.0));
        }
        case 7u: {
            let burn = 1.0 - min(vec3<f32>(1.0), (1.0 - b) / max(s, vec3<f32>(1e-6)));
            return select(select(burn, vec3<f32>(0.0), s <= vec3<f32>(0.0)), vec3<f32>(1.0), b >= vec3<f32>(1.0));
        }
        case 8u: {
            return group_hard_light(b, s);
        }
        case 9u: {
            let d = select(sqrt(b), ((16.0 * b - 12.0) * b + 4.0) * b, b <= vec3<f32>(0.25));
            return select(b + (2.0 * s - 1.0) * (d - b), b - (1.0 - 2.0 * s) * b * (1.0 - b), s <= vec3<f32>(0.5));
        }
        case 10u: {
            return abs(b - s);
        }
        case 11u: {
            return b + s - 2.0 * b * s;
        }
        case 12u: {
            return group_set_lum(group_set_sat(s, group_sat(b)), group_lum(b));
        }
        case 13u: {
            return group_set_lum(group_set_sat(b, group_sat(s)), group_lum(b));
        }
        case 14u: {
            return group_set_lum(s, group_lum(b));
        }
        case 15u: {
            return group_set_lum(b, group_lum(s));
        }
        default: {
            return s;
        }
    }
}

// Normal and most modes are drawn over the target (premultiplied source over),
// plus-lighter adds, and plus-darker replaces the target with the whole result.
@fragment
fn fs_group_composite(input: GroupVarying) -> @location(0) vec4<f32> {
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        discard;
    }
    let position = input.position.xy;
    let source_uv = (position - group_params.source_region.xy) / group_params.source_region.zw;
    var color = textureSampleLevel(t_source, s_source, source_uv, 0.0);
    if (group_params.filter_count > 0u) {
        var rgb = color.rgb / max(color.a, 1e-6);
        for (var i = 0u; i < group_params.filter_count; i += 1u) {
            rgb = group_color_filter(
                rgb,
                group_params.filter_kinds[i / 4u][i % 4u],
                group_params.filter_amounts[i / 4u][i % 4u],
            );
        }
        color = vec4<f32>(rgb * color.a, color.a);
    }
    color *= group_params.opacity;
    let mode = group_params.blend_mode;
    if (mode == 0u || mode == 16u) {
        return color;
    }
    let backdrop_uv = (position - group_params.backdrop_region.xy) / group_params.backdrop_region.zw;
    let backdrop = textureSampleLevel(t_group_backdrop, s_group_backdrop, backdrop_uv, 0.0);
    if (mode == 17u) {
        let alpha = min(1.0, color.a + backdrop.a);
        let rgb = max(vec3<f32>(0.0), alpha - (backdrop.a - backdrop.rgb) - (color.a - color.rgb));
        return vec4<f32>(rgb, alpha);
    }
    let source = color.rgb / max(color.a, 1e-6);
    let behind = backdrop.rgb / max(backdrop.a, 1e-6);
    let mixed = (1.0 - backdrop.a) * source + backdrop.a * saturate(group_blend(behind, source, mode));
    return vec4<f32>(mixed * color.a, color.a);
}
