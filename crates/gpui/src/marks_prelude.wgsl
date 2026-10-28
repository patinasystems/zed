// The prelude of every `gpui::MarksShader`: the bindings GPUI owns, `MarkVertex`,
// `MarkFragment`, helpers, and the entry points that call the program's
// `mark_vertex` and `mark_fragment`. Layouts match `MarksUniforms` in gpui/src/marks.rs.

// A rounded-rectangle clip: device px map to its space by `m` (x' = m.x x +
// m.z y + t.x, y' = m.y x + m.w y + t.y), where it is centered on `half_size`;
// a negative half size is an empty clip.
struct GpuiMarksClip {
    m: vec4<f32>,
    t: vec2<f32>,
    half_size: vec2<f32>,
    radii: vec4<f32>,
}

struct GpuiMarks {
    // The buffer's space to device px, row-major: x' = a x + b y + tx, y' = c x + d y + ty.
    transform: vec4<f32>,
    translation: vec2<f32>,
    viewport_size: vec2<f32>,
    // Layer 1, in the buffer's space, the same way.
    layer: vec4<f32>,
    layer_translation: vec2<f32>,
    opacity: f32,
    clip_count: u32,
    // x, y, width, height in device px.
    content_mask: vec4<f32>,
    clips: array<GpuiMarksClip, 4>,
}

@group(0) @binding(0) var<uniform> gpui_marks: GpuiMarks;
// Linear, clamped to the edge.
@group(0) @binding(8) var gpui_marks_sampler: sampler;

// `MarkVertex.layer` bits.
const GPUI_MARK_LAYER_1: u32 = 1u;
const GPUI_MARK_HIDDEN: u32 = 2u;

// What `mark_vertex` returns for one vertex.
struct MarkVertex {
    // In the buffer's space (before the draw's transformations).
    position: vec2<f32>,
    // Added after the transformations, in device px: sizes that do not zoom
    // (a 3 px point, a hairline).
    offset_px: vec2<f32>,
    // `GPUI_MARK_LAYER_1`: moved by the layer transformation. `GPUI_MARK_HIDDEN`:
    // not drawn (set it on every vertex of the instance).
    layer: u32,
    // Interpolated to `mark_fragment`. `color` is straight alpha by convention.
    color: vec4<f32>,
    local: vec4<f32>,
    data: vec4<f32>,
    // Not interpolated: the first vertex's value.
    flat_data: vec4<u32>,
}

// What `mark_fragment` gets for each pixel.
struct MarkFragment {
    // The pixel's center in device px.
    position: vec2<f32>,
    color: vec4<f32>,
    local: vec4<f32>,
    data: vec4<f32>,
    flat_data: vec4<u32>,
}

// A vertex at `position` with everything else zero.
fn gpui_mark_vertex(position: vec2<f32>) -> MarkVertex {
    return MarkVertex(position, vec2<f32>(0.0), 0u, vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<u32>(0u));
}

// Corner `vertex` (0 to 5) of a unit quad drawn as two triangles: (0,0) to (1,1).
fn gpui_marks_quad_corner(vertex: u32) -> vec2<f32> {
    let i = vertex % 6u;
    let k = select(i, i - 2u, i >= 3u); // corners 0 1 2 | 1 2 3
    return vec2<f32>(f32(k & 1u), f32((k >> 1u) & 1u));
}

// Corner `vertex` (0 to 3) of a unit quad drawn as a triangle strip
// (`MarksTopology::TriangleStrip`): (0,0) to (1,1).
fn gpui_marks_strip_corner(vertex: u32) -> vec2<f32> {
    return vec2<f32>(f32(vertex & 1u), f32((vertex >> 1u) & 1u));
}

// `p`, in the buffer's space, in device px; through layer 1 first when `layer_1`.
fn gpui_marks_to_device(p: vec2<f32>, layer_1: bool) -> vec2<f32> {
    var q = p;
    if layer_1 {
        let l = gpui_marks.layer;
        q = vec2<f32>(l.x * p.x + l.y * p.y, l.z * p.x + l.w * p.y) + gpui_marks.layer_translation;
    }
    let t = gpui_marks.transform;
    return vec2<f32>(t.x * q.x + t.y * q.y, t.z * q.x + t.w * q.y) + gpui_marks.translation;
}

// Device px per unit of the buffer's space (the transformation's mean scale).
fn gpui_marks_px_per_unit() -> f32 {
    let t = gpui_marks.transform;
    return sqrt(abs(t.x * t.w - t.y * t.z));
}

// `p` from the center, y down; `radii` top-left, top-right, bottom-right,
// bottom-left, each at most half the shorter side.
fn gpui_marks_rect_sdf(p: vec2<f32>, half_size: vec2<f32>, radii: vec4<f32>) -> f32 {
    let side = select(radii.xw, radii.yz, p.x >= 0.0);
    let radius = select(side.x, side.y, p.y >= 0.0);
    let q = abs(p) - half_size + radius;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

// How much of the device pixel at `pos` is inside every clip, antialiased as
// retained shapes' clips are.
fn gpui_marks_clip_coverage(pos: vec2<f32>) -> f32 {
    var coverage = 1.0;
    for (var i = 0u; i < gpui_marks.clip_count; i++) {
        let c = gpui_marks.clips[i];
        if c.half_size.x < 0.0 {
            return 0.0;
        }
        let p = vec2<f32>(c.m.x * pos.x + c.m.z * pos.y, c.m.y * pos.x + c.m.w * pos.y) + c.t - c.half_size;
        let d = gpui_marks_rect_sdf(p, c.half_size, c.radii);
        let side = select(c.radii.xw, c.radii.yz, p.x >= 0.0);
        let radius = select(side.x, side.y, p.y >= 0.0);
        let q = abs(p) - c.half_size + radius;
        var g = select(select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), q.x > q.y), normalize(q), q.x > 0.0 && q.y > 0.0);
        g *= vec2<f32>(select(1.0, -1.0, p.x < 0.0), select(1.0, -1.0, p.y < 0.0));
        let per_px = length(vec2<f32>(c.m.x * g.x + c.m.y * g.y, c.m.z * g.x + c.m.w * g.y));
        coverage = min(coverage, clamp(0.5 - d / max(per_px, 1e-30), 0.0, 1.0));
    }
    return coverage;
}

struct GpuiMarksVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local: vec4<f32>,
    @location(2) data: vec4<f32>,
    @location(3) @interpolate(flat) flat_data: vec4<u32>,
}

@vertex
fn gpui_marks_vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> GpuiMarksVarying {
    let m = mark_vertex(vertex, instance);
    var out: GpuiMarksVarying;
    if (m.layer & GPUI_MARK_HIDDEN) != 0u {
        out.position = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
        return out;
    }
    let device = gpui_marks_to_device(m.position, (m.layer & GPUI_MARK_LAYER_1) != 0u) + m.offset_px;
    out.position = vec4<f32>(device / gpui_marks.viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.color = m.color;
    out.local = m.local;
    out.data = m.data;
    out.flat_data = m.flat_data;
    return out;
}

@fragment
fn gpui_marks_fragment(input: GpuiMarksVarying) -> @location(0) vec4<f32> {
    let pos = input.position.xy;
    let mask = gpui_marks.content_mask;
    if pos.x < mask.x || pos.y < mask.y || pos.x >= mask.x + mask.z || pos.y >= mask.y + mask.w {
        discard;
    }
    var clip = 1.0;
    if gpui_marks.clip_count > 0u {
        clip = gpui_marks_clip_coverage(pos);
    }
    let color = mark_fragment(MarkFragment(pos, input.color, input.local, input.data, input.flat_data));
    let alpha = clamp(color.a, 0.0, 1.0) * clip * gpui_marks.opacity;
    // Premultiplied, as renderers blend it.
    return vec4<f32>(clamp(color.rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * alpha, alpha);
}
