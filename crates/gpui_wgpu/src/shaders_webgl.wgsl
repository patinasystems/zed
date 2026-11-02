@group(1) @binding(0) var t_instances: texture_2d<u32>;
// The frame's gradient records (`gpui::GradientRecord`), for quads, written first in
// the frame's instance data, so they start at texel 0.
@group(2) @binding(0) var t_gradients: texture_2d<u32>;
// The scene's record table (`gpui::InstanceTables`), in the same texture: renderers offset
// the records' table indices by the table's first texel, so they index it absolutely.
@group(3) @binding(0) var t_tables: texture_2d<u32>;

fn load_table_texel(index: u32) -> vec4<f32> {
    let width = textureDimensions(t_tables).x;
    let coordinate = vec2<i32>(i32(index % width), i32(index / width));
    return bitcast<vec4<f32>>(textureLoad(t_tables, coordinate, 0));
}

fn load_gradient_texel(index: u32) -> vec4<f32> {
    let width = textureDimensions(t_gradients).x;
    let coordinate = vec2<i32>(i32(index % width), i32(index / width));
    return bitcast<vec4<f32>>(textureLoad(t_gradients, coordinate, 0));
}

// Each texel of `t_instances` packs four 32-bit words of instance data. Most
// records are read strictly front to back through a cursor that keeps the
// most recently fetched texel and only touches the texture again when the next
// word crosses a texel boundary. This fetches each texel exactly once per
// record load instead of once per word. The `read_*` functions must therefore
// consume their words in struct declaration order; skipping or re-reading words
// would still return correct data (the cursor re-fetches whenever the texel
// index changes) but would forfeit the single-fetch-per-texel guarantee.
struct InstanceCursor {
    word_index: u32,
    texel_index: u32,
    texel: vec4<u32>,
    width: u32,
}

fn fetch_instance_texel(texel_index: u32, width: u32) -> vec4<u32> {
    let coordinate = vec2<i32>(
        i32(texel_index % width),
        i32(texel_index / width),
    );
    return textureLoad(t_instances, coordinate, 0);
}

fn instance_cursor(word_index: u32) -> InstanceCursor {
    let width = textureDimensions(t_instances).x;
    let texel_index = word_index / 4u;
    return InstanceCursor(
        word_index,
        texel_index,
        fetch_instance_texel(texel_index, width),
        width,
    );
}

fn read_word(cursor: ptr<function, InstanceCursor>) -> u32 {
    let word_index = (*cursor).word_index;
    let texel_index = word_index / 4u;
    if texel_index != (*cursor).texel_index {
        (*cursor).texel = fetch_instance_texel(texel_index, (*cursor).width);
        (*cursor).texel_index = texel_index;
    }
    (*cursor).word_index = word_index + 1u;
    return (*cursor).texel[word_index % 4u];
}

fn read_f32(cursor: ptr<function, InstanceCursor>) -> f32 {
    return bitcast<f32>(read_word(cursor));
}

fn read_vec2_f32(cursor: ptr<function, InstanceCursor>) -> vec2<f32> {
    return vec2<f32>(read_f32(cursor), read_f32(cursor));
}

fn read_bounds(cursor: ptr<function, InstanceCursor>) -> Bounds {
    return Bounds(read_vec2_f32(cursor), read_vec2_f32(cursor));
}

fn instance_texel(texel_index: u32) -> vec4<u32> {
    return fetch_instance_texel(texel_index, textureDimensions(t_instances).x);
}

// Records of whole texels, decoded at fixed positions: some WebGL shader compilers fail to
// optimize the cursor's branches and dynamic vector indexing.
fn load_quad_record(instance_id: u32) -> QuadRecord {
    let first = instance_id * 4u;
    let t0 = bitcast<vec4<f32>>(instance_texel(first));
    let t1 = bitcast<vec4<f32>>(instance_texel(first + 1u));
    let t2 = bitcast<vec4<f32>>(instance_texel(first + 2u));
    let t3 = instance_texel(first + 3u);
    return QuadRecord(
        Bounds(t0.xy, t0.zw),
        Corners(t1.x, t1.y, t1.z, t1.w),
        Hsla(t2.x, t2.y, t2.z, t2.w),
        t3.x,
        t3.y,
        t3.z,
        t3.w,
    );
}

fn load_shadow_record(instance_id: u32) -> ShadowRecord {
    let first = instance_id * 6u;
    let t0 = bitcast<vec4<f32>>(instance_texel(first));
    let t1 = bitcast<vec4<f32>>(instance_texel(first + 1u));
    let t2 = bitcast<vec4<f32>>(instance_texel(first + 2u));
    let t3 = bitcast<vec4<f32>>(instance_texel(first + 3u));
    let t4 = bitcast<vec4<f32>>(instance_texel(first + 4u));
    let t5 = instance_texel(first + 5u);
    return ShadowRecord(
        Bounds(t0.xy, t0.zw),
        Corners(t1.x, t1.y, t1.z, t1.w),
        Hsla(t2.x, t2.y, t2.z, t2.w),
        Bounds(t3.xy, t3.zw),
        Corners(t4.x, t4.y, t4.z, t4.w),
        bitcast<f32>(t5.x),
        t5.y,
        t5.z,
        t5.w,
    );
}

fn load_path_vertex_record(vertex_id: u32) -> PathVertexRecord {
    var cursor = instance_cursor(vertex_id * 6u);
    return PathVertexRecord(
        read_vec2_f32(&cursor),
        read_vec2_f32(&cursor),
        read_word(&cursor),
        read_word(&cursor),
    );
}

fn load_path_sprite(instance_id: u32) -> PathSprite {
    var cursor = instance_cursor(instance_id * 4u);
    return PathSprite(read_bounds(&cursor));
}

fn load_underline_record(instance_id: u32) -> UnderlineRecord {
    let first = instance_id * 3u;
    let t0 = bitcast<vec4<f32>>(instance_texel(first));
    let t1 = bitcast<vec4<f32>>(instance_texel(first + 1u));
    let t2 = instance_texel(first + 2u);
    return UnderlineRecord(
        Bounds(t0.xy, t0.zw),
        Hsla(t1.x, t1.y, t1.z, t1.w),
        bitcast<f32>(t2.x),
        t2.y,
        t2.z,
        t2.w,
    );
}

fn load_mono_sprite_record(instance_id: u32) -> SpriteRecord {
    let first = instance_id * 4u;
    let t0 = bitcast<vec4<i32>>(instance_texel(first));
    let t1 = bitcast<vec4<f32>>(instance_texel(first + 1u));
    let t2 = bitcast<vec4<f32>>(instance_texel(first + 2u));
    let t3 = instance_texel(first + 3u);
    return SpriteRecord(
        AtlasBounds(t0.xy, t0.zw),
        Bounds(t1.xy, t1.zw),
        Hsla(t2.x, t2.y, t2.z, t2.w),
        t3.x,
        t3.y,
        t3.zw,
    );
}

fn load_poly_sprite_record(instance_id: u32) -> PolychromeSpriteRecord {
    let first = instance_id * 4u;
    let t0 = bitcast<vec4<i32>>(instance_texel(first));
    let t1 = bitcast<vec4<f32>>(instance_texel(first + 1u));
    let t2 = bitcast<vec4<f32>>(instance_texel(first + 2u));
    let t3 = instance_texel(first + 3u);
    return PolychromeSpriteRecord(
        AtlasBounds(t0.xy, t0.zw),
        Bounds(t1.xy, t1.zw),
        Corners(t2.x, t2.y, t2.z, t2.w),
        bitcast<f32>(t3.x),
        t3.y,
        t3.z,
        t3.w,
    );
}

// Retained shapes: 16 words per record, read from the buffer's own texture
// (bound in place of the frame's instance texture).
fn load_retained_shape(instance_id: u32) -> RetainedShape {
    var cursor = instance_cursor(instance_id * 16u);
    let a = read_f32(&cursor);
    let b = read_f32(&cursor);
    let c = read_f32(&cursor);
    let d = read_f32(&cursor);
    let tx = read_f32(&cursor);
    let ty = read_f32(&cursor);
    let width = read_f32(&cursor);
    let height = read_f32(&cursor);
    let top_left = read_f32(&cursor);
    let top_right = read_f32(&cursor);
    let bottom_right = read_f32(&cursor);
    let bottom_left = read_f32(&cursor);
    let corner_radii = vec4<f32>(top_left, top_right, bottom_right, bottom_left);
    let stroke_width = read_f32(&cursor);
    let fill = read_word(&cursor);
    let stroke = read_word(&cursor);
    let kind = read_word(&cursor);
    return RetainedShape(a, b, c, d, tx, ty, width, height, corner_radii, stroke_width, fill, stroke, kind);
}
