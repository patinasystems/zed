// Storage-buffer instance transport. Concatenated after `shaders.wgsl` on backends
// with storage buffer support; `shaders_webgl.wgsl` is the WebGL2 counterpart that
// decodes the same records from a texture instead.
//
// All buffers share `@group(1) @binding(0)` because each pipeline binds only the
// buffer its own entry points read.

@group(1) @binding(0) var<storage, read> b_quads: array<Quad>;
@group(1) @binding(0) var<storage, read> b_shadows: array<ShadowRecord>;
@group(1) @binding(0) var<storage, read> b_path_vertices: array<PathVertexRecord>;
@group(1) @binding(0) var<storage, read> b_path_sprites: array<PathSprite>;
@group(1) @binding(0) var<storage, read> b_underlines: array<UnderlineRecord>;
@group(1) @binding(0) var<storage, read> b_mono_sprites: array<SpriteRecord>;
@group(1) @binding(0) var<storage, read> b_poly_sprites: array<PolychromeSpriteRecord>;

// Quads are fill-heavy: shaders read the whole quad in place, never expanding a record per
// pixel (`WgpuRenderer::write_scene_records`).
fn load_quad(instance_id: u32) -> Quad {
    return b_quads[instance_id];
}

fn load_shadow_record(instance_id: u32) -> ShadowRecord {
    return b_shadows[instance_id];
}

fn load_path_vertex_record(vertex_id: u32) -> PathVertexRecord {
    return b_path_vertices[vertex_id];
}

fn load_path_sprite(instance_id: u32) -> PathSprite {
    return b_path_sprites[instance_id];
}

fn load_underline_record(instance_id: u32) -> UnderlineRecord {
    return b_underlines[instance_id];
}

fn load_mono_sprite_record(instance_id: u32) -> SpriteRecord {
    return b_mono_sprites[instance_id];
}

fn load_poly_sprite_record(instance_id: u32) -> PolychromeSpriteRecord {
    return b_poly_sprites[instance_id];
}

// The scene's record table (`gpui::InstanceTables`), which the records index.
@group(3) @binding(0) var<storage, read> b_tables: array<vec4<f32>>;

fn load_table_texel(index: u32) -> vec4<f32> {
    return b_tables[index];
}

@group(1) @binding(0) var<storage, read> b_retained_shapes: array<RetainedShape>;

fn load_retained_shape(instance_id: u32) -> RetainedShape {
    return b_retained_shapes[instance_id];
}

// The frame's gradient records (`gpui::GradientRecord`), for quads.
@group(2) @binding(0) var<storage, read> b_gradients: array<vec4<f32>>;

fn load_gradient_texel(index: u32) -> vec4<f32> {
    return b_gradients[index];
}
