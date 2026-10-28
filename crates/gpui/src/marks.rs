//! Marks: instanced draws of a WGSL program over GPU buffers the caller owns.
//!
//! [`Window::paint_marks`](crate::Window::paint_marks) draws `instances` of a
//! [`MarksShader`] inside the frame's own render pass, in paint order with every
//! other primitive: no pass, upload or copy of its own. The shader pulls whatever
//! it needs from up to [`MAX_MARKS_BUFFERS`] storage buffers and
//! [`MAX_MARKS_TEXTURES`] external textures, and GPUI applies the draw's
//! transformations, clips, content mask and opacity around it.
//!
//! GPUI keeps no data here. A [`MarksBuffer`] is the caller's own GPU buffer
//! (Metal: a `metal::Buffer` on the window's device; wgpu: a `wgpu::Buffer` on
//! the renderer's device, with `STORAGE` usage); the caller writes it, on the
//! queue GPUI renders with, before the frame that reads it. There are no ids,
//! versions or uploads: an unchanged mark costs its draw call.
//!
//! On Metal the program is translated to MSL with Naga, as
//! [`Shader`](crate::Shader)s are; on wgpu it is compiled as is.

use crate::{
    Bounds, ContentMask, DrawOrder, ExternalTextureId, MAX_RETAINED_CLIPS, RetainedClip,
    RetainedClipUniform, ScaledPixels, TransformationMatrix,
};
use smallvec::SmallVec;
use std::{any::Any, ops::Range, sync::Arc};

/// Most storage buffers one marks draw binds (`@binding(2)` to `@binding(5)`).
pub const MAX_MARKS_BUFFERS: usize = 4;
/// Most textures one marks draw binds (`@binding(6)` and `@binding(7)`).
pub const MAX_MARKS_TEXTURES: usize = 2;
/// Most bytes of a draw's own uniforms (`@binding(1)`); a shader that declares
/// more than it was given reads zeros.
pub const MARKS_UNIFORM_BYTES: usize = 256;
/// Storage buffer offsets are multiples of this (wgpu's portable minimum).
pub const MARKS_BUFFER_OFFSET_ALIGNMENT: u64 = 256;

/// The WGSL every [`MarksShader`] is compiled with: the bindings GPUI owns, the
/// [`MarkVertex`]/`MarkFragment` structs, helpers, and the entry points
/// `gpui_marks_vertex` and `gpui_marks_fragment`, which call the shader's
/// `mark_vertex` and `mark_fragment`.
pub const MARKS_PRELUDE: &str = include_str!("marks_prelude.wgsl");

/// A marks program: WGSL that defines
///
/// ```wgsl
/// fn mark_vertex(vertex: u32, instance: u32) -> MarkVertex
/// fn mark_fragment(input: MarkFragment) -> vec4<f32> // straight alpha
/// ```
///
/// over its own bindings, all in `@group(0)`:
/// - `@binding(1)`: `var<uniform>`, the draw's [`MarksBindings::uniforms`];
/// - `@binding(2)` to `@binding(5)`: `var<storage, read>`, [`MarksBindings::buffers`] in order;
/// - `@binding(6)`, `@binding(7)`: `texture_2d<f32>`, [`MarksBindings::textures`] in order,
///   sampled with the prelude's `gpui_marks_sampler` (linear, clamped) or loaded.
///
/// `instance` counts from the draw's first instance, so a sub-range draw reads
/// the same buffer slots a whole draw would.
#[derive(Clone, Debug, PartialEq)]
pub struct MarksShader {
    /// A hash of the source; renderers cache pipelines by it.
    pub id: MarksShaderId,
    /// The WGSL source, without the prelude.
    pub source: Arc<str>,
}

/// Identifies a [`MarksShader`] by its source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MarksShaderId(pub u64);

impl MarksShader {
    /// A program from WGSL source.
    pub fn new(source: impl Into<Arc<str>>) -> Self {
        use std::hash::{Hash, Hasher};
        let source = source.into();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        "gpui-marks".hash(&mut hasher);
        source.hash(&mut hasher);
        Self {
            id: MarksShaderId(hasher.finish()),
            source,
        }
    }

    /// The prelude and the source: the WGSL module renderers compile.
    pub fn module_source(&self) -> String {
        format!("{MARKS_PRELUDE}\n{}", self.source)
    }
}

/// What a resource declared in a marks module is, for [`marks_binding_allowed`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarksResource {
    /// `var<uniform>`.
    Uniform,
    /// `var<storage, read>`.
    StorageRead,
    /// `var<storage, read_write>` (never allowed).
    StorageWrite,
    /// A `texture_2d<f32>` (sampled, not multisampled).
    Texture2d,
    /// A filtering sampler.
    Sampler,
    /// Anything else.
    Other,
}

/// Whether a marks module may declare `resource` at `group`/`binding`: the
/// prelude's (bindings 0 and 8, by name), and the program's (see [`MarksShader`]).
/// `name` is the variable's name.
pub fn marks_binding_allowed(
    group: u32,
    binding: u32,
    name: Option<&str>,
    resource: MarksResource,
) -> bool {
    match (group, binding) {
        (0, 0) => name == Some("gpui_marks"),
        (0, 1) => resource == MarksResource::Uniform,
        (0, 2..=5) => resource == MarksResource::StorageRead,
        (0, 6..=7) => resource == MarksResource::Texture2d,
        (0, 8) => name == Some("gpui_marks_sampler"),
        _ => false,
    }
}

/// The error a renderer reports for a resource [`marks_binding_allowed`] refuses.
pub fn marks_binding_error(group: u32, binding: u32) -> String {
    format!(
        "@group({group}) @binding({binding}) is not bound by GPUI: marks bind uniforms at @group(0) @binding(1), \
         `var<storage, read>` buffers at 2 to 5 and `texture_2d<f32>` at 6 and 7"
    )
}

/// A GPU buffer the caller owns, bound from `offset` bytes on. GPUI only binds it.
///
/// The resource is the platform's buffer type: a `metal::Buffer` on the
/// window's Metal device, or a `wgpu::Buffer` with `STORAGE` usage on the wgpu
/// renderer's device. A renderer skips a draw whose buffer it cannot use, or
/// whose `offset` is not a multiple of [`MARKS_BUFFER_OFFSET_ALIGNMENT`].
/// Clone the handle to bind the same buffer every frame without allocating.
#[derive(Clone)]
pub struct MarksBuffer {
    resource: Arc<dyn Any>,
    /// Where the binding starts, in bytes.
    pub offset: u64,
}

impl MarksBuffer {
    /// Binds `resource` (see [`MarksBuffer`]) from its start.
    pub fn new(resource: Arc<dyn Any>) -> Self {
        Self {
            resource,
            offset: 0,
        }
    }

    /// The same buffer from `offset` bytes on.
    pub fn at(&self, offset: u64) -> Self {
        Self {
            resource: self.resource.clone(),
            offset,
        }
    }

    /// The platform buffer, if it is a `T`.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.resource.downcast_ref()
    }
}

impl std::fmt::Debug for MarksBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MarksBuffer")
            .field("resource", &Arc::as_ptr(&self.resource))
            .field("offset", &self.offset)
            .finish()
    }
}

/// What a marks draw binds besides GPUI's own uniforms.
#[derive(Clone, Debug, Default)]
pub struct MarksBindings {
    /// Storage buffers, `@binding(2)` on; at most [`MAX_MARKS_BUFFERS`].
    pub buffers: SmallVec<[MarksBuffer; MAX_MARKS_BUFFERS]>,
    /// External textures (see `Window::share_external_texture`), `@binding(6)`
    /// on; at most [`MAX_MARKS_TEXTURES`]. A missing one binds a transparent texel.
    pub textures: SmallVec<[ExternalTextureId; MAX_MARKS_TEXTURES]>,
    /// The program's uniforms (`@binding(1)`), at most [`MARKS_UNIFORM_BYTES`].
    pub uniforms: Arc<[u8]>,
}

impl MarksBindings {
    /// Binds `buffers` with no textures or uniforms.
    pub fn buffers(buffers: impl IntoIterator<Item = MarksBuffer>) -> Self {
        Self {
            buffers: buffers.into_iter().take(MAX_MARKS_BUFFERS).collect(),
            ..Self::default()
        }
    }

    /// Also binds `texture`; past [`MAX_MARKS_TEXTURES`] it is ignored.
    pub fn texture(mut self, texture: ExternalTextureId) -> Self {
        if self.textures.len() < MAX_MARKS_TEXTURES {
            self.textures.push(texture);
        }
        self
    }

    /// Sets the program's uniforms.
    pub fn uniforms(mut self, uniforms: impl Into<Arc<[u8]>>) -> Self {
        self.uniforms = uniforms.into();
        self
    }
}

/// How each instance's vertices make triangles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MarksTopology {
    /// Every three vertices are a triangle (6 for a quad).
    #[default]
    Triangles,
    /// A strip per instance (4 for a quad: `gpui_marks_strip_corner`). A quad's four
    /// vertices run the vertex function four times, not six, which is what a
    /// mostly-culled draw of many small quads costs.
    TriangleStrip,
}

/// How to draw a marks program; see
/// [`Window::paint_marks`](crate::Window::paint_marks).
#[derive(Clone, Debug)]
pub struct MarksPaint {
    /// Instances to draw: `mark_vertex` sees these instance indices.
    pub instances: Range<u32>,
    /// Vertices per instance, made into triangles by `topology`.
    pub vertices: u32,
    /// How the vertices make triangles.
    pub topology: MarksTopology,
    /// The buffer's space to window pixels.
    pub transformation: TransformationMatrix,
    /// Applied in the buffer's space, before `transformation`, to vertices
    /// `mark_vertex` flags with `GPUI_MARK_LAYER_1` (a live drag).
    pub layer_transformation: TransformationMatrix,
    /// The draw shows only inside every one of these (their intersection), with
    /// antialiased edges, in the buffer's space. At most [`MAX_RETAINED_CLIPS`].
    pub clips: SmallVec<[RetainedClip; MAX_RETAINED_CLIPS]>,
}

impl MarksPaint {
    /// Draw `instances` of `vertices` each under `transformation`.
    pub fn new(instances: Range<u32>, vertices: u32, transformation: TransformationMatrix) -> Self {
        Self {
            instances,
            vertices,
            topology: MarksTopology::Triangles,
            transformation,
            layer_transformation: TransformationMatrix::unit(),
            clips: SmallVec::new(),
        }
    }

    /// Set the layer-1 transformation.
    pub fn layer(mut self, layer_transformation: TransformationMatrix) -> Self {
        self.layer_transformation = layer_transformation;
        self
    }

    /// Draw each instance as one triangle strip.
    pub fn strip(mut self) -> Self {
        self.topology = MarksTopology::TriangleStrip;
        self
    }

    /// Also clip to `clip`. Past [`MAX_RETAINED_CLIPS`] clips, later ones are ignored.
    pub fn clip(mut self, clip: RetainedClip) -> Self {
        if self.clips.len() < MAX_RETAINED_CLIPS {
            self.clips.push(clip);
        }
        self
    }

    /// Axis-aligned bounds, in `transformation`'s target, of the clips'
    /// intersection; `None` without clips.
    pub fn clip_bounds(&self) -> Option<Bounds<ScaledPixels>> {
        crate::retained_shapes::clips_bounds(
            &self.transformation,
            &self.layer_transformation,
            &self.clips,
        )
    }
}

/// A marks draw, as recorded in the scene.
#[derive(Clone, Debug)]
pub struct MarksDraw {
    /// Assigned by the scene.
    pub order: DrawOrder,
    /// Window area the draw covers, for ordering against other primitives.
    pub bounds: Bounds<ScaledPixels>,
    /// Clip; renderers also clip to `bounds`.
    pub content_mask: ContentMask<ScaledPixels>,
    /// The program.
    pub shader: MarksShader,
    /// What it binds.
    pub bindings: MarksBindings,
    /// What to draw; `transformation` is already in device pixels.
    pub paint: MarksPaint,
    /// Opacity multiplier for the whole draw.
    pub opacity: f32,
}

impl MarksDraw {
    /// The prelude's uniforms (`gpui_marks`) on a target of `viewport` device pixels.
    pub fn uniforms(&self, viewport: [f32; 2]) -> MarksUniforms {
        let mask = self.content_mask.bounds.intersect(&self.bounds);
        let (clips, clip_count) = crate::retained_shapes::clip_uniforms(
            &self.paint.transformation,
            &self.paint.layer_transformation,
            &self.paint.clips,
            &self.content_mask,
        );
        let t = &self.paint.transformation;
        let l = &self.paint.layer_transformation;
        MarksUniforms {
            transform: [
                t.rotation_scale[0][0],
                t.rotation_scale[0][1],
                t.rotation_scale[1][0],
                t.rotation_scale[1][1],
            ],
            translation: t.translation,
            viewport_size: viewport,
            layer: [
                l.rotation_scale[0][0],
                l.rotation_scale[0][1],
                l.rotation_scale[1][0],
                l.rotation_scale[1][1],
            ],
            layer_translation: l.translation,
            opacity: self.opacity,
            clip_count,
            content_mask: [
                mask.origin.x.0,
                mask.origin.y.0,
                mask.size.width.0.max(0.0),
                mask.size.height.0.max(0.0),
            ],
            clips,
        }
    }

    /// The instance range and vertex count a renderer draws, or `None` when
    /// there is nothing to draw.
    pub fn draw_range(&self) -> Option<(Range<u32>, u32)> {
        let vertices = match self.paint.topology {
            MarksTopology::Triangles => self.paint.vertices - self.paint.vertices % 3,
            MarksTopology::TriangleStrip if self.paint.vertices >= 3 => self.paint.vertices,
            MarksTopology::TriangleStrip => 0,
        };
        (!self.paint.instances.is_empty() && vertices > 0)
            .then(|| (self.paint.instances.clone(), vertices))
    }

    /// The program's uniform bytes, cut to [`MARKS_UNIFORM_BYTES`] and padded with
    /// zeros to it.
    pub fn uniform_block(&self) -> [u8; MARKS_UNIFORM_BYTES] {
        let mut block = [0u8; MARKS_UNIFORM_BYTES];
        let bytes = &self.bindings.uniforms;
        let len = bytes.len().min(MARKS_UNIFORM_BYTES);
        block[..len].copy_from_slice(&bytes[..len]);
        block
    }
}

/// `gpui_marks` in the prelude (272 bytes). Transforms are row-major:
/// x' = a·x + b·y + tx, y' = c·x + d·y + ty for `[a, b, c, d]`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct MarksUniforms {
    /// The buffer's space to device pixels, 2x2 part.
    pub transform: [f32; 4],
    /// The buffer's space to device pixels, translation.
    pub translation: [f32; 2],
    /// Target size in device pixels.
    pub viewport_size: [f32; 2],
    /// Layer-1 transformation in the buffer's space, 2x2 part.
    pub layer: [f32; 4],
    /// Layer-1 translation.
    pub layer_translation: [f32; 2],
    /// Opacity multiplier.
    pub opacity: f32,
    /// How many of `clips` apply.
    pub clip_count: u32,
    /// Clip: x, y, width, height in device pixels.
    pub content_mask: [f32; 4],
    /// The draw's clips.
    pub clips: [RetainedClipUniform; MAX_RETAINED_CLIPS],
}

impl MarksUniforms {
    /// The block's bytes.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `repr(C)`, made of 4-byte scalars: no padding.
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self).cast::<u8>(),
                std::mem::size_of::<Self>(),
            )
        }
    }
}

/// Marks for renderer tests: every backend draws them over buffers it makes and
/// must hit the same probe colors.
#[cfg(any(test, feature = "test-support", feature = "bench-support"))]
#[doc(hidden)]
pub mod marks_test_scene {
    use super::*;
    pub use crate::retained_shapes_test_scene::{Probe, SIZE, check};
    use crate::{Quad, Rgba, Scene, point, size};

    /// Rectangles pulled from one storage buffer of `Rect`s (x, y, w, h, rgba8, flags).
    pub const RECTS: &str = "
struct Rect { x: f32, y: f32, w: f32, h: f32, color: u32, layer: u32, pad0: u32, pad1: u32 }
@group(0) @binding(2) var<storage, read> rects: array<Rect>;
struct Tint { color: vec4<f32> }
@group(0) @binding(1) var<uniform> tint: Tint;

fn mark_vertex(vertex: u32, instance: u32) -> MarkVertex {
    let r = rects[instance];
    let corner = gpui_marks_quad_corner(vertex);
    var out = gpui_mark_vertex(vec2<f32>(r.x, r.y) + corner * vec2<f32>(r.w, r.h));
    out.layer = r.layer;
    out.color = unpack4x8unorm(r.color).wzyx * tint.color;
    return out;
}

fn mark_fragment(input: MarkFragment) -> vec4<f32> {
    return input.color;
}
";

    /// One record of [`RECTS`]' buffer.
    #[derive(Clone, Copy, Debug)]
    #[repr(C)]
    pub struct Rect {
        /// Origin and size in the buffer's space.
        pub xywh: [f32; 4],
        /// RGBA, 8 bits each, red in the high byte.
        pub color: u32,
        /// `GPUI_MARK_LAYER_1` (1) or `GPUI_MARK_HIDDEN` (2) bits.
        pub layer: u32,
        #[doc(hidden)]
        pub pad: [u32; 2],
    }

    /// The rectangles in the scene's buffer, as bytes.
    pub fn buffer_bytes() -> Vec<u8> {
        let rect = |x, y, w, h, color, layer| Rect {
            xywh: [x, y, w, h],
            color,
            layer,
            pad: [0; 2],
        };
        let rects = [
            // 0: red, drawn whole by the first draw.
            rect(4., 4., 12., 12., 0xff0000ff, 0),
            // 1: green, moved by the layer transformation (+20 x).
            rect(4., 20., 12., 12., 0x00ff00ff, 1),
            // 2: hidden.
            rect(20., 4., 12., 12., 0xffffffff, 2),
            // 3: blue, drawn by the second draw, clipped to a 6-wide clip.
            rect(40., 4., 20., 12., 0x0000ffff, 0),
            // 4: white, drawn by the second draw at half opacity.
            rect(40., 40., 12., 12., 0xffffffff, 0),
        ];
        // SAFETY: `Rect` is `repr(C)` of 4-byte scalars.
        unsafe {
            std::slice::from_raw_parts(rects.as_ptr().cast::<u8>(), std::mem::size_of_val(&rects))
        }
        .to_vec()
    }

    fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(w), ScaledPixels(h)),
        )
    }

    fn tint(rgba: [f32; 4]) -> Arc<[u8]> {
        rgba.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// The scene over `buffer` (holding [`buffer_bytes`]), and the probes.
    pub fn scene(buffer: MarksBuffer) -> (Scene, Vec<Probe>) {
        let full = bounds(0., 0., SIZE.0 as f32, SIZE.1 as f32);
        let mut scene = Scene::default();
        let shader = MarksShader::new(RECTS);
        // A gray backdrop.
        scene.insert_primitive(Quad {
            bounds: full,
            content_mask: ContentMask::from(full),
            background: crate::Hsla::from(Rgba {
                r: 0.,
                g: 0.,
                b: 0.,
                a: 1.,
            })
            .into(),
            ..Default::default()
        });
        let bindings = MarksBindings::buffers([buffer]).uniforms(tint([1.; 4]));
        // Rects 0..3: identity, layer 1 moves +20 x.
        scene.insert_primitive(MarksDraw {
            order: 0,
            bounds: full,
            content_mask: ContentMask::from(full),
            shader: shader.clone(),
            bindings: bindings.clone(),
            paint: MarksPaint::new(0..3, 6, TransformationMatrix::unit()).layer(
                TransformationMatrix::unit().translate(point(ScaledPixels(20.), ScaledPixels(0.))),
            ),
            opacity: 1.,
        });
        // Rect 3: clipped to x 40..46.
        scene.insert_primitive(MarksDraw {
            order: 0,
            bounds: full,
            content_mask: ContentMask::from(full),
            shader: shader.clone(),
            bindings: bindings.clone(),
            paint: MarksPaint::new(3..4, 6, TransformationMatrix::unit()).clip(RetainedClip {
                transform: [1., 0., 0., 1., 40., 0.],
                width: 6.,
                height: SIZE.1 as f32,
                corner_radii: [0.; 4],
                layer_1: false,
            }),
            opacity: 1.,
        });
        // Rect 4: half opacity, tinted to keep only green.
        scene.insert_primitive(MarksDraw {
            order: 0,
            bounds: full,
            content_mask: ContentMask::from(full),
            shader,
            bindings: MarksBindings::buffers(bindings.buffers.iter().cloned())
                .uniforms(tint([0., 1., 0., 1.])),
            paint: MarksPaint::new(4..5, 6, TransformationMatrix::unit()),
            opacity: 0.5,
        });
        scene.finish();
        let probe = |what, x, y, rgba| Probe { what, x, y, rgba };
        let probes = vec![
            probe("red rect", 10, 10, [255, 0, 0, 255]),
            probe("green rect moved by the layer", 30, 26, [0, 255, 0, 255]),
            probe("green rect's old place", 10, 26, [0, 0, 0, 255]),
            probe("hidden rect", 26, 10, [0, 0, 0, 255]),
            probe("blue inside the clip", 43, 10, [0, 0, 255, 255]),
            probe("blue outside the clip", 50, 10, [0, 0, 0, 255]),
            probe("tinted, half opacity", 46, 46, [0, 128, 0, 255]),
        ];
        (scene, probes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniforms_layout_matches_the_prelude() {
        assert_eq!(std::mem::size_of::<MarksUniforms>(), 272);
        assert_eq!(std::mem::offset_of!(MarksUniforms, content_mask), 64);
        assert_eq!(std::mem::offset_of!(MarksUniforms, clips), 80);
    }

    #[test]
    fn bindings_follow_the_program_contract() {
        use MarksResource::*;
        assert!(marks_binding_allowed(0, 0, Some("gpui_marks"), Uniform));
        assert!(!marks_binding_allowed(0, 0, Some("mine"), Uniform));
        assert!(marks_binding_allowed(0, 1, Some("u"), Uniform));
        assert!(marks_binding_allowed(0, 2, Some("xs"), StorageRead));
        assert!(!marks_binding_allowed(0, 2, Some("xs"), StorageWrite));
        assert!(marks_binding_allowed(0, 7, Some("t"), Texture2d));
        assert!(!marks_binding_allowed(0, 9, Some("t"), Texture2d));
        assert!(!marks_binding_allowed(1, 0, Some("gpui_marks"), Uniform));
    }

    #[test]
    fn draw_range_rounds_vertices_to_triangles() {
        let draw = |instances: Range<u32>, vertices| MarksDraw {
            order: 0,
            bounds: Bounds::default(),
            content_mask: ContentMask::default(),
            shader: MarksShader::new(""),
            bindings: MarksBindings::default(),
            paint: MarksPaint::new(instances, vertices, TransformationMatrix::unit()),
            opacity: 1.,
        };
        assert_eq!(draw(2..5, 7).draw_range(), Some((2..5, 6)));
        assert_eq!(draw(2..2, 6).draw_range(), None);
        assert_eq!(draw(0..1, 2).draw_range(), None);
        let mut strip = draw(0..1, 4);
        strip.paint = strip.paint.strip();
        assert_eq!(strip.draw_range(), Some((0..1, 4)));
    }

    #[test]
    fn uniform_block_pads_and_cuts() {
        let mut draw = MarksDraw {
            order: 0,
            bounds: Bounds::default(),
            content_mask: ContentMask::default(),
            shader: MarksShader::new(""),
            bindings: MarksBindings::default().uniforms(vec![7u8; 4]),
            paint: MarksPaint::new(0..1, 6, TransformationMatrix::unit()),
            opacity: 1.,
        };
        let block = draw.uniform_block();
        assert_eq!(&block[..5], &[7, 7, 7, 7, 0]);
        draw.bindings.uniforms = vec![1u8; 1000].into();
        assert!(draw.uniform_block().iter().all(|b| *b == 1));
    }
}
