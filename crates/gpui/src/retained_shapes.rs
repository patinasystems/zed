//! Retained, GPU-resident shape instances.
//!
//! A [`RetainedShapes`] buffer holds [`ShapeInstance`]s (rectangles, rounded
//! rectangles and ellipses with a fill and a stroke) in a space of the caller's
//! choosing, such as a canvas page. Each renderer keeps a GPU copy keyed by the
//! buffer's id and uploads only the ranges written since its last upload.
//! [`Window::paint_retained_shapes`](crate::Window::paint_retained_shapes) draws a
//! range of the buffer under a transformation, so panning and zooming upload
//! nothing, and a [layer transformation](RetainedShapesPaint::layer_transformation)
//! moves the instances flagged [`ShapeInstance::LAYER_1`] (a live drag) without
//! touching the buffer. A draw can be [clipped](RetainedShapesPaint::clips)
//! to rounded rectangles in the buffer's space, rotated or not (a canvas
//! frame that clips its content); unclipped draws pay one uniform check.

use crate::{Bounds, ContentMask, DrawOrder, ScaledPixels, TransformationMatrix};
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

/// One shape, drawn as a signed-distance field in its own space.
///
/// `transform` maps shape-local points (0..`width`, 0..`height`) into the
/// buffer's space: `x' = a·x + c·y + tx`, `y' = b·x + d·y + ty`. The layout is
/// shared with the shaders (16 four-byte words).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct ShapeInstance {
    /// `[a, b, c, d, tx, ty]`.
    pub transform: [f32; 6],
    /// Local width.
    pub width: f32,
    /// Local height.
    pub height: f32,
    /// Corner radii in local units, rectangles only: top-left, top-right,
    /// bottom-right, bottom-left (local y grows down). Each is clamped to
    /// half the shorter side when drawn.
    pub corner_radii: [f32; 4],
    /// Stroke width in local units, centered on the outline. 0 draws no stroke.
    pub stroke_width: f32,
    /// Fill color, `0xRRGGBBAA` (straight alpha). Alpha 0 draws no fill.
    pub fill: u32,
    /// Stroke color, `0xRRGGBBAA`.
    pub stroke: u32,
    /// Kind in the low byte ([`ShapeInstance::RECT`], ...), flags above it
    /// ([`ShapeInstance::HIDDEN`], [`ShapeInstance::LAYER_1`]).
    pub kind: u32,
}

impl ShapeInstance {
    /// A rectangle, each corner rounded by its entry of `corner_radii`.
    pub const RECT: u32 = 0;
    /// An ellipse inscribed in the local rectangle.
    pub const ELLIPSE: u32 = 1;
    /// A line from (0, 0) to (`width`, 0), `stroke_width` thick with round caps,
    /// in the stroke color.
    pub const SEGMENT: u32 = 2;
    /// A quadratic curve from (0, 0) through control point (`width` / 2,
    /// `height`) to (`width`, 0), stroked like a segment (an arrow's bent body).
    pub const ARC: u32 = 3;
    /// A triangle (0, 0), (`width`, `height` / 2), (0, `height`): an arrowhead
    /// pointing along +x, with fill and stroke.
    pub const TRIANGLE: u32 = 4;
    /// A rounded bar (radius `height` / 2, fill only) standing in for a line of
    /// text. Drawn only while its on-screen height is inside the paint's
    /// [text bar band](RetainedShapesPaint::text_bar_band_px).
    pub const TEXT_BAR: u32 = 5;
    /// Not drawn; keeps its slot so ranges stay stable.
    pub const HIDDEN: u32 = 1 << 8;
    /// Drawn through the paint's layer transformation (for example a live drag).
    pub const LAYER_1: u32 = 1 << 16;

    /// The kind without flags.
    pub fn base_kind(&self) -> u32 {
        self.kind & 0xff
    }
}

/// Packs an [`Rgba`](crate::Rgba) as `0xRRGGBBAA`.
pub fn rgba8(color: crate::Rgba) -> u32 {
    let c = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
    (c(color.r) << 24) | (c(color.g) << 16) | (c(color.b) << 8) | c(color.a)
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// The CPU side of a retained instance buffer. Clones share the buffer.
#[derive(Clone)]
pub struct RetainedShapes {
    id: u64,
    inner: Arc<Mutex<Inner>>,
}

struct Inner {
    instances: Vec<ShapeInstance>,
    /// Bumps on every write.
    version: u64,
    /// Ranges written since version `dirty_since`. A renderer whose copy is at
    /// `dirty_since` becomes current by copying these ranges.
    dirty: Vec<Range<usize>>,
    dirty_since: u64,
}

/// What a renderer copies to bring its GPU copy up to date, from
/// [`RetainedShapes::sync`].
pub enum RetainedUpload<'a> {
    /// Already current.
    None,
    /// Copy these ranges of the instances; the length is unchanged.
    Ranges(&'a [ShapeInstance], &'a [Range<usize>]),
    /// Replace everything (the length may have changed).
    All(&'a [ShapeInstance]),
}

impl Default for RetainedShapes {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for RetainedShapes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedShapes")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl RetainedShapes {
    /// An empty buffer with a fresh id.
    pub fn new() -> Self {
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            inner: Arc::new(Mutex::new(Inner {
                instances: Vec::new(),
                version: 1,
                dirty: Vec::new(),
                dirty_since: 0,
            })),
        }
    }

    /// Identifies the buffer to renderers.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Bumps on every write.
    pub fn version(&self) -> u64 {
        self.inner.lock().version
    }

    /// Number of instances.
    pub fn len(&self) -> usize {
        self.inner.lock().instances.len()
    }

    /// True when there are no instances.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Replace every instance. Renderers upload the whole buffer.
    pub fn set_all(&self, instances: Vec<ShapeInstance>) {
        let mut inner = self.inner.lock();
        inner.instances = instances;
        inner.version += 1;
        inner.dirty.clear();
        // No renderer can use ranges against the old contents.
        inner.dirty_since = 0;
    }

    /// Overwrite instances starting at `start`. Renderers upload this range.
    ///
    /// Panics if the range is out of bounds.
    pub fn write(&self, start: usize, values: &[ShapeInstance]) {
        if values.is_empty() {
            return;
        }
        let mut inner = self.inner.lock();
        inner.instances[start..start + values.len()].copy_from_slice(values);
        inner.mark(start..start + values.len());
    }

    /// Edit instances in place. `f` returns the range it changed, which
    /// renderers upload; an empty range uploads nothing.
    pub fn update(&self, f: impl FnOnce(&mut [ShapeInstance]) -> Range<usize>) {
        let mut inner = self.inner.lock();
        let range = f(&mut inner.instances);
        let len = inner.instances.len();
        let range = range.start.min(len)..range.end.min(len);
        if !range.is_empty() {
            inner.mark(range);
        }
    }

    /// For renderers: given the version their copy holds (0 for none), call
    /// `upload` with what to copy, and return the version they then hold.
    pub fn sync(&self, have: u64, upload: impl FnOnce(RetainedUpload<'_>)) -> u64 {
        let mut inner = self.inner.lock();
        let version = inner.version;
        if have == version {
            upload(RetainedUpload::None);
            return version;
        }
        if have != 0 && have == inner.dirty_since {
            upload(RetainedUpload::Ranges(&inner.instances, &inner.dirty));
        } else {
            upload(RetainedUpload::All(&inner.instances));
        }
        inner.dirty.clear();
        inner.dirty_since = version;
        version
    }
}

impl Inner {
    fn mark(&mut self, range: Range<usize>) {
        self.version += 1;
        // Coalesce with the last range when they touch, so writes in order stay
        // one copy.
        match self.dirty.last_mut() {
            Some(last) if range.start <= last.end && range.end >= last.start => {
                last.start = last.start.min(range.start);
                last.end = last.end.max(range.end);
            }
            _ => self.dirty.push(range),
        }
        // Many scattered writes: one copy of their span beats many small ones.
        if self.dirty.len() > 64 {
            let start = self.dirty.iter().map(|r| r.start).min().unwrap_or(0);
            let end = self.dirty.iter().map(|r| r.end).max().unwrap_or(0);
            self.dirty.clear();
            self.dirty.push(start..end);
        }
    }
}

/// How to draw a range of a [`RetainedShapes`] buffer; see
/// [`Window::paint_retained_shapes`](crate::Window::paint_retained_shapes).
#[derive(Clone, Debug)]
pub struct RetainedShapesPaint {
    /// Instances to draw.
    pub range: Range<u32>,
    /// Buffer space to window pixels.
    pub transformation: TransformationMatrix,
    /// Applied in buffer space, before `transformation`, to instances flagged
    /// [`ShapeInstance::LAYER_1`].
    pub layer_transformation: TransformationMatrix,
    /// Shapes whose longer side is below this many device pixels draw as a
    /// solid quad of at least one pixel, with alpha carrying their coverage.
    pub lod_min_device_px: f32,
    /// [`ShapeInstance::TEXT_BAR`]s draw while their height on screen, in
    /// device pixels, is at least the first value and below the second; the
    /// caller draws real glyphs above the band and nothing below it.
    pub text_bar_band_px: [f32; 2],
    /// The draw shows only inside every one of these (their intersection),
    /// with antialiased edges. At most [`MAX_RETAINED_CLIPS`].
    pub clips: SmallVec<[RetainedClip; MAX_RETAINED_CLIPS]>,
}

/// Most [`RetainedClip`]s one draw takes.
pub const MAX_RETAINED_CLIPS: usize = 4;

/// A rounded rectangle that clips a retained-shape draw, in the buffer's
/// space: a canvas frame that clips its content, even rotated or rounded.
/// Drawing outside it is discarded; its edge is antialiased like a shape's.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RetainedClip {
    /// Maps the clip's own space, where it is the rectangle (0, 0)-(`width`,
    /// `height`), into the buffer's space, as [`ShapeInstance::transform`] does.
    pub transform: [f32; 6],
    /// Local width.
    pub width: f32,
    /// Local height.
    pub height: f32,
    /// Corner radii, as [`ShapeInstance::corner_radii`] (each clamped to half
    /// the shorter side).
    pub corner_radii: [f32; 4],
    /// Moves with the paint's layer transformation, as instances flagged
    /// [`ShapeInstance::LAYER_1`] do.
    pub layer_1: bool,
}

impl RetainedShapesPaint {
    /// Draw `range` under `transformation`, with no layer transformation.
    pub fn new(range: Range<u32>, transformation: TransformationMatrix) -> Self {
        Self {
            range,
            transformation,
            layer_transformation: TransformationMatrix::unit(),
            lod_min_device_px: 1.5,
            text_bar_band_px: [4.0, 8.0],
            clips: SmallVec::new(),
        }
    }

    /// Set the layer-1 transformation.
    pub fn layer(mut self, layer_transformation: TransformationMatrix) -> Self {
        self.layer_transformation = layer_transformation;
        self
    }

    /// Also clip to `clip`. Past [`MAX_RETAINED_CLIPS`] clips, later ones
    /// are ignored.
    pub fn clip(mut self, clip: RetainedClip) -> Self {
        if self.clips.len() < MAX_RETAINED_CLIPS {
            self.clips.push(clip);
        }
        self
    }

    /// Axis-aligned bounds, in `transformation`'s target, that hold the
    /// intersection of the clips and their antialiased edges; `None` without clips.
    pub fn clip_bounds(&self) -> Option<Bounds<ScaledPixels>> {
        clips_bounds(
            &self.transformation,
            &self.layer_transformation,
            &self.clips,
        )
    }
}

/// `clip`'s space to `transformation`'s target (through `layer` first when the
/// clip is [`RetainedClip::layer_1`]), as `[a, b, c, d, tx, ty]` in f64
/// (x' = a·x + c·y + tx).
pub(crate) fn clip_to_target(
    transformation: &TransformationMatrix,
    layer: &TransformationMatrix,
    clip: &RetainedClip,
) -> [f64; 6] {
    let mut m = clip.transform.map(f64::from);
    let mut then = |t: &TransformationMatrix| {
        let [[a, c], [b, d]] = t.rotation_scale.map(|row| row.map(f64::from));
        let [tx, ty] = t.translation.map(f64::from);
        m = [
            a * m[0] + c * m[1],
            b * m[0] + d * m[1],
            a * m[2] + c * m[3],
            b * m[2] + d * m[3],
            a * m[4] + c * m[5] + tx,
            b * m[4] + d * m[5] + ty,
        ];
    };
    if clip.layer_1 {
        then(layer);
    }
    then(transformation);
    m
}

/// Axis-aligned bounds, in `transformation`'s target, that hold the intersection
/// of `clips` and their antialiased edges; `None` without clips.
pub(crate) fn clips_bounds(
    transformation: &TransformationMatrix,
    layer: &TransformationMatrix,
    clips: &[RetainedClip],
) -> Option<Bounds<ScaledPixels>> {
    clips
        .iter()
        .take(MAX_RETAINED_CLIPS)
        .map(|clip| {
            let m = clip_to_target(transformation, layer, clip);
            let (w, h) = (f64::from(clip.width), f64::from(clip.height));
            let corners = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)]
                .map(|(x, y)| (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]));
            let (x0, y0, x1, y1) = corners
                .iter()
                .fold((f64::MAX, f64::MAX, f64::MIN, f64::MIN), |b, c| {
                    (b.0.min(c.0), b.1.min(c.1), b.2.max(c.0), b.3.max(c.1))
                });
            Bounds::from_corners(
                crate::point(
                    ScaledPixels(x0.floor() as f32 - 1.0),
                    ScaledPixels(y0.floor() as f32 - 1.0),
                ),
                crate::point(
                    ScaledPixels(x1.ceil() as f32 + 1.0),
                    ScaledPixels(y1.ceil() as f32 + 1.0),
                ),
            )
        })
        .reduce(|a, b| a.intersect(&b))
}

/// A clip's uniform: the map from device pixels to the clip's space (the
/// inverse of the clip's transform followed by the draw's).
pub(crate) fn clip_uniform(
    transformation: &TransformationMatrix,
    layer: &TransformationMatrix,
    clip: &RetainedClip,
) -> RetainedClipUniform {
    // A clip with no area hides everything.
    const EMPTY: RetainedClipUniform = RetainedClipUniform {
        matrix: [0.0; 4],
        translation: [0.0; 2],
        half_size: [-1.0, -1.0],
        corner_radii: [0.0; 4],
    };
    let (w, h) = (clip.width, clip.height);
    if !(w > 0.0 && h > 0.0) {
        return EMPTY;
    }
    let m = clip_to_target(transformation, layer, clip);
    let det = m[0] * m[3] - m[2] * m[1];
    let inverse = [m[3] / det, -m[1] / det, -m[2] / det, m[0] / det];
    let [a, b, c, d] = inverse.map(|v| v as f32);
    let (tx, ty) = (
        -(inverse[0] * m[4] + inverse[2] * m[5]),
        -(inverse[1] * m[4] + inverse[3] * m[5]),
    );
    let uniform = RetainedClipUniform {
        matrix: [a, b, c, d],
        translation: [tx as f32, ty as f32],
        half_size: [0.5 * w, 0.5 * h],
        corner_radii: clip.corner_radii.map(|r| r.clamp(0.0, 0.5 * w.min(h))),
    };
    let finite = uniform
        .matrix
        .iter()
        .chain(&uniform.translation)
        .all(|v| v.is_finite());
    if det == 0.0 || !finite {
        EMPTY
    } else {
        uniform
    }
}

/// A rounded content mask (an `overflow` clip with a rounded border) as one more
/// clip, already in device pixels; `None` for a square mask.
pub(crate) fn rounded_mask_clip(mask: &ContentMask<ScaledPixels>) -> Option<RetainedClipUniform> {
    mask.is_rounded().then(|| {
        let b = mask.bounds;
        let (w, h) = (b.size.width.0.max(0.0), b.size.height.0.max(0.0));
        let r = mask.corner_radii;
        RetainedClipUniform {
            matrix: [1.0, 0.0, 0.0, 1.0],
            translation: [-b.origin.x.0, -b.origin.y.0],
            half_size: [0.5 * w, 0.5 * h],
            corner_radii: [
                r.top_left.0,
                r.top_right.0,
                r.bottom_right.0,
                r.bottom_left.0,
            ]
            .map(|r| r.clamp(0.0, 0.5 * w.min(h))),
        }
    })
}

/// The clip uniforms of a draw: `clips`, then the rounded content mask, at most
/// [`MAX_RETAINED_CLIPS`] in all (past it the mask clips square). Returns them and
/// how many apply.
pub(crate) fn clip_uniforms(
    transformation: &TransformationMatrix,
    layer: &TransformationMatrix,
    clips: &[RetainedClip],
    mask: &ContentMask<ScaledPixels>,
) -> ([RetainedClipUniform; MAX_RETAINED_CLIPS], u32) {
    let rounded = rounded_mask_clip(mask);
    let count = (clips.len() + usize::from(rounded.is_some())).min(MAX_RETAINED_CLIPS) as u32;
    let uniforms = std::array::from_fn(|i| {
        clips.get(i).map_or_else(
            || rounded.filter(|_| i == clips.len()).unwrap_or_default(),
            |clip| clip_uniform(transformation, layer, clip),
        )
    });
    (uniforms, count)
}

/// A draw of retained shapes, as recorded in the scene.
#[derive(Clone, Debug)]
pub struct RetainedShapesDraw {
    /// Assigned by the scene.
    pub order: DrawOrder,
    /// Window area the draw covers, for ordering against other primitives.
    pub bounds: Bounds<ScaledPixels>,
    /// Clip; renderers also clip to `bounds`.
    pub content_mask: ContentMask<ScaledPixels>,
    /// The buffer.
    pub shapes: RetainedShapes,
    /// What to draw; `transformation` is already in device pixels.
    pub paint: RetainedShapesPaint,
    /// Opacity multiplier for the whole draw.
    pub opacity: f32,
}

impl RetainedShapesDraw {
    /// Shader uniforms for this draw on a target of `viewport` device pixels.
    pub fn uniforms(&self, viewport: [f32; 2]) -> RetainedShapesUniforms {
        let mask = self.content_mask.bounds.intersect(&self.bounds);
        let m = |t: &TransformationMatrix| {
            let [[a, b], [c, d]] = t.rotation_scale;
            let [x, y] = t.translation;
            [a, b, c, d, x, y]
        };
        let (clips, clip_count) = clip_uniforms(
            &self.paint.transformation,
            &self.paint.layer_transformation,
            &self.paint.clips,
            &self.content_mask,
        );
        RetainedShapesUniforms {
            transform: m(&self.paint.transformation),
            viewport_size: viewport,
            content_mask: [
                mask.origin.x.0,
                mask.origin.y.0,
                mask.size.width.0.max(0.0),
                mask.size.height.0.max(0.0),
            ],
            opacity: self.opacity,
            lod_min_px: self.paint.lod_min_device_px,
            greek_band: self.paint.text_bar_band_px,
            layer1: m(&self.paint.layer_transformation),
            clip_count,
            _pad: 0.0,
            clips,
        }
    }
}

/// Per-draw uniforms shared by the Metal and WGSL shaders (288 bytes).
/// Transforms are row-major 2x2 then translation.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct RetainedShapesUniforms {
    /// Buffer space to device pixels.
    pub transform: [f32; 6],
    /// Target size in device pixels.
    pub viewport_size: [f32; 2],
    /// Clip: x, y, width, height in device pixels.
    pub content_mask: [f32; 4],
    /// Opacity multiplier.
    pub opacity: f32,
    /// Level-of-detail threshold in device pixels.
    pub lod_min_px: f32,
    /// Device-pixel height band in which text bars draw (unused by rects and ellipses).
    pub greek_band: [f32; 2],
    /// Layer-1 transform in buffer space.
    pub layer1: [f32; 6],
    /// How many of `clips` apply.
    pub clip_count: u32,
    #[doc(hidden)]
    pub _pad: f32,
    /// The draw's clips.
    pub clips: [RetainedClipUniform; MAX_RETAINED_CLIPS],
}

/// One clip in [`RetainedShapesUniforms`] (48 bytes): device pixels map to
/// its space by `matrix` (`[a, b, c, d]`, x' = a·x + c·y + tx) and
/// `translation`, where it is a rounded rectangle centered on `half_size`.
/// A negative `half_size` is an empty clip.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct RetainedClipUniform {
    /// Device pixels to the clip's space, 2x2 part.
    pub matrix: [f32; 4],
    /// Device pixels to the clip's space, translation.
    pub translation: [f32; 2],
    /// Half the clip's size.
    pub half_size: [f32; 2],
    /// Corner radii, clamped.
    pub corner_radii: [f32; 4],
}

/// A small scene that exercises retained shapes, for renderer tests: every
/// backend renders it and must hit the same probe colors.
#[cfg(any(test, feature = "test-support", feature = "bench-support"))]
#[doc(hidden)]
pub mod retained_shapes_test_scene {
    use super::*;
    use crate::{Scene, point, size};

    /// Target size in device pixels.
    pub const SIZE: (i32, i32) = (64, 64);

    /// A probe: device pixel and expected RGBA.
    pub struct Probe {
        /// What the probe checks.
        pub what: &'static str,
        /// Device pixel.
        pub x: u32,
        /// Device pixel.
        pub y: u32,
        /// Expected RGBA.
        pub rgba: [u8; 4],
    }

    fn rect(x: f32, y: f32, w: f32, h: f32, fill: u32) -> ShapeInstance {
        ShapeInstance {
            transform: [1.0, 0.0, 0.0, 1.0, x, y],
            width: w,
            height: h,
            fill,
            kind: ShapeInstance::RECT,
            ..Default::default()
        }
    }

    /// The buffer's instances.
    pub fn instances() -> Vec<ShapeInstance> {
        let mut moved = rect(4.0, 36.0, 12.0, 12.0, 0xffffffff);
        moved.kind |= ShapeInstance::LAYER_1;
        let mut hidden = rect(0.0, 0.0, 64.0, 64.0, 0xff00ffff);
        hidden.kind |= ShapeInstance::HIDDEN;
        vec![
            // Red square.
            rect(4.0, 4.0, 24.0, 24.0, 0xff0000ff),
            // Green ellipse with a 4 px blue stroke.
            ShapeInstance {
                transform: [1.0, 0.0, 0.0, 1.0, 36.0, 4.0],
                width: 24.0,
                height: 24.0,
                stroke_width: 4.0,
                fill: 0x00ff00ff,
                stroke: 0x0000ffff,
                kind: ShapeInstance::ELLIPSE,
                ..Default::default()
            },
            // White square on layer 1: drawn 20 px to the right.
            moved,
            // Yellow square crossing the bottom of the clip.
            rect(44.0, 52.0, 16.0, 12.0, 0xffff00ff),
            // Hidden magenta square over everything.
            hidden,
        ]
    }

    /// The scene (clip: the top 58 rows) and the probes every renderer must hit.
    pub fn scene(shapes: &RetainedShapes) -> (Scene, Vec<Probe>) {
        shapes.set_all(instances());
        let full = Bounds::new(
            point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size(ScaledPixels(SIZE.0 as f32), ScaledPixels(SIZE.1 as f32)),
        );
        let clip = Bounds::new(
            point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size(ScaledPixels(SIZE.0 as f32), ScaledPixels(58.0)),
        );
        let mut scene = Scene::default();
        scene.insert_primitive(RetainedShapesDraw {
            order: 0,
            bounds: full,
            content_mask: ContentMask::from(clip),
            shapes: shapes.clone(),
            paint: RetainedShapesPaint::new(0..5, TransformationMatrix::unit()).layer(
                TransformationMatrix {
                    rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
                    translation: [20.0, 0.0],
                },
            ),
            opacity: 1.0,
        });
        scene.finish();
        let probe = |what, x, y, rgba| Probe { what, x, y, rgba };
        let probes = vec![
            probe("rect fill", 16, 16, [255, 0, 0, 255]),
            probe("background", 32, 16, [0, 0, 0, 255]),
            probe("ellipse fill", 48, 16, [0, 255, 0, 255]),
            probe("ellipse stroke", 48, 4, [0, 0, 255, 255]),
            probe("ellipse corner stays empty", 37, 5, [0, 0, 0, 255]),
            probe("layer-1 rect moved", 30, 42, [255, 255, 255, 255]),
            probe("layer-1 rect left its slot", 8, 42, [0, 0, 0, 255]),
            probe("inside the clip", 52, 55, [255, 255, 0, 255]),
            probe("outside the clip", 52, 61, [0, 0, 0, 255]),
        ];
        (scene, probes)
    }

    /// Segment, arc, triangle and text bars (one inside the default 4-8 px band,
    /// one above it), with probes.
    pub fn kinds_scene(shapes: &RetainedShapes) -> (Scene, Vec<Probe>) {
        let at = |x: f32, y: f32| [1.0, 0.0, 0.0, 1.0, x, y];
        shapes.set_all(vec![
            ShapeInstance {
                transform: at(4.0, 8.0),
                width: 40.0,
                stroke_width: 4.0,
                stroke: 0xffffffff,
                kind: ShapeInstance::SEGMENT,
                ..Default::default()
            },
            ShapeInstance {
                transform: at(4.0, 20.0),
                width: 40.0,
                height: 16.0,
                stroke_width: 4.0,
                stroke: 0xff0000ff,
                kind: ShapeInstance::ARC,
                ..Default::default()
            },
            ShapeInstance {
                transform: at(4.0, 36.0),
                width: 20.0,
                height: 20.0,
                fill: 0x00ff00ff,
                kind: ShapeInstance::TRIANGLE,
                ..Default::default()
            },
            ShapeInstance {
                transform: at(34.0, 40.0),
                width: 24.0,
                height: 6.0,
                fill: 0xffff00ff,
                kind: ShapeInstance::TEXT_BAR,
                ..Default::default()
            },
            ShapeInstance {
                transform: at(34.0, 50.0),
                width: 24.0,
                height: 12.0,
                fill: 0x00ffffff,
                kind: ShapeInstance::TEXT_BAR,
                ..Default::default()
            },
        ]);
        let full = Bounds::new(
            point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size(ScaledPixels(SIZE.0 as f32), ScaledPixels(SIZE.1 as f32)),
        );
        let mut scene = Scene::default();
        scene.insert_primitive(RetainedShapesDraw {
            order: 0,
            bounds: full,
            content_mask: ContentMask::from(full),
            shapes: shapes.clone(),
            paint: RetainedShapesPaint::new(0..5, TransformationMatrix::unit()),
            opacity: 1.0,
        });
        scene.finish();
        let probe = |what, x, y, rgba| Probe { what, x, y, rgba };
        let probes = vec![
            probe("segment", 24, 8, [255, 255, 255, 255]),
            probe("beside the segment", 24, 13, [0, 0, 0, 255]),
            probe("arc apex", 24, 27, [255, 0, 0, 255]),
            probe("inside the arc's bend", 24, 22, [0, 0, 0, 255]),
            probe("triangle", 10, 46, [0, 255, 0, 255]),
            probe("outside the triangle", 22, 38, [0, 0, 0, 255]),
            probe("text bar in band", 46, 43, [255, 255, 0, 255]),
            probe("text bar above band", 46, 56, [0, 0, 0, 255]),
        ];
        (scene, probes)
    }

    /// A 56 x 56 rectangle with four different corner radii (0, 8, 24, and
    /// 40, which is clamped to half the side, 28), with probes inside and
    /// outside each corner.
    pub fn corners_scene(shapes: &RetainedShapes) -> (Scene, Vec<Probe>) {
        shapes.set_all(vec![ShapeInstance {
            corner_radii: [0.0, 8.0, 24.0, 40.0],
            ..rect(4.0, 4.0, 56.0, 56.0, 0xffffffff)
        }]);
        let full = Bounds::new(
            point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size(ScaledPixels(SIZE.0 as f32), ScaledPixels(SIZE.1 as f32)),
        );
        let mut scene = Scene::default();
        scene.insert_primitive(RetainedShapesDraw {
            order: 0,
            bounds: full,
            content_mask: ContentMask::from(full),
            shapes: shapes.clone(),
            paint: RetainedShapesPaint::new(0..1, TransformationMatrix::unit()),
            opacity: 1.0,
        });
        scene.finish();
        let probe = |what, x, y, rgba| Probe { what, x, y, rgba };
        const WHITE: [u8; 4] = [255, 255, 255, 255];
        const BLACK: [u8; 4] = [0, 0, 0, 255];
        let probes = vec![
            probe("top-left (0) is square", 5, 5, WHITE),
            probe("top-right (8) cuts its tip", 58, 5, BLACK),
            probe("top-right (8) keeps its arc", 55, 8, WHITE),
            probe("bottom-right (24) cuts its tip", 56, 56, BLACK),
            probe("bottom-right (24) keeps its arc", 50, 50, WHITE),
            probe("bottom-left (40) cuts its tip", 8, 54, BLACK),
            probe("bottom-left is clamped to 28", 14, 48, WHITE),
        ];
        (scene, probes)
    }

    /// Clips: a red square clipped to a rotated, rounded square (45°, side
    /// 30, radius 6, centered on (20, 32)); a green square clipped to the
    /// intersection of two rectangles, the second moved up 10 px by the
    /// layer transformation; a blue square clipped to a clip stretched 200
    /// times more across than down, whose edges stay one pixel sharp.
    /// Probes inside and outside each.
    pub fn clip_scene(shapes: &RetainedShapes) -> (Scene, Vec<Probe>) {
        shapes.set_all(vec![
            rect(0.0, 0.0, 64.0, 64.0, 0xff0000ff),
            rect(0.0, 0.0, 64.0, 64.0, 0x00ff00ff),
            rect(0.0, 0.0, 64.0, 64.0, 0x0000ffff),
        ]);
        let full = Bounds::new(
            point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size(ScaledPixels(SIZE.0 as f32), ScaledPixels(SIZE.1 as f32)),
        );
        let (sin, cos) = std::f32::consts::FRAC_PI_4.sin_cos();
        // The rotated square's own (0, 0) corner, so that its center lands on (20, 32).
        let (cx, cy) = (
            20.0 - (cos * 15.0 - sin * 15.0),
            32.0 - (sin * 15.0 + cos * 15.0),
        );
        let diamond = RetainedClip {
            transform: [cos, sin, -sin, cos, cx, cy],
            width: 30.0,
            height: 30.0,
            corner_radii: [6.0; 4],
            layer_1: false,
        };
        let right = RetainedClip {
            transform: [1.0, 0.0, 0.0, 1.0, 44.0, 0.0],
            width: 20.0,
            height: 64.0,
            ..Default::default()
        };
        let low = RetainedClip {
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 40.0],
            width: 64.0,
            height: 24.0,
            layer_1: true,
            ..Default::default()
        };
        let up = TransformationMatrix {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, -10.0],
        };
        // Stretched 100x across and 0.5x down: 10 x 4 device pixels at (2, 54).
        let stretched = RetainedClip {
            transform: [100.0, 0.0, 0.0, 0.5, 2.0, 54.0],
            width: 0.1,
            height: 8.0,
            ..Default::default()
        };
        // Two clips whose top edges both lie at y = 61.5 (the middle of row 61).
        let shared = RetainedClip {
            transform: [1.0, 0.0, 0.0, 1.0, 24.0, 61.5],
            width: 12.0,
            height: 10.0,
            ..Default::default()
        };
        let shared_too = RetainedClip {
            transform: [1.0, 0.0, 0.0, 1.0, 20.0, 61.5],
            width: 20.0,
            height: 10.0,
            ..Default::default()
        };
        let mut scene = Scene::default();
        for (order, paint) in [
            (
                0,
                RetainedShapesPaint::new(0..1, TransformationMatrix::unit()).clip(diamond),
            ),
            (
                1,
                RetainedShapesPaint::new(1..2, TransformationMatrix::unit())
                    .layer(up)
                    .clip(right)
                    .clip(low),
            ),
            (
                2,
                RetainedShapesPaint::new(2..3, TransformationMatrix::unit()).clip(stretched),
            ),
            (
                3,
                RetainedShapesPaint::new(2..3, TransformationMatrix::unit())
                    .clip(shared)
                    .clip(shared_too),
            ),
        ] {
            let bounds = paint.clip_bounds().map_or(full, |b| full.intersect(&b));
            scene.insert_primitive(RetainedShapesDraw {
                order,
                bounds,
                content_mask: ContentMask::from(full),
                shapes: shapes.clone(),
                paint,
                opacity: 1.0,
            });
        }
        scene.finish();
        let probe = |what, x, y, rgba| Probe { what, x, y, rgba };
        const BLACK: [u8; 4] = [0, 0, 0, 255];
        const RED: [u8; 4] = [255, 0, 0, 255];
        const GREEN: [u8; 4] = [0, 255, 0, 255];
        const BLUE: [u8; 4] = [0, 0, 255, 255];
        let probes = vec![
            probe("inside the rotated clip", 20, 32, RED),
            probe("inside, near its rounded tip", 35, 31, RED),
            probe("past the rounded tip, inside the sharp one", 40, 31, BLACK),
            probe("outside the rotated clip's side", 32, 20, BLACK),
            probe("a corner of the target", 2, 2, BLACK),
            probe("inside both clips, the second moved up", 54, 35, GREEN),
            probe("below the moved clip", 54, 58, BLACK),
            probe("above the moved clip", 54, 20, BLACK),
            probe("left of the first clip", 40, 45, BLACK),
            probe("inside the stretched clip", 6, 55, BLUE),
            probe("its last column: the edge is sharp across", 11, 55, BLUE),
            probe("past its right edge", 12, 55, BLACK),
            probe("its last row: the edge is sharp down", 6, 57, BLUE),
            probe("below it", 6, 58, BLACK),
            // Two clips sharing an edge halfway through a pixel: half covered, not a quarter.
            probe("a shared half-covered edge", 30, 61, [0, 0, 128, 255]),
        ];
        (scene, probes)
    }

    /// A denser scene for comparing backends pixel by pixel: `n` random shapes
    /// of every kind (rotated, stroked, translucent, some below one pixel, some on
    /// layer 1) on a 256x256 target.
    pub fn random_scene(shapes: &RetainedShapes, n: usize, seed: u64) -> (Scene, (i32, i32)) {
        let mut state = seed | 1;
        let mut unit = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        let instances = (0..n)
            .map(|i| {
                let angle = (unit() - 0.5) * 3.0;
                let (sin, cos) = angle.sin_cos();
                let tiny = i % 10 == 0;
                let (w, h) = if tiny {
                    (0.3 + unit(), 0.3 + unit())
                } else {
                    (4.0 + unit() * 56.0, 4.0 + unit() * 56.0)
                };
                let fill_rgb = ((unit() * 16777215.0) as u32) << 8;
                let stroke_rgb = ((unit() * 16777215.0) as u32) << 8;
                let (x, y) = (unit() * 256.0, unit() * 256.0);
                let radius = unit() * 10.0;
                let stroke_width = 0.5 + unit() * 4.0;
                ShapeInstance {
                    transform: [cos, sin, -sin, cos, x, y],
                    width: w,
                    height: h,
                    corner_radii: match i % 6 {
                        0 => [radius; 4],
                        3 => [radius, 0.0, radius * 2.0, radius * 0.5],
                        _ => [0.0; 4],
                    },
                    stroke_width: if i % 4 == 0 { 0.0 } else { stroke_width },
                    fill: if i % 2 == 0 {
                        fill_rgb | if i % 6 == 0 { 120 } else { 255 }
                    } else {
                        0
                    },
                    stroke: stroke_rgb | 255,
                    kind: [
                        ShapeInstance::RECT,
                        ShapeInstance::ELLIPSE,
                        ShapeInstance::SEGMENT,
                        ShapeInstance::ARC,
                        ShapeInstance::TRIANGLE,
                        ShapeInstance::TEXT_BAR,
                    ][(unit() * 5.999) as usize]
                        | if i % 7 == 0 {
                            ShapeInstance::LAYER_1
                        } else {
                            0
                        },
                }
            })
            .collect();
        shapes.set_all(instances);
        let full = Bounds::new(
            point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size(ScaledPixels(256.0), ScaledPixels(256.0)),
        );
        let layer = TransformationMatrix {
            rotation_scale: [[0.9, 0.2], [-0.2, 0.9]],
            translation: [12.0, -7.0],
        };
        // The second half draws inside a rotated rounded clip and a clip on layer 1.
        let (sin, cos) = 0.4f32.sin_cos();
        let tilted = RetainedClip {
            transform: [cos, sin, -sin, cos, 70.0, 20.0],
            width: 170.0,
            height: 150.0,
            corner_radii: [40.0, 0.0, 12.0, 70.0],
            layer_1: false,
        };
        let moved = RetainedClip {
            transform: [1.0, 0.0, 0.0, 1.0, 10.0, 30.0],
            width: 200.0,
            height: 200.0,
            corner_radii: [24.0; 4],
            layer_1: true,
        };
        let half = (n / 2) as u32;
        let mut scene = Scene::default();
        for (order, paint) in [
            (
                0,
                RetainedShapesPaint::new(0..half, TransformationMatrix::unit()).layer(layer),
            ),
            (
                1,
                RetainedShapesPaint::new(half..n as u32, TransformationMatrix::unit())
                    .layer(layer)
                    .clip(tilted)
                    .clip(moved),
            ),
        ] {
            scene.insert_primitive(RetainedShapesDraw {
                order,
                bounds: full,
                content_mask: ContentMask::from(full),
                shapes: shapes.clone(),
                paint,
                opacity: 0.9,
            });
        }
        scene.finish();
        (scene, (256, 256))
    }

    /// Panics with every failing probe (channels within `tolerance`).
    pub fn check(pixel: impl Fn(u32, u32) -> [u8; 4], probes: &[Probe], tolerance: u8) {
        let failures: Vec<String> = probes
            .iter()
            .filter_map(|p| {
                let actual = pixel(p.x, p.y);
                let ok = actual
                    .iter()
                    .zip(p.rgba)
                    .all(|(a, e)| a.abs_diff(e) <= tolerance);
                (!ok).then(|| {
                    format!(
                        "{} at ({}, {}): {actual:?}, expected {:?}",
                        p.what, p.x, p.y, p.rgba
                    )
                })
            })
            .collect();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_the_shaders() {
        // 16 four-byte words: the Metal and WGSL `RetainedShape` records.
        assert_eq!(std::mem::size_of::<ShapeInstance>(), 16 * 4);
        // Six floats, float2, float4 (16-aligned at 32), two floats, float2,
        // six floats, uint, float, then four 48-byte clips at 96.
        assert_eq!(std::mem::size_of::<RetainedClipUniform>(), 48);
        assert_eq!(std::mem::size_of::<RetainedShapesUniforms>(), 96 + 4 * 48);
        assert_eq!(std::mem::offset_of!(RetainedShapesUniforms, clips), 96);
    }

    fn describe(u: RetainedUpload<'_>) -> String {
        match u {
            RetainedUpload::All(d) => format!("all {}", d.len()),
            RetainedUpload::Ranges(_, r) => format!("ranges {r:?}"),
            RetainedUpload::None => "none".into(),
        }
    }

    #[test]
    fn sync_uploads_all_then_nothing_then_ranges() {
        let shapes = RetainedShapes::new();
        shapes.set_all(vec![ShapeInstance::default(); 10]);
        let mut seen = vec![];
        let v1 = shapes.sync(0, |u| seen.push(describe(u)));
        let v2 = shapes.sync(v1, |u| seen.push(describe(u)));
        assert_eq!(v1, v2);
        shapes.write(3, &[ShapeInstance::default(); 2]);
        shapes.write(5, &[ShapeInstance::default()]);
        shapes.update(|s| {
            s[8].width = 1.0;
            8..9
        });
        let v3 = shapes.sync(v2, |u| seen.push(describe(u)));
        assert!(v3 > v2);
        assert_eq!(seen, ["all 10", "none", "ranges [3..6, 8..9]"]);
    }

    #[test]
    fn a_second_renderer_behind_the_ranges_gets_everything() {
        let shapes = RetainedShapes::new();
        shapes.set_all(vec![ShapeInstance::default(); 4]);
        let a = shapes.sync(0, |_| {});
        let b = shapes.sync(0, |_| {});
        assert_eq!(a, b);
        shapes.write(1, &[ShapeInstance::default()]);
        let mut first = String::new();
        shapes.sync(a, |u| first = describe(u));
        assert_eq!(first, "ranges [1..2]");
        // Window A consumed the ranges; window B, at the same old version, cannot
        // use them any more.
        let mut second = String::new();
        shapes.sync(b, |u| second = describe(u));
        assert_eq!(second, "all 4");
    }

    #[test]
    fn scattered_writes_coalesce_past_64_ranges() {
        let shapes = RetainedShapes::new();
        shapes.set_all(vec![ShapeInstance::default(); 1000]);
        let v = shapes.sync(0, |_| {});
        for i in 0..100 {
            shapes.write(i * 10, &[ShapeInstance::default()]);
        }
        shapes.sync(v, |u| match u {
            RetainedUpload::Ranges(_, r) => {
                assert!(r.len() <= 64);
                assert_eq!(r.first().unwrap().start, 0);
                assert_eq!(r.last().unwrap().end, 991);
            }
            _ => panic!("expected ranges"),
        });
    }

    #[test]
    fn clip_uniforms_map_device_pixels_into_the_clip() {
        let scale = TransformationMatrix::unit().scale(crate::size(2.0, 2.0));
        let layer = TransformationMatrix {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [5.0, 0.0],
        };
        // A quarter turn.
        let clip = RetainedClip {
            transform: [0.0, 1.0, -1.0, 0.0, 100.0, 10.0],
            width: 40.0,
            height: 20.0,
            corner_radii: [4.0, 100.0, 0.0, 0.0],
            layer_1: true,
        };
        let paint = RetainedShapesPaint::new(0..1, scale)
            .layer(layer)
            .clip(clip);
        let draw = RetainedShapesDraw {
            order: 0,
            bounds: Bounds::default(),
            content_mask: ContentMask::default(),
            shapes: RetainedShapes::new(),
            paint: paint.clone(),
            opacity: 1.0,
        };
        let u = draw.uniforms([400.0, 400.0]);
        assert_eq!(u.clip_count, 1);
        let c = u.clips[0];
        // The clip's (10, 5) is buffer (100 - 5 + 5, 10 + 10) after the layer: device (200, 40).
        let (x, y) = (200.0, 40.0);
        let local = (
            c.matrix[0] * x + c.matrix[2] * y + c.translation[0],
            c.matrix[1] * x + c.matrix[3] * y + c.translation[1],
        );
        assert!(
            (local.0 - 10.0).abs() < 1e-4 && (local.1 - 5.0).abs() < 1e-4,
            "{local:?}"
        );
        assert_eq!(c.half_size, [20.0, 10.0]);
        assert_eq!(
            c.corner_radii,
            [4.0, 10.0, 0.0, 0.0],
            "clamped to half the shorter side"
        );
        // Bounds: the rotated 40 x 20 box at buffer (80..100, 10..50), moved 5 right, scaled by 2, plus a pixel.
        let b = paint.clip_bounds().unwrap();
        assert_eq!(
            (b.origin.x.0, b.origin.y.0, b.size.width.0, b.size.height.0),
            (169.0, 19.0, 42.0, 82.0)
        );
        // No clips, no bounds; a clip with no area hides everything.
        assert!(
            RetainedShapesPaint::new(0..1, scale)
                .clip_bounds()
                .is_none()
        );
        let flat = RetainedShapesDraw {
            paint: RetainedShapesPaint::new(0..1, scale).clip(RetainedClip {
                transform: [0.0; 6],
                width: 1.0,
                height: 1.0,
                ..Default::default()
            }),
            ..draw
        };
        assert!(flat.uniforms([1.0, 1.0]).clips[0].half_size[0] < 0.0);
    }

    #[test]
    fn a_rounded_content_mask_is_one_more_clip_in_device_pixels() {
        use crate::{Corners, point, size};
        let mask = ContentMask {
            bounds: Bounds::new(
                point(ScaledPixels(10.0), ScaledPixels(20.0)),
                size(ScaledPixels(40.0), ScaledPixels(30.0)),
            ),
            corner_radii: Corners {
                top_left: ScaledPixels(8.0),
                top_right: ScaledPixels(0.0),
                bottom_right: ScaledPixels(40.0),
                bottom_left: ScaledPixels(2.0),
            },
        };
        let draw = RetainedShapesDraw {
            order: 0,
            bounds: Bounds::default(),
            content_mask: mask,
            shapes: RetainedShapes::new(),
            paint: RetainedShapesPaint::new(0..1, TransformationMatrix::unit()),
            opacity: 1.0,
        };
        let u = draw.uniforms([100.0, 100.0]);
        assert_eq!(u.clip_count, 1);
        let c = u.clips[0];
        // Device (10, 20) is the clip's (0, 0).
        assert_eq!(
            (c.matrix, c.translation),
            ([1.0, 0.0, 0.0, 1.0], [-10.0, -20.0])
        );
        assert_eq!(c.half_size, [20.0, 15.0]);
        assert_eq!(
            c.corner_radii,
            [8.0, 0.0, 15.0, 2.0],
            "clamped to half the shorter side"
        );
        // A square mask adds no clip; past the clip limit a rounded one clips square.
        let square = RetainedShapesDraw {
            content_mask: ContentMask::from(mask.bounds),
            ..draw.clone()
        };
        assert_eq!(square.uniforms([100.0, 100.0]).clip_count, 0);
        let mut paint = RetainedShapesPaint::new(0..1, TransformationMatrix::unit());
        for _ in 0..MAX_RETAINED_CLIPS {
            paint = paint.clip(RetainedClip {
                width: 1.0,
                height: 1.0,
                ..Default::default()
            });
        }
        let full = RetainedShapesDraw { paint, ..draw };
        assert_eq!(
            full.uniforms([100.0, 100.0]).clip_count as usize,
            MAX_RETAINED_CLIPS
        );
    }

    #[test]
    fn rgba8_packs_rgba() {
        let c = crate::Rgba {
            r: 1.0,
            g: 0.0,
            b: 0.5,
            a: 1.0,
        };
        assert_eq!(rgba8(c), 0xff0080ff);
    }
}
