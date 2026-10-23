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
//! touching the buffer.

use crate::{Bounds, ContentMask, DrawOrder, ScaledPixels, TransformationMatrix};
use parking_lot::Mutex;
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
/// shared with the shaders (13 four-byte words).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct ShapeInstance {
    /// `[a, b, c, d, tx, ty]`.
    pub transform: [f32; 6],
    /// Local width.
    pub width: f32,
    /// Local height.
    pub height: f32,
    /// Corner radius in local units (rectangles only).
    pub corner_radius: f32,
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
    /// A rectangle, rounded by `corner_radius`.
    pub const RECT: u32 = 0;
    /// An ellipse inscribed in the local rectangle.
    pub const ELLIPSE: u32 = 1;
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
}

impl RetainedShapesPaint {
    /// Draw `range` under `transformation`, with no layer transformation.
    pub fn new(range: Range<u32>, transformation: TransformationMatrix) -> Self {
        Self {
            range,
            transformation,
            layer_transformation: TransformationMatrix::unit(),
            lod_min_device_px: 1.5,
        }
    }

    /// Set the layer-1 transformation.
    pub fn layer(mut self, layer_transformation: TransformationMatrix) -> Self {
        self.layer_transformation = layer_transformation;
        self
    }
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
            greek_band: [0.0, 0.0],
            layer1: m(&self.paint.layer_transformation),
            _pad: [0.0; 2],
        }
    }
}

/// Per-draw uniforms shared by the Metal and WGSL shaders (96 bytes).
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
    #[doc(hidden)]
    pub _pad: [f32; 2],
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
            content_mask: ContentMask { bounds: clip },
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

    /// A denser scene for comparing backends pixel by pixel: `n` random rects
    /// and ellipses (rotated, stroked, translucent, some below one pixel, some on
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
                    corner_radius: if i % 3 == 0 { radius } else { 0.0 },
                    stroke_width: if i % 4 == 0 { 0.0 } else { stroke_width },
                    fill: if i % 2 == 0 {
                        fill_rgb | if i % 6 == 0 { 120 } else { 255 }
                    } else {
                        0
                    },
                    stroke: stroke_rgb | 255,
                    kind: (if unit() < 0.5 {
                        ShapeInstance::ELLIPSE
                    } else {
                        ShapeInstance::RECT
                    }) | if i % 7 == 0 {
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
        let mut scene = Scene::default();
        scene.insert_primitive(RetainedShapesDraw {
            order: 0,
            bounds: full,
            content_mask: ContentMask { bounds: full },
            shapes: shapes.clone(),
            paint: RetainedShapesPaint::new(0..n as u32, TransformationMatrix::unit()).layer(
                TransformationMatrix {
                    rotation_scale: [[0.9, 0.2], [-0.2, 0.9]],
                    translation: [12.0, -7.0],
                },
            ),
            opacity: 0.9,
        });
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
        // 13 four-byte words: the Metal and WGSL `RetainedShape` records.
        assert_eq!(std::mem::size_of::<ShapeInstance>(), 13 * 4);
        // Six floats, float2, float4 (16-aligned at 32), two floats, float2,
        // six floats, float2.
        assert_eq!(std::mem::size_of::<RetainedShapesUniforms>(), 96);
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
