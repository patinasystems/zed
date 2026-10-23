//! Retained, GPU-resident shape instances.
//!
//! A [`RetainedShapes`] holds a list of [`ShapeInstance`]s (rectangles, rounded
//! rectangles and ellipses with a fill and a stroke) in a space of the caller's
//! choosing (for example a canvas page). The renderer keeps a GPU copy keyed by
//! the buffer's id and re-uploads only the ranges written since its last upload.
//! Each frame the caller paints a range of it with
//! [`Window::paint_retained_shapes`](crate::Window::paint_retained_shapes) and a
//! transformation from that space to the window, so panning and zooming upload
//! nothing.

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
/// buffer's space: `x' = a·x + c·y + tx`, `y' = b·x + d·y + ty`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct ShapeInstance {
    /// `[a, b, c, d, tx, ty]`.
    pub transform: [f32; 6],
    /// Local size.
    pub width: f32,
    /// Local size.
    pub height: f32,
    /// Corner radius in local units (rectangles only).
    pub corner_radius: f32,
    /// Stroke width in local units, centered on the outline. 0 draws no stroke.
    pub stroke_width: f32,
    /// Fill color, `0xRRGGBBAA`. Alpha 0 draws no fill.
    pub fill: u32,
    /// Stroke color, `0xRRGGBBAA`.
    pub stroke: u32,
    /// Low 8 bits: [`ShapeInstance::RECT`] or [`ShapeInstance::ELLIPSE`].
    /// Bit 8: [`ShapeInstance::HIDDEN`]. Bits 16..24: layer; layer 1 is drawn
    /// through the draw's `layer_transform` (a live drag), layer 0 without it.
    pub kind: u32,
}

impl ShapeInstance {
    /// A rectangle (rounded by `corner_radius`).
    pub const RECT: u32 = 0;
    /// An ellipse inscribed in the local rectangle.
    pub const ELLIPSE: u32 = 1;
    /// Not drawn (keeps its slot so ranges stay stable).
    pub const HIDDEN: u32 = 1 << 8;
    /// Layer 1: moves with the draw's layer transform.
    pub const LAYER_1: u32 = 1 << 16;
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// The CPU side of a retained instance buffer. Cloning shares the buffer.
#[derive(Clone)]
pub struct RetainedShapes {
    id: u64,
    inner: Arc<Mutex<Inner>>,
}

struct Inner {
    instances: Vec<ShapeInstance>,
    /// Bumps on every write.
    version: u64,
    /// Ranges written since `dirty_since`. Empty with `version == dirty_since`
    /// means a renderer that uploaded `dirty_since` is current.
    dirty: Vec<Range<usize>>,
    dirty_since: u64,
}

/// What a renderer must copy to bring its GPU copy (uploaded at `have`) current.
pub enum RetainedUpload<'a> {
    /// Already current.
    None,
    /// Copy these ranges of `instances` (the length is unchanged).
    Ranges(&'a [ShapeInstance], &'a [Range<usize>]),
    /// Replace everything.
    All(&'a [ShapeInstance]),
}

impl Default for RetainedShapes {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for RetainedShapes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedShapes").field("id", &self.id).finish()
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

    /// No instances.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Replace every instance (a full upload).
    pub fn set_all(&self, instances: Vec<ShapeInstance>) {
        let mut inner = self.inner.lock();
        inner.instances = instances;
        inner.version += 1;
        inner.dirty.clear();
        // No renderer can be at this version, so the next upload is full.
        inner.dirty_since = 0;
    }

    /// Overwrite `values.len()` instances starting at `start` (a range upload).
    /// Panics if the range is out of bounds.
    pub fn write(&self, start: usize, values: &[ShapeInstance]) {
        if values.is_empty() {
            return;
        }
        let mut inner = self.inner.lock();
        inner.instances[start..start + values.len()].copy_from_slice(values);
        inner.version += 1;
        let range = start..start + values.len();
        // Merge with the last range when adjacent or overlapping, so a run of
        // single writes in order stays one copy.
        match inner.dirty.last_mut() {
            Some(last) if range.start <= last.end && range.end >= last.start => {
                last.start = last.start.min(range.start);
                last.end = last.end.max(range.end);
            }
            _ => inner.dirty.push(range),
        }
    }

    /// Edit instances in place; `f` returns the range it changed.
    pub fn update(&self, f: impl FnOnce(&mut [ShapeInstance]) -> Range<usize>) {
        let mut inner = self.inner.lock();
        let range = f(&mut inner.instances);
        if range.is_empty() {
            return;
        }
        inner.version += 1;
        inner.dirty.push(range);
    }

    /// Called by a renderer whose GPU copy is at version `have` (0 for none):
    /// `upload` gets what to copy, and afterwards the renderer is current.
    /// Returns the version the renderer now has.
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

/// A draw of `range` of a [`RetainedShapes`] buffer, as recorded in the scene.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct RetainedShapesDraw {
    pub order: crate::DrawOrder,
    /// Window area the draw covers (device pixels), for ordering and culling.
    pub bounds: crate::Bounds<crate::ScaledPixels>,
    pub content_mask: crate::ContentMask<crate::ScaledPixels>,
    pub shapes: RetainedShapes,
    pub range: Range<u32>,
    /// Buffer space to device pixels.
    pub transformation: crate::TransformationMatrix,
    /// Opacity multiplier for the whole draw.
    pub opacity: f32,
    /// Applied in buffer space to instances on layer 1 (before `transformation`).
    pub layer_transform: crate::TransformationMatrix,
}

/// Per-draw uniforms shared by the Metal and WGSL shaders. Must match
/// `RetainedShapesUniforms` in both shader sources.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
#[allow(missing_docs)]
pub struct RetainedShapesUniforms {
    /// Row-major 2x2 then translation: buffer space to device pixels.
    pub transform: [f32; 6],
    pub viewport_size: [f32; 2],
    /// x, y, width, height in device pixels.
    pub content_mask: [f32; 4],
    pub opacity: f32,
    /// Below this many device pixels on the longer side, a shape is drawn as a
    /// solid quad of its dominant color (no SDF).
    pub lod_min_px: f32,
    pub _pad: [f32; 2],
    /// Layer 1 transform in buffer space, row-major 2x2 then translation.
    pub layer1: [f32; 6],
    pub _pad2: [f32; 2],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_the_shaders() {
        // 13 four-byte words: the Metal and WGSL `RetainedShape` records.
        assert_eq!(std::mem::size_of::<ShapeInstance>(), 13 * 4);
        // Metal: float m[6], float2, float4 (16-aligned at 32), float, float, float2.
        assert_eq!(std::mem::size_of::<RetainedShapesUniforms>(), 96);
    }

    #[test]
    fn sync_uploads_all_then_ranges_then_nothing() {
        let shapes = RetainedShapes::new();
        shapes.set_all(vec![ShapeInstance::default(); 10]);
        let mut seen = vec![];
        let v = shapes.sync(0, |u| {
            seen.push(match u {
                RetainedUpload::All(d) => format!("all {}", d.len()),
                RetainedUpload::Ranges(_, r) => format!("ranges {r:?}"),
                RetainedUpload::None => "none".into(),
            })
        });
        let v2 = shapes.sync(v, |u| assert!(matches!(u, RetainedUpload::None)));
        assert_eq!(v, v2);
        shapes.write(3, &[ShapeInstance::default(); 2]);
        shapes.write(5, &[ShapeInstance::default()]);
        let v3 = shapes.sync(v2, |u| match u {
            RetainedUpload::Ranges(_, r) => assert_eq!(r, &[3..6]),
            _ => panic!("expected ranges"),
        });
        assert!(v3 > v2);
        // A renderer that missed a version gets everything.
        shapes.write(0, &[ShapeInstance::default()]);
        shapes.sync(v2, |u| assert!(matches!(u, RetainedUpload::All(_))));
        assert_eq!(seen, ["all 10"]);
    }
}
