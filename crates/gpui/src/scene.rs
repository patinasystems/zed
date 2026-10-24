// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, Hsla, Pixels,
    Point, Radians, RetainedShapesDraw, ScaledPixels, Size, bounds_tree::BoundsTree, point,
};
use std::{
    fmt::Debug,
    iter::Peekable,
    ops::{Add, Range, Sub},
    slice,
};

#[allow(non_camel_case_types, unused)]
#[expect(missing_docs)]
pub type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

#[expect(missing_docs)]
pub type DrawOrder = u32;

/// A boolean stored as a `u32` so that GPU-facing structs contain no
/// compiler-inserted padding bytes, which would be undefined behavior to
/// reinterpret as `&[u8]` when writing instance buffers. Guaranteed to be
/// `0` or `1` by construction; shaders read it as a `u32`/`uint`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PaddedBool32(u32);

impl From<bool> for PaddedBool32 {
    fn from(value: bool) -> Self {
        PaddedBool32(value as u32)
    }
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    /// Each open layer's draw order and bounds.
    layer_stack: Vec<(DrawOrder, Bounds<ScaledPixels>)>,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub paths: Vec<Path<ScaledPixels>>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    pub retained_shapes: Vec<RetainedShapesDraw>,
    pub backdrop_filters: Vec<BackdropFilter>,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.retained_shapes.clear();
        self.backdrop_filters.clear();
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push((order, bounds));
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let mut primitive = primitive.into();
        let mut clipped_bounds = primitive
            .transformed_bounds()
            .intersect(&primitive.content_mask().bounds);
        // A backdrop filter draws inside its mask but reads three blur radii beyond it.
        if let Primitive::BackdropFilter(filter) = &primitive {
            if filter
                .bounds
                .intersect(&filter.content_mask.bounds)
                .is_empty()
            {
                return;
            }
            clipped_bounds = filter
                .sample_bounds()
                .intersect(&filter.content_mask.bounds.dilate(filter.blur_radius * 3.));
        }

        if clipped_bounds.is_empty() {
            return;
        }

        let order = match (&primitive, self.layer_stack.last_mut()) {
            // Inside a layer a filter needs its own order, after what the layer drew so
            // far; the rest of the layer then draws after the filter.
            (Primitive::BackdropFilter(_), Some(layer)) => {
                let order = self.primitive_bounds.insert(clipped_bounds);
                layer.0 = self.primitive_bounds.insert(layer.1);
                order
            }
            (_, Some(layer)) => layer.0,
            (_, None) => self.primitive_bounds.insert(clipped_bounds),
        };
        match &mut primitive {
            Primitive::Shadow(shadow) => {
                shadow.order = order;
                self.shadows.push(*shadow);
            }
            Primitive::Quad(quad) => {
                quad.order = order;
                self.quads.push(*quad);
            }
            Primitive::Path(path) => {
                path.order = order;
                path.id = PathId(self.paths.len());
                self.paths.push(path.clone());
            }
            Primitive::Underline(underline) => {
                underline.order = order;
                self.underlines.push(*underline);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.order = order;
                self.monochrome_sprites.push(*sprite);
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.order = order;
                self.subpixel_sprites.push(*sprite);
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.order = order;
                self.polychrome_sprites.push(*sprite);
            }
            Primitive::Surface(surface) => {
                surface.order = order;
                self.surfaces.push(surface.clone());
            }
            Primitive::RetainedShapes(draw) => {
                draw.order = order;
                self.retained_shapes.push(draw.clone());
            }
            Primitive::BackdropFilter(filter) => {
                filter.order = order;
                self.backdrop_filters.push(*filter);
            }
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
            }
        }
    }

    pub fn finish(&mut self) {
        self.shadows.sort_by_key(|shadow| shadow.order);
        self.quads.sort_by_key(|quad| quad.order);
        self.paths.sort_by_key(|path| path.order);
        self.underlines.sort_by_key(|underline| underline.order);
        self.monochrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.texture_id.index));
        self.subpixel_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.texture_id.index));
        self.polychrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.texture_id.index));
        self.surfaces.sort_by_key(|surface| surface.order);
        self.retained_shapes.sort_by_key(|draw| draw.order);
        self.backdrop_filters.sort_by_key(|filter| filter.order);
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        BatchIterator {
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: self.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
            retained_start: 0,
            retained_iter: self.retained_shapes.iter().peekable(),
            backdrop_filters_start: 0,
            backdrop_filters_iter: self.backdrop_filters.iter().peekable(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    // First: at an equal order, a backdrop filter reads what is behind the element
    // before the element draws.
    BackdropFilter,
    Shadow,
    #[default]
    Quad,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
    RetainedShapes,
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
    RetainedShapes(RetainedShapesDraw),
    BackdropFilter(BackdropFilter),
}

#[expect(missing_docs)]
impl Primitive {
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
            Primitive::RetainedShapes(draw) => &draw.bounds,
            Primitive::BackdropFilter(filter) => &filter.bounds,
        }
    }

    /// The bounds this primitive covers once its transformation, if any, is applied.
    ///
    /// Content masks are applied after the transformation, so culling and draw ordering
    /// must use these bounds rather than [`Primitive::bounds`].
    pub fn transformed_bounds(&self) -> Bounds<ScaledPixels> {
        match self {
            Primitive::MonochromeSprite(sprite) => {
                sprite.transformation.transform_bounds(sprite.bounds)
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.transformation.transform_bounds(sprite.bounds)
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.transformation.transform_bounds(sprite.bounds)
            }
            Primitive::Quad(quad) => quad.transformation.transform_bounds(quad.bounds),
            // It reads up to three blur radii beyond its bounds: order it after anything
            // drawn there.
            Primitive::BackdropFilter(filter) => filter.sample_bounds(),
            Primitive::Underline(underline) => {
                underline.transformation.transform_bounds(underline.bounds)
            }
            Primitive::Shadow(shadow) => {
                // Drop shadows paint up to three blur radii outside their bounds.
                let bounds = if shadow.inset != 0 {
                    shadow.element_bounds
                } else {
                    shadow.bounds.dilate(shadow.blur_radius * 3.)
                };
                shadow.transformation.transform_bounds(bounds)
            }
            primitive => *primitive.bounds(),
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
            Primitive::RetainedShapes(draw) => &draw.content_mask,
            Primitive::BackdropFilter(filter) => &filter.content_mask,
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
    retained_start: usize,
    retained_iter: Peekable<slice::Iter<'a, RetainedShapesDraw>>,
    backdrop_filters_start: usize,
    backdrop_filters_iter: Peekable<slice::Iter<'a, BackdropFilter>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
            (
                self.retained_iter.peek().map(|s| s.order),
                PrimitiveKind::RetainedShapes,
            ),
            (
                self.backdrop_filters_iter.peek().map(|f| f.order),
                PrimitiveKind::BackdropFilter,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(shadows_start..shadows_end))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(quads_start..quads_end))
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(paths_start..paths_end))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(underlines_start..underlines_end))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(surfaces_start..surfaces_end))
            }
            PrimitiveKind::RetainedShapes => {
                let start = self.retained_start;
                let mut end = start + 1;
                self.retained_iter.next();
                while self
                    .retained_iter
                    .next_if(|draw| (draw.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    end += 1;
                }
                self.retained_start = end;
                Some(PrimitiveBatch::RetainedShapes(start..end))
            }
            PrimitiveKind::BackdropFilter => {
                let start = self.backdrop_filters_start;
                let mut end = start + 1;
                self.backdrop_filters_iter.next();
                while self
                    .backdrop_filters_iter
                    .next_if(|filter| (filter.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    end += 1;
                }
                self.backdrop_filters_start = end;
                Some(PrimitiveBatch::BackdropFilters(start..end))
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
#[allow(missing_docs)]
pub enum PrimitiveBatch {
    Shadows(Range<usize>),
    Quads(Range<usize>),
    Paths(Range<usize>),
    Underlines(Range<usize>),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    SubpixelSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    Surfaces(Range<usize>),
    /// Draws of retained shape buffers, each its own draw call.
    RetainedShapes(Range<usize>),
    /// Backdrop filters, each read from and drawn back to the target in turn.
    BackdropFilters(Range<usize>),
}

impl PrimitiveBatch {
    #[expect(missing_docs)]
    pub fn label(&self) -> String {
        match self {
            Self::Shadows(range) => format!("shadows ({})", range.len()),
            Self::Quads(range) => format!("quads ({})", range.len()),
            Self::Paths(range) => format!("paths ({})", range.len()),
            Self::Underlines(range) => format!("underlines ({})", range.len()),
            Self::BackdropFilters(range) => format!("backdrop filters ({})", range.len()),
            Self::MonochromeSprites { texture_id, range } => {
                format!(
                    "monochrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::SubpixelSprites { texture_id, range } => {
                format!(
                    "subpixel sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::PolychromeSprites { texture_id, range } => {
                format!(
                    "polychrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::Surfaces(range) => format!("surfaces ({})", range.len()),
            Self::RetainedShapes(range) => format!("retained shapes ({})", range.len()),
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
    /// Applied to the quad after layout, in device pixels (see [`Window::with_element_transformation`]).
    pub transformation: TransformationMatrix,
    pub pad: [u32; 2], // align to 16 bytes (WebGL reads quads as whole texels)
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: PaddedBool32,
    /// Applied to the underline after layout, in device pixels.
    pub transformation: TransformationMatrix,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub element_bounds: Bounds<ScaledPixels>,
    pub element_corner_radii: Corners<ScaledPixels>,
    /// 0 = drop shadow (rendered outside the element), 1 = inset shadow (rendered inside).
    pub inset: u32,
    pub pad: u32, // align to 8 bytes
    /// Applied to the shadow after layout, in device pixels.
    pub transformation: TransformationMatrix,
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// Filters what is already drawn behind `bounds` (CSS `backdrop-filter`): a gaussian
/// blur, then saturation, brightness and contrast, drawn back inside `bounds` with
/// `corner_radii`. Renderers read the target, so everything behind must be drawn
/// first; the scene orders it after anything within three blur radii.
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct BackdropFilter {
    pub order: DrawOrder,
    /// The element's opacity: 0 leaves the backdrop as it is.
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    /// Gaussian standard deviation, in device pixels.
    pub blur_radius: ScaledPixels,
    /// 1 leaves colors as they are; 0 is grayscale.
    pub saturate: f32,
    /// A multiplier.
    pub brightness: f32,
    /// 1 leaves colors as they are; 0 is mid gray.
    pub contrast: f32,
}

impl Default for BackdropFilter {
    /// A filter that changes nothing (no blur, neutral color controls, fully opaque).
    fn default() -> Self {
        Self {
            order: 0,
            opacity: 1.,
            bounds: Bounds::default(),
            content_mask: ContentMask::default(),
            corner_radii: Corners::default(),
            blur_radius: ScaledPixels(0.),
            saturate: 1.,
            brightness: 1.,
            contrast: 1.,
        }
    }
}

impl BackdropFilter {
    /// What the filter reads: its bounds and three blur radii around them.
    pub fn sample_bounds(&self) -> Bounds<ScaledPixels> {
        self.bounds.dilate(self.blur_radius * 3.)
    }
}

impl From<BackdropFilter> for Primitive {
    fn from(filter: BackdropFilter) -> Self {
        Primitive::BackdropFilter(filter)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(mut self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Returns the smallest axis-aligned bounds that contain `bounds` after this
    /// transformation is applied to it.
    pub fn transform_bounds(&self, bounds: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
        if *self == Self::unit() {
            return bounds;
        }
        let corners = [
            bounds.origin,
            bounds.top_right(),
            bounds.bottom_right(),
            bounds.bottom_left(),
        ]
        .map(|corner| {
            let [[a, b], [c, d]] = self.rotation_scale;
            let [tx, ty] = self.translation;
            (
                a * corner.x.0 + b * corner.y.0 + tx,
                c * corner.x.0 + d * corner.y.0 + ty,
            )
        });
        let (mut min_x, mut min_y) = corners[0];
        let (mut max_x, mut max_y) = corners[0];
        for (x, y) in &corners[1..] {
            min_x = min_x.min(*x);
            min_y = min_y.min(*y);
            max_x = max_x.max(*x);
            max_y = max_y.max(*y);
        }
        Bounds::from_corners(
            point(ScaledPixels(min_x), ScaledPixels(min_y)),
            point(ScaledPixels(max_x), ScaledPixels(max_y)),
        )
    }

    /// Whether this is the unit matrix.
    pub fn is_unit(&self) -> bool {
        *self == Self::unit()
    }

    /// The inverse transformation, or `None` if this one is singular (it
    /// collapses the plane onto a line or a point).
    pub fn inverse(&self) -> Option<Self> {
        let [[a, b], [c, d]] = self.rotation_scale;
        let determinant = a * d - b * c;
        if determinant == 0. || !determinant.is_finite() {
            return None;
        }
        let rotation_scale = [
            [d / determinant, -b / determinant],
            [-c / determinant, a / determinant],
        ];
        let [tx, ty] = self.translation;
        Some(Self {
            rotation_scale,
            translation: [
                -(rotation_scale[0][0] * tx + rotation_scale[0][1] * ty),
                -(rotation_scale[1][0] * tx + rotation_scale[1][1] * ty),
            ],
        })
    }

    /// The same transformation for coordinates scaled by `scale_factor`
    /// (logical to device pixels): the translation scales, the rest does not.
    pub fn scale_translation(mut self, scale_factor: f32) -> Self {
        self.translation[0] *= scale_factor;
        self.translation[1] *= scale_factor;
        self
    }

    /// Apply transformation to a point in device pixels.
    pub fn apply_scaled(&self, point: Point<ScaledPixels>) -> Point<ScaledPixels> {
        let applied = self.apply(Point::new(Pixels(point.x.0), Pixels(point.y.0)));
        Point::new(ScaledPixels(applied.x.0), ScaledPixels(applied.y.0))
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub tile: AtlasTile,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub transformation: TransformationMatrix,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub tile: AtlasTile,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub tile: AtlasTile,
    pub grayscale: PaddedBool32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub transformation: TransformationMatrix,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub image_buffer: objc2_core_foundation::CFRetained<objc2_core_video::CVPixelBuffer>,
}

impl From<RetainedShapesDraw> for Primitive {
    fn from(draw: RetainedShapesDraw) -> Self {
        Primitive::RetainedShapes(draw)
    }
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[expect(missing_docs)]
pub struct PathId(pub usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub id: PathId,
    pub order: DrawOrder,
    pub bounds: Bounds<P>,
    pub content_mask: ContentMask<P>,
    pub vertices: Vec<PathVertex<P>>,
    pub color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl Path<ScaledPixels> {
    /// Applies `transformation` to every vertex; the bounds become the box that covers the
    /// transformed path.
    pub fn transform(&mut self, transformation: &TransformationMatrix) {
        if transformation.is_unit() {
            return;
        }
        let apply = |p: Point<ScaledPixels>| transformation.apply_scaled(p);
        for vertex in &mut self.vertices {
            vertex.xy_position = apply(vertex.xy_position);
        }
        self.start = apply(self.start);
        self.current = apply(self.current);
        self.bounds = transformation.transform_bounds(self.bounds);
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    #[expect(missing_docs)]
    pub fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub xy_position: Point<P>,
    pub st_position: Point<f32>,
    pub content_mask: ContentMask<P>,
}

#[expect(missing_docs)]
impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
    }
}

/// Quads, borders, shadows and underlines painted under element transformations, for
/// renderer tests: every backend renders it and must hit the same probe colors. Each
/// probe fails if its primitive is drawn untransformed.
#[cfg(any(test, feature = "test-support", feature = "bench-support"))]
#[doc(hidden)]
pub mod element_transformation_test_scene {
    use super::*;
    pub use crate::retained_shapes_test_scene::{Probe, SIZE, check};
    use crate::{Rgba, radians, size};

    fn color(r: f32, g: f32, b: f32) -> Hsla {
        Rgba { r, g, b, a: 1.0 }.into()
    }

    fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(w), ScaledPixels(h)),
        )
    }

    /// Rotates by `degrees` (clockwise on screen) and scales by `scale` about `(x, y)`.
    pub fn about(x: f32, y: f32, degrees: f32, scale: f32) -> TransformationMatrix {
        let pivot = point(ScaledPixels(x), ScaledPixels(y));
        TransformationMatrix::unit()
            .translate(pivot)
            .rotate(radians(degrees.to_radians()))
            .scale(size(scale, scale))
            .translate(-pivot)
    }

    /// The scene and the probes every renderer must hit.
    pub fn scene() -> (Scene, Vec<Probe>) {
        let full = ContentMask {
            bounds: bounds(0., 0., SIZE.0 as f32, SIZE.1 as f32),
        };
        let mut scene = Scene::default();
        // A red square turned 45 degrees: a diamond reaching 11.3 px from its center.
        scene.insert_primitive(Quad {
            bounds: bounds(8., 8., 16., 16.),
            content_mask: full,
            background: color(1., 0., 0.).into(),
            transformation: about(16., 16., 45., 1.),
            ..Default::default()
        });
        // A blue square with a 3 px green border, scaled 1.5x about its center: 36..60
        // with a 4.5 px border.
        scene.insert_primitive(Quad {
            bounds: bounds(40., 8., 16., 16.),
            content_mask: full,
            background: color(0., 0., 1.).into(),
            border_color: color(0., 1., 0.),
            border_widths: Edges::all(ScaledPixels(3.)),
            transformation: about(48., 16., 0., 1.5),
            ..Default::default()
        });
        // A white underline turned upright: x 15..17, y 33..49.
        scene.insert_primitive(Underline {
            order: 0,
            pad: 0,
            bounds: bounds(8., 40., 16., 2.),
            content_mask: full,
            color: color(1., 1., 1.),
            thickness: ScaledPixels(2.),
            wavy: false.into(),
            transformation: about(16., 41., 90., 1.),
        });
        // A sharp yellow drop shadow turned 45 degrees: a diamond reaching 8.5 px.
        scene.insert_primitive(Shadow {
            order: 0,
            blur_radius: ScaledPixels(0.),
            bounds: bounds(40., 40., 12., 12.),
            corner_radii: Corners::default(),
            content_mask: full,
            color: color(1., 1., 0.),
            element_bounds: bounds(40., 40., 12., 12.),
            element_corner_radii: Corners::default(),
            inset: 0,
            pad: 0,
            transformation: about(46., 46., 45., 1.),
        });
        // A magenta 4x8 quad stretched 4x wide and squeezed to half height: x 22..38,
        // y 54..58. Its edges must stay sharp along both axes.
        scene.insert_primitive(Quad {
            bounds: bounds(28., 52., 4., 8.),
            content_mask: full,
            background: color(1., 0., 1.).into(),
            transformation: TransformationMatrix::unit()
                .translate(point(ScaledPixels(30.), ScaledPixels(56.)))
                .scale(size(4., 0.5))
                .translate(point(ScaledPixels(-30.), ScaledPixels(-56.))),
            ..Default::default()
        });
        scene.finish();
        let probe = |what, x, y, rgba| Probe { what, x, y, rgba };
        let magenta = [255, 0, 255, 255];
        let (black, red, green, blue, white, yellow) = (
            [0, 0, 0, 255],
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 255, 255, 255],
            [255, 255, 0, 255],
        );
        let probes = vec![
            probe("rotated quad center", 16, 16, red),
            probe("rotated quad tip above its box", 16, 6, red),
            probe("rotated quad leaves its corner", 9, 9, black),
            probe("scaled quad fill", 48, 16, blue),
            probe("scaled quad border", 37, 16, green),
            probe("scaled quad border past its box", 38, 22, green),
            probe("outside the scaled quad", 34, 16, black),
            probe("upright underline", 16, 35, white),
            probe("upright underline leaves its row", 10, 41, black),
            probe("rotated shadow tip", 46, 39, yellow),
            probe("rotated shadow leaves its corner", 40, 40, black),
            probe("stretched quad past its box", 24, 56, magenta),
            probe("stretched quad's edge does not blur right", 38, 56, black),
            probe("stretched quad's edge does not blur left", 21, 56, black),
            probe("squeezed quad leaves its rows", 30, 52, black),
        ];
        (scene, probes)
    }
}

/// Backdrop filters over a striped and a red backdrop, for renderer tests: every
/// backend renders it and must hit the same probe colors.
#[cfg(any(test, feature = "test-support", feature = "bench-support"))]
#[doc(hidden)]
pub mod backdrop_filter_test_scene {
    use super::*;
    pub use crate::retained_shapes_test_scene::{Probe, SIZE, check};
    use crate::{Rgba, size};

    fn color(r: f32, g: f32, b: f32) -> Hsla {
        Rgba { r, g, b, a: 1.0 }.into()
    }

    fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(w), ScaledPixels(h)),
        )
    }

    /// The scene and the probes every renderer must hit.
    pub fn scene() -> (Scene, Vec<Probe>) {
        let full = ContentMask {
            bounds: bounds(0., 0., SIZE.0 as f32, SIZE.1 as f32),
        };
        let mut scene = Scene::default();
        // White one-pixel columns on black, over the top 40 rows.
        for x in (1..SIZE.0).step_by(2) {
            scene.insert_primitive(Quad {
                bounds: bounds(x as f32, 0., 1., 40.),
                content_mask: full,
                background: color(1., 1., 1.).into(),
                ..Default::default()
            });
        }
        // A red band below.
        scene.insert_primitive(Quad {
            bounds: bounds(0., 44., SIZE.0 as f32, 20.),
            content_mask: full,
            background: color(1., 0., 0.).into(),
            ..Default::default()
        });
        // Blur the stripes inside a rounded square: they average to gray.
        scene.insert_primitive(BackdropFilter {
            bounds: bounds(8., 4., 32., 32.),
            content_mask: full,
            corner_radii: Corners::all(ScaledPixels(10.)),
            blur_radius: ScaledPixels(3.),
            opacity: 1.,
            saturate: 1.,
            brightness: 1.,
            contrast: 1.,
            ..Default::default()
        });
        // Desaturate part of the red band, darken another part, and half-fade a third
        // desaturation (element opacity 0.5).
        scene.insert_primitive(BackdropFilter {
            bounds: bounds(4., 48., 16., 12.),
            content_mask: full,
            opacity: 1.,
            saturate: 0.,
            brightness: 1.,
            contrast: 1.,
            ..Default::default()
        });
        scene.insert_primitive(BackdropFilter {
            bounds: bounds(40., 48., 16., 12.),
            content_mask: full,
            opacity: 1.,
            saturate: 1.,
            brightness: 0.5,
            contrast: 1.,
            ..Default::default()
        });
        scene.insert_primitive(BackdropFilter {
            bounds: bounds(24., 48., 12., 12.),
            content_mask: full,
            opacity: 0.5,
            saturate: 0.,
            brightness: 1.,
            contrast: 1.,
            ..Default::default()
        });
        scene.finish();
        let probe = |what, x, y, rgba| Probe { what, x, y, rgba };
        let gray = 128;
        let luma = (0.213f32 * 255.).round() as u8;
        let probes = vec![
            probe("blurred stripes average", 24, 20, [gray, gray, gray, 255]),
            probe(
                "blurred stripes average (odd column)",
                25,
                20,
                [gray, gray, gray, 255],
            ),
            probe("stripes outside stay black", 46, 20, [0, 0, 0, 255]),
            probe("stripes outside stay white", 47, 20, [255, 255, 255, 255]),
            probe("rounded corner leaves the backdrop", 8, 4, [0, 0, 0, 255]),
            probe("desaturated red", 12, 54, [luma, luma, luma, 255]),
            probe("red outside the filters", 22, 54, [255, 0, 0, 255]),
            probe(
                "half-faded desaturation",
                30,
                54,
                [((255 + luma as u16) / 2) as u8, luma / 2, luma / 2, 255],
            ),
            probe("darkened red", 48, 54, [128, 0, 0, 255]),
        ];
        (scene, probes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AtlasTextureId, AtlasTextureKind, DevicePixels, TileId, radians, size};

    fn scaled_bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(width), ScaledPixels(height)),
        )
    }

    fn rotation_about(angle: f32, pivot: Point<ScaledPixels>) -> TransformationMatrix {
        TransformationMatrix::unit()
            .translate(pivot)
            .rotate(radians(angle))
            .translate(-pivot)
    }

    fn polychrome_sprite(
        bounds: Bounds<ScaledPixels>,
        content_mask: Bounds<ScaledPixels>,
        transformation: TransformationMatrix,
    ) -> PolychromeSprite {
        PolychromeSprite {
            order: 0,
            pad: 0,
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: 0,
                    kind: AtlasTextureKind::Polychrome,
                },
                tile_id: TileId(0),
                padding: 0,
                bounds: Bounds::new(
                    point(DevicePixels(0), DevicePixels(0)),
                    size(DevicePixels(1), DevicePixels(1)),
                ),
            },
            grayscale: false.into(),
            opacity: 1.,
            bounds,
            content_mask: ContentMask {
                bounds: content_mask,
            },
            corner_radii: Corners::default(),
            transformation,
        }
    }

    fn assert_bounds_eq(actual: Bounds<ScaledPixels>, expected: Bounds<ScaledPixels>) {
        let close = |a: ScaledPixels, b: ScaledPixels| (a.0 - b.0).abs() < 1e-4;
        assert!(
            close(actual.origin.x, expected.origin.x)
                && close(actual.origin.y, expected.origin.y)
                && close(actual.size.width, expected.size.width)
                && close(actual.size.height, expected.size.height),
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn test_transform_bounds() {
        let bounds = scaled_bounds(0., 0., 20., 10.);
        assert_eq!(
            TransformationMatrix::unit().transform_bounds(bounds),
            bounds
        );

        let quarter_turn = rotation_about(
            std::f32::consts::FRAC_PI_2,
            point(ScaledPixels(10.), ScaledPixels(5.)),
        );
        assert_bounds_eq(
            quarter_turn.transform_bounds(bounds),
            scaled_bounds(5., -5., 10., 20.),
        );

        let scale_and_translate = TransformationMatrix::unit()
            .translate(point(ScaledPixels(100.), ScaledPixels(50.)))
            .scale(size(2., 3.));
        assert_bounds_eq(
            scale_and_translate.transform_bounds(bounds),
            scaled_bounds(100., 50., 40., 30.),
        );
    }

    #[test]
    fn test_insert_primitive_culls_transformed_sprites_by_transformed_bounds() {
        let content_mask = scaled_bounds(0., 0., 100., 100.);
        let move_right =
            TransformationMatrix::unit().translate(point(ScaledPixels(200.), ScaledPixels(0.)));
        let move_left =
            TransformationMatrix::unit().translate(point(ScaledPixels(-200.), ScaledPixels(0.)));

        let mut scene = Scene::default();
        // Painted outside the mask, but transformed into it.
        scene.insert_primitive(polychrome_sprite(
            scaled_bounds(-190., 10., 20., 20.),
            content_mask,
            move_right,
        ));
        // Painted inside the mask, but transformed out of it.
        scene.insert_primitive(polychrome_sprite(
            scaled_bounds(10., 10., 20., 20.),
            content_mask,
            move_left,
        ));
        scene.finish();

        assert_eq!(scene.polychrome_sprites.len(), 1);
        assert_eq!(scene.polychrome_sprites[0].transformation, move_right);
        assert_eq!(
            scene.polychrome_sprites[0].bounds,
            scaled_bounds(-190., 10., 20., 20.)
        );
    }

    #[test]
    fn backdrop_filters_inside_a_layer_keep_paint_order() {
        let mut scene = Scene::default();
        let layer = scaled_bounds(0., 0., 40., 40.);
        scene.push_layer(layer);
        scene.insert_primitive(quad(scaled_bounds(0., 0., 40., 40.)));
        scene.insert_primitive(BackdropFilter {
            bounds: scaled_bounds(5., 5., 20., 20.),
            content_mask: ContentMask { bounds: layer },
            ..Default::default()
        });
        scene.insert_primitive(quad(scaled_bounds(5., 5., 20., 20.)));
        scene.pop_layer();
        scene.finish();
        let filter = scene.backdrop_filters[0].order;
        assert!(scene.quads[0].order < filter, "{:?}", scene.quads);
        assert!(filter < scene.quads[1].order, "{:?}", scene.quads);
        assert_eq!(BackdropFilter::default().opacity, 1.);
        assert_eq!(BackdropFilter::default().brightness, 1.);
    }

    #[test]
    fn clipped_backdrop_filters_order_after_what_their_blur_reads() {
        let mut scene = Scene::default();
        // Outside the filter's mask, but within its blur reach.
        scene.insert_primitive(quad(scaled_bounds(0., 0., 10., 10.)));
        scene.insert_primitive(BackdropFilter {
            bounds: scaled_bounds(12., 0., 20., 20.),
            content_mask: ContentMask {
                bounds: scaled_bounds(12., 0., 20., 20.),
            },
            blur_radius: ScaledPixels(2.),
            ..Default::default()
        });
        scene.finish();
        assert!(scene.quads[0].order < scene.backdrop_filters[0].order);
    }

    #[test]
    fn backdrop_filters_order_after_what_they_read_and_before_what_covers_them() {
        let mut scene = Scene::default();
        // A quad near, but outside, the filter's bounds: inside its blur reach.
        scene.insert_primitive(quad(scaled_bounds(0., 0., 10., 10.)));
        let filter_bounds = scaled_bounds(14., 0., 20., 20.);
        scene.insert_primitive(BackdropFilter {
            bounds: filter_bounds,
            content_mask: ContentMask {
                bounds: scaled_bounds(-100., -100., 1000., 1000.),
            },
            blur_radius: ScaledPixels(2.),
            ..Default::default()
        });
        // The element's own background over it.
        scene.insert_primitive(quad(filter_bounds));
        scene.finish();
        let filter = scene.backdrop_filters[0].order;
        assert!(scene.quads[0].order < filter);
        assert!(filter < scene.quads[1].order);
        let kinds: Vec<_> = scene
            .batches()
            .map(|batch| match batch {
                PrimitiveBatch::Quads(_) => "quads",
                PrimitiveBatch::BackdropFilters(_) => "backdrop",
                _ => "other",
            })
            .collect();
        assert_eq!(kinds, vec!["quads", "backdrop", "quads"]);
    }

    fn quad(bounds: Bounds<ScaledPixels>) -> Quad {
        Quad {
            bounds,
            content_mask: ContentMask { bounds },
            ..Default::default()
        }
    }

    fn retained(
        bounds: Bounds<ScaledPixels>,
        shapes: &crate::RetainedShapes,
    ) -> RetainedShapesDraw {
        RetainedShapesDraw {
            order: 0,
            bounds,
            content_mask: ContentMask { bounds },
            shapes: shapes.clone(),
            paint: crate::RetainedShapesPaint::new(0..1, TransformationMatrix::unit()),
            opacity: 1.,
        }
    }

    #[test]
    fn test_retained_shapes_order_between_overlapping_primitives() {
        let shapes = crate::RetainedShapes::new();
        let canvas = scaled_bounds(0., 0., 100., 100.);
        let mut scene = Scene::default();
        scene.insert_primitive(quad(scaled_bounds(10., 10., 10., 10.)));
        scene.insert_primitive(retained(canvas, &shapes));
        scene.insert_primitive(quad(scaled_bounds(10., 10., 10., 10.)));
        scene.insert_primitive(retained(canvas, &shapes));
        scene.finish();
        let batches: Vec<String> = scene.batches().map(|b| b.label()).collect();
        assert_eq!(
            batches,
            [
                "quads (1)",
                "retained shapes (1)",
                "quads (1)",
                "retained shapes (1)"
            ]
        );
    }

    #[test]
    fn test_retained_shapes_outside_the_mask_are_culled() {
        let shapes = crate::RetainedShapes::new();
        let mut scene = Scene::default();
        let mut draw = retained(scaled_bounds(0., 0., 10., 10.), &shapes);
        draw.content_mask.bounds = scaled_bounds(50., 50., 10., 10.);
        scene.insert_primitive(draw);
        scene.finish();
        assert!(scene.retained_shapes.is_empty());
    }
}
