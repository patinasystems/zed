//! The records renderers upload for a scene's primitives, and the per-scene table they index.
//!
//! A [`Quad`] or a sprite carries everything about itself: its content mask, its transformation,
//! elliptical radii, borders and gradient stops, inline. Most of that is the same for most
//! primitives or absent: a frame has a few distinct masks, most primitives are untransformed,
//! circular, solid and unbordered. So renderers upload a compact record per primitive and one
//! table of 16-byte texels per scene ([`InstanceTables`]) that the records index:
//!
//! - each distinct content mask once, as three texels (bounds, corner radii, rounded bounds),
//!   and a fourth, its vertical corner radii, when a corner is elliptical: the index a record
//!   holds then has [`MASK_ELLIPTICAL`] set, so circular masks cost no more to store or load;
//! - each distinct transformation once, as two texels, the identity at texel 0;
//! - per primitive, only what it uses beyond the common case (its "aux" record): a border, the
//!   vertical radii of elliptical corners, superellipse corner shapes, a background other than a
//!   solid color, and the per-path fill of a path.
//!
//! Shaders expand a record back into the primitive's full layout (`load_quad` and friends), so
//! everything after the load reads the same values as before.

use crate::{
    Background, BackgroundTag, Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla,
    MonochromeSprite, Path, Point, PolychromeSprite, Quad, ScaledPixels, Shadow, SubpixelSprite,
    TransformationMatrix, Underline,
};
use collections::FxHashMap;

/// A 16-byte table entry: four 32-bit words, floats or (bit-cast) unsigned integers.
pub type Texel = [f32; 4];

/// [`QuadRecord::flags`]: the border is dashed ([`crate::BorderStyle::Dashed`]).
pub const QUAD_DASHED: u32 = 1;
/// [`QuadRecord::flags`] (and a path's flags): the background tag ([`BackgroundTag`]) in bits 1–3.
pub const BACKGROUND_TAG_SHIFT: u32 = 1;
/// [`QuadRecord::flags`] (and a path's flags): the background interpolates in Oklab.
pub const BACKGROUND_OKLAB: u32 = 1 << 4;
/// [`QuadRecord::flags`]: the aux record starts with the border: widths, then color.
pub const QUAD_BORDER: u32 = 1 << 5;
/// [`QuadRecord::flags`]: the aux record has the corners' vertical radii next.
pub const QUAD_ELLIPTICAL: u32 = 1 << 6;
/// [`QuadRecord::flags`]: the aux record has the corner shapes next ([`Quad::corner_shapes`]).
pub const QUAD_SHAPED: u32 = 1 << 7;

/// A [`Quad`] as renderers upload it (64 bytes).
///
/// After the aux record's optional border (2 texels), vertical radii (1) and corner shapes (1), a
/// background other than [`BackgroundTag::Solid`] has its block (see
/// [`InstanceTables::background_block`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct QuadRecord {
    pub bounds: Bounds<ScaledPixels>,
    /// Each corner's horizontal radius (and vertical one, unless [`QUAD_ELLIPTICAL`]).
    pub corner_radii: Corners<ScaledPixels>,
    /// The background's solid color.
    pub background: Hsla,
    /// [`QUAD_DASHED`], the background tag and color space, and which parts the aux record has.
    pub flags: u32,
    /// The first texel of the content mask.
    pub content_mask: u32,
    /// The first texel of the transformation; 0 is the identity.
    pub transformation: u32,
    /// The first texel of the aux record; unused when `flags` names no part of it.
    pub aux: u32,
}

/// A [`Shadow`] as renderers upload it (96 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct ShadowRecord {
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub color: Hsla,
    pub element_bounds: Bounds<ScaledPixels>,
    pub element_corner_radii: Corners<ScaledPixels>,
    pub blur_radius: ScaledPixels,
    pub content_mask: u32,
    pub transformation: u32,
    /// Bit 0: an inset shadow. Bits 1 and up: the first texel of the vertical radii (the shadow's,
    /// then the element's), or 0 when both are circular.
    pub info: u32,
}

/// An [`Underline`] as renderers upload it (48 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct UnderlineRecord {
    pub bounds: Bounds<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    /// Nonzero for a wavy underline.
    pub wavy: u32,
    pub content_mask: u32,
    pub transformation: u32,
}

/// A [`MonochromeSprite`] or [`SubpixelSprite`] as renderers upload it (64 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SpriteRecord {
    /// The tile's bounds within its atlas texture.
    pub tile_bounds: Bounds<DevicePixels>,
    pub bounds: Bounds<ScaledPixels>,
    pub color: Hsla,
    pub content_mask: u32,
    pub transformation: u32,
    pub pad: [u32; 2],
}

/// A [`PolychromeSprite`] as renderers upload it (64 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSpriteRecord {
    /// The tile's bounds within its atlas texture.
    pub tile_bounds: Bounds<DevicePixels>,
    pub bounds: Bounds<ScaledPixels>,
    /// Each corner's horizontal radius (and vertical one, unless `info` names vertical radii).
    pub corner_radii: Corners<ScaledPixels>,
    pub opacity: f32,
    pub content_mask: u32,
    pub transformation: u32,
    /// Bit 0: [`PolychromeSprite::exact_scale`]. Bit 1: grayscale. Bits 2 and up: the texel of
    /// the vertical radii, or 0 when the corners are circular.
    pub info: u32,
}

/// A path vertex as renderers upload it (24 bytes): its positions, its path's record, whose
/// texels are the clipped bounds, the background's solid color, then the background tag and color
/// space (bits as in [`QuadRecord::flags`]), then a background block for a background other than
/// a solid color; and the path's content mask. Table texels never hold table indices, so only
/// records need [`RebaseRecord`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathVertexRecord {
    pub xy_position: Point<ScaledPixels>,
    pub st_position: Point<f32>,
    /// The first texel of the path's record.
    pub path: u32,
    pub content_mask: u32,
}

/// A path's place in the table, which its vertices share ([`InstanceTables::path`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[expect(missing_docs)]
pub struct PathRecord {
    pub texel: u32,
    pub content_mask: u32,
}

/// A record's table indices, which a renderer whose shaders index the table absolutely (WebGL's
/// instance texture) offsets by the table's first texel.
pub trait RebaseRecord {
    /// Adds `base` to every table index the record holds.
    fn rebase(&mut self, base: u32);
}

impl RebaseRecord for QuadRecord {
    fn rebase(&mut self, base: u32) {
        self.content_mask += base;
        self.transformation += base;
        self.aux += base;
    }
}

impl RebaseRecord for ShadowRecord {
    fn rebase(&mut self, base: u32) {
        self.content_mask += base;
        self.transformation += base;
        if self.info >> 1 != 0 {
            self.info += base << 1;
        }
    }
}

impl RebaseRecord for UnderlineRecord {
    fn rebase(&mut self, base: u32) {
        self.content_mask += base;
        self.transformation += base;
    }
}

impl RebaseRecord for SpriteRecord {
    fn rebase(&mut self, base: u32) {
        self.content_mask += base;
        self.transformation += base;
    }
}

impl RebaseRecord for PolychromeSpriteRecord {
    fn rebase(&mut self, base: u32) {
        self.content_mask += base;
        self.transformation += base;
        if self.info >> 2 != 0 {
            self.info += base << 2;
        }
    }
}

impl RebaseRecord for PathVertexRecord {
    fn rebase(&mut self, base: u32) {
        self.path += base;
        self.content_mask += base;
    }
}

impl RebaseRecord for PathRecord {
    fn rebase(&mut self, base: u32) {
        self.texel += base;
        self.content_mask += base;
    }
}

/// A scene's table of texels: its distinct content masks and transformations, and its
/// primitives' aux records. Renderers keep one, encode a scene's primitives through it, upload
/// [`InstanceTables::texels`] beside the records and [`clear`](Self::clear) it for the next scene.
pub struct InstanceTables {
    texels: Vec<Texel>,
    masks: FxHashMap<[u32; 16], u32>,
    last_mask: Option<([u32; 16], u32)>,
    transforms: FxHashMap<[u32; 6], u32>,
}

impl Default for InstanceTables {
    fn default() -> Self {
        let mut tables = Self {
            texels: Vec::new(),
            masks: FxHashMap::default(),
            last_mask: None,
            transforms: FxHashMap::default(),
        };
        tables.clear();
        tables
    }
}

fn bits(value: u32) -> f32 {
    f32::from_bits(value)
}

fn bounds_texel(bounds: &Bounds<ScaledPixels>) -> Texel {
    [
        bounds.origin.x.0,
        bounds.origin.y.0,
        bounds.size.width.0,
        bounds.size.height.0,
    ]
}

fn corners_texel(corners: &Corners<ScaledPixels>) -> Texel {
    [
        corners.top_left.0,
        corners.top_right.0,
        corners.bottom_right.0,
        corners.bottom_left.0,
    ]
}

fn edges_texel(edges: &Edges<ScaledPixels>) -> Texel {
    [edges.top.0, edges.right.0, edges.bottom.0, edges.left.0]
}

fn hsla_texel(color: &Hsla) -> Texel {
    [color.h, color.s, color.l, color.a]
}

/// Set in a record's content mask index when the mask has a fourth texel, its vertical corner
/// radii (an elliptical corner).
pub const MASK_ELLIPTICAL: u32 = 1 << 31;

fn mask_key(mask: &ContentMask<ScaledPixels>) -> [u32; 16] {
    let [a, b, c, d] = bounds_texel(&mask.bounds);
    let [e, f, g, h] = corners_texel(&mask.corner_radii);
    let [i, j, k, l] = bounds_texel(&mask.rounded_bounds);
    let [m, n, o, p] = corners_texel(&mask.corner_radii_y);
    [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p].map(f32::to_bits)
}

fn is_identity(transformation: &TransformationMatrix) -> bool {
    *transformation == TransformationMatrix::unit()
}

/// The background's tag and color space, as [`QuadRecord::flags`] holds them.
fn background_flags(background: &Background) -> u32 {
    (background.tag as u32) << BACKGROUND_TAG_SHIFT
        | if background.color_space as u32 == 1 {
            BACKGROUND_OKLAB
        } else {
            0
        }
}

impl InstanceTables {
    /// The texels encoded since the last [`clear`](Self::clear), the identity first.
    pub fn texels(&self) -> &[Texel] {
        &self.texels
    }

    /// Empties the table for another scene.
    pub fn clear(&mut self) {
        self.texels.clear();
        self.masks.clear();
        self.last_mask = None;
        self.transforms.clear();
        // The identity transformation, at texel 0.
        self.texels.push([1., 0., 0., 1.]);
        self.texels.push([0.; 4]);
    }

    fn next(&self) -> u32 {
        self.texels.len() as u32
    }

    /// The first texel of `mask`, added when the scene has not used it yet, with
    /// [`MASK_ELLIPTICAL`] set when it has a fourth texel.
    pub fn content_mask(&mut self, mask: &ContentMask<ScaledPixels>) -> u32 {
        let key = mask_key(mask);
        if let Some((last, index)) = self.last_mask
            && last == key
        {
            return index;
        }
        let elliptical = mask.corner_radii_y != Corners::default();
        let next = self.next() | if elliptical { MASK_ELLIPTICAL } else { 0 };
        let index = *self.masks.entry(key).or_insert(next);
        if index == next {
            self.texels.extend_from_slice(&[
                bounds_texel(&mask.bounds),
                corners_texel(&mask.corner_radii),
                bounds_texel(&mask.rounded_bounds),
            ]);
            if elliptical {
                self.texels.push(corners_texel(&mask.corner_radii_y));
            }
        }
        self.last_mask = Some((key, index));
        index
    }

    /// The first texel of `transformation`, added when the scene has not used it yet; 0 for the
    /// identity.
    pub fn transformation(&mut self, transformation: &TransformationMatrix) -> u32 {
        if is_identity(transformation) {
            return 0;
        }
        let [[a, b], [c, d]] = transformation.rotation_scale;
        let [tx, ty] = transformation.translation;
        let key = [a, b, c, d, tx, ty].map(f32::to_bits);
        let next = self.next();
        let index = *self.transforms.entry(key).or_insert(next);
        if index == next {
            self.texels.extend_from_slice(&[[a, b, c, d], [tx, ty, 0., 0.]]);
        }
        index
    }

    /// Appends a background's block when it is not a solid color: one texel of the angle or
    /// pattern height, the gradient record's first texel (bit-cast) and the opacity a
    /// [`BackgroundTag::Gradient`] is drawn with; then, for a [`BackgroundTag::LinearGradient`],
    /// its two stops' colors and their percentages.
    fn background_block(&mut self, background: &Background) {
        if background.tag == BackgroundTag::Solid {
            return;
        }
        self.texels.push([
            background.gradient_angle_or_pattern_height,
            bits(background.gradient),
            background.colors[0].color.a,
            0.,
        ]);
        if background.tag == BackgroundTag::LinearGradient {
            let [from, to] = &background.colors;
            self.texels.extend_from_slice(&[
                hsla_texel(&from.color),
                hsla_texel(&to.color),
                [from.percentage, to.percentage, 0., 0.],
            ]);
        }
    }

    /// `quad`'s record; its mask, transformation and aux record go into the table.
    pub fn quad(&mut self, quad: &Quad) -> QuadRecord {
        let content_mask = self.content_mask(&quad.content_mask);
        let transformation = self.transformation(&quad.transformation);
        let widths = &quad.border_widths;
        let bordered = [widths.top, widths.right, widths.bottom, widths.left]
            .iter()
            .any(|width| width.0 != 0.);
        let elliptical = quad.corner_radii_y != quad.corner_radii;
        let shaped = quad.corner_shapes != [0, 0];
        let solid = quad.background.tag == BackgroundTag::Solid;

        let mut flags = background_flags(&quad.background);
        if quad.border_style as u32 == 1 {
            flags |= QUAD_DASHED;
        }
        let aux = self.next();
        if bordered {
            flags |= QUAD_BORDER;
            self.texels
                .extend_from_slice(&[edges_texel(widths), hsla_texel(&quad.border_color)]);
        }
        if elliptical {
            flags |= QUAD_ELLIPTICAL;
            self.texels.push(corners_texel(&quad.corner_radii_y));
        }
        if shaped {
            flags |= QUAD_SHAPED;
            let [first, second] = quad.corner_shapes;
            self.texels.push([bits(first), bits(second), 0., 0.]);
        }
        if !solid {
            self.background_block(&quad.background);
        }
        QuadRecord {
            bounds: quad.bounds,
            corner_radii: quad.corner_radii,
            background: quad.background.solid,
            flags,
            content_mask,
            transformation,
            aux: if bordered || elliptical || shaped || !solid {
                aux
            } else {
                0
            },
        }
    }

    /// `shadow`'s record; its mask, transformation and vertical radii go into the table.
    pub fn shadow(&mut self, shadow: &Shadow) -> ShadowRecord {
        let content_mask = self.content_mask(&shadow.content_mask);
        let transformation = self.transformation(&shadow.transformation);
        let elliptical = shadow.corner_radii_y != shadow.corner_radii
            || shadow.element_corner_radii_y != shadow.element_corner_radii;
        let aux = if elliptical {
            let aux = self.next();
            self.texels.extend_from_slice(&[
                corners_texel(&shadow.corner_radii_y),
                corners_texel(&shadow.element_corner_radii_y),
            ]);
            aux
        } else {
            0
        };
        ShadowRecord {
            bounds: shadow.bounds,
            corner_radii: shadow.corner_radii,
            color: shadow.color,
            element_bounds: shadow.element_bounds,
            element_corner_radii: shadow.element_corner_radii,
            blur_radius: shadow.blur_radius,
            content_mask,
            transformation,
            info: aux << 1 | u32::from(shadow.inset != 0),
        }
    }

    /// `underline`'s record; its mask and transformation go into the table.
    pub fn underline(&mut self, underline: &Underline) -> UnderlineRecord {
        UnderlineRecord {
            bounds: underline.bounds,
            color: underline.color,
            thickness: underline.thickness,
            wavy: u32::from(bool::from(underline.wavy)),
            content_mask: self.content_mask(&underline.content_mask),
            transformation: self.transformation(&underline.transformation),
        }
    }

    /// `sprite`'s record; its mask and transformation go into the table.
    pub fn monochrome_sprite(&mut self, sprite: &MonochromeSprite) -> SpriteRecord {
        SpriteRecord {
            tile_bounds: sprite.tile.bounds,
            bounds: sprite.bounds,
            color: sprite.color,
            content_mask: self.content_mask(&sprite.content_mask),
            transformation: self.transformation(&sprite.transformation),
            pad: [0; 2],
        }
    }

    /// `sprite`'s record; its mask and transformation go into the table.
    pub fn subpixel_sprite(&mut self, sprite: &SubpixelSprite) -> SpriteRecord {
        SpriteRecord {
            tile_bounds: sprite.tile.bounds,
            bounds: sprite.bounds,
            color: sprite.color,
            content_mask: self.content_mask(&sprite.content_mask),
            transformation: self.transformation(&sprite.transformation),
            pad: [0; 2],
        }
    }

    /// `sprite`'s record; its mask, transformation and vertical radii go into the table.
    pub fn polychrome_sprite(&mut self, sprite: &PolychromeSprite) -> PolychromeSpriteRecord {
        let content_mask = self.content_mask(&sprite.content_mask);
        let transformation = self.transformation(&sprite.transformation);
        let aux = if sprite.corner_radii_y != sprite.corner_radii {
            let aux = self.next();
            self.texels.push(corners_texel(&sprite.corner_radii_y));
            aux
        } else {
            0
        };
        PolychromeSpriteRecord {
            tile_bounds: sprite.tile.bounds,
            bounds: sprite.bounds,
            corner_radii: sprite.corner_radii,
            opacity: sprite.opacity,
            content_mask,
            transformation,
            info: aux << 2
                | u32::from(bool::from(sprite.grayscale)) << 1
                | u32::from(sprite.exact_scale != 0),
        }
    }

    /// `path`'s record (see [`PathVertexRecord`]) and content mask, which its vertices share.
    pub fn path(&mut self, path: &Path<ScaledPixels>) -> PathRecord {
        let content_mask = self.content_mask(&path.content_mask);
        let record = self.next();
        self.texels.extend_from_slice(&[
            bounds_texel(&path.clipped_bounds()),
            hsla_texel(&path.color.solid),
            [bits(background_flags(&path.color)), 0., 0., 0.],
        ]);
        self.background_block(&path.color);
        PathRecord {
            texel: record,
            content_mask,
        }
    }

    /// `path`'s vertices, each pointing at `record` ([`InstanceTables::path`]).
    pub fn path_vertices<'a>(
        path: &'a Path<ScaledPixels>,
        record: PathRecord,
    ) -> impl ExactSizeIterator<Item = PathVertexRecord> + 'a {
        path.vertices.iter().map(move |vertex| PathVertexRecord {
            xy_position: vertex.xy_position,
            st_position: vertex.st_position,
            path: record.texel,
            content_mask: record.content_mask,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AtlasTextureId, AtlasTextureKind, AtlasTile, BorderStyle, CornerShapes, PaddedBool32,
        TileId, linear_color_stop, linear_gradient, point, size,
    };

    fn sp(value: f32) -> ScaledPixels {
        ScaledPixels(value)
    }

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(sp(x), sp(y)),
            size: size(sp(w), sp(h)),
        }
    }

    fn mask(x: f32) -> ContentMask<ScaledPixels> {
        ContentMask::from(rect(x, 0., 100., 100.))
    }

    fn quad() -> Quad {
        Quad {
            bounds: rect(1., 2., 3., 4.),
            content_mask: mask(0.),
            background: crate::solid_background(Hsla {
                h: 0.1,
                s: 0.2,
                l: 0.3,
                a: 0.4,
            }),
            corner_radii: Corners::all(sp(2.)),
            corner_radii_y: Corners::all(sp(2.)),
            transformation: TransformationMatrix::unit(),
            ..Default::default()
        }
    }

    fn texels_u32(texel: Texel) -> [u32; 4] {
        texel.map(f32::to_bits)
    }

    #[test]
    fn record_sizes() {
        assert_eq!(size_of::<QuadRecord>(), 64);
        assert_eq!(size_of::<ShadowRecord>(), 96);
        assert_eq!(size_of::<UnderlineRecord>(), 48);
        assert_eq!(size_of::<SpriteRecord>(), 64);
        assert_eq!(size_of::<PolychromeSpriteRecord>(), 64);
        assert_eq!(size_of::<PathVertexRecord>(), 24);
    }

    #[test]
    fn plain_quads_share_their_mask_and_need_no_aux() {
        let mut tables = InstanceTables::default();
        let first = tables.quad(&quad());
        let second = tables.quad(&Quad {
            bounds: rect(10., 10., 5., 5.),
            ..quad()
        });
        assert_eq!(first.content_mask, 2);
        assert_eq!(second.content_mask, 2);
        assert_eq!(first.transformation, 0);
        assert_eq!(first.aux, 0);
        assert_eq!(first.flags, 0);
        // The identity, then one circular mask.
        assert_eq!(tables.texels().len(), 5);
        assert_eq!(tables.texels()[0], [1., 0., 0., 1.]);
        assert_eq!(tables.texels()[2], [0., 0., 100., 100.]);
    }

    #[test]
    fn masks_and_transformations_dedupe_across_kinds() {
        let mut tables = InstanceTables::default();
        let other = mask(5.);
        let a = tables.quad(&quad()).content_mask;
        let b = tables
            .quad(&Quad {
                content_mask: other,
                ..quad()
            })
            .content_mask;
        let c = tables.underline(&Underline {
            order: 0,
            pad: 0,
            bounds: rect(0., 0., 1., 1.),
            content_mask: mask(0.),
            color: Hsla::default(),
            thickness: sp(1.),
            wavy: PaddedBool32::from(false),
            transformation: TransformationMatrix::unit().translate(point(sp(3.), sp(4.))),
        });
        assert_eq!(a, c.content_mask);
        assert_ne!(a, b);
        assert_eq!(c.transformation, 2 + 3 + 3);
        assert_eq!(tables.texels()[c.transformation as usize + 1], [3., 4., 0., 0.]);
        let again = tables.transformation(&TransformationMatrix::unit().translate(point(sp(3.), sp(4.))));
        assert_eq!(again, c.transformation);
    }

    #[test]
    fn quad_aux_holds_border_radii_y_shapes_and_linear_stops_in_order() {
        let mut tables = InstanceTables::default();
        let background = linear_gradient(
            45.,
            linear_color_stop(Hsla { h: 0., s: 1., l: 0.5, a: 1. }, 0.25),
            linear_color_stop(Hsla { h: 0.5, s: 1., l: 0.5, a: 0.5 }, 0.75),
        );
        let mut quad = Quad {
            border_style: BorderStyle::Dashed,
            border_widths: Edges {
                top: sp(1.),
                right: sp(2.),
                bottom: sp(3.),
                left: sp(4.),
            },
            border_color: Hsla { h: 0.7, s: 0.6, l: 0.5, a: 0.4 },
            corner_radii_y: Corners::all(sp(5.)),
            corner_shapes: CornerShapes::all(2.).gpu(),
            ..quad()
        };
        quad.background = background;
        let record = tables.quad(&quad);
        let tag = (record.flags >> BACKGROUND_TAG_SHIFT) & 7;
        assert_eq!(tag, BackgroundTag::LinearGradient as u32);
        assert_ne!(record.flags & QUAD_DASHED, 0);
        assert_ne!(record.flags & QUAD_BORDER, 0);
        assert_ne!(record.flags & QUAD_ELLIPTICAL, 0);
        assert_ne!(record.flags & QUAD_SHAPED, 0);
        let aux = &tables.texels()[record.aux as usize..];
        assert_eq!(aux[0], [1., 2., 3., 4.]);
        assert_eq!(aux[1], [0.7, 0.6, 0.5, 0.4]);
        assert_eq!(aux[2], [5.; 4]);
        assert_eq!(
            texels_u32(aux[3])[..2],
            CornerShapes::all(2.).gpu()
        );
        assert_eq!(aux[4][0], 45.);
        assert_eq!(aux[5], [0., 1., 0.5, 1.]);
        assert_eq!(aux[6], [0.5, 1., 0.5, 0.5]);
        assert_eq!(aux[7][..2], [0.25, 0.75]);
        assert_eq!(tables.texels().len(), record.aux as usize + 8);
    }

    #[test]
    fn rebase_offsets_every_index_and_keeps_absent_aux_absent() {
        let mut tables = InstanceTables::default();
        let mut shadow = tables.shadow(&Shadow {
            order: 0,
            blur_radius: sp(1.),
            bounds: rect(0., 0., 1., 1.),
            corner_radii: Corners::default(),
            corner_radii_y: Corners::default(),
            content_mask: mask(0.),
            color: Hsla::default(),
            element_bounds: rect(0., 0., 1., 1.),
            element_corner_radii: Corners::default(),
            element_corner_radii_y: Corners::default(),
            inset: 1,
            pad: 0,
            transformation: TransformationMatrix::unit(),
        });
        assert_eq!(shadow.info, 1);
        shadow.rebase(100);
        assert_eq!(shadow.info, 1);
        assert_eq!(shadow.content_mask, 102);
        assert_eq!(shadow.transformation, 100);

        let mut sprite = tables.polychrome_sprite(&PolychromeSprite {
            order: 0,
            exact_scale: 1,
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: 0,
                    kind: AtlasTextureKind::Polychrome,
                },
                tile_id: TileId(0),
                padding: 0,
                bounds: Bounds::default(),
            },
            grayscale: PaddedBool32::from(true),
            opacity: 1.,
            bounds: rect(0., 0., 1., 1.),
            content_mask: mask(0.),
            corner_radii: Corners::all(sp(1.)),
            corner_radii_y: Corners::all(sp(2.)),
            transformation: TransformationMatrix::unit(),
        });
        let aux = sprite.info >> 2;
        assert_eq!(tables.texels()[aux as usize], [2.; 4]);
        assert_eq!(sprite.info & 3, 3);
        sprite.rebase(10);
        assert_eq!(sprite.info >> 2, aux + 10);
        assert_eq!(sprite.info & 3, 3);
    }
}
