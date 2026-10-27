// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, BackgroundTag, Bounds, ContentMask, Corners,
    DevicePixels, Edges, Hsla, Pixels, Point, Radians, RetainedShapesDraw, ScaledPixels, Size,
    bounds_tree::BoundsTree, point,
};
use smallvec::SmallVec;
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
    /// Groups being painted, innermost last. Only the scene a window paints
    /// into has them; what they draw goes to the innermost one's scene.
    open_groups: Vec<OpenGroup>,
    /// Whether this is a group's scene, which tracks what it covers below.
    in_group: bool,
    /// Whether a primitive was ordered after one it overlaps.
    overlapping: bool,
    /// Whether a layer was pushed.
    has_layers: bool,
    /// The union of what the primitives and groups cover, if anything.
    drawn_bounds: Option<Bounds<ScaledPixels>>,
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
    /// The sources of the shaders this frame's backdrop filters run, once each.
    pub shaders: Vec<Shader>,
    /// Their uniforms, each draw's at [`ShaderDraw::uniforms`], 16-byte aligned.
    pub shader_uniforms: Vec<u8>,
    /// Groups drawn offscreen and composited as one image ([`Scene::push_group`]).
    pub groups: Vec<CompositedGroup>,
    /// The records of the gradients this frame's quads paint ([`crate::GradientRecord`]),
    /// in 16-byte texels. A group's scene refers to the window's.
    pub gradients: Vec<[f32; 4]>,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.open_groups.clear();
        self.overlapping = false;
        self.has_layers = false;
        self.drawn_bounds = None;
        self.groups.clear();
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
        self.shaders.clear();
        self.shader_uniforms.clear();
        self.gradients.clear();
    }

    /// Records a gradient's texels; returns the index of its first.
    pub fn add_gradient(&mut self, texels: &[[f32; 4]]) -> u32 {
        let start = self.gradients.len() as u32;
        self.gradients.extend_from_slice(texels);
        start
    }

    /// The texels of the gradient a quad from `from` paints, copied into this scene,
    /// so that the quad can be inserted here.
    fn adopt_gradient(&mut self, quad: &mut Quad, from: &Scene) {
        let start = quad.background.gradient as usize;
        let Some(header) = from.gradients.get(start) else {
            return;
        };
        let len = 2 + 2 * header[1] as usize;
        if let Some(texels) = from.gradients.get(start..start + len) {
            quad.background.gradient = self.add_gradient(texels);
        }
    }

    /// Records a shader draw's source (once per frame) and uniforms.
    pub fn add_shader(
        &mut self,
        shader: &Shader,
        uniforms: &[u8],
        reads_backdrop: bool,
        scale_factor: f32,
    ) -> ShaderDraw {
        let index = match self.shaders.iter().position(|known| known.id == shader.id) {
            Some(index) => index,
            None => {
                self.shaders.push(shader.clone());
                self.shaders.len() - 1
            }
        };
        let start = self.shader_uniforms.len();
        self.shader_uniforms.extend_from_slice(uniforms);
        let padded = self.shader_uniforms.len().next_multiple_of(16);
        self.shader_uniforms.resize(padded, 0);
        ShaderDraw {
            shader: index as u32 + 1,
            uniforms_start: start as u32,
            uniforms_len: uniforms.len() as u32,
            reads_backdrop: reads_backdrop.into(),
            scale_factor,
        }
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
        match self.open_groups.last_mut() {
            Some(group) => group.start_layer(bounds),
            None => self.start_layer(bounds),
        }
    }

    pub fn pop_layer(&mut self) {
        self.paint_operations.push(PaintOperation::EndLayer);
        match self.open_groups.last_mut() {
            Some(group) => group.end_layer(),
            None => self.end_layer(),
        }
    }

    fn start_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push((order, bounds));
        self.has_layers = true;
    }

    fn end_layer(&mut self) {
        self.layer_stack.pop();
    }

    /// Starts a compositing group: what is painted until [`Scene::pop_group`] is drawn
    /// into an offscreen image, which is then filtered, faded and blended onto what is
    /// behind it as one (CSS group opacity, `filter`, `mix-blend-mode`, `isolation`).
    ///
    /// Groups that need no offscreen image are painted directly: an empty group or one
    /// of opacity 0 paints nothing; a group without a filter or blend mode whose content
    /// does not overlap itself fades each primitive instead (when every primitive can
    /// be faded alone), and one of opacity 1 that holds no blending group paints as if
    /// it were not there. Backdrop filters inside a group drawn offscreen read only
    /// what the group drew. A group with opacity below 1, a filter or a blend mode is
    /// their Backdrop Root (Filter Effects 2), so it is always drawn offscreen when it
    /// holds one; an opaque group without those is not a Backdrop Root.
    pub fn push_group(&mut self, mut effects: GroupEffects) {
        // `opacity()` filters fade the image as the group's opacity does (they commute
        // with the color filters, which act on straight colors).
        let faded: f32 = effects
            .color_filters
            .iter()
            .map(|filter| match filter {
                ColorFilter::Opacity(amount) => amount.clamp(0., 1.),
                _ => 1.,
            })
            .product();
        effects.opacity *= faded;
        effects
            .color_filters
            .retain(|filter| !matches!(filter, ColorFilter::Opacity(_)));
        self.paint_operations
            .push(PaintOperation::StartGroup(effects.clone()));
        self.open_groups.push(OpenGroup::new(effects));
    }

    /// Ends the innermost group ([`Scene::push_group`]).
    pub fn pop_group(&mut self) {
        self.paint_operations.push(PaintOperation::EndGroup);
        let Some(group) = self.open_groups.pop() else {
            return;
        };
        let masks_parent = group.effects.masks_parent;
        let closed = group.close();
        // A mask group's image masks the group around it (outside any group, nothing).
        if let Some(mode) = masks_parent {
            if let Some(parent) = self.open_groups.last_mut() {
                let image = match closed {
                    ClosedGroup::Offscreen(image) => Some(image),
                    _ => None,
                };
                parent.set_mask(mode, image);
            }
            return;
        }
        match self.open_groups.last_mut() {
            Some(parent) => closed.apply(parent),
            None => closed.apply(self),
        }
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let primitive = primitive.into();
        match self.open_groups.last_mut() {
            Some(group) => {
                self.paint_operations
                    .push(PaintOperation::Primitive(primitive.clone()));
                group.insert(primitive);
            }
            None => {
                if let Some(primitive) = self.place(primitive) {
                    self.paint_operations
                        .push(PaintOperation::Primitive(primitive));
                }
            }
        }
    }

    /// Orders and stores a primitive; returns it, with its order, unless it was culled.
    fn place(&mut self, mut primitive: Primitive) -> Option<Primitive> {
        let mut clipped_bounds = primitive
            .transformed_bounds()
            .intersect(&primitive.content_mask().bounds);
        let mut drawn_bounds = clipped_bounds;
        // A backdrop filter draws inside its mask but reads three blur radii beyond it.
        if let Primitive::BackdropFilter(filter) = &primitive {
            drawn_bounds = filter.bounds.intersect(&filter.content_mask.bounds);
            if drawn_bounds.is_empty() {
                return None;
            }
            clipped_bounds = filter
                .sample_bounds()
                .intersect(&filter.content_mask.bounds.dilate(filter.blur_radius * 3.));
        }

        if clipped_bounds.is_empty() {
            return None;
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
        self.note_drawn(order, drawn_bounds);
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
        Some(primitive)
    }

    /// Orders and stores a group drawn offscreen. Like a backdrop filter, inside a
    /// layer it takes its own order and the rest of the layer draws after it.
    fn place_group(&mut self, mut group: CompositedGroup) {
        let bounds = group.bounds;
        if bounds.is_empty() {
            return;
        }
        // Its backdrop reads three blur radii around its bounds: order it after that.
        let ordered = match &group.effects.backdrop {
            Some(filter) => bounds.union(
                &filter
                    .sample_bounds()
                    .intersect(&filter.content_mask.bounds.dilate(filter.blur_radius * 3.)),
            ),
            None => bounds,
        };
        let order = match self.layer_stack.last_mut() {
            Some(layer) => {
                let order = self.primitive_bounds.insert(ordered);
                layer.0 = self.primitive_bounds.insert(layer.1);
                order
            }
            None => self.primitive_bounds.insert(ordered),
        };
        self.note_drawn(order, bounds);
        group.order = order;
        self.groups.push(group);
    }

    fn note_drawn(&mut self, order: DrawOrder, bounds: Bounds<ScaledPixels>) {
        if !self.in_group {
            return;
        }
        // Orders start at 1, so a higher one overlaps something drawn earlier.
        self.overlapping |= order > 1;
        self.drawn_bounds = Some(match self.drawn_bounds {
            Some(drawn) => drawn.union(&bounds),
            None => bounds,
        });
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                // A shader draw refers to its frame's sources and uniforms: copy them over.
                PaintOperation::Primitive(Primitive::BackdropFilter(filter))
                    if filter.shader.shader != 0 =>
                {
                    let mut filter = *filter;
                    if let Some(shader) = filter.shader.shader(prev_scene) {
                        filter.shader = self.add_shader(
                            shader,
                            filter.shader.uniforms(prev_scene),
                            filter.shader.reads_backdrop(),
                            filter.shader.scale_factor,
                        );
                        self.insert_primitive(filter);
                    }
                }
                // So does a gradient quad, to its frame's gradients.
                PaintOperation::Primitive(Primitive::Quad(quad))
                    if quad.background.tag == BackgroundTag::Gradient =>
                {
                    let mut quad = *quad;
                    self.adopt_gradient(&mut quad, prev_scene);
                    self.insert_primitive(quad);
                }
                PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
                PaintOperation::StartGroup(effects) => self.push_group(effects.clone()),
                PaintOperation::EndGroup => self.pop_group(),
            }
        }
    }

    pub fn finish(&mut self) {
        // Groups are closed by the closures that open them; close any left open.
        while !self.open_groups.is_empty() {
            self.pop_group();
        }
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
        self.groups.sort_by_key(|group| group.order);
    }

    /// This scene and the scenes of its groups, nested ones included, depth first in
    /// the order renderers draw them.
    pub fn scenes(&self) -> SmallVec<[&Scene; 4]> {
        fn visit<'a>(scene: &'a Scene, out: &mut SmallVec<[&'a Scene; 4]>) {
            out.push(scene);
            for group in &scene.groups {
                visit(&group.scene, out);
                if let Some(mask) = &group.mask {
                    visit(&mask.image.scene, out);
                }
            }
        }
        let mut out = SmallVec::new();
        visit(self, &mut out);
        out
    }

    /// This frame painted without offscreen groups, for renderers that cannot draw
    /// them: each group's opacity fades its primitives, as GPUI faded elements before
    /// groups, and filters and blend modes are dropped. A group's backdrop paints as
    /// a backdrop filter.
    pub fn without_groups(&self) -> Scene {
        let mut scene = Scene::default();
        let mut opacities = vec![1f32];
        // Inside a mask group, how many groups deep: masks are not drawn (content
        // draws unmasked).
        let mut in_mask = 0usize;
        for operation in &self.paint_operations {
            let opacity = opacities.last().copied().unwrap_or(1.);
            if in_mask > 0 {
                match operation {
                    PaintOperation::StartGroup(_) => in_mask += 1,
                    PaintOperation::EndGroup => in_mask -= 1,
                    _ => {}
                }
                continue;
            }
            if let PaintOperation::StartGroup(effects) = operation
                && effects.masks_parent.is_some()
            {
                in_mask = 1;
                continue;
            }
            match operation {
                PaintOperation::Primitive(primitive) => {
                    let mut primitive = primitive.clone();
                    if let Primitive::BackdropFilter(filter) = &mut primitive
                        && let Some(shader) = filter.shader.shader(self)
                    {
                        filter.shader = scene.add_shader(
                            shader,
                            filter.shader.uniforms(self),
                            filter.shader.reads_backdrop(),
                            filter.shader.scale_factor,
                        );
                    }
                    if let Primitive::Quad(quad) = &mut primitive
                        && quad.background.tag == BackgroundTag::Gradient
                    {
                        scene.adopt_gradient(quad, self);
                    }
                    if opacity < 1. {
                        primitive.fade(opacity);
                    }
                    scene.insert_primitive(primitive);
                }
                PaintOperation::StartLayer(bounds) => scene.push_layer(*bounds),
                PaintOperation::EndLayer => scene.pop_layer(),
                PaintOperation::StartGroup(effects) => {
                    let opacity = opacity * effects.opacity;
                    if let Some(mut filter) = effects.backdrop {
                        filter.opacity *= opacity;
                        scene.insert_primitive(filter);
                    }
                    opacities.push(opacity);
                }
                PaintOperation::EndGroup => {
                    opacities.pop();
                }
            }
        }
        scene.finish();
        scene
    }

    /// Moves everything this finished scene draws by `offset`, as if painted that much
    /// further right and down: a group's scene moves to its own image's origin.
    fn translate(&mut self, offset: Point<ScaledPixels>) {
        let bounds = |b: &mut Bounds<ScaledPixels>| b.origin = b.origin + offset;
        let mask = |m: &mut ContentMask<ScaledPixels>| m.bounds.origin = m.bounds.origin + offset;
        // A transformation applied to content in window pixels, conjugated: content
        // moved by `offset`, transformed, lands `offset` from where it did.
        let conjugate = |t: &mut TransformationMatrix| {
            let [[a, b], [c, d]] = t.rotation_scale;
            let (x, y) = (offset.x.0, offset.y.0);
            t.translation[0] += x - (a * x + b * y);
            t.translation[1] += y - (c * x + d * y);
        };
        for quad in &mut self.quads {
            bounds(&mut quad.bounds);
            mask(&mut quad.content_mask);
            conjugate(&mut quad.transformation);
        }
        for shadow in &mut self.shadows {
            bounds(&mut shadow.bounds);
            bounds(&mut shadow.element_bounds);
            mask(&mut shadow.content_mask);
            conjugate(&mut shadow.transformation);
        }
        for path in &mut self.paths {
            bounds(&mut path.bounds);
            mask(&mut path.content_mask);
            for vertex in &mut path.vertices {
                vertex.xy_position = vertex.xy_position + offset;
                mask(&mut vertex.content_mask);
            }
        }
        for underline in &mut self.underlines {
            bounds(&mut underline.bounds);
            mask(&mut underline.content_mask);
            conjugate(&mut underline.transformation);
        }
        for sprite in &mut self.monochrome_sprites {
            bounds(&mut sprite.bounds);
            mask(&mut sprite.content_mask);
            conjugate(&mut sprite.transformation);
        }
        for sprite in &mut self.subpixel_sprites {
            bounds(&mut sprite.bounds);
            mask(&mut sprite.content_mask);
            conjugate(&mut sprite.transformation);
        }
        for sprite in &mut self.polychrome_sprites {
            bounds(&mut sprite.bounds);
            mask(&mut sprite.content_mask);
            conjugate(&mut sprite.transformation);
        }
        for surface in &mut self.surfaces {
            bounds(&mut surface.bounds);
            mask(&mut surface.content_mask);
        }
        for draw in &mut self.retained_shapes {
            bounds(&mut draw.bounds);
            mask(&mut draw.content_mask);
            // Buffer space to window pixels: the window side moves.
            draw.paint.transformation.translation[0] += offset.x.0;
            draw.paint.transformation.translation[1] += offset.y.0;
        }
        for filter in &mut self.backdrop_filters {
            bounds(&mut filter.bounds);
            mask(&mut filter.content_mask);
        }
        for group in &mut self.groups {
            bounds(&mut group.bounds);
            mask(&mut group.effects.content_mask);
            if let Some(filter) = &mut group.effects.backdrop {
                bounds(&mut filter.bounds);
                mask(&mut filter.content_mask);
            }
            // A mask's image draws in its own coordinates; where it sits moves.
            if let Some(group_mask) = &mut group.mask {
                bounds(&mut group_mask.image.bounds);
                mask(&mut group_mask.image.effects.content_mask);
            }
        }
    }

    /// The deepest nesting of groups drawn offscreen (0 without any): how many
    /// offscreen images a renderer holds at once.
    pub fn group_depth(&self) -> usize {
        self.groups
            .iter()
            .map(|group| {
                let mask = group
                    .mask
                    .as_ref()
                    .map_or(0, |mask| 1 + mask.image.scene.group_depth());
                (1 + group.scene.group_depth()).max(mask)
            })
            .max()
            .unwrap_or(0)
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
            groups_start: 0,
            groups_iter: self.groups.iter().peekable(),
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
    Group,
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
    StartGroup(GroupEffects),
    EndGroup,
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

    /// Whether fading this primitive alone equals fading it as a group: it paints
    /// nothing over itself (a quad with both a background and a border does) and can
    /// be faded (surfaces, retained shapes and backdrop filters cannot).
    fn fades_alone(&self) -> bool {
        match self {
            Primitive::Quad(quad) => {
                let border = quad.border_color.a > 0.
                    && (quad.border_widths.top.0 > 0.
                        || quad.border_widths.right.0 > 0.
                        || quad.border_widths.bottom.0 > 0.
                        || quad.border_widths.left.0 > 0.);
                !(border && !quad.background.is_transparent())
            }
            Primitive::Shadow(_)
            | Primitive::Path(_)
            | Primitive::Underline(_)
            | Primitive::MonochromeSprite(_)
            | Primitive::SubpixelSprite(_)
            | Primitive::PolychromeSprite(_) => true,
            Primitive::Surface(_) | Primitive::RetainedShapes(_) | Primitive::BackdropFilter(_) => {
                false
            }
        }
    }

    /// Multiplies its alpha by `opacity` (for [`Primitive::fades_alone`] primitives).
    fn fade(&mut self, opacity: f32) {
        match self {
            Primitive::Quad(quad) => {
                quad.background = quad.background.opacity(opacity);
                quad.border_color = quad.border_color.opacity(opacity);
            }
            Primitive::Shadow(shadow) => shadow.color = shadow.color.opacity(opacity),
            Primitive::Path(path) => path.color = path.color.opacity(opacity),
            Primitive::Underline(underline) => underline.color = underline.color.opacity(opacity),
            Primitive::MonochromeSprite(sprite) => sprite.color = sprite.color.opacity(opacity),
            Primitive::SubpixelSprite(sprite) => sprite.color = sprite.color.opacity(opacity),
            Primitive::PolychromeSprite(sprite) => sprite.opacity *= opacity,
            Primitive::RetainedShapes(draw) => draw.opacity *= opacity,
            Primitive::BackdropFilter(filter) => filter.opacity *= opacity,
            Primitive::Surface(_) => {}
        }
    }

    /// Intersects its content mask with `mask`.
    fn clip(&mut self, mask: &ContentMask<ScaledPixels>) {
        let content_mask = match self {
            Primitive::Shadow(shadow) => &mut shadow.content_mask,
            Primitive::Quad(quad) => &mut quad.content_mask,
            Primitive::Path(path) => &mut path.content_mask,
            Primitive::Underline(underline) => &mut underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &mut sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &mut sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &mut sprite.content_mask,
            Primitive::Surface(surface) => &mut surface.content_mask,
            Primitive::RetainedShapes(draw) => &mut draw.content_mask,
            Primitive::BackdropFilter(filter) => &mut filter.content_mask,
        };
        *content_mask = content_mask.intersect(mask);
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
    groups_start: usize,
    groups_iter: Peekable<slice::Iter<'a, CompositedGroup>>,
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
            (
                self.groups_iter.peek().map(|g| g.order),
                PrimitiveKind::Group,
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
            PrimitiveKind::Group => {
                let start = self.groups_start;
                let mut end = start + 1;
                self.groups_iter.next();
                while self
                    .groups_iter
                    .next_if(|group| (group.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    end += 1;
                }
                self.groups_start = end;
                Some(PrimitiveBatch::Groups(start..end))
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
    /// Groups ([`Scene::groups`]), each drawn offscreen and composited in turn.
    Groups(Range<usize>),
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
            Self::Groups(range) => format!("groups ({})", range.len()),
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
    /// The superellipse shape of each corner ([`CornerShapes::gpu`]); zero is round.
    /// Also aligns the quad to 16 bytes (WebGL reads quads as whole texels).
    pub corner_shapes: [u32; 2],
}

/// The shape of each corner of a rounded quad (CSS Borders 4 `corner-shape`): the
/// superellipse parameter `K` of `superellipse(K)`, whose corner curve is
/// `|x|^(2^K) + |y|^(2^K) = 1`; a negative `K` is the concave curve of `-K`. `1` is
/// `round` (the default), `2` `squircle`, `0` `bevel`, `-1` `scoop`, infinity
/// `square` and minus infinity `notch`. Corners without a radius stay square.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CornerShapes {
    /// `K` of the top left corner.
    pub top_left: f32,
    /// `K` of the top right corner.
    pub top_right: f32,
    /// `K` of the bottom right corner.
    pub bottom_right: f32,
    /// `K` of the bottom left corner.
    pub bottom_left: f32,
}

impl Default for CornerShapes {
    fn default() -> Self {
        Self::ROUND
    }
}

impl CornerShapes {
    /// Every corner round.
    pub const ROUND: CornerShapes = CornerShapes {
        top_left: 1.,
        top_right: 1.,
        bottom_right: 1.,
        bottom_left: 1.,
    };

    /// The same shape at every corner.
    pub fn all(k: f32) -> Self {
        CornerShapes {
            top_left: k,
            top_right: k,
            bottom_right: k,
            bottom_left: k,
        }
    }

    /// Whether every corner is round.
    pub fn is_round(&self) -> bool {
        *self == Self::ROUND
    }

    /// The renderers' encoding ([`Quad::corner_shapes`]): `K - 1` in 1/1024 steps as
    /// a signed 16-bit value per corner (zero is round; the extremes are square and
    /// notch, and anything beyond ±31 rounds to them), top left and top right in the
    /// first word's low and high halves, bottom right and bottom left in the second's.
    pub fn gpu(&self) -> [u32; 2] {
        let code = |k: f32| -> u32 {
            let v = if k.is_nan() {
                0
            } else if k >= 32. {
                i16::MAX
            } else if k <= -30. {
                i16::MIN
            } else {
                ((k - 1.) * 1024.).round().clamp(-32767., 32766.) as i16
            };
            v as u16 as u32
        };
        let c = self;
        [
            code(c.top_left) | code(c.top_right) << 16,
            code(c.bottom_right) | code(c.bottom_left) << 16,
        ]
    }
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
    /// A custom shader that draws the element's box instead of the filters above
    /// ([`crate::Window::paint_shader`]); [`ShaderDraw::NONE`] for none.
    pub shader: ShaderDraw,
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
            shader: ShaderDraw::NONE,
        }
    }
}

impl BackdropFilter {
    /// The same filter moved by `offset` (bounds and content mask).
    pub fn translated(mut self, offset: Point<ScaledPixels>) -> Self {
        self.bounds.origin = self.bounds.origin + offset;
        self.content_mask.bounds.origin = self.content_mask.bounds.origin + offset;
        self
    }

    /// What the filter reads: its bounds and three blur radii around them.
    pub fn sample_bounds(&self) -> Bounds<ScaledPixels> {
        self.bounds.dilate(self.blur_radius * 3.)
    }
}

/// A WGSL shader that draws an element's box ([`crate::Window::paint_shader`]).
///
/// The source defines `fn shade(input: ShaderInput) -> vec4<f32>`, returning a
/// straight-alpha color for each pixel; [`SHADER_PRELUDE`] declares `ShaderInput`
/// and `backdrop(uv)`, which samples what is drawn behind the element. Uniforms
/// passed to `paint_shader` bind at `@group(0) @binding(3)`. GPUI clips the result
/// to the element's corner radii and content mask and fades it with the element
/// opacity.
#[derive(Clone, Debug, PartialEq)]
pub struct Shader {
    /// A hash of the source; renderers cache pipelines by it.
    pub id: ShaderId,
    /// The WGSL source, without the prelude.
    pub source: std::sync::Arc<str>,
}

impl Shader {
    /// A shader from WGSL source.
    pub fn new(source: impl Into<std::sync::Arc<str>>) -> Self {
        use std::hash::{Hash, Hasher};
        let source = source.into();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut hasher);
        Self {
            id: ShaderId(hasher.finish()),
            source,
        }
    }

    /// The prelude and the source: the WGSL module renderers compile.
    pub fn module_source(&self) -> String {
        format!("{SHADER_PRELUDE}\n{}", self.source)
    }
}

impl Serialize for Shader {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.source)
    }
}

impl<'de> Deserialize<'de> for Shader {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer).map(Shader::new)
    }
}

impl JsonSchema for Shader {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Shader".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({ "type": "string", "description": "WGSL that defines `shade`" })
    }
}

/// A [`Shader`] that paints an element's background ([`crate::Style::shader`]),
/// with its uniforms and what it reads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ShaderFill {
    /// The shader.
    pub shader: Shader,
    /// Its uniforms (`@group(0) @binding(3)`), at most 4096 bytes.
    pub uniforms: Vec<u8>,
    /// What it reads besides its uniforms.
    pub input: ShaderInput,
}

/// Identifies a [`Shader`] by its source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct ShaderId(pub u64);

/// What a [`Shader`] reads besides its uniforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum ShaderInput {
    /// Nothing: `backdrop(uv)` returns transparent black.
    None,
    /// What is drawn behind the element, through `backdrop(uv)`.
    Backdrop,
}

/// A shader draw within a [`BackdropFilter`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct ShaderDraw {
    /// One more than the shader's index in [`Scene::shaders`]; 0 for none.
    pub shader: u32,
    /// Where its uniforms start in [`Scene::shader_uniforms`], in bytes.
    pub uniforms_start: u32,
    /// Their length in bytes.
    pub uniforms_len: u32,
    /// Whether it reads the backdrop.
    pub reads_backdrop: PaddedBool32,
    /// Device pixels per logical pixel, for `ShaderInput`.
    pub scale_factor: f32,
}

impl ShaderDraw {
    /// No shader.
    pub const NONE: Self = Self {
        shader: 0,
        uniforms_start: 0,
        uniforms_len: 0,
        reads_backdrop: PaddedBool32(0),
        scale_factor: 1.,
    };

    /// The shader this draws, if any.
    pub fn shader<'a>(&self, scene: &'a Scene) -> Option<&'a Shader> {
        (self.shader as usize)
            .checked_sub(1)
            .and_then(|index| scene.shaders.get(index))
    }

    /// Whether the shader reads the backdrop.
    pub fn reads_backdrop(&self) -> bool {
        self.reads_backdrop.0 != 0
    }

    /// Its uniform bytes in `scene`.
    pub fn uniforms<'a>(&self, scene: &'a Scene) -> &'a [u8] {
        let start = self.uniforms_start as usize;
        scene
            .shader_uniforms
            .get(start..start + self.uniforms_len as usize)
            .unwrap_or_default()
    }
}

/// Whether a shader module may declare a resource at `group`/`binding`: only the
/// prelude's (bindings 0 to 2, by name) and a uniform buffer at binding 3, which
/// are what renderers bind. `name` is the variable's name; `uniform` whether it
/// is in the uniform address space.
pub fn shader_binding_allowed(group: u32, binding: u32, name: Option<&str>, uniform: bool) -> bool {
    match (group, binding) {
        (0, 0) => name == Some("gpui_shader"),
        (0, 1) => name == Some("gpui_backdrop_texture"),
        (0, 2) => name == Some("gpui_backdrop_sampler"),
        (0, 3) => uniform,
        _ => false,
    }
}

/// The WGSL every [`Shader`] is compiled with: its bindings (group 0, bindings 0
/// to 2; uniforms are binding 3), `ShaderInput`, `backdrop(uv)`, and the entry
/// points `gpui_shader_vertex` and `gpui_shader_fragment`, which call `shade`.
pub const SHADER_PRELUDE: &str = include_str!("shader_prelude.wgsl");

impl From<BackdropFilter> for Primitive {
    fn from(filter: BackdropFilter) -> Self {
        Primitive::BackdropFilter(filter)
    }
}

/// How a group's image blends onto what is behind it: CSS `mix-blend-mode`
/// (Compositing and Blending Level 1) and the `plus-lighter` and `plus-darker`
/// compositing operators.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(u32)]
#[expect(missing_docs)]
pub enum BlendMode {
    /// Source over, without blending.
    #[default]
    Normal = 0,
    Multiply = 1,
    Screen = 2,
    Overlay = 3,
    Darken = 4,
    Lighten = 5,
    ColorDodge = 6,
    ColorBurn = 7,
    HardLight = 8,
    SoftLight = 9,
    Difference = 10,
    Exclusion = 11,
    Hue = 12,
    Saturation = 13,
    Color = 14,
    Luminosity = 15,
    /// `plus-lighter`: premultiplied colors and alphas add, clamped to 1.
    PlusLighter = 16,
    /// `plus-darker` (WebKit, Core Graphics `kCGBlendModePlusDarker`): on
    /// premultiplied colors, `max(0, a - ((a_b - b) + (a_s - s)))` with alpha
    /// `a = min(1, a_s + a_b)`; for opaque colors `max(0, 1 - ((1 - b) + (1 - s)))`.
    PlusDarker = 17,
}

impl BlendMode {
    /// Whether compositing reads what is already drawn behind the group: every mode
    /// but normal and plus-lighter, which blending hardware draws.
    pub fn reads_backdrop(self) -> bool {
        !matches!(self, BlendMode::Normal | BlendMode::PlusLighter)
    }
}

/// A CSS color filter function (Filter Effects 1). Renderers apply it to straight
/// (unpremultiplied) colors and clamp them to 0..1 after each.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum ColorFilter {
    /// `brightness()`: a multiplier.
    Brightness(f32),
    /// `contrast()`: 1 leaves colors as they are, 0 is mid gray.
    Contrast(f32),
    /// `grayscale()`: 0 leaves colors as they are, 1 is gray.
    Grayscale(f32),
    /// `hue-rotate()`, in degrees.
    HueRotate(f32),
    /// `invert()`: 0 leaves colors as they are, 1 inverts them.
    Invert(f32),
    /// `saturate()`: 1 leaves colors as they are, 0 is gray.
    Saturate(f32),
    /// `sepia()`: 0 leaves colors as they are, 1 is sepia.
    Sepia(f32),
    /// `opacity()`: multiplies alpha. A group folds it into its opacity.
    Opacity(f32),
}

impl ColorFilter {
    /// The code and amount shaders take: 1 brightness, 2 contrast, 3 grayscale,
    /// 4 hue-rotate (in radians), 5 invert, 6 saturate, 7 sepia. Amounts of
    /// grayscale, invert and sepia are clamped to 0..1, others to 0 and above.
    pub fn gpu(self) -> (u32, f32) {
        let unit = |amount: f32| amount.clamp(0., 1.);
        let positive = |amount: f32| amount.max(0.);
        match self {
            ColorFilter::Brightness(amount) => (1, positive(amount)),
            ColorFilter::Contrast(amount) => (2, positive(amount)),
            ColorFilter::Grayscale(amount) => (3, unit(amount)),
            ColorFilter::HueRotate(degrees) => (4, degrees.to_radians()),
            ColorFilter::Invert(amount) => (5, unit(amount)),
            ColorFilter::Saturate(amount) => (6, positive(amount)),
            ColorFilter::Sepia(amount) => (7, unit(amount)),
            // Folded into the group's opacity; shaders skip code 0.
            ColorFilter::Opacity(amount) => (0, unit(amount)),
        }
    }
}

/// The most color filters a group applies ([`GroupEffects::color_filters`]).
pub const MAX_COLOR_FILTERS: usize = 8;

/// How a group's image is composited, in device pixels ([`Scene::push_group`]):
/// blurred, color filtered in order, faded by `opacity`, then blended with
/// `blend_mode`, inside `content_mask`.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupEffects {
    /// Multiplies the image's alpha; 0 hides it.
    pub opacity: f32,
    /// The standard deviation of a gaussian blur of the image; zero for none.
    pub blur_radius: ScaledPixels,
    /// Applied in order after the blur; only the first [`MAX_COLOR_FILTERS`] apply.
    pub color_filters: SmallVec<[ColorFilter; 4]>,
    /// How the image blends onto what is behind it.
    pub blend_mode: BlendMode,
    /// Clips the image, and so what the blur spreads.
    pub content_mask: ContentMask<ScaledPixels>,
    /// A backdrop filter the image starts with: what is behind the group, filtered
    /// (an element's own `backdrop-filter` under its `filter`, `opacity` and
    /// `mix-blend-mode`, Filter Effects 2). It reads the target the group composites
    /// onto, not the group, and is drawn before what the group paints.
    pub backdrop: Option<BackdropFilter>,
    /// Set on a group painted inside another: its image is not drawn but masks the
    /// group around it, by its alpha or its luminance (CSS `mask-image`, SVG
    /// `<mask>`; [`GroupMask`]). Its other effects do not apply.
    pub masks_parent: Option<MaskMode>,
}

/// What a mask image's pixels mask with (CSS Masking 1 `mask-mode`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MaskMode {
    /// The image's alpha.
    #[default]
    Alpha,
    /// The image's luminance times its alpha (sRGB coefficients 0.2125, 0.7154,
    /// 0.0721, as SVG `<mask>` uses by default).
    Luminance,
}

impl MaskMode {
    /// The code renderers composite with (`GroupCompositeUniforms::mask_mode`).
    pub fn gpu(self) -> u32 {
        match self {
            MaskMode::Alpha => 1,
            MaskMode::Luminance => 2,
        }
    }
}

/// A group's mask: an image whose alpha or luminance multiplies the group's image
/// where they overlap; outside it the group draws nothing.
pub struct GroupMask {
    /// How its pixels mask.
    pub mode: MaskMode,
    /// The mask's image: drawn offscreen like a group's (its own scene, in the
    /// coordinates of its image), never composited itself. Its bounds are in the
    /// same space as the masked group's.
    pub image: Box<CompositedGroup>,
}

impl Default for GroupEffects {
    /// An opaque group without a filter or blend mode: an isolated group.
    fn default() -> Self {
        Self {
            opacity: 1.,
            blur_radius: ScaledPixels(0.),
            color_filters: SmallVec::new(),
            blend_mode: BlendMode::Normal,
            content_mask: ContentMask::default(),
            backdrop: None,
            masks_parent: None,
        }
    }
}

impl GroupEffects {
    /// Whether the image is filtered.
    pub fn has_filter(&self) -> bool {
        self.blur_radius.0 > 0. || !self.color_filters.is_empty()
    }

    /// The color filters renderers apply.
    pub fn applied_color_filters(&self) -> &[ColorFilter] {
        &self.color_filters[..self.color_filters.len().min(MAX_COLOR_FILTERS)]
    }
}

/// A group drawn offscreen and composited onto the scene as one image
/// ([`Scene::push_group`]).
pub struct CompositedGroup {
    /// Assigned by the scene.
    pub order: DrawOrder,
    /// What the image covers in the target: the group's content, spread by its blur
    /// and clipped by its content mask, in whole device pixels.
    pub bounds: Bounds<ScaledPixels>,
    /// How the image is composited.
    pub effects: GroupEffects,
    /// What the group draws, finished: its primitives and its own groups, in the
    /// coordinates of its image, whose origin is `bounds.origin` in the target.
    /// Shader draws in it refer to the sources and uniforms of the scene the window
    /// painted.
    pub scene: Scene,
    /// Masks its image after its filters, before its opacity (CSS Masking 1); its
    /// bounds lie inside the mask's.
    pub mask: Option<GroupMask>,
}

impl CompositedGroup {
    /// Narrows its bounds to `clip`, in whole pixels, keeping its scene where it was
    /// drawn (the scene moves with the image's origin).
    fn clip(&mut self, clip: &Bounds<ScaledPixels>) {
        let clipped = round_out(self.bounds.intersect(clip));
        let shift = self.bounds.origin - clipped.origin;
        if shift != Point::default() {
            self.scene.translate(shift);
        }
        self.bounds = clipped;
    }

    /// The size of its image, device pixels, on a target of `target` device pixels:
    /// its bounds (whole pixels) up to the target's right and bottom edges. Its scene
    /// draws in the image's coordinates, from `bounds.origin`.
    pub fn image_size(&self, target: Size<DevicePixels>) -> Size<DevicePixels> {
        let right = self.bounds.bottom_right().x.0.min(target.width.0 as f32);
        let bottom = self.bounds.bottom_right().y.0.min(target.height.0 as f32);
        Size {
            width: DevicePixels((right - self.bounds.origin.x.0).round().max(0.) as i32),
            height: DevicePixels((bottom - self.bounds.origin.y.0).round().max(0.) as i32),
        }
    }

    /// What compositing reads and draws in a target of `viewport` device pixels: its
    /// bounds within the viewport. `None` when that is empty.
    pub fn target_region(&self, viewport: Size<DevicePixels>) -> Option<Bounds<ScaledPixels>> {
        let viewport = Bounds::new(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            Size {
                width: ScaledPixels(viewport.width.0 as f32),
                height: ScaledPixels(viewport.height.0 as f32),
            },
        );
        let region = self.bounds.intersect(&viewport);
        (!region.is_empty()).then_some(region)
    }

    /// The uniforms renderers composite the group with. `source_region` is where its
    /// (blurred) image sits in the target, `backdrop_region` where the copy of what
    /// is behind it does and `mask_region` where its mask's image does (origin and
    /// size, device pixels; unused without a mask); `viewport` is the target's size.
    pub fn composite_uniforms(
        &self,
        source_region: [f32; 4],
        backdrop_region: [f32; 4],
        mask_region: [f32; 4],
        viewport: [f32; 2],
    ) -> GroupCompositeUniforms {
        let rect =
            |b: Bounds<ScaledPixels>| [b.origin.x.0, b.origin.y.0, b.size.width.0, b.size.height.0];
        let filters = self.effects.applied_color_filters();
        let mut filter_kinds = [[0; 4]; 2];
        let mut filter_amounts = [[0.; 4]; 2];
        for (i, filter) in filters.iter().enumerate() {
            let (kind, amount) = filter.gpu();
            filter_kinds[i / 4][i % 4] = kind;
            filter_amounts[i / 4][i % 4] = amount;
        }
        GroupCompositeUniforms {
            bounds: rect(self.bounds),
            content_mask: rect(self.effects.content_mask.bounds),
            source_region,
            backdrop_region,
            filter_kinds,
            filter_amounts,
            viewport_size: viewport,
            opacity: self.effects.opacity.clamp(0., 1.),
            blend_mode: self.effects.blend_mode as u32,
            filter_count: filters.len() as u32,
            mask_mode: self.mask.as_ref().map_or(0, |mask| mask.mode.gpu()),
            pad: [0; 2],
            mask_region,
        }
    }
}

/// What renderers composite a group with (`GroupCompositeParams` in the Metal and
/// WGSL shaders): 176 bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct GroupCompositeUniforms {
    /// What to draw: origin and size, device pixels.
    pub bounds: [f32; 4],
    pub content_mask: [f32; 4],
    /// Where the group's image sits in the target.
    pub source_region: [f32; 4],
    /// Where the copy of what is behind the group sits in the target.
    pub backdrop_region: [f32; 4],
    /// [`ColorFilter::gpu`] codes, four to a vector; [`Self::filter_count`] apply.
    pub filter_kinds: [[u32; 4]; 2],
    pub filter_amounts: [[f32; 4]; 2],
    pub viewport_size: [f32; 2],
    pub opacity: f32,
    /// [`BlendMode`] as `u32`.
    pub blend_mode: u32,
    pub filter_count: u32,
    /// 0 without a mask, else [`MaskMode::gpu`].
    pub mask_mode: u32,
    pub pad: [u32; 2],
    /// Where the mask's image sits in the target.
    pub mask_region: [f32; 4],
}

impl GroupCompositeUniforms {
    /// Its bytes, for uniform buffers.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `repr(C)` and made of 4-byte scalars, so it has no padding.
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self).cast::<u8>(),
                std::mem::size_of::<Self>(),
            )
        }
    }
}

/// A group being painted ([`Scene::push_group`]).
struct OpenGroup {
    effects: GroupEffects,
    scene: Scene,
    /// Its mask, once painted ([`GroupEffects::masks_parent`]).
    mask: Option<GroupMask>,
    /// A mask was painted and drew nothing: so does the group.
    masked_out: bool,
    /// What was painted, in order, while the group may still be painted directly
    /// instead of offscreen.
    operations: Option<Vec<GroupOperation>>,
}

enum GroupOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
    /// The group at this index of the open group's scene.
    Group(usize),
}

impl OpenGroup {
    fn new(effects: GroupEffects) -> Self {
        // A translucent group's backdrop and content fade together: offscreen. So
        // does a mask's image.
        let direct = !effects.has_filter()
            && effects.blend_mode == BlendMode::Normal
            && (effects.backdrop.is_none() || effects.opacity >= 1.)
            && effects.masks_parent.is_none();
        Self {
            mask: None,
            masked_out: false,
            effects,
            scene: Scene {
                in_group: true,
                ..Scene::default()
            },
            operations: direct.then(Vec::new),
        }
    }

    /// Whether painting it directly fades each primitive.
    fn fades(&self) -> bool {
        self.effects.opacity < 1.
    }

    /// Stops recording once fading each primitive would differ from fading the whole:
    /// something overlaps something else.
    fn check_overlap(&mut self) {
        if self.fades() && self.scene.overlapping {
            self.operations = None;
        }
    }

    fn insert(&mut self, primitive: Primitive) {
        if self.fades() && !primitive.fades_alone() {
            self.operations = None;
        }
        if let Some(operations) = &mut self.operations {
            operations.push(GroupOperation::Primitive(primitive.clone()));
        }
        self.scene.place(primitive);
        self.check_overlap();
    }

    fn start_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        // What a layer draws may overlap without the scene knowing.
        if self.fades() {
            self.operations = None;
        }
        if let Some(operations) = &mut self.operations {
            operations.push(GroupOperation::StartLayer(bounds));
        }
        self.scene.start_layer(bounds);
    }

    fn end_layer(&mut self) {
        if let Some(operations) = &mut self.operations {
            operations.push(GroupOperation::EndLayer);
        }
        self.scene.end_layer();
    }

    fn add_group(&mut self, group: CompositedGroup) {
        // A blending group blends with this group's image only: it isolates it.
        if group.effects.blend_mode != BlendMode::Normal {
            self.operations = None;
        }
        let index = self.scene.groups.len();
        self.scene.place_group(group);
        if self.scene.groups.len() > index
            && let Some(operations) = &mut self.operations
        {
            operations.push(GroupOperation::Group(index));
        }
        self.check_overlap();
    }

    /// Its mask: `image` is the mask group's, `None` when it drew nothing. A masked
    /// group draws offscreen; a later mask replaces an earlier one.
    fn set_mask(&mut self, mode: MaskMode, image: Option<CompositedGroup>) {
        self.operations = None;
        self.mask = image.map(|image| GroupMask {
            mode,
            image: Box::new(image),
        });
        self.masked_out = self.mask.is_none();
    }

    fn close(mut self) -> ClosedGroup {
        if self.effects.opacity <= 0. || self.masked_out {
            return ClosedGroup::Nothing;
        }
        let backdrop = self
            .effects
            .backdrop
            .map(|filter| filter.bounds.intersect(&filter.content_mask.bounds))
            .filter(|bounds| !bounds.is_empty());
        let drawn = match (self.scene.drawn_bounds, backdrop) {
            (Some(drawn), Some(backdrop)) => drawn.union(&backdrop),
            (drawn, backdrop) => match drawn.or(backdrop) {
                Some(drawn) => drawn,
                None => return ClosedGroup::Nothing,
            },
        };
        if let Some(operations) = self.operations.take() {
            return ClosedGroup::Direct {
                backdrop: self.effects.backdrop,
                operations,
                groups: std::mem::take(&mut self.scene.groups)
                    .into_iter()
                    .map(Some)
                    .collect(),
                opacity: self.effects.opacity,
                content_mask: self.effects.content_mask,
            };
        }
        self.scene.finish();
        let mut bounds = drawn
            .dilate(self.effects.blur_radius * 3.)
            .intersect(&self.effects.content_mask.bounds);
        // Outside its mask's image a masked group draws nothing.
        if let Some(mask) = &self.mask {
            bounds = bounds.intersect(&mask.image.bounds);
        }
        let bounds = round_out(bounds);
        if self.mask.is_some() && bounds.is_empty() {
            return ClosedGroup::Nothing;
        }
        // The image covers `bounds`: draw the scene in its coordinates.
        self.scene.translate(-bounds.origin);
        ClosedGroup::Offscreen(CompositedGroup {
            order: 0,
            bounds,
            effects: self.effects,
            scene: self.scene,
            mask: self.mask,
        })
    }
}

/// The whole device pixels `bounds` touches.
fn round_out(bounds: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
    if bounds.is_empty() {
        return Bounds::default();
    }
    Bounds::from_corners(
        point(
            ScaledPixels(bounds.origin.x.0.floor()),
            ScaledPixels(bounds.origin.y.0.floor()),
        ),
        point(
            ScaledPixels(bounds.bottom_right().x.0.ceil()),
            ScaledPixels(bounds.bottom_right().y.0.ceil()),
        ),
    )
}

/// A closed group, as it goes into the scene or group around it.
enum ClosedGroup {
    /// It draws nothing.
    Nothing,
    /// Painted directly: its backdrop (only when opaque), then each primitive and
    /// group faded by `opacity` and clipped by `content_mask`.
    Direct {
        backdrop: Option<BackdropFilter>,
        operations: Vec<GroupOperation>,
        groups: Vec<Option<CompositedGroup>>,
        opacity: f32,
        content_mask: ContentMask<ScaledPixels>,
    },
    Offscreen(CompositedGroup),
}

impl ClosedGroup {
    fn apply(self, target: &mut impl GroupTarget) {
        match self {
            ClosedGroup::Nothing => {}
            ClosedGroup::Offscreen(group) => target.add_group(group),
            ClosedGroup::Direct {
                backdrop,
                operations,
                mut groups,
                opacity,
                content_mask,
            } => {
                if let Some(filter) = backdrop {
                    target.add_primitive(Primitive::BackdropFilter(filter));
                }
                for operation in operations {
                    match operation {
                        GroupOperation::Primitive(mut primitive) => {
                            if opacity < 1. {
                                primitive.fade(opacity);
                            }
                            primitive.clip(&content_mask);
                            target.add_primitive(primitive);
                        }
                        GroupOperation::StartLayer(bounds) => target.add_start_layer(bounds),
                        GroupOperation::EndLayer => target.add_end_layer(),
                        GroupOperation::Group(index) => {
                            if let Some(mut group) = groups.get_mut(index).and_then(Option::take) {
                                group.effects.opacity *= opacity;
                                let mask = &mut group.effects.content_mask.bounds;
                                *mask = mask.intersect(&content_mask.bounds);
                                group.clip(&content_mask.bounds);
                                target.add_group(group);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Where a closed group goes: the window's scene or the group around it.
trait GroupTarget {
    fn add_primitive(&mut self, primitive: Primitive);
    fn add_start_layer(&mut self, bounds: Bounds<ScaledPixels>);
    fn add_end_layer(&mut self);
    fn add_group(&mut self, group: CompositedGroup);
}

impl GroupTarget for Scene {
    // The window's scene recorded these paint operations already.
    fn add_primitive(&mut self, primitive: Primitive) {
        self.place(primitive);
    }

    fn add_start_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        self.start_layer(bounds);
    }

    fn add_end_layer(&mut self) {
        self.end_layer();
    }

    fn add_group(&mut self, group: CompositedGroup) {
        self.place_group(group);
    }
}

impl GroupTarget for OpenGroup {
    fn add_primitive(&mut self, primitive: Primitive) {
        self.insert(primitive);
    }

    fn add_start_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        self.start_layer(bounds);
    }

    fn add_end_layer(&mut self) {
        self.end_layer();
    }

    fn add_group(&mut self, group: CompositedGroup) {
        OpenGroup::add_group(self, group);
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

/// The regions to copy a batch of paths from the path intermediate texture to the target,
/// each pixel at most once (paths can be translucent): the paths' clipped bounds, merged
/// where they overlap. Paths of one draw order overlap when they are painted in one
/// [`Window::paint_layer`](crate::Window::paint_layer), so the order alone does not make
/// them disjoint. Batches of more than 256 paths copy their union.
pub fn path_copy_bounds(paths: &[Path<ScaledPixels>]) -> Vec<Bounds<ScaledPixels>> {
    let mut regions: Vec<Bounds<ScaledPixels>> = Vec::new();
    if paths.len() > 256 {
        let mut union = paths[0].clipped_bounds();
        for path in &paths[1..] {
            union = union.union(&path.clipped_bounds());
        }
        regions.push(union);
        return regions;
    }
    for path in paths {
        let mut bounds = path.clipped_bounds();
        if bounds.is_empty() {
            continue;
        }
        // Merge every region the new one overlaps, again until none overlaps it.
        loop {
            let Some(i) = regions.iter().position(|r| r.intersects(&bounds)) else {
                break;
            };
            bounds = bounds.union(&regions.swap_remove(i));
        }
        regions.push(bounds);
    }
    regions
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
        let full = ContentMask::from(bounds(0., 0., SIZE.0 as f32, SIZE.1 as f32));
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
        let full = ContentMask::from(bounds(0., 0., SIZE.0 as f32, SIZE.1 as f32));
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

/// Compositing groups, for renderer tests: every backend renders it and must hit the
/// same probe colors. Over a white page:
/// - a half-opaque group of overlapping red and blue squares (the blue covers the
///   red where they overlap, as one image);
/// - a half-opaque group whose backdrop filter blackens what the group drew, but not
///   the red square behind the group (the group is its Backdrop Root);
/// - nested half-opaque groups;
/// - groups filtered with grayscale, invert and a blur;
/// - an inverting group whose image starts with its backdrop, darkened.
#[cfg(any(test, feature = "test-support", feature = "bench-support"))]
#[doc(hidden)]
pub mod compositing_group_test_scene {
    use super::*;
    pub use crate::retained_shapes_test_scene::{Probe, SIZE, check};
    use crate::{Rgba, size};

    fn color(r: f32, g: f32, b: f32, a: f32) -> Hsla {
        Rgba { r, g, b, a }.into()
    }

    fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(w), ScaledPixels(h)),
        )
    }

    fn full() -> ContentMask<ScaledPixels> {
        ContentMask::from(bounds(0., 0., SIZE.0 as f32, SIZE.1 as f32))
    }

    fn quad(scene: &mut Scene, x: f32, y: f32, w: f32, h: f32, fill: Hsla) {
        scene.insert_primitive(Quad {
            bounds: bounds(x, y, w, h),
            content_mask: full(),
            background: fill.into(),
            ..Default::default()
        });
    }

    fn group(opacity: f32) -> GroupEffects {
        GroupEffects {
            opacity,
            content_mask: full(),
            ..GroupEffects::default()
        }
    }

    /// The scene and the probes every renderer must hit.
    pub fn scene() -> (Scene, Vec<Probe>) {
        let mut scene = Scene::default();
        let (red, green, blue) = (
            color(1., 0., 0., 1.),
            color(0., 1., 0., 1.),
            color(0., 0., 1., 1.),
        );
        quad(
            &mut scene,
            0.,
            0.,
            SIZE.0 as f32,
            SIZE.1 as f32,
            color(1., 1., 1., 1.),
        );

        // Overlapping squares at half opacity: one image.
        scene.push_group(group(0.5));
        quad(&mut scene, 2., 2., 16., 16., red);
        quad(&mut scene, 10., 10., 16., 16., blue);
        scene.pop_group();

        // A red square behind a half-opaque group that draws a green square and
        // blackens its own drawing with a backdrop filter.
        quad(&mut scene, 34., 2., 28., 28., red);
        scene.push_group(group(0.5));
        quad(&mut scene, 36., 4., 12., 12., green);
        scene.insert_primitive(BackdropFilter {
            bounds: bounds(36., 4., 24., 12.),
            content_mask: full(),
            brightness: 0.,
            ..Default::default()
        });
        scene.pop_group();

        // Nested: a half-opaque group of green and blue squares over a red one, all in
        // a half-opaque group.
        scene.push_group(group(0.5));
        quad(&mut scene, 6., 38., 8., 8., red);
        scene.push_group(group(0.5));
        quad(&mut scene, 2., 34., 16., 16., green);
        quad(&mut scene, 10., 42., 16., 16., blue);
        scene.pop_group();
        scene.pop_group();

        // Filters: grayscale red, inverted green, blurred black.
        scene.push_group(GroupEffects {
            color_filters: [ColorFilter::Grayscale(1.)].into_iter().collect(),
            ..group(1.)
        });
        quad(&mut scene, 34., 34., 12., 12., red);
        scene.pop_group();
        scene.push_group(GroupEffects {
            color_filters: [ColorFilter::Invert(1.)].into_iter().collect(),
            ..group(1.)
        });
        quad(&mut scene, 50., 34., 12., 12., green);
        scene.pop_group();
        scene.push_group(GroupEffects {
            blur_radius: ScaledPixels(2.),
            ..group(1.)
        });
        quad(&mut scene, 36., 50., 8., 8., color(0., 0., 0., 1.));
        scene.pop_group();

        // A red square behind an inverting group whose image starts with its backdrop,
        // darkened: the group's filter applies to the backdrop too.
        quad(&mut scene, 50., 50., 12., 12., red);
        scene.push_group(GroupEffects {
            color_filters: [ColorFilter::Invert(1.)].into_iter().collect(),
            backdrop: Some(BackdropFilter {
                bounds: bounds(52., 52., 8., 8.),
                content_mask: full(),
                brightness: 0.5,
                ..Default::default()
            }),
            ..group(1.)
        });
        scene.pop_group();
        scene.finish();

        let probe = |what, x, y, rgba| Probe { what, x, y, rgba };
        let gray = (0.2126f32 * 255.).round() as u8;
        let probes = vec![
            probe("red alone, half opaque", 4, 4, [255, 128, 128, 255]),
            probe("overlap shows only blue", 14, 14, [128, 128, 255, 255]),
            probe("blue alone, half opaque", 24, 24, [128, 128, 255, 255]),
            probe(
                "the backdrop filter blackens the group's green",
                40,
                8,
                [128, 0, 0, 255],
            ),
            probe(
                "the backdrop filter leaves the red behind the group",
                54,
                8,
                [255, 0, 0, 255],
            ),
            probe("nested: green alone", 4, 36, [191, 255, 191, 255]),
            probe("nested: inner group over red", 8, 40, [191, 191, 128, 255]),
            probe(
                "nested: blue over green over red",
                12,
                44,
                [191, 128, 191, 255],
            ),
            probe("grayscale red", 40, 40, [gray, gray, gray, 255]),
            probe("inverted green", 56, 40, [255, 0, 255, 255]),
            // A gaussian of standard deviation 2 over the 8 px square: 90% at the
            // center pixel, 10% at the pixel whose center is 2.5 px outside.
            probe("blurred square's center", 40, 54, [26, 26, 26, 255]),
            probe(
                "blur spreads beyond the square",
                40,
                47,
                [230, 230, 230, 255],
            ),
            probe("white beside the groups", 30, 30, [255, 255, 255, 255]),
            probe(
                "darkened backdrop, inverted by its group",
                56,
                56,
                [127, 255, 255, 255],
            ),
            probe("red beside the group's backdrop", 51, 51, [255, 0, 0, 255]),
        ];
        (scene, probes)
    }
}

/// Paths and transformed quads in an offscreen group away from the target's origin,
/// for renderer tests: the group draws them in its own image's coordinates. Over a
/// white page, a half-opaque group holds a red triangle (a path) and a blue square
/// turned 45 degrees over it.
#[cfg(any(test, feature = "test-support", feature = "bench-support"))]
#[doc(hidden)]
pub mod compositing_group_content_test_scene {
    use super::*;
    pub use crate::retained_shapes_test_scene::{Probe, SIZE, check};
    use crate::{Rgba, px, radians, size};

    /// The scene and the probes every renderer must hit.
    pub fn scene() -> (Scene, Vec<Probe>) {
        let full = ContentMask::from(Bounds::new(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            size(ScaledPixels(SIZE.0 as f32), ScaledPixels(SIZE.1 as f32)),
        ));
        let color = |r: f32, g: f32, b: f32| -> Hsla { Rgba { r, g, b, a: 1. }.into() };
        let mut scene = Scene::default();
        scene.insert_primitive(Quad {
            bounds: full.bounds,
            content_mask: full,
            background: color(1., 1., 1.).into(),
            ..Default::default()
        });
        scene.push_group(GroupEffects {
            opacity: 0.5,
            content_mask: full,
            ..GroupEffects::default()
        });
        let mut triangle = Path::new(point(px(20.), px(20.)));
        triangle.line_to(point(px(56.), px(20.)));
        triangle.line_to(point(px(20.), px(56.)));
        triangle.line_to(point(px(20.), px(20.)));
        let mut triangle = triangle.scale(1.);
        triangle.content_mask = full;
        triangle.color = color(1., 0., 0.).into();
        scene.insert_primitive(triangle);
        let center = point(ScaledPixels(36.), ScaledPixels(36.));
        scene.insert_primitive(Quad {
            bounds: Bounds::new(
                point(ScaledPixels(31.), ScaledPixels(31.)),
                size(ScaledPixels(10.), ScaledPixels(10.)),
            ),
            content_mask: full,
            background: color(0., 0., 1.).into(),
            transformation: TransformationMatrix::unit()
                .translate(center)
                .rotate(radians(std::f32::consts::FRAC_PI_4))
                .translate(-center),
            ..Default::default()
        });
        scene.pop_group();
        scene.finish();
        assert_eq!(scene.groups.len(), 1, "drawn offscreen");
        let probe = |what, x, y, rgba| Probe { what, x, y, rgba };
        let probes = vec![
            probe("the triangle, half opaque", 24, 24, [255, 128, 128, 255]),
            probe("the turned square over it", 36, 36, [128, 128, 255, 255]),
            probe("the turned square's tip", 36, 30, [128, 128, 255, 255]),
            probe("outside the square's corner", 31, 31, [255, 128, 128, 255]),
            probe("outside the triangle", 50, 50, [255, 255, 255, 255]),
            probe("left of the group", 10, 30, [255, 255, 255, 255]),
        ];
        (scene, probes)
    }
}

/// Group masks ([`GroupMask`]), for renderer tests. Over a white page: a red square
/// whose mask is opaque on its left half and half opaque on its right (alpha); a
/// blue square masked by white and 50% gray (luminance), inside a multiplying group
/// drawn offscreen, so the mask moves with that group's image; a green square
/// masked by a smaller square; and a black square whose mask draws nothing.
#[cfg(any(test, feature = "test-support", feature = "bench-support"))]
#[doc(hidden)]
pub mod group_mask_test_scene {
    use super::*;
    pub use crate::retained_shapes_test_scene::{Probe, SIZE, check};
    use crate::{Rgba, size};

    /// The scene and the probes every renderer must hit.
    pub fn scene() -> (Scene, Vec<Probe>) {
        let full = ContentMask::from(Bounds::new(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            size(ScaledPixels(SIZE.0 as f32), ScaledPixels(SIZE.1 as f32)),
        ));
        let quad = |x: f32, y: f32, w: f32, h: f32, r: f32, g: f32, b: f32, a: f32| Quad {
            bounds: Bounds::new(
                point(ScaledPixels(x), ScaledPixels(y)),
                size(ScaledPixels(w), ScaledPixels(h)),
            ),
            content_mask: full,
            background: Hsla::from(Rgba { r, g, b, a }).into(),
            ..Default::default()
        };
        let group = GroupEffects {
            content_mask: full,
            ..GroupEffects::default()
        };
        let mask = |mode| GroupEffects {
            masks_parent: Some(mode),
            ..group.clone()
        };
        let mut scene = Scene::default();
        scene.insert_primitive(quad(0., 0., 64., 64., 1., 1., 1., 1.));
        // Alpha: the mask after the content.
        scene.push_group(group.clone());
        scene.insert_primitive(quad(4., 4., 24., 24., 1., 0., 0., 1.));
        scene.push_group(mask(MaskMode::Alpha));
        scene.insert_primitive(quad(4., 4., 12., 24., 0., 0., 0., 1.));
        scene.insert_primitive(quad(16., 4., 12., 24., 0., 0., 0., 0.5));
        scene.pop_group();
        scene.pop_group();
        // Luminance, the mask before the content, inside an offscreen group.
        scene.push_group(GroupEffects {
            blend_mode: BlendMode::Multiply,
            ..group.clone()
        });
        scene.push_group(group.clone());
        scene.push_group(mask(MaskMode::Luminance));
        scene.insert_primitive(quad(36., 4., 12., 24., 1., 1., 1., 1.));
        scene.insert_primitive(quad(48., 4., 12., 24., 0.5, 0.5, 0.5, 1.));
        scene.pop_group();
        scene.insert_primitive(quad(36., 4., 24., 24., 0., 0., 1., 1.));
        scene.pop_group();
        scene.pop_group();
        // A mask smaller than the content.
        scene.push_group(group.clone());
        scene.insert_primitive(quad(4., 36., 24., 24., 0., 1., 0., 1.));
        scene.push_group(mask(MaskMode::Alpha));
        scene.insert_primitive(quad(10., 42., 12., 12., 0., 0., 0., 1.));
        scene.pop_group();
        scene.pop_group();
        // A mask that draws nothing.
        scene.push_group(group.clone());
        scene.insert_primitive(quad(36., 36., 24., 24., 0., 0., 0., 1.));
        scene.push_group(mask(MaskMode::Alpha));
        scene.pop_group();
        scene.pop_group();
        scene.finish();
        let probe = |what, x, y, rgba| Probe { what, x, y, rgba };
        let probes = vec![
            probe(
                "red under the opaque half of its mask",
                8,
                16,
                [255, 0, 0, 255],
            ),
            probe(
                "red under the half-opaque half",
                22,
                16,
                [255, 128, 128, 255],
            ),
            probe("blue under white (luminance 1)", 40, 16, [0, 0, 255, 255]),
            probe("blue under 50% gray", 54, 16, [127, 127, 255, 255]),
            probe("green inside its smaller mask", 16, 48, [0, 255, 0, 255]),
            probe(
                "green outside its mask: not drawn",
                6,
                38,
                [255, 255, 255, 255],
            ),
            probe(
                "black under an empty mask: not drawn",
                48,
                48,
                [255, 255, 255, 255],
            ),
        ];
        (scene, probes)
    }
}

/// Rounded content masks (an `overflow` clip that follows a rounded border), for
/// renderer tests: four masks with 12 px corners over white, each overfilled by one
/// kind of primitive (a quad, a sharp drop shadow, a path, and a quad in a half
/// opaque offscreen group), probed along each corner's diagonal inside and outside
/// its curve.
#[cfg(any(test, feature = "test-support", feature = "bench-support"))]
#[doc(hidden)]
pub mod rounded_clip_test_scene {
    use super::*;
    pub use crate::retained_shapes_test_scene::{Probe, SIZE, check};
    use crate::{Rgba, px, size};

    /// The corner radius of every mask.
    pub const RADIUS: f32 = 12.;

    /// The masks (x, y, width, height): one per primitive kind.
    pub const MASKS: [[f32; 4]; 4] = [
        [2., 2., 28., 28.],
        [34., 2., 28., 28.],
        [2., 34., 28., 28.],
        [34., 34., 28., 28.],
    ];

    fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(w), ScaledPixels(h)),
        )
    }

    /// The scene and the probes every renderer must hit.
    pub fn scene() -> (Scene, Vec<Probe>) {
        let full = ContentMask::from(bounds(0., 0., SIZE.0 as f32, SIZE.1 as f32));
        let color = |r: f32, g: f32, b: f32| -> Hsla { Rgba { r, g, b, a: 1. }.into() };
        let mut scene = Scene::default();
        scene.insert_primitive(Quad {
            bounds: full.bounds,
            content_mask: full,
            background: color(1., 1., 1.).into(),
            ..Default::default()
        });
        let masks = MASKS.map(|[x, y, w, h]| ContentMask {
            bounds: bounds(x, y, w, h),
            corner_radii: Corners::all(ScaledPixels(RADIUS)),
        });
        // Each primitive overfills its mask by 4 px on every side.
        let over = |m: &ContentMask<ScaledPixels>| {
            let b = m.bounds;
            bounds(
                b.origin.x.0 - 4.,
                b.origin.y.0 - 4.,
                b.size.width.0 + 8.,
                b.size.height.0 + 8.,
            )
        };
        scene.insert_primitive(Quad {
            bounds: over(&masks[0]),
            content_mask: masks[0],
            background: color(1., 0., 0.).into(),
            ..Default::default()
        });
        scene.insert_primitive(Shadow {
            order: 0,
            blur_radius: ScaledPixels(0.),
            bounds: over(&masks[1]),
            corner_radii: Corners::default(),
            content_mask: masks[1],
            color: color(0., 1., 0.),
            element_bounds: over(&masks[1]),
            element_corner_radii: Corners::default(),
            inset: 0,
            pad: 0,
            transformation: TransformationMatrix::unit(),
        });
        let b = over(&masks[2]);
        let (x0, y0) = (b.origin.x.0, b.origin.y.0);
        let (x1, y1) = (x0 + b.size.width.0, y0 + b.size.height.0);
        let mut square = Path::new(point(px(x0), px(y0)));
        square.line_to(point(px(x1), px(y0)));
        square.line_to(point(px(x1), px(y1)));
        square.line_to(point(px(x0), px(y1)));
        square.line_to(point(px(x0), px(y0)));
        let mut square = square.scale(1.);
        square.content_mask = masks[2];
        square.color = color(0., 0., 1.).into();
        scene.insert_primitive(square);
        scene.push_group(GroupEffects {
            opacity: 0.5,
            content_mask: full,
            ..GroupEffects::default()
        });
        scene.insert_primitive(Quad {
            bounds: over(&masks[3]),
            content_mask: masks[3],
            background: color(1., 0., 0.).into(),
            ..Default::default()
        });
        scene.pop_group();
        scene.finish();

        let inside_colors = [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 128, 128, 255],
        ];
        let mut probes = Vec::new();
        for ([x, y, w, h], rgba) in MASKS.into_iter().zip(inside_colors) {
            let corners = [
                ([x + RADIUS, y + RADIUS], [-1., -1.]),
                ([x + w - RADIUS, y + RADIUS], [1., -1.]),
                ([x + w - RADIUS, y + h - RADIUS], [1., 1.]),
                ([x + RADIUS, y + h - RADIUS], [-1., 1.]),
            ];
            for (center, [dx, dy]) in corners {
                for step in 1..18 {
                    let t = step as f32 / 12.;
                    let px = (center[0] + dx * t * RADIUS).floor();
                    let py = (center[1] + dy * t * RADIUS).floor();
                    // Inside the circle, or outside it but still inside the mask's
                    // rectangle (the part a square clip would wrongly draw), clear of the
                    // antialiased edge.
                    let distance = |d: f32| {
                        let cx = px + 0.5 + dx * d - center[0];
                        let cy = py + 0.5 + dy * d - center[1];
                        (cx * cx + cy * cy).sqrt() - RADIUS
                    };
                    let in_rect = px >= x && py >= y && px + 1. <= x + w && py + 1. <= y + h;
                    if !in_rect || distance(0.).abs() < 1.5 {
                        continue;
                    }
                    let here = distance(0.) < 0.;
                    probes.push(Probe {
                        what: if here {
                            "inside a rounded clip"
                        } else {
                            "outside a rounded clip's curve"
                        },
                        x: px as u32,
                        y: py as u32,
                        rgba: if here { rgba } else { [255, 255, 255, 255] },
                    });
                }
            }
        }
        (scene, probes)
    }
}

/// Shaped corners (CSS `corner-shape`), for renderer tests: two red quads over white,
/// each corner a different shape, probed along each corner's diagonal well inside
/// and outside its curve.
#[cfg(any(test, feature = "test-support", feature = "bench-support"))]
#[doc(hidden)]
pub mod corner_shape_test_scene {
    use super::*;
    pub use crate::retained_shapes_test_scene::{Probe, SIZE, check};
    use crate::{Rgba, size};

    const RADIUS: f32 = 12.;

    /// (bounds, shapes: top left, top right, bottom right, bottom left).
    pub const QUADS: [([f32; 4], [f32; 4]); 2] = [
        ([2., 2., 60., 28.], [1., 2., 0., -1.]),
        (
            [2., 34., 60., 28.],
            [f32::INFINITY, f32::NEG_INFINITY, 4., -2.],
        ),
    ];

    /// Whether `u` (from a corner's center toward its outer corner, over its radius)
    /// is inside a corner of shape `k`.
    pub fn inside(u: [f32; 2], k: f32) -> bool {
        let [x, y] = u;
        if x <= 0. || y <= 0. {
            // Beside the corner's box: the straight edges.
            return true;
        }
        if k == f32::INFINITY {
            return x <= 1. && y <= 1.;
        }
        if k == f32::NEG_INFINITY {
            return false;
        }
        if k >= 0. {
            let n = 2f32.powf(k);
            x.powf(n) + y.powf(n) <= 1.
        } else {
            let m = 2f32.powf(-k);
            x <= 1. && y <= 1. && (1. - x).powf(m) + (1. - y).powf(m) >= 1.
        }
    }

    /// The scene and the probes every renderer must hit.
    pub fn scene() -> (Scene, Vec<Probe>) {
        let full = ContentMask::from(Bounds::new(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            size(ScaledPixels(SIZE.0 as f32), ScaledPixels(SIZE.1 as f32)),
        ));
        let color = |r: f32, g: f32, b: f32| -> Hsla { Rgba { r, g, b, a: 1. }.into() };
        let mut scene = Scene::default();
        scene.insert_primitive(Quad {
            bounds: full.bounds,
            content_mask: full,
            background: color(1., 1., 1.).into(),
            ..Default::default()
        });
        let mut probes = Vec::new();
        for ([x, y, w, h], [tl, tr, br, bl]) in QUADS {
            scene.insert_primitive(Quad {
                bounds: Bounds::new(
                    point(ScaledPixels(x), ScaledPixels(y)),
                    size(ScaledPixels(w), ScaledPixels(h)),
                ),
                content_mask: full,
                background: color(1., 0., 0.).into(),
                corner_radii: Corners::all(ScaledPixels(RADIUS)),
                corner_shapes: CornerShapes {
                    top_left: tl,
                    top_right: tr,
                    bottom_right: br,
                    bottom_left: bl,
                }
                .gpu(),
                ..Default::default()
            });
            // Each corner: its center and the direction to its outer corner.
            let corners = [
                ([x + RADIUS, y + RADIUS], [-1., -1.], tl),
                ([x + w - RADIUS, y + RADIUS], [1., -1.], tr),
                ([x + w - RADIUS, y + h - RADIUS], [1., 1.], br),
                ([x + RADIUS, y + h - RADIUS], [-1., 1.], bl),
            ];
            for (center, [dx, dy], k) in corners {
                for step in 1..12 {
                    let t = step as f32 / 12.;
                    let px = (center[0] + dx * t * RADIUS).floor();
                    let py = (center[1] + dy * t * RADIUS).floor();
                    // The pixel's center, and points 1.5 px either way along the diagonal:
                    // keep the probe only when all three agree (clear of the antialiased edge).
                    let at = |d: f32| {
                        let cx = px + 0.5 + dx * d;
                        let cy = py + 0.5 + dy * d;
                        inside(
                            [
                                (cx - center[0]) * dx / RADIUS,
                                (cy - center[1]) * dy / RADIUS,
                            ],
                            k,
                        )
                    };
                    let here = at(0.);
                    if at(-1.5) != here || at(1.5) != here {
                        continue;
                    }
                    probes.push(Probe {
                        what: if here {
                            "inside a shaped corner"
                        } else {
                            "outside a shaped corner"
                        },
                        x: px as u32,
                        y: py as u32,
                        rgba: if here {
                            [255, 0, 0, 255]
                        } else {
                            [255, 255, 255, 255]
                        },
                    });
                }
            }
        }
        scene.finish();
        (scene, probes)
    }
}

/// Scenes for cost measurements of compositing groups.
#[cfg(any(test, feature = "test-support", feature = "bench-support"))]
#[doc(hidden)]
pub mod compositing_group_bench_scene {
    use super::*;
    use crate::{Rgba, size};

    fn quad(x: f32, y: f32, w: f32, h: f32, a: f32, mask: ContentMask<ScaledPixels>) -> Quad {
        Quad {
            bounds: Bounds::new(
                point(ScaledPixels(x), ScaledPixels(y)),
                size(ScaledPixels(w), ScaledPixels(h)),
            ),
            content_mask: mask,
            background: Hsla::from(Rgba {
                r: 0.3,
                g: 0.5,
                b: 0.9,
                a,
            })
            .into(),
            ..Default::default()
        }
    }

    fn mask(width: f32, height: f32) -> ContentMask<ScaledPixels> {
        ContentMask::from(Bounds::new(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            size(ScaledPixels(width), ScaledPixels(height)),
        ))
    }

    /// `count` quads in rows (each overlapping its neighbor), as a busy UI paints:
    /// no groups.
    pub fn flat(count: usize, width: f32, height: f32) -> Scene {
        let mut scene = Scene::default();
        let mask = mask(width, height);
        let columns = (width / 20.).max(1.) as usize;
        for i in 0..count {
            let (x, y) = (
                (i % columns) as f32 * 20.,
                (i / columns) as f32 * 12. % height,
            );
            scene.insert_primitive(quad(x, y, 24., 16., 1., mask));
        }
        scene.finish();
        scene
    }

    /// `depth` nested half-opaque cards of `width` x `height` device pixels at
    /// (40, 40), each inset 4 px in the one around it and drawing a translucent quad
    /// over itself first (so none folds into another), around two overlapping quads.
    pub fn cards(depth: usize, width: f32, height: f32, target: (f32, f32)) -> Scene {
        let mut scene = Scene::default();
        let mask = mask(target.0, target.1);
        for level in 0..depth {
            let inset = 4. * level as f32;
            scene.push_group(GroupEffects {
                opacity: 0.9,
                content_mask: mask,
                ..GroupEffects::default()
            });
            scene.insert_primitive(quad(
                40. + inset,
                40. + inset,
                width - 2. * inset,
                height - 2. * inset,
                0.1,
                mask,
            ));
        }
        let inset = 4. * depth as f32;
        scene.insert_primitive(quad(40. + inset, 40. + inset, 40., 40., 1., mask));
        scene.insert_primitive(quad(60. + inset, 60. + inset, 40., 40., 1., mask));
        for _ in 0..depth {
            scene.pop_group();
        }
        scene.finish();
        scene
    }

    /// `depth` nested half-opaque groups, each drawing a translucent quad over the
    /// whole target first (so none folds into another), around `flat(count)`.
    pub fn nested(depth: usize, count: usize, width: f32, height: f32) -> Scene {
        let mut scene = Scene::default();
        let mask = mask(width, height);
        for _ in 0..depth {
            scene.push_group(GroupEffects {
                opacity: 0.9,
                content_mask: mask,
                ..GroupEffects::default()
            });
            scene.insert_primitive(quad(0., 0., width, height, 0.1, mask));
        }
        let columns = (width / 20.).max(1.) as usize;
        for i in 0..count {
            let (x, y) = (
                (i % columns) as f32 * 20.,
                (i / columns) as f32 * 12. % height,
            );
            scene.insert_primitive(quad(x, y, 24., 16., 1., mask));
        }
        for _ in 0..depth {
            scene.pop_group();
        }
        scene.finish();
        scene
    }
}

/// Every [`BlendMode`] composited over four backdrops, and four ordered chains of
/// color filters, for renderer tests, with
/// expected colors from [`blend_mode_test_scene::reference`], a CPU version of the
/// Compositing and Blending Level 1 formulas. Each cell is an isolated group on a
/// white page, so backdrops may be translucent.
#[cfg(any(test, feature = "test-support", feature = "bench-support"))]
#[doc(hidden)]
pub mod blend_mode_test_scene {
    use super::*;
    pub use crate::retained_shapes_test_scene::{Probe, SIZE, check};
    use crate::{Rgba, size};

    /// Straight backdrop and source colors (RGBA) of each column.
    pub const PAIRS: [([f32; 4], [f32; 4]); 4] = [
        ([0.8, 0.3, 0.2, 1.], [0.2, 0.6, 0.9, 1.]),
        // Unclamped color dodge and burn amplify 8-bit rounding near their poles
        // (a source channel near 1 or 0): these stay clear of them.
        ([0.1, 0.7, 0.4, 1.], [0.7, 0.5, 0.15, 0.6]),
        ([0.5, 0.5, 0.5, 0.5], [0.7, 0.2, 0.6, 1.]),
        ([0.95, 0.9, 0.1, 1.], [0.3, 0.3, 0.8, 0.8]),
    ];

    /// Every mode, one row each.
    pub const MODES: [BlendMode; 18] = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::ColorDodge,
        BlendMode::ColorBurn,
        BlendMode::HardLight,
        BlendMode::SoftLight,
        BlendMode::Difference,
        BlendMode::Exclusion,
        BlendMode::Hue,
        BlendMode::Saturation,
        BlendMode::Color,
        BlendMode::Luminosity,
        BlendMode::PlusLighter,
        BlendMode::PlusDarker,
    ];

    fn lum(c: [f32; 3]) -> f32 {
        0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
    }

    fn clip_color(c: [f32; 3]) -> [f32; 3] {
        let l = lum(c);
        let n = c[0].min(c[1]).min(c[2]);
        let x = c[0].max(c[1]).max(c[2]);
        let mut c = c;
        if n < 0. {
            c = c.map(|v| l + (v - l) * l / (l - n));
        }
        if x > 1. {
            c = c.map(|v| l + (v - l) * (1. - l) / (x - l));
        }
        c
    }

    fn set_lum(c: [f32; 3], l: f32) -> [f32; 3] {
        let d = l - lum(c);
        clip_color(c.map(|v| v + d))
    }

    fn sat(c: [f32; 3]) -> f32 {
        c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
    }

    fn set_sat(c: [f32; 3], s: f32) -> [f32; 3] {
        let n = c[0].min(c[1]).min(c[2]);
        let x = c[0].max(c[1]).max(c[2]);
        if x > n {
            c.map(|v| (v - n) * s / (x - n))
        } else {
            [0.; 3]
        }
    }

    fn separable(mode: BlendMode, b: f32, s: f32) -> f32 {
        let screen = |b: f32, s: f32| b + s - b * s;
        let hard_light = |b: f32, s: f32| {
            if s <= 0.5 {
                b * 2. * s
            } else {
                screen(b, 2. * s - 1.)
            }
        };
        match mode {
            BlendMode::Multiply => b * s,
            BlendMode::Screen => screen(b, s),
            BlendMode::Overlay => hard_light(s, b),
            BlendMode::Darken => b.min(s),
            BlendMode::Lighten => b.max(s),
            // Unclamped (see `reference`), as the shaders: the divisor floors at 1e-6.
            BlendMode::ColorDodge => {
                if b <= 0. {
                    0.
                } else {
                    b / (1. - s).max(1e-6)
                }
            }
            BlendMode::ColorBurn => {
                if b >= 1. {
                    1.
                } else {
                    1. - (1. - b) / s.max(1e-6)
                }
            }
            BlendMode::HardLight => hard_light(b, s),
            // WebKit's (Core Graphics') soft light: one curve for every source. CSS
            // Compositing 1 switches to a square root above 0.5.
            BlendMode::SoftLight => b + (2. * s - 1.) * b * (1. - b),
            BlendMode::Difference => (b - s).abs(),
            BlendMode::Exclusion => b + s - 2. * b * s,
            _ => s,
        }
    }

    /// The premultiplied result of compositing straight `source` onto straight
    /// `backdrop` with `mode`.
    pub fn reference(mode: BlendMode, backdrop: [f32; 4], source: [f32; 4]) -> [f32; 4] {
        let (b, ab) = ([backdrop[0], backdrop[1], backdrop[2]], backdrop[3]);
        let (s, a_s) = ([source[0], source[1], source[2]], source[3]);
        match mode {
            BlendMode::PlusLighter => {
                let a = (a_s + ab).min(1.);
                let c: [f32; 3] = std::array::from_fn(|i| (s[i] * a_s + b[i] * ab).min(1.));
                return [c[0], c[1], c[2], a];
            }
            BlendMode::PlusDarker => {
                let a = (a_s + ab).min(1.);
                let c: [f32; 3] =
                    std::array::from_fn(|i| (a - (ab - b[i] * ab) - (a_s - s[i] * a_s)).max(0.));
                return [c[0], c[1], c[2], a];
            }
            _ => {}
        }
        let blended = match mode {
            BlendMode::Hue => set_lum(set_sat(s, sat(b)), lum(b)),
            BlendMode::Saturation => set_lum(set_sat(b, sat(s)), lum(b)),
            BlendMode::Color => set_lum(s, lum(b)),
            BlendMode::Luminosity => set_lum(b, lum(s)),
            _ => std::array::from_fn(|i| separable(mode, b[i], s[i])),
        };
        let a = a_s + ab * (1. - a_s);
        // WebKit (Core Graphics) mixes color dodge and burn unclamped and clamps the
        // result: a translucent source that dodges past 1 still lands at 1.
        let unclamped = matches!(mode, BlendMode::ColorDodge | BlendMode::ColorBurn);
        let c: [f32; 3] = std::array::from_fn(|i| {
            let blended = if unclamped {
                blended[i]
            } else {
                blended[i].clamp(0., 1.)
            };
            let mixed = (1. - ab) * s[i] + ab * blended;
            (a_s * mixed + ab * b[i] * (1. - a_s)).clamp(0., a)
        });
        [c[0], c[1], c[2], a]
    }

    fn rgba(c: [f32; 4]) -> Hsla {
        Rgba {
            r: c[0],
            g: c[1],
            b: c[2],
            a: c[3],
        }
        .into()
    }

    /// The scene and the probes every renderer must hit. Cell `(column, row)` covers
    /// 16 x 3 device pixels at `(16 column, 3 row)`.
    pub fn scene() -> (Scene, Vec<Probe>) {
        let full = ContentMask::from(Bounds::new(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            size(ScaledPixels(SIZE.0 as f32), ScaledPixels(SIZE.1 as f32)),
        ));
        let mut scene = Scene::default();
        scene.insert_primitive(Quad {
            bounds: full.bounds,
            content_mask: full,
            background: rgba([1.; 4]).into(),
            ..Default::default()
        });
        let mut probes = Vec::new();
        for (row, mode) in MODES.into_iter().enumerate() {
            for (column, (backdrop, source)) in PAIRS.into_iter().enumerate() {
                let cell = Bounds::new(
                    point(
                        ScaledPixels(16. * column as f32),
                        ScaledPixels(3. * row as f32),
                    ),
                    size(ScaledPixels(16.), ScaledPixels(3.)),
                );
                scene.push_group(GroupEffects {
                    content_mask: full,
                    ..GroupEffects::default()
                });
                scene.insert_primitive(Quad {
                    bounds: cell,
                    content_mask: full,
                    background: rgba(backdrop).into(),
                    ..Default::default()
                });
                scene.push_group(GroupEffects {
                    blend_mode: mode,
                    content_mask: full,
                    ..GroupEffects::default()
                });
                scene.insert_primitive(Quad {
                    bounds: cell,
                    content_mask: full,
                    background: rgba(source).into(),
                    ..Default::default()
                });
                scene.pop_group();
                scene.pop_group();
                // The isolated cell over white.
                let cell = reference(mode, backdrop, source);
                let expected = [0, 1, 2].map(|i| cell[i] + 1. - cell[3]);
                let expected = [expected[0], expected[1], expected[2], 1.];
                probes.push(Probe {
                    what: "blend mode cell",
                    x: 16 * column as u32 + 8,
                    y: 3 * row as u32 + 1,
                    rgba: expected.map(|v| (v.clamp(0., 1.) * 255.).round() as u8),
                });
            }
        }
        // Color filter chains, below the blend modes.
        for (column, (rgb, filters)) in chains().into_iter().enumerate() {
            let cell = Bounds::new(
                point(ScaledPixels(16. * column as f32), ScaledPixels(56.)),
                size(ScaledPixels(16.), ScaledPixels(6.)),
            );
            scene.push_group(GroupEffects {
                color_filters: filters.iter().copied().collect(),
                content_mask: full,
                ..GroupEffects::default()
            });
            scene.insert_primitive(Quad {
                bounds: cell,
                content_mask: full,
                background: rgba([rgb[0], rgb[1], rgb[2], 1.]).into(),
                ..Default::default()
            });
            scene.pop_group();
            let expected = color_filters_reference(rgb, &filters);
            probes.push(Probe {
                what: "filter chain cell",
                x: 16 * column as u32 + 8,
                y: 59,
                rgba: [expected[0], expected[1], expected[2], 1.]
                    .map(|v| (v.clamp(0., 1.) * 255.).round() as u8),
            });
        }
        scene.finish();
        (scene, probes)
    }

    /// The mode and column of the probe at `index`.
    pub fn describe(index: usize) -> String {
        match MODES.get(index / PAIRS.len()) {
            Some(mode) => format!("{mode:?}, pair {}", index % PAIRS.len()),
            None => format!("filter chain {}", index - MODES.len() * PAIRS.len()),
        }
    }

    /// Colors and color filter chains (five to eight functions, in order) of the cells
    /// below the blend modes.
    pub fn chains() -> [([f32; 3], Vec<ColorFilter>); 4] {
        use ColorFilter::*;
        [
            (
                [0.3, 0.5, 0.7],
                vec![
                    Brightness(1.4),
                    Contrast(0.8),
                    Saturate(1.5),
                    HueRotate(45.),
                    Sepia(0.3),
                ],
            ),
            (
                [0.9, 0.2, 0.4],
                vec![
                    Invert(0.25),
                    Grayscale(0.5),
                    Brightness(0.9),
                    HueRotate(-120.),
                    Contrast(1.3),
                    Saturate(0.2),
                ],
            ),
            (
                [0.2, 0.8, 0.3],
                vec![
                    HueRotate(90.),
                    Saturate(2.),
                    Brightness(1.2),
                    Contrast(1.5),
                    Invert(0.1),
                    Sepia(0.5),
                    Grayscale(0.2),
                    Brightness(0.8),
                ],
            ),
            (
                [0.6, 0.6, 0.1],
                vec![
                    Contrast(2.),
                    Brightness(0.5),
                    Invert(1.),
                    HueRotate(200.),
                    Saturate(0.),
                ],
            ),
        ]
    }

    /// `filters` applied to straight `rgb` in order, each clamped (Filter Effects 1).
    pub fn color_filters_reference(rgb: [f32; 3], filters: &[ColorFilter]) -> [f32; 3] {
        let matrix = |c: [f32; 3], m: [[f32; 3]; 3]| -> [f32; 3] {
            std::array::from_fn(|i| {
                (m[i][0] * c[0] + m[i][1] * c[1] + m[i][2] * c[2]).clamp(0., 1.)
            })
        };
        let mut c = rgb;
        for filter in filters {
            let (kind, a) = filter.gpu();
            c = match kind {
                1 => c.map(|v| (v * a).clamp(0., 1.)),
                2 => c.map(|v| ((v - 0.5) * a + 0.5).clamp(0., 1.)),
                3 => {
                    let s = 1. - a;
                    matrix(
                        c,
                        [
                            [
                                0.2126 + 0.7874 * s,
                                0.7152 - 0.7152 * s,
                                0.0722 - 0.0722 * s,
                            ],
                            [
                                0.2126 - 0.2126 * s,
                                0.7152 + 0.2848 * s,
                                0.0722 - 0.0722 * s,
                            ],
                            [
                                0.2126 - 0.2126 * s,
                                0.7152 - 0.7152 * s,
                                0.0722 + 0.9278 * s,
                            ],
                        ],
                    )
                }
                4 => {
                    let (sin, cos) = a.sin_cos();
                    matrix(
                        c,
                        [
                            [
                                0.213 + cos * 0.787 - sin * 0.213,
                                0.715 - cos * 0.715 - sin * 0.715,
                                0.072 - cos * 0.072 + sin * 0.928,
                            ],
                            [
                                0.213 - cos * 0.213 + sin * 0.143,
                                0.715 + cos * 0.285 + sin * 0.140,
                                0.072 - cos * 0.072 - sin * 0.283,
                            ],
                            [
                                0.213 - cos * 0.213 - sin * 0.787,
                                0.715 - cos * 0.715 + sin * 0.715,
                                0.072 + cos * 0.928 + sin * 0.072,
                            ],
                        ],
                    )
                }
                0 => c,
                5 => c.map(|v| (v + (1. - 2. * v) * a).clamp(0., 1.)),
                6 => matrix(
                    c,
                    [
                        [0.213 + 0.787 * a, 0.715 - 0.715 * a, 0.072 - 0.072 * a],
                        [0.213 - 0.213 * a, 0.715 + 0.285 * a, 0.072 - 0.072 * a],
                        [0.213 - 0.213 * a, 0.715 - 0.715 * a, 0.072 + 0.928 * a],
                    ],
                ),
                _ => {
                    let s = 1. - a;
                    matrix(
                        c,
                        [
                            [0.393 + 0.607 * s, 0.769 - 0.769 * s, 0.189 - 0.189 * s],
                            [0.349 - 0.349 * s, 0.686 + 0.314 * s, 0.168 - 0.168 * s],
                            [0.272 - 0.272 * s, 0.534 - 0.534 * s, 0.131 + 0.869 * s],
                        ],
                    )
                }
            };
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AtlasTextureId, AtlasTextureKind, DevicePixels, TileId, radians, size};

    fn path_at(bounds: Bounds<ScaledPixels>, order: DrawOrder) -> Path<ScaledPixels> {
        let mut path = Path::new(point(crate::px(0.), crate::px(0.))).scale(1.);
        path.bounds = bounds;
        path.order = order;
        path.content_mask = ContentMask::from(scaled_bounds(-1000., -1000., 4000., 4000.));
        path
    }

    #[test]
    fn path_copy_bounds_merge_overlapping_paths_of_one_order() {
        // One order (a paint layer), overlapping: one region, so each pixel is copied once.
        let a = path_at(scaled_bounds(0., 0., 10., 10.), 3);
        let b = path_at(scaled_bounds(8., 0., 10., 10.), 3);
        let c = path_at(scaled_bounds(16., 0., 10., 10.), 3);
        assert_eq!(
            path_copy_bounds(&[a.clone(), b, c]),
            vec![scaled_bounds(0., 0., 26., 10.)]
        );
        // Disjoint paths keep their own regions (no copy of the space between them).
        let far = path_at(scaled_bounds(100., 100., 5., 5.), 3);
        assert_eq!(
            path_copy_bounds(&[a, far]),
            vec![
                scaled_bounds(0., 0., 10., 10.),
                scaled_bounds(100., 100., 5., 5.)
            ]
        );
        // A region that grows into another merges with it too.
        let p = path_at(scaled_bounds(0., 0., 4., 4.), 1);
        let q = path_at(scaled_bounds(10., 0., 4., 4.), 2);
        let bridge = path_at(scaled_bounds(2., 0., 10., 4.), 3);
        assert_eq!(
            path_copy_bounds(&[p, q, bridge]),
            vec![scaled_bounds(0., 0., 14., 4.)]
        );
    }

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
            content_mask: ContentMask::from(content_mask),
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
    fn shader_draws_keep_one_source_and_aligned_uniforms() {
        let mut scene = Scene::default();
        let shader =
            Shader::new("fn shade(input: ShaderInput) -> vec4<f32> { return vec4<f32>(1.0); }");
        let a = scene.add_shader(&shader, &[1, 2, 3, 4, 5], true, 2.);
        let b = scene.add_shader(&shader, &[6; 8], false, 2.);
        assert_eq!(scene.shaders.len(), 1, "one source per frame");
        assert_eq!(a.shader(&scene), Some(&shader));
        assert_eq!(a.uniforms(&scene), &[1, 2, 3, 4, 5]);
        assert_eq!(b.uniforms_start, 16, "uniforms start 16-byte aligned");
        assert_eq!(b.uniforms(&scene), &[6; 8]);
        assert!(a.reads_backdrop() && !b.reads_backdrop());
        assert_eq!(ShaderDraw::NONE.shader(&scene), None);
        scene.clear();
        assert!(scene.shaders.is_empty() && scene.shader_uniforms.is_empty());
    }

    #[test]
    fn replayed_shader_draws_bring_their_source_and_uniforms() {
        let mut previous = Scene::default();
        let other =
            Shader::new("fn shade(input: ShaderInput) -> vec4<f32> { return vec4<f32>(0.0); }");
        let shader =
            Shader::new("fn shade(input: ShaderInput) -> vec4<f32> { return vec4<f32>(1.0); }");
        previous.add_shader(&other, &[9; 32], false, 1.);
        let draw = previous.add_shader(&shader, &[1, 2, 3, 4], true, 2.);
        previous.insert_primitive(BackdropFilter {
            bounds: scaled_bounds(0., 0., 10., 10.),
            content_mask: ContentMask::from(scaled_bounds(0., 0., 10., 10.)),
            shader: draw,
            ..Default::default()
        });
        let mut next = Scene::default();
        next.replay(0..previous.paint_operations.len(), &previous);
        let replayed = next.backdrop_filters[0].shader;
        assert_eq!(replayed.shader(&next), Some(&shader));
        assert_eq!(replayed.uniforms(&next), &[1, 2, 3, 4]);
        assert!(replayed.reads_backdrop());
        assert_eq!(replayed.scale_factor, 2.);
    }

    #[test]
    fn backdrop_filters_inside_a_layer_keep_paint_order() {
        let mut scene = Scene::default();
        let layer = scaled_bounds(0., 0., 40., 40.);
        scene.push_layer(layer);
        scene.insert_primitive(quad(scaled_bounds(0., 0., 40., 40.)));
        scene.insert_primitive(BackdropFilter {
            bounds: scaled_bounds(5., 5., 20., 20.),
            content_mask: ContentMask::from(layer),
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
            content_mask: ContentMask::from(scaled_bounds(12., 0., 20., 20.)),
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
            content_mask: ContentMask::from(scaled_bounds(-100., -100., 1000., 1000.)),
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
            content_mask: ContentMask::from(bounds),
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
            content_mask: ContentMask::from(bounds),
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

    fn opaque_quad(bounds: Bounds<ScaledPixels>) -> Quad {
        Quad {
            background: crate::red().into(),
            ..quad(bounds)
        }
    }

    fn effects(opacity: f32) -> GroupEffects {
        GroupEffects {
            opacity,
            content_mask: ContentMask::from(scaled_bounds(0., 0., 100., 100.)),
            ..GroupEffects::default()
        }
    }

    #[test]
    fn a_translucent_group_of_disjoint_primitives_fades_each() {
        let mut scene = Scene::default();
        scene.push_group(effects(0.5));
        scene.insert_primitive(opaque_quad(scaled_bounds(0., 0., 10., 10.)));
        scene.insert_primitive(opaque_quad(scaled_bounds(20., 0., 10., 10.)));
        scene.pop_group();
        scene.finish();
        assert!(scene.groups.is_empty());
        assert_eq!(scene.quads.len(), 2);
        for quad in &scene.quads {
            assert_eq!(quad.background.solid.a, 0.5);
        }
    }

    #[test]
    fn a_translucent_group_of_overlapping_primitives_draws_offscreen() {
        let mut scene = Scene::default();
        scene.insert_primitive(opaque_quad(scaled_bounds(0., 0., 100., 100.)));
        scene.push_group(effects(0.5));
        scene.insert_primitive(opaque_quad(scaled_bounds(0.5, 0., 10., 10.)));
        scene.insert_primitive(opaque_quad(scaled_bounds(5., 5., 10.2, 10.)));
        scene.pop_group();
        scene.finish();
        assert_eq!(scene.quads.len(), 1);
        assert_eq!(scene.groups.len(), 1);
        let group = &scene.groups[0];
        assert_eq!(group.scene.quads.len(), 2);
        assert_eq!(group.scene.quads[0].background.solid.a, 1.);
        // Its content, in whole pixels, after what it covers.
        assert_eq!(group.bounds, scaled_bounds(0., 0., 16., 15.));
        assert!(group.order > scene.quads[0].order);
        assert_eq!(scene.group_depth(), 1);
        let batches: Vec<String> = scene.batches().map(|b| b.label()).collect();
        assert_eq!(batches, ["quads (1)", "groups (1)"]);
    }

    #[test]
    fn a_quad_with_a_border_and_a_background_overlaps_itself() {
        let mut scene = Scene::default();
        scene.push_group(effects(0.5));
        scene.insert_primitive(Quad {
            border_widths: Edges::all(ScaledPixels(1.)),
            border_color: crate::blue(),
            ..opaque_quad(scaled_bounds(0., 0., 10., 10.))
        });
        scene.pop_group();
        scene.finish();
        assert_eq!(scene.groups.len(), 1);
    }

    #[test]
    fn a_translucent_group_is_the_backdrop_root_of_its_filters() {
        let mut scene = Scene::default();
        scene.push_group(effects(0.5));
        scene.insert_primitive(BackdropFilter {
            bounds: scaled_bounds(0., 0., 10., 10.),
            content_mask: effects(1.).content_mask,
            saturate: 0.,
            ..Default::default()
        });
        scene.pop_group();
        scene.finish();
        assert_eq!(scene.groups.len(), 1);
        assert_eq!(scene.groups[0].scene.backdrop_filters.len(), 1);
    }

    #[test]
    fn empty_and_transparent_groups_paint_nothing() {
        let mut scene = Scene::default();
        scene.push_group(effects(0.5));
        scene.pop_group();
        scene.push_group(effects(0.));
        scene.insert_primitive(opaque_quad(scaled_bounds(0., 0., 10., 10.)));
        scene.insert_primitive(opaque_quad(scaled_bounds(5., 5., 10., 10.)));
        scene.pop_group();
        scene.finish();
        assert!(scene.groups.is_empty());
        assert!(scene.quads.is_empty());
    }

    #[test]
    fn an_isolating_group_draws_offscreen_only_around_blending_groups() {
        let mut scene = Scene::default();
        scene.push_group(effects(1.));
        scene.insert_primitive(opaque_quad(scaled_bounds(0., 0., 10., 10.)));
        scene.insert_primitive(opaque_quad(scaled_bounds(5., 5., 10., 10.)));
        scene.pop_group();
        scene.finish();
        assert!(scene.groups.is_empty());
        assert_eq!(scene.quads.len(), 2);

        let mut scene = Scene::default();
        scene.push_group(effects(1.));
        scene.insert_primitive(opaque_quad(scaled_bounds(0., 0., 10., 10.)));
        scene.push_group(GroupEffects {
            blend_mode: BlendMode::Multiply,
            ..effects(1.)
        });
        scene.insert_primitive(opaque_quad(scaled_bounds(5., 5., 10., 10.)));
        scene.pop_group();
        scene.pop_group();
        scene.finish();
        assert_eq!(scene.groups.len(), 1);
        assert_eq!(scene.groups[0].effects.blend_mode, BlendMode::Normal);
        assert_eq!(scene.groups[0].scene.groups.len(), 1);
        assert_eq!(scene.group_depth(), 2);
    }

    #[test]
    fn a_translucent_group_around_one_group_folds_into_it() {
        let mut scene = Scene::default();
        scene.push_group(effects(0.5));
        scene.push_group(effects(0.5));
        scene.insert_primitive(opaque_quad(scaled_bounds(0., 0., 10., 10.)));
        scene.insert_primitive(opaque_quad(scaled_bounds(5., 5., 10., 10.)));
        scene.pop_group();
        scene.pop_group();
        scene.finish();
        assert_eq!(scene.groups.len(), 1);
        assert_eq!(scene.groups[0].effects.opacity, 0.25);
        assert_eq!(scene.group_depth(), 1);
    }

    #[test]
    fn a_blur_spreads_a_group_within_its_mask() {
        let mut scene = Scene::default();
        scene.push_group(GroupEffects {
            blur_radius: ScaledPixels(2.),
            ..effects(1.)
        });
        scene.insert_primitive(opaque_quad(scaled_bounds(0., 10., 10., 10.)));
        scene.pop_group();
        scene.finish();
        assert_eq!(scene.groups[0].bounds, scaled_bounds(0., 4., 16., 22.));
    }

    #[test]
    fn groups_inside_a_layer_keep_paint_order() {
        let mut scene = Scene::default();
        scene.push_layer(scaled_bounds(0., 0., 100., 100.));
        scene.insert_primitive(opaque_quad(scaled_bounds(0., 0., 50., 50.)));
        scene.push_group(GroupEffects {
            blend_mode: BlendMode::Screen,
            ..effects(1.)
        });
        scene.insert_primitive(opaque_quad(scaled_bounds(10., 10., 10., 10.)));
        scene.pop_group();
        scene.insert_primitive(opaque_quad(scaled_bounds(0., 0., 50., 50.)));
        scene.pop_layer();
        scene.finish();
        let batches: Vec<String> = scene.batches().map(|b| b.label()).collect();
        assert_eq!(batches, ["quads (1)", "groups (1)", "quads (1)"]);
    }

    #[test]
    fn replayed_groups_match_painted_ones() {
        let mut painted = Scene::default();
        painted.push_group(effects(0.5));
        painted.insert_primitive(opaque_quad(scaled_bounds(0., 0., 10., 10.)));
        painted.insert_primitive(opaque_quad(scaled_bounds(5., 5., 10., 10.)));
        painted.pop_group();
        painted.push_group(effects(0.5));
        painted.insert_primitive(opaque_quad(scaled_bounds(50., 50., 10., 10.)));
        painted.pop_group();
        let mut replayed = Scene::default();
        replayed.replay(0..painted.len(), &painted);
        painted.finish();
        replayed.finish();
        for scene in [&painted, &replayed] {
            assert_eq!(scene.groups.len(), 1);
            assert_eq!(scene.groups[0].scene.quads.len(), 2);
            assert_eq!(scene.quads.len(), 1);
            assert_eq!(scene.quads[0].background.solid.a, 0.5);
        }
    }

    #[test]
    fn blend_mode_reference_matches_known_values() {
        use blend_mode_test_scene::reference;
        let b = [0.8, 0.3, 0.2, 1.];
        let s = [0.2, 0.6, 0.9, 1.];
        let close = |a: [f32; 4], e: [f32; 4]| a.iter().zip(e).all(|(a, e)| (a - e).abs() < 1e-5);
        assert!(close(reference(BlendMode::Normal, b, s), s));
        assert!(close(
            reference(BlendMode::Multiply, b, s),
            [0.16, 0.18, 0.18, 1.]
        ));
        assert!(close(
            reference(BlendMode::Screen, b, s),
            [0.84, 0.72, 0.92, 1.]
        ));
        assert!(close(
            reference(BlendMode::Difference, b, s),
            [0.6, 0.3, 0.7, 1.]
        ));
        assert!(close(
            reference(BlendMode::PlusDarker, b, s),
            [0., 0., 0.1, 1.]
        ));
        assert!(close(
            reference(BlendMode::PlusLighter, b, s),
            [1., 0.9, 1., 1.]
        ));
        // A transparent backdrop leaves the source.
        let clear = [0., 0., 0., 0.];
        for mode in blend_mode_test_scene::MODES {
            assert!(close(reference(mode, clear, s), s), "{mode:?}");
        }
    }

    #[test]
    fn opacity_filters_fold_into_the_group_opacity() {
        let mut scene = Scene::default();
        scene.push_group(GroupEffects {
            color_filters: [ColorFilter::Opacity(0.5), ColorFilter::Invert(1.)]
                .into_iter()
                .collect(),
            ..effects(0.5)
        });
        scene.insert_primitive(opaque_quad(scaled_bounds(0., 0., 10., 10.)));
        scene.pop_group();
        scene.finish();
        assert_eq!(scene.groups[0].effects.opacity, 0.25);
        assert_eq!(
            scene.groups[0].effects.color_filters.as_slice(),
            &[ColorFilter::Invert(1.)]
        );
        // Only an opacity filter: a plain opacity group, painted directly here.
        let mut scene = Scene::default();
        scene.push_group(GroupEffects {
            color_filters: [ColorFilter::Opacity(0.5)].into_iter().collect(),
            ..effects(1.)
        });
        scene.insert_primitive(opaque_quad(scaled_bounds(0., 0., 10., 10.)));
        scene.pop_group();
        scene.finish();
        assert!(scene.groups.is_empty());
        assert_eq!(scene.quads[0].background.solid.a, 0.5);
    }

    #[test]
    fn offscreen_groups_draw_in_their_image_coordinates() {
        let mut scene = Scene::default();
        scene.push_group(effects(0.5));
        scene.insert_primitive(opaque_quad(scaled_bounds(40., 30., 10., 10.)));
        scene.push_group(effects(0.5));
        scene.insert_primitive(opaque_quad(scaled_bounds(44., 34., 10., 10.)));
        scene.insert_primitive(Quad {
            transformation: TransformationMatrix::unit()
                .translate(point(ScaledPixels(46.), ScaledPixels(36.)))
                .rotate(radians(1.))
                .translate(point(ScaledPixels(-46.), ScaledPixels(-36.))),
            ..opaque_quad(scaled_bounds(45., 35., 2., 2.))
        });
        scene.pop_group();
        scene.pop_group();
        scene.finish();
        let outer = &scene.groups[0];
        assert_eq!(
            outer.bounds.origin,
            point(ScaledPixels(40.), ScaledPixels(30.))
        );
        assert_eq!(
            outer.scene.quads[0].bounds,
            scaled_bounds(0., 0., 10., 10.),
            "relative to the outer image"
        );
        let inner = &outer.scene.groups[0];
        assert_eq!(
            inner.bounds.origin,
            point(ScaledPixels(4.), ScaledPixels(4.))
        );
        assert_eq!(inner.scene.quads[0].bounds, scaled_bounds(0., 0., 10., 10.));
        // A rotation about (46, 36) in the window turns about (2, 2) in the image.
        let rotated = &inner.scene.quads[1];
        let pivot = point(ScaledPixels(2.), ScaledPixels(2.));
        let turned = rotated.transformation.apply_scaled(pivot);
        assert!((turned.x.0 - 2.).abs() < 1e-4 && (turned.y.0 - 2.).abs() < 1e-4);
    }

    #[test]
    fn a_nested_group_clipped_by_a_direct_parent_keeps_its_content_in_place() {
        // A translucent parent around one group folds into it (painted directly), and
        // its mask cuts the child's left half.
        let mut scene = Scene::default();
        scene.push_group(GroupEffects {
            content_mask: ContentMask::from(scaled_bounds(20., 0., 80., 80.)),
            ..effects(0.5)
        });
        scene.push_group(effects(0.5));
        scene.insert_primitive(opaque_quad(scaled_bounds(10., 10., 20., 20.)));
        scene.insert_primitive(opaque_quad(scaled_bounds(15., 15., 10., 10.)));
        scene.pop_group();
        scene.pop_group();
        scene.finish();
        assert_eq!(scene.groups.len(), 1);
        let group = &scene.groups[0];
        assert_eq!(group.effects.opacity, 0.25);
        assert_eq!(group.bounds, scaled_bounds(20., 10., 10., 20.));
        // The quad still covers window x 10..30: image x -10..10.
        assert_eq!(
            group.scene.quads[0].bounds,
            scaled_bounds(-10., 0., 20., 20.)
        );
    }

    #[test]
    fn a_directly_painted_group_keeps_its_mask() {
        let mut scene = Scene::default();
        scene.push_group(GroupEffects {
            content_mask: ContentMask::from(scaled_bounds(0., 0., 5., 5.)),
            ..effects(0.5)
        });
        scene.insert_primitive(opaque_quad(scaled_bounds(0., 0., 10., 10.)));
        scene.pop_group();
        scene.finish();
        assert!(scene.groups.is_empty());
        assert_eq!(
            scene.quads[0].content_mask.bounds,
            scaled_bounds(0., 0., 5., 5.)
        );
    }

    #[test]
    fn a_group_backdrop_starts_its_image() {
        let backdrop = BackdropFilter {
            bounds: scaled_bounds(0., 0., 10., 10.),
            content_mask: effects(1.).content_mask,
            blur_radius: ScaledPixels(2.),
            ..Default::default()
        };
        // Translucent: drawn offscreen, the backdrop with the content.
        let mut scene = Scene::default();
        scene.insert_primitive(opaque_quad(scaled_bounds(14., 0., 4., 4.)));
        scene.push_group(GroupEffects {
            backdrop: Some(backdrop),
            ..effects(0.5)
        });
        scene.insert_primitive(opaque_quad(scaled_bounds(2., 2., 4., 4.)));
        scene.pop_group();
        scene.finish();
        assert_eq!(scene.groups.len(), 1);
        assert_eq!(scene.groups[0].effects.backdrop, Some(backdrop));
        // Ordered after what its blur reads (the quad 4 px to the right).
        assert!(scene.groups[0].order > scene.quads[0].order);
        // Opaque without effects: the backdrop paints as a backdrop filter first.
        let mut scene = Scene::default();
        scene.push_group(GroupEffects {
            backdrop: Some(backdrop),
            ..effects(1.)
        });
        scene.insert_primitive(opaque_quad(scaled_bounds(2., 2., 4., 4.)));
        scene.pop_group();
        scene.finish();
        assert!(scene.groups.is_empty());
        assert_eq!(scene.backdrop_filters.len(), 1);
        let batches: Vec<String> = scene.batches().map(|b| b.label()).collect();
        assert_eq!(batches, ["backdrop filters (1)", "quads (1)"]);
        // A group with only a backdrop is not empty.
        let mut scene = Scene::default();
        scene.push_group(GroupEffects {
            backdrop: Some(backdrop),
            blend_mode: BlendMode::Screen,
            ..effects(1.)
        });
        scene.pop_group();
        scene.finish();
        assert_eq!(scene.groups.len(), 1);
    }

    #[test]
    fn without_groups_fades_each_primitive() {
        let mut scene = Scene::default();
        scene.push_group(GroupEffects {
            blend_mode: BlendMode::Multiply,
            ..effects(0.5)
        });
        scene.insert_primitive(opaque_quad(scaled_bounds(0., 0., 10., 10.)));
        scene.push_group(effects(0.5));
        scene.insert_primitive(opaque_quad(scaled_bounds(5., 5., 10., 10.)));
        scene.insert_primitive(opaque_quad(scaled_bounds(6., 6., 10., 10.)));
        scene.pop_group();
        scene.pop_group();
        scene.finish();
        assert_eq!(scene.groups.len(), 1);
        let flat = scene.without_groups();
        assert!(flat.groups.is_empty());
        let alphas: Vec<f32> = flat.quads.iter().map(|q| q.background.solid.a).collect();
        assert_eq!(alphas, [0.5, 0.25, 0.25]);
    }

    #[test]
    fn a_mask_group_masks_its_parent_offscreen_inside_its_bounds() {
        let (scene, _) = group_mask_test_scene::scene();
        // The red, the multiplying and the green groups; the empty mask hides the black.
        assert_eq!(scene.groups.len(), 3);
        let red = &scene.groups[0];
        let mask = red.mask.as_ref().expect("masked");
        assert_eq!(mask.mode, MaskMode::Alpha);
        assert_eq!(red.bounds, scaled_bounds(4., 4., 24., 24.));
        assert_eq!(mask.image.bounds, scaled_bounds(4., 4., 24., 24.));
        // The mask's image draws in its own coordinates.
        assert_eq!(
            mask.image.scene.quads[0].bounds,
            scaled_bounds(0., 0., 12., 24.)
        );
        let uniforms = red.composite_uniforms([0.; 4], [0.; 4], [1., 2., 3., 4.], [64., 64.]);
        assert_eq!(
            (uniforms.mask_mode, uniforms.mask_region),
            (1, [1., 2., 3., 4.])
        );
        // Inside the multiplying group's image (origin 36, 4), the blue group and its
        // mask moved with it.
        let outer = &scene.groups[1];
        assert_eq!(outer.bounds, scaled_bounds(36., 4., 24., 24.));
        let blue = &outer.scene.groups[0];
        assert_eq!(blue.bounds, scaled_bounds(0., 0., 24., 24.));
        let mask = blue.mask.as_ref().expect("masked");
        assert_eq!(
            (mask.mode, mask.image.bounds),
            (MaskMode::Luminance, scaled_bounds(0., 0., 24., 24.))
        );
        assert!(outer.mask.is_none());
        assert_eq!(scene.group_depth(), 2);
        // A group draws only inside its mask's image.
        assert_eq!(scene.groups[2].bounds, scaled_bounds(10., 42., 12., 12.));
        // Renderers without groups draw the content unmasked and the masks not at all.
        let flat = scene.without_groups();
        let colors: Vec<f32> = flat.quads.iter().map(|q| q.background.solid.a).collect();
        assert_eq!(
            colors,
            [1., 1., 1., 1., 1.],
            "page, red, blue, green, black"
        );
        // `scenes` visits masks' images.
        assert_eq!(
            scene.scenes().len(),
            8,
            "the page, three groups and their masks, the blue group"
        );
    }

    #[test]
    fn a_mask_outside_any_group_masks_nothing() {
        let mut scene = Scene::default();
        scene.push_group(GroupEffects {
            masks_parent: Some(MaskMode::Alpha),
            ..effects(1.)
        });
        scene.insert_primitive(opaque_quad(scaled_bounds(0., 0., 10., 10.)));
        scene.pop_group();
        scene.insert_primitive(opaque_quad(scaled_bounds(20., 0., 10., 10.)));
        scene.finish();
        assert!(scene.groups.is_empty());
        assert_eq!(scene.quads.len(), 1);
    }

    #[test]
    fn blend_reference_matches_webkit() {
        use blend_mode_test_scene::reference;
        // Measured in Playwright WebKit (macOS): a source over the opaque backdrop
        // #4f7cff, then 8-bit channels.
        let backdrop = [79. / 255., 124. / 255., 1., 1.];
        let px = |c: [f32; 4]| [c[0], c[1], c[2]].map(|v| (v * 255.).round() as i32);
        let near =
            |got: [i32; 3], want: [i32; 3]| got.iter().zip(want).all(|(g, w)| (g - w).abs() <= 2);
        let purple = [192. / 255., 75. / 255., 216. / 255., 0.6];
        let slate = [117. / 255., 128. / 255., 165. / 255., 0.8];
        // Color dodge and burn mix unclamped, then clamp.
        assert!(near(
            px(reference(BlendMode::ColorDodge, backdrop, purple)),
            [222, 155, 255]
        ));
        assert!(near(
            px(reference(BlendMode::ColorBurn, backdrop, purple)),
            [44, 0, 255]
        ));
        assert!(near(
            px(reference(BlendMode::ColorBurn, backdrop, slate)),
            [0, 19, 255]
        ));
        // Soft light keeps one curve above 0.5 (opaque grays: backdrop 32, source 255 → 60).
        let gray = |v: f32| [v / 255., v / 255., v / 255., 1.];
        assert!(near(
            px(reference(BlendMode::SoftLight, gray(32.), gray(255.))),
            [60; 3]
        ));
        assert!(near(
            px(reference(BlendMode::SoftLight, gray(192.), gray(160.))),
            [204; 3]
        ));
    }

    #[test]
    fn group_composite_uniforms_match_the_shader_layout() {
        assert_eq!(std::mem::size_of::<GroupCompositeUniforms>(), 176);
        assert_eq!(
            std::mem::offset_of!(GroupCompositeUniforms, mask_region),
            160
        );
        assert_eq!(
            std::mem::offset_of!(GroupCompositeUniforms, filter_kinds),
            64
        );
        assert_eq!(
            std::mem::offset_of!(GroupCompositeUniforms, viewport_size),
            128
        );
        assert_eq!(
            std::mem::offset_of!(GroupCompositeUniforms, filter_count),
            144
        );
        let mut scene = Scene::default();
        scene.push_group(GroupEffects {
            color_filters: (0..10).map(|_| ColorFilter::HueRotate(180.)).collect(),
            ..effects(1.)
        });
        scene.insert_primitive(opaque_quad(scaled_bounds(0., 0., 10., 10.)));
        scene.pop_group();
        scene.finish();
        let uniforms = scene.groups[0].composite_uniforms([0.; 4], [0.; 4], [0.; 4], [100., 100.]);
        assert_eq!(uniforms.filter_count, MAX_COLOR_FILTERS as u32);
        assert_eq!(uniforms.filter_kinds[1][3], 4);
        assert!((uniforms.filter_amounts[0][0] - std::f32::consts::PI).abs() < 1e-6);
    }
}

/// Custom shaders ([`Shader`]), for renderer tests: every backend renders it and
/// must hit the same probe colors.
#[cfg(any(test, feature = "test-support", feature = "bench-support"))]
#[doc(hidden)]
pub mod custom_shader_test_scene {
    use super::*;
    pub use crate::retained_shapes_test_scene::{Probe, SIZE, check};
    use crate::{Rgba, size};

    /// A uniform color.
    pub const SOLID: &str = "
struct Params { color: vec4<f32> }
@group(0) @binding(3) var<uniform> params: Params;
fn shade(input: ShaderInput) -> vec4<f32> { return params.color; }
";
    /// The backdrop, inverted.
    pub const INVERT: &str = "
fn shade(input: ShaderInput) -> vec4<f32> {
    let behind = backdrop(input.uv);
    return vec4<f32>(1.0 - behind.rgb, 1.0);
}
";
    /// Red from 0 to 1 across the element, green 1 past 8 logical pixels down.
    pub const GRADIENT: &str = "
fn shade(input: ShaderInput) -> vec4<f32> {
    return vec4<f32>(input.uv.x, select(0.0, 1.0, input.position.y > 8.0), 0.0, 1.0);
}
";

    fn color(r: f32, g: f32, b: f32) -> Hsla {
        Rgba { r, g, b, a: 1.0 }.into()
    }

    fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(w), ScaledPixels(h)),
        )
    }

    fn draw(
        scene: &mut Scene,
        source: &str,
        uniforms: &[f32],
        at: Bounds<ScaledPixels>,
        radius: f32,
        opacity: f32,
        reads_backdrop: bool,
    ) {
        let full = ContentMask::from(bounds(0., 0., SIZE.0 as f32, SIZE.1 as f32));
        let bytes: Vec<u8> = uniforms.iter().flat_map(|f| f.to_le_bytes()).collect();
        // Scale factor 2: the gradient's 8 logical pixels are 16 device pixels.
        let shader = scene.add_shader(&Shader::new(source), &bytes, reads_backdrop, 2.);
        scene.insert_primitive(BackdropFilter {
            bounds: at,
            content_mask: full,
            corner_radii: Corners::all(ScaledPixels(radius)),
            opacity,
            shader,
            ..Default::default()
        });
    }

    /// The scene and the probes every renderer must hit.
    pub fn scene() -> (Scene, Vec<Probe>) {
        let full = ContentMask::from(bounds(0., 0., SIZE.0 as f32, SIZE.1 as f32));
        let mut scene = Scene::default();
        // A red band behind the inverting shader.
        scene.insert_primitive(Quad {
            bounds: bounds(0., 40., SIZE.0 as f32, 24.),
            content_mask: full,
            background: color(1., 0., 0.).into(),
            ..Default::default()
        });
        // A blue rounded square from a uniform, and a half-faded one.
        draw(
            &mut scene,
            SOLID,
            &[0., 0., 1., 1.],
            bounds(4., 4., 24., 24.),
            10.,
            1.,
            false,
        );
        draw(
            &mut scene,
            SOLID,
            &[0., 1., 0., 1.],
            bounds(36., 4., 12., 12.),
            0.,
            0.5,
            false,
        );
        // A gradient across x 36..52, y 20..36 (green below 16 device pixels).
        draw(
            &mut scene,
            GRADIENT,
            &[],
            bounds(36., 20., 16., 32.),
            0.,
            1.,
            false,
        );
        // The red band inverted to cyan.
        draw(
            &mut scene,
            INVERT,
            &[],
            bounds(4., 44., 24., 16.),
            0.,
            1.,
            true,
        );
        scene.finish();
        let probe = |what, x, y, rgba| Probe { what, x, y, rgba };
        let probes = vec![
            probe("uniform color", 16, 16, [0, 0, 255, 255]),
            probe("rounded corner stays clear", 4, 4, [0, 0, 0, 255]),
            probe("half-faded uniform color", 42, 10, [0, 128, 0, 255]),
            probe("gradient start", 36, 24, [8, 0, 0, 255]),
            probe("gradient middle", 44, 24, [135, 0, 0, 255]),
            probe("gradient below 8 logical pixels", 36, 40, [8, 255, 0, 255]),
            probe("inverted red", 16, 52, [0, 255, 255, 255]),
            probe("red outside the shader", 32, 52, [255, 0, 0, 255]),
        ];
        (scene, probes)
    }
}
