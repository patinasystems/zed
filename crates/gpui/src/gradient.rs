//! CSS gradients ([`Gradient`]): linear, radial and conic, repeating or not, with any
//! number of color stops and transition hints.
//!
//! A gradient resolves against the box it fills when it is painted
//! ([`crate::Window::paint_gradient`]): stop positions, the gradient line, the radial
//! ending shape and the conic center become a [`GradientRecord`] of 16-byte texels in
//! the frame's [`crate::Scene::gradients`]. The quad that draws it refers to the record,
//! and its fragment shader finds the stop pair at each pixel and interpolates between
//! them. Nothing is rasterized on the CPU.

use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{ColorSpace, Hsla, Pixels, Rgba, Size, px};

/// A length along a gradient: `px` plus `fraction` of a reference length, like CSS
/// `calc(<fraction × 100>% + <px>px)`. The reference depends on where it is used: the
/// gradient line or ray for stop positions, the box's width or height for centers and
/// ellipse radii. Conic gradients measure stop positions in turns (`fraction`) and
/// ignore `px`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GradientLength {
    /// The absolute part, in logical pixels.
    pub px: Pixels,
    /// The relative part, a fraction of the reference length (`0.5` is `50%`).
    pub fraction: f32,
}

impl GradientLength {
    /// A fraction of the reference length (`0.5` is `50%`).
    pub fn fraction(fraction: f32) -> Self {
        Self {
            px: px(0.),
            fraction,
        }
    }

    /// An absolute length in logical pixels.
    pub fn px(length: impl Into<Pixels>) -> Self {
        Self {
            px: length.into(),
            fraction: 0.,
        }
    }

    /// The length in pixels against `reference` pixels.
    pub fn resolve(&self, reference: f32) -> f32 {
        self.px.0 + self.fraction * reference
    }
}

/// One entry of a gradient's stop list.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum GradientStop {
    /// A color stop. Without a position it sits between its neighbors (CSS color stop
    /// fixup): the first at 0%, the last at 100%, runs spread evenly.
    Color {
        /// The stop's color.
        color: Hsla,
        /// Where along the gradient it sits.
        position: Option<GradientLength>,
    },
    /// A transition hint: where the color halfway between the stops on either side of it
    /// is reached (CSS `<linear-color-hint>`). Ignored unless it sits between two color
    /// stops.
    Hint(GradientLength),
}

/// A color stop at `position`.
pub fn gradient_stop(color: impl Into<Hsla>, position: GradientLength) -> GradientStop {
    GradientStop::Color {
        color: color.into(),
        position: Some(position),
    }
}

/// The direction of a linear gradient.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum LinearDirection {
    /// An angle in degrees: `0` points up, `90` right (CSS `<angle>`).
    Angle(f32),
    /// Toward a corner (CSS `to top right`): the gradient line is perpendicular to the
    /// diagonal between the two other corners, so their colors match.
    ToCorner {
        /// Toward a right corner, else a left one.
        right: bool,
        /// Toward a bottom corner, else a top one.
        bottom: bool,
    },
}

/// The ending shape of a radial gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum RadialShape {
    /// A circle.
    Circle,
    /// An axis-aligned ellipse.
    Ellipse,
}

/// The size of a radial gradient's ending shape.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum RadialSize {
    /// Meets the side of the box closest to the center (per axis for an ellipse).
    ClosestSide,
    /// Meets the side farthest from the center (per axis for an ellipse).
    FarthestSide,
    /// Passes through the closest corner; an ellipse keeps the `ClosestSide` aspect ratio.
    ClosestCorner,
    /// Passes through the farthest corner; an ellipse keeps the `FarthestSide` aspect ratio.
    FarthestCorner,
    /// Explicit radii: horizontal against the box width, vertical against its height. A
    /// circle uses the first.
    Explicit(GradientLength, GradientLength),
}

/// The geometry of a [`Gradient`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum GradientKind {
    /// CSS `linear-gradient`.
    Linear(LinearDirection),
    /// CSS `radial-gradient`.
    Radial {
        /// Circle or ellipse.
        shape: RadialShape,
        /// The ending shape's size.
        size: RadialSize,
        /// The center, against the box's width and height.
        center: (GradientLength, GradientLength),
    },
    /// CSS `conic-gradient`.
    Conic {
        /// Where the gradient starts, in degrees clockwise from up.
        from: f32,
        /// The center, against the box's width and height.
        center: (GradientLength, GradientLength),
    },
}

/// A CSS gradient image: its geometry, its stops, whether it repeats, and the color space
/// it interpolates in. Paint it with [`crate::Window::paint_gradient`] or as an element's
/// background ([`crate::Fill::Gradient`]).
///
/// Colors interpolate with premultiplied alpha, as CSS does: a stop that is transparent
/// takes on its neighbor's color. [`ColorSpace::Srgb`] interpolates gamma-encoded sRGB,
/// the CSS default for these colors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Gradient {
    /// The geometry.
    pub kind: GradientKind,
    /// Color stops and hints, in order.
    pub stops: Arc<[GradientStop]>,
    /// Whether the stops repeat past the first and last (`repeating-*-gradient`).
    pub repeating: bool,
    /// The color space colors interpolate in.
    pub color_space: ColorSpace,
}

impl Gradient {
    /// A linear gradient.
    pub fn linear(direction: LinearDirection, stops: impl Into<Arc<[GradientStop]>>) -> Self {
        Self::new(GradientKind::Linear(direction), stops)
    }

    /// A radial gradient.
    pub fn radial(
        shape: RadialShape,
        size: RadialSize,
        center: (GradientLength, GradientLength),
        stops: impl Into<Arc<[GradientStop]>>,
    ) -> Self {
        Self::new(
            GradientKind::Radial {
                shape,
                size,
                center,
            },
            stops,
        )
    }

    /// A conic gradient starting `from` degrees clockwise from up.
    pub fn conic(
        from: f32,
        center: (GradientLength, GradientLength),
        stops: impl Into<Arc<[GradientStop]>>,
    ) -> Self {
        Self::new(GradientKind::Conic { from, center }, stops)
    }

    fn new(kind: GradientKind, stops: impl Into<Arc<[GradientStop]>>) -> Self {
        Self {
            kind,
            stops: stops.into(),
            repeating: false,
            color_space: ColorSpace::Srgb,
        }
    }

    /// Repeats the stops past the first and last.
    pub fn repeating(mut self, repeating: bool) -> Self {
        self.repeating = repeating;
        self
    }

    /// Interpolates in `color_space`.
    pub fn color_space(mut self, color_space: ColorSpace) -> Self {
        self.color_space = color_space;
        self
    }

    /// Whether every color stop is transparent (or there are none).
    pub fn is_transparent(&self) -> bool {
        self.stops.iter().all(|stop| match stop {
            GradientStop::Color { color, .. } => color.is_transparent(),
            GradientStop::Hint(_) => true,
        })
    }

    /// The first color stop's color, for renderers that draw a flat stand-in.
    pub fn first_color(&self) -> Hsla {
        self.stops
            .iter()
            .find_map(|stop| match stop {
                GradientStop::Color { color, .. } => Some(*color),
                GradientStop::Hint(_) => None,
            })
            .unwrap_or_default()
    }

    /// Resolves the gradient against a box of `size` device pixels, `scale_factor` device
    /// pixels per logical pixel, into the texels the quad shaders read. `None` when the
    /// box is empty or there are no color stops.
    pub fn resolve(&self, size: Size<f32>, scale_factor: f32) -> Option<GradientRecord> {
        let (width, height) = (size.width, size.height);
        if !(width > 0. && height > 0.) {
            return None;
        }
        // Lengths below resolve in device pixels.
        let device = |length: &GradientLength, reference: f32| {
            length.px.0 * scale_factor + length.fraction * reference
        };
        let (kind, geometry, line) = match self.kind {
            GradientKind::Linear(direction) => {
                let (dx, dy) = match direction {
                    LinearDirection::Angle(degrees) => {
                        let radians = degrees.to_radians();
                        (radians.sin(), -radians.cos())
                    }
                    LinearDirection::ToCorner { right, bottom } => {
                        // Perpendicular to the diagonal between the other two corners.
                        let (x, y) = (
                            if right { height } else { -height },
                            if bottom { width } else { -width },
                        );
                        let length = x.hypot(y);
                        (x / length, y / length)
                    }
                };
                // The gradient line passes through the center, and the corners it points
                // from and to lie on the perpendiculars at its ends.
                let length = (width * dx).abs() + (height * dy).abs();
                let (cx, cy) = (width / 2., height / 2.);
                let start = (cx - dx * length / 2., cy - dy * length / 2.);
                let end = (cx + dx * length / 2., cy + dy * length / 2.);
                (
                    GradientRecord::LINEAR,
                    [
                        start.0 / width,
                        start.1 / height,
                        end.0 / width,
                        end.1 / height,
                    ],
                    length,
                )
            }
            GradientKind::Radial {
                shape,
                size: ending,
                center,
            } => {
                let cx = device(&center.0, width);
                let cy = device(&center.1, height);
                let (rx, ry) = radial_radii(shape, ending, cx, cy, width, height, &device);
                // CSS draws a degenerate ending shape as a vanishingly small one.
                let (rx, ry) = (rx.max(RADIUS_EPSILON), ry.max(RADIUS_EPSILON));
                (
                    GradientRecord::RADIAL,
                    [cx / width, cy / height, rx / width, ry / height],
                    rx,
                )
            }
            GradientKind::Conic { from, center } => {
                let cx = device(&center.0, width);
                let cy = device(&center.1, height);
                (
                    GradientRecord::CONIC,
                    [cx / width, cy / height, from.to_radians(), 0.],
                    0.,
                )
            }
        };

        let conic = kind == GradientRecord::CONIC;
        let position = |length: &GradientLength| {
            if conic {
                length.fraction
            } else if line > 0. {
                length.px.0 * scale_factor / line + length.fraction
            } else {
                length.fraction
            }
        };
        let mut stops = fixup_stops(&self.stops, position)?;
        let mut repeating = self.repeating;
        if repeating {
            let first = stops.first().map_or(0., |stop| stop.position);
            let last = stops.last().map_or(0., |stop| stop.position);
            if last - first <= f32::EPSILON {
                // A zero-length repeating gradient paints its average color.
                stops = vec![ResolvedStop {
                    color: average_color(&stops),
                    position: 0.,
                    hint: 0.5,
                }];
                repeating = false;
            }
        }

        let mut texels = Vec::with_capacity(2 + stops.len() * 2);
        texels.push([
            kind as f32,
            stops.len() as f32,
            repeating as u32 as f32,
            match self.color_space {
                ColorSpace::Srgb => 0.,
                ColorSpace::Oklab => 1.,
            },
        ]);
        texels.push(geometry);
        for stop in &stops {
            texels.push(premultiplied(stop.color, self.color_space));
            texels.push([stop.position, stop.hint, 0., 0.]);
        }
        Some(GradientRecord { texels })
    }
}

impl From<Gradient> for crate::Fill {
    fn from(gradient: Gradient) -> Self {
        crate::Fill::Gradient(gradient)
    }
}

/// Radii below this many device pixels draw as this, as CSS draws degenerate ending
/// shapes: the gradient's last color everywhere but its center.
const RADIUS_EPSILON: f32 = 1e-3;

fn radial_radii(
    shape: RadialShape,
    size: RadialSize,
    cx: f32,
    cy: f32,
    width: f32,
    height: f32,
    device: &impl Fn(&GradientLength, f32) -> f32,
) -> (f32, f32) {
    let (left, right) = (cx.abs(), (width - cx).abs());
    let (top, bottom) = (cy.abs(), (height - cy).abs());
    let corners = [(left, top), (right, top), (right, bottom), (left, bottom)];
    let corner = |farthest: bool| {
        let mut best = corners[0];
        for &candidate in &corners[1..] {
            let (a, b) = (candidate.0.hypot(candidate.1), best.0.hypot(best.1));
            if (farthest && a > b) || (!farthest && a < b) {
                best = candidate;
            }
        }
        best
    };
    match (shape, size) {
        (RadialShape::Circle, RadialSize::ClosestSide) => {
            let r = left.min(right).min(top).min(bottom);
            (r, r)
        }
        (RadialShape::Circle, RadialSize::FarthestSide) => {
            let r = left.max(right).max(top).max(bottom);
            (r, r)
        }
        (RadialShape::Circle, RadialSize::ClosestCorner) => {
            let (x, y) = corner(false);
            let r = x.hypot(y);
            (r, r)
        }
        (RadialShape::Circle, RadialSize::FarthestCorner) => {
            let (x, y) = corner(true);
            let r = x.hypot(y);
            (r, r)
        }
        (RadialShape::Circle, RadialSize::Explicit(radius, _)) => {
            let r = device(&radius, width).max(0.);
            (r, r)
        }
        (RadialShape::Ellipse, RadialSize::ClosestSide) => (left.min(right), top.min(bottom)),
        (RadialShape::Ellipse, RadialSize::FarthestSide) => (left.max(right), top.max(bottom)),
        (RadialShape::Ellipse, RadialSize::ClosestCorner | RadialSize::FarthestCorner) => {
            let farthest = size == RadialSize::FarthestCorner;
            let (sx, sy) = if farthest {
                (left.max(right), top.max(bottom))
            } else {
                (left.min(right), top.min(bottom))
            };
            let (x, y) = corner(farthest);
            if sx <= 0. || sy <= 0. {
                return (sx, sy);
            }
            // The ellipse with the sides' aspect ratio through that corner.
            let ratio = sx / sy;
            let ry = ((x / ratio).powi(2) + y * y).sqrt();
            (ratio * ry, ry)
        }
        (RadialShape::Ellipse, RadialSize::Explicit(rx, ry)) => {
            (device(&rx, width).max(0.), device(&ry, height).max(0.))
        }
    }
}

/// A color stop after fixup: its position along the gradient (0 to 1 over the gradient
/// line or ray, or a turn) and the hint toward it from the stop before, as a fraction of
/// the segment between them (`0.5` interpolates linearly).
#[derive(Debug, Clone, Copy, PartialEq)]
struct ResolvedStop {
    color: Hsla,
    position: f32,
    hint: f32,
}

/// CSS color stop fixup: the first and last stops default to 0 and 1, no stop or hint
/// sits before an earlier one, and runs without positions spread evenly. Hints become
/// fractions of their segment.
fn fixup_stops(
    stops: &[GradientStop],
    position: impl Fn(&GradientLength) -> f32,
) -> Option<Vec<ResolvedStop>> {
    // Color stops with optional positions, each with the hint before it.
    let mut colors: Vec<(Hsla, Option<f32>, Option<f32>)> = Vec::new();
    let mut pending_hint = None;
    for stop in stops {
        match stop {
            GradientStop::Color {
                color,
                position: at,
            } => {
                colors.push((*color, at.as_ref().map(&position), pending_hint.take()));
            }
            // A hint counts only between two color stops, one per segment.
            GradientStop::Hint(at) if !colors.is_empty() && pending_hint.is_none() => {
                pending_hint = Some(position(at));
            }
            GradientStop::Hint(_) => {}
        }
    }
    let count = colors.len();
    if count == 0 {
        return None;
    }
    if colors[0].1.is_none() {
        colors[0].1 = Some(0.);
    }
    if count > 1 && colors[count - 1].1.is_none() {
        colors[count - 1].1 = Some(1.);
    }
    // Clamp positions (and hints) to the largest before them.
    let mut largest = f32::NEG_INFINITY;
    for (_, at, hint) in &mut colors {
        if let Some(hint) = hint {
            *hint = hint.max(largest);
            largest = *hint;
        }
        if let Some(at) = at {
            *at = at.max(largest);
            largest = *at;
        }
    }
    // Spread runs without positions between their neighbors.
    let mut index = 0;
    while index < count {
        if colors[index].1.is_some() {
            index += 1;
            continue;
        }
        let start = index - 1;
        let mut end = index;
        while colors[end].1.is_none() {
            end += 1;
        }
        let (from, to) = (colors[start].1.unwrap_or(0.), colors[end].1.unwrap_or(1.));
        for (step, entry) in colors[index..end].iter_mut().enumerate() {
            let t = (step + 1) as f32 / (end - start) as f32;
            entry.1 = Some(from + (to - from) * t);
        }
        index = end;
    }
    let mut resolved = Vec::with_capacity(count);
    for (index, (color, at, hint)) in colors.iter().enumerate() {
        let at = at.unwrap_or(0.);
        let hint = match (hint, index) {
            (Some(hint), 1..) => {
                let before = resolved
                    .last()
                    .map_or(at, |stop: &ResolvedStop| stop.position);
                // Spread positions can leave a hint outside its segment.
                let hint = hint.clamp(before, at.max(before));
                if at > before {
                    (hint - before) / (at - before)
                } else {
                    0.5
                }
            }
            _ => 0.5,
        };
        resolved.push(ResolvedStop {
            color: *color,
            position: at,
            hint,
        });
    }
    Some(resolved)
}

/// The average color of a gradient whose stops spread evenly over one length, with
/// premultiplied alpha.
fn average_color(stops: &[ResolvedStop]) -> Hsla {
    let premultiplied = |color: Hsla| {
        let rgba = Rgba::from(color);
        [rgba.r * rgba.a, rgba.g * rgba.a, rgba.b * rgba.a, rgba.a]
    };
    let mut sum = [0f32; 4];
    if stops.len() == 1 {
        sum = premultiplied(stops[0].color);
    } else {
        // Each segment contributes the mean of its ends.
        let segments = (stops.len() - 1) as f32;
        for pair in stops.windows(2) {
            let (a, b) = (premultiplied(pair[0].color), premultiplied(pair[1].color));
            for channel in 0..4 {
                sum[channel] += (a[channel] + b[channel]) / 2. / segments;
            }
        }
    }
    let alpha = sum[3];
    let unpremultiply = |value: f32| if alpha > 0. { value / alpha } else { 0. };
    Rgba {
        r: unpremultiply(sum[0]),
        g: unpremultiply(sum[1]),
        b: unpremultiply(sum[2]),
        a: alpha,
    }
    .into()
}

/// A stop's color in the interpolation space, alpha premultiplied.
fn premultiplied(color: Hsla, color_space: ColorSpace) -> [f32; 4] {
    let rgba = Rgba::from(color);
    let channels = match color_space {
        ColorSpace::Srgb => [rgba.r, rgba.g, rgba.b],
        ColorSpace::Oklab => srgb_to_oklab([rgba.r, rgba.g, rgba.b]),
    };
    [
        channels[0] * rgba.a,
        channels[1] * rgba.a,
        channels[2] * rgba.a,
        rgba.a,
    ]
}

fn srgb_to_oklab(srgb: [f32; 3]) -> [f32; 3] {
    let linear = srgb.map(|c| {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    });
    let [r, g, b] = linear;
    let l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b).cbrt();
    let m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b).cbrt();
    let s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b).cbrt();
    [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]
}

/// A resolved gradient as the quad shaders read it, in 16-byte texels:
///
/// - header: kind ([`Self::LINEAR`], [`Self::RADIAL`], [`Self::CONIC`]), stop count,
///   repeating (0 or 1), color space (0 sRGB, 1 Oklab);
/// - geometry, in fractions of the box's width and height: a linear gradient's line
///   (start x, y, end x, y); a radial one's center and radii (x, y, rx, ry); a conic
///   one's center and start angle in radians (x, y, angle, 0);
/// - per stop, its premultiplied color in the interpolation space, then its position
///   and the hint before it (position, hint, 0, 0).
#[derive(Debug, Clone, PartialEq)]
pub struct GradientRecord {
    /// The texels.
    pub texels: Vec<[f32; 4]>,
}

impl GradientRecord {
    /// A linear gradient's kind.
    pub const LINEAR: u32 = 0;
    /// A radial gradient's kind.
    pub const RADIAL: u32 = 1;
    /// A conic gradient's kind.
    pub const CONIC: u32 = 2;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{hsla, size};

    fn red() -> Hsla {
        hsla(0., 1., 0.5, 1.)
    }

    fn blue() -> Hsla {
        hsla(2. / 3., 1., 0.5, 1.)
    }

    fn stop(color: Hsla, at: Option<f32>) -> GradientStop {
        GradientStop::Color {
            color,
            position: at.map(GradientLength::fraction),
        }
    }

    fn positions(gradient: &Gradient, width: f32, height: f32) -> Vec<(f32, f32)> {
        let record = gradient.resolve(size(width, height), 1.).unwrap();
        record.texels[2..]
            .chunks(2)
            .map(|pair| (pair[1][0], pair[1][1]))
            .collect()
    }

    #[test]
    fn fixup_spreads_and_clamps_positions() {
        let gradient = Gradient::linear(
            LinearDirection::Angle(90.),
            vec![
                stop(red(), None),
                stop(blue(), Some(0.6)),
                stop(red(), Some(0.2)),
                stop(blue(), None),
                stop(red(), None),
            ],
        );
        let at: Vec<f32> = positions(&gradient, 100., 10.)
            .iter()
            .map(|p| p.0)
            .collect();
        assert_eq!(at, vec![0., 0.6, 0.6, 0.8, 1.]);
    }

    #[test]
    fn pixel_positions_resolve_against_the_gradient_line() {
        // 90deg over 200 x 50: the line is 200 px long.
        let gradient = Gradient::linear(
            LinearDirection::Angle(90.),
            vec![
                gradient_stop(red(), GradientLength::px(50.)),
                gradient_stop(
                    blue(),
                    GradientLength {
                        px: px(-20.),
                        fraction: 1.,
                    },
                ),
            ],
        );
        let at = positions(&gradient, 200., 50.);
        assert_eq!(at[0].0, 0.25);
        assert_eq!(at[1].0, 0.9);
        // At scale factor 2, the same logical pixels against a 400 device px line.
        let record = gradient.resolve(size(400., 100.), 2.).unwrap();
        assert_eq!(record.texels[3][0], 0.25);
    }

    #[test]
    fn hints_become_fractions_of_their_segment() {
        let gradient = Gradient::linear(
            LinearDirection::Angle(180.),
            vec![
                stop(red(), Some(0.2)),
                GradientStop::Hint(GradientLength::fraction(0.3)),
                stop(blue(), Some(0.6)),
            ],
        );
        let at = positions(&gradient, 10., 10.);
        assert_eq!(at[0], (0.2, 0.5));
        assert!((at[1].1 - 0.25).abs() < 1e-6, "{at:?}");
    }

    #[test]
    fn linear_line_spans_the_box_corners() {
        // 45deg over a 200 x 100 box: length = 200 sin 45 + 100 cos 45.
        let gradient = Gradient::linear(
            LinearDirection::Angle(45.),
            vec![stop(red(), None), stop(blue(), None)],
        );
        let record = gradient.resolve(size(200., 100.), 1.).unwrap();
        let [sx, sy, ex, ey] = record.texels[1];
        let length = ((ex - sx) * 200.).hypot((ey - sy) * 100.);
        assert!((length - 300. * std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3);
        // `to top right` on a square box is 45deg.
        let corner = Gradient::linear(
            LinearDirection::ToCorner {
                right: true,
                bottom: false,
            },
            vec![stop(red(), None), stop(blue(), None)],
        );
        let square = Gradient {
            kind: GradientKind::Linear(LinearDirection::Angle(45.)),
            ..corner.clone()
        };
        let a = corner.resolve(size(100., 100.), 1.).unwrap().texels[1];
        let b = square.resolve(size(100., 100.), 1.).unwrap().texels[1];
        for i in 0..4 {
            assert!((a[i] - b[i]).abs() < 1e-5, "{a:?} {b:?}");
        }
    }

    #[test]
    fn radial_sizes() {
        let center = (GradientLength::px(20.), GradientLength::px(10.));
        let radii = |shape, size_keyword| {
            let gradient = Gradient::radial(
                shape,
                size_keyword,
                center,
                vec![stop(red(), None), stop(blue(), None)],
            );
            let texel = gradient.resolve(size(100., 50.), 1.).unwrap().texels[1];
            (texel[2] * 100., texel[3] * 50.)
        };
        let close = |(a, b): (f32, f32), (c, d): (f32, f32)| {
            assert!(
                (a - c).abs() < 1e-3 && (b - d).abs() < 1e-3,
                "{a},{b} vs {c},{d}"
            );
        };
        // Sides: left 20, right 80, top 10, bottom 40.
        close(
            radii(RadialShape::Circle, RadialSize::ClosestSide),
            (10., 10.),
        );
        close(
            radii(RadialShape::Circle, RadialSize::FarthestSide),
            (80., 80.),
        );
        let near = 20f32.hypot(10.);
        let far = 80f32.hypot(40.);
        close(
            radii(RadialShape::Circle, RadialSize::ClosestCorner),
            (near, near),
        );
        close(
            radii(RadialShape::Circle, RadialSize::FarthestCorner),
            (far, far),
        );
        close(
            radii(RadialShape::Ellipse, RadialSize::ClosestSide),
            (20., 10.),
        );
        close(
            radii(RadialShape::Ellipse, RadialSize::FarthestSide),
            (80., 40.),
        );
        // Through the corner with the sides' aspect ratio: sqrt(2) times the sides.
        let sqrt2 = std::f32::consts::SQRT_2;
        close(
            radii(RadialShape::Ellipse, RadialSize::ClosestCorner),
            (20. * sqrt2, 10. * sqrt2),
        );
        close(
            radii(RadialShape::Ellipse, RadialSize::FarthestCorner),
            (80. * sqrt2, 40. * sqrt2),
        );
        close(
            radii(
                RadialShape::Ellipse,
                RadialSize::Explicit(GradientLength::fraction(0.5), GradientLength::px(5.)),
            ),
            (50., 5.),
        );
    }

    #[test]
    fn zero_length_repeating_gradients_paint_their_average() {
        let gradient = Gradient::linear(
            LinearDirection::Angle(90.),
            vec![stop(red(), Some(0.5)), stop(blue(), Some(0.5))],
        )
        .repeating(true);
        let record = gradient.resolve(size(10., 10.), 1.).unwrap();
        assert_eq!(record.texels[0], [0., 1., 0., 0.]);
        let color = record.texels[2];
        assert!((color[0] - 0.5).abs() < 1e-3 && (color[2] - 0.5).abs() < 1e-3);
    }

    fn gradient_quad(scene: &mut crate::Scene, gradient: &Gradient) -> crate::Quad {
        let record = gradient.resolve(size(10., 10.), 1.).unwrap();
        let start = scene.add_gradient(&record.texels);
        crate::Quad {
            bounds: crate::Bounds::new(
                crate::point(crate::ScaledPixels(0.), crate::ScaledPixels(0.)),
                size(crate::ScaledPixels(10.), crate::ScaledPixels(10.)),
            ),
            content_mask: crate::ContentMask::from(crate::Bounds::new(
                crate::point(crate::ScaledPixels(0.), crate::ScaledPixels(0.)),
                size(crate::ScaledPixels(100.), crate::ScaledPixels(100.)),
            )),
            background: crate::Background::painted_gradient(start, gradient.first_color()),
            ..Default::default()
        }
    }

    fn texels_of(scene: &crate::Scene, quad: &crate::Quad) -> Vec<[f32; 4]> {
        let start = quad.background.gradient as usize;
        let count = scene.gradients[start][1] as usize;
        scene.gradients[start..start + 2 + 2 * count].to_vec()
    }

    #[test]
    fn replayed_and_ungrouped_quads_take_their_gradients_along() {
        let first = Gradient::linear(
            LinearDirection::Angle(0.),
            vec![stop(red(), None), stop(blue(), None)],
        );
        let second = Gradient::conic(
            30.,
            (GradientLength::fraction(0.5), GradientLength::fraction(0.5)),
            vec![stop(blue(), None), stop(red(), None), stop(blue(), None)],
        );
        let mut previous = crate::Scene::default();
        let a = gradient_quad(&mut previous, &first);
        previous.insert_primitive(a);
        let b = gradient_quad(&mut previous, &second);
        previous.push_group(crate::GroupEffects {
            opacity: 0.5,
            content_mask: a.content_mask,
            ..Default::default()
        });
        previous.insert_primitive(b);
        previous.pop_group();
        previous.finish();
        let expected = [texels_of(&previous, &a), texels_of(&previous, &b)];

        // Replaying the second quad into a scene that has a gradient of its own.
        let mut next = crate::Scene::default();
        let own = gradient_quad(&mut next, &second);
        next.insert_primitive(own);
        next.replay(1..previous.len(), &previous);
        next.finish();
        // The group fades its one quad instead of drawing offscreen.
        assert!(next.groups.is_empty());
        let replayed = next.quads[1];
        assert_eq!(texels_of(&next, &next.quads[0]), expected[1]);
        assert_eq!(texels_of(&next, &replayed), expected[1]);
        assert_ne!(replayed.background.gradient, b.background.gradient);

        // Without groups, both quads keep their records.
        let flat = previous.without_groups();
        assert_eq!(flat.quads.len(), 2);
        assert_eq!(texels_of(&flat, &flat.quads[0]), expected[0]);
        assert_eq!(texels_of(&flat, &flat.quads[1]), expected[1]);
        assert_eq!(flat.quads[1].background.colors[0].color.a, 0.5, "faded");
    }

    #[test]
    fn colors_are_premultiplied() {
        let gradient = Gradient::linear(
            LinearDirection::Angle(90.),
            vec![stop(red().opacity(0.5), None), stop(blue(), None)],
        );
        let record = gradient.resolve(size(10., 10.), 1.).unwrap();
        assert_eq!(record.texels[2], [0.5, 0., 0., 0.5]);
        assert!(gradient.resolve(size(0., 10.), 1.).is_none());
    }
}

/// Gradients for renderer tests: every backend renders the scene and must match the
/// probes, which evaluate each gradient's record on the CPU at the probed pixel's
/// center (over a white page). The scene holds:
/// - linear gradients: four hard stops; a transition hint toward a translucent stop; a
///   repeating one with pixel stops at 45 degrees; one toward a corner in Oklab;
/// - radial gradients: circles and ellipses for every size keyword and explicit radii,
///   off-center, and a repeating one;
/// - conic gradients, plain and repeating;
/// - a gradient with rounded corners, one rotated and scaled, and one in a half-opaque
///   compositing group over a red square.
#[cfg(any(test, feature = "test-support", feature = "bench-support"))]
#[doc(hidden)]
pub mod gradient_test_scene {
    use super::*;
    pub use crate::retained_shapes_test_scene::{Probe, check};
    use crate::{
        Background, Bounds, ContentMask, Corners, GroupEffects, Quad, ScaledPixels, Scene,
        TransformationMatrix, element_transformation_test_scene::about, point, rgba, size,
    };

    /// The scene's size in device pixels.
    pub const SIZE: (i32, i32) = (128, 112);

    fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(w), ScaledPixels(h)),
        )
    }

    fn full() -> ContentMask<ScaledPixels> {
        ContentMask::from(bounds(0., 0., SIZE.0 as f32, SIZE.1 as f32))
    }

    fn color(hex: u32) -> Hsla {
        rgba(hex).into()
    }

    fn at(color: u32, fraction: f32) -> GradientStop {
        gradient_stop(self::color(color), GradientLength::fraction(fraction))
    }

    fn auto(color: u32) -> GradientStop {
        GradientStop::Color {
            color: self::color(color),
            position: None,
        }
    }

    fn center(x: f32, y: f32) -> (GradientLength, GradientLength) {
        (GradientLength::fraction(x), GradientLength::fraction(y))
    }

    /// One drawn gradient: where it is and how to evaluate it.
    struct Drawn {
        bounds: Bounds<ScaledPixels>,
        texels: Vec<[f32; 4]>,
        transformation: TransformationMatrix,
        radius: f32,
        /// A solid color drawn under it inside a half-opaque group, if any.
        group_under: Option<[f32; 4]>,
    }

    /// The gradient's straight-alpha color at `point` in its bounds' frame, as the
    /// shaders compute it.
    pub fn evaluate(texels: &[[f32; 4]], size: (f32, f32), point: (f32, f32)) -> [f32; 4] {
        let [kind, count, repeating, space] = texels[0];
        let geometry = texels[1];
        let count = count as usize;
        let (w, h) = size;
        let mut t = if kind == 0. {
            let (sx, sy) = (geometry[0] * w, geometry[1] * h);
            let (lx, ly) = (
                (geometry[2] - geometry[0]) * w,
                (geometry[3] - geometry[1]) * h,
            );
            ((point.0 - sx) * lx + (point.1 - sy) * ly) / (lx * lx + ly * ly)
        } else if kind == 1. {
            let dx = (point.0 - geometry[0] * w) / (geometry[2] * w);
            let dy = (point.1 - geometry[1] * h) / (geometry[3] * h);
            dx.hypot(dy)
        } else {
            let (dx, dy) = (point.0 - geometry[0] * w, point.1 - geometry[1] * h);
            let turn = (dx.atan2(-dy) - geometry[2]) / (2. * std::f32::consts::PI);
            turn - turn.floor()
        };
        let stop = |i: usize| (texels[2 + 2 * i], texels[3 + 2 * i]);
        let first = stop(0).1[0];
        if repeating != 0. {
            let period = stop(count - 1).1[0] - first;
            let r = (t - first) / period;
            t = first + period * (r - r.floor());
        }
        let mut color = stop(0).0;
        if t > first {
            color = stop(count - 1).0;
            for i in 1..count {
                let (after, place) = stop(i);
                if t < place[0] {
                    let (before, previous) = stop(i - 1);
                    let local = (t - previous[0]) / (place[0] - previous[0]);
                    let hint = place[1];
                    let weight = if hint == 0.5 {
                        local
                    } else if hint <= 0. {
                        1.
                    } else if hint >= 1. {
                        0.
                    } else {
                        local.powf(0.5f32.ln() / hint.ln())
                    };
                    for c in 0..4 {
                        color[c] = before[c] + (after[c] - before[c]) * weight;
                    }
                    break;
                }
            }
        }
        let alpha = color[3];
        let mut rgb = if alpha > 0. {
            [color[0] / alpha, color[1] / alpha, color[2] / alpha]
        } else {
            [0.; 3]
        };
        if space == 1. {
            rgb = oklab_to_srgb(rgb);
        }
        [rgb[0], rgb[1], rgb[2], alpha]
    }

    fn oklab_to_srgb(lab: [f32; 3]) -> [f32; 3] {
        let l_ = lab[0] + 0.3963377774 * lab[1] + 0.2158037573 * lab[2];
        let m_ = lab[0] - 0.1055613458 * lab[1] - 0.0638541728 * lab[2];
        let s_ = lab[0] - 0.0894841775 * lab[1] - 1.2914855480 * lab[2];
        let (l, m, s) = (l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
        [
            4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        ]
        .map(|c| {
            if c < 0.0031308 {
                c * 12.92
            } else {
                1.055 * c.max(0.).powf(1. / 2.4) - 0.055
            }
        })
    }

    fn over(below: [f32; 3], above: [f32; 4]) -> [f32; 3] {
        [0, 1, 2].map(|c| above[c] * above[3] + below[c] * (1. - above[3]))
    }

    /// `count` overlapping quads, each a quarter of a `width` x `height` target, filled
    /// with a solid color, or with `stops` stops of a radial or linear gradient: a
    /// fill-rate benchmark of gradients against solid fills.
    pub fn bench(count: usize, width: f32, height: f32, fill: BenchFill) -> Scene {
        let mut scene = Scene::default();
        let mask = ContentMask::from(bounds(0., 0., width, height));
        for i in 0..count {
            let (x, y) = (
                (i * 37 % 100) as f32 / 100. * width / 2.,
                (i * 61 % 100) as f32 / 100. * height / 2.,
            );
            let at = bounds(x, y, width / 2., height / 2.);
            let background = match fill {
                BenchFill::Solid => color(0x4f7cffff).into(),
                BenchFill::TwoStop => crate::linear_gradient(
                    30.,
                    crate::linear_color_stop(color(0xff0000ff), 0.),
                    crate::linear_color_stop(color(0x0000ffff), 1.),
                ),
                BenchFill::Linear(stops) | BenchFill::Radial(stops) => {
                    let colors = [0xff0000ff, 0x00ff00ff, 0x0000ffff, 0xffffffff];
                    let stops: Vec<GradientStop> =
                        (0..stops).map(|k| auto(colors[k % colors.len()])).collect();
                    let gradient = match fill {
                        BenchFill::Radial(_) => Gradient::radial(
                            RadialShape::Ellipse,
                            RadialSize::FarthestCorner,
                            center(0.3, 0.6),
                            stops,
                        ),
                        _ => Gradient::linear(LinearDirection::Angle(30.), stops),
                    };
                    let record = gradient
                        .resolve(size(at.size.width.0, at.size.height.0), 1.)
                        .expect("resolves");
                    let start = scene.add_gradient(&record.texels);
                    Background::painted_gradient(start, gradient.first_color())
                }
            };
            scene.insert_primitive(Quad {
                bounds: at,
                content_mask: mask,
                background,
                corner_radii: Corners::all(ScaledPixels(12.)),
                corner_radii_y: Corners::all(ScaledPixels(12.)),
                ..Default::default()
            });
        }
        scene.finish();
        scene
    }

    /// What [`bench`] fills its quads with.
    #[derive(Clone, Copy, Debug)]
    pub enum BenchFill {
        /// A solid color.
        Solid,
        /// GPUI's two-stop [`crate::linear_gradient`] background.
        TwoStop,
        /// A linear gradient with this many stops.
        Linear(usize),
        /// A radial gradient with this many stops.
        Radial(usize),
    }

    /// The scene and the probes every renderer must hit.
    pub fn scene() -> (Scene, Vec<Probe>) {
        let mut scene = Scene::default();
        scene.insert_primitive(Quad {
            bounds: bounds(0., 0., SIZE.0 as f32, SIZE.1 as f32),
            content_mask: full(),
            background: color(0xffffffff).into(),
            ..Default::default()
        });
        let mut drawn = Vec::new();
        let mut paint = |scene: &mut Scene,
                         gradient: Gradient,
                         at: Bounds<ScaledPixels>,
                         radius: f32,
                         transformation: TransformationMatrix,
                         group_under: Option<[f32; 4]>| {
            let record = gradient
                .resolve(size(at.size.width.0, at.size.height.0), 1.)
                .expect("resolves");
            let start = scene.add_gradient(&record.texels);
            scene.insert_primitive(Quad {
                bounds: at,
                content_mask: full(),
                background: Background::painted_gradient(start, gradient.first_color()),
                corner_radii: Corners::all(ScaledPixels(radius)),
                corner_radii_y: Corners::all(ScaledPixels(radius)),
                transformation,
                ..Default::default()
            });
            drawn.push(Drawn {
                bounds: at,
                texels: record.texels,
                transformation,
                radius,
                group_under,
            });
        };
        let unit = TransformationMatrix::unit();
        let (red, green, blue, white) = (0xff0000ff, 0x00ff00ff, 0x0000ffff, 0xffffffff);

        // Linear: four hard stops (CSS `90deg, red 0 25%, green 25% 50%, …`).
        let hard = Gradient::linear(
            LinearDirection::Angle(90.),
            vec![
                at(red, 0.),
                at(red, 0.25),
                at(green, 0.25),
                at(green, 0.5),
                at(blue, 0.5),
                at(blue, 0.75),
                at(white, 0.75),
            ],
        );
        paint(&mut scene, hard, bounds(0., 0., 64., 16.), 0., unit, None);
        // A hint at 25% toward a translucent blue.
        let hinted = Gradient::linear(
            LinearDirection::Angle(90.),
            vec![
                auto(red),
                GradientStop::Hint(GradientLength::fraction(0.25)),
                auto(0x0000ff33),
            ],
        );
        paint(
            &mut scene,
            hinted,
            bounds(64., 0., 64., 16.),
            0.,
            unit,
            None,
        );
        // Repeating 8 px stripes at 45 degrees.
        let stripes = Gradient::linear(
            LinearDirection::Angle(45.),
            vec![
                gradient_stop(color(red), GradientLength::px(0.)),
                gradient_stop(color(red), GradientLength::px(4.)),
                gradient_stop(color(blue), GradientLength::px(4.)),
                gradient_stop(color(blue), GradientLength::px(8.)),
            ],
        )
        .repeating(true);
        paint(
            &mut scene,
            stripes,
            bounds(0., 16., 64., 16.),
            0.,
            unit,
            None,
        );
        // Toward the top right corner, in Oklab.
        let corner = Gradient::linear(
            LinearDirection::ToCorner {
                right: true,
                bottom: false,
            },
            vec![auto(red), auto(0xffff00ff), auto(blue)],
        )
        .color_space(ColorSpace::Oklab);
        paint(
            &mut scene,
            corner,
            bounds(64., 16., 64., 16.),
            0.,
            unit,
            None,
        );

        // Radial: every size keyword, circles then ellipses, off-center in 32 x 16.
        let sizes = [
            RadialSize::ClosestSide,
            RadialSize::FarthestSide,
            RadialSize::ClosestCorner,
            RadialSize::FarthestCorner,
        ];
        for (row, shape) in [RadialShape::Circle, RadialShape::Ellipse]
            .into_iter()
            .enumerate()
        {
            for (column, size_keyword) in sizes.into_iter().enumerate() {
                let gradient = Gradient::radial(
                    shape,
                    size_keyword,
                    center(0.4, 0.6),
                    vec![auto(0xffff00ff), auto(blue), auto(0x000000ff)],
                );
                let at = bounds(column as f32 * 32., 32. + row as f32 * 16., 32., 16.);
                paint(&mut scene, gradient, at, 0., unit, None);
            }
        }
        // Explicit radii: a 6 px circle, and a 50% x 4 px ellipse, repeating.
        let circle = Gradient::radial(
            RadialShape::Circle,
            RadialSize::Explicit(GradientLength::px(6.), GradientLength::px(6.)),
            center(0.5, 0.5),
            vec![auto(red), auto(green)],
        );
        paint(
            &mut scene,
            circle,
            bounds(0., 64., 32., 16.),
            0.,
            unit,
            None,
        );
        let rings = Gradient::radial(
            RadialShape::Ellipse,
            RadialSize::Explicit(GradientLength::fraction(0.5), GradientLength::px(4.)),
            center(0.5, 0.5),
            vec![at(red, 0.), at(blue, 0.3)],
        )
        .repeating(true);
        paint(
            &mut scene,
            rings,
            bounds(32., 64., 32., 16.),
            0.,
            unit,
            None,
        );

        // Conic from 90 degrees, and repeating in quarter turns.
        let conic = Gradient::conic(
            90.,
            center(0.5, 0.5),
            vec![auto(red), auto(green), auto(blue), auto(red)],
        );
        paint(
            &mut scene,
            conic,
            bounds(64., 64., 32., 16.),
            0.,
            unit,
            None,
        );
        // Off the pixel grid, so no probe sits on a hard edge.
        let pinwheel = Gradient::conic(
            0.,
            center(0.26, 0.47),
            vec![
                at(red, 0.),
                at(red, 0.125),
                at(white, 0.125),
                at(white, 0.25),
            ],
        )
        .repeating(true);
        paint(
            &mut scene,
            pinwheel,
            bounds(96., 64., 32., 16.),
            0.,
            unit,
            None,
        );

        // Rounded corners; rotated 30 degrees and scaled 1.25; in a half-opaque group
        // over a red square (the group draws offscreen: its content overlaps).
        let ramp = || {
            Gradient::linear(
                LinearDirection::Angle(180.),
                vec![auto(0x00ff00ff), auto(0x0000ff80)],
            )
        };
        paint(
            &mut scene,
            ramp(),
            bounds(4., 84., 24., 24.),
            8.,
            unit,
            None,
        );
        paint(
            &mut scene,
            ramp(),
            bounds(44., 88., 16., 16.),
            0.,
            about(52., 96., 30., 1.25),
            None,
        );
        scene.push_group(GroupEffects {
            opacity: 0.5,
            content_mask: full(),
            ..GroupEffects::default()
        });
        scene.insert_primitive(Quad {
            bounds: bounds(80., 84., 24., 24.),
            content_mask: full(),
            background: color(red).into(),
            ..Default::default()
        });
        paint(
            &mut scene,
            ramp(),
            bounds(84., 88., 16., 16.),
            0.,
            unit,
            Some([1., 0., 0., 1.]),
        );
        scene.pop_group();
        scene.finish();

        // Probe a grid of pixel centers inside each gradient, away from its edges.
        let mut probes = Vec::new();
        for drawn in &drawn {
            let inverse = drawn.transformation.inverse().expect("invertible");
            let b = drawn.bounds;
            let (x0, y0) = (b.origin.x.0, b.origin.y.0);
            let (w, h) = (b.size.width.0, b.size.height.0);
            let device = drawn.transformation.transform_bounds(b);
            let mut y = device.origin.y.0.floor() as i32;
            while y < (device.origin.y.0 + device.size.height.0) as i32 {
                let mut x = device.origin.x.0.floor() as i32;
                while x < (device.origin.x.0 + device.size.width.0) as i32 {
                    let center = point(ScaledPixels(x as f32 + 0.5), ScaledPixels(y as f32 + 0.5));
                    let local = inverse.apply_scaled(center);
                    let (lx, ly) = (local.x.0 - x0, local.y.0 - y0);
                    // Two pixels inside the edges and rounded corners, so edge
                    // antialiasing never matters.
                    let r = drawn.radius;
                    let (cx, cy) = (lx.clamp(r, w - r), ly.clamp(r, h - r));
                    let inside = lx > 2.
                        && ly > 2.
                        && lx < w - 2.
                        && ly < h - 2.
                        && (r == 0. || (lx - cx).hypot(ly - cy) < r - 2.);
                    if inside {
                        let color = evaluate(&drawn.texels, (w, h), (lx, ly));
                        let rgb = match drawn.group_under {
                            Some(under) => {
                                let inner = over([under[0], under[1], under[2]], color);
                                over([1.; 3], [inner[0], inner[1], inner[2], 0.5])
                            }
                            None => over([1.; 3], color),
                        };
                        let byte = |c: f32| (c.clamp(0., 1.) * 255.).round() as u8;
                        probes.push(Probe {
                            what: "gradient pixel",
                            x: x as u32,
                            y: y as u32,
                            rgba: [byte(rgb[0]), byte(rgb[1]), byte(rgb[2]), 255],
                        });
                    }
                    x += 3;
                }
                y += 3;
            }
        }
        let probe = |what, x, y, rgba| Probe { what, x, y, rgba };
        probes.extend([
            probe("first hard stop", 8, 8, [255, 0, 0, 255]),
            probe("second hard stop", 24, 8, [0, 255, 0, 255]),
            probe("third hard stop", 40, 8, [0, 0, 255, 255]),
            probe("last hard stop", 56, 8, [255, 255, 255, 255]),
            probe("rounded corner stays white", 5, 85, [255, 255, 255, 255]),
        ]);
        (scene, probes)
    }
}
