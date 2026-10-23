#![cfg_attr(target_family = "wasm", no_main)]

//! Paints text and images with affine transformations next to untransformed references.
//!
//! Red squares mark the point each transformation pivots about. Gray outlines show the
//! untransformed bounds, which are also the bounds used for layout and hit testing.

#[path = "example_support/fonts.rs"]
mod example_support;

use std::{f32::consts::PI, sync::Arc};

use gpui::{
    App, BorderStyle, Bounds, ContentMask, Context, Corners, Hsla, Image, ImageFormat, Path,
    Pixels, Point, ShapedLine, SharedString, TextAlign, TransformationMatrix, Window, WindowBounds,
    WindowOptions, canvas, div, fill, outline, point, prelude::*, px, radians, rgb, rgba, size,
};
use gpui_platform::application;

const IMAGE_SIZE: Pixels = px(96.);

struct TransformedPaintExample {
    image: Arc<Image>,
}

/// Rotates clockwise by `angle` radians and scales by `scale` about `pivot`, given in
/// window coordinates.
fn transformation_about(
    pivot: Point<Pixels>,
    angle: f32,
    scale: f32,
    window: &Window,
) -> TransformationMatrix {
    let pivot = pivot.scale(window.scale_factor());
    TransformationMatrix::unit()
        .translate(pivot)
        .rotate(radians(angle))
        .scale(size(scale, scale))
        .translate(-pivot)
}

fn pivot_marker(pivot: Point<Pixels>, window: &mut Window) {
    window.paint_quad(fill(
        Bounds::centered_at(pivot, size(px(6.), px(6.))),
        rgb(0xe5484d),
    ));
}

fn shape_label(text: &'static str, font_size: Pixels, window: &Window) -> ShapedLine {
    let mut run = window.text_style().to_run(text.len());
    run.color = rgb(0x1f2328).into();
    window
        .text_system()
        .shape_line(SharedString::new_static(text), font_size, &[run], None)
}

fn paint_label(
    line: &ShapedLine,
    origin: Point<Pixels>,
    line_height: Pixels,
    transformation: TransformationMatrix,
    window: &mut Window,
    cx: &mut App,
) {
    line.paint_with_transformation(
        origin,
        line_height,
        TextAlign::Left,
        None,
        transformation,
        window,
        cx,
    )
    .expect("painting a shaped line should succeed");
}

/// Fills `bounds` after applying `transformation`, as a four-point path. Quads cannot be
/// transformed, so this is how carets and selections follow a transformed label.
fn paint_transformed_rect(
    bounds: Bounds<Pixels>,
    transformation: TransformationMatrix,
    color: impl Into<Hsla>,
    window: &mut Window,
) {
    let scale_factor = window.scale_factor();
    let transform = |corner: Point<Pixels>| {
        transformation.apply(corner.scale(scale_factor).map(|value| px(value.0))) / scale_factor
    };
    let mut path = Path::new(transform(bounds.origin));
    path.line_to(transform(bounds.top_right()));
    path.line_to(transform(bounds.bottom_right()));
    path.line_to(transform(bounds.bottom_left()));
    path.line_to(transform(bounds.origin));
    window.paint_path(path, color.into());
}

impl Render for TransformedPaintExample {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let image = self.image.clone().use_render_image(window, cx);

        div().size_full().bg(rgb(0xf6f8fa)).child(
            canvas(
                |_, _, _| {},
                move |bounds, _, window, cx| {
                    let at = |x: f32, y: f32| bounds.origin + point(px(x), px(y));
                    let line_height = px(28.);
                    let label = shape_label("Transformed text", px(20.), window);

                    // Rotation: an untransformed reference, then the line rotated as a unit.
                    let reference = at(40., 30.);
                    paint_label(
                        &label,
                        reference,
                        line_height,
                        TransformationMatrix::unit(),
                        window,
                        cx,
                    );
                    pivot_marker(reference, window);

                    let rotated = at(340., 30.);
                    let rotation = transformation_about(rotated, PI / 6., 1., window);
                    paint_label(&label, rotated, line_height, rotation, window, cx);
                    pivot_marker(rotated, window);

                    let vertical = at(700., 20.);
                    let vertical_line = shape_label("Vertical", px(20.), window);
                    paint_label(
                        &vertical_line,
                        vertical,
                        line_height,
                        transformation_about(vertical, PI / 2., 1., window),
                        window,
                        cx,
                    );
                    pivot_marker(vertical, window);

                    // Scale: a 10px line drawn at 3x next to the same text shaped at 30px.
                    let small = shape_label("Zoomed 3x", px(10.), window);
                    let large = shape_label("Zoomed 3x", px(30.), window);
                    let zoom_reference = at(40., 160.);
                    paint_label(
                        &large,
                        zoom_reference,
                        px(42.),
                        TransformationMatrix::unit(),
                        window,
                        cx,
                    );
                    let zoomed = at(340., 160.);
                    paint_label(
                        &small,
                        zoomed,
                        px(14.),
                        transformation_about(zoomed, 0., 3., window),
                        window,
                        cx,
                    );
                    pivot_marker(zoomed, window);
                    let zoomed_rotated = at(560., 130.);
                    paint_label(
                        &small,
                        zoomed_rotated,
                        px(14.),
                        transformation_about(zoomed_rotated, PI / 8., 2.5, window),
                        window,
                        cx,
                    );
                    pivot_marker(zoomed_rotated, window);

                    // Editing: selection and caret painted as transformed paths behind and
                    // over a rotated label.
                    let edited = at(80., 250.);
                    let editing = transformation_about(edited, -PI / 10., 1.5, window);
                    let selection = Bounds::from_corners(
                        point(edited.x + label.x_for_index(0), edited.y),
                        point(edited.x + label.x_for_index(11), edited.y + line_height),
                    );
                    paint_transformed_rect(selection, editing, rgba(0x0969da40), window);
                    paint_label(&label, edited, line_height, editing, window, cx);
                    let caret = Bounds::new(
                        point(edited.x + label.x_for_index(11), edited.y),
                        size(px(2.), line_height),
                    );
                    paint_transformed_rect(caret, editing, rgb(0x0969da), window);
                    pivot_marker(edited, window);

                    let Some(image) = image else {
                        return;
                    };
                    let corner_radii = Corners::all(px(16.));

                    // Images: an untransformed reference, then rotated about the center.
                    let reference_bounds = Bounds::new(at(40., 330.), size(IMAGE_SIZE, IMAGE_SIZE));
                    window
                        .paint_image(
                            reference_bounds,
                            reference_bounds,
                            corner_radii,
                            image.clone(),
                            0,
                            false,
                        )
                        .expect("painting an image should succeed");
                    window.paint_quad(outline(reference_bounds, rgb(0x8c959f), BorderStyle::Solid));

                    let rotated_bounds = Bounds::new(at(340., 330.), size(IMAGE_SIZE, IMAGE_SIZE));
                    window
                        .paint_image_with_transformation(
                            rotated_bounds,
                            rotated_bounds,
                            corner_radii,
                            image.clone(),
                            0,
                            false,
                            transformation_about(rotated_bounds.center(), PI / 6., 1., window),
                        )
                        .expect("painting an image should succeed");
                    window.paint_quad(outline(rotated_bounds, rgb(0x8c959f), BorderStyle::Solid));
                    pivot_marker(rotated_bounds.center(), window);

                    // Clipping: the content mask applies after the transformation.
                    let mask = Bounds::new(at(40., 490.), size(px(680.), px(60.)));
                    window.paint_quad(outline(mask, rgb(0x0969da), BorderStyle::Dashed));
                    window.with_content_mask(Some(ContentMask { bounds: mask }), |window| {
                        let clipped_bounds =
                            Bounds::new(at(120., 470.), size(IMAGE_SIZE, IMAGE_SIZE));
                        window
                            .paint_image_with_transformation(
                                clipped_bounds,
                                clipped_bounds,
                                corner_radii,
                                image,
                                0,
                                false,
                                transformation_about(clipped_bounds.center(), PI / 4., 1., window),
                            )
                            .expect("painting an image should succeed");

                        let clipped_text = at(320., 530.);
                        let clipped_line =
                            shape_label("Clipped by the content mask", px(20.), window);
                        paint_label(
                            &clipped_line,
                            clipped_text,
                            line_height,
                            transformation_about(clipped_text, -PI / 12., 1., window),
                            window,
                            cx,
                        );
                    });
                },
            )
            .size_full(),
        )
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        if !example_support::load_fonts(cx) {
            return;
        }
        let image = Arc::new(Image::from_bytes(
            ImageFormat::Png,
            include_bytes!("image/app-icon.png").to_vec(),
        ));
        let bounds = Bounds::centered(None, size(px(760.), px(580.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| TransformedPaintExample { image }),
        )
        .expect("opening the window should succeed");
        cx.activate(true);
    });
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    run_example();
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    gpui_platform::web_init();
    run_example();
}
