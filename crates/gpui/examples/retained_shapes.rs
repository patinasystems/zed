#![cfg_attr(target_family = "wasm", no_main)]

//! 100,000 retained, instanced shapes (rectangles and ellipses, fill and stroke,
//! plus 2% bent arrows and 4% text bars)
//! under an animated pan and zoom. The instances upload once; each frame only the
//! camera transformation changes.
//!
//! RETAINED_N=100000 (shapes), RETAINED_MODE=pan|edit|drag|paths, RETAINED_SECONDS=10.
//! `edit` also rewrites one instance per frame (a range upload); `drag` puts every
//! tenth shape on layer 1 and moves it with the layer transformation (no upload);
//! `paths` is the baseline: every shape through `paint_path` each frame.
//! GPUI_RETAINED_SHAPES_STATS=1 makes the Metal renderer report encode CPU, GPU
//! time and upload bytes every 120 frames. RETAINED_STAY=1 keeps the window open.
//! The last line reports frame pacing (`frame_interval_ms`).

#[path = "example_support/fonts.rs"]
mod example_support;

use std::time::Instant;

use gpui::{
    App, Bounds, Context, PathBuilder, Pixels, Render, RetainedShapes, RetainedShapesPaint,
    ShapeInstance, TitlebarOptions, TransformationMatrix, Window, WindowBounds, WindowOptions,
    canvas, div, point, prelude::*, px, rgb, size,
};
use gpui_platform::application;

const PITCH: f32 = 64.0;

static FRAMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Pan,
    Edit,
    Drag,
    Paths,
}

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

const PALETTE: [u32; 6] = [0x569cff, 0xa78bfa, 0x4ac88c, 0xfaaa46, 0xf06464, 0x64c8e6];

fn make(n: usize) -> (Vec<ShapeInstance>, f32, f32) {
    let cols = ((n as f32 * 1.6).sqrt().ceil() as usize).max(1);
    let rows = n.div_ceil(cols);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (col, row) = ((i % cols) as f32, (i / cols) as f32);
        let (w, h) = (24.0 + rng.unit() * 30.0, 20.0 + rng.unit() * 30.0);
        let angle = if rng.unit() < 0.15 {
            (rng.unit() - 0.5) * 1.2
        } else {
            0.0
        };
        let (s, c) = angle.sin_cos();
        let color = PALETTE[i % PALETTE.len()];
        let filled = rng.unit() < 0.25;
        out.push(ShapeInstance {
            transform: [c, s, -s, c, col * PITCH, row * PITCH],
            width: w,
            height: h,
            corner_radii: if i % 7 == 0 { [6.0; 4] } else { [0.0; 4] },
            stroke_width: 2.0,
            fill: if filled { (color << 8) | 90 } else { 0 },
            stroke: (color << 8) | 0xff,
            kind: if rng.unit() < 0.45 {
                ShapeInstance::ELLIPSE
            } else {
                ShapeInstance::RECT
            },
        });
        // 4% carry a label, drawn as a text bar at low zoom.
        if i % 25 == 3 {
            out.push(ShapeInstance {
                transform: [
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    col * PITCH + 4.0,
                    row * PITCH + h * 0.5 - 3.0,
                ],
                width: (w - 8.0).max(4.0),
                height: 6.0,
                fill: 0xd0d4dcff,
                kind: ShapeInstance::TEXT_BAR,
                ..Default::default()
            });
        }
        // 2% are joined to the next cell by a bent arrow: an arc and a head.
        if i % 50 == 25 && (i + 1) % cols != 0 {
            let (x, y) = (col * PITCH + w, row * PITCH + h * 0.5);
            let length = PITCH - w + 4.0;
            out.push(ShapeInstance {
                transform: [1.0, 0.0, 0.0, 1.0, x, y],
                width: length,
                height: -10.0,
                stroke_width: 1.5,
                stroke: 0xe8e8e8ff,
                kind: ShapeInstance::ARC,
                ..Default::default()
            });
            out.push(ShapeInstance {
                transform: [1.0, 0.0, 0.0, 1.0, x + length - 7.0, y - 3.5],
                width: 7.0,
                height: 7.0,
                fill: 0xe8e8e8ff,
                kind: ShapeInstance::TRIANGLE,
                ..Default::default()
            });
        }
    }
    (out, cols as f32 * PITCH, rows as f32 * PITCH)
}

struct RetainedShapesExample {
    mode: Mode,
    shapes: RetainedShapes,
    instances: Vec<ShapeInstance>,
    page: (f32, f32),
    started: Instant,
    seconds: f64,
    frames: u64,
    last_frame: Option<Instant>,
    intervals: Vec<f64>,
    cpu: Vec<f64>,
    reported: bool,
}

impl RetainedShapesExample {
    fn new(n: usize, mode: Mode, seconds: f64) -> Self {
        let (instances, w, h) = make(n);
        let shapes = RetainedShapes::new();
        let mut instances = instances;
        if mode == Mode::Drag {
            // Select every tenth shape: one write when the drag starts.
            for s in instances.iter_mut().step_by(10) {
                s.kind |= ShapeInstance::LAYER_1;
            }
        }
        shapes.set_all(instances.clone());
        Self {
            mode,
            shapes,
            instances,
            page: (w, h),
            started: Instant::now(),
            seconds,
            frames: 0,
            last_frame: None,
            intervals: Vec::new(),
            cpu: Vec::new(),
            reported: false,
        }
    }
}

fn q(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
}

impl Render for RetainedShapesExample {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let frame_started = Instant::now();
        window.request_animation_frame();
        if let Some(last) = self.last_frame.replace(frame_started) {
            if self.frames > 30 {
                self.intervals
                    .push((frame_started - last).as_secs_f64() * 1e3);
            }
        }
        self.frames += 1;
        FRAMES.store(self.frames, std::sync::atomic::Ordering::Relaxed);
        let t = self.started.elapsed().as_secs_f64();
        if t > self.seconds && !self.reported {
            self.reported = true;
            let n = self.instances.len();
            let mode = match self.mode {
                Mode::Pan => "pan",
                Mode::Edit => "edit",
                Mode::Drag => "drag",
                Mode::Paths => "paths",
            };
            println!(
                "RETAINED_EXAMPLE n {n} mode {mode} frames {} frame_interval_ms p50 {:.2} p95 {:.2} cpu_render_to_paint_ms p50 {:.3} p95 {:.3}",
                self.frames,
                q(&mut self.intervals, 0.5),
                q(&mut self.intervals, 0.95),
                q(&mut self.cpu, 0.5),
                q(&mut self.cpu, 0.95),
            );
            if std::env::var("RETAINED_STAY").as_deref() != Ok("1") {
                cx.defer(|cx| cx.quit());
            }
        }
        if self.mode == Mode::Edit {
            // One instance changes per frame: a 52-byte range upload.
            let i = (self.frames as usize * 7919) % self.instances.len();
            let mut one = self.instances[i];
            one.transform[4] += ((self.frames % 20) as f32 - 10.0) * 0.5;
            self.shapes.write(i, &[one]);
        }
        // Camera: a slow zoom between "all visible" and 4x that, plus a pan.
        let (pw, ph) = self.page;
        let wobble = (t * 0.7).sin() as f32;
        let mode = self.mode;
        let shapes = self.shapes.clone();
        let n = self.instances.len() as u32;
        let instances = if mode == Mode::Paths {
            Some(self.instances.clone())
        } else {
            None
        };
        let entity = cx.entity();
        div().size_full().bg(rgb(0x16181d)).child(
            canvas(
                move |_, _, _| {},
                move |bounds: Bounds<Pixels>, _, window, cx| {
                    let (vw, vh) = (f32::from(bounds.size.width), f32::from(bounds.size.height));
                    let fit = (vw / pw).min(vh / ph) * 0.95;
                    let z = fit * (1.0 + 1.5 * (0.5 + 0.5 * wobble));
                    let cx_ = pw * 0.5 + wobble * pw * 0.1;
                    let cy_ = ph * 0.5 + wobble * ph * 0.05;
                    let origin = bounds.origin;
                    let camera = TransformationMatrix {
                        rotation_scale: [[z, 0.0], [0.0, z]],
                        translation: [
                            f32::from(origin.x) + vw * 0.5 - cx_ * z,
                            f32::from(origin.y) + vh * 0.5 - cy_ * z,
                        ],
                    };
                    match mode {
                        Mode::Paths => {
                            for s in instances.iter().flatten() {
                                paint_as_path(window, s, &camera);
                            }
                        }
                        Mode::Drag => {
                            // Every tenth shape is on layer 1 (selected); the drag
                            // offset is one uniform, z-order is untouched.
                            let offset = (t * 3.0).sin() as f32 * 200.0;
                            let layer = TransformationMatrix {
                                rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
                                translation: [offset, 0.0],
                            };
                            window.paint_retained_shapes(
                                bounds,
                                &shapes,
                                RetainedShapesPaint::new(0..n, camera).layer(layer),
                            );
                        }
                        Mode::Pan | Mode::Edit => {
                            window.paint_retained_shapes(
                                bounds,
                                &shapes,
                                RetainedShapesPaint::new(0..n, camera),
                            );
                        }
                    }
                    let cpu = frame_started.elapsed().as_secs_f64() * 1e3;
                    entity.update(cx, |this, _| {
                        if this.frames > 30 {
                            this.cpu.push(cpu);
                        }
                    });
                },
            )
            .size_full(),
        )
    }
}

/// The baseline: tessellate the outline every frame and submit it as a path.
fn paint_as_path(window: &mut Window, s: &ShapeInstance, camera: &TransformationMatrix) {
    let [a, b, c, d, tx, ty] = s.transform;
    let [[m00, m01], [m10, m11]] = camera.rotation_scale;
    let [mtx, mty] = camera.translation;
    let map = |x: f32, y: f32| {
        let (px_, py_) = (a * x + c * y + tx, b * x + d * y + ty);
        point(
            px(m00 * px_ + m01 * py_ + mtx),
            px(m10 * px_ + m11 * py_ + mty),
        )
    };
    let pts: Vec<_> = if s.kind & 0xff == ShapeInstance::ELLIPSE {
        (0..24)
            .map(|i| {
                let t = i as f32 / 24.0 * std::f32::consts::TAU;
                map(
                    s.width * 0.5 * (1.0 + t.cos()),
                    s.height * 0.5 * (1.0 + t.sin()),
                )
            })
            .collect()
    } else {
        vec![
            map(0.0, 0.0),
            map(s.width, 0.0),
            map(s.width, s.height),
            map(0.0, s.height),
        ]
    };
    let color = |c: u32| gpui::Rgba {
        r: ((c >> 24) & 255) as f32 / 255.0,
        g: ((c >> 16) & 255) as f32 / 255.0,
        b: ((c >> 8) & 255) as f32 / 255.0,
        a: (c & 255) as f32 / 255.0,
    };
    for (stroke, col) in [(false, s.fill), (true, s.stroke)] {
        if col & 255 == 0 {
            continue;
        }
        let mut builder = if stroke {
            PathBuilder::stroke(px(s.stroke_width * camera.rotation_scale[0][0]))
        } else {
            PathBuilder::fill()
        };
        builder.move_to(pts[0]);
        for p in &pts[1..] {
            builder.line_to(*p);
        }
        builder.close();
        if let Ok(path) = builder.build() {
            window.paint_path(path, color(col));
        }
    }
}

fn run_example() {
    let n: usize = std::env::var("RETAINED_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);
    let seconds: f64 = std::env::var("RETAINED_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10.0);
    let mode = match std::env::var("RETAINED_MODE").as_deref() {
        Ok("edit") => Mode::Edit,
        Ok("drag") => Mode::Drag,
        Ok("paths") => Mode::Paths,
        _ => Mode::Pan,
    };
    application().run(move |cx: &mut App| {
        if !example_support::load_fonts(cx) {
            return;
        }
        cx.open_window(
            WindowOptions {
                titlebar: Some(TitlebarOptions {
                    title: Some("retained shapes".into()),
                    ..Default::default()
                }),
                focus: false,
                window_bounds: Some(WindowBounds::Windowed(Bounds::centered(
                    None,
                    size(px(1400.0), px(900.0)),
                    cx,
                ))),
                ..Default::default()
            },
            |_, cx| cx.new(|_| RetainedShapesExample::new(n, mode, seconds)),
        )
        .unwrap();
        // A window the system stops drawing (occluded, minimized) renders no
        // frames; the timer still reports and quits.
        if std::env::var("RETAINED_STAY").as_deref() != Ok("1") {
            cx.spawn(async move |cx| {
                cx.background_executor()
                    .timer(std::time::Duration::from_secs_f64(seconds + 3.0))
                    .await;
                println!(
                    "RETAINED_EXAMPLE timeout: {} frames rendered (is the window visible?)",
                    FRAMES.load(std::sync::atomic::Ordering::Relaxed)
                );
                cx.update(|cx| cx.quit());
            })
            .detach();
        }
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
