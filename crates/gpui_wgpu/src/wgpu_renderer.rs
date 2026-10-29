use crate::{CompositorGpuHint, DeviceErrorState, WgpuAtlas, WgpuContext};
use anyhow::{Context as _, Result};
use bytemuck::{Pod, Zeroable};
use collections::FxHashMap;
use gpui::{
    AtlasTextureId, BackdropFilter, Background, Bounds, ContentMask, DevicePixels, GpuSpecs, Path,
    Point, PrimitiveBatch, ScaledPixels, Scene, Size, get_gamma_correction_ratios,
};
use log::warn;
#[cfg(not(target_family = "wasm"))]
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::num::NonZeroU64;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

#[path = "marks.rs"]
mod marks;
#[path = "retained_shapes.rs"]
mod retained_shapes;

const MAX_INSTANCE_BUFFER_SIZE: u64 = 256 * 1024 * 1024;

const INSTANCE_TEXTURE_TEXEL_SIZE: u64 = 16;

/// Shader variant for backends with storage buffer support: the shared shader
/// logic plus the storage-buffer instance transport.
const STORAGE_BUFFER_SHADERS: &str = concat!(
    include_str!("shaders.wgsl"),
    include_str!("shaders_storage.wgsl"),
);

/// Shader variant for WebGL2, which has no storage buffers: the shared shader
/// logic plus the texture-based instance transport.
const WEBGL_SHADERS: &str = concat!(
    include_str!("shaders.wgsl"),
    include_str!("shaders_webgl.wgsl"),
);

/// Subpixel text rendering requires dual-source blending, which WebGL2 lacks, so
/// this variant only ever runs with the storage-buffer transport. The `enable`
/// directive must precede all declarations.
const SUBPIXEL_SHADERS: &str = concat!(
    "enable dual_source_blending;\n",
    include_str!("shaders.wgsl"),
    include_str!("shaders_storage.wgsl"),
    include_str!("shaders_subpixel.wgsl"),
);

fn least_common_multiple(left: u64, right: u64) -> u64 {
    let mut first = left;
    let mut second = right;
    while second != 0 {
        let remainder = first % second;
        first = second;
        second = remainder;
    }
    left / first * right
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalParams {
    viewport_size: [f32; 2],
    premultiplied_alpha: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PodBounds {
    origin: [f32; 2],
    size: [f32; 2],
}

impl From<Bounds<ScaledPixels>> for PodBounds {
    fn from(bounds: Bounds<ScaledPixels>) -> Self {
        Self {
            origin: [bounds.origin.x.0, bounds.origin.y.0],
            size: [bounds.size.width.0, bounds.size.height.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SurfaceParams {
    bounds: PodBounds,
    content_mask: PodBounds,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GammaParams {
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    is_bgr: u32,
    _pad: u32,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct PathSprite {
    bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct PathRasterizationVertex {
    xy_position: Point<ScaledPixels>,
    st_position: Point<f32>,
    color: Background,
    bounds: Bounds<ScaledPixels>,
    /// The path's content mask, for a rounded clip (`bounds` is already clipped to it).
    content_mask: ContentMask<ScaledPixels>,
}

pub struct WgpuSurfaceConfig {
    pub size: Size<DevicePixels>,
    pub transparent: bool,
    /// Preferred presentation mode. When `Some`, the renderer will use this
    /// mode if supported by the surface, falling back to `Fifo`.
    /// When `None`, defaults to `Fifo` (VSync).
    ///
    /// Mobile platforms may prefer `Mailbox` (triple-buffering) to avoid
    /// blocking in `get_current_texture()` during lifecycle transitions.
    pub preferred_present_mode: Option<wgpu::PresentMode>,
}

struct WgpuPipelines {
    quads: wgpu::RenderPipeline,
    shadows: wgpu::RenderPipeline,
    path_rasterization: wgpu::RenderPipeline,
    paths: wgpu::RenderPipeline,
    underlines: wgpu::RenderPipeline,
    mono_sprites: wgpu::RenderPipeline,
    subpixel_sprites: Option<wgpu::RenderPipeline>,
    poly_sprites: wgpu::RenderPipeline,
    #[allow(dead_code)]
    surfaces: wgpu::RenderPipeline,
    retained_shapes: wgpu::RenderPipeline,
    backdrop: BackdropPipelines,
}

/// Backdrop filter passes (`shaders_backdrop.wgsl`).
struct BackdropPipelines {
    /// Copies a texture to a render target (downsampling, and the final frame copy).
    copy: wgpu::RenderPipeline,
    blur: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    /// Composite a group's image over the target, adding to it (plus-lighter) and
    /// replacing it (plus-darker).
    group_over: wgpu::RenderPipeline,
    group_add: wgpu::RenderPipeline,
    group_replace: wgpu::RenderPipeline,
    /// Draws a group's backdrop into its (transparent) image.
    backdrop_group: wgpu::RenderPipeline,
    /// Group 1: the pass's uniforms.
    uniform_layout: wgpu::BindGroupLayout,
}

impl BackdropPipelines {
    fn group_composite(&self, blend_mode: gpui::BlendMode) -> &wgpu::RenderPipeline {
        match blend_mode {
            gpui::BlendMode::PlusLighter => &self.group_add,
            gpui::BlendMode::PlusDarker
            | gpui::BlendMode::ColorDodge
            | gpui::BlendMode::ColorBurn => &self.group_replace,
            _ => &self.group_over,
        }
    }
}

/// A premultiplied group image over the target, or added to it.
const fn group_blend(destination: wgpu::BlendFactor) -> wgpu::BlendState {
    let component = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: destination,
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState {
        color: component,
        alpha: component,
    }
}

/// The filtered color (premultiplied by coverage) over the target, keeping the
/// target's alpha.
const BACKDROP_COMPOSITE_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

/// `BlurParams` in shaders_backdrop.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BackdropBlurParams {
    direction: [f32; 2],
    sigma: f32,
    radius: i32,
}

/// `CompositeParams` in shaders_backdrop.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BackdropCompositeParams {
    bounds: [f32; 4],
    content_mask: [f32; 4],
    corner_radii: [f32; 4],
    region: [f32; 4],
    viewport_size: [f32; 2],
    saturate: f32,
    brightness: f32,
    contrast: f32,
    opacity: f32,
    pad: [f32; 2],
}

/// The scenes a frame draws (the window's first, then its groups', depth first) with
/// their instance data.
struct FrameScenes<'a> {
    scenes: SmallVec<[FrameScene<'a>; 4]>,
    /// The retained shape uniform slot stride.
    retained_stride: u64,
    /// The window scene's gradient records, which quads in every scene refer to
    /// (`@group(2)` of the quad pipeline).
    gradients: InstanceBinding,
}

struct FrameScene<'a> {
    scene: &'a Scene,
    instance_bindings: InstanceBindings,
    /// The uniform slot of its first retained shape draw.
    retained_base: usize,
    /// The frame's index of its first marks draw.
    marks_base: usize,
}

impl<'a> FrameScenes<'a> {
    fn get(&self, scene: &Scene) -> Option<&FrameScene<'a>> {
        self.scenes
            .iter()
            .find(|drawn| std::ptr::eq(drawn.scene, scene))
    }
}

/// Where a scene draws: a view, its texture when copies may read it (the offscreen
/// frame and group images; not a surface), and the size drawn, which may be less
/// than the texture's (a group's image).
#[derive(Clone, Copy)]
struct DrawTarget<'a> {
    view: &'a wgpu::TextureView,
    texture: Option<&'a wgpu::Texture>,
    size: Size<DevicePixels>,
}

/// A pooled texture and its view.
#[derive(Clone)]
struct Scratch {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// Group images are allocated in steps of this many device pixels per side, so a
/// group that changes size a little reuses its image.
const GROUP_IMAGE_STEP: u32 = 64;

fn bucket(size: u32) -> u32 {
    size.max(1).next_multiple_of(GROUP_IMAGE_STEP)
}

/// Offscreen textures (group images, blur and backdrop copies) by size, reused
/// within and across frames. A texture released during a frame is reused later in
/// it: wgpu orders the later pass after the earlier ones. Textures a frame did not
/// use are freed at its end.
#[derive(Default)]
struct TexturePool {
    entries: RefCell<Vec<PooledTexture>>,
}

struct PooledTexture {
    scratch: Scratch,
    in_use: bool,
    used: bool,
}

impl TexturePool {
    fn acquire(
        &self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Scratch {
        let (width, height) = (width.max(1), height.max(1));
        let mut entries = self.entries.borrow_mut();
        if let Some(entry) = entries.iter_mut().find(|entry| {
            !entry.in_use
                && entry.scratch.texture.width() == width
                && entry.scratch.texture.height() == height
        }) {
            entry.in_use = true;
            entry.used = true;
            return entry.scratch.clone();
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let scratch = Scratch { texture, view };
        entries.push(PooledTexture {
            scratch: scratch.clone(),
            in_use: true,
            used: true,
        });
        scratch
    }

    fn release(&self, texture: &wgpu::Texture) {
        if let Some(entry) = self
            .entries
            .borrow_mut()
            .iter_mut()
            .find(|entry| entry.scratch.texture == *texture)
        {
            entry.in_use = false;
        }
    }

    fn end_frame(&self) {
        let mut entries = self.entries.borrow_mut();
        entries.retain(|entry| entry.used);
        for entry in entries.iter_mut() {
            entry.used = false;
            entry.in_use = false;
        }
    }

    fn bytes(&self) -> u64 {
        self.entries
            .borrow()
            .iter()
            .map(|entry| {
                let texture = &entry.scratch.texture;
                u64::from(texture.width()) * u64::from(texture.height()) * 4
            })
            .sum()
    }
}

/// Globals for a group image's size ([`WgpuRendererCore::group_globals`]).
struct GroupGlobals {
    buffer: wgpu::Buffer,
    globals: wgpu::BindGroup,
    path_globals: wgpu::BindGroup,
    /// The frame that last used (and wrote) it; frames count from 1.
    last_used: u64,
}

/// One frame allocation of instance data, ready to bind.
struct InstanceBinding {
    bind_group: wgpu::BindGroup,
    /// Index of the allocation's first instance within the bound data. Always
    /// zero on the storage-buffer path, where the binding offset already
    /// positions the array; on the WebGL texture path the shader indexes the
    /// shared instance texture absolutely, so draws must offset their
    /// instance (or vertex) ranges by this value.
    first_instance: u32,
}

struct InstanceBindings {
    quads: InstanceBinding,
    shadows: InstanceBinding,
    underlines: InstanceBinding,
    monochrome_sprites: InstanceBinding,
    subpixel_sprites: InstanceBinding,
    polychrome_sprites: InstanceBinding,
}

struct WgpuBindGroupLayouts {
    globals: wgpu::BindGroupLayout,
    instances: wgpu::BindGroupLayout,
    texture: wgpu::BindGroupLayout,
    surfaces: wgpu::BindGroupLayout,
    /// Per-draw retained shape uniforms, at a dynamic offset.
    retained_uniforms: wgpu::BindGroupLayout,
}

/// Shared GPU context reference, used to coordinate device recovery across multiple windows.
pub type GpuContext = Rc<RefCell<Option<WgpuContext>>>;

enum InstanceData {
    Storage(wgpu::Buffer),
    // WebGL2 has no storage buffers. A uint texture keeps the records available to both shader
    // stages while preserving integer and floating-point bit patterns exactly.
    Texture {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        width: u32,
        height: u32,
    },
}

/// GPU resources that must be dropped together during device recovery.
struct WgpuResources {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: WgpuPipelines,
    bind_group_layouts: WgpuBindGroupLayouts,
    atlas_sampler: wgpu::Sampler,
    /// For external textures drawn pixelated ([`gpui::TextureSampling::Nearest`]).
    nearest_sampler: wgpu::Sampler,
    atlas_texture_bind_groups: FxHashMap<AtlasTextureId, CachedTextureBindGroup>,
    globals_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    path_globals_bind_group: wgpu::BindGroup,
    instance_data: InstanceData,
    path_intermediate_texture: Option<wgpu::Texture>,
    path_intermediate_view: Option<wgpu::TextureView>,
    path_msaa_texture: Option<wgpu::Texture>,
    path_msaa_view: Option<wgpu::TextureView>,
    retained: retained_shapes::RetainedState,
    marks: marks::MarksState,
    /// Group images and their blur and backdrop copies, reused across frames and
    /// freed when a frame does not use them.
    texture_pool: TexturePool,
    /// Globals for group images, by image size (their viewport differs from the
    /// frame's), freed when a frame does not use them.
    group_globals: FxHashMap<(i32, i32), GroupGlobals>,
    /// Frames drawn, for the group globals' lifetime.
    group_frame: u64,
}

struct CachedTextureBindGroup {
    texture_generation: u64,
    bind_group: wgpu::BindGroup,
}

impl WgpuResources {
    fn invalidate_intermediate_textures(&mut self) {
        self.path_intermediate_texture = None;
        self.path_intermediate_view = None;
        self.path_msaa_texture = None;
        self.path_msaa_view = None;
    }
}

struct WgpuRendererCore {
    resources: WgpuResources,
    /// Pipelines of custom shaders (`gpui::Shader`).
    custom_shaders: crate::custom_shaders::CustomShaders,
    atlas: Arc<WgpuAtlas>,
    path_globals_offset: u64,
    gamma_offset: u64,
    instance_data_capacity: u64,
    max_instance_data_size: u64,
    instance_data_alignment: u64,
    uses_webgl_instance_data: bool,
    rendering_params: RenderingParameters,
    is_bgr: bool,
    dual_source_blending: bool,
    adapter_info: wgpu::AdapterInfo,
    target_format: wgpu::TextureFormat,
    max_texture_size: u32,
    /// The frame's `GlobalParams::premultiplied_alpha`, for group globals.
    premultiplied_alpha: bool,
}

/// GPU resources of a windowed renderer. A surface is only ever configured against the
/// device that owns `core`, so it cannot outlive it: there is no surface-only state.
enum RendererState {
    /// Frames can be drawn.
    Ready {
        surface: wgpu::Surface<'static>,
        core: WgpuRendererCore,
    },
    /// The native surface is gone (Android `TerminateWindow`, browser context loss) but
    /// the device, pipelines, and atlas remain so `replace_surface` can resume without
    /// re-uploading cached textures.
    Unconfigured { core: WgpuRendererCore },
    /// Released by `destroy`, or dropped for a device recovery that has not succeeded yet.
    Released,
}

pub struct WgpuRenderer {
    /// Shared GPU context for device recovery coordination (unused on WASM).
    #[allow(dead_code)]
    context: Option<GpuContext>,
    /// Compositor GPU hint for adapter selection (unused on WASM).
    #[allow(dead_code)]
    compositor_gpu: Option<CompositorGpuHint>,
    state: RendererState,
    surface_config: wgpu::SurfaceConfiguration,
    atlas: Arc<WgpuAtlas>,
    transparent_alpha_mode: wgpu::CompositeAlphaMode,
    opaque_alpha_mode: wgpu::CompositeAlphaMode,
    max_texture_size: u32,
    is_bgr: bool,
    failed_frame_count: u32,
    device_errors: Arc<DeviceErrorState>,
    observed_error_generation: u64,
    last_surface_error: Option<String>,
    needs_redraw: bool,
}

impl WgpuRenderer {
    fn core(&self) -> Option<&WgpuRendererCore> {
        match &self.state {
            RendererState::Ready { core, .. } | RendererState::Unconfigured { core } => Some(core),
            RendererState::Released => None,
        }
    }

    fn core_mut(&mut self) -> Option<&mut WgpuRendererCore> {
        match &mut self.state {
            RendererState::Ready { core, .. } | RendererState::Unconfigured { core } => Some(core),
            RendererState::Released => None,
        }
    }

    /// Creates a new WgpuRenderer from raw window handles.
    ///
    /// The `gpu_context` is a shared reference that coordinates GPU context across
    /// multiple windows. The first window to create a renderer will initialize the
    /// context; subsequent windows will share it.
    ///
    /// # Safety
    /// The caller must ensure that the window handle remains valid for the lifetime
    /// of the returned renderer.
    #[cfg(not(target_family = "wasm"))]
    pub fn new<W>(
        gpu_context: GpuContext,
        window: &W,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
    ) -> anyhow::Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
    {
        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;

        let target = wgpu::SurfaceTargetUnsafe::RawHandle {
            // Fall back to the display handle already provided via InstanceDescriptor::display.
            raw_display_handle: None,
            raw_window_handle: window_handle.as_raw(),
        };

        // Use the existing context's instance if available, otherwise create a new one.
        // The surface must be created with the same instance that will be used for
        // adapter selection, otherwise wgpu will panic.
        let instance = gpu_context
            .borrow()
            .as_ref()
            .map(|ctx| ctx.instance.clone())
            .unwrap_or_else(|| WgpuContext::instance(Some(Box::new(window.clone()))));

        // Safety: The caller guarantees that the window handle is valid for the
        // lifetime of this renderer. In practice, the RawWindow struct is created
        // from the native window handles and the surface is dropped before the window.
        let surface = unsafe {
            instance
                .create_surface_unsafe(target)
                .map_err(|e| anyhow::anyhow!("Failed to create surface: {e}"))?
        };

        let mut ctx_ref = gpu_context.borrow_mut();
        let context = match ctx_ref.as_mut() {
            Some(context) => {
                context.check_compatible_with_surface(&surface)?;
                context
            }
            None => ctx_ref.insert(WgpuContext::new(instance, &surface, compositor_gpu)?),
        };

        let atlas = Arc::new(WgpuAtlas::from_context(context));

        Self::new_internal(
            Some(Rc::clone(&gpu_context)),
            context,
            surface,
            config,
            compositor_gpu,
            atlas,
        )
    }

    #[cfg(target_family = "wasm")]
    pub fn new_from_canvas(
        context: &WgpuContext,
        canvas: &web_sys::HtmlCanvasElement,
        config: WgpuSurfaceConfig,
    ) -> anyhow::Result<Self> {
        let surface = context
            .instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| anyhow::anyhow!("Failed to create surface: {e}"))?;
        Self::new_from_surface(context, surface, config)
    }

    #[cfg(target_family = "wasm")]
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new_from_surface(
        context: &WgpuContext,
        surface: wgpu::Surface<'static>,
        config: WgpuSurfaceConfig,
    ) -> anyhow::Result<Self> {
        let atlas = Arc::new(WgpuAtlas::from_context(context));
        Self::new_internal(None, context, surface, config, None, atlas)
    }

    fn new_internal(
        gpu_context: Option<GpuContext>,
        context: &WgpuContext,
        surface: wgpu::Surface<'static>,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
        atlas: Arc<WgpuAtlas>,
    ) -> anyhow::Result<Self> {
        let surface_caps = surface.get_capabilities(&context.adapter);
        let preferred_formats = [
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ];
        let surface_format = preferred_formats
            .iter()
            .find(|f| surface_caps.formats.contains(f))
            .copied()
            .or_else(|| {
                surface_caps
                    .formats
                    .iter()
                    .find(|format| !format.has_srgb_suffix())
                    .copied()
            })
            .or_else(|| surface_caps.formats.first().copied())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Surface reports no supported texture formats for adapter {:?}",
                    context.adapter.get_info().name
                )
            })?;

        let pick_alpha_mode =
            |preferences: &[wgpu::CompositeAlphaMode]| -> anyhow::Result<wgpu::CompositeAlphaMode> {
                preferences
                    .iter()
                    .find(|p| surface_caps.alpha_modes.contains(p))
                    .copied()
                    .or_else(|| surface_caps.alpha_modes.first().copied())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Surface reports no supported alpha modes for adapter {:?}",
                            context.adapter.get_info().name
                        )
                    })
            };

        let transparent_alpha_mode = pick_alpha_mode(&[
            wgpu::CompositeAlphaMode::PreMultiplied,
            wgpu::CompositeAlphaMode::Inherit,
        ])?;

        let opaque_alpha_mode = pick_alpha_mode(&[
            wgpu::CompositeAlphaMode::Opaque,
            wgpu::CompositeAlphaMode::Inherit,
        ])?;

        let alpha_mode = if config.transparent {
            transparent_alpha_mode
        } else {
            opaque_alpha_mode
        };

        let device = Arc::clone(&context.device);
        let max_texture_size = device.limits().max_texture_dimension_2d;

        let requested_width = config.size.width.0 as u32;
        let requested_height = config.size.height.0 as u32;
        let clamped_width = requested_width.min(max_texture_size);
        let clamped_height = requested_height.min(max_texture_size);

        if clamped_width != requested_width || clamped_height != requested_height {
            warn!(
                "Requested surface size ({}, {}) exceeds maximum texture dimension {}. \
                 Clamping to ({}, {}). Window content may not fill the entire window.",
                requested_width, requested_height, max_texture_size, clamped_width, clamped_height
            );
        }

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: clamped_width.max(1),
            height: clamped_height.max(1),
            present_mode: config
                .preferred_present_mode
                .filter(|mode| surface_caps.present_modes.contains(mode))
                .unwrap_or(wgpu::PresentMode::Fifo),
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![],
        };
        // Configure the surface immediately. The adapter selection process already validated
        // that this adapter can successfully configure this surface.
        surface.configure(&context.device, &surface_config);

        let core = WgpuRendererCore::new(context, atlas.clone(), surface_format, alpha_mode);

        Ok(Self {
            context: gpu_context,
            compositor_gpu,
            state: RendererState::Ready { surface, core },
            surface_config,
            atlas,
            transparent_alpha_mode,
            opaque_alpha_mode,
            max_texture_size,
            is_bgr: false,
            failed_frame_count: 0,
            device_errors: Arc::clone(context.errors()),
            observed_error_generation: 0,
            last_surface_error: None,
            needs_redraw: false,
        })
    }
}

impl WgpuRendererCore {
    fn create_bind_group_layouts(
        device: &wgpu::Device,
        uses_webgl_instance_data: bool,
    ) -> WgpuBindGroupLayouts {
        let globals =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("globals_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<GlobalParams>() as u64
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<GammaParams>() as u64
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let instance_data_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: if uses_webgl_instance_data {
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }
            } else {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            },
            count: None,
        };

        let instances = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instances_layout"),
            entries: &[instance_data_entry],
        });

        let texture = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let surfaces = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surfaces_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<SurfaceParams>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let retained_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("retained_shapes_uniforms_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<
                        gpui::RetainedShapesUniforms,
                    >() as u64),
                },
                count: None,
            }],
        });

        WgpuBindGroupLayouts {
            globals,
            instances,
            texture,
            surfaces,
            retained_uniforms,
        }
    }

    fn create_instance_texture(
        device: &wgpu::Device,
        requested_capacity: u64,
        max_texture_dimension: u32,
    ) -> (InstanceData, u64) {
        let texel_count = requested_capacity.div_ceil(INSTANCE_TEXTURE_TEXEL_SIZE);
        let width = texel_count.min(u64::from(max_texture_dimension)).max(1) as u32;
        let height = texel_count
            .div_ceil(u64::from(width))
            .min(u64::from(max_texture_dimension))
            .max(1) as u32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("instance_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let capacity = u64::from(width) * u64::from(height) * INSTANCE_TEXTURE_TEXEL_SIZE;
        (
            InstanceData::Texture {
                texture,
                view,
                width,
                height,
            },
            capacity,
        )
    }

    fn create_pipelines(
        device: &wgpu::Device,
        layouts: &WgpuBindGroupLayouts,
        surface_format: wgpu::TextureFormat,
        alpha_mode: wgpu::CompositeAlphaMode,
        path_sample_count: u32,
        dual_source_blending: bool,
        uses_webgl_instance_data: bool,
    ) -> WgpuPipelines {
        // Diagnostic guard: verify the device actually has
        // DUAL_SOURCE_BLENDING. We have a crash report (ZED-5G1) where a
        // feature mismatch caused a wgpu-hal abort, but we haven't
        // identified the code path that produces the mismatch. This
        // guard prevents the crash and logs more evidence.
        // Remove this check once:
        // a) We find and fix the root cause, or
        // b) There are no reports of this warning appearing for some time.
        let device_has_feature = device
            .features()
            .contains(wgpu::Features::DUAL_SOURCE_BLENDING);
        if dual_source_blending && !device_has_feature {
            log::error!(
                "BUG: dual_source_blending flag is true but device does not \
                 have DUAL_SOURCE_BLENDING enabled (device features: {:?}). \
                 Falling back to mono text rendering. Please report this at \
                 https://github.com/zed-industries/zed/issues",
                device.features(),
            );
        }
        let dual_source_blending =
            dual_source_blending && device_has_feature && !uses_webgl_instance_data;

        let shader_source = if uses_webgl_instance_data {
            WEBGL_SHADERS
        } else {
            STORAGE_BUFFER_SHADERS
        };
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpui_shaders"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let subpixel_shader_module = if dual_source_blending {
            Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpui_subpixel_shaders"),
                source: wgpu::ShaderSource::Wgsl(SUBPIXEL_SHADERS.into()),
            }))
        } else {
            None
        };

        let blend_mode = match alpha_mode {
            wgpu::CompositeAlphaMode::PreMultiplied => {
                wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
            }
            _ => wgpu::BlendState::ALPHA_BLENDING,
        };

        let color_target = wgpu::ColorTargetState {
            format: surface_format,
            blend: Some(blend_mode),
            write_mask: wgpu::ColorWrites::ALL,
        };

        let create_pipeline = |name: &str,
                               vs_entry: &str,
                               fs_entry: &str,
                               globals_layout: &wgpu::BindGroupLayout,
                               data_layout: &wgpu::BindGroupLayout,
                               texture_layout: Option<&wgpu::BindGroupLayout>,
                               topology: wgpu::PrimitiveTopology,
                               color_targets: &[Option<wgpu::ColorTargetState>],
                               sample_count: u32,
                               module: &wgpu::ShaderModule| {
            let mut bind_group_layouts = vec![Some(globals_layout), Some(data_layout)];
            bind_group_layouts.extend(texture_layout.map(Some));
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{name}_layout")),
                bind_group_layouts: &bind_group_layouts,
                immediate_size: 0,
            });

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(vs_entry),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(fs_entry),
                    targets: color_targets,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };

        // `@group(2)` holds the frame's gradient records, bound like instance data.
        let quads = create_pipeline(
            "quads",
            "vs_quad",
            "fs_quad",
            &layouts.globals,
            &layouts.instances,
            Some(&layouts.instances),
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let shadows = create_pipeline(
            "shadows",
            "vs_shadow",
            "fs_shadow",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let path_rasterization = create_pipeline(
            "path_rasterization",
            "vs_path_rasterization",
            "fs_path_rasterization",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleList,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            path_sample_count,
            &shader_module,
        );

        let paths = create_pipeline(
            "paths",
            "vs_path",
            "fs_path",
            &layouts.globals,
            &layouts.instances,
            Some(&layouts.texture),
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            1,
            &shader_module,
        );

        let underlines = create_pipeline(
            "underlines",
            "vs_underline",
            "fs_underline",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let mono_sprites = create_pipeline(
            "mono_sprites",
            "vs_mono_sprite",
            "fs_mono_sprite",
            &layouts.globals,
            &layouts.instances,
            Some(&layouts.texture),
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let subpixel_sprites = if let Some(subpixel_module) = &subpixel_shader_module {
            let subpixel_blend = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Src1,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc1,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            };

            Some(create_pipeline(
                "subpixel_sprites",
                "vs_subpixel_sprite",
                "fs_subpixel_sprite",
                &layouts.globals,
                &layouts.instances,
                Some(&layouts.texture),
                wgpu::PrimitiveTopology::TriangleStrip,
                &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(subpixel_blend),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
                1,
                subpixel_module,
            ))
        } else {
            None
        };

        let poly_sprites = create_pipeline(
            "poly_sprites",
            "vs_poly_sprite",
            "fs_poly_sprite",
            &layouts.globals,
            &layouts.instances,
            Some(&layouts.texture),
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let surfaces = create_pipeline(
            "surfaces",
            "vs_surface",
            "fs_surface",
            &layouts.globals,
            &layouts.surfaces,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let retained_shapes = create_pipeline(
            "retained_shapes",
            "vs_retained_shape",
            "fs_retained_shape",
            &layouts.globals,
            &layouts.instances,
            Some(&layouts.retained_uniforms),
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target)],
            1,
            &shader_module,
        );

        let backdrop = Self::create_backdrop_pipelines(device, &layouts.texture, surface_format);

        WgpuPipelines {
            backdrop,
            retained_shapes,
            quads,
            shadows,
            path_rasterization,
            paths,
            underlines,
            mono_sprites,
            subpixel_sprites,
            poly_sprites,
            surfaces,
        }
    }

    fn create_backdrop_pipelines(
        device: &wgpu::Device,
        texture_layout: &wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
    ) -> BackdropPipelines {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpui_backdrop_shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders_backdrop.wgsl").into()),
        });
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("backdrop_uniform_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline = |name: &str,
                        vs: &str,
                        fs: &str,
                        with_uniforms: bool,
                        blend: Option<wgpu::BlendState>,
                        topology| {
            let mut layouts = vec![Some(texture_layout)];
            if with_uniforms {
                layouts.push(Some(&uniform_layout));
            }
            // Group composites also sample what is behind the group, and its mask.
            if vs == "vs_group_composite" {
                layouts.push(Some(texture_layout));
                layouts.push(Some(texture_layout));
            }
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{name}_layout")),
                bind_group_layouts: &layouts,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(name),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vs),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        BackdropPipelines {
            copy: pipeline(
                "backdrop_copy",
                "vs_backdrop_fullscreen",
                "fs_backdrop_copy",
                false,
                None,
                wgpu::PrimitiveTopology::TriangleList,
            ),
            blur: pipeline(
                "backdrop_blur",
                "vs_backdrop_fullscreen",
                "fs_backdrop_blur",
                true,
                None,
                wgpu::PrimitiveTopology::TriangleList,
            ),
            composite: pipeline(
                "backdrop_composite",
                "vs_backdrop_composite",
                "fs_backdrop_composite",
                true,
                Some(BACKDROP_COMPOSITE_BLEND),
                wgpu::PrimitiveTopology::TriangleStrip,
            ),
            backdrop_group: pipeline(
                "backdrop_group",
                "vs_backdrop_composite",
                "fs_backdrop_group",
                true,
                Some(group_blend(wgpu::BlendFactor::OneMinusSrcAlpha)),
                wgpu::PrimitiveTopology::TriangleStrip,
            ),
            group_over: pipeline(
                "group_composite_over",
                "vs_group_composite",
                "fs_group_composite",
                true,
                Some(group_blend(wgpu::BlendFactor::OneMinusSrcAlpha)),
                wgpu::PrimitiveTopology::TriangleStrip,
            ),
            group_add: pipeline(
                "group_composite_add",
                "vs_group_composite",
                "fs_group_composite",
                true,
                Some(group_blend(wgpu::BlendFactor::One)),
                wgpu::PrimitiveTopology::TriangleStrip,
            ),
            group_replace: pipeline(
                "group_composite_replace",
                "vs_group_composite",
                "fs_group_composite",
                true,
                None,
                wgpu::PrimitiveTopology::TriangleStrip,
            ),
            uniform_layout,
        }
    }

    fn create_path_intermediate(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_intermediate"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_msaa_if_needed(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Option<(wgpu::Texture, wgpu::TextureView)> {
        if sample_count <= 1 {
            return None;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_msaa"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Some((texture, view))
    }
}

impl WgpuRenderer {
    /// Records the new drawable size and reconfigures the surface when one is present.
    /// The size is kept even without GPU resources so that recovery restores it.
    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        let width = size.width.0 as u32;
        let height = size.height.0 as u32;

        if width == self.surface_config.width && height == self.surface_config.height {
            return;
        }

        let clamped_width = width.min(self.max_texture_size);
        let clamped_height = height.min(self.max_texture_size);
        if clamped_width != width || clamped_height != height {
            warn!(
                "Requested surface size ({}, {}) exceeds maximum texture dimension {}. \
                 Clamping to ({}, {}). Window content may not fill the entire window.",
                width, height, self.max_texture_size, clamped_width, clamped_height
            );
        }
        self.surface_config.width = clamped_width.max(1);
        self.surface_config.height = clamped_height.max(1);

        let Some(core) = self.core_mut() else {
            return;
        };
        let resources = &mut core.resources;

        // Wait for any in-flight GPU work to complete before destroying textures
        if let Err(e) = resources.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        }) {
            warn!("Failed to poll device during resize: {e:?}");
        }

        // Destroy old textures before allocating new ones to avoid GPU memory spikes
        if let Some(ref texture) = resources.path_intermediate_texture {
            texture.destroy();
        }
        if let Some(ref texture) = resources.path_msaa_texture {
            texture.destroy();
        }

        // Invalidate intermediate textures - they will be lazily recreated
        // in draw() after we confirm the surface is healthy. This avoids
        // panics when the device/surface is in an invalid state during resize.
        resources.invalidate_intermediate_textures();

        if let RendererState::Ready { surface, core } = &self.state {
            surface.configure(&core.resources.device, &self.surface_config);
        }
    }

    pub fn set_subpixel_layout(&mut self, is_bgr: bool) {
        self.is_bgr = is_bgr;
        if let Some(core) = self.core_mut() {
            core.is_bgr = is_bgr;
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        let new_alpha_mode = if transparent {
            self.transparent_alpha_mode
        } else {
            self.opaque_alpha_mode
        };
        if new_alpha_mode == self.surface_config.alpha_mode {
            return;
        }
        self.surface_config.alpha_mode = new_alpha_mode;
        let format = self.surface_config.format;

        let Some(core) = self.core_mut() else {
            return;
        };
        let resources = &mut core.resources;
        resources.pipelines = WgpuRendererCore::create_pipelines(
            &resources.device,
            &resources.bind_group_layouts,
            format,
            new_alpha_mode,
            core.rendering_params.path_sample_count,
            core.dual_source_blending,
            core.uses_webgl_instance_data,
        );

        if let RendererState::Ready { surface, core } = &self.state {
            surface.configure(&core.resources.device, &self.surface_config);
        }
    }

    #[allow(dead_code)]
    pub fn viewport_size(&self) -> Size<DevicePixels> {
        Size {
            width: DevicePixels(self.surface_config.width as i32),
            height: DevicePixels(self.surface_config.height as i32),
        }
    }

    pub fn sprite_atlas(&self) -> &Arc<WgpuAtlas> {
        &self.atlas
    }

    pub fn supports_dual_source_blending(&self) -> bool {
        self.core().is_some_and(|core| core.dual_source_blending)
    }

    /// Returns `None` once GPU resources have been released by `destroy` or a pending
    /// device recovery.
    pub fn gpu_specs(&self) -> Option<GpuSpecs> {
        let adapter_info = &self.core()?.adapter_info;
        Some(GpuSpecs {
            is_software_emulated: adapter_info.device_type == wgpu::DeviceType::Cpu,
            device_name: adapter_info.name.clone(),
            driver_name: adapter_info.driver.clone(),
            driver_info: adapter_info.driver_info.clone(),
        })
    }

    pub fn max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    pub fn draw(&mut self, scene: &Scene) -> bool {
        #[cfg(target_family = "wasm")]
        if self.device_lost() {
            if matches!(self.state, RendererState::Ready { .. }) {
                log::error!(
                    "Browser graphics context was lost; rendering has stopped. Reload the page to recover."
                );
                self.unconfigure_surface();
            }
            return false;
        }

        // Bail out early if the surface has been unconfigured (e.g. during
        // Android background/rotation transitions).  Attempting to acquire
        // a texture from an unconfigured surface can block indefinitely on
        // some drivers (Adreno).
        let RendererState::Ready { surface, core } = &mut self.state else {
            return false;
        };

        if let Some(error) = self.last_surface_error.take().or_else(|| {
            self.device_errors
                .observe_error(&mut self.observed_error_generation)
        }) {
            self.failed_frame_count += 1;
            log::error!(
                "GPU error during frame (failure {} of 10): {error}",
                self.failed_frame_count
            );

            // TBD. Does retrying more actually help?
            if self.failed_frame_count > 10 {
                panic!("Too many consecutive GPU errors. Last error: {error}");
            } else if self.failed_frame_count > 5 {
                core.resources.invalidate_intermediate_textures();
                self.atlas.clear();
                self.needs_redraw = true;
                self.failed_frame_count = 0;
                return false;
            }
        } else {
            self.failed_frame_count = 0;
        }

        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                // Textures must be destroyed before the surface can be reconfigured.
                drop(frame);
                surface.configure(&core.resources.device, &self.surface_config);
                return false;
            }
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                surface.configure(&core.resources.device, &self.surface_config);
                return false;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return false;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                self.last_surface_error = Some("Surface texture validation error".to_string());
                return false;
            }
        };

        // The acquired texture is the authority on frame dimensions; the surface
        // configuration is only a request.
        let size = Size {
            width: DevicePixels(frame.texture.width() as i32),
            height: DevicePixels(frame.texture.height() as i32),
        };
        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let premultiplied_alpha =
            self.surface_config.alpha_mode == wgpu::CompositeAlphaMode::PreMultiplied;
        if let Err(error) = core.render_frame(
            scene,
            &frame_view,
            size,
            premultiplied_alpha,
            wgpu::Color::TRANSPARENT,
        ) {
            log::error!("{error:#}");
            return false;
        }

        core.resources.queue.present(frame);
        true
    }
}

impl WgpuRendererCore {
    fn new(
        context: &WgpuContext,
        atlas: Arc<WgpuAtlas>,
        target_format: wgpu::TextureFormat,
        alpha_mode: wgpu::CompositeAlphaMode,
    ) -> Self {
        let device = Arc::clone(&context.device);
        let queue = Arc::clone(&context.queue);
        // External textures are drawn into with this renderer's pipelines (texture
        // targets), so they take its format.
        atlas.set_external_format(target_format);
        let rendering_params = RenderingParameters::new(&context.adapter, target_format);
        let uses_webgl_instance_data = context.uses_webgl_instance_data();
        let dual_source_blending =
            context.supports_dual_source_blending() && !uses_webgl_instance_data;
        let bind_group_layouts = Self::create_bind_group_layouts(&device, uses_webgl_instance_data);
        let pipelines = Self::create_pipelines(
            &device,
            &bind_group_layouts,
            target_format,
            alpha_mode,
            rendering_params.path_sample_count,
            dual_source_blending,
            uses_webgl_instance_data,
        );
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nearest_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let globals_size = std::mem::size_of::<GlobalParams>() as u64;
        let gamma_size = std::mem::size_of::<GammaParams>() as u64;
        let path_globals_offset = globals_size.next_multiple_of(uniform_alignment);
        let gamma_offset = (path_globals_offset + globals_size).next_multiple_of(uniform_alignment);
        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals_buffer"),
            size: gamma_offset + gamma_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (
            instance_data,
            instance_data_capacity,
            max_instance_data_size,
            instance_data_alignment,
        ) = if uses_webgl_instance_data {
            let max_texture_dimension = device.limits().max_texture_dimension_2d;
            let max_instance_data_size = (u64::from(max_texture_dimension).pow(2)
                * INSTANCE_TEXTURE_TEXEL_SIZE)
                .min(MAX_INSTANCE_BUFFER_SIZE);
            let initial_capacity = (2 * 1024 * 1024).min(max_instance_data_size);
            let (instance_data, capacity) =
                Self::create_instance_texture(&device, initial_capacity, max_texture_dimension);
            (
                instance_data,
                capacity,
                max_instance_data_size,
                INSTANCE_TEXTURE_TEXEL_SIZE,
            )
        } else {
            let max_buffer_size = device
                .limits()
                .max_buffer_size
                .min(device.limits().max_storage_buffer_binding_size)
                .min(MAX_INSTANCE_BUFFER_SIZE);
            let initial_capacity = (2 * 1024 * 1024).min(max_buffer_size);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instance_buffer"),
                size: initial_capacity,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (
                InstanceData::Storage(buffer),
                initial_capacity,
                max_buffer_size,
                device.limits().min_storage_buffer_offset_alignment as u64,
            )
        };
        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals_bind_group"),
            layout: &bind_group_layouts.globals,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: 0,
                        size: NonZeroU64::new(globals_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: gamma_offset,
                        size: NonZeroU64::new(gamma_size),
                    }),
                },
            ],
        });
        let path_globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("path_globals_bind_group"),
            layout: &bind_group_layouts.globals,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: path_globals_offset,
                        size: NonZeroU64::new(globals_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: gamma_offset,
                        size: NonZeroU64::new(gamma_size),
                    }),
                },
            ],
        });
        let max_texture_size = device.limits().max_texture_dimension_2d;
        let custom_shaders = crate::custom_shaders::CustomShaders::new(&device, &queue);

        Self {
            custom_shaders,
            resources: WgpuResources {
                device,
                queue,
                pipelines,
                bind_group_layouts,
                atlas_sampler,
                nearest_sampler,
                atlas_texture_bind_groups: FxHashMap::default(),
                globals_buffer,
                globals_bind_group,
                path_globals_bind_group,
                instance_data,
                path_intermediate_texture: None,
                path_intermediate_view: None,
                path_msaa_texture: None,
                path_msaa_view: None,
                retained: Default::default(),
                marks: Default::default(),
                texture_pool: TexturePool::default(),
                group_globals: FxHashMap::default(),
                group_frame: 0,
            },
            atlas,
            path_globals_offset,
            gamma_offset,
            instance_data_capacity,
            max_instance_data_size,
            instance_data_alignment,
            uses_webgl_instance_data,
            rendering_params,
            is_bgr: false,
            dual_source_blending,
            adapter_info: context.adapter.get_info(),
            target_format,
            max_texture_size,
            premultiplied_alpha: false,
        }
    }

    fn resources(&self) -> &WgpuResources {
        &self.resources
    }

    fn resources_mut(&mut self) -> &mut WgpuResources {
        &mut self.resources
    }

    /// Path rendering goes through an intermediate texture that must match the frame
    /// target's dimensions, so the textures are keyed by size and rebuilt on mismatch.
    fn ensure_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        let width = size.width.0 as u32;
        let height = size.height.0 as u32;
        if self
            .resources
            .path_intermediate_texture
            .as_ref()
            .is_some_and(|texture| texture.width() == width && texture.height() == height)
        {
            return;
        }

        let format = self.target_format;
        let path_sample_count = self.rendering_params.path_sample_count;
        let resources = &mut self.resources;

        let (texture, view) =
            Self::create_path_intermediate(&resources.device, format, width, height);
        resources.path_intermediate_texture = Some(texture);
        resources.path_intermediate_view = Some(view);

        let (path_msaa_texture, path_msaa_view) = Self::create_msaa_if_needed(
            &resources.device,
            format,
            width,
            height,
            path_sample_count,
        )
        .map(|(texture, view)| (Some(texture), Some(view)))
        .unwrap_or((None, None));
        resources.path_msaa_texture = path_msaa_texture;
        resources.path_msaa_view = path_msaa_view;
    }

    fn render_frame(
        &mut self,
        scene: &Scene,
        target_view: &wgpu::TextureView,
        size: Size<DevicePixels>,
        premultiplied_alpha: bool,
        clear_color: wgpu::Color,
    ) -> Result<wgpu::SubmissionIndex> {
        anyhow::ensure!(
            size.width.0 > 0 && size.height.0 > 0,
            "invalid render target size: {size:?}"
        );
        anyhow::ensure!(
            size.width.0 as u32 <= self.max_texture_size
                && size.height.0 as u32 <= self.max_texture_size,
            "render target size {size:?} exceeds maximum texture dimension {}",
            self.max_texture_size
        );

        self.atlas.before_frame();
        self.ensure_intermediate_textures(size);

        let gamma_params = GammaParams {
            gamma_ratios: self.rendering_params.gamma_ratios,
            grayscale_enhanced_contrast: self.rendering_params.grayscale_enhanced_contrast,
            subpixel_enhanced_contrast: self.rendering_params.subpixel_enhanced_contrast,
            is_bgr: self.is_bgr as u32,
            _pad: 0,
        };
        self.premultiplied_alpha = premultiplied_alpha;
        let globals = GlobalParams {
            viewport_size: [size.width.0 as f32, size.height.0 as f32],
            premultiplied_alpha: premultiplied_alpha as u32,
            pad: 0,
        };
        let path_globals = GlobalParams {
            premultiplied_alpha: 0,
            ..globals
        };
        self.resources.queue.write_buffer(
            &self.resources.globals_buffer,
            0,
            bytemuck::bytes_of(&globals),
        );
        self.resources.queue.write_buffer(
            &self.resources.globals_buffer,
            self.path_globals_offset,
            bytemuck::bytes_of(&path_globals),
        );
        self.resources.queue.write_buffer(
            &self.resources.globals_buffer,
            self.gamma_offset,
            bytemuck::bytes_of(&gamma_params),
        );

        self.record_frame(scene, target_view, size, clear_color)
            .inspect_err(|_| {
                // Queue writes are staged before encoding; flush them even if the frame fails.
                self.resources.queue.submit(std::iter::empty());
            })
    }

    fn record_frame(
        &mut self,
        scene: &Scene,
        frame_view: &wgpu::TextureView,
        size: Size<DevicePixels>,
        clear_color: wgpu::Color,
    ) -> Result<wgpu::SubmissionIndex> {
        // Every scene the frame draws, depth first, with its target's size: the frame's
        // for the window's scene, its image's for a group's.
        fn visit<'a>(
            scene: &'a Scene,
            viewport: [f32; 2],
            out: &mut SmallVec<[(&'a Scene, [f32; 2]); 4]>,
        ) {
            out.push((scene, viewport));
            let target = Size {
                width: DevicePixels(viewport[0] as i32),
                height: DevicePixels(viewport[1] as i32),
            };
            for group in &scene.groups {
                let size = group.image_size(target);
                visit(
                    &group.scene,
                    [size.width.0 as f32, size.height.0 as f32],
                    out,
                );
                if let Some(mask) = &group.mask {
                    let size = mask.image.image_size(target);
                    visit(
                        &mask.image.scene,
                        [size.width.0 as f32, size.height.0 as f32],
                        out,
                    );
                }
            }
        }
        let mut viewports = SmallVec::new();
        visit(
            scene,
            [size.width.0 as f32, size.height.0 as f32],
            &mut viewports,
        );
        // Texture targets (`Window::paint_to_texture`), after the window's scene, which
        // stays first.
        for target in &scene.texture_targets {
            visit(
                &target.scene,
                [target.size.width.0 as f32, target.size.height.0 as f32],
                &mut viewports,
            );
        }
        let scenes: SmallVec<[&Scene; 4]> = viewports.iter().map(|(scene, _)| *scene).collect();
        let retained_stride = self
            .prepare_retained_shapes(&viewports)
            .context("retained shapes")?;
        self.prepare_marks(&viewports);
        // No pass is open yet: a marks failure may panic here (debug).
        gpui::check_marks_reports();
        let mut instance_offset = 0;
        // First, at offset 0: the WebGL texture transport indexes its texture
        // absolutely and quads index gradient records from 0.
        let gradients = self
            .write_instance_binding(
                "gradients_bind_group",
                &mut instance_offset,
                &scene.gradients,
            )
            .context("gradients")?;
        debug_assert_eq!(gradients.first_instance, 0);
        let mut frame = FrameScenes {
            scenes: SmallVec::new(),
            retained_stride,
            gradients,
        };
        let mut retained_base = 0;
        let mut marks_base = 0;
        for drawn in &scenes {
            let instance_bindings = self
                .write_instances(drawn, &mut instance_offset)
                .with_context(|| {
                    format!(
                        "scene too large: {} paths, {} shadows, {} quads, {} underlines, {} monochrome sprites, {} subpixel sprites, {} polychrome sprites",
                        drawn.paths.len(),
                        drawn.shadows.len(),
                        drawn.quads.len(),
                        drawn.underlines.len(),
                        drawn.monochrome_sprites.len(),
                        drawn.subpixel_sprites.len(),
                        drawn.polychrome_sprites.len(),
                    )
                })?;
            frame.scenes.push(FrameScene {
                scene: drawn,
                instance_bindings,
                retained_base,
                marks_base,
            });
            retained_base += drawn.retained_shapes.len();
            marks_base += drawn.marks.len();
        }
        self.prepare_texture_bind_groups(&scenes);

        let mut encoder =
            self.resources()
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("main_encoder"),
                });

        // Backdrop filters and blend modes read back what is drawn, which the frame (a
        // surface) may not allow: draw into an offscreen texture and copy it to the
        // frame at the end. Groups draw into their own images, which allow it.
        let reads_frame = !scene.backdrop_filters.is_empty()
            || scene.groups.iter().any(|group| {
                group.effects.blend_mode.reads_backdrop() || group.effects.backdrop.is_some()
            });
        let offscreen = reads_frame.then(|| {
            self.create_backdrop_texture(
                size.width.0 as u32,
                size.height.0 as u32,
                "backdrop_frame",
            )
        });
        // Texture targets first, in order: the window and later targets sample them.
        let mut result = Ok(());
        for target in &scene.texture_targets {
            result = self.draw_texture_target(&mut encoder, &frame, target, &mut instance_offset);
            if result.is_err() {
                break;
            }
        }
        let target = DrawTarget {
            view: offscreen.as_ref().map_or(frame_view, |(_, view)| view),
            texture: offscreen.as_ref().map(|(texture, _)| texture),
            size,
        };
        let result = result.and_then(|()| {
            self.draw_scene(
                &mut encoder,
                &frame,
                scene,
                target,
                Some(clear_color),
                &mut instance_offset,
            )
        });
        // Keep the images, copies and globals this frame used; free the rest.
        self.resources.texture_pool.end_frame();
        // Group globals live 60 frames unused (sizes that animate come back), at most 64.
        self.resources.group_frame += 1;
        let frame_index = self.resources.group_frame;
        let globals = &mut self.resources.group_globals;
        globals.retain(|_, globals| frame_index - globals.last_used < 60);
        if globals.len() > 64 {
            globals.retain(|_, globals| globals.last_used == frame_index);
        }
        result?;

        if let Some((_, offscreen_view)) = &offscreen {
            let mut copy = begin_load_pass(&mut encoder, frame_view, "backdrop_frame_copy");
            let bind_group =
                self.create_texture_bind_group("backdrop_frame_source", offscreen_view);
            copy.set_pipeline(&self.resources().pipelines.backdrop.copy);
            copy.set_bind_group(0, &bind_group, &[]);
            copy.draw(0..3, 0..1);
        }

        let submission = self
            .resources()
            .queue
            .submit(std::iter::once(encoder.finish()));
        Ok(submission)
    }

    /// Draws `scene` (the window's, `frame.scenes[0]`, or a group's at `depth`) into
    /// `target`, cleared first when `clear` is set (else drawn over). Passes reopen
    /// only when a batch draws into them.
    fn draw_scene(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameScenes<'_>,
        scene: &Scene,
        target: DrawTarget<'_>,
        clear: Option<wgpu::Color>,
        instance_offset: &mut u64,
    ) -> Result<()> {
        let size = target.size;
        let root = frame.scenes[0].scene;
        let drawn = frame
            .get(scene)
            .context("scene not prepared for the frame")?;
        let instance_bindings = &drawn.instance_bindings;
        let target_view = target.view;
        let mut pass = clear.map(|color| target_pass(encoder, target, Some(color), "main_pass"));

        for batch in scene.batches() {
            match batch {
                PrimitiveBatch::Quads(range) => {
                    let pass = open_pass(&mut pass, encoder, target);
                    // Quads read gradient records at `@group(2)`.
                    pass.set_bind_group(2, &frame.gradients.bind_group, &[]);
                    self.draw_instances(
                        &instance_bindings.quads,
                        &self.resources().pipelines.quads,
                        instance_range(range),
                        pass,
                    )
                }
                PrimitiveBatch::Shadows(range) => self.draw_instances(
                    &instance_bindings.shadows,
                    &self.resources().pipelines.shadows,
                    instance_range(range),
                    open_pass(&mut pass, encoder, target),
                ),
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    if paths.is_empty() {
                        continue;
                    }

                    pass = None;
                    let rasterized =
                        self.draw_paths_to_intermediate(encoder, paths, instance_offset, size)?;
                    if rasterized {
                        self.draw_paths_from_intermediate(
                            paths,
                            instance_offset,
                            open_pass(&mut pass, encoder, target),
                        )?;
                    }
                }
                PrimitiveBatch::Underlines(range) => self.draw_instances(
                    &instance_bindings.underlines,
                    &self.resources().pipelines.underlines,
                    instance_range(range),
                    open_pass(&mut pass, encoder, target),
                ),
                PrimitiveBatch::MonochromeSprites { texture_id, range } => {
                    self.draw_sprites(
                        &instance_bindings.monochrome_sprites,
                        texture_id,
                        &self.resources().pipelines.mono_sprites,
                        instance_range(range),
                        open_pass(&mut pass, encoder, target),
                    )?;
                }
                PrimitiveBatch::SubpixelSprites { texture_id, range } => {
                    let resources = self.resources();
                    self.draw_sprites(
                        &instance_bindings.subpixel_sprites,
                        texture_id,
                        resources
                            .pipelines
                            .subpixel_sprites
                            .as_ref()
                            .unwrap_or(&resources.pipelines.mono_sprites),
                        instance_range(range),
                        open_pass(&mut pass, encoder, target),
                    )?;
                }
                PrimitiveBatch::PolychromeSprites { texture_id, range } => {
                    self.draw_sprites(
                        &instance_bindings.polychrome_sprites,
                        texture_id,
                        &self.resources().pipelines.poly_sprites,
                        instance_range(range),
                        open_pass(&mut pass, encoder, target),
                    )?;
                }
                // Surfaces are macOS-only for video playback and are not
                // implemented by the WGPU renderer.
                PrimitiveBatch::Surfaces(_surfaces) => {}
                PrimitiveBatch::RetainedShapes(range) => self.draw_retained_shapes(
                    &scene.retained_shapes[range.clone()],
                    drawn.retained_base + range.start,
                    frame.retained_stride,
                    open_pass(&mut pass, encoder, target),
                ),
                PrimitiveBatch::Marks(range) => self.draw_marks(
                    &scene.marks[range.clone()],
                    drawn.marks_base + range.start,
                    open_pass(&mut pass, encoder, target),
                ),
                PrimitiveBatch::BackdropFilters(range) => {
                    pass = None;
                    let target_texture = target
                        .texture
                        .context("backdrop filters draw to an offscreen frame")?;
                    for filter in &scene.backdrop_filters[range] {
                        if let Some(shader) = filter.shader.shader(root) {
                            let backdrop = filter
                                .shader
                                .reads_backdrop()
                                .then(|| {
                                    self.filter_backdrop(encoder, filter, target_texture, size)
                                })
                                .flatten();
                            let resources = &self.resources;
                            self.custom_shaders.draw(
                                &resources.device,
                                &resources.queue,
                                self.target_format,
                                encoder,
                                target_view,
                                shader,
                                filter,
                                filter.shader.uniforms(root),
                                backdrop
                                    .as_ref()
                                    .map(|(scratch, region)| (&scratch.view, *region)),
                                [size.width.0 as f32, size.height.0 as f32],
                            );
                            if let Some((scratch, _)) = &backdrop {
                                self.resources.texture_pool.release(&scratch.texture);
                            }
                            continue;
                        }
                        let filtered = self.filter_backdrop(encoder, filter, target_texture, size);
                        let Some((filtered, region)) = filtered else {
                            continue;
                        };
                        let mut composite =
                            target_pass(encoder, target, None, "backdrop_composite");
                        self.draw_backdrop(&mut composite, filter, &filtered.view, region, size);
                        drop(composite);
                        self.resources.texture_pool.release(&filtered.texture);
                    }
                }
                PrimitiveBatch::Groups(range) => {
                    pass = None;
                    for group in &scene.groups[range] {
                        self.draw_group(encoder, frame, group, target, instance_offset)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Draws a group's scene into an image the size of its bounds, starting with its
    /// filtered backdrop, then blurs it, reads what is behind it when its blend mode
    /// needs that, and composites it onto `target`.
    /// Draws a texture target's scene (`Window::paint_to_texture`) into its external
    /// texture, cleared to transparent, with the texture's viewport. A texture that is
    /// missing, another size or format, or not a render attachment is skipped (its
    /// pixels stay).
    fn draw_texture_target(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameScenes<'_>,
        target: &gpui::TextureTargetScene,
        instance_offset: &mut u64,
    ) -> Result<()> {
        let Some(texture) = self.atlas.external_texture(target.id) else {
            log::warn!("texture target {:?}: no such external texture", target.id);
            return Ok(());
        };
        if texture.width() != target.size.width.0 as u32
            || texture.height() != target.size.height.0 as u32
            || texture.format() != self.target_format
            || !texture
                .usage()
                .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
        {
            log::warn!(
                "texture target {:?}: the texture is {}x{} {:?} ({:?}), not a {:?} {:?} render attachment",
                target.id,
                texture.width(),
                texture.height(),
                texture.format(),
                texture.usage(),
                target.size,
                self.target_format
            );
            return Ok(());
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let draw_target = DrawTarget {
            view: &view,
            texture: Some(&texture),
            size: target.size,
        };
        let globals = self.group_globals(target.size);
        let frame_globals = (
            std::mem::replace(&mut self.resources.globals_bind_group, globals.0),
            std::mem::replace(&mut self.resources.path_globals_bind_group, globals.1),
        );
        let result = self.draw_scene(
            encoder,
            frame,
            &target.scene,
            draw_target,
            Some(wgpu::Color::TRANSPARENT),
            instance_offset,
        );
        self.resources.globals_bind_group = frame_globals.0;
        self.resources.path_globals_bind_group = frame_globals.1;
        result
    }

    fn draw_group(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameScenes<'_>,
        group: &gpui::CompositedGroup,
        target: DrawTarget<'_>,
        instance_offset: &mut u64,
    ) -> Result<()> {
        let size = group.image_size(target.size);
        if group.target_region(target.size).is_none() || size.width.0 <= 0 || size.height.0 <= 0 {
            return Ok(());
        }
        let largest = self.max_texture_size.saturating_sub(GROUP_IMAGE_STEP);
        if size.width.0 as u32 > largest || size.height.0 as u32 > largest {
            log::error!("compositing group image of {size:?} is too large; not drawn");
            return Ok(());
        }
        let origin = [group.bounds.origin.x.0, group.bounds.origin.y.0];
        let image = self.resources.texture_pool.acquire(
            &self.resources.device,
            self.target_format,
            bucket(size.width.0 as u32),
            bucket(size.height.0 as u32),
        );
        let image_target = DrawTarget {
            view: &image.view,
            texture: Some(&image.texture),
            size,
        };
        // The image starts with what is behind the group, filtered (its backdrop).
        let mut clear = Some(wgpu::Color::TRANSPARENT);
        if let Some(filter) = &group.effects.backdrop {
            let target_texture = target
                .texture
                .context("group backdrops draw to an offscreen frame")?;
            if let Some((filtered, mut region)) =
                self.filter_backdrop(encoder, filter, target_texture, target.size)
            {
                let filter = filter.translated(-group.bounds.origin);
                region[0] -= origin[0];
                region[1] -= origin[1];
                let mut pass = target_pass(encoder, image_target, clear.take(), "group_backdrop");
                let pipeline = &self.resources().pipelines.backdrop.backdrop_group;
                self.draw_backdrop_with(&mut pass, pipeline, &filter, &filtered.view, region, size);
                drop(pass);
                self.resources.texture_pool.release(&filtered.texture);
            }
        }
        // The group's scene draws with its image's viewport.
        let globals = self.group_globals(size);
        let frame_globals = (
            std::mem::replace(&mut self.resources.globals_bind_group, globals.0),
            std::mem::replace(&mut self.resources.path_globals_bind_group, globals.1),
        );
        let result = self.draw_scene(
            encoder,
            frame,
            &group.scene,
            image_target,
            clear,
            instance_offset,
        );
        self.resources.globals_bind_group = frame_globals.0;
        self.resources.path_globals_bind_group = frame_globals.1;
        result?;
        let viewport = [target.size.width.0 as f32, target.size.height.0 as f32];
        let blurred = if group.effects.blur_radius.0 >= 0.5 {
            let whole = Bounds::new(
                gpui::point(ScaledPixels(0.), ScaledPixels(0.)),
                group.bounds.size,
            );
            match self.copy_region(encoder, &image.texture, whole, group.effects.blur_radius.0) {
                Some(blurred) => Some(blurred),
                None => {
                    self.resources.texture_pool.release(&image.texture);
                    return Ok(());
                }
            }
        } else {
            None
        };
        let (source, source_region) = match &blurred {
            Some((scratch, region)) => (
                &scratch.view,
                [
                    region[0] + origin[0],
                    region[1] + origin[1],
                    region[2],
                    region[3],
                ],
            ),
            None => (
                &image.view,
                [
                    origin[0],
                    origin[1],
                    image.texture.width() as f32,
                    image.texture.height() as f32,
                ],
            ),
        };
        let backdrop = match group.target_region(target.size) {
            Some(region) if group.effects.blend_mode.reads_backdrop() => {
                let target_texture = target
                    .texture
                    .context("blend modes draw to an offscreen frame")?;
                self.copy_region(encoder, target_texture, region, 0.)
            }
            _ => None,
        };
        let (backdrop_view, backdrop_region) = match &backdrop {
            Some((scratch, region)) => (&scratch.view, *region),
            None => (source, source_region),
        };
        // The mask's image, drawn like a group's (it is never composited itself).
        let mask = match &group.mask {
            Some(mask) => {
                let image_size = mask.image.image_size(target.size);
                if image_size.width.0 <= 0 || image_size.height.0 <= 0 {
                    // An empty mask draws nothing of the group.
                    let pool = &self.resources.texture_pool;
                    pool.release(&image.texture);
                    for (scratch, _) in blurred.iter().chain(backdrop.iter()) {
                        pool.release(&scratch.texture);
                    }
                    return Ok(());
                }
                let texture = self.resources.texture_pool.acquire(
                    &self.resources.device,
                    self.target_format,
                    bucket(image_size.width.0 as u32),
                    bucket(image_size.height.0 as u32),
                );
                let mask_target = DrawTarget {
                    view: &texture.view,
                    texture: Some(&texture.texture),
                    size: image_size,
                };
                let globals = self.group_globals(image_size);
                let frame_globals = (
                    std::mem::replace(&mut self.resources.globals_bind_group, globals.0),
                    std::mem::replace(&mut self.resources.path_globals_bind_group, globals.1),
                );
                let result = self.draw_scene(
                    encoder,
                    frame,
                    &mask.image.scene,
                    mask_target,
                    Some(wgpu::Color::TRANSPARENT),
                    instance_offset,
                );
                self.resources.globals_bind_group = frame_globals.0;
                self.resources.path_globals_bind_group = frame_globals.1;
                result?;
                let region = [
                    mask.image.bounds.origin.x.0,
                    mask.image.bounds.origin.y.0,
                    texture.texture.width() as f32,
                    texture.texture.height() as f32,
                ];
                Some((texture, region))
            }
            None => None,
        };
        let mask_region = mask.as_ref().map_or([0.; 4], |(_, region)| *region);
        let uniforms =
            group.composite_uniforms(source_region, backdrop_region, mask_region, viewport);
        let uniforms = self.backdrop_uniforms(uniforms.as_bytes());
        let source_group = self.create_texture_bind_group("group_image", source);
        let backdrop_group = self.create_texture_bind_group("group_backdrop", backdrop_view);
        let mask_view = mask.as_ref().map_or(source, |(texture, _)| &texture.view);
        let mask_group = self.create_texture_bind_group("group_mask", mask_view);
        let mut pass = target_pass(encoder, target, None, "group_composite");
        pass.set_pipeline(
            self.resources()
                .pipelines
                .backdrop
                .group_composite(group.effects.blend_mode),
        );
        pass.set_bind_group(0, &source_group, &[]);
        pass.set_bind_group(1, &uniforms, &[]);
        pass.set_bind_group(2, &backdrop_group, &[]);
        pass.set_bind_group(3, &mask_group, &[]);
        pass.draw(0..4, 0..1);
        drop(pass);
        // Later passes reuse these; wgpu orders them after this composite.
        let pool = &self.resources.texture_pool;
        pool.release(&image.texture);
        for (scratch, _) in blurred.iter().chain(backdrop.iter()) {
            pool.release(&scratch.texture);
        }
        if let Some((texture, _)) = &mask {
            pool.release(&texture.texture);
        }
        Ok(())
    }

    /// Bind groups of globals for a target of `size` (a group's image): the globals
    /// and the path globals, with the frame's gamma.
    fn group_globals(&mut self, size: Size<DevicePixels>) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let globals = GlobalParams {
            viewport_size: [size.width.0 as f32, size.height.0 as f32],
            premultiplied_alpha: self.premultiplied_alpha as u32,
            pad: 0,
        };
        let path_globals = GlobalParams {
            premultiplied_alpha: 0,
            ..globals
        };
        let path_offset = self.path_globals_offset;
        let gamma_offset = self.gamma_offset;
        let resources = &mut self.resources;
        let globals_size = std::mem::size_of::<GlobalParams>() as u64;
        let gamma_size = std::mem::size_of::<GammaParams>() as u64;
        let entry = resources
            .group_globals
            .entry((size.width.0, size.height.0))
            .or_insert_with(|| {
                let buffer = resources.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("group_globals"),
                    size: path_offset + globals_size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind_group = |offset| {
                    resources
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("group_globals"),
                            layout: &resources.bind_group_layouts.globals,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &buffer,
                                        offset,
                                        size: NonZeroU64::new(globals_size),
                                    }),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &resources.globals_buffer,
                                        offset: gamma_offset,
                                        size: NonZeroU64::new(gamma_size),
                                    }),
                                },
                            ],
                        })
                };
                GroupGlobals {
                    globals: bind_group(0),
                    path_globals: bind_group(path_offset),
                    buffer,
                    last_used: 0,
                }
            });
        // The frame drawing now is `group_frame + 1` (counted at its end).
        let frame = resources.group_frame + 1;
        if entry.last_used != frame {
            entry.last_used = frame;
            resources
                .queue
                .write_buffer(&entry.buffer, 0, bytemuck::bytes_of(&globals));
            resources.queue.write_buffer(
                &entry.buffer,
                path_offset,
                bytemuck::bytes_of(&path_globals),
            );
        }
        (entry.globals.clone(), entry.path_globals.clone())
    }

    /// Bytes of the group images and copies kept now (for tests and stats).
    pub fn group_texture_bytes(&self) -> u64 {
        self.resources.texture_pool.bytes()
    }

    /// A texture the backdrop passes render to, sample and copy into.
    fn create_backdrop_texture(
        &self,
        width: u32,
        height: u32,
        label: &str,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = self
            .resources()
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: width.max(1),
                    height: height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// Copies the region a backdrop filter reads from the frame, blurs it (at a lower
    /// resolution for large radii) and returns the filtered view and where it sits in
    /// the frame (origin and size, device pixels). `None` if it reads nothing.
    fn filter_backdrop(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        filter: &BackdropFilter,
        frame_texture: &wgpu::Texture,
        viewport_size: Size<DevicePixels>,
    ) -> Option<(Scratch, [f32; 4])> {
        let viewport = Bounds::new(
            gpui::point(ScaledPixels(0.), ScaledPixels(0.)),
            gpui::size(
                ScaledPixels(viewport_size.width.0 as f32),
                ScaledPixels(viewport_size.height.0 as f32),
            ),
        );
        let region = filter
            .sample_bounds()
            .intersect(&filter.content_mask.bounds.dilate(filter.blur_radius * 3.))
            .intersect(&viewport);
        self.copy_region(encoder, frame_texture, region, filter.blur_radius.0)
    }

    /// Copies `region` (within the texture) of `frame_texture` and blurs it with
    /// standard deviation `sigma` (at a lower resolution for large ones); returns the
    /// view and where it sits (origin and size, whole device pixels). `None` if the
    /// region is empty.
    fn copy_region(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame_texture: &wgpu::Texture,
        region: Bounds<ScaledPixels>,
        sigma: f32,
    ) -> Option<(Scratch, [f32; 4])> {
        let (x, y) = (region.origin.x.0.floor(), region.origin.y.0.floor());
        let width = (region.origin.x.0 + region.size.width.0).ceil() - x;
        let height = (region.origin.y.0 + region.size.height.0).ceil() - y;
        if width < 1. || height < 1. {
            return None;
        }
        let (width, height) = (width as u32, height as u32);
        let pool = &self.resources().texture_pool;
        let device = &self.resources().device;
        let copy = pool.acquire(device, self.target_format, width, height);
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: frame_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: x as u32,
                    y: y as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &copy.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let region = [x, y, width as f32, height as f32];
        if sigma < 0.5 {
            return Some((copy, region));
        }
        // Blur at most 8 texels wide: downsample larger radii and let the composite's
        // bilinear sampling scale back up.
        let scale = (sigma / 8.).ceil().max(1.);
        let small_width = ((width as f32) / scale).ceil().max(1.) as u32;
        let small_height = ((height as f32) / scale).ceil().max(1.) as u32;
        let a = pool.acquire(device, self.target_format, small_width, small_height);
        let b = pool.acquire(device, self.target_format, small_width, small_height);
        let (copy_view, a_view, b_view) = (&copy.view, &a.view, &b.view);
        let sigma = sigma / scale;
        let radius = (sigma * 3.).ceil() as i32;
        self.backdrop_pass(encoder, &copy_view, &a_view, None);
        self.backdrop_pass(
            encoder,
            &a_view,
            &b_view,
            Some(BackdropBlurParams {
                direction: [1. / small_width as f32, 0.],
                sigma,
                radius,
            }),
        );
        self.backdrop_pass(
            encoder,
            &b_view,
            &a_view,
            Some(BackdropBlurParams {
                direction: [0., 1. / small_height as f32],
                sigma,
                radius,
            }),
        );
        pool.release(&copy.texture);
        pool.release(&b.texture);
        Some((a, region))
    }

    /// One fullscreen pass: a copy (`blur: None`) or one axis of the blur.
    fn backdrop_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        destination: &wgpu::TextureView,
        blur: Option<BackdropBlurParams>,
    ) {
        let resources = self.resources();
        let source_group = self.create_texture_bind_group("backdrop_source", source);
        let uniforms = blur.map(|params| self.backdrop_uniforms(bytemuck::bytes_of(&params)));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("backdrop_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: destination,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        match &uniforms {
            Some(group) => {
                pass.set_pipeline(&resources.pipelines.backdrop.blur);
                pass.set_bind_group(1, group, &[]);
            }
            None => pass.set_pipeline(&resources.pipelines.backdrop.copy),
        }
        pass.set_bind_group(0, &source_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn backdrop_uniforms(&self, bytes: &[u8]) -> wgpu::BindGroup {
        let resources = self.resources();
        let buffer = resources.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("backdrop_uniforms"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        resources.queue.write_buffer(&buffer, 0, bytes);
        resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("backdrop_uniforms"),
                layout: &resources.pipelines.backdrop.uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
    }

    fn draw_backdrop(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        filter: &BackdropFilter,
        filtered: &wgpu::TextureView,
        region: [f32; 4],
        viewport_size: Size<DevicePixels>,
    ) {
        let pipeline = &self.resources().pipelines.backdrop.composite;
        self.draw_backdrop_with(pass, pipeline, filter, filtered, region, viewport_size);
    }

    fn draw_backdrop_with(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        pipeline: &wgpu::RenderPipeline,
        filter: &BackdropFilter,
        filtered: &wgpu::TextureView,
        region: [f32; 4],
        viewport_size: Size<DevicePixels>,
    ) {
        let rect =
            |b: Bounds<ScaledPixels>| [b.origin.x.0, b.origin.y.0, b.size.width.0, b.size.height.0];
        let radii = filter.corner_radii;
        let params = BackdropCompositeParams {
            bounds: rect(filter.bounds),
            content_mask: rect(filter.content_mask.bounds),
            corner_radii: [
                radii.top_left.0,
                radii.top_right.0,
                radii.bottom_right.0,
                radii.bottom_left.0,
            ],
            region,
            viewport_size: [viewport_size.width.0 as f32, viewport_size.height.0 as f32],
            saturate: filter.saturate,
            brightness: filter.brightness,
            contrast: filter.contrast,
            opacity: filter.opacity,
            pad: [0.; 2],
        };
        let uniforms = self.backdrop_uniforms(bytemuck::bytes_of(&params));
        let source = self.create_texture_bind_group("backdrop_filtered", filtered);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &source, &[]);
        pass.set_bind_group(1, &uniforms, &[]);
        pass.draw(0..4, 0..1);
    }

    fn write_instances(
        &mut self,
        scene: &Scene,
        instance_offset: &mut u64,
    ) -> Result<InstanceBindings> {
        Ok(InstanceBindings {
            quads: self.write_instance_binding(
                "quads_bind_group",
                instance_offset,
                &scene.quads,
            )?,
            shadows: self.write_instance_binding(
                "shadows_bind_group",
                instance_offset,
                &scene.shadows,
            )?,
            underlines: self.write_instance_binding(
                "underlines_bind_group",
                instance_offset,
                &scene.underlines,
            )?,
            monochrome_sprites: self.write_instance_binding(
                "monochrome_sprites_bind_group",
                instance_offset,
                &scene.monochrome_sprites,
            )?,
            subpixel_sprites: self.write_instance_binding(
                "subpixel_sprites_bind_group",
                instance_offset,
                &scene.subpixel_sprites,
            )?,
            polychrome_sprites: self.write_instance_binding(
                "polychrome_sprites_bind_group",
                instance_offset,
                &scene.polychrome_sprites,
            )?,
        })
    }

    fn create_texture_bind_group(
        &self,
        label: &str,
        texture_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        self.create_sampled_texture_bind_group(label, texture_view, gpui::TextureSampling::Linear)
    }

    fn create_sampled_texture_bind_group(
        &self,
        label: &str,
        texture_view: &wgpu::TextureView,
        sampling: gpui::TextureSampling,
    ) -> wgpu::BindGroup {
        let resources = self.resources();
        let sampler = match sampling {
            gpui::TextureSampling::Linear => &resources.atlas_sampler,
            gpui::TextureSampling::Nearest => &resources.nearest_sampler,
        };
        resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &resources.bind_group_layouts.texture,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
    }

    fn prepare_texture_bind_groups(&mut self, scenes: &[&Scene]) {
        let mut texture_ids = SmallVec::<[AtlasTextureId; 8]>::new();
        for batch in scenes.iter().flat_map(|scene| scene.batches()) {
            let texture_id = match batch {
                PrimitiveBatch::MonochromeSprites { texture_id, .. }
                | PrimitiveBatch::SubpixelSprites { texture_id, .. }
                | PrimitiveBatch::PolychromeSprites { texture_id, .. } => texture_id,
                _ => continue,
            };
            if !texture_ids.contains(&texture_id) {
                texture_ids.push(texture_id);
            }
        }

        self.resources_mut()
            .atlas_texture_bind_groups
            .retain(|texture_id, _| texture_ids.contains(texture_id));

        for texture_id in texture_ids {
            let Some(texture_info) = self.atlas.get_texture_info(texture_id) else {
                self.resources_mut()
                    .atlas_texture_bind_groups
                    .remove(&texture_id);
                continue;
            };
            let is_current = self
                .resources()
                .atlas_texture_bind_groups
                .get(&texture_id)
                .is_some_and(|cached| cached.texture_generation == texture_info.generation);
            if is_current {
                continue;
            }

            let bind_group = self.create_sampled_texture_bind_group(
                "atlas_texture_bind_group",
                &texture_info.view,
                texture_info.sampling,
            );
            self.resources_mut().atlas_texture_bind_groups.insert(
                texture_id,
                CachedTextureBindGroup {
                    texture_generation: texture_info.generation,
                    bind_group,
                },
            );
        }
    }

    fn draw_instances(
        &self,
        instances: &InstanceBinding,
        pipeline: &wgpu::RenderPipeline,
        range: Range<u32>,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        if range.is_empty() {
            return;
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.resources().globals_bind_group, &[]);
        pass.set_bind_group(1, &instances.bind_group, &[]);
        pass.draw(
            0..4,
            instances.first_instance + range.start..instances.first_instance + range.end,
        );
    }

    fn draw_sprites(
        &self,
        sprite_instances: &InstanceBinding,
        texture_id: AtlasTextureId,
        pipeline: &wgpu::RenderPipeline,
        range: Range<u32>,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> Result<()> {
        if range.is_empty() {
            return Ok(());
        }
        let resources = self.resources();
        // The atlas has released this texture; the batch belongs to a stale
        // paint that will be replaced once its view re-renders.
        let Some(texture) = resources.atlas_texture_bind_groups.get(&texture_id) else {
            return Ok(());
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &resources.globals_bind_group, &[]);
        pass.set_bind_group(1, &sprite_instances.bind_group, &[]);
        pass.set_bind_group(2, &texture.bind_group, &[]);
        pass.draw(
            0..4,
            sprite_instances.first_instance + range.start
                ..sprite_instances.first_instance + range.end,
        );
        Ok(())
    }

    unsafe fn instance_bytes<T>(instances: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                instances.as_ptr() as *const u8,
                std::mem::size_of_val(instances),
            )
        }
    }

    fn draw_paths_from_intermediate(
        &mut self,
        paths: &[Path<ScaledPixels>],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> Result<()> {
        // Each pixel is copied at most once (paths can be translucent): overlapping paths'
        // bounds are copied as one region.
        let sprites: Vec<PathSprite> = gpui::path_copy_bounds(paths)
            .into_iter()
            .map(|bounds| PathSprite { bounds })
            .collect();

        let Some(path_intermediate_view) = self.resources().path_intermediate_view.clone() else {
            return Ok(());
        };
        let instances =
            self.write_instance_binding("path_sprites_bind_group", instance_offset, &sprites)?;
        let texture = self.create_texture_bind_group(
            "path_intermediate_texture_bind_group",
            &path_intermediate_view,
        );
        let resources = self.resources();
        pass.set_pipeline(&resources.pipelines.paths);
        pass.set_bind_group(0, &resources.globals_bind_group, &[]);
        pass.set_bind_group(1, &instances.bind_group, &[]);
        pass.set_bind_group(2, &texture, &[]);
        pass.draw(
            0..4,
            instances.first_instance..instances.first_instance + sprites.len() as u32,
        );
        Ok(())
    }

    fn draw_paths_to_intermediate(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        paths: &[Path<ScaledPixels>],
        instance_offset: &mut u64,
        size: Size<DevicePixels>,
    ) -> Result<bool> {
        let mut vertices = Vec::new();
        for path in paths {
            let bounds = path.clipped_bounds();
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds,
                content_mask: path.content_mask,
            }));
        }

        if vertices.is_empty() {
            return Ok(false);
        }

        let vertex_binding = self.write_instance_binding(
            "path_rasterization_bind_group",
            instance_offset,
            &vertices,
        )?;

        let resources = self.resources();
        let Some(path_intermediate_view) = resources.path_intermediate_view.as_ref() else {
            return Ok(false);
        };

        let (target_view, resolve_target) = if let Some(ref msaa_view) = resources.path_msaa_view {
            (msaa_view, Some(path_intermediate_view))
        } else {
            (path_intermediate_view, None)
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("path_rasterization_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            // The target may be smaller than the intermediate texture (a group's image).
            pass.set_viewport(0., 0., size.width.0 as f32, size.height.0 as f32, 0., 1.);
            pass.set_pipeline(&resources.pipelines.path_rasterization);
            pass.set_bind_group(0, &resources.path_globals_bind_group, &[]);
            pass.set_bind_group(1, &vertex_binding.bind_group, &[]);
            // The path rasterization shader loads records by vertex index
            // rather than instance index, so the allocation's base shifts the
            // vertex range here.
            pass.draw(
                vertex_binding.first_instance
                    ..vertex_binding.first_instance + vertices.len() as u32,
                0..1,
            );
        }

        Ok(true)
    }

    fn write_instance_binding<T>(
        &mut self,
        label: &str,
        instance_offset: &mut u64,
        instances: &[T],
    ) -> Result<InstanceBinding> {
        let data = unsafe { Self::instance_bytes(instances) };
        // wgpu rejects zero-sized bindings, so empty primitive arrays still
        // reserve the 16-byte minimum.
        let size = (data.len() as u64).max(16);
        let stride = (std::mem::size_of::<T>() as u64).max(1);
        let (alignment, allocation_size) = if self.uses_webgl_instance_data {
            // The texture transport has no binding offset: the shader indexes
            // the instance texture absolutely, so each allocation must start on
            // a whole instance (a stride multiple) and a whole texel, and must
            // end on a texel boundary so the zero padding of its final partial
            // texel cannot overlap the next allocation.
            (
                least_common_multiple(self.instance_data_alignment, stride),
                size.next_multiple_of(INSTANCE_TEXTURE_TEXEL_SIZE),
            )
        } else {
            (self.instance_data_alignment.max(1), size)
        };
        let mut offset = (*instance_offset).next_multiple_of(alignment);
        if offset + allocation_size > self.instance_data_capacity {
            self.grow_instance_data(allocation_size)?;
            offset = 0;
        }
        *instance_offset = offset + allocation_size;

        let first_instance = if self.uses_webgl_instance_data {
            u32::try_from(offset / stride).context("instance index exceeds u32 range")?
        } else {
            0
        };

        let resources = self.resources();
        if !data.is_empty() {
            match &resources.instance_data {
                InstanceData::Storage(buffer) => resources.queue.write_buffer(buffer, offset, data),
                InstanceData::Texture { .. } => {
                    Self::write_instance_texture(resources, offset, data)
                }
            }
        }
        let bind_group = resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &resources.bind_group_layouts.instances,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: match &resources.instance_data {
                        InstanceData::Storage(buffer) => {
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer,
                                offset,
                                size: NonZeroU64::new(size),
                            })
                        }
                        InstanceData::Texture { view, .. } => {
                            wgpu::BindingResource::TextureView(view)
                        }
                    },
                }],
            });
        Ok(InstanceBinding {
            bind_group,
            first_instance,
        })
    }

    fn write_instance_texture(resources: &WgpuResources, offset: u64, data: &[u8]) {
        let InstanceData::Texture {
            texture,
            width,
            height,
            ..
        } = &resources.instance_data
        else {
            return;
        };
        Self::write_texture_bytes(&resources.queue, texture, *width, *height, offset, data);
    }

    /// Write `data` into an instance texture starting at byte `offset` (a whole
    /// texel), padding the final texel.
    fn write_texture_bytes(
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
        offset: u64,
        data: &[u8],
    ) {
        let (width, height) = (&width, &height);
        let mut byte_offset = 0usize;
        let mut texel_offset = offset / INSTANCE_TEXTURE_TEXEL_SIZE;
        while byte_offset < data.len() {
            let x = (texel_offset % u64::from(*width)) as u32;
            let y = (texel_offset / u64::from(*width)) as u32;
            if y >= *height {
                // The capacity check in write_instance_binding should make this
                // unreachable. Truncating silently would leave stale bytes in the
                // texture and draw garbage for the remaining instances.
                debug_assert!(
                    false,
                    "instance texture write out of bounds: row {y} >= height {}",
                    *height
                );
                log::error!(
                    "instance texture write out of bounds; dropping {} bytes of instance data",
                    data.len() - byte_offset
                );
                return;
            }
            let available_texels = u64::from(*width - x);
            let remaining_bytes = data.len() - byte_offset;
            let complete_texels = remaining_bytes as u64 / INSTANCE_TEXTURE_TEXEL_SIZE;
            let texels = complete_texels.min(available_texels);
            if texels > 0 {
                let byte_count = (texels * INSTANCE_TEXTURE_TEXEL_SIZE) as usize;
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x, y, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &data[byte_offset..byte_offset + byte_count],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(byte_count as u32),
                        rows_per_image: None,
                    },
                    wgpu::Extent3d {
                        width: texels as u32,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
                byte_offset += byte_count;
                texel_offset += texels;
                continue;
            }

            let mut final_texel = [0; INSTANCE_TEXTURE_TEXEL_SIZE as usize];
            final_texel[..remaining_bytes].copy_from_slice(&data[byte_offset..]);
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                &final_texel,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(INSTANCE_TEXTURE_TEXEL_SIZE as u32),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            break;
        }
    }

    fn grow_instance_data(&mut self, required: u64) -> Result<()> {
        let capacity = (self.instance_data_capacity * 2)
            .max(required.next_power_of_two())
            .min(self.max_instance_data_size);
        anyhow::ensure!(
            capacity >= required,
            "instance data needs {required} bytes, above the maximum of {}",
            self.max_instance_data_size
        );
        anyhow::ensure!(
            capacity > self.instance_data_capacity,
            "frame instance data exceeds the {}-byte maximum",
            self.max_instance_data_size
        );
        log::debug!(
            "instance data grown from {} to {capacity}",
            self.instance_data_capacity
        );
        // Bind groups created earlier in the frame keep the previous buffer or
        // texture alive, so allocations written before the grow remain valid;
        // only subsequent writes land in the new allocation.
        let uses_webgl_instance_data = self.uses_webgl_instance_data;
        let resources = self.resources_mut();
        if uses_webgl_instance_data {
            let max_texture_dimension = resources.device.limits().max_texture_dimension_2d;
            let (instance_data, actual_capacity) =
                Self::create_instance_texture(&resources.device, capacity, max_texture_dimension);
            resources.instance_data = instance_data;
            self.instance_data_capacity = actual_capacity;
        } else {
            resources.instance_data =
                InstanceData::Storage(resources.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("instance_buffer"),
                    size: capacity,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            self.instance_data_capacity = capacity;
        }
        Ok(())
    }
}

impl WgpuRenderer {
    /// Mark the surface as unconfigured so rendering is skipped until a new
    /// surface is provided via [`replace_surface`](Self::replace_surface).
    ///
    /// This does **not** drop the renderer — the device, queue, atlas, and
    /// pipelines stay alive.  Use this when the native window is destroyed
    /// (e.g. Android `TerminateWindow`) but you intend to re-create the
    /// surface later without losing cached atlas textures.
    pub fn unconfigure_surface(&mut self) {
        self.state = match std::mem::replace(&mut self.state, RendererState::Released) {
            RendererState::Ready { core, surface } => {
                drop(surface);
                RendererState::Unconfigured { core }
            }
            state @ (RendererState::Unconfigured { .. } | RendererState::Released) => state,
        };
        // Drop intermediate textures since they reference the old surface size.
        if let Some(core) = self.core_mut() {
            core.resources.invalidate_intermediate_textures();
        }
    }

    /// Replace the wgpu surface with a new one (e.g. after Android destroys
    /// and recreates the native window).  Keeps the device, queue, atlas, and
    /// all pipelines intact so cached `AtlasTextureId`s remain valid.
    ///
    /// The `instance` **must** be the same [`wgpu::Instance`] that was used to
    /// create the adapter and device (i.e. from the [`WgpuContext`]).  Using a
    /// different instance will cause a "Device does not exist" panic because
    /// the wgpu device is bound to its originating instance.
    #[cfg(not(target_family = "wasm"))]
    pub fn replace_surface<W: HasWindowHandle>(
        &mut self,
        window: &W,
        config: WgpuSurfaceConfig,
        instance: &wgpu::Instance,
    ) -> anyhow::Result<()> {
        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;

        let surface = create_surface(instance, window_handle.as_raw())?;

        let width = (config.size.width.0 as u32).max(1);
        let height = (config.size.height.0 as u32).max(1);

        let alpha_mode = if config.transparent {
            self.transparent_alpha_mode
        } else {
            self.opaque_alpha_mode
        };

        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface_config.alpha_mode = alpha_mode;
        if let Some(mode) = config.preferred_present_mode {
            self.surface_config.present_mode = mode;
        }

        let mut core = match std::mem::replace(&mut self.state, RendererState::Released) {
            RendererState::Ready {
                core,
                surface: old_surface,
            } => {
                drop(old_surface);
                core
            }
            RendererState::Unconfigured { core } => core,
            RendererState::Released => {
                anyhow::bail!("Cannot replace the surface: GPU resources have been released")
            }
        };
        surface.configure(&core.resources.device, &self.surface_config);
        core.resources.invalidate_intermediate_textures();
        self.state = RendererState::Ready { surface, core };

        Ok(())
    }

    pub fn destroy(&mut self) {
        // Release surface-bound GPU resources eagerly so the underlying native
        // window can be destroyed before the renderer itself is dropped.
        self.state = RendererState::Released;
    }

    /// Returns true if the GPU device was lost and recovery is needed.
    pub fn device_lost(&self) -> bool {
        self.device_errors.device_lost()
    }

    /// Returns true if a redraw is needed because GPU state was cleared.
    /// Calling this method clears the flag.
    pub fn needs_redraw(&mut self) -> bool {
        std::mem::take(&mut self.needs_redraw)
    }

    /// Recovers from a lost GPU device by recreating the renderer with a new context.
    ///
    /// Call this after detecting `device_lost()` returns true.
    ///
    /// This method coordinates recovery across multiple windows:
    /// - The first window to call this will recreate the shared context
    /// - Subsequent windows will adopt the already-recovered context
    #[cfg(not(target_family = "wasm"))]
    pub fn recover<W>(&mut self, window: &W) -> anyhow::Result<()>
    where
        W: HasWindowHandle + HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
    {
        let gpu_context = self.context.as_ref().expect("recover requires gpu_context");

        // Check if another window already recovered the context
        let needs_new_context = gpu_context
            .borrow()
            .as_ref()
            .is_none_or(|ctx| ctx.device_lost());

        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;

        let surface = if needs_new_context {
            log::warn!("GPU device lost, recreating context...");

            // Drop old resources to release Arc<Device>/Arc<Queue> and GPU resources
            self.state = RendererState::Released;
            *gpu_context.borrow_mut() = None;

            // Wait briefly for the GPU driver to stabilize, then try to
            // recreate the context without software renderers. If this fails
            // the caller should request another frame and retry — the real GPU
            // may need more time to come back (e.g. after suspend/resume).
            std::thread::sleep(std::time::Duration::from_millis(350));

            let instance = WgpuContext::instance(Some(Box::new(window.clone())));
            let surface = create_surface(&instance, window_handle.as_raw())?;
            let new_context =
                WgpuContext::new_rejecting_software(instance, &surface, self.compositor_gpu)?;
            *gpu_context.borrow_mut() = Some(new_context);
            surface
        } else {
            let ctx_ref = gpu_context.borrow();
            let instance = &ctx_ref.as_ref().unwrap().instance;
            create_surface(instance, window_handle.as_raw())?
        };

        let config = WgpuSurfaceConfig {
            size: gpui::Size {
                width: gpui::DevicePixels(self.surface_config.width as i32),
                height: gpui::DevicePixels(self.surface_config.height as i32),
            },
            transparent: self.surface_config.alpha_mode != wgpu::CompositeAlphaMode::Opaque,
            preferred_present_mode: Some(self.surface_config.present_mode),
        };
        let gpu_context = Rc::clone(gpu_context);
        let ctx_ref = gpu_context.borrow();
        let context = ctx_ref.as_ref().expect("context should exist");

        self.state = RendererState::Released;
        self.atlas.handle_device_lost(context);

        let is_bgr = self.is_bgr;
        *self = Self::new_internal(
            Some(gpu_context.clone()),
            context,
            surface,
            config,
            self.compositor_gpu,
            self.atlas.clone(),
        )?;
        self.set_subpixel_layout(is_bgr);

        log::info!("GPU recovery complete");
        Ok(())
    }
}

#[cfg(all(
    not(target_family = "wasm"),
    any(test, feature = "bench-support", feature = "test-support")
))]
struct HeadlessRenderTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[cfg(all(
    not(target_family = "wasm"),
    any(test, feature = "bench-support", feature = "test-support")
))]
impl HeadlessRenderTarget {
    fn size(&self) -> Size<DevicePixels> {
        Size {
            width: DevicePixels(self.texture.width() as i32),
            height: DevicePixels(self.texture.height() as i32),
        }
    }
}

#[cfg(all(
    not(target_family = "wasm"),
    any(test, feature = "bench-support", feature = "test-support")
))]
pub struct WgpuHeadlessRenderer {
    context: WgpuContext,
    core: WgpuRendererCore,
    render_target: Option<HeadlessRenderTarget>,
    observed_error_generation: u64,
}

#[cfg(all(
    not(target_family = "wasm"),
    any(test, feature = "bench-support", feature = "test-support")
))]
impl WgpuHeadlessRenderer {
    pub fn new() -> anyhow::Result<Self> {
        let (context, target_format) = WgpuContext::new_headless()?;
        // The device may be shared with earlier renderers; observe only errors raised from
        // here on, including any while this renderer builds its pipelines.
        let observed_error_generation = context.errors().current_generation();
        let atlas = Arc::new(WgpuAtlas::from_context(&context));
        let core = WgpuRendererCore::new(
            &context,
            atlas,
            target_format,
            wgpu::CompositeAlphaMode::Opaque,
        );

        Ok(Self {
            context,
            core,
            render_target: None,
            observed_error_generation,
        })
    }

    /// The device and queue this renderer draws with (`WgpuContext::device`, `queue`), so
    /// other GPU work (a Canvas 2D backend, a 3D pass) can make textures it shows and
    /// submit work it orders after; on Linux without a display, lavapipe's.
    pub fn context(&self) -> &WgpuContext {
        &self.context
    }

    /// The format it renders in.
    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.core.target_format
    }

    fn ensure_render_target(&mut self, size: Size<DevicePixels>) -> anyhow::Result<()> {
        anyhow::ensure!(
            size.width.0 > 0 && size.height.0 > 0,
            "invalid headless render target size: {size:?}"
        );
        anyhow::ensure!(
            size.width.0 as u32 <= self.core.max_texture_size
                && size.height.0 as u32 <= self.core.max_texture_size,
            "headless render target size {size:?} exceeds maximum texture dimension {}",
            self.core.max_texture_size
        );
        if self
            .render_target
            .as_ref()
            .is_some_and(|target| target.size() == size)
        {
            return Ok(());
        }

        let texture = self
            .core
            .resources
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("headless_render_target"),
                size: wgpu::Extent3d {
                    width: size.width.0 as u32,
                    height: size.height.0 as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.core.target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.render_target = Some(HeadlessRenderTarget { texture, view });
        Ok(())
    }

    fn check_gpu_errors(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.context.device_lost(),
            "GPU device was lost during headless rendering"
        );
        if let Some(error) = self
            .context
            .errors()
            .observe_error(&mut self.observed_error_generation)
        {
            anyhow::bail!("GPU error during headless rendering: {error}");
        }
        Ok(())
    }

    fn render(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.check_gpu_errors()?;
        self.ensure_render_target(size)?;
        let view = self
            .render_target
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Headless render target was not created"))?
            .view
            .clone();
        self.core
            .render_frame(scene, &view, size, false, wgpu::Color::BLACK)?;
        Ok(())
    }

    /// Copies the current render target back to the CPU. Dimensions come from the
    /// target texture itself, so the copy can never disagree with what was rendered.
    fn read_image(&mut self) -> anyhow::Result<image::RgbaImage> {
        let texture = self
            .render_target
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Headless render target was not created"))?
            .texture
            .clone();
        self.read_texture(&texture)
    }

    /// `texture`'s pixels (the render target's format), as RGBA, after the work submitted so far.
    fn read_texture(&mut self, texture: &wgpu::Texture) -> anyhow::Result<image::RgbaImage> {
        let width = texture.width();
        let height = texture.height();
        let bytes_per_row = width
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("Headless render target row size overflowed"))?;
        let padded_bytes_per_row = bytes_per_row
            .checked_next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            .ok_or_else(|| anyhow::anyhow!("Headless padded row size overflowed"))?;
        let buffer_size = u64::from(padded_bytes_per_row)
            .checked_mul(u64::from(height))
            .ok_or_else(|| anyhow::anyhow!("Headless readback buffer size overflowed"))?;
        anyhow::ensure!(
            buffer_size <= self.core.resources.device.limits().max_buffer_size,
            "Headless readback buffer size {buffer_size} exceeds maximum buffer size {}",
            self.core.resources.device.limits().max_buffer_size
        );
        let readback_buffer = self
            .core
            .resources
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("headless_readback_buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
        let mut encoder =
            self.core
                .resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("headless_readback_encoder"),
                });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let submission = self
            .core
            .resources
            .queue
            .submit(std::iter::once(encoder.finish()));
        let (sender, receiver) = std::sync::mpsc::channel();
        readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if sender.send(result).is_err() {
                    log::error!("Headless readback receiver was dropped before mapping completed");
                }
            });
        self.core
            .resources
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .map_err(|error| anyhow::anyhow!("Failed to wait for headless rendering: {error}"))?;
        receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .map_err(|error| anyhow::anyhow!("Failed to receive headless mapping result: {error}"))?
            .map_err(|error| anyhow::anyhow!("Failed to map headless readback buffer: {error}"))?;
        self.check_gpu_errors()?;

        let mapped_data = readback_buffer.slice(..).get_mapped_range()?;
        let pixel_capacity = usize::try_from(u64::from(bytes_per_row) * u64::from(height))
            .map_err(|_| anyhow::anyhow!("Headless image size exceeds addressable memory"))?;
        let mut pixels = Vec::with_capacity(pixel_capacity);
        for row in mapped_data
            .chunks_exact(padded_bytes_per_row as usize)
            .take(height as usize)
        {
            pixels.extend_from_slice(&row[..bytes_per_row as usize]);
        }
        drop(mapped_data);
        readback_buffer.unmap();

        if self.core.target_format == wgpu::TextureFormat::Bgra8Unorm {
            for pixel in pixels.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }

        image::RgbaImage::from_raw(width, height, pixels)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image from headless pixel data"))
    }
}

#[cfg(all(
    not(target_family = "wasm"),
    any(test, feature = "bench-support", feature = "test-support")
))]
impl gpui::PlatformHeadlessRenderer for WgpuHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        self.render(scene, size)?;
        self.read_image()
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.render(scene, size)
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.core.atlas.clone()
    }
}

fn instance_range(range: Range<usize>) -> Range<u32> {
    range.start as u32..range.end as u32
}

#[cfg(not(target_family = "wasm"))]
fn create_surface(
    instance: &wgpu::Instance,
    raw_window_handle: raw_window_handle::RawWindowHandle,
) -> anyhow::Result<wgpu::Surface<'static>> {
    unsafe {
        instance
            .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                // Fall back to the display handle already provided via InstanceDescriptor::display.
                raw_display_handle: None,
                raw_window_handle,
            })
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

struct RenderingParameters {
    path_sample_count: u32,
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
}

impl RenderingParameters {
    fn new(adapter: &wgpu::Adapter, surface_format: wgpu::TextureFormat) -> Self {
        use std::env;

        let format_features = adapter.get_texture_format_features(surface_format);
        let path_sample_count = [4, 2, 1]
            .into_iter()
            .find(|&n| format_features.flags.sample_count_supported(n))
            .unwrap_or(1);

        let gamma = env::var("ZED_FONTS_GAMMA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.8_f32)
            .clamp(1.0, 2.2);
        let gamma_ratios = get_gamma_correction_ratios(gamma);

        let grayscale_enhanced_contrast = env::var("ZED_FONTS_GRAYSCALE_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0_f32)
            .max(0.0);

        let subpixel_enhanced_contrast = env::var("ZED_FONTS_SUBPIXEL_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5_f32)
            .max(0.0);

        Self {
            path_sample_count,
            gamma_ratios,
            grayscale_enhanced_contrast,
            subpixel_enhanced_contrast,
        }
    }
}

/// A render pass that keeps what `view` holds.
/// The pass `pass` holds, or a new one that loads `target`.
fn open_pass<'p>(
    pass: &'p mut Option<wgpu::RenderPass<'static>>,
    encoder: &mut wgpu::CommandEncoder,
    target: DrawTarget<'_>,
) -> &'p mut wgpu::RenderPass<'static> {
    pass.get_or_insert_with(|| target_pass(encoder, target, None, "main_pass_continued"))
}

/// A pass on `target`, cleared to `clear` or loaded, drawing into its size.
fn target_pass(
    encoder: &mut wgpu::CommandEncoder,
    target: DrawTarget<'_>,
    clear: Option<wgpu::Color>,
    label: &str,
) -> wgpu::RenderPass<'static> {
    let mut pass = encoder
        .begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: clear.map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        })
        .forget_lifetime();
    pass.set_viewport(
        0.,
        0.,
        target.size.width.0 as f32,
        target.size.height.0 as f32,
        0.,
        1.,
    );
    pass
}

pub(crate) fn begin_load_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    view: &'a wgpu::TextureView,
    label: &str,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{
        BorderStyle, ColorSpace, ContentMask, Corners, Edges, Hsla, MonochromeSprite,
        PolychromeSprite, Quad, Shadow, Size, SubpixelSprite, TransformationMatrix, Underline,
        linear_color_stop, linear_gradient,
    };
    #[cfg(target_os = "linux")]
    use gpui::{DevicePixels, PlatformHeadlessRenderer, Scene};

    #[cfg(target_os = "linux")]
    fn device_size(width: i32, height: i32) -> Size<DevicePixels> {
        Size {
            width: DevicePixels(width),
            height: DevicePixels(height),
        }
    }

    #[cfg(target_os = "linux")]
    fn solid_quad(x: f32, y: f32, width: f32, height: f32, color: Hsla) -> Quad {
        let bounds = Bounds {
            origin: Point {
                x: x.into(),
                y: y.into(),
            },
            size: Size {
                width: width.into(),
                height: height.into(),
            },
        };
        Quad {
            order: 0,
            border_style: BorderStyle::Solid,
            bounds,
            content_mask: ContentMask::from(bounds),
            background: color.into(),
            border_color: color,
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
            ..Default::default()
        }
    }

    /// Channels are compared with a small tolerance so the assertions hold across
    /// drivers without pinning exact rasterizer output.
    #[cfg(target_os = "linux")]
    fn assert_pixel(image: &image::RgbaImage, x: u32, y: u32, expected: [u8; 4]) {
        let actual = image.get_pixel(x, y).0;
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| actual.abs_diff(expected) <= 3),
            "pixel ({x}, {y}) was {actual:?}, expected {expected:?}"
        );
    }

    #[cfg(target_os = "linux")]
    const RED: [u8; 4] = [255, 0, 0, 255];
    #[cfg(target_os = "linux")]
    const BLUE: [u8; 4] = [0, 0, 255, 255];
    #[cfg(target_os = "linux")]
    const BLACK: [u8; 4] = [0, 0, 0, 255];

    #[cfg(target_os = "linux")]
    #[test]
    fn headless_renderer_draws_quads_with_distinct_colors() -> anyhow::Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let mut scene = Scene::default();
        scene.insert_primitive(solid_quad(0.0, 0.0, 32.0, 32.0, gpui::red()));
        scene.insert_primitive(solid_quad(32.0, 0.0, 32.0, 32.0, gpui::blue()));
        scene.finish();

        let image = renderer.render_scene_to_image(&scene, device_size(64, 32))?;
        assert_eq!(image.dimensions(), (64, 32));
        assert_pixel(&image, 8, 16, RED);
        assert_pixel(&image, 24, 16, RED);
        assert_pixel(&image, 40, 16, BLUE);
        assert_pixel(&image, 56, 16, BLUE);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn headless_renderer_captures_each_requested_size() -> anyhow::Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let mut scene = Scene::default();
        scene.insert_primitive(solid_quad(2.0, 2.0, 4.0, 3.0, gpui::red()));
        scene.finish();

        // 13 px rows are 52 bytes, forcing readback to strip copy-row padding.
        let image = renderer.render_scene_to_image(&scene, device_size(13, 7))?;
        assert_eq!(image.dimensions(), (13, 7));
        assert_pixel(&image, 0, 0, BLACK);
        assert_pixel(&image, 3, 3, RED);
        assert_pixel(&image, 12, 6, BLACK);

        let image = renderer.render_scene_to_image(&scene, device_size(17, 9))?;
        assert_eq!(image.dimensions(), (17, 9));
        assert_pixel(&image, 3, 3, RED);
        assert_pixel(&image, 16, 8, BLACK);

        let image = renderer.render_scene_to_image(&Scene::default(), device_size(13, 7))?;
        assert_eq!(image.dimensions(), (13, 7));
        assert!(image.pixels().all(|pixel| pixel.0 == BLACK));
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn headless_renderer_reuses_target_for_same_size() -> anyhow::Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let target_texture = |renderer: &WgpuHeadlessRenderer| {
            renderer
                .render_target
                .as_ref()
                .map(|target| target.texture.clone())
        };

        renderer.render_scene(&Scene::default(), device_size(16, 16))?;
        let first = target_texture(&renderer);
        assert!(first.is_some());

        renderer.render_scene(&Scene::default(), device_size(16, 16))?;
        assert_eq!(target_texture(&renderer), first);

        renderer.render_scene(&Scene::default(), device_size(16, 17))?;
        assert_ne!(target_texture(&renderer), first);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn headless_renderer_rejects_invalid_sizes() -> anyhow::Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let too_large = renderer.core.max_texture_size as i32 + 1;

        for size in [
            device_size(0, 8),
            device_size(8, 0),
            device_size(-1, 8),
            device_size(too_large, 8),
        ] {
            assert!(
                renderer.render_scene(&Scene::default(), size).is_err(),
                "{size:?} should be rejected"
            );
            assert!(
                renderer
                    .render_scene_to_image(&Scene::default(), size)
                    .is_err(),
                "{size:?} should be rejected"
            );
        }

        // Rejection must leave the renderer usable.
        let image = renderer.render_scene_to_image(&Scene::default(), device_size(4, 4))?;
        assert_eq!(image.dimensions(), (4, 4));
        Ok(())
    }

    /// The shared backdrop filter scene on the wgpu backend.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn backdrop_filters_render() -> anyhow::Result<()> {
        use gpui::backdrop_filter_test_scene as t;
        use gpui::{DevicePixels, PlatformHeadlessRenderer};
        let mut renderer = match WgpuHeadlessRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("skipped: no wgpu adapter ({error:#})");
                return Ok(());
            }
        };
        let (scene, probes) = t::scene();
        let image = renderer.render_scene_to_image(
            &scene,
            Size {
                width: DevicePixels(t::SIZE.0),
                height: DevicePixels(t::SIZE.1),
            },
        )?;
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image.save(std::path::Path::new(&dir).join("backdrop_wgpu.png"))?;
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 12);
        Ok(())
    }

    /// The shared gradient scene on the wgpu backend.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn gradients_render() -> anyhow::Result<()> {
        use gpui::gradient_test_scene as t;
        use gpui::{DevicePixels, PlatformHeadlessRenderer};
        let mut renderer = match WgpuHeadlessRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("skipped: no wgpu adapter ({error:#})");
                return Ok(());
            }
        };
        let (scene, probes) = t::scene();
        let image = renderer.render_scene_to_image(
            &scene,
            Size {
                width: DevicePixels(t::SIZE.0),
                height: DevicePixels(t::SIZE.1),
            },
        )?;
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image.save(std::path::Path::new(&dir).join("gradients_wgpu.png"))?;
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 2);
        Ok(())
    }

    /// Frame costs with and without compositing groups. Run with
    /// `cargo test -p gpui_wgpu --release --lib group_frame_cost -- --ignored --nocapture`.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    #[ignore]
    fn group_frame_cost() -> anyhow::Result<()> {
        use gpui::compositing_group_bench_scene as b;
        use gpui::{DevicePixels, PlatformHeadlessRenderer};
        let (width, height) = (2880, 1800);
        let size = Size {
            width: DevicePixels(width),
            height: DevicePixels(height),
        };
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let flat = b::flat(10_000, width as f32, height as f32);
        let mut encode = Vec::new();
        for _ in 0..200 {
            let start = std::time::Instant::now();
            renderer.render_scene(&flat, size)?;
            encode.push(start.elapsed().as_secs_f64() * 1e3);
        }
        println!("encode 10k quads, no groups: {:.3} ms", median(encode));
        for depth in [0, 1, 10] {
            let scene = b::nested(depth, 2_000, width as f32, height as f32);
            let mut frames = Vec::new();
            for _ in 0..40 {
                let start = std::time::Instant::now();
                renderer.render_scene_to_image(&scene, size)?;
                frames.push(start.elapsed().as_secs_f64() * 1e3);
            }
            let bytes = renderer.core.group_texture_bytes();
            println!(
                "{depth} nested groups, 2k quads, {width}x{height}: {:.3} ms per frame (rendered and read back), {bytes} group image bytes",
                median(frames)
            );
        }
        let cases = [
            (
                "one 300x200 card at 2x",
                b::cards(1, 600., 400., (width as f32, height as f32)),
            ),
            (
                "ten nested 300x200 cards",
                b::cards(10, 600., 400., (width as f32, height as f32)),
            ),
            (
                "one full-screen group",
                b::nested(1, 2_000, width as f32, height as f32),
            ),
        ];
        for (label, scene) in cases {
            let mut frames = Vec::new();
            for _ in 0..40 {
                let start = std::time::Instant::now();
                renderer.render_scene_to_image(&scene, size)?;
                frames.push(start.elapsed().as_secs_f64() * 1e3);
            }
            println!(
                "{label}: {:.3} ms per frame, {} group image bytes",
                median(frames),
                renderer.core.group_texture_bytes()
            );
        }
        Ok(())
    }

    /// Other GPU work runs on the headless renderer's own device: a texture made and drawn
    /// there can be shown by its atlas.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn headless_renderers_share_their_device() -> anyhow::Result<()> {
        use gpui::{ExternalTextureId, PlatformHeadlessRenderer as _, TextureSampling};
        let renderer = match WgpuHeadlessRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("skipped: no wgpu adapter ({error:#})");
                return Ok(());
            }
        };
        let context = renderer.context();
        let atlas = renderer.sprite_atlas();
        // The atlas takes textures in its color format: one of these two.
        let shared = [
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ]
        .into_iter()
        .any(|format| {
            let texture = context.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("producer"),
                size: wgpu::Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let mut encoder = context.device.create_command_encoder(&Default::default());
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            context.queue.submit([encoder.finish()]);
            atlas
                .share_external_texture(ExternalTextureId(1), &texture, TextureSampling::Linear)
                .is_ok()
        });
        assert!(
            shared,
            "a texture made on the renderer's device is shared with its atlas"
        );
        assert!(atlas.external_texture_tile(ExternalTextureId(1)).is_some());
        Ok(())
    }

    /// The shared texture target scene on the wgpu backend: targets draw into their
    /// external textures, not the window.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn texture_targets_draw_into_their_textures_not_the_window() -> anyhow::Result<()> {
        use gpui::texture_target_test_scene as t;
        use gpui::{DevicePixels, PlatformAtlas as _, PlatformHeadlessRenderer};
        let mut renderer = match WgpuHeadlessRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("skipped: no wgpu adapter ({error:#})");
                return Ok(());
            }
        };
        let atlas = renderer.core.atlas.clone();
        for (id, size) in t::targets() {
            atlas.set_external_texture(id, size, gpui::TextureSampling::Linear)?;
        }
        let size = Size {
            width: DevicePixels(t::SIZE.0),
            height: DevicePixels(t::SIZE.1),
        };
        let (scene, window, textures) = t::scene();
        let image = renderer.render_scene_to_image(&scene, size)?;
        t::check(|x, y| image.get_pixel(x, y).0, &window, 3);
        for (id, probes) in textures {
            let texture = atlas.external_texture(id).expect("the texture exists");
            let pixels = renderer.read_texture(&texture)?;
            t::check(|x, y| pixels.get_pixel(x, y).0, &probes, 3);
        }
        Ok(())
    }

    /// The shared compositing group scene on the wgpu backend; images are freed when
    /// frames stop drawing groups.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn compositing_groups_render_and_free_their_images() -> anyhow::Result<()> {
        use gpui::compositing_group_test_scene as t;
        use gpui::{DevicePixels, PlatformHeadlessRenderer};
        let mut renderer = match WgpuHeadlessRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("skipped: no wgpu adapter ({error:#})");
                return Ok(());
            }
        };
        let size = Size {
            width: DevicePixels(t::SIZE.0),
            height: DevicePixels(t::SIZE.1),
        };
        let (scene, probes) = t::scene();
        let image = renderer.render_scene_to_image(&scene, size)?;
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image.save(std::path::Path::new(&dir).join("groups_wgpu.png"))?;
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
        let kept = renderer.core.group_texture_bytes();
        assert!(kept > 0 && kept <= 6 * 64 * 64 * 4, "{kept} bytes");
        let mut empty = Scene::default();
        empty.finish();
        renderer.render_scene_to_image(&empty, size)?;
        assert_eq!(renderer.core.group_texture_bytes(), 0);
        Ok(())
    }

    /// The shared rounded content mask scene on the wgpu backend.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn rounded_clips_render() -> anyhow::Result<()> {
        use gpui::rounded_clip_test_scene as t;
        use gpui::{DevicePixels, PlatformHeadlessRenderer};
        let mut renderer = match WgpuHeadlessRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("skipped: no wgpu adapter ({error:#})");
                return Ok(());
            }
        };
        let (scene, probes) = t::scene();
        let image = renderer.render_scene_to_image(
            &scene,
            Size {
                width: DevicePixels(t::SIZE.0),
                height: DevicePixels(t::SIZE.1),
            },
        )?;
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image.save(std::path::Path::new(&dir).join("rounded_clips_wgpu.png"))?;
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
        Ok(())
    }

    /// The shared shaped-corner scene (CSS `corner-shape`) on the wgpu backend.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn corner_shapes_render() -> anyhow::Result<()> {
        use gpui::corner_shape_test_scene as t;
        use gpui::{DevicePixels, PlatformHeadlessRenderer};
        let mut renderer = match WgpuHeadlessRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("skipped: no wgpu adapter ({error:#})");
                return Ok(());
            }
        };
        let (scene, probes) = t::scene();
        let image = renderer.render_scene_to_image(
            &scene,
            Size {
                width: DevicePixels(t::SIZE.0),
                height: DevicePixels(t::SIZE.1),
            },
        )?;
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image.save(std::path::Path::new(&dir).join("corner_shapes_wgpu.png"))?;
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
        Ok(())
    }

    /// The shared group mask scene (alpha and luminance masks, a mask moving with an
    /// offscreen group, a smaller mask, an empty one) on the wgpu backend.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn group_masks_render() -> anyhow::Result<()> {
        use gpui::group_mask_test_scene as t;
        use gpui::{DevicePixels, PlatformHeadlessRenderer};
        let mut renderer = match WgpuHeadlessRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("skipped: no wgpu adapter ({error:#})");
                return Ok(());
            }
        };
        let (scene, probes) = t::scene();
        let image = renderer.render_scene_to_image(
            &scene,
            Size {
                width: DevicePixels(t::SIZE.0),
                height: DevicePixels(t::SIZE.1),
            },
        )?;
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image.save(std::path::Path::new(&dir).join("group_masks_wgpu.png"))?;
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
        Ok(())
    }

    /// The shared group content scene (a path and a turned square in an offset group)
    /// on the wgpu backend.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn group_content_draws_in_image_coordinates() -> anyhow::Result<()> {
        use gpui::compositing_group_content_test_scene as t;
        use gpui::{DevicePixels, PlatformHeadlessRenderer};
        let mut renderer = match WgpuHeadlessRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("skipped: no wgpu adapter ({error:#})");
                return Ok(());
            }
        };
        let (scene, probes) = t::scene();
        let image = renderer.render_scene_to_image(
            &scene,
            Size {
                width: DevicePixels(t::SIZE.0),
                height: DevicePixels(t::SIZE.1),
            },
        )?;
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image.save(std::path::Path::new(&dir).join("group_content_wgpu.png"))?;
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
        Ok(())
    }

    /// Every blend mode against the CPU reference on the wgpu backend.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn blend_modes_match_the_reference() -> anyhow::Result<()> {
        use gpui::blend_mode_test_scene as t;
        use gpui::{DevicePixels, PlatformHeadlessRenderer};
        let mut renderer = match WgpuHeadlessRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("skipped: no wgpu adapter ({error:#})");
                return Ok(());
            }
        };
        let (scene, probes) = t::scene();
        let image = renderer.render_scene_to_image(
            &scene,
            Size {
                width: DevicePixels(t::SIZE.0),
                height: DevicePixels(t::SIZE.1),
            },
        )?;
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image.save(std::path::Path::new(&dir).join("blend_wgpu.png"))?;
        }
        let failures: Vec<String> = probes
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let actual = image.get_pixel(p.x, p.y).0;
                let ok = actual.iter().zip(p.rgba).all(|(a, e)| a.abs_diff(e) <= 3);
                (!ok).then(|| format!("{}: {actual:?}, expected {:?}", t::describe(i), p.rgba))
            })
            .collect();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        Ok(())
    }

    /// The shared custom shader scene on the wgpu backend.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn custom_shaders_render() -> anyhow::Result<()> {
        use gpui::custom_shader_test_scene as t;
        use gpui::{DevicePixels, PlatformHeadlessRenderer};
        let mut renderer = match WgpuHeadlessRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("skipped: no wgpu adapter ({error:#})");
                return Ok(());
            }
        };
        let (scene, probes) = t::scene();
        let image = renderer.render_scene_to_image(
            &scene,
            Size {
                width: DevicePixels(t::SIZE.0),
                height: DevicePixels(t::SIZE.1),
            },
        )?;
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image.save(std::path::Path::new(&dir).join("custom_shaders_wgpu.png"))?;
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 4);
        Ok(())
    }

    /// The shared marks scene on the wgpu backend, over a storage buffer on the
    /// renderer's device; Metal draws the same probes.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn marks_render() -> anyhow::Result<()> {
        use gpui::marks_test_scene as t;
        use gpui::{DevicePixels, PlatformHeadlessRenderer};
        use wgpu::util::DeviceExt as _;
        let mut renderer = match WgpuHeadlessRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("skipped: no wgpu adapter ({error:#})");
                return Ok(());
            }
        };
        let buffer =
            renderer
                .context()
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("marks_test"),
                    contents: &t::buffer_bytes(),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let (scene, probes) = t::scene(gpui::MarksBuffer::new(std::sync::Arc::new(buffer)));
        let image = renderer.render_scene_to_image(
            &scene,
            Size {
                width: DevicePixels(t::SIZE.0),
                height: DevicePixels(t::SIZE.1),
            },
        )?;
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image.save(std::path::Path::new(&dir).join("marks_wgpu.png"))?;
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
        Ok(())
    }

    /// A program wgpu cannot build says so ([`gpui::MarksShader::compile_error`],
    /// and a panic in debug builds), and so does a draw bound to a buffer
    /// smaller than its program reads.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn marks_failures_are_reported() -> anyhow::Result<()> {
        use gpui::marks_test_scene as t;
        use gpui::{DevicePixels, PlatformHeadlessRenderer};
        use wgpu::util::DeviceExt as _;
        let px = Size { width: DevicePixels(t::SIZE.0), height: DevicePixels(t::SIZE.1) };
        let new = || WgpuHeadlessRenderer::new();
        let mut renderer = match new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("skipped: no wgpu adapter ({error:#})");
                return Ok(());
            }
        };
        let buffer = |renderer: &WgpuHeadlessRenderer, bytes: &[u8]| {
            let buffer = renderer.context().device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("marks_test"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
            gpui::MarksBuffer::new(std::sync::Arc::new(buffer))
        };
        let broken = t::broken("wgpu");
        let full = buffer(&renderer, &t::buffer_bytes());
        t::expect_report(|| {
            let _ = renderer.render_scene_to_image(&t::one_draw(broken.clone(), full), px);
        });
        let error = broken.compile_error().expect("the failure is recorded");
        assert!(error.contains("target"), "{error}");

        let mut renderer = new()?;
        let small = buffer(&renderer, &[0u8; 16]);
        let rects = t::rects("wgpu small buffer");
        t::expect_report(|| {
            let _ = renderer.render_scene_to_image(&t::one_draw(rects.clone(), small), px);
        });
        let (skipped, why) = rects.skipped_draws();
        assert!(skipped >= 1 && why.as_deref().is_some_and(|w| w.contains("32 bytes")), "{skipped} {why:?}");
        Ok(())
    }

    /// The shared transformed scene on the wgpu backend; it must also match Metal's
    /// rendering pixel for pixel within the probe tolerance.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn element_transformations_render() -> anyhow::Result<()> {
        use gpui::element_transformation_test_scene as t;
        use gpui::{DevicePixels, PlatformHeadlessRenderer};
        let mut renderer = match WgpuHeadlessRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("skipped: no wgpu adapter ({error:#})");
                return Ok(());
            }
        };
        let (scene, probes) = t::scene();
        let image = renderer.render_scene_to_image(
            &scene,
            Size {
                width: DevicePixels(t::SIZE.0),
                height: DevicePixels(t::SIZE.1),
            },
        )?;
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image.save(std::path::Path::new(&dir).join("transformed_wgpu.png"))?;
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
        Ok(())
    }

    /// The shared retained-shape scene, rendered by the wgpu backend (Metal on
    /// macOS, Vulkan or GL on Linux). Skips when no adapter is available.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn retained_shapes_render_upload_ranges_and_match_metal() -> anyhow::Result<()> {
        use gpui::retained_shapes_test_scene as t;
        use gpui::{DevicePixels, PlatformHeadlessRenderer};
        let mut renderer = match WgpuHeadlessRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("skipped: no wgpu adapter ({error:#})");
                return Ok(());
            }
        };
        let size = |(w, h): (i32, i32)| Size {
            width: DevicePixels(w),
            height: DevicePixels(h),
        };
        let shapes = gpui::RetainedShapes::new();
        let (scene, probes) = t::scene(&shapes);
        let image = renderer.render_scene_to_image(&scene, size(t::SIZE))?;
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
        let uploaded = renderer.core.resources.retained.uploaded_bytes;
        assert_eq!(uploaded, 5 * 64);

        // Same scene again: nothing uploads. A range write uploads one record.
        renderer.render_scene_to_image(&scene, size(t::SIZE))?;
        assert_eq!(renderer.core.resources.retained.uploaded_bytes, uploaded);
        let mut first = t::instances()[0];
        first.fill = 0x0000ffff;
        shapes.write(0, &[first]);
        let image = renderer.render_scene_to_image(&scene, size(t::SIZE))?;
        assert_eq!(
            renderer.core.resources.retained.uploaded_bytes,
            uploaded + 64
        );
        let blue = image.get_pixel(16, 16).0;
        assert!(
            blue[2] > 250 && blue[0] < 3,
            "range write not drawn: {blue:?}"
        );

        // Arrow parts and text bars.
        let kinds = gpui::RetainedShapes::new();
        let (scene, probes) = t::kinds_scene(&kinds);
        let image = renderer.render_scene_to_image(&scene, size(t::SIZE))?;
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);

        // Four corner radii.
        let corners = gpui::RetainedShapes::new();
        let (scene, probes) = t::corners_scene(&corners);
        let image = renderer.render_scene_to_image(&scene, size(t::SIZE))?;
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);

        // Rounded, rotated and layer-moved clips.
        let clipped = gpui::RetainedShapes::new();
        let (scene, probes) = t::clip_scene(&clipped);
        let image = renderer.render_scene_to_image(&scene, size(t::SIZE))?;
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);

        // The dense scene, kept for the Metal comparison below.
        let dense = gpui::RetainedShapes::new();
        let (scene, target) = t::random_scene(&dense, 400, 7);
        let image = renderer.render_scene_to_image(&scene, size(target))?;
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            let dir = std::path::Path::new(&dir);
            image.save(dir.join("retained_random_wgpu.png"))?;
            // Written by gpui_apple's `retained_shapes_render_and_upload_only_changes`.
            let metal = dir.join("retained_random_metal.png");
            if metal.exists() {
                let metal = image::open(metal)?.to_rgba8();
                let (mut worst, mut off) = (0u8, 0usize);
                for (a, b) in image.pixels().zip(metal.pixels()) {
                    let d =
                        a.0.iter()
                            .zip(b.0)
                            .map(|(x, y)| x.abs_diff(y))
                            .max()
                            .unwrap_or(0);
                    worst = worst.max(d);
                    off += usize::from(d > 8);
                }
                let total = (image.width() * image.height()) as usize;
                println!(
                    "RETAINED_COMPARE wgpu vs metal: max channel diff {worst}, pixels over 8: {off} of {total}"
                );
                assert!(
                    off * 1000 <= total,
                    "{off} of {total} pixels differ by more than 8"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn webgl_shader_is_valid_wgsl_without_storage_buffers() {
        assert!(!WEBGL_SHADERS.contains("var<storage"));
        validate_wgsl(WEBGL_SHADERS, naga::valid::Capabilities::empty());
    }

    #[test]
    fn storage_buffer_shader_is_valid_wgsl() {
        validate_wgsl(STORAGE_BUFFER_SHADERS, naga::valid::Capabilities::empty());
        validate_wgsl(
            include_str!("shaders_backdrop.wgsl"),
            naga::valid::Capabilities::empty(),
        );
    }

    #[test]
    fn subpixel_shader_is_valid_wgsl() {
        validate_wgsl(
            SUBPIXEL_SHADERS,
            naga::valid::Capabilities::DUAL_SOURCE_BLENDING,
        );
    }

    fn validate_wgsl(source: &str, capabilities: naga::valid::Capabilities) {
        let module = naga::front::wgsl::parse_str(source).expect("shader should parse");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
            .validate(&module)
            .expect("shader should validate");
    }

    #[test]
    fn webgl_record_sizes_match_shader_word_strides() {
        assert_eq!(std::mem::size_of::<Quad>(), 52 * 4);
        assert_eq!(std::mem::size_of::<Shadow>(), 38 * 4);
        assert_eq!(std::mem::size_of::<PathRasterizationVertex>(), 34 * 4);
        assert_eq!(std::mem::size_of::<PathSprite>(), 4 * 4);
        assert_eq!(std::mem::size_of::<Underline>(), 26 * 4);
        assert_eq!(std::mem::size_of::<MonochromeSprite>(), 32 * 4);
        assert_eq!(std::mem::size_of::<SubpixelSprite>(), 32 * 4);
        assert_eq!(std::mem::size_of::<PolychromeSprite>(), 34 * 4);
    }

    #[test]
    fn webgl_quad_layout_matches_fixed_decoder() {
        let quad = Quad {
            order: 41,
            border_style: BorderStyle::Dashed,
            bounds: Bounds {
                origin: Point {
                    x: 1.0.into(),
                    y: 2.0.into(),
                },
                size: Size {
                    width: 3.0.into(),
                    height: 4.0.into(),
                },
            },
            content_mask: ContentMask {
                bounds: Bounds {
                    origin: Point {
                        x: 5.0.into(),
                        y: 6.0.into(),
                    },
                    size: Size {
                        width: 7.0.into(),
                        height: 8.0.into(),
                    },
                },
                corner_radii: Corners {
                    top_left: 90.0.into(),
                    top_right: 91.0.into(),
                    bottom_right: 92.0.into(),
                    bottom_left: 93.0.into(),
                },
            },
            background: linear_gradient(
                11.0,
                linear_color_stop(
                    Hsla {
                        h: 12.0,
                        s: 13.0,
                        l: 14.0,
                        a: 15.0,
                    },
                    16.0,
                ),
                linear_color_stop(
                    Hsla {
                        h: 17.0,
                        s: 18.0,
                        l: 19.0,
                        a: 20.0,
                    },
                    21.0,
                ),
            )
            .color_space(ColorSpace::Oklab),
            border_color: Hsla {
                h: 22.0,
                s: 23.0,
                l: 24.0,
                a: 25.0,
            },
            corner_radii: Corners {
                top_left: 26.0.into(),
                top_right: 27.0.into(),
                bottom_right: 28.0.into(),
                bottom_left: 29.0.into(),
            },
            border_widths: Edges {
                top: 30.0.into(),
                right: 31.0.into(),
                bottom: 32.0.into(),
                left: 33.0.into(),
            },
            transformation: TransformationMatrix {
                rotation_scale: [[34.0, 35.0], [36.0, 37.0]],
                translation: [38.0, 39.0],
            },
            corner_shapes: [0; 2],
        };

        let bytes = unsafe { WgpuRendererCore::instance_bytes(std::slice::from_ref(&quad)) };
        let words: &[u32] = bytemuck::cast_slice(bytes);
        assert_eq!(
            words,
            &[
                41,
                1,
                1.0_f32.to_bits(),
                2.0_f32.to_bits(),
                3.0_f32.to_bits(),
                4.0_f32.to_bits(),
                5.0_f32.to_bits(),
                6.0_f32.to_bits(),
                7.0_f32.to_bits(),
                8.0_f32.to_bits(),
                90.0_f32.to_bits(),
                91.0_f32.to_bits(),
                92.0_f32.to_bits(),
                93.0_f32.to_bits(),
                1,
                1,
                0,
                0,
                0,
                0,
                11.0_f32.to_bits(),
                12.0_f32.to_bits(),
                13.0_f32.to_bits(),
                14.0_f32.to_bits(),
                15.0_f32.to_bits(),
                16.0_f32.to_bits(),
                17.0_f32.to_bits(),
                18.0_f32.to_bits(),
                19.0_f32.to_bits(),
                20.0_f32.to_bits(),
                21.0_f32.to_bits(),
                0,
                22.0_f32.to_bits(),
                23.0_f32.to_bits(),
                24.0_f32.to_bits(),
                25.0_f32.to_bits(),
                26.0_f32.to_bits(),
                27.0_f32.to_bits(),
                28.0_f32.to_bits(),
                29.0_f32.to_bits(),
                30.0_f32.to_bits(),
                31.0_f32.to_bits(),
                32.0_f32.to_bits(),
                33.0_f32.to_bits(),
                34.0_f32.to_bits(),
                35.0_f32.to_bits(),
                36.0_f32.to_bits(),
                37.0_f32.to_bits(),
                38.0_f32.to_bits(),
                39.0_f32.to_bits(),
                0,
                0,
            ]
        );
    }
}
