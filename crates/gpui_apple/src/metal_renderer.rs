use crate::metal_atlas::MetalAtlas;
use anyhow::{Context as _, Result};
use block2::RcBlock;
use core_graphics::geometry::CGSize;
use gpui::{
    AtlasTextureId, BackdropFilter, Background, Bounds, ContentMask, DevicePixels, PaintSurface,
    Path, PlatformAtlas as _, Point, PrimitiveBatch, RetainedShapesDraw, RetainedUpload,
    ScaledPixels, Scene, ShapeInstance, Size, point, size,
};
#[cfg(any(test, feature = "bench-support", feature = "test-support", feature = "window-capture"))]
use image::RgbaImage;
use objc2::runtime::AnyObject;
use smallvec::SmallVec;
use std::collections::HashMap;

use core_graphics::color_space::{CGColorSpace, kCGColorSpaceDisplayP3};
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    CAMetalLayer, CommandQueue, MTLGPUFamily, MTLPixelFormat, MTLResourceOptions, NSRange,
    NSUInteger,
};
use objc2_core_foundation::CFRetained;
use objc2_core_video::{
    CVMetalTexture, CVMetalTextureCache, CVMetalTextureGetTexture, CVPixelBufferGetHeight,
    CVPixelBufferGetHeightOfPlane, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
    CVPixelBufferGetWidthOfPlane, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVReturnSuccess,
};
use parking_lot::Mutex;

use std::{cell::Cell, ffi::c_void, mem, mem::MaybeUninit, ops::Range, ptr, slice, sync::Arc};

// Exported to metal
pub(crate) type PointF = gpui::Point<f32>;

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;
/// Metal requires the offset a buffer is bound at to be 256-byte aligned.
const INSTANCE_BUFFER_ALIGNMENT: usize = 256;
const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;

pub type Context = Arc<Mutex<InstanceBufferPool>>;
pub type Renderer = MetalRenderer;

pub unsafe fn new_renderer(
    context: self::Context,
    _native_window: *mut c_void,
    _native_view: *mut c_void,
    _bounds: gpui::Size<f32>,
    transparent: bool,
    use_display_p3: bool,
) -> Renderer {
    MetalRenderer::new(context, transparent, use_display_p3)
}

pub struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<metal::Buffer>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

pub(crate) struct InstanceBuffer {
    metal_buffer: metal::Buffer,
    size: usize,
}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    pub(crate) fn acquire(
        &mut self,
        device: &metal::Device,
        unified_memory: bool,
    ) -> InstanceBuffer {
        let buffer = self.buffers.pop().unwrap_or_else(|| {
            let options = if unified_memory {
                MTLResourceOptions::StorageModeShared
                    // Buffers are write only which can benefit from the combined cache
                    // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            };

            device.new_buffer(self.buffer_size as u64, options)
        });
        InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        }
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer)
        }
    }
}

/// The Metal device GPUI's windows render with: on Intel Macs the low-power
/// integrated GPU when there is one (on Apple silicon and iOS there is only
/// one). Code that hands GPUI textures (external textures) makes them on it.
pub fn preferred_device() -> metal::Device {
    #[cfg(target_os = "macos")]
    if let Some(d) = metal::Device::all()
        .into_iter()
        .min_by_key(|d| (d.is_removable(), !d.is_low_power()))
    {
        return d;
    }
    // For some reason `all()` can return an empty list, see https://github.com/zed-industries/zed/issues/37689
    // In that case, we fall back to the system default device.
    #[cfg(target_os = "macos")]
    log::error!("Unable to enumerate Metal devices; attempting to use system default device");
    metal::Device::system_default().unwrap_or_else(|| {
        log::error!("unable to access a compatible graphics device");
        std::process::exit(1);
    })
}

/// The command queue every GPUI renderer and atlas on `device` encodes on: one per device,
/// made on first use. Code that renders into textures GPUI samples (external textures) encodes
/// on it too, so its command buffers run in commit order with GPUI's frames: a texture written
/// before a frame commits is ready for that frame, with no fence and no CPU wait.
pub fn shared_command_queue(device: &metal::DeviceRef) -> metal::CommandQueue {
    static QUEUES: std::sync::Mutex<Vec<(u64, SendQueue)>> = std::sync::Mutex::new(Vec::new());
    let mut queues = QUEUES.lock().unwrap_or_else(|e| e.into_inner());
    let id = device.registry_id();
    if let Some((_, q)) = queues.iter().find(|(d, _)| *d == id) {
        return q.0.clone();
    }
    // Room for as many uncompleted command buffers as wgpu allows: wgpu on this queue opens
    // one command buffer per pass, and `newCommandQueue`'s limit of 64 would block a large
    // submission in `commandBuffer` until earlier ones complete (a deadlock when they wait
    // on work committed later).
    let queue = device.new_command_queue_with_max_command_buffer_count(MAX_COMMAND_BUFFERS);
    queues.push((id, SendQueue(queue.clone())));
    queue
}

/// Uncompleted command buffers [`shared_command_queue`] allows: wgpu-hal's Metal
/// `MAX_COMMAND_BUFFERS`.
pub const MAX_COMMAND_BUFFERS: u64 = 4096;

/// Metal command queues are thread-safe (`MTLCommandQueue` may be used from any thread).
struct SendQueue(metal::CommandQueue);
// SAFETY: see above.
unsafe impl Send for SendQueue {}

pub struct MetalRenderer {
    device: metal::Device,
    layer: Option<metal::MetalLayer>,
    is_apple_gpu: bool,
    is_unified_memory: bool,
    presents_with_transaction: bool,
    /// For headless rendering, tracks whether output should be opaque
    opaque: bool,
    command_queue: CommandQueue,
    paths_rasterization_pipeline_state: metal::RenderPipelineState,
    path_sprites_pipeline_state: metal::RenderPipelineState,
    shadows_pipeline_state: metal::RenderPipelineState,
    quads_pipeline_state: metal::RenderPipelineState,
    underlines_pipeline_state: metal::RenderPipelineState,
    monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    surfaces_pipeline_state: metal::RenderPipelineState,
    retained_shapes_pipeline_state: metal::RenderPipelineState,
    backdrop_copy_pipeline_state: metal::RenderPipelineState,
    backdrop_blur_pipeline_state: metal::RenderPipelineState,
    backdrop_composite_pipeline_state: metal::RenderPipelineState,
    /// Composite a group's image over the target, adding to it (plus-lighter) and
    /// replacing it (plus-darker).
    group_composite_pipeline_states: GroupCompositePipelines,
    /// Draws a group's backdrop into its (transparent) image.
    backdrop_group_pipeline_state: metal::RenderPipelineState,
    /// Group images and their blur and backdrop copies, reused across frames and
    /// freed when a frame does not use them.
    texture_pool: TexturePool,
    /// Pipelines of custom shaders by id; `None` for one that failed to compile
    /// (logged once).
    custom_shaders: HashMap<gpui::ShaderId, Option<metal::RenderPipelineState>>,
    /// Pipelines of marks programs by id; `None` for one that failed to compile
    /// (logged once).
    marks_pipelines: HashMap<gpui::MarksShaderId, Option<crate::metal_marks::MarksPipeline>>,
    /// Samples a custom shader's backdrop.
    custom_shader_sampler: metal::SamplerState,
    /// Polychrome sprites: linear, and nearest for external textures that ask for it.
    sprite_samplers: [metal::SamplerState; 2],
    /// The backdrop of a custom shader that reads none.
    blank_texture: metal::Texture,
    /// GPU copies of retained shape buffers, by buffer id.
    retained_shapes: HashMap<u64, RetainedGpu>,
    frame_index: u64,
    /// Bytes of retained shape instances uploaded so far (for tests and stats).
    retained_uploaded_bytes: u64,
    unit_vertices: metal::Buffer,
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: CFRetained<CVMetalTextureCache>,
    path_intermediate_texture: Option<metal::Texture>,
    path_intermediate_msaa_texture: Option<metal::Texture>,
    path_sample_count: u32,
    /// Offscreen render target reused across `render_scene` calls when
    /// rendering headlessly without reading pixels back.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    headless_render_target: Option<metal::Texture>,
    /// Whether the last [`MetalRenderer::draw`] committed its frame.
    rendered: bool,
}

#[repr(C)]
pub struct PathRasterizationVertex {
    pub xy_position: Point<ScaledPixels>,
    pub st_position: Point<f32>,
    pub color: Background,
    pub bounds: Bounds<ScaledPixels>,
    /// The path's content mask, for a rounded clip (`bounds` is already clipped to it).
    pub content_mask: ContentMask<ScaledPixels>,
}

impl MetalRenderer {
    /// Creates a new MetalRenderer with a CAMetalLayer for window-based rendering.
    pub fn new(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        transparent: bool,
        use_display_p3: bool,
    ) -> Self {
        let device = Self::create_device();
        let layer = metal::MetalLayer::new();
        Self::configure_layer(&layer, &device, transparent, use_display_p3);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    /// Creates a renderer for a CAMetalLayer owned by a platform view, such as
    /// the backing layer UIKit creates for a view whose `layerClass` is
    /// `CAMetalLayer`. The renderer retains the layer.
    pub fn from_layer(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        layer: &objc2_quartz_core::CAMetalLayer,
        transparent: bool,
    ) -> Self {
        let device = Self::create_device();
        // Both types bind the same Objective-C class, so this only changes which
        // Rust wrapper views the live layer. `to_owned` retains it.
        let layer = unsafe {
            metal::MetalLayerRef::from_ptr(ptr::from_ref(layer).cast_mut().cast::<CAMetalLayer>())
        }
        .to_owned();
        Self::configure_layer(&layer, &device, transparent, false);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    fn configure_layer(
        layer: &metal::MetalLayerRef,
        device: &metal::DeviceRef,
        transparent: bool,
        use_display_p3: bool,
    ) {
        layer.set_device(device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        // metal-rs doesn't bind these setters, so view the same object through
        // objc2's typed CAMetalLayer binding.
        let objc2_layer: &objc2_quartz_core::CAMetalLayer = unsafe { &*layer.as_ptr().cast() };
        if use_display_p3 {
            // SAFETY: The constant is extern, from Core Graphics
            let display_p3_name = unsafe { kCGColorSpaceDisplayP3 };
            if let Some(display_p3_color_space) = CGColorSpace::create_with_name(display_p3_name) {
                // SAFETY: `layer` is a CAMetalLayer and the color space remains valid for the call.
                // 'metal-rs' does not expose CAMetalLayer's typed `colorspace` setter.
                // Could be replaced with a native binding via objc2-* in future
                unsafe {
                    let _: () = objc2::msg_send![
                        objc2_layer,
                        setColorspace: display_p3_color_space.as_ptr().cast::<c_void>()
                    ];
                }
            } else {
                log::error!("unable to create the Display P3 color space");
            }
        }
        // Support direct-to-display rendering if the window is not transparent
        // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
        layer.set_opaque(!transparent);
        layer.set_maximum_drawable_count(3);
        // Allow texture reading for visual tests and UI automation screenshots
        // (captures without ScreenCaptureKit). Only debug builds pay the
        // presentation cost, even when `test-support` is compiled in.
        #[cfg(all(feature = "test-support", debug_assertions))]
        layer.set_framebuffer_only(false);
        objc2_layer.setAllowsNextDrawableTimeout(false);
        objc2_layer.setNeedsDisplayOnBoundsChange(true);
        // UIKit sizes a view's backing layer itself; only AppKit-hosted
        // layers need to track their superlayer's bounds.
        #[cfg(target_os = "macos")]
        objc2_layer.setAutoresizingMask(
            objc2_quartz_core::CAAutoresizingMask::LayerWidthSizable
                | objc2_quartz_core::CAAutoresizingMask::LayerHeightSizable,
        );
    }

    /// Creates a new headless MetalRenderer for offscreen rendering without a window.
    ///
    /// This renderer can render scenes to images without requiring a CAMetalLayer,
    /// window, or AppKit. Use `render_scene_to_image()` to render scenes.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn new_headless(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>) -> Self {
        let device = Self::create_device();
        Self::new_internal(device, None, true, instance_buffer_pool)
    }

    fn create_device() -> metal::Device {
        preferred_device()
    }

    fn new_internal(
        device: metal::Device,
        layer: Option<metal::MetalLayer>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    ) -> Self {
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .new_library_with_source(&SHADERS_SOURCE_FILE, &metal::CompileOptions::new())
            .expect("error building metal library");
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .new_library_with_data(SHADERS_METALLIB)
            .expect("error building metal library");

        fn to_float2_bits(point: PointF) -> u64 {
            let mut output = point.y.to_bits() as u64;
            output <<= 32;
            output |= point.x.to_bits() as u64;
            output
        }

        // Shared memory can be used only if CPU and GPU share the same memory space.
        // https://developer.apple.com/documentation/metal/setting-resource-storage-modes
        // iOS does not support managed resources. Its simulator may report a
        // non-unified host GPU even though resources must still use shared storage.
        let is_unified_memory = cfg!(target_os = "ios") || device.has_unified_memory();
        // Apple GPU families support memoryless textures, which can significantly reduce
        // memory usage by keeping render targets in on-chip tile memory instead of
        // allocating backing store in system memory.
        // https://developer.apple.com/documentation/metal/mtlgpufamily
        let is_apple_gpu = device.supports_family(MTLGPUFamily::Apple1);

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        let unit_vertices = device.new_buffer_with_data(
            unit_vertices.as_ptr() as *const c_void,
            mem::size_of_val(&unit_vertices) as u64,
            if is_unified_memory {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            },
        );

        let custom_shader_sampler = {
            let descriptor = metal::SamplerDescriptor::new();
            descriptor.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
            descriptor.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
            descriptor.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
            descriptor.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
            device.new_sampler(&descriptor)
        };
        let sprite_samplers = [metal::MTLSamplerMinMagFilter::Linear, metal::MTLSamplerMinMagFilter::Nearest].map(|filter| {
            let descriptor = metal::SamplerDescriptor::new();
            descriptor.set_min_filter(filter);
            descriptor.set_mag_filter(filter);
            device.new_sampler(&descriptor)
        });
        let blank_texture = {
            let descriptor = metal::TextureDescriptor::new();
            descriptor.set_width(1);
            descriptor.set_height(1);
            descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
            descriptor.set_usage(metal::MTLTextureUsage::ShaderRead);
            let texture = device.new_texture(&descriptor);
            texture.replace_region(
                metal::MTLRegion::new_2d(0, 0, 1, 1),
                0,
                [0u8; 4].as_ptr() as *const _,
                4,
            );
            texture
        };
        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        );
        let backdrop_copy_pipeline_state = build_opaque_pipeline_state(
            &device,
            &library,
            "backdrop_copy",
            "backdrop_fullscreen_vertex",
            "backdrop_copy_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let backdrop_blur_pipeline_state = build_opaque_pipeline_state(
            &device,
            &library,
            "backdrop_blur",
            "backdrop_fullscreen_vertex",
            "backdrop_blur_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let backdrop_composite_pipeline_state =
            build_backdrop_composite_pipeline_state(&device, &library, MTLPixelFormat::BGRA8Unorm);
        let group_composite_pipeline_states =
            GroupCompositePipelines::new(&device, &library, MTLPixelFormat::BGRA8Unorm);
        let backdrop_group_pipeline_state =
            build_backdrop_group_pipeline_state(&device, &library, MTLPixelFormat::BGRA8Unorm);
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );

        let retained_shapes_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "retained_shapes",
            "retained_shape_vertex",
            "retained_shape_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );

        let command_queue = shared_command_queue(&device);
        let supports_shared_storage = cfg!(target_os = "ios") || is_apple_gpu;
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), supports_shared_storage));
        // metal and objc2-metal expose the same Objective-C device; the borrowed
        // protocol object remains valid while `device` owns it.
        let core_video_texture_cache = unsafe {
            let mut cache = ptr::null_mut();
            let status = CVMetalTextureCache::create(
                None,
                None,
                &*device
                    .as_ptr()
                    .cast::<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>(),
                None,
                ptr::NonNull::from(&mut cache),
            );
            assert_eq!(
                status, kCVReturnSuccess,
                "failed to create CoreVideo Metal texture cache"
            );
            CFRetained::from_raw(ptr::NonNull::new(cache).expect("CoreVideo returned a null cache"))
        };

        Self {
            device,
            layer,
            presents_with_transaction: false,
            is_apple_gpu,
            is_unified_memory,
            opaque,
            command_queue,
            paths_rasterization_pipeline_state,
            path_sprites_pipeline_state,
            shadows_pipeline_state,
            quads_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            surfaces_pipeline_state,
            retained_shapes_pipeline_state,
            backdrop_copy_pipeline_state,
            backdrop_blur_pipeline_state,
            backdrop_composite_pipeline_state,
            group_composite_pipeline_states,
            backdrop_group_pipeline_state,
            texture_pool: TexturePool::default(),
            custom_shaders: HashMap::new(),
            marks_pipelines: HashMap::new(),
            custom_shader_sampler,
            sprite_samplers,
            blank_texture,
            retained_shapes: HashMap::new(),
            frame_index: 0,
            retained_uploaded_bytes: 0,
            unit_vertices,
            instance_buffer_pool,
            sprite_atlas,
            core_video_texture_cache,
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            path_sample_count: PATH_SAMPLE_COUNT,
            #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
            headless_render_target: None,
            rendered: false,
        }
    }

    pub fn layer(&self) -> Option<&metal::MetalLayerRef> {
        self.layer.as_ref().map(|l| l.as_ref())
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer
            .as_ref()
            .map(|l| l.as_ptr())
            .unwrap_or(ptr::null_mut())
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        if let Some(layer) = &self.layer {
            layer.set_presents_with_transaction(presents_with_transaction);
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        if let Some(layer) = &self.layer {
            layer.set_drawable_size(CGSize::new(size.width.0 as f64, size.height.0 as f64));
        }
        self.update_path_intermediate_textures(size);
    }

    fn update_path_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        // We are uncertain when this happens, but sometimes size can be 0 here. Most likely before
        // the layout pass on window creation. Zero-sized texture creation causes SIGABRT.
        // https://github.com/zed-industries/zed/issues/36229
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.path_intermediate_texture = None;
            self.path_intermediate_msaa_texture = None;
            return;
        }

        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        self.path_intermediate_texture = Some(self.device.new_texture(&texture_descriptor));

        if self.path_sample_count > 1 {
            // https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
            // Rendering MSAA textures are done in a single pass, so we can use memory-less storage on Apple Silicon
            let storage_mode = if self.is_apple_gpu {
                metal::MTLStorageMode::Memoryless
            } else {
                metal::MTLStorageMode::Private
            };

            let msaa_descriptor = texture_descriptor;
            msaa_descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
            msaa_descriptor.set_storage_mode(storage_mode);
            msaa_descriptor.set_sample_count(self.path_sample_count as _);
            self.path_intermediate_msaa_texture = Some(self.device.new_texture(&msaa_descriptor));
        } else {
            self.path_intermediate_msaa_texture = None;
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.opaque = !transparent;
        if let Some(layer) = &self.layer {
            layer.set_opaque(!transparent);
        }
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    /// Whether the last [`MetalRenderer::draw`] committed its frame: false when it found no
    /// layer or drawable, or failed to encode.
    pub fn rendered(&self) -> bool {
        self.rendered
    }

    pub fn draw(&mut self, scene: &Scene) {
        self.rendered = false;
        let layer = match &self.layer {
            Some(l) => l.clone(),
            None => {
                log::error!(
                    "draw() called on headless renderer - use render_scene_to_image() instead"
                );
                return;
            }
        };
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = if let Some(drawable) = layer.next_drawable() {
            drawable
        } else {
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            return;
        };

        let encode_started = std::time::Instant::now();
        let command_buffer = match self.render_frame(scene, drawable.texture(), viewport_size) {
            Ok(command_buffer) => command_buffer,
            Err(error) => {
                log::error!("failed to render: {error:#}");
                return;
            }
        };

        if self.presents_with_transaction {
            command_buffer.commit();
            command_buffer.wait_until_scheduled();
            drawable.present();
        } else {
            command_buffer.present_drawable(drawable);
            command_buffer.commit();
        }
        self.rendered = true;
        retained_stats::record_encode(encode_started.elapsed().as_secs_f64() * 1e3);
    }

    fn render_frame(
        &mut self,
        scene: &Scene,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        let mut writer = InstanceBufferWriter::new(
            &self.device,
            &self.instance_buffer_pool,
            self.is_unified_memory,
        );
        let gradients = write_gradients(scene, &mut writer)?;
        let instance_bindings = write_instances(&gradients, scene, &mut writer).with_context(|| {
            format!(
                "scene too large: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                scene.paths.len(),
                scene.shadows.len(),
                scene.quads.len(),
                scene.underlines.len(),
                scene.monochrome_sprites.len(),
                scene.polychrome_sprites.len(),
                scene.surfaces.len(),
            )
        })?;
        let command_buffer = self.draw_primitives_to_texture(
            scene,
            &instance_bindings,
            &mut writer,
            texture,
            viewport_size,
        )?;

        let instance_buffer_pool = self.instance_buffer_pool.clone();
        let instance_buffer = Cell::new(Some(writer.finish()));
        let block = RcBlock::new(move |command_buffer: ptr::NonNull<AnyObject>| {
            if let Some(instance_buffer) = instance_buffer.take() {
                instance_buffer_pool.lock().release(instance_buffer);
            }
            if retained_stats::enabled() {
                // SAFETY: Metal passes the completed command buffer.
                let (start, end): (f64, f64) = unsafe {
                    let cb = command_buffer.as_ref();
                    (
                        objc2::msg_send![cb, GPUStartTime],
                        objc2::msg_send![cb, GPUEndTime],
                    )
                };
                retained_stats::record_gpu((end - start) * 1e3);
            }
        });
        // SAFETY: Both pointee types are opaque views of the same Objective-C block pointer ABI.
        unsafe {
            command_buffer.add_completed_handler(&*RcBlock::as_ptr(&block).cast());
        }

        Ok(command_buffer)
    }

    /// Renders the scene at the layer's drawable size to an offscreen texture
    /// and returns the pixel data as an RGBA image. Nothing is presented and
    /// no drawable is taken, so the window may be hidden: useful for visual
    /// tests and for exporting what a window draws.
    ///
    /// Note: This requires a layer-backed renderer (for its size). For
    /// headless rendering, use `render_scene_to_image()` instead.
    #[cfg(any(test, feature = "test-support", feature = "window-capture"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        let layer = self
            .layer
            .clone()
            .ok_or_else(|| anyhow::anyhow!("render_to_image requires a layer-backed renderer"))?;
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        self.render_scene_to_image(scene, viewport_size)
    }

    /// Renders a scene to an image without requiring a window or CAMetalLayer.
    ///
    /// This is the primary method for headless rendering. It creates an offscreen
    /// texture, renders the scene to it, and returns the pixel data as an RGBA image.
    #[cfg(any(test, feature = "bench-support", feature = "test-support", feature = "window-capture"))]
    pub fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene_to_image: {:?}", size);
        }

        // Headless callers do not have a Cocoa event-loop pool to release
        // autoreleased command buffers and render-pass descriptors.
        objc2::rc::autoreleasepool(|_| {
            // Update path intermediate textures for this size
            self.update_path_intermediate_textures(size);

            // Create an offscreen texture as render target
            let texture_descriptor = metal::TextureDescriptor::new();
            texture_descriptor.set_width(size.width.0 as u64);
            texture_descriptor.set_height(size.height.0 as u64);
            texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            texture_descriptor.set_usage(
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            // Like the atlas, only Apple GPUs can create shared textures on macOS;
            // Intel Macs cannot, even with unified memory. iOS has no managed storage.
            let uses_shared_storage = cfg!(target_os = "ios") || self.is_apple_gpu;
            texture_descriptor.set_storage_mode(if uses_shared_storage {
                metal::MTLStorageMode::Shared
            } else {
                metal::MTLStorageMode::Managed
            });
            let target_texture = self.device.new_texture(&texture_descriptor);

            let command_buffer = self.render_frame(scene, &target_texture, size)?;

            // Managed textures require an explicit blit synchronize before the CPU
            // can read back the rendered data. Without this, get_bytes returns
            // stale zeros.
            if !uses_shared_storage {
                let blit = command_buffer.new_blit_command_encoder();
                blit.synchronize_resource(&target_texture);
                blit.end_encoding();
            }

            // Commit and wait for completion
            command_buffer.commit();
            command_buffer.wait_until_completed();

            read_texture_to_image(&target_texture)
        })
    }

    /// Renders a scene to a reused offscreen texture without reading pixels
    /// back or blocking on GPU completion.
    ///
    /// This mirrors the CPU cost of presenting a frame to a window (scene
    /// encoding, instance buffer writes, command submission) and is used by
    /// headless benchmark rendering, where the produced pixels are never
    /// inspected.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<()> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene: {:?}", size);
        }

        objc2::rc::autoreleasepool(|_| {
            self.update_path_intermediate_textures(size);

            let needs_new_target = self.headless_render_target.as_ref().is_none_or(|texture| {
                texture.width() != size.width.0 as u64 || texture.height() != size.height.0 as u64
            });
            if needs_new_target {
                let texture_descriptor = metal::TextureDescriptor::new();
                texture_descriptor.set_width(size.width.0 as u64);
                texture_descriptor.set_height(size.height.0 as u64);
                texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
                texture_descriptor.set_usage(
                    metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
                );
                texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
                self.headless_render_target = Some(self.device.new_texture(&texture_descriptor));
            }
            let target_texture = self
                .headless_render_target
                .clone()
                .expect("just ensured the render target exists");

            let command_buffer = self.render_frame(scene, &target_texture, size)?;

            // Commit without waiting, mirroring presentation to a real window where
            // the CPU doesn't block on the GPU.
            command_buffer.commit();
            Ok(())
        })
    }

    fn draw_primitives_to_texture(
        &mut self,
        scene: &Scene,
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        let alpha = if self.opaque { 1. } else { 0. };
        self.sync_retained_shapes(scene, writer, command_buffer)?;

        // Texture targets first, in order: the window and later targets sample them.
        for target in &scene.texture_targets {
            if let Err(error) = self.draw_texture_target(
                scene,
                target,
                &instance_bindings.gradients,
                writer,
                command_buffer,
            ) {
                self.texture_pool.end_frame();
                return Err(error);
            }
        }

        let result = self.draw_scene(
            scene,
            scene,
            instance_bindings,
            writer,
            texture,
            viewport_size,
            Some(metal::MTLClearColor::new(0., 0., 0., alpha)),
            command_buffer,
        );
        // Keep the images and copies this frame used; free the rest.
        self.texture_pool.end_frame();
        // Every encoder is closed: a marks failure may panic here (debug).
        gpui::check_marks_reports();
        result?;
        Ok(command_buffer.to_owned())
    }

    /// Draws `scene` (the window's, or a group's at `depth`) into `texture`, cleared
    /// to `clear` first (or drawn over). `root` is the window's scene, which holds the
    /// shader sources and uniforms. Passes reopen only when a batch draws into them.
    fn draw_scene(
        &mut self,
        root: &Scene,
        scene: &Scene,
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
        clear: Option<metal::MTLClearColor>,
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<()> {
        let mut encoder = clear.map(|clear| {
            new_command_encoder_for_texture(command_buffer, texture, viewport_size, Some(clear))
        });
        let open =
            |encoder: &mut Option<_>| open_encoder(encoder, command_buffer, texture, viewport_size);

        for batch in scene.batches() {
            match batch {
                PrimitiveBatch::Shadows(range) => {
                    self.draw_shadows(range, instance_bindings, viewport_size, open(&mut encoder))
                }
                PrimitiveBatch::Quads(range) => {
                    self.draw_quads(range, instance_bindings, viewport_size, open(&mut encoder))
                }
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    close_encoder(&mut encoder);
                    let did_draw = self.draw_paths_to_intermediate(
                        paths,
                        writer,
                        viewport_size,
                        command_buffer,
                    )?;
                    if did_draw {
                        let encoder = open(&mut encoder);
                        if let Err(error) =
                            self.draw_paths_from_intermediate(paths, writer, viewport_size, encoder)
                        {
                            encoder.end_encoding();
                            return Err(error);
                        }
                    }
                }
                PrimitiveBatch::Underlines(range) => self.draw_underlines(
                    range,
                    instance_bindings,
                    viewport_size,
                    open(&mut encoder),
                ),
                PrimitiveBatch::MonochromeSprites { texture_id, range } => self
                    .draw_monochrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        viewport_size,
                        open(&mut encoder),
                    ),
                PrimitiveBatch::PolychromeSprites { texture_id, range } => self
                    .draw_polychrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        viewport_size,
                        open(&mut encoder),
                    ),
                PrimitiveBatch::Surfaces(range) => self.draw_surfaces(
                    &scene.surfaces[range.clone()],
                    range.start,
                    instance_bindings,
                    viewport_size,
                    open(&mut encoder),
                ),
                PrimitiveBatch::RetainedShapes(range) => self.draw_retained_shapes(
                    &scene.retained_shapes[range],
                    viewport_size,
                    open(&mut encoder),
                ),
                PrimitiveBatch::Marks(range) => self.draw_marks(
                    &scene.marks[range],
                    viewport_size,
                    open(&mut encoder),
                ),
                PrimitiveBatch::BackdropFilters(range) => {
                    close_encoder(&mut encoder);
                    for filter in &scene.backdrop_filters[range] {
                        if let Some(shader) = filter.shader.shader(root) {
                            self.draw_custom_shader(
                                root,
                                filter,
                                shader,
                                texture,
                                viewport_size,
                                command_buffer,
                            );
                            continue;
                        }
                        // Read what is drawn so far, filter it offscreen, draw it back.
                        let filtered =
                            self.filter_backdrop(filter, texture, viewport_size, command_buffer);
                        if let Some((filtered, region)) = filtered {
                            let encoder = new_command_encoder_for_texture(
                                command_buffer,
                                texture,
                                viewport_size,
                                None,
                            );
                            self.draw_backdrop(filter, &filtered, region, viewport_size, encoder);
                            encoder.end_encoding();
                            self.texture_pool.release(&filtered);
                        }
                    }
                }
                PrimitiveBatch::Groups(range) => {
                    close_encoder(&mut encoder);
                    for group in &scene.groups[range] {
                        self.draw_group(
                            root,
                            group,
                            &instance_bindings.gradients,
                            writer,
                            texture,
                            viewport_size,
                            command_buffer,
                        )?;
                    }
                }
                PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
            }
        }

        close_encoder(&mut encoder);
        Ok(())
    }

    /// Draws a texture target's scene ([`gpui::Window::paint_to_texture`]) into its
    /// external texture, cleared to transparent. A texture that is missing, another
    /// size, or not a render target is skipped (its pixels stay).
    fn draw_texture_target(
        &mut self,
        root: &Scene,
        target: &gpui::TextureTargetScene,
        gradients: &InstanceBinding,
        writer: &mut InstanceBufferWriter,
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<()> {
        let Some(texture) = self
            .sprite_atlas
            .external_texture_tile(target.id)
            .and_then(|tile| self.sprite_atlas.metal_texture(tile.texture_id))
        else {
            log::warn!("texture target {:?}: no such external texture", target.id);
            return Ok(());
        };
        if texture.width() != target.size.width.0 as u64
            || texture.height() != target.size.height.0 as u64
            || !texture
                .usage()
                .contains(metal::MTLTextureUsage::RenderTarget)
        {
            log::warn!(
                "texture target {:?}: the texture is {}x{} ({:?}), not a {:?} render target",
                target.id,
                texture.width(),
                texture.height(),
                texture.usage(),
                target.size
            );
            return Ok(());
        }
        let bindings = write_instances(gradients, &target.scene, writer)?;
        self.draw_scene(
            root,
            &target.scene,
            &bindings,
            writer,
            &texture,
            target.size,
            Some(metal::MTLClearColor::new(0., 0., 0., 0.)),
            command_buffer,
        )
    }

    /// Draws a group's scene into an image the size of its bounds, starting with its
    /// filtered backdrop, then blurs it, reads what is behind it when its blend mode
    /// needs that, and composites it onto `target` (`viewport_size` device pixels).
    fn draw_group(
        &mut self,
        root: &Scene,
        group: &gpui::CompositedGroup,
        gradients: &InstanceBinding,
        writer: &mut InstanceBufferWriter,
        target: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<()> {
        let size = group.image_size(viewport_size);
        if group.target_region(viewport_size).is_none() || size.width.0 <= 0 || size.height.0 <= 0 {
            return Ok(());
        }
        if size.width.0 > MAX_GROUP_IMAGE || size.height.0 > MAX_GROUP_IMAGE {
            log::error!("compositing group image of {size:?} is too large; not drawn");
            return Ok(());
        }
        let origin = [group.bounds.origin.x.0, group.bounds.origin.y.0];
        let image = self.texture_pool.acquire(
            &self.device,
            bucket(size.width.0 as u64),
            bucket(size.height.0 as u64),
        );
        let transparent = metal::MTLClearColor::new(0., 0., 0., 0.);
        // The image starts with what is behind the group, filtered (its backdrop).
        let mut clear = Some(transparent);
        if let Some(filter) = &group.effects.backdrop
            && let Some((filtered, mut region)) =
                self.filter_backdrop(filter, target, viewport_size, command_buffer)
        {
            let filter = filter.translated(-group.bounds.origin);
            region[0] -= origin[0];
            region[1] -= origin[1];
            let encoder =
                new_command_encoder_for_texture(command_buffer, &image, size, clear.take());
            self.draw_backdrop_with(
                &self.backdrop_group_pipeline_state,
                &filter,
                &filtered,
                region,
                size,
                encoder,
            );
            encoder.end_encoding();
            self.texture_pool.release(&filtered);
        }
        let instance_bindings = write_instances(gradients, &group.scene, writer)?;
        self.draw_scene(
            root,
            &group.scene,
            &instance_bindings,
            writer,
            &image,
            size,
            clear,
            command_buffer,
        )?;
        let viewport = [viewport_size.width.0 as f32, viewport_size.height.0 as f32];
        let blurred = if group.effects.blur_radius.0 >= 0.5 {
            let whole = Bounds::new(point(ScaledPixels(0.), ScaledPixels(0.)), group.bounds.size);
            match self.copy_region(&image, whole, group.effects.blur_radius.0, command_buffer) {
                Some((texture, region)) => Some((texture, region)),
                None => {
                    self.texture_pool.release(&image);
                    return Ok(());
                }
            }
        } else {
            None
        };
        let (source, source_region) = match &blurred {
            Some((texture, region)) => (
                texture.clone(),
                [
                    region[0] + origin[0],
                    region[1] + origin[1],
                    region[2],
                    region[3],
                ],
            ),
            None => (
                image.clone(),
                [
                    origin[0],
                    origin[1],
                    image.width() as f32,
                    image.height() as f32,
                ],
            ),
        };
        let backdrop = match group.target_region(viewport_size) {
            Some(region) if group.effects.blend_mode.reads_backdrop() => {
                self.copy_region(target, region, 0., command_buffer)
            }
            _ => None,
        };
        let (backdrop_texture, backdrop_region) = match &backdrop {
            Some((texture, region)) => (texture.as_ref(), *region),
            None => (source.as_ref(), source_region),
        };
        // The mask's image, drawn like a group's (it is never composited itself).
        let mask = match &group.mask {
            Some(mask) => {
                let image_size = mask.image.image_size(viewport_size);
                if image_size.width.0 <= 0 || image_size.height.0 <= 0 {
                    // An empty mask draws nothing of the group.
                    self.texture_pool.release(&image);
                    for (texture, _) in blurred.iter().chain(backdrop.iter()) {
                        self.texture_pool.release(texture);
                    }
                    return Ok(());
                }
                let texture = self.texture_pool.acquire(
                    &self.device,
                    bucket(image_size.width.0 as u64),
                    bucket(image_size.height.0 as u64),
                );
                let bindings = write_instances(gradients, &mask.image.scene, writer)?;
                self.draw_scene(
                    root,
                    &mask.image.scene,
                    &bindings,
                    writer,
                    &texture,
                    image_size,
                    Some(transparent),
                    command_buffer,
                )?;
                let region = [
                    mask.image.bounds.origin.x.0,
                    mask.image.bounds.origin.y.0,
                    texture.width() as f32,
                    texture.height() as f32,
                ];
                Some((texture, region))
            }
            None => None,
        };
        let mask_region = mask.as_ref().map_or([0.; 4], |(_, region)| *region);
        let uniforms =
            group.composite_uniforms(source_region, backdrop_region, mask_region, viewport);
        let encoder = new_command_encoder_for_texture(command_buffer, target, viewport_size, None);
        encoder.set_render_pipeline_state(
            self.group_composite_pipeline_states
                .for_blend_mode(group.effects.blend_mode),
        );
        encoder.set_vertex_buffer(0, Some(&self.unit_vertices), 0);
        let uniforms_size = mem::size_of_val(&uniforms) as u64;
        let pointer = &uniforms as *const gpui::GroupCompositeUniforms as *const _;
        encoder.set_vertex_bytes(1, uniforms_size, pointer);
        encoder.set_fragment_bytes(1, uniforms_size, pointer);
        encoder.set_fragment_texture(0, Some(&source));
        encoder.set_fragment_texture(1, Some(backdrop_texture));
        let mask_texture = mask
            .as_ref()
            .map_or(source.as_ref(), |(texture, _)| texture.as_ref());
        encoder.set_fragment_texture(2, Some(mask_texture));
        encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
        encoder.end_encoding();
        // Later passes reuse these; Metal orders them after this composite.
        self.texture_pool.release(&image);
        if let Some((texture, _)) = &blurred {
            self.texture_pool.release(texture);
        }
        if let Some((texture, _)) = &backdrop {
            self.texture_pool.release(texture);
        }
        if let Some((texture, _)) = &mask {
            self.texture_pool.release(texture);
        }
        Ok(())
    }

    /// Bytes of the group images and copies kept now (for tests and stats).
    pub fn group_texture_bytes(&self) -> u64 {
        self.texture_pool.bytes()
    }

    /// Bring each retained buffer drawn this frame up to date: changed ranges are
    /// staged in this frame's instance buffer and blitted into the retained GPU
    /// buffer, ordered before the render pass on the GPU timeline, so frames still
    /// in flight never see a half-written buffer.
    fn sync_retained_shapes(
        &mut self,
        scene: &Scene,
        writer: &mut InstanceBufferWriter,
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<()> {
        self.frame_index += 1;
        let scenes = scene.scenes();
        if scenes.iter().all(|scene| scene.retained_shapes.is_empty()) {
            return Ok(());
        }
        let mut blit: Option<&metal::BlitCommandEncoderRef> = None;
        let mut uploaded = 0usize;
        for draw in scenes.iter().flat_map(|scene| &scene.retained_shapes) {
            let id = draw.shapes.id();
            let frame = self.frame_index;
            let entry = self
                .retained_shapes
                .entry(id)
                .or_insert_with(|| RetainedGpu {
                    buffer: None,
                    capacity: 0,
                    len: 0,
                    version: 0,
                    last_frame: frame,
                });
            if entry.last_frame == frame && entry.version != 0 {
                continue;
            }
            entry.last_frame = frame;
            let mut result: Result<()> = Ok(());
            let device = &self.device;
            let version = draw.shapes.sync(entry.version, |upload| {
                let (data, ranges): (&[ShapeInstance], Vec<std::ops::Range<usize>>) = match upload {
                    RetainedUpload::None => return,
                    RetainedUpload::All(data) => (data, vec![0..data.len()]),
                    RetainedUpload::Ranges(data, ranges) => (data, ranges.to_vec()),
                };
                entry.len = data.len();
                let bytes = mem::size_of_val(data);
                if bytes == 0 {
                    return;
                }
                if entry.capacity < bytes {
                    // Grow (by half again) and copy everything into the new buffer.
                    let capacity = (bytes + bytes / 2).next_multiple_of(4096);
                    entry.buffer = Some(
                        device.new_buffer(capacity as u64, MTLResourceOptions::StorageModePrivate),
                    );
                    entry.capacity = capacity;
                    return copy_ranges(
                        writer,
                        command_buffer,
                        &mut blit,
                        entry,
                        data,
                        &[0..data.len()],
                        &mut uploaded,
                        &mut result,
                    );
                }
                copy_ranges(
                    writer,
                    command_buffer,
                    &mut blit,
                    entry,
                    data,
                    &ranges,
                    &mut uploaded,
                    &mut result,
                );
            });
            if let Err(error) = result {
                // The copy is incomplete: upload everything next time.
                entry.version = 0;
                if let Some(blit) = blit {
                    blit.end_encoding();
                }
                return Err(error);
            }
            entry.version = version;
        }
        if let Some(blit) = blit {
            blit.end_encoding();
        }
        // Drop GPU copies not drawn for a while (their buffers were dropped or hidden).
        let frame = self.frame_index;
        self.retained_shapes
            .retain(|_, e| frame - e.last_frame < 600);
        self.retained_uploaded_bytes += uploaded as u64;
        retained_stats::record_upload(uploaded);
        Ok(())
    }

    fn draw_retained_shapes(
        &self,
        draws: &[RetainedShapesDraw],
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        command_encoder.set_render_pipeline_state(&self.retained_shapes_pipeline_state);
        command_encoder.set_vertex_buffer(0, Some(&self.unit_vertices), 0);
        for draw in draws {
            let Some(gpu) = self.retained_shapes.get(&draw.shapes.id()) else {
                continue;
            };
            let Some(buffer) = &gpu.buffer else {
                continue;
            };
            let count = gpu.len as u32;
            let range = draw.paint.range.start.min(count)..draw.paint.range.end.min(count);
            if range.is_empty() {
                continue;
            }
            let uniforms =
                draw.uniforms([viewport_size.width.0 as f32, viewport_size.height.0 as f32]);
            command_encoder.set_vertex_buffer(1, Some(buffer), 0);
            command_encoder.set_vertex_bytes(
                2,
                mem::size_of_val(&uniforms) as u64,
                &uniforms as *const gpui::RetainedShapesUniforms as *const _,
            );
            command_encoder.set_fragment_bytes(
                2,
                mem::size_of_val(&uniforms) as u64,
                &uniforms as *const gpui::RetainedShapesUniforms as *const _,
            );
            command_encoder.draw_primitives_instanced_base_instance(
                metal::MTLPrimitiveType::Triangle,
                0,
                6,
                range.len() as u64,
                range.start as u64,
            );
        }
    }

    /// Draws marks into the open pass: no pass, upload or copy of their own.
    /// Programs compile on first use; one that fails to compile draws nothing,
    /// and says so ([`gpui::MarksShader::report_compile_error`]).
    fn draw_marks(
        &mut self,
        draws: &[gpui::MarksDraw],
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        let viewport = [viewport_size.width.0 as f32, viewport_size.height.0 as f32];
        let mut bound = None;
        for draw in draws {
            let device = &self.device;
            let pipeline = self
                .marks_pipelines
                .entry(draw.shader.id)
                .or_insert_with(|| {
                    crate::metal_marks::build_pipeline(
                        device,
                        &draw.shader,
                        MTLPixelFormat::BGRA8Unorm,
                    )
                    .inspect_err(|error| draw.shader.report_compile_error(format!("{error:#}")))
                    .ok()
                });
            let Some(pipeline) = pipeline.as_ref() else {
                continue;
            };
            let textures: SmallVec<[Option<metal::Texture>; 2]> = draw
                .bindings
                .textures
                .iter()
                .map(|id| {
                    self.sprite_atlas
                        .external_texture_tile(*id)
                        .and_then(|tile| self.sprite_atlas.metal_texture(tile.texture_id))
                })
                .collect();
            crate::metal_marks::encode(
                command_encoder,
                pipeline,
                draw,
                viewport,
                &textures,
                &self.blank_texture,
                &self.custom_shader_sampler,
                &mut bound,
            );
        }
    }

    fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<bool> {
        if paths.is_empty() {
            return Ok(false);
        }
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        let mut vertices = Vec::new();
        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.bounds.intersect(&path.content_mask.bounds),
                content_mask: path.content_mask,
            }));
        }
        let vertex_instance_bindings = writer.write(&vertices)?;

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));

        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.set_texture(Some(msaa_texture));
            color_attachment.set_resolve_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
        }

        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        // The target may be smaller than the intermediate texture (a group's image).
        command_encoder.set_viewport(metal::MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: viewport_size.width.0 as f64,
            height: viewport_size.height.0 as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        command_encoder.set_render_pipeline_state(&self.paths_rasterization_pipeline_state);
        command_encoder.set_vertex_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&vertex_instance_bindings.buffer),
            vertex_instance_bindings.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            PathRasterizationInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&vertex_instance_bindings.buffer),
            vertex_instance_bindings.offset as u64,
        );
        command_encoder.draw_primitives(
            metal::MTLPrimitiveType::Triangle,
            0,
            vertices.len() as u64,
        );

        command_encoder.end_encoding();
        Ok(true)
    }

    fn draw_shadows(
        &self,
        shadows: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if shadows.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.shadows_pipeline_state);
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_bindings.shadows.buffer),
            instance_bindings.shadows.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_bindings.shadows.buffer),
            instance_bindings.shadows.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            ShadowInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            shadows.len() as u64,
            shadows.start as u64,
        );
    }

    fn draw_quads(
        &self,
        quads: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if quads.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.quads_pipeline_state);
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_bindings.quads.buffer),
            instance_bindings.quads.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_bindings.quads.buffer),
            instance_bindings.quads.offset as u64,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Gradients as u64,
            Some(&instance_bindings.gradients.buffer),
            instance_bindings.gradients.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Gradients as u64,
            Some(&instance_bindings.gradients.buffer),
            instance_bindings.gradients.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            QuadInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            quads.len() as u64,
            quads.start as u64,
        );
    }

    /// Copies the region a backdrop filter reads from `target`, blurs it (at a lower
    /// resolution for large radii) and returns the filtered texture and where it sits
    /// in the target (origin and size, device pixels). `None` if it reads nothing.
    fn filter_backdrop(
        &self,
        filter: &BackdropFilter,
        target: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
    ) -> Option<(metal::Texture, [f32; 4])> {
        let viewport = Bounds::new(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            size(
                ScaledPixels(viewport_size.width.0 as f32),
                ScaledPixels(viewport_size.height.0 as f32),
            ),
        );
        let region = filter
            .sample_bounds()
            .intersect(&filter.content_mask.bounds.dilate(filter.blur_radius * 3.))
            .intersect(&viewport);
        self.copy_region(target, region, filter.blur_radius.0, command_buffer)
    }

    /// Copies `region` (within the target) of `target` and blurs it with standard
    /// deviation `sigma` (at a lower resolution for large ones); returns the texture
    /// and where it sits in the target (origin and size, whole device pixels). `None`
    /// if the region is empty.
    fn copy_region(
        &self,
        target: &metal::TextureRef,
        region: Bounds<ScaledPixels>,
        sigma: f32,
        command_buffer: &metal::CommandBufferRef,
    ) -> Option<(metal::Texture, [f32; 4])> {
        let (x, y) = (region.origin.x.0.floor(), region.origin.y.0.floor());
        let width = (region.origin.x.0 + region.size.width.0).ceil() - x;
        let height = (region.origin.y.0 + region.size.height.0).ceil() - y;
        if width < 1. || height < 1. {
            return None;
        }
        let (width, height) = (width as u64, height as u64);
        let copy = self.texture_pool.acquire(&self.device, width, height);
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_texture(
            target,
            0,
            0,
            metal::MTLOrigin {
                x: x as u64,
                y: y as u64,
                z: 0,
            },
            metal::MTLSize {
                width,
                height,
                depth: 1,
            },
            &copy,
            0,
            0,
            metal::MTLOrigin { x: 0, y: 0, z: 0 },
        );
        blit.end_encoding();

        let region = [x, y, width as f32, height as f32];
        if sigma < 0.5 {
            return Some((copy, region));
        }
        // Blur at most 8 texels wide: downsample larger radii and let the composite's
        // bilinear sampling scale back up.
        let scale = (sigma / 8.).ceil().max(1.);
        let small = Size {
            width: DevicePixels(((width as f32) / scale).ceil().max(1.) as i32),
            height: DevicePixels(((height as f32) / scale).ceil().max(1.) as i32),
        };
        let a =
            self.texture_pool
                .acquire(&self.device, small.width.0 as u64, small.height.0 as u64);
        let b =
            self.texture_pool
                .acquire(&self.device, small.width.0 as u64, small.height.0 as u64);
        let pass = |source: &metal::TextureRef,
                    destination: &metal::TextureRef,
                    blur: Option<BackdropBlurParams>| {
            let encoder = new_command_encoder_for_texture(
                command_buffer,
                destination,
                small,
                Some(metal::MTLClearColor::new(0., 0., 0., 0.)),
            );
            match blur {
                Some(params) => {
                    encoder.set_render_pipeline_state(&self.backdrop_blur_pipeline_state);
                    encoder.set_fragment_bytes(
                        0,
                        mem::size_of::<BackdropBlurParams>() as u64,
                        &params as *const BackdropBlurParams as *const _,
                    );
                }
                None => encoder.set_render_pipeline_state(&self.backdrop_copy_pipeline_state),
            }
            encoder.set_fragment_texture(0, Some(source));
            encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 3);
            encoder.end_encoding();
        };
        let sigma = sigma / scale;
        let radius = (sigma * 3.).ceil() as i32;
        pass(&copy, &a, None);
        pass(
            &a,
            &b,
            Some(BackdropBlurParams {
                direction: [1. / small.width.0 as f32, 0.],
                sigma,
                radius,
            }),
        );
        pass(
            &b,
            &a,
            Some(BackdropBlurParams {
                direction: [0., 1. / small.height.0 as f32],
                sigma,
                radius,
            }),
        );
        self.texture_pool.release(&copy);
        self.texture_pool.release(&b);
        Some((a, region))
    }

    /// Draws a custom shader: reads its backdrop (when it does) and runs its pipeline,
    /// compiled on first use. A shader that fails to compile draws nothing.
    fn draw_custom_shader(
        &mut self,
        root: &Scene,
        filter: &BackdropFilter,
        shader: &gpui::Shader,
        target: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
    ) {
        let pipeline = self
            .custom_shaders
            .entry(shader.id)
            .or_insert_with(|| {
                crate::metal_custom_shader::build_pipeline(
                    &self.device,
                    shader,
                    MTLPixelFormat::BGRA8Unorm,
                )
                .inspect_err(|error| log::error!("custom shader: {error:#}"))
                .ok()
            })
            .clone();
        let Some(pipeline) = pipeline else { return };
        let filtered = if filter.shader.reads_backdrop() {
            self.filter_backdrop(filter, target, viewport_size, command_buffer)
        } else {
            None
        };
        let (backdrop, region) = match &filtered {
            Some((texture, region)) => (texture.as_ref(), *region),
            None => (self.blank_texture.as_ref(), [0., 0., 1., 1.]),
        };
        let params = crate::metal_custom_shader::ShaderParams::new(
            filter,
            region,
            [viewport_size.width.0 as f32, viewport_size.height.0 as f32],
        );
        let encoder = new_command_encoder_for_texture(command_buffer, target, viewport_size, None);
        crate::metal_custom_shader::draw(
            encoder,
            &pipeline,
            &self.custom_shader_sampler,
            &params,
            filter.shader.uniforms(root),
            backdrop,
        );
        encoder.end_encoding();
        if let Some((texture, _)) = &filtered {
            self.texture_pool.release(texture);
        }
    }

    fn draw_backdrop(
        &self,
        filter: &BackdropFilter,
        filtered: &metal::TextureRef,
        region: [f32; 4],
        viewport_size: Size<DevicePixels>,
        encoder: &metal::RenderCommandEncoderRef,
    ) {
        self.draw_backdrop_with(
            &self.backdrop_composite_pipeline_state,
            filter,
            filtered,
            region,
            viewport_size,
            encoder,
        );
    }

    fn draw_backdrop_with(
        &self,
        pipeline: &metal::RenderPipelineStateRef,
        filter: &BackdropFilter,
        filtered: &metal::TextureRef,
        region: [f32; 4],
        viewport_size: Size<DevicePixels>,
        encoder: &metal::RenderCommandEncoderRef,
    ) {
        encoder.set_render_pipeline_state(pipeline);
        encoder.set_vertex_buffer(0, Some(&self.unit_vertices), 0);
        let filter_size = mem::size_of::<BackdropFilter>() as u64;
        let filter_ptr = filter as *const BackdropFilter as *const _;
        encoder.set_vertex_bytes(1, filter_size, filter_ptr);
        encoder.set_vertex_bytes(
            2,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        encoder.set_fragment_bytes(1, filter_size, filter_ptr);
        encoder.set_fragment_bytes(
            3,
            mem::size_of_val(&region) as u64,
            region.as_ptr() as *const _,
        );
        encoder.set_fragment_texture(0, Some(filtered));
        encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> Result<()> {
        if paths.is_empty() {
            return Ok(());
        }
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        command_encoder.set_render_pipeline_state(&self.path_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.set_fragment_texture(
            SpriteInputIndex::AtlasTexture as u64,
            Some(intermediate_texture),
        );

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths:
        // overlapping paths' bounds are copied as one region.
        let sprites: Vec<PathSprite> = gpui::path_copy_bounds(paths)
            .into_iter()
            .map(|bounds| PathSprite { bounds })
            .collect();

        let sprite_instance_bindings = writer.write(&sprites)?;
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&sprite_instance_bindings.buffer),
            sprite_instance_bindings.offset as u64,
        );

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        Ok(())
    }

    fn draw_underlines(
        &self,
        underlines: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if underlines.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.underlines_pipeline_state);
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_bindings.underlines.buffer),
            instance_bindings.underlines.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_bindings.underlines.buffer),
            instance_bindings.underlines.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            UnderlineInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            underlines.len() as u64,
            underlines.start as u64,
        );
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if sprites.is_empty() {
            return;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.monochrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.monochrome_sprites.buffer),
            instance_bindings.monochrome_sprites.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.monochrome_sprites.buffer),
            instance_bindings.monochrome_sprites.offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
            sprites.start as u64,
        );
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if sprites.is_empty() {
            return;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.polychrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.polychrome_sprites.buffer),
            instance_bindings.polychrome_sprites.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.polychrome_sprites.buffer),
            instance_bindings.polychrome_sprites.offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));
        let sampler = match self.sprite_atlas.sampling(texture_id) {
            gpui::TextureSampling::Linear => &self.sprite_samplers[0],
            gpui::TextureSampling::Nearest => &self.sprite_samplers[1],
        };
        command_encoder.set_fragment_sampler_state(0, Some(sampler));

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
            sprites.start as u64,
        );
    }

    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        first_surface: usize,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if surfaces.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.surfaces_pipeline_state);
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Surfaces as u64,
            Some(&instance_bindings.surfaces.buffer),
            instance_bindings.surfaces.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SurfaceInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        for (index, surface) in surfaces.iter().enumerate() {
            let image_buffer = &surface.image_buffer;
            let texture_size = size(
                DevicePixels::from(CVPixelBufferGetWidth(image_buffer) as i32),
                DevicePixels::from(CVPixelBufferGetHeight(image_buffer) as i32),
            );

            assert_eq!(
                CVPixelBufferGetPixelFormatType(image_buffer),
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            );

            let plane_texture = |plane, format| unsafe {
                let mut texture = ptr::null_mut();
                let status = CVMetalTextureCache::create_texture_from_image(
                    None,
                    &self.core_video_texture_cache,
                    image_buffer,
                    None,
                    format,
                    CVPixelBufferGetWidthOfPlane(image_buffer, plane),
                    CVPixelBufferGetHeightOfPlane(image_buffer, plane),
                    plane,
                    ptr::NonNull::from(&mut texture),
                );
                assert_eq!(
                    status, kCVReturnSuccess,
                    "failed to create CoreVideo Metal texture"
                );
                CFRetained::<CVMetalTexture>::from_raw(
                    ptr::NonNull::new(texture).expect("CoreVideo returned a null texture"),
                )
            };
            let y_texture = plane_texture(0, objc2_metal::MTLPixelFormat::R8Unorm);
            let cb_cr_texture = plane_texture(1, objc2_metal::MTLPixelFormat::RG8Unorm);

            command_encoder.set_vertex_bytes(
                SurfaceInputIndex::TextureSize as u64,
                mem::size_of_val(&texture_size) as u64,
                &texture_size as *const Size<DevicePixels> as *const _,
            );
            command_encoder.set_fragment_texture(SurfaceInputIndex::YTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(&y_texture)
                    .expect("CoreVideo returned no Metal texture");
                Some(metal::TextureRef::from_ptr(
                    objc2::rc::Retained::as_ptr(&texture) as *mut _,
                ))
            });
            command_encoder.set_fragment_texture(SurfaceInputIndex::CbCrTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(&cb_cr_texture)
                    .expect("CoreVideo returned no Metal texture");
                Some(metal::TextureRef::from_ptr(
                    objc2::rc::Retained::as_ptr(&texture) as *mut _,
                ))
            });

            command_encoder.draw_primitives_instanced_base_instance(
                metal::MTLPrimitiveType::Triangle,
                0,
                6,
                1,
                (first_surface + index) as u64,
            );
        }
    }
}

/// The pass `encoder` holds, or a new one that loads `texture`.
fn open_encoder<'a>(
    encoder: &mut Option<&'a metal::RenderCommandEncoderRef>,
    command_buffer: &'a metal::CommandBufferRef,
    texture: &'a metal::TextureRef,
    viewport_size: Size<DevicePixels>,
) -> &'a metal::RenderCommandEncoderRef {
    encoder.get_or_insert_with(|| {
        new_command_encoder_for_texture(command_buffer, texture, viewport_size, None)
    })
}

/// Ends the pass `encoder` holds, if any.
fn close_encoder(encoder: &mut Option<&metal::RenderCommandEncoderRef>) {
    if let Some(encoder) = encoder.take() {
        encoder.end_encoding();
    }
}

fn new_command_encoder_for_texture<'a>(
    command_buffer: &'a metal::CommandBufferRef,
    texture: &'a metal::TextureRef,
    viewport_size: Size<DevicePixels>,
    clear_color: Option<metal::MTLClearColor>,
) -> &'a metal::RenderCommandEncoderRef {
    #[cfg(test)]
    render_passes::opened();
    let render_pass_descriptor = metal::RenderPassDescriptor::new();
    let color_attachment = render_pass_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();
    color_attachment.set_texture(Some(texture));
    color_attachment.set_store_action(metal::MTLStoreAction::Store);
    if let Some(clear_color) = clear_color {
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(clear_color);
    } else {
        color_attachment.set_load_action(metal::MTLLoadAction::Load);
    }

    let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
    command_encoder.set_viewport(metal::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    command_encoder
}

#[cfg(any(test, feature = "bench-support", feature = "test-support", feature = "window-capture"))]
fn read_texture_to_image(texture: &metal::TextureRef) -> Result<RgbaImage> {
    let width = texture.width() as u32;
    let height = texture.height() as u32;
    let bytes_per_row = width as usize * 4;
    let mut pixels = vec![0u8; height as usize * bytes_per_row];

    let region = metal::MTLRegion {
        origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
        size: metal::MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
    };
    texture.get_bytes(
        pixels.as_mut_ptr() as *mut std::ffi::c_void,
        bytes_per_row as u64,
        region,
        0,
    );

    // Convert BGRA to RGBA (swap B and R channels)
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    RgbaImage::from_raw(width, height, pixels).context("failed to create RgbaImage from pixel data")
}

fn build_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

/// A pipeline that overwrites its target (backdrop copies and blur passes).
fn build_opaque_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");
    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(false);
    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

/// The filtered color (premultiplied by coverage) over the target, keeping the
/// target's alpha.
fn build_backdrop_composite_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label("backdrop_composite");
    let vertex_fn = library
        .get_function("backdrop_composite_vertex", None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function("backdrop_composite_fragment", None)
        .expect("error locating fragment function");
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::Zero);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);
    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

/// Group images are allocated in steps of this many device pixels per side, so a
/// group that changes size a little reuses its image.
const GROUP_IMAGE_STEP: u64 = 64;

/// The largest group image, per side: Metal's 2D texture limit on every GPU family
/// GPUI supports.
const MAX_GROUP_IMAGE: i32 = 16384 - GROUP_IMAGE_STEP as i32;

fn bucket(size: u64) -> u64 {
    size.max(1).next_multiple_of(GROUP_IMAGE_STEP)
}

/// Offscreen textures (group images, blur and backdrop copies) by size, reused
/// within and across frames. A texture released during a frame is reused later in
/// it: Metal orders the later pass after the earlier ones. Textures a frame did not
/// use are freed at its end.
#[derive(Default)]
struct TexturePool {
    entries: std::cell::RefCell<Vec<PooledTexture>>,
}

struct PooledTexture {
    texture: metal::Texture,
    in_use: bool,
    used: bool,
}

impl TexturePool {
    fn acquire(&self, device: &metal::DeviceRef, width: u64, height: u64) -> metal::Texture {
        let mut entries = self.entries.borrow_mut();
        if let Some(entry) = entries.iter_mut().find(|entry| {
            !entry.in_use && entry.texture.width() == width && entry.texture.height() == height
        }) {
            entry.in_use = true;
            entry.used = true;
            return entry.texture.clone();
        }
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(width);
        descriptor.set_height(height);
        descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        let texture = device.new_texture(&descriptor);
        entries.push(PooledTexture {
            texture: texture.clone(),
            in_use: true,
            used: true,
        });
        texture
    }

    fn release(&self, texture: &metal::TextureRef) {
        if let Some(entry) = self
            .entries
            .borrow_mut()
            .iter_mut()
            .find(|entry| std::ptr::eq(entry.texture.as_ref(), texture))
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
            .map(|entry| entry.texture.width() * entry.texture.height() * 4)
            .sum()
    }
}

/// The group composite pipelines, one per kind of hardware blending.
struct GroupCompositePipelines {
    /// Premultiplied source over: normal and the blend modes, which read the backdrop.
    over: metal::RenderPipelineState,
    /// Adds (plus-lighter).
    add: metal::RenderPipelineState,
    /// Replaces the target with the shader's result (plus-darker).
    replace: metal::RenderPipelineState,
}

impl GroupCompositePipelines {
    fn new(
        device: &metal::DeviceRef,
        library: &metal::LibraryRef,
        pixel_format: metal::MTLPixelFormat,
    ) -> Self {
        let build = |label: &str, destination: Option<metal::MTLBlendFactor>| {
            let descriptor = metal::RenderPipelineDescriptor::new();
            descriptor.set_label(label);
            let vertex_fn = library
                .get_function("group_composite_vertex", None)
                .expect("error locating vertex function");
            let fragment_fn = library
                .get_function("group_composite_fragment", None)
                .expect("error locating fragment function");
            descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
            descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
            let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
            color_attachment.set_pixel_format(pixel_format);
            if let Some(destination) = destination {
                color_attachment.set_blending_enabled(true);
                color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
                color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
                color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
                color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
                color_attachment.set_destination_rgb_blend_factor(destination);
                color_attachment.set_destination_alpha_blend_factor(destination);
            } else {
                color_attachment.set_blending_enabled(false);
            }
            device
                .new_render_pipeline_state(&descriptor)
                .expect("could not create render pipeline state")
        };
        Self {
            over: build(
                "group_composite_over",
                Some(metal::MTLBlendFactor::OneMinusSourceAlpha),
            ),
            add: build("group_composite_add", Some(metal::MTLBlendFactor::One)),
            replace: build("group_composite_replace", None),
        }
    }

    fn for_blend_mode(&self, blend_mode: gpui::BlendMode) -> &metal::RenderPipelineStateRef {
        match blend_mode {
            gpui::BlendMode::PlusLighter => &self.add,
            gpui::BlendMode::PlusDarker
            | gpui::BlendMode::ColorDodge
            | gpui::BlendMode::ColorBurn => &self.replace,
            _ => &self.over,
        }
    }
}

/// A group's filtered backdrop, premultiplied, over its transparent image.
fn build_backdrop_group_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label("backdrop_group");
    let vertex_fn = library
        .get_function("backdrop_composite_vertex", None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function("backdrop_group_fragment", None)
        .expect("error locating fragment function");
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

/// `BackdropBlurParams` in shaders.metal.
#[repr(C)]
struct BackdropBlurParams {
    direction: [f32; 2],
    sigma: f32,
    radius: i32,
}

fn build_path_sprite_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_rasterization_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    path_sample_count: u32,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.set_raster_sample_count(path_sample_count as _);
        descriptor.set_alpha_to_coverage_enabled(false);
    }
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

#[derive(Clone)]
struct InstanceBinding {
    buffer: metal::Buffer,
    offset: usize,
}

struct InstanceBindings {
    quads: InstanceBinding,
    shadows: InstanceBinding,
    underlines: InstanceBinding,
    monochrome_sprites: InstanceBinding,
    polychrome_sprites: InstanceBinding,
    surfaces: InstanceBinding,
    /// The window scene's gradient records, which quads in every scene refer to.
    gradients: InstanceBinding,
}

/// Writes the window scene's gradient records once per frame; every scene's quads
/// read them. One texel at least, so the binding never sits at the end of the buffer.
fn write_gradients(root: &Scene, writer: &mut InstanceBufferWriter) -> Result<InstanceBinding> {
    if root.gradients.is_empty() {
        writer.write(&[[0f32; 4]])
    } else {
        writer.write(&root.gradients)
    }
}

/// Writes `scene`'s instances; its quads read `gradients` (the window scene's).
fn write_instances(
    gradients: &InstanceBinding,
    scene: &Scene,
    writer: &mut InstanceBufferWriter,
) -> Result<InstanceBindings> {
    Ok(InstanceBindings {
        gradients: gradients.clone(),
        quads: writer.write(&scene.quads)?,
        shadows: writer.write(&scene.shadows)?,
        underlines: writer.write(&scene.underlines)?,
        monochrome_sprites: writer.write(&scene.monochrome_sprites)?,
        polychrome_sprites: writer.write(&scene.polychrome_sprites)?,
        surfaces: writer.write_iter(scene.surfaces.iter().map(|surface| SurfaceBounds {
            bounds: surface.bounds,
            content_mask: surface.content_mask,
        }))?,
    })
}

struct InstanceBufferWriter {
    device: metal::Device,
    pool: Arc<Mutex<InstanceBufferPool>>,
    unified_memory: bool,
    filled: Vec<(InstanceBuffer, usize)>,
    current: InstanceBuffer,
    offset: usize,
}

impl InstanceBufferWriter {
    fn new(
        device: &metal::Device,
        pool: &Arc<Mutex<InstanceBufferPool>>,
        unified_memory: bool,
    ) -> Self {
        let current = pool.lock().acquire(device, unified_memory);
        Self {
            device: device.clone(),
            pool: pool.clone(),
            unified_memory,
            filled: Vec::new(),
            current,
            offset: 0,
        }
    }

    fn allocate<T>(&mut self, count: usize) -> Result<(InstanceBinding, &mut [MaybeUninit<T>])> {
        let size = mem::size_of::<T>() * count;
        let mut offset = self.offset.next_multiple_of(INSTANCE_BUFFER_ALIGNMENT);
        if offset + size > self.current.size {
            self.grow(size)?;
            offset = 0;
        }
        self.offset = offset + size;

        let binding = InstanceBinding {
            buffer: self.current.metal_buffer.clone(),
            offset,
        };
        // Safety: the reservation lies within a buffer this frame owns
        // exclusively, and never overlaps one handed out earlier.
        let values = unsafe {
            let start = (self.current.metal_buffer.contents() as *mut u8).add(offset);
            slice::from_raw_parts_mut(start.cast::<MaybeUninit<T>>(), count)
        };
        Ok((binding, values))
    }

    fn write<T>(&mut self, values: &[T]) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                destination.as_mut_ptr().cast::<T>(),
                values.len(),
            );
        }
        Ok(binding)
    }

    fn write_iter<T>(
        &mut self,
        values: impl ExactSizeIterator<Item = T>,
    ) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        for (slot, value) in destination.iter_mut().zip(values) {
            slot.write(value);
        }
        Ok(binding)
    }

    fn grow(&mut self, required: usize) -> Result<()> {
        let mut pool = self.pool.lock();
        let buffer_size = (pool.buffer_size * 2)
            .max(required.next_power_of_two())
            .min(MAX_INSTANCE_BUFFER_SIZE);
        anyhow::ensure!(
            buffer_size >= required,
            "instance buffer needs {required} bytes, above the maximum of {MAX_INSTANCE_BUFFER_SIZE}"
        );
        anyhow::ensure!(
            buffer_size > self.current.size,
            "frame instance data exceeds the {MAX_INSTANCE_BUFFER_SIZE}-byte maximum"
        );
        if buffer_size != pool.buffer_size {
            log::info!("increased instance buffer size to {buffer_size}");
            pool.reset(buffer_size);
        }
        let buffer = pool.acquire(&self.device, self.unified_memory);
        drop(pool);

        let filled = mem::replace(&mut self.current, buffer);
        self.filled.push((filled, self.offset));
        self.offset = 0;
        Ok(())
    }

    fn finish(self) -> InstanceBuffer {
        let Self {
            unified_memory,
            filled,
            current,
            offset,
            ..
        } = self;

        if !unified_memory {
            for (buffer, written) in &filled {
                if *written == 0 {
                    continue;
                }
                buffer.metal_buffer.did_modify_range(NSRange {
                    location: 0,
                    length: *written as NSUInteger,
                });
            }
            if offset > 0 {
                current.metal_buffer.did_modify_range(NSRange {
                    location: 0,
                    length: offset as NSUInteger,
                });
            }
        }

        // Metal retains encoded resources until the command buffer completes.
        // Only the final, largest buffer is worth keeping in the pool.
        drop(filled);
        current
    }
}

#[repr(C)]
enum ShadowInputIndex {
    Vertices = 0,
    Shadows = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum QuadInputIndex {
    Vertices = 0,
    Quads = 1,
    ViewportSize = 2,
    Gradients = 3,
}

#[repr(C)]
enum UnderlineInputIndex {
    Vertices = 0,
    Underlines = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum SpriteInputIndex {
    Vertices = 0,
    Sprites = 1,
    ViewportSize = 2,
    AtlasTextureSize = 3,
    AtlasTexture = 4,
}

#[repr(C)]
enum SurfaceInputIndex {
    Vertices = 0,
    Surfaces = 1,
    ViewportSize = 2,
    TextureSize = 3,
    YTexture = 4,
    CbCrTexture = 5,
}

#[repr(C)]
enum PathRasterizationInputIndex {
    Vertices = 0,
    ViewportSize = 1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PathSprite {
    pub bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SurfaceBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
pub struct MetalHeadlessRenderer {
    renderer: MetalRenderer,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl MetalHeadlessRenderer {
    pub fn new() -> Self {
        let instance_buffer_pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let renderer = MetalRenderer::new_headless(instance_buffer_pool);
        Self { renderer }
    }
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl gpui::PlatformHeadlessRenderer for MetalHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        self.renderer.render_scene_to_image(scene, size)
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.renderer.render_scene(scene, size)
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.renderer.sprite_atlas().clone()
    }

    fn set_transparent(&mut self, transparent: bool) {
        self.renderer.update_transparency(transparent);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frame cost of 100k retained shapes under a moving camera, offscreen at
    /// 2800x1800: CPU to encode, GPU time, and bytes uploaded after the first
    /// frame. `cargo test -p gpui_apple --release --lib retained_shapes_100k -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn retained_shapes_100k_frame_cost() {
        use gpui::{RetainedShapes, RetainedShapesDraw, RetainedShapesPaint, TransformationMatrix};
        let n: usize = std::env::var("RETAINED_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100_000);
        let cols = ((n as f32 * 1.6).sqrt().ceil() as usize).max(1);
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut unit = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        let instances: Vec<ShapeInstance> = (0..n)
            .map(|i| {
                let angle = if unit() < 0.15 {
                    (unit() - 0.5) * 1.2
                } else {
                    0.0
                };
                let (s, c) = angle.sin_cos();
                ShapeInstance {
                    transform: [
                        c,
                        s,
                        -s,
                        c,
                        (i % cols) as f32 * 64.0,
                        (i / cols) as f32 * 64.0,
                    ],
                    width: 24.0 + unit() * 30.0,
                    height: 20.0 + unit() * 30.0,
                    corner_radii: if i % 7 == 0 { [6.0; 4] } else { [0.0; 4] },
                    stroke_width: 2.0,
                    fill: if unit() < 0.25 { 0x569cff5a } else { 0 },
                    stroke: 0xa78bfaff,
                    kind: if unit() < 0.45 {
                        ShapeInstance::ELLIPSE
                    } else {
                        ShapeInstance::RECT
                    },
                }
            })
            .collect();
        let shapes = RetainedShapes::new();
        shapes.set_all(instances);
        let (w, h) = (2800, 1800);
        let page_w = cols as f32 * 64.0;
        let mut renderer =
            MetalRenderer::new_headless(Arc::new(Mutex::new(InstanceBufferPool::default())));
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(w as u64);
        descriptor.set_height(h as u64);
        descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        descriptor.set_usage(metal::MTLTextureUsage::RenderTarget);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        let texture = renderer.device.new_texture(&descriptor);
        let size = size(DevicePixels(w), DevicePixels(h));
        let bounds = Bounds::new(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            gpui::size(ScaledPixels(w as f32), ScaledPixels(h as f32)),
        );
        let (mut scene_ms, mut encode_ms, mut gpu_ms) = (vec![], vec![], vec![]);
        let mut uploaded_after_first = 0;
        for frame in 0..240 {
            let t = frame as f32 / 60.0;
            let z = w as f32 / page_w * (1.0 + 1.5 * (0.5 + 0.5 * t.sin()));
            let camera = TransformationMatrix {
                rotation_scale: [[z, 0.0], [0.0, z]],
                translation: [-(t * 37.0) % 500.0, -(t * 23.0) % 300.0],
            };
            let started = std::time::Instant::now();
            let mut scene = Scene::default();
            scene.insert_primitive(RetainedShapesDraw {
                order: 0,
                bounds,
                content_mask: ContentMask::from(bounds),
                shapes: shapes.clone(),
                paint: RetainedShapesPaint::new(0..n as u32, camera),
                opacity: 1.0,
            });
            scene.finish();
            let scene_done = started.elapsed();
            let before = renderer.retained_uploaded_bytes;
            objc2::rc::autoreleasepool(|_| {
                let started = std::time::Instant::now();
                let mut writer = InstanceBufferWriter::new(
                    &renderer.device,
                    &renderer.instance_buffer_pool,
                    renderer.is_unified_memory,
                );
                let gradients = write_gradients(&scene, &mut writer).unwrap();
                let bindings = write_instances(&gradients, &scene, &mut writer).unwrap();
                let cb = renderer
                    .draw_primitives_to_texture(&scene, &bindings, &mut writer, &texture, size)
                    .unwrap();
                let encoded = started.elapsed();
                cb.commit();
                cb.wait_until_completed();
                let (start, end): (f64, f64) = unsafe {
                    let cb = &*(cb.as_ptr() as *const AnyObject);
                    (
                        objc2::msg_send![cb, GPUStartTime],
                        objc2::msg_send![cb, GPUEndTime],
                    )
                };
                drop(writer);
                if frame >= 20 {
                    scene_ms.push(scene_done.as_secs_f64() * 1e3);
                    encode_ms.push(encoded.as_secs_f64() * 1e3);
                    gpu_ms.push((end - start) * 1e3);
                }
            });
            if frame > 0 {
                uploaded_after_first += renderer.retained_uploaded_bytes - before;
            }
        }
        let q = |v: &mut Vec<f64>, p: f64| {
            v.sort_by(f64::total_cmp);
            v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
        };
        println!(
            "RETAINED_100K n {n} target {w}x{h} scene_build_ms p50 {:.3} encode_cpu_ms p50 {:.3} p95 {:.3} gpu_ms p50 {:.3} p95 {:.3} first_upload_bytes {} later_upload_bytes {uploaded_after_first}",
            q(&mut scene_ms, 0.5),
            q(&mut encode_ms, 0.5),
            q(&mut encode_ms, 0.95),
            q(&mut gpu_ms, 0.5),
            q(&mut gpu_ms, 0.95),
            n * mem::size_of::<ShapeInstance>(),
        );
        assert_eq!(uploaded_after_first, 0, "pan frames must upload nothing");
    }

    /// Two overlapping translucent paths in one paint layer (one draw order): each pixel
    /// is copied from the intermediate once, so the overlap is the two composited, not
    /// that composited twice.
    #[test]
    fn overlapping_paths_in_one_layer_composite_once() {
        use gpui::{Bounds, ContentMask, Path, ScaledPixels, px};
        let mut renderer = MetalHeadlessRenderer::new();
        renderer.renderer.update_transparency(true);
        let all = Bounds::new(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            size(ScaledPixels(64.), ScaledPixels(32.)),
        );
        let rect = |x: f32| {
            let mut path = Path::new(point(px(x), px(4.)));
            path.line_to(point(px(x + 30.), px(4.)));
            path.line_to(point(px(x + 30.), px(28.)));
            path.line_to(point(px(x), px(28.)));
            path.line_to(point(px(x), px(4.)));
            let mut path = path.scale(1.);
            path.content_mask = ContentMask::from(all);
            path.color = gpui::Hsla {
                h: 0.,
                s: 0.,
                l: 0.,
                a: 0.5,
            }
            .into();
            path
        };
        let mut scene = Scene::default();
        scene.push_layer(all);
        scene.insert_primitive(rect(4.));
        scene.insert_primitive(rect(20.));
        scene.pop_layer();
        scene.finish();
        let image = gpui::PlatformHeadlessRenderer::render_scene_to_image(
            &mut renderer,
            &scene,
            size(DevicePixels(64), DevicePixels(32)),
        )
        .expect("render");
        // One path: alpha 0.5. Both: 0.75 (composited once), not 0.9375 (twice).
        let alpha = |x: u32| image.get_pixel(x, 16).0[3] as f32 / 255.;
        assert!((alpha(10) - 0.5).abs() < 0.02, "one path {}", alpha(10));
        assert!((alpha(27) - 0.75).abs() < 0.02, "the overlap {}", alpha(27));
    }

    #[test]
    fn scaled_up_images_fill_to_their_edges() {
        // A 1x1 image drawn 40 px square fills it flat: linear filtering must not read past the
        // tile into its (transparent) atlas neighbours, which faded the edges to 1/4 alpha.
        use gpui::{
            AtlasKey, Bounds, ContentMask, Corners, ImageId, PaddedBool32, PolychromeSprite,
            RenderImageParams, ScaledPixels, TransformationMatrix,
        };
        let mut renderer = MetalHeadlessRenderer::new();
        renderer.renderer.update_transparency(true);
        let atlas = gpui::PlatformHeadlessRenderer::sprite_atlas(&renderer);
        // BGRA: opaque green.
        let pixel: [u8; 4] = [0, 128, 0, 255];
        let tile = atlas
            .get_or_insert_with(
                AtlasKey::Image(RenderImageParams { image_id: ImageId(1), frame_index: 0 }),
                &mut || Ok(Some((size(DevicePixels(1), DevicePixels(1)), std::borrow::Cow::Borrowed(&pixel[..])))),
            )
            .expect("insert")
            .expect("tile");
        let all = Bounds::new(point(ScaledPixels(0.), ScaledPixels(0.)), size(ScaledPixels(48.), ScaledPixels(48.)));
        let mut scene = Scene::default();
        scene.insert_primitive(PolychromeSprite {
            order: 0,
            exact_scale: 0,
            tile,
            grayscale: PaddedBool32::from(false),
            opacity: 1.,
            bounds: Bounds::new(point(ScaledPixels(4.), ScaledPixels(4.)), size(ScaledPixels(40.), ScaledPixels(40.))),
            content_mask: ContentMask::from(all),
            corner_radii: Corners::default(),
            transformation: TransformationMatrix::unit(),
        });
        scene.finish();
        let image = gpui::PlatformHeadlessRenderer::render_scene_to_image(&mut renderer, &scene, size(DevicePixels(48), DevicePixels(48)))
            .expect("render");
        for (x, y) in [(4, 4), (43, 4), (4, 43), (43, 43), (24, 4), (4, 24), (24, 24)] {
            let p = image.get_pixel(x, y).0;
            assert!(p[3] >= 254 && p[1] >= 126 && p[0] <= 1, "pixel ({x}, {y}) is {p:?}");
        }
        assert_eq!(image.get_pixel(2, 2).0[3], 0, "nothing outside the sprite");
    }

    #[test]
    fn images_at_whole_scales_keep_their_hard_edges() {
        // A 1x2 image (green over transparent) drawn 2x: with `exact_scale` each texel fills its
        // 2x2 device pixels; without it, linear filtering bleeds a quarter of the green into the
        // first transparent row.
        use gpui::{
            AtlasKey, Bounds, ContentMask, Corners, ImageId, PaddedBool32, PolychromeSprite,
            RenderImageParams, ScaledPixels, TransformationMatrix,
        };
        let draw = |exact_scale: u32| {
            let mut renderer = MetalHeadlessRenderer::new();
            renderer.renderer.update_transparency(true);
            let atlas = gpui::PlatformHeadlessRenderer::sprite_atlas(&renderer);
            // BGRA rows: opaque green, then transparent.
            let pixels: [u8; 8] = [0, 128, 0, 255, 0, 0, 0, 0];
            let tile = atlas
                .get_or_insert_with(
                    AtlasKey::Image(RenderImageParams { image_id: ImageId(2), frame_index: 0 }),
                    &mut || Ok(Some((size(DevicePixels(1), DevicePixels(2)), std::borrow::Cow::Borrowed(&pixels[..])))),
                )
                .expect("insert")
                .expect("tile");
            let all = Bounds::new(point(ScaledPixels(0.), ScaledPixels(0.)), size(ScaledPixels(8.), ScaledPixels(8.)));
            let mut scene = Scene::default();
            scene.insert_primitive(PolychromeSprite {
                order: 0,
                exact_scale,
                tile,
                grayscale: PaddedBool32::from(false),
                opacity: 1.,
                bounds: Bounds::new(point(ScaledPixels(2.), ScaledPixels(2.)), size(ScaledPixels(2.), ScaledPixels(4.))),
                content_mask: ContentMask::from(all),
                corner_radii: Corners::default(),
                transformation: TransformationMatrix::unit(),
            });
            scene.finish();
            gpui::PlatformHeadlessRenderer::render_scene_to_image(&mut renderer, &scene, size(DevicePixels(8), DevicePixels(8)))
                .expect("render")
        };
        let exact = draw(1);
        for y in [2, 3] {
            let p = exact.get_pixel(2, y).0;
            assert!(p[3] >= 254 && p[1] >= 126, "green row at y {y}: {p:?}");
        }
        for y in [4, 5] {
            assert_eq!(exact.get_pixel(2, y).0[3], 0, "transparent row at y {y}");
        }
        // Filtered, the device row just past the edge picks up green.
        assert!(draw(0).get_pixel(2, 4).0[3] > 0, "linear filtering blends across the edge");
    }

    #[test]
    fn backdrop_filters_keep_transparent_backdrops_transparent() {
        use gpui::{BackdropFilter, Bounds, ContentMask, Corners, ScaledPixels};
        let mut renderer = MetalHeadlessRenderer::new();
        renderer.renderer.update_transparency(true);
        let bounds = Bounds::new(
            point(ScaledPixels(8.), ScaledPixels(8.)),
            size(ScaledPixels(16.), ScaledPixels(16.)),
        );
        let mut scene = Scene::default();
        scene.insert_primitive(BackdropFilter {
            bounds,
            content_mask: ContentMask::from(bounds),
            corner_radii: Corners::default(),
            blur_radius: ScaledPixels(2.),
            opacity: 1.,
            saturate: 1.,
            brightness: 1.,
            // Mid gray where there is color; nothing where there is none.
            contrast: 0.,
            order: 0,
            shader: gpui::ShaderDraw::NONE,
        });
        scene.finish();
        let image = gpui::PlatformHeadlessRenderer::render_scene_to_image(
            &mut renderer,
            &scene,
            size(DevicePixels(32), DevicePixels(32)),
        )
        .expect("render");
        assert_eq!(image.get_pixel(16, 16).0, [0, 0, 0, 0]);
    }

    #[test]
    fn backdrop_filters_render() {
        use gpui::backdrop_filter_test_scene as t;
        let mut renderer = MetalHeadlessRenderer::new();
        let (scene, probes) = t::scene();
        let image = gpui::PlatformHeadlessRenderer::render_scene_to_image(
            &mut renderer,
            &scene,
            size(DevicePixels(t::SIZE.0), DevicePixels(t::SIZE.1)),
        )
        .expect("render");
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image
                .save(std::path::Path::new(&dir).join("backdrop_metal.png"))
                .expect("dump");
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 12);
    }

    #[test]
    fn gradients_render() {
        use gpui::gradient_test_scene as t;
        let mut renderer = MetalHeadlessRenderer::new();
        let (scene, probes) = t::scene();
        let image = gpui::PlatformHeadlessRenderer::render_scene_to_image(
            &mut renderer,
            &scene,
            size(DevicePixels(t::SIZE.0), DevicePixels(t::SIZE.1)),
        )
        .expect("render");
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image
                .save(std::path::Path::new(&dir).join("gradients_metal.png"))
                .expect("dump");
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 2);
    }

    /// Gradients against solid fills: 200 overlapping quarter-screen quads with rounded
    /// corners at 2880x1800 (about 260 million shaded pixels). Fills alternate each
    /// round; each line is the median GPU time of a frame. Run with
    /// `GPUI_RETAINED_SHAPES_STATS=1 cargo test -p gpui_apple --release --lib gradient_frame_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gradient_frame_cost() {
        use gpui::gradient_test_scene::{BenchFill, bench};
        assert!(
            retained_stats::enabled(),
            "set GPUI_RETAINED_SHAPES_STATS=1 to record GPU times"
        );
        let (width, height) = (2880, 1800);
        let size = size(DevicePixels(width), DevicePixels(height));
        let mut renderer = MetalHeadlessRenderer::new();
        let fills = [
            BenchFill::Solid,
            BenchFill::TwoStop,
            BenchFill::Linear(2),
            BenchFill::Linear(8),
            BenchFill::Radial(4),
            BenchFill::Radial(16),
        ];
        let scenes: Vec<Scene> = fills
            .iter()
            .map(|fill| bench(200, width as f32, height as f32, *fill))
            .collect();
        let mut gpu = vec![Vec::new(); scenes.len()];
        for _ in 0..60 {
            for (scene, times) in scenes.iter().zip(&mut gpu) {
                gpui::PlatformHeadlessRenderer::render_scene_to_image(&mut renderer, scene, size)
                    .expect("render");
                // The completion handler runs after the wait returns: let it record.
                let mut recorded = retained_stats::take_gpu();
                for _ in 0..100 {
                    if !recorded.is_empty() {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    recorded = retained_stats::take_gpu();
                }
                times.extend(recorded);
            }
        }
        for (fill, times) in fills.iter().zip(&mut gpu) {
            times.sort_by(f64::total_cmp);
            println!(
                "{fill:?}: GPU {:.3} ms per frame (median of {})",
                times[times.len() / 2],
                times.len()
            );
        }
    }

    /// Frame costs with and without compositing groups. Run with
    /// `cargo test -p gpui_apple --release --lib group_frame_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn group_frame_cost() {
        use gpui::compositing_group_bench_scene as b;
        let (width, height) = (2880, 1800);
        let size = size(DevicePixels(width), DevicePixels(height));
        let mut renderer = MetalHeadlessRenderer::new();
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let time = |f: &mut dyn FnMut()| {
            let start = std::time::Instant::now();
            f();
            start.elapsed().as_secs_f64() * 1e3
        };
        // Scene building (CPU) without groups: 10,000 quads.
        let build: Vec<f64> = (0..200)
            .map(|_| time(&mut || drop(b::flat(10_000, width as f32, height as f32))))
            .collect();
        println!("build 10k quads, no groups: {:.3} ms", median(build));
        let flat = b::flat(10_000, width as f32, height as f32);
        let encode: Vec<f64> = (0..200)
            .map(|_| {
                time(&mut || {
                    gpui::PlatformHeadlessRenderer::render_scene(&mut renderer, &flat, size)
                        .expect("render");
                })
            })
            .collect();
        println!("encode 10k quads, no groups: {:.3} ms", median(encode));
        for depth in [0, 1, 10] {
            let scene = b::nested(depth, 2_000, width as f32, height as f32);
            assert_eq!(scene.group_depth(), depth);
            let frames: Vec<f64> = (0..40)
                .map(|_| {
                    time(&mut || {
                        gpui::PlatformHeadlessRenderer::render_scene_to_image(
                            &mut renderer,
                            &scene,
                            size,
                        )
                        .expect("render");
                    })
                })
                .collect();
            println!(
                "{depth} nested groups, 2k quads, {width}x{height}: {:.3} ms per frame (rendered and read back), {} group image bytes",
                median(frames),
                renderer.renderer.group_texture_bytes()
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
            let frames: Vec<f64> = (0..40)
                .map(|_| {
                    time(&mut || {
                        gpui::PlatformHeadlessRenderer::render_scene_to_image(
                            &mut renderer,
                            &scene,
                            size,
                        )
                        .expect("render");
                    })
                })
                .collect();
            println!(
                "{label}: {:.3} ms per frame, {} group image bytes",
                median(frames),
                renderer.renderer.group_texture_bytes()
            );
        }
    }

    /// More than 64 uncompleted command buffers (`newCommandQueue`'s limit) on the shared
    /// queue: the first waits on an event no one has signaled, so none completes while the
    /// rest are made. With the old limit the 65th blocks forever.
    #[test]
    fn the_shared_queue_holds_more_than_64_uncompleted_command_buffers() {
        struct Send<T>(T);
        unsafe impl<T> std::marker::Send for Send<T> {}
        let device = preferred_device();
        let queue = Send(shared_command_queue(&device));
        let event = device.new_shared_event();
        let first = queue.0.new_command_buffer().to_owned();
        first.encode_wait_for_event(&event, 1);
        first.commit();
        let (done, made) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let queue = queue;
            let mut buffers = Vec::new();
            for _ in 0..200 {
                let buffer = queue.0.new_command_buffer().to_owned();
                buffer.commit();
                buffers.push(buffer);
            }
            done.send(Send(buffers)).ok();
        });
        // With room, they are made at once. With `newCommandQueue`'s 64, the 65th waits until
        // Metal's watchdog fails the first (about 5 s on macOS 27), so 2 s tells them apart.
        let result = made.recv_timeout(std::time::Duration::from_secs(2));
        // Let them all complete either way.
        event.set_signaled_value(1);
        let buffers = result.expect("200 uncompleted command buffers were made without blocking");
        for buffer in &buffers.0 {
            buffer.wait_until_completed();
        }
        first.wait_until_completed();
    }

    /// External texture `id`'s pixels as RGBA, read on the shared queue.
    fn external_rgba(atlas: &MetalAtlas, id: gpui::ExternalTextureId) -> (u32, Vec<u8>) {
        use gpui::PlatformAtlas as _;
        let tile = atlas.external_texture_tile(id).expect("the texture exists");
        let texture = atlas.metal_texture(tile.texture_id).unwrap();
        let (w, h) = (texture.width(), texture.height());
        let device = texture.device().to_owned();
        let buffer = device.new_buffer(w * h * 4, metal::MTLResourceOptions::StorageModeShared);
        let cb = shared_command_queue(&device)
            .new_command_buffer()
            .to_owned();
        let blit = cb.new_blit_command_encoder();
        blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            metal::MTLOrigin { x: 0, y: 0, z: 0 },
            metal::MTLSize::new(w, h, 1),
            &buffer,
            0,
            w * 4,
            0,
            metal::MTLBlitOption::empty(),
        );
        blit.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let bgra = unsafe {
            std::slice::from_raw_parts(buffer.contents() as *const u8, (w * h * 4) as usize)
        };
        let rgba = bgra
            .chunks(4)
            .flat_map(|p| [p[2], p[1], p[0], p[3]])
            .collect();
        (w as u32, rgba)
    }

    #[test]
    fn texture_targets_draw_into_their_textures_not_the_window() {
        use gpui::PlatformAtlas as _;
        use gpui::texture_target_test_scene as t;
        let mut renderer = MetalHeadlessRenderer::new();
        let atlas = renderer.renderer.sprite_atlas().clone();
        for (id, size) in t::targets() {
            atlas
                .set_external_texture(id, size, gpui::TextureSampling::Linear)
                .unwrap();
        }
        let (scene, window, textures) = t::scene();
        let size = size(DevicePixels(t::SIZE.0), DevicePixels(t::SIZE.1));
        let image =
            gpui::PlatformHeadlessRenderer::render_scene_to_image(&mut renderer, &scene, size)
                .expect("render");
        t::check(|x, y| image.get_pixel(x, y).0, &window, 3);
        for (id, probes) in textures {
            let (width, rgba) = external_rgba(&atlas, id);
            t::check(
                |x, y| {
                    let i = ((y * width + x) * 4) as usize;
                    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
                },
                &probes,
                3,
            );
        }
    }

    /// A `draw` with nothing to draw into (no layer, so no drawable) does not count as rendered,
    /// so the window keeps the frame's texture targets for the next frame.
    #[test]
    fn a_draw_without_a_drawable_is_not_rendered() {
        let mut renderer = MetalHeadlessRenderer::new();
        let mut scene = Scene::default();
        scene.finish();
        renderer.renderer.draw(&scene);
        assert!(!renderer.renderer.rendered());
    }

    #[test]
    fn compositing_groups_render_and_free_their_images() {
        use gpui::compositing_group_test_scene as t;
        let mut renderer = MetalHeadlessRenderer::new();
        let (scene, probes) = t::scene();
        let size = size(DevicePixels(t::SIZE.0), DevicePixels(t::SIZE.1));
        let image =
            gpui::PlatformHeadlessRenderer::render_scene_to_image(&mut renderer, &scene, size)
                .expect("render");
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image
                .save(std::path::Path::new(&dir).join("groups_metal.png"))
                .expect("dump");
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
        // Images sized to the groups' bounds (in steps of 64 px) and the blur's copies,
        // reused across the frame: well under one window-sized image per group.
        let kept = renderer.renderer.group_texture_bytes();
        assert!(kept > 0 && kept <= 6 * 64 * 64 * 4, "{kept} bytes");
        // A frame without groups frees them.
        let mut empty = Scene::default();
        empty.finish();
        gpui::PlatformHeadlessRenderer::render_scene_to_image(&mut renderer, &empty, size)
            .expect("render");
        assert_eq!(renderer.renderer.group_texture_bytes(), 0);
    }

    #[test]
    fn texture_pool_reuses_released_textures_and_frees_unused_ones() {
        let renderer = MetalHeadlessRenderer::new();
        let device = &renderer.renderer.device;
        let pool = TexturePool::default();
        let one = 64 * 64 * 4;
        let a = pool.acquire(device, 64, 64);
        let b = pool.acquire(device, 64, 64);
        assert!(
            !std::ptr::eq(a.as_ref(), b.as_ref()),
            "both in use: two textures"
        );
        pool.release(&a);
        let c = pool.acquire(device, 64, 64);
        assert!(std::ptr::eq(a.as_ref(), c.as_ref()), "released: reused");
        let d = pool.acquire(device, 128, 64);
        assert_eq!(pool.bytes(), 4 * one);
        pool.end_frame();
        assert_eq!(pool.bytes(), 4 * one, "all used this frame: kept");
        let e = pool.acquire(device, 64, 64);
        pool.release(&e);
        pool.end_frame();
        assert_eq!(pool.bytes(), one, "unused last frame: freed");
        pool.end_frame();
        assert_eq!(pool.bytes(), 0);
        drop((b, d));
    }

    #[test]
    fn group_content_draws_in_image_coordinates() {
        use gpui::compositing_group_content_test_scene as t;
        let mut renderer = MetalHeadlessRenderer::new();
        let (scene, probes) = t::scene();
        let image = gpui::PlatformHeadlessRenderer::render_scene_to_image(
            &mut renderer,
            &scene,
            size(DevicePixels(t::SIZE.0), DevicePixels(t::SIZE.1)),
        )
        .expect("render");
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image
                .save(std::path::Path::new(&dir).join("group_content_metal.png"))
                .expect("dump");
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
    }

    #[test]
    fn group_masks_render() {
        use gpui::group_mask_test_scene as t;
        let mut renderer = MetalHeadlessRenderer::new();
        let (scene, probes) = t::scene();
        let image = gpui::PlatformHeadlessRenderer::render_scene_to_image(
            &mut renderer,
            &scene,
            size(DevicePixels(t::SIZE.0), DevicePixels(t::SIZE.1)),
        )
        .expect("render");
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image
                .save(std::path::Path::new(&dir).join("group_masks_metal.png"))
                .expect("dump");
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
    }

    /// GPU cost of rounded content masks: `count` overlapping clipped quads (each the
    /// size of a card, overfilling a rounded or a square mask) rendered and waited for,
    /// alternating the two, at 2x. Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn rounded_clip_cost() {
        use gpui::{ContentMask, Corners, Quad, ScaledPixels, Scene};
        let (width, height) = (1600i32, 1200i32);
        let scene = |count: usize, radius: f32| {
            let mut scene = Scene::default();
            for i in 0..count {
                let (x, y) = ((i % 12) as f32 * 130., ((i / 12) % 9) as f32 * 130.);
                let bounds = |inset: f32| {
                    Bounds::new(
                        point(ScaledPixels(x + inset), ScaledPixels(y + inset)),
                        size(
                            ScaledPixels(240. - 2. * inset),
                            ScaledPixels(240. - 2. * inset),
                        ),
                    )
                };
                scene.insert_primitive(Quad {
                    bounds: bounds(0.),
                    content_mask: ContentMask {
                        bounds: bounds(8.),
                        corner_radii: Corners::all(ScaledPixels(radius)),
                    },
                    background: gpui::Hsla {
                        h: 0.6,
                        s: 0.5,
                        l: 0.5,
                        a: 1.,
                    }
                    .into(),
                    ..Default::default()
                });
            }
            scene.finish();
            scene
        };
        let mut renderer = MetalHeadlessRenderer::new();
        let viewport = size(DevicePixels(width), DevicePixels(height));
        for count in [100usize, 1000] {
            let (square, rounded) = (scene(count, 0.), scene(count, 24.));
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..40 {
                for (k, s) in [&square, &rounded].into_iter().enumerate() {
                    let t = std::time::Instant::now();
                    gpui::PlatformHeadlessRenderer::render_scene_to_image(
                        &mut renderer,
                        s,
                        viewport,
                    )
                    .expect("render");
                    if round >= 5 {
                        times[k].push(t.elapsed().as_secs_f64() * 1000.);
                    }
                }
            }
            let median = |v: &mut Vec<f64>| {
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            let (sq, ro) = (median(&mut times[0]), median(&mut times[1]));
            println!(
                "{count} clipped quads (224 px masks): square {sq:.3} ms, rounded {ro:.3} ms, {:.2} us per clipped node",
                (ro - sq) * 1000. / count as f64
            );
        }
    }

    #[test]
    fn rounded_clips_render() {
        use gpui::rounded_clip_test_scene as t;
        let mut renderer = MetalHeadlessRenderer::new();
        let (scene, probes) = t::scene();
        assert!(probes.len() > 60, "{} probes", probes.len());
        let image = gpui::PlatformHeadlessRenderer::render_scene_to_image(
            &mut renderer,
            &scene,
            size(DevicePixels(t::SIZE.0), DevicePixels(t::SIZE.1)),
        )
        .expect("render");
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image
                .save(std::path::Path::new(&dir).join("rounded_clips_metal.png"))
                .expect("dump");
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
    }

    #[test]
    fn corner_shapes_render() {
        use gpui::corner_shape_test_scene as t;
        let mut renderer = MetalHeadlessRenderer::new();
        let (scene, probes) = t::scene();
        assert!(probes.len() > 40, "{} probes", probes.len());
        let image = gpui::PlatformHeadlessRenderer::render_scene_to_image(
            &mut renderer,
            &scene,
            size(DevicePixels(t::SIZE.0), DevicePixels(t::SIZE.1)),
        )
        .expect("render");
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image
                .save(std::path::Path::new(&dir).join("corner_shapes_metal.png"))
                .expect("dump");
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
    }

    #[test]
    fn elliptical_corners_render() {
        use gpui::elliptical_corner_test_scene as t;
        let mut renderer = MetalHeadlessRenderer::new();
        let (scene, probes) = t::scene();
        assert!(probes.len() > 300, "{} probes", probes.len());
        let image = gpui::PlatformHeadlessRenderer::render_scene_to_image(
            &mut renderer,
            &scene,
            size(DevicePixels(t::SIZE.0), DevicePixels(t::SIZE.1)),
        )
        .expect("render");
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image
                .save(std::path::Path::new(&dir).join("elliptical_corners_metal.png"))
                .expect("dump");
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
    }

    #[test]
    fn blend_modes_match_the_reference() {
        use gpui::blend_mode_test_scene as t;
        let mut renderer = MetalHeadlessRenderer::new();
        let (scene, probes) = t::scene();
        let image = gpui::PlatformHeadlessRenderer::render_scene_to_image(
            &mut renderer,
            &scene,
            size(DevicePixels(t::SIZE.0), DevicePixels(t::SIZE.1)),
        )
        .expect("render");
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image
                .save(std::path::Path::new(&dir).join("blend_metal.png"))
                .expect("dump");
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
    }

    #[test]
    fn custom_shaders_render() {
        use gpui::custom_shader_test_scene as t;
        let mut renderer = MetalHeadlessRenderer::new();
        let (scene, probes) = t::scene();
        let image = gpui::PlatformHeadlessRenderer::render_scene_to_image(
            &mut renderer,
            &scene,
            size(DevicePixels(t::SIZE.0), DevicePixels(t::SIZE.1)),
        )
        .expect("render");
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image
                .save(std::path::Path::new(&dir).join("custom_shaders_metal.png"))
                .expect("dump");
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 4);
    }

    fn marks_buffer(device: &metal::DeviceRef, bytes: &[u8]) -> gpui::MarksBuffer {
        let buffer = device.new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        gpui::MarksBuffer::new(Arc::new(buffer))
    }

    /// A program Metal cannot build says so: [`gpui::MarksShader::compile_error`],
    /// and a panic in debug builds. So does a draw bound to a buffer smaller
    /// than its program reads.
    #[test]
    fn marks_failures_are_reported() {
        use gpui::marks_test_scene as t;
        let px = size(DevicePixels(t::SIZE.0), DevicePixels(t::SIZE.1));
        let mut renderer = MetalHeadlessRenderer::new();
        let buffer = marks_buffer(&renderer.renderer.device, &t::buffer_bytes());
        let broken = t::broken("metal");
        t::expect_report(|| {
            let scene = t::one_draw(broken.clone(), buffer.clone());
            let _ = gpui::PlatformHeadlessRenderer::render_scene_to_image(&mut renderer, &scene, px);
        });
        let error = broken.compile_error().expect("the failure is recorded");
        assert!(error.contains("target"), "{error}");

        let mut renderer = MetalHeadlessRenderer::new();
        let small = marks_buffer(&renderer.renderer.device, &[0u8; 16]);
        let rects = t::rects("metal small buffer");
        t::expect_report(|| {
            let scene = t::one_draw(rects.clone(), small);
            let _ = gpui::PlatformHeadlessRenderer::render_scene_to_image(&mut renderer, &scene, px);
        });
        let (skipped, why) = rects.skipped_draws();
        assert!(skipped >= 1 && why.as_deref().is_some_and(|w| w.contains("32 bytes")), "{skipped} {why:?}");
        assert_eq!(t::rects("metal fine").compile_error(), None);
    }

    #[test]
    fn marks_render() {
        use gpui::marks_test_scene as t;
        let mut renderer = MetalHeadlessRenderer::new();
        let buffer = marks_buffer(&renderer.renderer.device, &t::buffer_bytes());
        let (scene, probes) = t::scene(buffer);
        let image = gpui::PlatformHeadlessRenderer::render_scene_to_image(
            &mut renderer,
            &scene,
            size(DevicePixels(t::SIZE.0), DevicePixels(t::SIZE.1)),
        )
        .expect("render");
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image
                .save(std::path::Path::new(&dir).join("marks_metal.png"))
                .expect("dump");
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
    }

    /// Marks draw inside the frame's pass: a scene with marks between two quads
    /// opens as many render passes as the same scene without them.
    #[test]
    fn marks_open_no_render_pass() {
        use gpui::marks_test_scene as t;
        let mut renderer = MetalHeadlessRenderer::new();
        let buffer = marks_buffer(&renderer.renderer.device, &t::buffer_bytes());
        let (with_marks, _) = t::scene(buffer);
        assert_eq!(with_marks.marks.len(), 3);
        let mut quads_only = Scene::default();
        let mut with = Scene::default();
        let backdrop = with_marks.quads[0];
        let mut cover = backdrop;
        cover.bounds = Bounds::new(
            point(ScaledPixels(8.), ScaledPixels(8.)),
            gpui::size(ScaledPixels(8.), ScaledPixels(8.)),
        );
        quads_only.insert_primitive(backdrop);
        quads_only.insert_primitive(cover);
        quads_only.finish();
        with.insert_primitive(backdrop);
        for draw in &with_marks.marks {
            with.insert_primitive(draw.clone());
        }
        // Over the marks: its own batch after them.
        with.insert_primitive(cover);
        with.finish();
        let labels: Vec<String> = with.batches().map(|b| b.label()).collect();
        assert_eq!(labels, ["quads (1)", "marks (3)", "quads (1)"]);
        let size = size(DevicePixels(t::SIZE.0), DevicePixels(t::SIZE.1));
        let mut passes = |scene: &Scene| {
            let before = render_passes::count();
            gpui::PlatformHeadlessRenderer::render_scene_to_image(&mut renderer, scene, size)
                .expect("render");
            render_passes::count() - before
        };
        let base = passes(&quads_only);
        assert_eq!(passes(&with), base, "marks opened a render pass");
    }

    /// The marks gate (r-marks §17 P0): 1M instanced points pulled from three f32
    /// storage buffers and a color table, as the spike's scatter scene draws them,
    /// drawn inside GPUI's frame pass at 2800x1800. Prints GPU time per frame (the
    /// command buffer: clear + the draw) next to a clear-only frame, CPU encode,
    /// encode per extra mark draw, and render passes per frame.
    /// `cargo test -p gpui_apple --release --lib marks_points_frame_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn marks_points_frame_cost() {
        use gpui::{MarksBindings, MarksDraw, MarksPaint, MarksShader, TransformationMatrix};
        const PROGRAM: &str = "
@group(0) @binding(2) var<storage, read> xs: array<f32>;
@group(0) @binding(3) var<storage, read> ys: array<f32>;
@group(0) @binding(4) var<storage, read> vs: array<f32>;
@group(0) @binding(5) var<storage, read> lut: array<vec4<f32>>;
struct Points { point_px: f32, d0: f32, d1: f32, pad: f32 }
@group(0) @binding(1) var<uniform> points: Points;

fn mark_vertex(vertex: u32, instance: u32) -> MarkVertex {
    let c = gpui_marks_strip_corner(vertex) * 2.0 - 1.0;
    let r = points.point_px * 0.5 + 0.5;
    var out = gpui_mark_vertex(vec2<f32>(xs[instance], ys[instance]));
    out.offset_px = c * r;
    out.local = vec4<f32>(c * r, 0.0, 0.0);
    let t = clamp((vs[instance] - points.d0) / max(points.d1 - points.d0, 1e-30), 0.0, 1.0);
    out.color = lut[u32(t * 255.0 + 0.5)];
    return out;
}

fn mark_fragment(input: MarkFragment) -> vec4<f32> {
    let a = clamp(points.point_px * 0.5 - length(input.local.xy) + 0.5, 0.0, 1.0);
    return vec4<f32>(input.color.rgb, input.color.a * a);
}
";
        let n: usize = std::env::var("MARKS_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000_000);
        // The spike's columns: half uniform, half in 16 clusters; value 0.7x + 0.3y.
        let mut seed = 7u64;
        let mut unit = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        let (mut xs, mut ys, mut vs) = (vec![], vec![], vec![]);
        for i in 0..n {
            let (x, y) = if i % 2 == 0 {
                (unit(), unit())
            } else {
                let c = (unit() * 16.0).floor();
                let (cx, cy) = (
                    (c * 0.37).fract() * 0.8 + 0.1,
                    (c * 0.61).fract() * 0.8 + 0.1,
                );
                let mut g = || (unit() + unit() + unit() - 1.5) * 0.05;
                (cx + g(), cy + g())
            };
            xs.push(x);
            ys.push(y);
            vs.push(x * 0.7 + y * 0.3);
        }
        let lut: Vec<f32> = (0..256)
            .flat_map(|i| {
                let t = i as f32 / 255.0;
                [t, 0.4 + 0.5 * t * (1.0 - t), 1.0 - t, 1.0]
            })
            .collect();
        let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_ne_bytes()).collect() };
        let mut renderer =
            MetalRenderer::new_headless(Arc::new(Mutex::new(InstanceBufferPool::default())));
        let device = renderer.device.clone();
        let buffers: Vec<gpui::MarksBuffer> = [&xs, &ys, &vs, &lut]
            .map(|v| marks_buffer(&device, &bytes(v)))
            .into();
        let uniforms: Arc<[u8]> = bytes(&[3.0, 0.0, 1.0, 0.0]).into();
        let shader = MarksShader::new(PROGRAM);
        let (w, h) = (2800, 1800);
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(w as u64);
        descriptor.set_height(h as u64);
        descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        descriptor.set_usage(metal::MTLTextureUsage::RenderTarget);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        let texture = device.new_texture(&descriptor);
        let target = size(DevicePixels(w), DevicePixels(h));
        let full = Bounds::new(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            gpui::size(ScaledPixels(w as f32), ScaledPixels(h as f32)),
        );
        let draw = |instances: std::ops::Range<u32>, camera: TransformationMatrix| MarksDraw {
            order: 0,
            bounds: full,
            content_mask: ContentMask::from(full),
            shader: shader.clone(),
            bindings: MarksBindings::buffers(buffers.iter().cloned()).uniforms(uniforms.clone()),
            paint: MarksPaint::new(instances, 4, camera).strip(),
            opacity: 1.0,
        };
        let camera = |zoom: f32, pan: f32| TransformationMatrix {
            rotation_scale: [[w as f32 * zoom, 0.0], [0.0, h as f32 * zoom]],
            translation: [-pan, -pan * 0.5],
        };
        let q = |v: &mut Vec<f64>, p: f64| {
            v.sort_by(f64::total_cmp);
            v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
        };
        // One frame: encode `scene`, commit, wait. Returns (encode ms, GPU ms, passes).
        let mut frame = |scene: &Scene| -> (f64, f64, usize) {
            objc2::rc::autoreleasepool(|_| {
                let passes = render_passes::count();
                let started = std::time::Instant::now();
                let mut writer = InstanceBufferWriter::new(
                    &renderer.device,
                    &renderer.instance_buffer_pool,
                    renderer.is_unified_memory,
                );
                let gradients = write_gradients(scene, &mut writer).unwrap();
                let bindings = write_instances(&gradients, scene, &mut writer).unwrap();
                let cb = renderer
                    .draw_primitives_to_texture(scene, &bindings, &mut writer, &texture, target)
                    .unwrap();
                let encoded = started.elapsed().as_secs_f64() * 1e3;
                cb.commit();
                cb.wait_until_completed();
                let (start, end): (f64, f64) = unsafe {
                    let cb = &*(cb.as_ptr() as *const AnyObject);
                    (
                        objc2::msg_send![cb, GPUStartTime],
                        objc2::msg_send![cb, GPUEndTime],
                    )
                };
                drop(writer);
                (
                    encoded,
                    (end - start) * 1e3,
                    render_passes::count() - passes,
                )
            })
        };
        let frames = 120;
        let mut run = |what: &str, make: &dyn Fn(usize) -> Scene| {
            let (mut encode, mut gpu, mut passes) = (vec![], vec![], 0);
            for i in 0..frames + 10 {
                let scene = make(i);
                let (e, g, p) = frame(&scene);
                if i >= 10 {
                    encode.push(e);
                    gpu.push(g);
                    passes = passes.max(p);
                }
            }
            println!(
                "MARKS_P0 {what:<36} n {n} target {w}x{h} encode_cpu_ms p50 {:.4} p95 {:.4} gpu_ms p50 {:.3} p95 {:.3} render_passes {passes}",
                q(&mut encode, 0.5),
                q(&mut encode, 0.95),
                q(&mut gpu, 0.5),
                q(&mut gpu, 0.95),
            );
            (q(&mut encode, 0.5), q(&mut gpu, 0.5))
        };
        let scene_of = |draws: Vec<MarksDraw>| {
            let mut scene = Scene::default();
            for d in draws {
                scene.insert_primitive(d);
            }
            scene.finish();
            scene
        };
        run("clear only (empty scene)", &|_| scene_of(vec![]));
        run("1M points, pan at fit", &|i| {
            scene_of(vec![draw(0..n as u32, camera(1.0, (i % 40) as f32 * 3.0))])
        });
        run("1M points, pan zoomed 4x", &|i| {
            scene_of(vec![draw(
                0..n as u32,
                camera(4.0, (i % 40) as f32 * 3.0 + w as f32 * 1.5),
            )])
        });
        // Encode per mark: 1 against 101 one-instance draws (each its own draw call).
        let (one, _) = run("1 one-point mark draw", &|_| {
            scene_of(vec![draw(0..1, camera(1.0, 0.0))])
        });
        let (many, _) = run("101 one-point mark draws", &|_| {
            scene_of((0..101).map(|k| draw(k..k + 1, camera(1.0, 0.0))).collect())
        });
        println!(
            "MARKS_P0 encode per extra mark draw {:.2} us (p50s)",
            (many - one) / 100.0 * 1e3
        );
    }

    #[test]
    fn element_transformations_render() {
        use gpui::element_transformation_test_scene as t;
        let mut renderer = MetalHeadlessRenderer::new();
        let (scene, probes) = t::scene();
        let image = gpui::PlatformHeadlessRenderer::render_scene_to_image(
            &mut renderer,
            &scene,
            size(DevicePixels(t::SIZE.0), DevicePixels(t::SIZE.1)),
        )
        .expect("render");
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image
                .save(std::path::Path::new(&dir).join("transformed_metal.png"))
                .expect("dump");
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
    }

    #[test]
    fn retained_shapes_render_and_upload_only_changes() {
        use gpui::retained_shapes_test_scene as t;
        let mut renderer = MetalHeadlessRenderer::new();
        let shapes = gpui::RetainedShapes::new();
        let (scene, probes) = t::scene(&shapes);
        let size = size(DevicePixels(t::SIZE.0), DevicePixels(t::SIZE.1));
        let image =
            gpui::PlatformHeadlessRenderer::render_scene_to_image(&mut renderer, &scene, size)
                .expect("render");
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            image
                .save(std::path::Path::new(&dir).join("retained_metal.png"))
                .expect("dump");
        }
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);
        let gpu = &renderer.renderer.retained_shapes[&shapes.id()];
        assert_eq!((gpu.len, gpu.version), (5, shapes.version()));

        // A range write: the next frame shows it.
        let mut red = t::instances()[0];
        red.fill = 0x0000ffff;
        shapes.write(0, &[red]);
        let image =
            gpui::PlatformHeadlessRenderer::render_scene_to_image(&mut renderer, &scene, size)
                .expect("render");
        assert_eq!(image.get_pixel(16, 16).0, [0, 0, 255, 255]);
        assert_eq!(
            renderer.renderer.retained_shapes[&shapes.id()].version,
            shapes.version()
        );

        // Arrow parts and text bars.
        let kinds = gpui::RetainedShapes::new();
        let (scene, probes) = t::kinds_scene(&kinds);
        let image =
            gpui::PlatformHeadlessRenderer::render_scene_to_image(&mut renderer, &scene, size)
                .expect("render");
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);

        // Four corner radii.
        let corners = gpui::RetainedShapes::new();
        let (scene, probes) = t::corners_scene(&corners);
        let image =
            gpui::PlatformHeadlessRenderer::render_scene_to_image(&mut renderer, &scene, size)
                .expect("render");
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);

        // Rounded, rotated and layer-moved clips.
        let clipped = gpui::RetainedShapes::new();
        let (scene, probes) = t::clip_scene(&clipped);
        let image =
            gpui::PlatformHeadlessRenderer::render_scene_to_image(&mut renderer, &scene, size)
                .expect("render");
        t::check(|x, y| image.get_pixel(x, y).0, &probes, 3);

        // The dense scene the wgpu test compares against.
        if let Some(dir) = std::env::var_os("GPUI_RETAINED_DUMP") {
            let dense = gpui::RetainedShapes::new();
            let (scene, (w, h)) = t::random_scene(&dense, 400, 7);
            let image = gpui::PlatformHeadlessRenderer::render_scene_to_image(
                &mut renderer,
                &scene,
                gpui::size(DevicePixels(w), DevicePixels(h)),
            )
            .expect("render");
            image
                .save(std::path::Path::new(&dir).join("retained_random_metal.png"))
                .expect("dump");
        }
    }
    use objc2_core_foundation::{CFDictionary, CFType};
    use objc2_core_video::{
        CVPixelBufferCreate, CVPixelBufferGetBaseAddressOfPlane,
        CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
        CVPixelBufferUnlockBaseAddress, kCVPixelBufferIOSurfacePropertiesKey,
    };

    #[test]
    fn render_core_video_surface() {
        let attributes = CFDictionary::from_slices(
            &[unsafe { kCVPixelBufferIOSurfacePropertiesKey }],
            &[&*CFDictionary::<CFType, CFType>::empty()],
        );
        let mut image_buffer = ptr::null_mut();
        assert_eq!(
            unsafe {
                CVPixelBufferCreate(
                    None,
                    16,
                    16,
                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                    Some(attributes.as_opaque()),
                    ptr::NonNull::from(&mut image_buffer),
                )
            },
            kCVReturnSuccess
        );
        let image_buffer =
            unsafe { CFRetained::from_raw(ptr::NonNull::new(image_buffer).unwrap()) };
        unsafe {
            assert_eq!(
                CVPixelBufferLockBaseAddress(&image_buffer, CVPixelBufferLockFlags::empty()),
                kCVReturnSuccess
            );
            for plane in 0..2 {
                let bytes = CVPixelBufferGetBaseAddressOfPlane(&image_buffer, plane).cast::<u8>();
                let length = CVPixelBufferGetBytesPerRowOfPlane(&image_buffer, plane)
                    * CVPixelBufferGetHeightOfPlane(&image_buffer, plane);
                slice::from_raw_parts_mut(bytes, length).fill(128);
            }
            assert_eq!(
                CVPixelBufferUnlockBaseAddress(&image_buffer, CVPixelBufferLockFlags::empty()),
                kCVReturnSuccess
            );
        }

        let bounds = Bounds {
            origin: Point::default(),
            size: size(ScaledPixels(16.0), ScaledPixels(16.0)),
        };
        let mut scene = Scene::default();
        scene.insert_primitive(PaintSurface {
            order: 0,
            bounds,
            content_mask: ContentMask::from(bounds),
            image_buffer: image_buffer.clone(),
        });
        drop(image_buffer);
        scene.finish();

        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(Default::default())));
        let image = renderer
            .render_scene_to_image(&scene, size(DevicePixels(16), DevicePixels(16)))
            .unwrap();
        for image::Rgba([red, green, blue, alpha]) in image.pixels() {
            assert!(red.abs_diff(128) <= 2);
            assert!(green.abs_diff(128) <= 2);
            assert!(blue.abs_diff(128) <= 2);
            assert_eq!(*alpha, 255);
        }
    }
}

/// A retained shape buffer's GPU copy.
struct RetainedGpu {
    buffer: Option<metal::Buffer>,
    capacity: usize,
    /// Instances in the GPU copy.
    len: usize,
    version: u64,
    last_frame: u64,
}

#[allow(clippy::too_many_arguments)]
fn copy_ranges<'a>(
    writer: &mut InstanceBufferWriter,
    command_buffer: &'a metal::CommandBufferRef,
    blit: &mut Option<&'a metal::BlitCommandEncoderRef>,
    entry: &RetainedGpu,
    data: &[ShapeInstance],
    ranges: &[std::ops::Range<usize>],
    uploaded: &mut usize,
    result: &mut Result<()>,
) {
    let stride = mem::size_of::<ShapeInstance>();
    let Some(dst) = &entry.buffer else { return };
    let encoder = *blit.get_or_insert_with(|| command_buffer.new_blit_command_encoder());
    for range in ranges {
        let range = range.start.min(data.len())..range.end.min(data.len());
        if range.is_empty() {
            continue;
        }
        let staged = match writer.write(&data[range.clone()]) {
            Ok(b) => b,
            Err(e) => {
                *result = Err(e);
                return;
            }
        };
        encoder.copy_from_buffer(
            &staged.buffer,
            staged.offset as u64,
            dst,
            (range.start * stride) as u64,
            (range.len() * stride) as u64,
        );
        *uploaded += range.len() * stride;
    }
}

/// Opt-in frame statistics for retained shapes (`GPUI_RETAINED_SHAPES_STATS=1`):
/// every 120 frames, stderr gets CPU encode time, GPU time and bytes uploaded.
/// Render passes opened for a texture, counted in tests: marks must open none.
#[cfg(test)]
mod render_passes {
    use std::cell::Cell;
    thread_local!(static OPENED: Cell<usize> = const { Cell::new(0) });
    pub fn opened() {
        OPENED.with(|n| n.set(n.get() + 1));
    }
    /// Passes opened on this thread so far.
    pub fn count() -> usize {
        OPENED.with(Cell::get)
    }
}

mod retained_stats {
    use parking_lot::Mutex;
    use std::sync::OnceLock;

    #[derive(Default)]
    struct Stats {
        encode: Vec<f64>,
        gpu: Vec<f64>,
        uploaded: Vec<usize>,
    }

    static STATS: Mutex<Option<Stats>> = Mutex::new(None);

    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("GPUI_RETAINED_SHAPES_STATS").is_some())
    }

    fn with(f: impl FnOnce(&mut Stats)) {
        if enabled() {
            f(STATS.lock().get_or_insert_with(Stats::default));
        }
    }

    pub fn record_upload(bytes: usize) {
        with(|s| s.uploaded.push(bytes));
    }

    pub fn record_gpu(ms: f64) {
        with(|s| s.gpu.push(ms));
    }

    /// The GPU times recorded so far, cleared (benchmarks).
    #[cfg(test)]
    pub fn take_gpu() -> Vec<f64> {
        STATS
            .lock()
            .as_mut()
            .map(|s| std::mem::take(&mut s.gpu))
            .unwrap_or_default()
    }

    pub fn record_encode(ms: f64) {
        with(|s| {
            s.encode.push(ms);
            if s.encode.len() >= 120 {
                let q = |v: &mut Vec<f64>, p: f64| {
                    if v.is_empty() {
                        return 0.0;
                    }
                    v.sort_by(f64::total_cmp);
                    v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
                };
                let uploads = s.uploaded.iter().filter(|b| **b > 0).count();
                let bytes: usize = s.uploaded.iter().sum();
                eprintln!(
                    "RETAINED_STATS frames {} encode_cpu_ms p50 {:.3} p95 {:.3} gpu_ms p50 {:.3} p95 {:.3} upload_frames {} upload_bytes {}",
                    s.encode.len(),
                    q(&mut s.encode, 0.5),
                    q(&mut s.encode, 0.95),
                    q(&mut s.gpu, 0.5),
                    q(&mut s.gpu, 0.95),
                    uploads,
                    bytes
                );
                *s = Stats::default();
            }
        });
    }
}
