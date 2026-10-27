use anyhow::{Context as _, Result};
use collections::FxHashMap;
use etagere::{BucketedAtlasAllocator, size2};
use gpui::{
    AtlasBackend, AtlasKey, AtlasState, AtlasTextureId, AtlasTextureKind, AtlasTextureList,
    AtlasTile, Bounds, DevicePixels, ExternalTextureId, PlatformAtlas, Point, Size,
    TextureSampling,
};
use parking_lot::Mutex;
use std::{borrow::Cow, ops, sync::Arc};

use crate::WgpuContext;

fn device_size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    size2(size.width.0, size.height.0)
}

fn etagere_point_to_device(point: etagere::Point) -> Point<DevicePixels> {
    Point {
        x: DevicePixels(point.x),
        y: DevicePixels(point.y),
    }
}

pub struct WgpuAtlas(Mutex<AtlasState<WgpuAtlasTextures>>);

struct PendingUpload {
    id: AtlasTextureId,
    bounds: Bounds<DevicePixels>,
    data: Vec<u8>,
}

struct WgpuAtlasTextures {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    max_texture_size: u32,
    color_texture_format: wgpu::TextureFormat,
    storage: WgpuAtlasStorage,
    pending_uploads: Vec<PendingUpload>,
    next_texture_generation: u64,
    /// External textures: each a whole polychrome texture of its own.
    externals: FxHashMap<ExternalTextureId, AtlasTextureId>,
}

pub struct WgpuTextureInfo {
    pub view: wgpu::TextureView,
    /// Distinguishes new textures that reuse an [`AtlasTextureId`].
    pub generation: u64,
    /// How the texture is sampled (external textures choose; the rest are linear).
    pub sampling: TextureSampling,
}

impl WgpuAtlas {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        color_texture_format: wgpu::TextureFormat,
    ) -> Self {
        let max_texture_size = device.limits().max_texture_dimension_2d;
        WgpuAtlas(Mutex::new(AtlasState::new(WgpuAtlasTextures {
            device,
            queue,
            max_texture_size,
            color_texture_format,
            storage: WgpuAtlasStorage::default(),
            pending_uploads: Vec::new(),
            next_texture_generation: 0,
            externals: FxHashMap::default(),
        })))
    }

    pub fn from_context(context: &WgpuContext) -> Self {
        Self::new(
            context.device.clone(),
            context.queue.clone(),
            context.color_texture_format(),
        )
    }

    pub fn before_frame(&self) {
        let mut lock = self.0.lock();
        lock.backend.flush_uploads();
    }

    /// Returns the view backing `id`, or `None` once every tile in it has been
    /// removed. A scene can still reference such a texture when a cached view
    /// replays a paint from before the image was dropped, so callers must skip
    /// those sprites rather than assume the texture exists.
    pub fn get_texture_info(&self, id: AtlasTextureId) -> Option<WgpuTextureInfo> {
        let lock = self.0.lock();
        let texture = lock.backend.storage.get(id)?;
        Some(WgpuTextureInfo {
            view: texture.view.clone(),
            generation: texture.generation,
            sampling: texture.external.unwrap_or_default(),
        })
    }

    /// Clears all cached textures and tiles, forcing them to be recreated.
    /// Use this for incremental recovery when the device is still valid.
    pub fn clear(&self) {
        self.0.lock().clear(|textures| {
            textures.storage = WgpuAtlasStorage::default();
            textures.pending_uploads.clear();
            textures.externals.clear();
        });
    }

    /// Handles device lost by clearing all textures and cached tiles.
    /// The atlas will lazily recreate textures as needed on subsequent frames.
    pub fn handle_device_lost(&self, context: &WgpuContext) {
        self.0.lock().clear(|textures| {
            textures.device = context.device.clone();
            textures.queue = context.queue.clone();
            textures.color_texture_format = context.color_texture_format();
            textures.storage = WgpuAtlasStorage::default();
            textures.pending_uploads.clear();
            textures.externals.clear();
        });
    }
}

impl PlatformAtlas for WgpuAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        self.0.lock().get_or_insert_with(key, build)
    }

    fn remove(&self, key: &AtlasKey) {
        self.0.lock().remove(key);
    }

    fn set_external_texture(
        &self,
        id: ExternalTextureId,
        size: Size<DevicePixels>,
        sampling: TextureSampling,
    ) -> Result<bool> {
        self.0.lock().backend.set_external(id, size, sampling)
    }

    fn external_texture_tile(&self, id: ExternalTextureId) -> Option<AtlasTile> {
        let lock = self.0.lock();
        let textures = &lock.backend;
        let texture = textures.storage.get(*textures.externals.get(&id)?)?;
        Some(texture.whole_tile())
    }

    fn copy_to_external_texture(
        &self,
        id: ExternalTextureId,
        source: &dyn std::any::Any,
    ) -> Result<()> {
        self.0.lock().backend.copy_external(id, source)
    }

    /// Takes a `wgpu::Texture` made on this atlas's device, in its color format, with
    /// `TEXTURE_BINDING` usage, and shows it as external texture `id` itself: no copy. Its
    /// producer submits its writes on this atlas's queue (`WgpuContext::queue`) before the
    /// frame that reads them.
    fn share_external_texture(
        &self,
        id: ExternalTextureId,
        source: &dyn std::any::Any,
        sampling: TextureSampling,
    ) -> Result<()> {
        self.0.lock().backend.share_external(id, source, sampling)
    }

    fn remove_external_texture(&self, id: ExternalTextureId) {
        self.0.lock().backend.remove_external(id);
    }

    fn external_texture_bytes(&self) -> u64 {
        let lock = self.0.lock();
        let textures = &lock.backend;
        textures
            .externals
            .values()
            .filter_map(|id| textures.storage.get(*id))
            .map(|t| {
                let s = t.texture.size();
                u64::from(s.width) * u64::from(s.height) * u64::from(t.bytes_per_pixel())
            })
            .sum()
    }
}

impl AtlasBackend for WgpuAtlasTextures {
    fn insert(
        &mut self,
        kind: AtlasTextureKind,
        size: Size<DevicePixels>,
        bytes: &[u8],
    ) -> Result<AtlasTile> {
        let tile = self.allocate(size, kind).context("failed to allocate")?;
        self.upload_texture(tile.texture_id, tile.bounds, bytes);
        Ok(tile)
    }

    fn remove(&mut self, tile: AtlasTile) {
        let id = tile.texture_id;
        let Some(texture_slot) = self.storage[id.kind].textures.get_mut(id.index as usize) else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                self.pending_uploads
                    .retain(|upload| upload.id != texture.id);
                self.storage[id.kind]
                    .free_list
                    .push(texture.id.index as usize);
            } else {
                *texture_slot = Some(texture);
            }
        }
    }
}

impl WgpuAtlasTextures {
    /// See [`PlatformAtlas::set_external_texture`].
    fn set_external(
        &mut self,
        id: ExternalTextureId,
        size: Size<DevicePixels>,
        sampling: TextureSampling,
    ) -> Result<bool> {
        let max = self.max_texture_size as i32;
        anyhow::ensure!(
            size.width.0 > 0 && size.height.0 > 0 && size.width.0 <= max && size.height.0 <= max,
            "external texture {}x{} is outside 1..={max} pixels a side",
            size.width.0,
            size.height.0
        );
        if let Some(texture_id) = self.externals.get(&id).copied() {
            let generation = self.next_texture_generation;
            let list = &mut self.storage[texture_id.kind];
            if let Some(texture) = list
                .textures
                .get_mut(texture_id.index as usize)
                .and_then(|t| t.as_mut())
            {
                let current = texture.texture.size();
                if current.width == size.width.0 as u32 && current.height == size.height.0 as u32 {
                    // Same size: keep the pixels; a new sampling only rebinds.
                    if texture.external != Some(sampling) {
                        texture.external = Some(sampling);
                        texture.generation = generation;
                        self.next_texture_generation = generation.wrapping_add(1);
                    }
                    return Ok(false);
                }
            }
        }
        self.remove_external(id);
        let format = self.color_texture_format;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("external_texture"),
            size: wgpu::Extent3d {
                width: size.width.0 as u32,
                height: size.height.0 as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // Copies from external images render into the texture.
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.insert_external(id, texture, size, sampling);
        Ok(true)
    }

    /// See [`PlatformAtlas::share_external_texture`].
    fn share_external(
        &mut self,
        id: ExternalTextureId,
        source: &dyn std::any::Any,
        sampling: TextureSampling,
    ) -> Result<()> {
        let source = source
            .downcast_ref::<wgpu::Texture>()
            .context("a wgpu external texture shares a wgpu::Texture")?;
        anyhow::ensure!(
            source.format() == self.color_texture_format,
            "an external texture shares a {:?} texture, not {:?}",
            self.color_texture_format,
            source.format()
        );
        anyhow::ensure!(
            source
                .usage()
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
                && source.dimension() == wgpu::TextureDimension::D2
                && source.sample_count() == 1,
            "an external texture shares a single-sampled 2D texture with TEXTURE_BINDING usage"
        );
        self.ensure_own_device(source)?;
        if let Some(texture_id) = self.externals.get(&id).copied() {
            let generation = self.next_texture_generation;
            if let Some(texture) = self.storage[texture_id.kind]
                .textures
                .get_mut(texture_id.index as usize)
                .and_then(|t| t.as_mut())
                && texture.texture == *source
            {
                // The same texture: its pixels are already the producer's; only rebind.
                if texture.external != Some(sampling) {
                    texture.external = Some(sampling);
                    texture.generation = generation;
                    self.next_texture_generation = generation.wrapping_add(1);
                }
                return Ok(());
            }
        }
        let extent = source.size();
        let size = Size {
            width: DevicePixels(extent.width as i32),
            height: DevicePixels(extent.height as i32),
        };
        self.remove_external(id);
        self.insert_external(id, source.clone(), size, sampling);
        Ok(())
    }

    /// Refuses a texture from another device (as Metal's atlas does): binding it would
    /// fail validation in a later frame. wgpu has no device getter, so this binds a
    /// view in a validation error scope. On the web, where an error scope resolves
    /// asynchronously, WebGPU reports a foreign texture when a frame binds it.
    #[cfg(not(target_family = "wasm"))]
    fn ensure_own_device(&self, texture: &wgpu::Texture) -> Result<()> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("share_external_device_check"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        drop(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("share_external_device_check"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        }));
        // On native backends the scope's result is ready when it is popped.
        let error = poll_ready(scope.pop()).flatten();
        anyhow::ensure!(
            error.is_none(),
            "an external texture shares a texture on its own wgpu device: {}",
            error.map(|e| e.to_string()).unwrap_or_default()
        );
        Ok(())
    }

    #[cfg(target_family = "wasm")]
    fn ensure_own_device(&self, _texture: &wgpu::Texture) -> Result<()> {
        Ok(())
    }

    /// Records `texture` as external texture `id`: one whole polychrome texture.
    fn insert_external(
        &mut self,
        id: ExternalTextureId,
        texture: wgpu::Texture,
        size: Size<DevicePixels>,
        sampling: TextureSampling,
    ) {
        let format = texture.format();
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let kind = AtlasTextureKind::Polychrome;
        let list = &mut self.storage[kind];
        let index = list.free_list.pop();
        let generation = self.next_texture_generation;
        self.next_texture_generation = self.next_texture_generation.wrapping_add(1);
        let texture_id = AtlasTextureId {
            index: index.unwrap_or(list.textures.len()) as u32,
            kind,
        };
        let entry = WgpuAtlasTexture {
            id: texture_id,
            generation,
            allocator: BucketedAtlasAllocator::new(device_size_to_etagere(size)),
            format,
            texture,
            view,
            live_atlas_keys: 0,
            external: Some(sampling),
        };
        match index {
            Some(ix) => list.textures[ix] = Some(entry),
            None => list.textures.push(Some(entry)),
        }
        self.externals.insert(id, texture_id);
    }

    fn remove_external(&mut self, id: ExternalTextureId) {
        let Some(texture_id) = self.externals.remove(&id) else {
            return;
        };
        let list = &mut self.storage[texture_id.kind];
        if let Some(slot) = list.textures.get_mut(texture_id.index as usize)
            && slot.take().is_some()
        {
            list.free_list.push(texture_id.index as usize);
        }
    }

    #[cfg(target_family = "wasm")]
    fn copy_external(&mut self, id: ExternalTextureId, source: &dyn std::any::Any) -> Result<()> {
        use wgpu::ExternalImageSource as Source;
        let texture_id = *self
            .externals
            .get(&id)
            .context("no such external texture")?;
        let texture = self
            .storage
            .get(texture_id)
            .context("the external texture was lost")?;
        let source = if let Some(c) = source.downcast_ref::<web_sys::OffscreenCanvas>() {
            Source::OffscreenCanvas(c.clone())
        } else if let Some(c) = source.downcast_ref::<web_sys::HtmlCanvasElement>() {
            Source::HTMLCanvasElement(c.clone())
        } else if let Some(b) = source.downcast_ref::<web_sys::ImageBitmap>() {
            Source::ImageBitmap(b.clone())
        } else {
            anyhow::bail!(
                "an external texture copies from an OffscreenCanvas, HtmlCanvasElement or ImageBitmap"
            );
        };
        let size = texture.texture.size();
        self.queue.copy_external_image_to_texture(
            &wgpu::CopyExternalImageSourceInfo {
                source,
                origin: wgpu::Origin2d::ZERO,
                flip_y: false,
            },
            wgpu::CopyExternalImageDestInfo {
                texture: &texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
                color_space: wgpu::PredefinedColorSpace::Srgb,
                // Sprites take straight alpha, as the atlas stores images.
                premultiplied_alpha: false,
            },
            size,
        );
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    fn copy_external(&mut self, _id: ExternalTextureId, _source: &dyn std::any::Any) -> Result<()> {
        anyhow::bail!("external image copies are a web feature")
    }

    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        {
            let textures = &mut self.storage[texture_kind];

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind);
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> &mut WgpuAtlasTexture {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };
        let max_texture_size = self.max_texture_size as i32;
        let max_atlas_size = Size {
            width: DevicePixels(max_texture_size),
            height: DevicePixels(max_texture_size),
        };

        let size = min_size.min(&max_atlas_size).max(&DEFAULT_ATLAS_SIZE);
        let format = match kind {
            AtlasTextureKind::Monochrome => wgpu::TextureFormat::R8Unorm,
            AtlasTextureKind::Subpixel | AtlasTextureKind::Polychrome => self.color_texture_format,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas"),
            size: wgpu::Extent3d {
                width: size.width.0 as u32,
                height: size.height.0 as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_list = &mut self.storage[kind];
        let index = texture_list.free_list.pop();
        let generation = self.next_texture_generation;
        self.next_texture_generation = self.next_texture_generation.wrapping_add(1);

        let atlas_texture = WgpuAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            generation,
            allocator: BucketedAtlasAllocator::new(device_size_to_etagere(size)),
            format,
            texture,
            view,
            live_atlas_keys: 0,
            external: None,
        };

        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list
                .textures
                .get_mut(ix)
                .and_then(|t| t.as_mut())
                .expect("texture must exist")
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list
                .textures
                .last_mut()
                .and_then(|t| t.as_mut())
                .expect("texture must exist")
        }
    }

    fn upload_texture(&mut self, id: AtlasTextureId, bounds: Bounds<DevicePixels>, bytes: &[u8]) {
        let data = self
            .storage
            .get(id)
            .map(|texture| swizzle_upload_data(bytes, texture.format))
            .unwrap_or_else(|| bytes.to_vec());

        self.pending_uploads
            .push(PendingUpload { id, bounds, data });
    }

    fn flush_uploads(&mut self) {
        for upload in self.pending_uploads.drain(..) {
            let Some(texture) = self.storage.get(upload.id) else {
                continue;
            };
            let bytes_per_pixel = texture.bytes_per_pixel();

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: upload.bounds.origin.x.0 as u32,
                        y: upload.bounds.origin.y.0 as u32,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &upload.data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(upload.bounds.size.width.0 as u32 * bytes_per_pixel as u32),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: upload.bounds.size.width.0 as u32,
                    height: upload.bounds.size.height.0 as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

#[derive(Default)]
struct WgpuAtlasStorage {
    monochrome_textures: AtlasTextureList<WgpuAtlasTexture>,
    subpixel_textures: AtlasTextureList<WgpuAtlasTexture>,
    polychrome_textures: AtlasTextureList<WgpuAtlasTexture>,
}

impl ops::Index<AtlasTextureKind> for WgpuAtlasStorage {
    type Output = AtlasTextureList<WgpuAtlasTexture>;
    fn index(&self, kind: AtlasTextureKind) -> &Self::Output {
        match kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Subpixel => &self.subpixel_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
        }
    }
}

impl ops::IndexMut<AtlasTextureKind> for WgpuAtlasStorage {
    fn index_mut(&mut self, kind: AtlasTextureKind) -> &mut Self::Output {
        match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
        }
    }
}

impl WgpuAtlasStorage {
    fn get(&self, id: AtlasTextureId) -> Option<&WgpuAtlasTexture> {
        self[id.kind]
            .textures
            .get(id.index as usize)
            .and_then(|t| t.as_ref())
    }
}

struct WgpuAtlasTexture {
    id: AtlasTextureId,
    generation: u64,
    allocator: BucketedAtlasAllocator,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    format: wgpu::TextureFormat,
    live_atlas_keys: u32,
    /// An external texture (and its sampling): one whole tile, never shared.
    external: Option<TextureSampling>,
}

impl WgpuAtlasTexture {
    /// The tile covering all of the texture (an external texture's).
    fn whole_tile(&self) -> AtlasTile {
        let size = self.texture.size();
        AtlasTile {
            texture_id: self.id,
            tile_id: gpui::TileId(0),
            padding: 0,
            bounds: Bounds {
                origin: Point::default(),
                size: Size {
                    width: DevicePixels(size.width as i32),
                    height: DevicePixels(size.height as i32),
                },
            },
        }
    }

    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        if self.external.is_some() {
            return None;
        }
        let allocation = self.allocator.allocate(device_size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            padding: 0,
            bounds: Bounds {
                origin: etagere_point_to_device(allocation.rectangle.min),
                size,
            },
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn bytes_per_pixel(&self) -> u8 {
        match self.format {
            wgpu::TextureFormat::R8Unorm => 1,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm => 4,
            _ => 4,
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&self) -> bool {
        self.live_atlas_keys == 0
    }
}

fn swizzle_upload_data(bytes: &[u8], format: wgpu::TextureFormat) -> Vec<u8> {
    match format {
        wgpu::TextureFormat::Rgba8Unorm => {
            let mut data = bytes.to_vec();
            for pixel in data.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
            data
        }
        _ => bytes.to_vec(),
    }
}

/// Polls `future` once: `None` if it is not ready.
#[cfg(not(target_family = "wasm"))]
fn poll_ready<F: std::future::Future>(future: F) -> Option<F::Output> {
    use std::task::{Context, Poll, Waker};
    let mut future = std::pin::pin!(future);
    match future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(output) => Some(output),
        Poll::Pending => None,
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use gpui::block_on;
    use gpui::{ImageId, RenderImageParams};
    use std::sync::Arc;

    fn test_device_and_queue() -> anyhow::Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                flags: wgpu::InstanceFlags::default(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: None,
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                    apply_limit_buckets: false,
                })
                .await
                .map_err(|error| anyhow::anyhow!("failed to request adapter: {error}"))?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("wgpu_atlas_test_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults()
                        .using_resolution(adapter.limits())
                        .using_alignment(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    default_queue: wgpu::QueueDescriptor::default(),
                })
                .await
                .map_err(|error| anyhow::anyhow!("failed to request device: {error}"))?;
            Ok((Arc::new(device), Arc::new(queue)))
        })
    }

    #[test]
    fn before_frame_skips_uploads_for_removed_texture() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;

        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
        let key = AtlasKey::Image(RenderImageParams {
            image_id: ImageId(1),
            frame_index: 0,
        });
        let size = Size {
            width: DevicePixels(1),
            height: DevicePixels(1),
        };
        let mut build = || Ok(Some((size, Cow::Owned(vec![0, 0, 0, 255]))));

        // Regression test: before the fix, this panicked in flush_uploads
        atlas
            .get_or_insert_with(key.clone(), &mut build)?
            .expect("tile should be created");
        atlas.remove(&key);
        atlas.before_frame();
        Ok(())
    }

    #[test]
    fn remove_deallocates_tile_space_for_reuse() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };
        let big = Size {
            width: DevicePixels(700),
            height: DevicePixels(700),
        };

        let make_key = |image_id: usize| {
            AtlasKey::Image(RenderImageParams {
                image_id: ImageId(image_id),
                frame_index: 0,
            })
        };
        let insert = |key: AtlasKey, size: Size<DevicePixels>| {
            let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
            atlas
                .get_or_insert_with(key, &mut || {
                    Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
                })
                .expect("allocation should succeed")
                .expect("callback returns Some")
        };

        let keeper_key = make_key(1);
        let big_key_a = make_key(2);
        let big_key_b = make_key(3);

        let keeper_tile = insert(keeper_key, small);
        let tile_a = insert(big_key_a.clone(), big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);
        let tile_b = insert(big_key_b, big);
        assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
        Ok(())
    }

    #[test]
    fn reused_texture_id_has_new_generation() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
        let size = Size {
            width: DevicePixels(700),
            height: DevicePixels(700),
        };
        let make_key = |image_id| {
            AtlasKey::Image(RenderImageParams {
                image_id: ImageId(image_id),
                frame_index: 0,
            })
        };
        let insert = |key: AtlasKey| {
            atlas
                .get_or_insert_with(key, &mut || {
                    Ok(Some((
                        size,
                        Cow::Owned(vec![0; size.width.0 as usize * size.height.0 as usize * 4]),
                    )))
                })
                .expect("allocation should succeed")
                .expect("callback returns Some")
        };

        let first_key = make_key(1);
        let first_tile = insert(first_key.clone());
        let first_generation = atlas
            .get_texture_info(first_tile.texture_id)
            .context("first texture should exist")?
            .generation;
        atlas.remove(&first_key);
        assert!(atlas.get_texture_info(first_tile.texture_id).is_none());

        let second_tile = insert(make_key(2));
        let second_generation = atlas
            .get_texture_info(second_tile.texture_id)
            .context("second texture should exist")?
            .generation;

        assert_eq!(second_tile.texture_id, first_tile.texture_id);
        assert_ne!(second_generation, first_generation);
        Ok(())
    }

    #[test]
    fn swizzle_upload_data_preserves_bgra_uploads() {
        let input = vec![0x10, 0x20, 0x30, 0x40];
        assert_eq!(
            swizzle_upload_data(&input, wgpu::TextureFormat::Bgra8Unorm),
            input
        );
    }

    #[test]
    fn swizzle_upload_data_converts_bgra_to_rgba() {
        let input = vec![0x10, 0x20, 0x30, 0x40, 0xAA, 0xBB, 0xCC, 0xDD];
        assert_eq!(
            swizzle_upload_data(&input, wgpu::TextureFormat::Rgba8Unorm),
            vec![0x30, 0x20, 0x10, 0x40, 0xCC, 0xBB, 0xAA, 0xDD]
        );
    }

    #[test]
    fn external_textures_are_whole_owned_tiles() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
        let id = ExternalTextureId(7);
        let size = |w, h| Size {
            width: DevicePixels(w),
            height: DevicePixels(h),
        };
        assert!(atlas.external_texture_tile(id).is_none());
        assert!(atlas.set_external_texture(id, size(300, 150), TextureSampling::Linear)?);
        let tile = atlas.external_texture_tile(id).expect("set makes it");
        assert_eq!(tile.bounds.size, size(300, 150));
        assert_eq!(atlas.external_texture_bytes(), 300 * 150 * 4);
        // The same size and sampling keep it; a new size or sampling remakes it.
        assert!(!atlas.set_external_texture(id, size(300, 150), TextureSampling::Linear)?);
        let before = atlas.get_texture_info(tile.texture_id).unwrap().generation;
        assert!(!atlas.set_external_texture(id, size(300, 150), TextureSampling::Nearest)?);
        assert_eq!(atlas.external_texture_tile(id).unwrap().texture_id, tile.texture_id, "a new sampling keeps the texture");
        assert_ne!(atlas.get_texture_info(tile.texture_id).unwrap().generation, before, "and rebinds it");
        let info = atlas
            .get_texture_info(atlas.external_texture_tile(id).unwrap().texture_id)
            .unwrap();
        assert_eq!(info.sampling, TextureSampling::Nearest);
        assert!(atlas.set_external_texture(id, size(64, 32), TextureSampling::Nearest)?);
        assert_eq!(atlas.external_texture_bytes(), 64 * 32 * 4);
        // Images never land in an external texture.
        let key = AtlasKey::Image(RenderImageParams {
            image_id: ImageId(1),
            frame_index: 0,
        });
        let image = atlas
            .get_or_insert_with(key, &mut || {
                Ok(Some((size(8, 8), Cow::Owned(vec![0; 256]))))
            })?
            .unwrap();
        assert_ne!(
            image.texture_id,
            atlas.external_texture_tile(id).unwrap().texture_id
        );
        assert!(
            atlas
                .set_external_texture(id, size(0, 4), TextureSampling::Linear)
                .is_err()
        );
        atlas.remove_external_texture(id);
        assert!(atlas.external_texture_tile(id).is_none());
        assert_eq!(atlas.external_texture_bytes(), 0);
        Ok(())
    }
    /// A producer's texture: drawn into, sampled and read back.
    fn producer_texture(
        device: &wgpu::Device,
        w: u32,
        h: u32,
        format: wgpu::TextureFormat,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("producer"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    /// Clears `texture` to `color` in a render pass submitted on `queue`, without waiting.
    fn clear(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        color: wgpu::Color,
    ) {
        let view = texture.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        queue.submit([encoder.finish()]);
    }

    /// The pixels of external texture `id` as the atlas holds it (a copy submitted on `queue`
    /// after everything before it; only this readback waits).
    fn external_pixels(
        atlas: &WgpuAtlas,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: ExternalTextureId,
    ) -> Vec<u8> {
        let texture = {
            let lock = atlas.0.lock();
            let textures = &lock.backend;
            textures
                .storage
                .get(*textures.externals.get(&id).unwrap())
                .unwrap()
                .texture
                .clone()
        };
        let size = texture.size();
        let row = (size.width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row * size.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: None,
                },
            },
            size,
        );
        queue.submit([encoder.finish()]);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        let data = buffer.slice(..).get_mapped_range().unwrap();
        let mut out = Vec::new();
        for y in 0..size.height {
            let start = (y * row) as usize;
            out.extend_from_slice(&data[start..start + (size.width * 4) as usize]);
        }
        out
    }

    #[test]
    fn shared_external_textures_are_the_producers_textures_with_no_copy() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let format = wgpu::TextureFormat::Bgra8Unorm;
        let atlas = WgpuAtlas::new(device.clone(), queue.clone(), format);
        let id = ExternalTextureId(3);
        let source = producer_texture(&device, 6, 4, format);
        clear(&device, &queue, &source, wgpu::Color::RED);
        atlas.share_external_texture(id, &source, TextureSampling::Linear)?;

        let tile = atlas.external_texture_tile(id).expect("shared");
        assert_eq!(tile.bounds.size.width, DevicePixels(6));
        assert_eq!(tile.bounds.size.height, DevicePixels(4));
        {
            let lock = atlas.0.lock();
            let shown = &lock.backend.storage.get(tile.texture_id).unwrap().texture;
            assert!(
                *shown == source,
                "the atlas shows the producer's texture itself"
            );
        }
        // BGRA bytes of red, read after the producer's submit with no wait between them.
        let pixels = external_pixels(&atlas, &device, &queue, id);
        assert!(pixels.chunks(4).all(|p| p == [0, 0, 255, 255]));

        // New pixels in the same texture: nothing to copy.
        clear(&device, &queue, &source, wgpu::Color::BLUE);
        let pixels = external_pixels(&atlas, &device, &queue, id);
        assert!(pixels.chunks(4).all(|p| p == [255, 0, 0, 255]));

        // Sharing it again with a new sampling keeps the texture and rebinds it.
        let before = atlas.get_texture_info(tile.texture_id).unwrap().generation;
        atlas.share_external_texture(id, &source, TextureSampling::Nearest)?;
        let again = atlas.external_texture_tile(id).unwrap();
        assert_eq!(again.texture_id, tile.texture_id);
        let info = atlas.get_texture_info(again.texture_id).unwrap();
        assert_eq!(info.sampling, TextureSampling::Nearest);
        assert_ne!(info.generation, before);

        // Another texture replaces it, as a new generation.
        let bigger = producer_texture(&device, 8, 8, format);
        atlas.share_external_texture(id, &bigger, TextureSampling::Linear)?;
        let replaced = atlas.external_texture_tile(id).unwrap();
        assert_eq!(replaced.bounds.size.width, DevicePixels(8));
        assert_ne!(
            atlas
                .get_texture_info(replaced.texture_id)
                .unwrap()
                .generation,
            info.generation
        );

        // A texture from another device is refused.
        let (other_device, _) = test_device_and_queue()?;
        let foreign = producer_texture(&other_device, 4, 4, format);
        assert!(
            atlas
                .share_external_texture(ExternalTextureId(5), &foreign, TextureSampling::Linear)
                .is_err()
        );
        assert!(atlas.external_texture_tile(ExternalTextureId(5)).is_none());

        // Only textures in the atlas's format with TEXTURE_BINDING, and only wgpu textures.
        let rgba = producer_texture(&device, 2, 2, wgpu::TextureFormat::Rgba8Unorm);
        assert!(
            atlas
                .share_external_texture(ExternalTextureId(4), &rgba, TextureSampling::Linear)
                .is_err()
        );
        assert!(
            atlas
                .share_external_texture(ExternalTextureId(4), &5u32, TextureSampling::Linear)
                .is_err()
        );
        assert!(atlas.external_texture_tile(ExternalTextureId(4)).is_none());

        atlas.remove_external_texture(id);
        assert!(atlas.external_texture_tile(id).is_none());
        Ok(())
    }
}
