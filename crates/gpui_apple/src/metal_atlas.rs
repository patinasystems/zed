use anyhow::{Context as _, Result};
use derive_more::{Deref, DerefMut};
use etagere::BucketedAtlasAllocator;
use foreign_types::ForeignType as _;
use gpui::{
    AtlasBackend, AtlasKey, AtlasState, AtlasTextureId, AtlasTextureKind, AtlasTextureList,
    AtlasTile, Bounds, DevicePixels, ExternalTextureId, PlatformAtlas, Point, Size,
    TextureSampling,
};
use metal::Device;
use parking_lot::Mutex;
use std::{borrow::Cow, collections::HashMap};

/// The largest texture side on all modern Apple GPUs.
const MAX_TEXTURE_SIDE: i32 = 16384;

pub struct MetalAtlas(Mutex<AtlasState<MetalAtlasTextures>>);

impl MetalAtlas {
    pub(crate) fn new(device: Device, supports_shared_storage: bool) -> Self {
        MetalAtlas(Mutex::new(AtlasState::new(MetalAtlasTextures {
            device: AssertSend(device),
            supports_shared_storage,
            monochrome_textures: Default::default(),
            polychrome_textures: Default::default(),
            externals: HashMap::new(),
        })))
    }

    /// How texture `id` is sampled: an external texture's own choice, else linear.
    pub(crate) fn sampling(&self, id: AtlasTextureId) -> TextureSampling {
        self.0
            .lock()
            .backend
            .texture(id)
            .and_then(|t| t.external)
            .unwrap_or_default()
    }

    /// Returns the GPU texture backing `id`, or `None` once every tile in it
    /// has been removed. A scene can still reference such a texture when a
    /// cached view replays a paint from before the image was dropped, so
    /// callers must skip those sprites rather than assume the texture exists.
    pub(crate) fn metal_texture(&self, id: AtlasTextureId) -> Option<metal::Texture> {
        Some(self.0.lock().backend.texture(id)?.metal_texture.clone())
    }
}

struct MetalAtlasTextures {
    device: AssertSend<Device>,
    supports_shared_storage: bool,
    monochrome_textures: AtlasTextureList<MetalAtlasTexture>,
    polychrome_textures: AtlasTextureList<MetalAtlasTexture>,
    /// External textures: each a whole polychrome texture of its own.
    externals: HashMap<ExternalTextureId, AtlasTextureId>,
}

impl PlatformAtlas for MetalAtlas {
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
        let texture = textures.texture(*textures.externals.get(&id)?)?;
        Some(texture.whole_tile())
    }

    /// Takes a `metal::Texture` of the same pixel format (BGRA8Unorm, straight
    /// alpha), on this atlas's device; copies the overlap of the two sizes and
    /// waits for the copy, so a frame encoded after it reads the new pixels.
    fn copy_to_external_texture(
        &self,
        id: ExternalTextureId,
        source: &dyn std::any::Any,
    ) -> Result<()> {
        self.0.lock().backend.copy_external(id, source)
    }

    /// Takes a `metal::Texture` (BGRA8Unorm, straight alpha, shader-readable) on this atlas's
    /// device and shows it as external texture `id` itself: no copy. Its producer encodes its
    /// writes on [`crate::metal_renderer::shared_command_queue`], so a frame committed after them
    /// reads them.
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
            .filter_map(|id| textures.texture(*id))
            .map(|t| t.metal_texture.width() * t.metal_texture.height() * u64::from(t.bytes_per_pixel()))
            .sum()
    }
}

impl AtlasBackend for MetalAtlasTextures {
    fn insert(
        &mut self,
        kind: AtlasTextureKind,
        size: Size<DevicePixels>,
        bytes: &[u8],
    ) -> Result<AtlasTile> {
        let tile = self.allocate(size, kind).context("failed to allocate")?;
        let texture = self
            .texture(tile.texture_id)
            .context("allocated tile refers to a missing texture")?;
        texture.upload(tile.bounds, bytes);
        Ok(tile)
    }

    fn remove(&mut self, tile: AtlasTile) {
        let id = tile.texture_id;

        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        };

        let Some(texture_slot) = textures
            .textures
            .iter_mut()
            .find(|texture| texture.as_ref().is_some_and(|v| v.id == id))
        else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                textures.free_list.push(id.index as usize);
            } else {
                *texture_slot = Some(texture);
            }
        }
    }
}

impl MetalAtlasTextures {
    /// See [`PlatformAtlas::set_external_texture`].
    fn set_external(
        &mut self,
        id: ExternalTextureId,
        size: Size<DevicePixels>,
        sampling: TextureSampling,
    ) -> Result<bool> {
        anyhow::ensure!(
            size.width.0 > 0
                && size.height.0 > 0
                && size.width.0 <= MAX_TEXTURE_SIDE
                && size.height.0 <= MAX_TEXTURE_SIDE,
            "external texture {}x{} is outside 1..={MAX_TEXTURE_SIDE} pixels a side",
            size.width.0,
            size.height.0
        );
        if let Some(texture_id) = self.externals.get(&id).copied()
            && let Some(texture) = self
                .polychrome_textures
                .textures
                .get_mut(texture_id.index as usize)
                .and_then(|t| t.as_mut())
            && texture.metal_texture.width() == size.width.0 as u64
            && texture.metal_texture.height() == size.height.0 as u64
        {
            // Same size: keep the pixels; a new sampling only rebinds.
            texture.external = Some(sampling);
            return Ok(false);
        }
        self.remove_external(id);
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(size.width.0 as u64);
        descriptor.set_height(size.height.0 as u64);
        descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        // Drawn into by texture targets (`Window::paint_to_texture`).
        descriptor
            .set_usage(metal::MTLTextureUsage::ShaderRead | metal::MTLTextureUsage::RenderTarget);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        let metal_texture = self.device.new_texture(&descriptor);
        // A new texture starts transparent.
        let bytes = (size.width.0 as u64) * (size.height.0 as u64) * 4;
        let zeros = self.device.new_buffer(bytes, metal::MTLResourceOptions::StorageModePrivate);
        self.blit(|blit| {
            blit.fill_buffer(&zeros, metal::NSRange::new(0, bytes), 0);
            blit.copy_from_buffer_to_texture(
                &zeros,
                0,
                size.width.0 as u64 * 4,
                0,
                metal::MTLSize::new(size.width.0 as u64, size.height.0 as u64, 1),
                &metal_texture,
                0,
                0,
                metal::MTLOrigin { x: 0, y: 0, z: 0 },
                metal::MTLBlitOption::empty(),
            );
        });
        self.insert_external(id, metal_texture, size, sampling);
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
            .downcast_ref::<metal::Texture>()
            .context("a Metal external texture shares a metal::Texture")?;
        anyhow::ensure!(
            source.device().registry_id() == self.device.registry_id(),
            "an external texture shares a texture on its own Metal device (`preferred_device`)"
        );
        anyhow::ensure!(
            source.pixel_format() == metal::MTLPixelFormat::BGRA8Unorm,
            "an external texture shares a BGRA8Unorm texture, not {:?}",
            source.pixel_format()
        );
        anyhow::ensure!(
            source.usage().contains(metal::MTLTextureUsage::ShaderRead),
            "an external texture shares a texture with ShaderRead usage"
        );
        if let Some(texture_id) = self.externals.get(&id).copied()
            && let Some(texture) = self
                .polychrome_textures
                .textures
                .get_mut(texture_id.index as usize)
                .and_then(|t| t.as_mut())
            && texture.metal_texture.as_ptr() == source.as_ptr()
        {
            // The same texture: its pixels are already the producer's; only rebind the sampling.
            texture.external = Some(sampling);
            return Ok(());
        }
        let size = Size {
            width: DevicePixels(source.width() as i32),
            height: DevicePixels(source.height() as i32),
        };
        self.remove_external(id);
        self.insert_external(id, source.clone(), size, sampling);
        Ok(())
    }

    /// Records `metal_texture` as external texture `id`: one whole polychrome texture.
    fn insert_external(
        &mut self,
        id: ExternalTextureId,
        metal_texture: metal::Texture,
        size: Size<DevicePixels>,
        sampling: TextureSampling,
    ) {
        let kind = AtlasTextureKind::Polychrome;
        let index = self.polychrome_textures.free_list.pop();
        let texture_id = AtlasTextureId {
            index: index.unwrap_or(self.polychrome_textures.textures.len()) as u32,
            kind,
        };
        let entry = MetalAtlasTexture {
            id: texture_id,
            allocator: etagere::BucketedAtlasAllocator::new(size_to_etagere(size)),
            metal_texture: AssertSend(metal_texture),
            live_atlas_keys: 0,
            external: Some(sampling),
        };
        match index {
            Some(ix) => self.polychrome_textures.textures[ix] = Some(entry),
            None => self.polychrome_textures.textures.push(Some(entry)),
        }
        self.externals.insert(id, texture_id);
    }

    fn remove_external(&mut self, id: ExternalTextureId) {
        let Some(texture_id) = self.externals.remove(&id) else {
            return;
        };
        if let Some(slot) = self
            .polychrome_textures
            .textures
            .get_mut(texture_id.index as usize)
            && slot.take().is_some()
        {
            self.polychrome_textures
                .free_list
                .push(texture_id.index as usize);
        }
    }

    fn copy_external(&mut self, id: ExternalTextureId, source: &dyn std::any::Any) -> Result<()> {
        let texture_id = *self
            .externals
            .get(&id)
            .context("no such external texture")?;
        let target = self
            .texture(texture_id)
            .context("the external texture was lost")?
            .metal_texture
            .clone();
        let source = source
            .downcast_ref::<metal::Texture>()
            .context("a Metal external texture copies from a metal::Texture")?;
        anyhow::ensure!(
            source.device().registry_id() == self.device.registry_id(),
            "an external texture copies from a texture on its own Metal device (`preferred_device`)"
        );
        anyhow::ensure!(
            source.pixel_format() == target.pixel_format(),
            "an external texture copies from a {:?} texture, not {:?}",
            target.pixel_format(),
            source.pixel_format()
        );
        let width = source.width().min(target.width());
        let height = source.height().min(target.height());
        self.blit(|blit| {
            blit.copy_from_texture(
                source,
                0,
                0,
                metal::MTLOrigin { x: 0, y: 0, z: 0 },
                metal::MTLSize::new(width, height, 1),
                &target,
                0,
                0,
                metal::MTLOrigin { x: 0, y: 0, z: 0 },
            );
        });
        Ok(())
    }

    /// Encodes blits on the renderers' queue ([`crate::metal_renderer::shared_command_queue`])
    /// and commits them: frames committed after them read the result, with no wait.
    fn blit(&mut self, encode: impl FnOnce(&metal::BlitCommandEncoderRef)) {
        let queue = crate::metal_renderer::shared_command_queue(&self.device);
        let command_buffer = queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        encode(blit);
        blit.end_encoding();
        command_buffer.commit();
    }

    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        {
            let textures = match texture_kind {
                AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
                AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
                AtlasTextureKind::Subpixel => unreachable!(),
            };

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .filter(|texture| texture.external.is_none())
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
    ) -> &mut MetalAtlasTexture {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };
        // Max texture size on all modern Apple GPUs. Anything bigger than that crashes in validateWithDevice.
        const MAX_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(16384),
            height: DevicePixels(16384),
        };
        let size = min_size.min(&MAX_ATLAS_SIZE).max(&DEFAULT_ATLAS_SIZE);
        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.into());
        texture_descriptor.set_height(size.height.into());
        let pixel_format;
        let usage;
        match kind {
            AtlasTextureKind::Monochrome => {
                pixel_format = metal::MTLPixelFormat::A8Unorm;
                usage = metal::MTLTextureUsage::ShaderRead;
            }
            AtlasTextureKind::Polychrome => {
                pixel_format = metal::MTLPixelFormat::BGRA8Unorm;
                usage = metal::MTLTextureUsage::ShaderRead;
            }
            AtlasTextureKind::Subpixel => unreachable!(),
        }
        texture_descriptor.set_pixel_format(pixel_format);
        texture_descriptor.set_usage(usage);
        // Shared memory mode can be used only on Apple GPU families
        // https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodeshared
        texture_descriptor.set_storage_mode(if self.supports_shared_storage {
            metal::MTLStorageMode::Shared
        } else {
            metal::MTLStorageMode::Managed
        });
        let metal_texture = self.device.new_texture(&texture_descriptor);

        let texture_list = match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        };

        let index = texture_list.free_list.pop();

        let atlas_texture = MetalAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            allocator: etagere::BucketedAtlasAllocator::new(size_to_etagere(size)),
            metal_texture: AssertSend(metal_texture),
            live_atlas_keys: 0,
            external: None,
        };

        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list.textures.get_mut(ix)
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list.textures.last_mut()
        }
        .unwrap()
        .as_mut()
        .unwrap()
    }

    fn texture(&self, id: AtlasTextureId) -> Option<&MetalAtlasTexture> {
        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        };
        textures.textures.get(id.index as usize)?.as_ref()
    }
}

struct MetalAtlasTexture {
    id: AtlasTextureId,
    allocator: BucketedAtlasAllocator,
    metal_texture: AssertSend<metal::Texture>,
    live_atlas_keys: u32,
    /// An external texture (and its sampling): one whole tile, never shared.
    external: Option<TextureSampling>,
}

impl MetalAtlasTexture {
    fn whole_tile(&self) -> AtlasTile {
        AtlasTile {
            texture_id: self.id,
            tile_id: gpui::TileId(0),
            bounds: Bounds {
                origin: Point::default(),
                size: Size {
                    width: DevicePixels(self.metal_texture.width() as i32),
                    height: DevicePixels(self.metal_texture.height() as i32),
                },
            },
            padding: 0,
        }
    }

    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            bounds: Bounds {
                origin: point_from_etagere(allocation.rectangle.min),
                size,
            },
            padding: 0,
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn upload(&self, bounds: Bounds<DevicePixels>, bytes: &[u8]) {
        let region = metal::MTLRegion::new_2d(
            bounds.origin.x.into(),
            bounds.origin.y.into(),
            bounds.size.width.into(),
            bounds.size.height.into(),
        );
        self.metal_texture.replace_region(
            region,
            0,
            bytes.as_ptr() as *const _,
            bounds.size.width.to_bytes(self.bytes_per_pixel()) as u64,
        );
    }

    fn bytes_per_pixel(&self) -> u8 {
        use metal::MTLPixelFormat::*;
        match self.metal_texture.pixel_format() {
            A8Unorm | R8Unorm => 1,
            RGBA8Unorm | BGRA8Unorm => 4,
            _ => unimplemented!(),
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&mut self) -> bool {
        self.live_atlas_keys == 0
    }
}

fn size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    etagere::Size::new(size.width.into(), size.height.into())
}

fn point_from_etagere(value: etagere::Point) -> Point<DevicePixels> {
    Point {
        x: DevicePixels::from(value.x),
        y: DevicePixels::from(value.y),
    }
}

#[derive(Deref, DerefMut)]
struct AssertSend<T>(T);

unsafe impl<T> Send for AssertSend<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::PlatformAtlas;
    use std::borrow::Cow;

    fn create_atlas() -> Option<MetalAtlas> {
        let device = metal::Device::system_default()?;
        Some(MetalAtlas::new(device, true))
    }

    fn make_image_key(image_id: usize, frame_index: usize) -> AtlasKey {
        AtlasKey::Image(gpui::RenderImageParams {
            image_id: gpui::ImageId(image_id),
            frame_index,
        })
    }

    fn insert_tile(atlas: &MetalAtlas, key: AtlasKey, size: Size<DevicePixels>) -> AtlasTile {
        atlas
            .get_or_insert_with(key, &mut || {
                let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
                Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
            })
            .expect("allocation should succeed")
            .expect("callback returns Some")
    }

    #[test]
    fn test_remove_clears_stale_keys_from_tiles_by_key() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };

        let key_a = make_image_key(1, 0);
        let key_b = make_image_key(2, 0);
        let key_c = make_image_key(3, 0);

        let tile_a = insert_tile(&atlas, key_a.clone(), small);
        let tile_b = insert_tile(&atlas, key_b.clone(), small);
        let tile_c = insert_tile(&atlas, key_c.clone(), small);

        assert_eq!(tile_a.texture_id, tile_b.texture_id);
        assert_eq!(tile_b.texture_id, tile_c.texture_id);

        // Remove A: texture still has B and C, so it stays.
        // The key for A must be removed from tiles_by_key.
        atlas.remove(&key_a);

        // Remove B: texture still has C.
        atlas.remove(&key_b);

        // Remove C: texture becomes unreferenced and is deleted.
        atlas.remove(&key_c);

        // Re-inserting A must allocate a fresh tile on a new texture,
        // NOT return a stale tile referencing the deleted texture.
        let tile_a2 = insert_tile(&atlas, key_a, small);

        // The texture must actually exist — this would panic before the fix.
        assert!(atlas.metal_texture(tile_a2.texture_id).is_some());
    }

    #[test]
    fn test_metal_texture_is_none_after_last_tile_removed() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let key = make_image_key(1, 0);
        let tile = insert_tile(
            &atlas,
            key.clone(),
            Size {
                width: DevicePixels(64),
                height: DevicePixels(64),
            },
        );
        assert!(atlas.metal_texture(tile.texture_id).is_some());

        // A scene built before the removal may still carry `tile`; looking its
        // texture up must report the gap instead of panicking.
        atlas.remove(&key);
        assert!(atlas.metal_texture(tile.texture_id).is_none());
    }

    #[test]
    fn test_remove_deallocates_tile_space_for_reuse() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };
        let big = Size {
            width: DevicePixels(700),
            height: DevicePixels(700),
        };

        let keeper_key = make_image_key(1, 0);
        let big_key_a = make_image_key(2, 0);
        let big_key_b = make_image_key(3, 0);

        let keeper_tile = insert_tile(&atlas, keeper_key, small);
        let tile_a = insert_tile(&atlas, big_key_a.clone(), big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);
        let tile_b = insert_tile(&atlas, big_key_b, big);
        assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
    }

    #[test]
    fn test_remove_nonexistent_key_is_noop() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let key = make_image_key(999, 0);
        atlas.remove(&key);
    }

    fn device_size(width: i32, height: i32) -> Size<DevicePixels> {
        Size {
            width: DevicePixels(width),
            height: DevicePixels(height),
        }
    }

    /// The pixels of external texture `id`, read back through a shared buffer.
    fn external_pixels(atlas: &MetalAtlas, id: ExternalTextureId) -> Vec<u8> {
        let tile = atlas.external_texture_tile(id).unwrap();
        let texture = atlas.metal_texture(tile.texture_id).unwrap();
        let (w, h) = (texture.width(), texture.height());
        let device = metal::Device::system_default().unwrap();
        let buffer = device.new_buffer(w * h * 4, metal::MTLResourceOptions::StorageModeShared);
        let queue = crate::metal_renderer::shared_command_queue(&device);
        let cb = queue.new_command_buffer();
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
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const u8, (w * h * 4) as usize) }.to_vec()
    }

    #[test]
    fn test_external_texture_is_one_whole_tile_that_images_never_share() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let id = ExternalTextureId(7);
        assert!(atlas.external_texture_tile(id).is_none());
        assert!(atlas.set_external_texture(id, device_size(40, 30), TextureSampling::Linear).unwrap());
        let tile = atlas.external_texture_tile(id).unwrap();
        assert_eq!(tile.bounds.size, device_size(40, 30));
        assert_eq!(tile.bounds.origin, Point::default());
        assert_eq!(atlas.external_texture_bytes(), 40 * 30 * 4);
        assert!(external_pixels(&atlas, id).iter().all(|b| *b == 0), "a new texture starts transparent");
        let image = insert_tile(&atlas, make_image_key(1, 0), device_size(8, 8));
        assert_ne!(image.texture_id, tile.texture_id, "images never share an external texture");
        atlas.remove_external_texture(id);
        assert!(atlas.external_texture_tile(id).is_none());
        assert_eq!(atlas.external_texture_bytes(), 0);
    }

    #[test]
    fn test_external_texture_keeps_its_pixels_at_its_size_and_sampling_rebinds() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let device = metal::Device::system_default().unwrap();
        let id = ExternalTextureId(1);
        atlas.set_external_texture(id, device_size(4, 2), TextureSampling::Linear).unwrap();
        // A shared BGRA texture with known bytes, copied in on the GPU.
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(4);
        descriptor.set_height(2);
        descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
        let source = device.new_texture(&descriptor);
        let bytes: Vec<u8> = (0..32).map(|i| i as u8 * 7).collect();
        source.replace_region(metal::MTLRegion::new_2d(0, 0, 4, 2), 0, bytes.as_ptr() as *const _, 16);
        atlas.copy_to_external_texture(id, &source).unwrap();
        assert_eq!(external_pixels(&atlas, id), bytes);
        let tile = atlas.external_texture_tile(id).unwrap();
        assert!(!atlas.set_external_texture(id, device_size(4, 2), TextureSampling::Nearest).unwrap());
        assert_eq!(atlas.sampling(tile.texture_id), TextureSampling::Nearest);
        assert_eq!(external_pixels(&atlas, id), bytes, "same size keeps the pixels");
        assert!(atlas.set_external_texture(id, device_size(5, 2), TextureSampling::Nearest).unwrap());
        assert!(external_pixels(&atlas, id).iter().all(|b| *b == 0), "a resize starts over");
        assert!(atlas.copy_to_external_texture(id, &5u32).is_err(), "only Metal textures copy in");
    }

    /// A producer's texture: BGRA8Unorm, private, drawn into and sampled.
    fn producer_texture(
        device: &metal::DeviceRef,
        w: u64,
        h: u64,
        format: metal::MTLPixelFormat,
    ) -> metal::Texture {
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(w);
        descriptor.set_height(h);
        descriptor.set_pixel_format(format);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        descriptor
            .set_usage(metal::MTLTextureUsage::ShaderRead | metal::MTLTextureUsage::RenderTarget);
        device.new_texture(&descriptor)
    }

    /// Clears `texture` to `bgra` in a render pass on the shared queue, without waiting.
    fn clear_on_shared_queue(texture: &metal::TextureRef, bgra: [u8; 4]) {
        let queue = crate::metal_renderer::shared_command_queue(&texture.device());
        let cb = queue.new_command_buffer();
        let pass = metal::RenderPassDescriptor::new();
        let color = pass.color_attachments().object_at(0).unwrap();
        color.set_texture(Some(texture));
        color.set_load_action(metal::MTLLoadAction::Clear);
        color.set_store_action(metal::MTLStoreAction::Store);
        let c = |v: u8| v as f64 / 255.0;
        color.set_clear_color(metal::MTLClearColor::new(
            c(bgra[2]),
            c(bgra[1]),
            c(bgra[0]),
            c(bgra[3]),
        ));
        cb.new_render_command_encoder(pass).end_encoding();
        cb.commit();
    }

    #[test]
    fn test_shared_external_texture_is_the_producers_texture_with_no_copy() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let device = metal::Device::system_default().unwrap();
        let id = ExternalTextureId(3);
        let source = producer_texture(&device, 6, 4, metal::MTLPixelFormat::BGRA8Unorm);
        clear_on_shared_queue(&source, [10, 20, 30, 255]);
        atlas
            .share_external_texture(id, &source, TextureSampling::Linear)
            .unwrap();

        let tile = atlas.external_texture_tile(id).unwrap();
        assert_eq!(tile.bounds.size, device_size(6, 4));
        let shown = atlas.metal_texture(tile.texture_id).unwrap();
        assert_eq!(
            shown.as_ptr(),
            source.as_ptr(),
            "the atlas shows the producer's texture itself"
        );
        // Read on the shared queue after the producer's clear, with no wait between them.
        assert!(
            external_pixels(&atlas, id)
                .chunks(4)
                .all(|p| p == [10, 20, 30, 255])
        );

        // New pixels in the same texture: nothing to copy, the next read sees them.
        clear_on_shared_queue(&source, [1, 2, 3, 255]);
        assert!(
            external_pixels(&atlas, id)
                .chunks(4)
                .all(|p| p == [1, 2, 3, 255])
        );

        // Sharing it again only rebinds the sampling.
        atlas
            .share_external_texture(id, &source, TextureSampling::Nearest)
            .unwrap();
        let again = atlas.external_texture_tile(id).unwrap();
        assert_eq!(again.texture_id, tile.texture_id);
        assert_eq!(atlas.sampling(again.texture_id), TextureSampling::Nearest);

        // Another texture replaces it.
        let bigger = producer_texture(&device, 8, 8, metal::MTLPixelFormat::BGRA8Unorm);
        atlas
            .share_external_texture(id, &bigger, TextureSampling::Linear)
            .unwrap();
        let replaced = atlas.external_texture_tile(id).unwrap();
        assert_eq!(replaced.bounds.size, device_size(8, 8));
        assert_eq!(
            atlas.metal_texture(replaced.texture_id).unwrap().as_ptr(),
            bigger.as_ptr()
        );

        atlas.remove_external_texture(id);
        assert!(atlas.external_texture_tile(id).is_none());
    }

    #[test]
    fn test_shared_external_textures_must_be_bgra_metal_textures() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let device = metal::Device::system_default().unwrap();
        let id = ExternalTextureId(4);
        let rgba = producer_texture(&device, 2, 2, metal::MTLPixelFormat::RGBA8Unorm);
        assert!(
            atlas
                .share_external_texture(id, &rgba, TextureSampling::Linear)
                .is_err()
        );
        assert!(
            atlas
                .share_external_texture(id, &5u32, TextureSampling::Linear)
                .is_err()
        );
        assert!(atlas.external_texture_tile(id).is_none());
    }
}
