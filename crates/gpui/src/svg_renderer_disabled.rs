//! The SVG renderer without the `svg` feature: no resvg, usvg or font database. `svg()`
//! elements draw nothing and SVG images fail to decode with an error that names the feature.

use crate::{AssetSource, DevicePixels, RenderImage, Result, SharedString, Size};
use std::sync::Arc;

/// When rendering SVGs, we render them at twice the size to get a higher-quality result.
pub const SMOOTH_SVG_SCALE_FACTOR: f32 = 2.;

#[derive(Clone, PartialEq, Hash, Eq)]
#[expect(missing_docs)]
pub struct RenderSvgParams {
    pub path: SharedString,
    pub size: Size<DevicePixels>,
}

#[derive(Clone)]
/// The SVG renderer, built without the `svg` feature: it renders nothing.
pub struct SvgRenderer;

impl SvgRenderer {
    /// Creates the renderer. It keeps nothing: there is nothing to render with.
    pub fn new(_asset_source: Arc<dyn AssetSource>) -> Self {
        Self
    }

    /// Fails: SVG rasterization needs gpui's `svg` feature.
    pub fn render_single_frame(&self, _bytes: &[u8], _scale_factor: f32) -> Result<Arc<RenderImage>> {
        anyhow::bail!("SVG images need gpui's `svg` feature")
    }

    /// Draws nothing: `svg()` needs gpui's `svg` feature.
    pub(crate) fn render_alpha_mask(&self, _params: &RenderSvgParams, _bytes: Option<&[u8]>) -> Result<Option<(Size<DevicePixels>, Vec<u8>)>> {
        Ok(None)
    }
}
