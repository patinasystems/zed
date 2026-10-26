use anyhow::anyhow;
use collections::{HashMap, HashSet};
use core_foundation::{
    array::{CFArray, CFArrayRef},
    attributed_string::CFMutableAttributedString,
    base::{CFRange, TCFType},
    number::CFNumber,
    string::CFString,
};
use core_graphics::{
    base::{CGFloat, CGGlyph, kCGImageAlphaPremultipliedLast},
    color_space::CGColorSpace,
    context::{CGContext, CGTextDrawingMode},
    geometry::CGPoint,
};
use core_text::{
    font::CTFont,
    font_collection::CTFontCollectionRef,
    font_descriptor::{
        CTFontDescriptor, kCTFontOrientationDefault, kCTFontSlantTrait, kCTFontSymbolicTrait,
        kCTFontWeightTrait, kCTFontWidthTrait,
    },
    line::CTLine,
    string_attributes::kCTFontAttributeName,
};
use font_kit::{
    font::Font as FontKitFont,
    handle::Handle,
    metrics::Metrics,
    properties::{Style as FontkitStyle, Weight as FontkitWeight},
    source::SystemSource,
    sources::mem::MemSource,
};
use gpui::{
    Bounds, DevicePixels, Font, FontFallbacks, FontFeatures, FontId, FontMetrics, FontRun,
    FontStyle, FontWeight, GlyphId, LineLayout, Pixels, PlatformTextSystem, RenderGlyphParams,
    Result, SUBPIXEL_VARIANTS_X, ShapedGlyph, ShapedRun, SharedString, Size, TextRenderingMode,
    point, px, size, swap_rgba_pa_to_bgra,
};
use parking_lot::{RwLock, RwLockUpgradableReadGuard};
use pathfinder_geometry::{
    rect::{RectF, RectI},
    transform2d::Transform2F,
    vector::Vector2F,
};
use smallvec::SmallVec;
use std::{borrow::Cow, char, convert::TryFrom, sync::Arc};

#[cfg(target_os = "macos")]
use {
    core_foundation::base::CFType,
    gpui::{Hsla, Rgba},
    std::sync::OnceLock,
};

use crate::open_type::apply_features_and_fallbacks;

#[allow(non_upper_case_globals)]
const kCGImageAlphaOnly: u32 = 7;

/// Apple text system using CoreText for font shaping.
pub struct AppleTextSystem(RwLock<AppleTextSystemState>);

#[derive(Clone, PartialEq, Eq, Hash)]
struct FontKey {
    font_family: SharedString,
    font_features: FontFeatures,
    font_fallbacks: Option<FontFallbacks>,
}

struct AppleTextSystemState {
    memory_source: MemSource,
    system_source: SystemSource,
    fonts: Vec<FontKitFont>,
    font_selections: HashMap<Font, FontId>,
    font_ids_by_postscript_name: HashMap<String, FontId>,
    font_ids_by_font_key: HashMap<FontKey, SmallVec<[FontId; 4]>>,
    postscript_names_by_font_id: HashMap<FontId, String>,
    /// The PostScript names in the system UI font's files and its cascade's, with their file,
    /// read on the first lookup Core Text cannot answer by name.
    system_ui_font_files: Option<HashMap<String, std::path::PathBuf>>,
    /// The same for every font file Core Text makes available, read on the first lookup the
    /// UI fonts cannot answer.
    available_font_files: Option<HashMap<String, std::path::PathBuf>>,
}

impl AppleTextSystem {
    /// Create a new AppleTextSystem.
    pub fn new() -> Self {
        Self(RwLock::new(AppleTextSystemState {
            memory_source: MemSource::empty(),
            system_source: SystemSource::new(),
            fonts: Vec::new(),
            font_selections: HashMap::default(),
            font_ids_by_postscript_name: HashMap::default(),
            font_ids_by_font_key: HashMap::default(),
            postscript_names_by_font_id: HashMap::default(),
            system_ui_font_files: None,
            available_font_files: None,
        }))
    }
}

impl Default for AppleTextSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl PlatformTextSystem for AppleTextSystem {
    fn add_fonts(&self, fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
        self.0.write().add_fonts(fonts)
    }

    fn all_font_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        let collection = core_text::font_collection::create_for_all_families();
        // NOTE: We intentionally avoid using `collection.get_descriptors()` here because
        // it has a memory leak bug in core-text v21.0.0. The upstream code uses
        // `wrap_under_get_rule` but `CTFontCollectionCreateMatchingFontDescriptors`
        // follows the Create Rule (caller owns the result), so it should use
        // `wrap_under_create_rule`. We call the function directly with correct memory management.
        unsafe extern "C" {
            fn CTFontCollectionCreateMatchingFontDescriptors(
                collection: CTFontCollectionRef,
            ) -> CFArrayRef;
        }
        let descriptors: Option<CFArray<CTFontDescriptor>> = unsafe {
            let array_ref =
                CTFontCollectionCreateMatchingFontDescriptors(collection.as_concrete_TypeRef());
            if array_ref.is_null() {
                None
            } else {
                Some(CFArray::wrap_under_create_rule(array_ref))
            }
        };
        let Some(descriptors) = descriptors else {
            return names;
        };
        for descriptor in descriptors.into_iter() {
            names.extend(lenient_font_attributes::family_name(&descriptor));
        }
        if let Ok(fonts_in_memory) = self.0.read().memory_source.all_families() {
            names.extend(fonts_in_memory);
        }
        names
    }

    fn font_id(&self, font: &Font) -> Result<FontId> {
        let lock = self.0.upgradable_read();
        if let Some(font_id) = lock.font_selections.get(font) {
            Ok(*font_id)
        } else {
            let mut lock = RwLockUpgradableReadGuard::upgrade(lock);
            let font_key = FontKey {
                font_family: font.family.clone(),
                font_features: font.features.clone(),
                font_fallbacks: font.fallbacks.clone(),
            };
            let candidates = if let Some(font_ids) = lock.font_ids_by_font_key.get(&font_key) {
                font_ids.as_slice()
            } else {
                let font_ids =
                    lock.load_family(&font.family, &font.features, font.fallbacks.as_ref())?;
                lock.font_ids_by_font_key.insert(font_key.clone(), font_ids);
                lock.font_ids_by_font_key[&font_key].as_ref()
            };

            let candidate_properties = candidates
                .iter()
                .map(|font_id| {
                    let font = &lock.fonts[font_id.0];
                    let mut properties = font.properties();
                    // font-kit maps Core Text's weight trait onto CSS weights with a curve
                    // that puts Medium at 530, so CSS matching sends 500 to Regular.
                    if let Some(weight) = core_text_css_weight(font) {
                        properties.weight = FontkitWeight(weight);
                    }
                    properties
                })
                .collect::<SmallVec<[_; 4]>>();

            let ix = font_kit::matching::find_best_match(
                &candidate_properties,
                &font_kit::properties::Properties {
                    style: fontkit_style(font.style),
                    weight: fontkit_weight(font.weight),
                    stretch: Default::default(),
                },
            )?;

            let font_id = candidates[ix];
            lock.font_selections.insert(font.clone(), font_id);
            Ok(font_id)
        }
    }

    fn font_id_for_postscript_name(&self, postscript_name: &str) -> Option<FontId> {
        if let Some(font_id) = self
            .0
            .read()
            .font_ids_by_postscript_name
            .get(postscript_name)
        {
            return Some(*font_id);
        }
        self.0.write().font_id_for_postscript_name(postscript_name)
    }

    fn font_metrics(&self, font_id: FontId) -> FontMetrics {
        font_kit_metrics_to_metrics(self.0.read().fonts[font_id.0].metrics())
    }

    fn typographic_bounds(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Bounds<f32>> {
        Ok(bounds_from_rect(
            self.0.read().fonts[font_id.0].typographic_bounds(glyph_id.0)?,
        ))
    }

    fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Size<f32>> {
        self.0.read().advance(font_id, glyph_id)
    }

    fn glyph_for_char(&self, font_id: FontId, ch: char) -> Option<GlyphId> {
        self.0.read().glyph_for_char(font_id, ch)
    }

    fn glyph_raster_bounds(&self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
        self.0.read().raster_bounds(params)
    }

    fn rasterize_glyph(
        &self,
        glyph_id: &RenderGlyphParams,
        raster_bounds: Bounds<DevicePixels>,
    ) -> Result<(Size<DevicePixels>, Vec<u8>)> {
        self.0.read().rasterize_glyph(glyph_id, raster_bounds)
    }

    fn layout_line(&self, text: &str, font_size: Pixels, font_runs: &[FontRun]) -> LineLayout {
        self.0.write().layout_line(text, font_size, font_runs)
    }

    fn recommended_rendering_mode(
        &self,
        _font_id: FontId,
        _font_size: Pixels,
    ) -> TextRenderingMode {
        TextRenderingMode::Grayscale
    }

    #[cfg(target_os = "macos")]
    fn glyph_dilation_for_color(&self, color: Hsla) -> u8 {
        // When font smoothing is enabled, CoreGraphics thickens glyph strokes by an amount that
        // depends on the foreground color's luminance. We replicate the logic used by CoreGraphics
        // to select between the different levels of dilation.
        if !font_smoothing_allowed_by_user() {
            return 0;
        }
        let rgba: Rgba = color.into();
        let luminance = 0.2126 * rgba.r + 0.7152 * rgba.g + 0.0722 * rgba.b;
        let level = ((4.0 * luminance) + 0.5).floor() as i32;
        level.clamp(0, 4) as u8
    }
}

unsafe extern "C" {
    fn CTFontCopyDefaultCascadeListForLanguages(
        font: core_text::font::CTFontRef,
        language_pref_list: core_foundation::array::CFArrayRef,
    ) -> CFArrayRef;
}

/// The faces in a font file, as Core Text reads them (one per named instance of a variable font).
fn file_descriptors(path: &std::path::Path) -> Vec<CTFontDescriptor> {
    let Some(url) = core_foundation::url::CFURL::from_path(path, false) else {
        return Vec::new();
    };
    let array = unsafe {
        core_text::font_manager::CTFontManagerCreateFontDescriptorsFromURL(
            url.as_concrete_TypeRef(),
        )
    };
    if array.is_null() {
        return Vec::new();
    }
    let array: CFArray<CTFontDescriptor> = unsafe { CFArray::wrap_under_create_rule(array) };
    array.iter().map(|descriptor| descriptor.clone()).collect()
}

/// Every PostScript name in the files of the system UI font and of the fonts its cascade lists
/// for the languages with their own fallbacks (CJK), with the file it is in.
fn system_ui_font_files() -> HashMap<String, std::path::PathBuf> {
    let ui = core_text::font::new_ui_font_for_language(
        core_text::font::kCTFontSystemFontType,
        12.,
        None,
    );
    let mut paths = Vec::new();
    paths.extend(ui.copy_descriptor().font_path());
    for language in ["en", "ja", "zh-Hans", "zh-Hant", "zh-HK", "ko"] {
        let languages = CFArray::from_CFTypes(&[CFString::new(language)]);
        let cascade = unsafe {
            CTFontCopyDefaultCascadeListForLanguages(
                ui.as_concrete_TypeRef(),
                languages.as_concrete_TypeRef(),
            )
        };
        if cascade.is_null() {
            continue;
        }
        let cascade: CFArray<CTFontDescriptor> =
            unsafe { CFArray::wrap_under_create_rule(cascade) };
        for descriptor in cascade.iter() {
            // Cascade entries can be aliases (`.AppleJapaneseFont`): the font they open has the file.
            let font = core_text::font::new_from_descriptor(&descriptor, 12.);
            paths.extend(font.copy_descriptor().font_path());
        }
    }
    names_in_files(paths)
}

/// Every PostScript name in the font files Core Text makes available and in the system font
/// directory (which also holds the hidden system fonts, such as SF Mono), with its file.
fn available_font_files() -> HashMap<String, std::path::PathBuf> {
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir("/System/Library/Fonts")
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect();
    let urls = unsafe { core_text::font_manager::CTFontManagerCopyAvailableFontURLs() };
    if !urls.is_null() {
        let urls: CFArray<core_foundation::url::CFURL> =
            unsafe { CFArray::wrap_under_create_rule(urls) };
        paths.extend(urls.iter().filter_map(|url| url.to_path()));
    }
    names_in_files(paths)
}

fn names_in_files(paths: Vec<std::path::PathBuf>) -> HashMap<String, std::path::PathBuf> {
    let mut names = HashMap::default();
    let mut seen = HashSet::default();
    for path in paths {
        if !seen.insert(path.clone()) {
            continue;
        }
        for descriptor in file_descriptors(&path) {
            names
                .entry(descriptor.font_name())
                .or_insert_with(|| path.clone());
        }
    }
    names
}

/// The face named `postscript_name` in the font file at `path`.
fn font_from_file(path: &std::path::Path, postscript_name: &str) -> Option<CTFont> {
    file_descriptors(path)
        .into_iter()
        .find(|descriptor| descriptor.font_name() == postscript_name)
        .map(|descriptor| core_text::font::new_from_descriptor(&descriptor, 12.))
}

#[cfg(target_os = "macos")]
fn font_smoothing_allowed_by_user() -> bool {
    static ALLOWED: OnceLock<bool> = OnceLock::new();
    *ALLOWED.get_or_init(|| {
        use core_foundation_sys::preferences::{
            CFPreferencesCopyAppValue, kCFPreferencesCurrentApplication,
        };

        let key = CFString::new("AppleFontSmoothing");
        let value_ref = unsafe {
            CFPreferencesCopyAppValue(key.as_concrete_TypeRef(), kCFPreferencesCurrentApplication)
        };
        if value_ref.is_null() {
            return true;
        }
        let value = unsafe { CFType::wrap_under_create_rule(value_ref) };
        let Some(number) = value.downcast_into::<CFNumber>() else {
            return true;
        };
        // Only an explicit value of `0` means that font smoothing is disabled.
        number.to_i64() != Some(0)
    })
}

impl AppleTextSystemState {
    fn add_fonts(&mut self, fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
        let fonts = fonts
            .into_iter()
            .map(|bytes| match bytes {
                Cow::Borrowed(embedded_font) => {
                    let data_provider = unsafe {
                        core_graphics::data_provider::CGDataProvider::from_slice(embedded_font)
                    };
                    let font = core_graphics::font::CGFont::from_data_provider(data_provider)
                        .map_err(|()| anyhow!("Could not load an embedded font."))?;
                    let font = font_kit::loaders::core_text::Font::from_core_graphics_font(font);
                    Ok(Handle::from_native(&font))
                }
                Cow::Owned(bytes) => Ok(Handle::from_memory(Arc::new(bytes), 0)),
            })
            .collect::<Result<Vec<_>>>()?;
        self.memory_source.add_fonts(fonts.into_iter())?;
        Ok(())
    }

    fn load_family(
        &mut self,
        name: &str,
        features: &FontFeatures,
        fallbacks: Option<&FontFallbacks>,
    ) -> Result<SmallVec<[FontId; 4]>> {
        let name = gpui::font_name_with_fallbacks(name, ".AppleSystemUIFont");

        let mut font_ids = SmallVec::new();
        let mut postscript_names_seen = HashSet::default();
        let family = self
            .memory_source
            .select_family_by_name(name)
            .or_else(|_| self.system_source.select_family_by_name(name))?;
        for font in family.fonts() {
            let mut font = font.load()?;

            apply_features_and_fallbacks(&mut font, features, fallbacks)?;
            // This block contains a precautionary fix to guard against loading fonts
            // that might cause panics due to `.unwrap()`s up the chain.
            {
                // We use the 'm' character for text measurements in various spots
                // (e.g., the editor). However, at time of writing some of those usages
                // will panic if the font has no 'm' glyph.
                //
                // Therefore, we check up front that the font has the necessary glyph.
                let has_m_glyph = font.glyph_for_char('m').is_some();

                // HACK: The 'Segoe Fluent Icons' font does not have an 'm' glyph,
                // but we need to be able to load it for rendering Windows icons in
                // the Storybook (on macOS).
                let is_segoe_fluent_icons = font.full_name() == "Segoe Fluent Icons";

                if !has_m_glyph && !is_segoe_fluent_icons {
                    // I spent far too long trying to track down why a font missing the 'm'
                    // character wasn't loading. This log statement will hopefully save
                    // someone else from suffering the same fate.
                    log::warn!(
                        "font '{}' has no 'm' character and was not loaded",
                        font.full_name()
                    );
                    continue;
                }
            }

            // We've seen a number of panics in production caused by calling font.properties()
            // which unwraps a downcast to CFNumber. This is an attempt to avoid the panic,
            // and to try and identify the incalcitrant font.
            let traits = font.native_font().all_traits();
            if unsafe {
                !(traits
                    .get(kCTFontSymbolicTrait)
                    .downcast::<CFNumber>()
                    .is_some()
                    && traits
                        .get(kCTFontWidthTrait)
                        .downcast::<CFNumber>()
                        .is_some()
                    && traits
                        .get(kCTFontWeightTrait)
                        .downcast::<CFNumber>()
                        .is_some()
                    && traits
                        .get(kCTFontSlantTrait)
                        .downcast::<CFNumber>()
                        .is_some())
            } {
                log::error!(
                    "Failed to read traits for font {:?} (PostScript name {:?})",
                    font.full_name(),
                    font.postscript_name(),
                );
                continue;
            }

            let Some(postscript_name) = font.postscript_name() else {
                log::warn!(
                    "font {:?} in family {:?} has no PostScript name; skipping",
                    font.full_name(),
                    name,
                );
                continue;
            };
            // Dedup is scoped to this single `load_family` call (issue #55472).
            // The same family can be reloaded later under a different `FontKey`
            // (different features/fallbacks); a global check against
            // `font_ids_by_postscript_name` would skip every already-registered
            // font and leave the second call's `font_ids` empty.
            if !postscript_names_seen.insert(postscript_name.clone()) {
                log::warn!(
                    "skipping duplicate font {:?} with PostScript name {:?} \
                     in family {:?}",
                    font.full_name(),
                    postscript_name,
                    name,
                );
                continue;
            }
            let font_id = FontId(self.fonts.len());
            font_ids.push(font_id);
            self.font_ids_by_postscript_name
                .insert(postscript_name.clone(), font_id);
            self.postscript_names_by_font_id
                .insert(font_id, postscript_name);
            self.fonts.push(font);
        }
        Ok(font_ids)
    }

    fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Size<f32>> {
        Ok(size_from_vector2f(
            self.fonts[font_id.0].advance(glyph_id.0)?,
        ))
    }

    fn glyph_for_char(&self, font_id: FontId, ch: char) -> Option<GlyphId> {
        self.fonts[font_id.0].glyph_for_char(ch).map(GlyphId)
    }

    fn font_id_for_postscript_name(&mut self, postscript_name: &str) -> Option<FontId> {
        if let Some(font_id) = self.font_ids_by_postscript_name.get(postscript_name) {
            return Some(*font_id);
        }
        // Fonts added with `add_fonts` are not registered with Core Text, so look there first.
        if let Ok(handle) = self
            .memory_source
            .select_by_postscript_name(postscript_name)
        {
            let font = handle.load().ok()?;
            let font_id = FontId(self.fonts.len());
            self.font_ids_by_postscript_name
                .insert(postscript_name.to_string(), font_id);
            self.postscript_names_by_font_id
                .insert(font_id, postscript_name.to_string());
            self.fonts.push(font);
            return Some(font_id);
        }
        // Core Text substitutes another font for a name it does not know: check the name.
        if let Ok(font) = core_text::font::new_from_name(postscript_name, 12.)
            && font.postscript_name() == postscript_name
        {
            return Some(self.id_for_native_font(font));
        }
        // Core Text opens the system UI fonts (their names start with a period) only through
        // the UI font APIs or their files, so look in the files of the UI font and its cascade.
        let mut path = self
            .system_ui_font_files
            .get_or_insert_with(system_ui_font_files)
            .get(postscript_name)
            .cloned();
        if path.is_none() {
            // Other hidden system fonts (`.SFNSMono-Regular`, the system monospaced font):
            // every font file Core Text makes available, read once.
            path = self
                .available_font_files
                .get_or_insert_with(available_font_files)
                .get(postscript_name)
                .cloned();
        }
        let font = font_from_file(&path?, postscript_name)?;
        Some(self.id_for_native_font(font))
    }

    fn id_for_native_font(&mut self, requested_font: CTFont) -> FontId {
        let postscript_name = requested_font.postscript_name();
        if let Some(font_id) = self.font_ids_by_postscript_name.get(&postscript_name) {
            *font_id
        } else {
            let font_id = FontId(self.fonts.len());
            self.font_ids_by_postscript_name
                .insert(postscript_name.clone(), font_id);
            self.postscript_names_by_font_id
                .insert(font_id, postscript_name);
            self.fonts
                .push(font_kit::font::Font::from_core_graphics_font(
                    requested_font.copy_to_CGFont(),
                ));
            font_id
        }
    }

    fn is_emoji(&self, font_id: FontId) -> bool {
        self.postscript_names_by_font_id
            .get(&font_id)
            .is_some_and(|postscript_name| {
                postscript_name == "AppleColorEmoji" || postscript_name == ".AppleColorEmojiUI"
            })
    }

    fn raster_bounds(&self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
        // Measure the glyph with a font at the size `rasterize_glyph` draws it at. Fonts with
        // optical sizes, such as the system font, change their outlines with the point size, so
        // bounds measured at another size and scaled can be too small and clip the glyph.
        let font = self.fonts[params.font_id.0]
            .native_font()
            .clone_with_font_size(f32::from(params.font_size) as CGFloat);
        let rect = font.get_bounding_rects_for_glyphs(
            kCTFontOrientationDefault,
            &[params.glyph_id.0 as CGGlyph],
        );
        // Core Text's y axis points up; raster bounds have their origin at the top left.
        let rect = RectF::new(
            Vector2F::new(
                rect.origin.x as f32,
                -(rect.origin.y + rect.size.height) as f32,
            ),
            Vector2F::new(rect.size.width as f32, rect.size.height as f32),
        );
        let scale = Transform2F::from_scale(params.scale_factor);
        let bounds = bounds_from_rect_i((scale * rect).round_out().to_i32());

        // Expand the bounds by 1 pixel on each side to give CG room for anti-aliasing.
        Ok(bounds.dilate(DevicePixels(1)))
    }

    fn rasterize_glyph(
        &self,
        params: &RenderGlyphParams,
        glyph_bounds: Bounds<DevicePixels>,
    ) -> Result<(Size<DevicePixels>, Vec<u8>)> {
        if glyph_bounds.size.width.0 == 0 || glyph_bounds.size.height.0 == 0 {
            anyhow::bail!("glyph bounds are empty");
        } else {
            // Add an extra pixel when the subpixel variant isn't zero to make room for anti-aliasing.
            let mut bitmap_size = glyph_bounds.size;
            if params.subpixel_variant.x > 0 {
                bitmap_size.width += DevicePixels(1);
            }
            if params.subpixel_variant.y > 0 {
                bitmap_size.height += DevicePixels(1);
            }
            let bitmap_size = bitmap_size;

            let mut bytes;
            let cx;
            if params.is_emoji {
                bytes = vec![0; bitmap_size.width.0 as usize * 4 * bitmap_size.height.0 as usize];
                cx = CGContext::create_bitmap_context(
                    Some(bytes.as_mut_ptr() as *mut _),
                    bitmap_size.width.0 as usize,
                    bitmap_size.height.0 as usize,
                    8,
                    bitmap_size.width.0 as usize * 4,
                    &CGColorSpace::create_device_rgb(),
                    kCGImageAlphaPremultipliedLast,
                );
            } else {
                bytes = vec![0; bitmap_size.width.0 as usize * bitmap_size.height.0 as usize];
                cx = CGContext::create_bitmap_context(
                    Some(bytes.as_mut_ptr() as *mut _),
                    bitmap_size.width.0 as usize,
                    bitmap_size.height.0 as usize,
                    8,
                    bitmap_size.width.0 as usize,
                    &CGColorSpace::create_device_gray(),
                    kCGImageAlphaOnly,
                );
            }

            // Move the origin to bottom left and account for scaling, this
            // makes drawing text consistent with the font-kit's raster_bounds.
            cx.translate(
                -glyph_bounds.origin.x.0 as CGFloat,
                (glyph_bounds.origin.y.0 + glyph_bounds.size.height.0) as CGFloat,
            );
            cx.scale(
                params.scale_factor as CGFloat,
                params.scale_factor as CGFloat,
            );

            let subpixel_shift = params
                .subpixel_variant
                .map(|v| v as f32 / SUBPIXEL_VARIANTS_X as f32);
            cx.set_text_drawing_mode(CGTextDrawingMode::CGTextFill);
            cx.set_allows_antialiasing(true);
            cx.set_should_antialias(true);
            cx.set_allows_font_subpixel_positioning(true);
            cx.set_should_subpixel_position_fonts(true);
            cx.set_allows_font_subpixel_quantization(false);
            cx.set_should_subpixel_quantize_fonts(false);

            if cfg!(target_os = "macos") && params.dilation > 0 {
                let luminance = params.dilation as f64 * 0.25;
                cx.set_should_smooth_fonts(true);
                cx.set_gray_fill_color(luminance, 1.0);
            } else {
                cx.set_gray_fill_color(0.0, 1.0);
            }
            self.fonts[params.font_id.0]
                .native_font()
                .clone_with_font_size(f32::from(params.font_size) as CGFloat)
                .draw_glyphs(
                    &[params.glyph_id.0 as CGGlyph],
                    &[CGPoint::new(
                        (subpixel_shift.x / params.scale_factor) as CGFloat,
                        (subpixel_shift.y / params.scale_factor) as CGFloat,
                    )],
                    cx,
                );

            if params.is_emoji {
                // Convert from RGBA with premultiplied alpha to BGRA with straight alpha.
                for pixel in bytes.chunks_exact_mut(4) {
                    swap_rgba_pa_to_bgra(pixel);
                }
            }

            Ok((bitmap_size, bytes))
        }
    }

    fn layout_line(&mut self, text: &str, font_size: Pixels, font_runs: &[FontRun]) -> LineLayout {
        // Construct the attributed string, converting UTF8 ranges to UTF16 ranges.
        let mut string = CFMutableAttributedString::new();
        let mut max_ascent = 0.0f32;
        let mut max_descent = 0.0f32;

        {
            let mut text = text;
            let mut break_ligature = true;
            for run in font_runs {
                let text_run;
                (text_run, text) = text.split_at(run.len);

                let utf16_start = string.char_len(); // insert at end of string
                // note: replace_str may silently ignore codepoints it dislikes (e.g., BOM at start of string)
                string.replace_str(&CFString::new(text_run), CFRange::init(utf16_start, 0));
                let utf16_end = string.char_len();

                let length = utf16_end - utf16_start;
                let cf_range = CFRange::init(utf16_start, length);
                let font = &self.fonts[run.font_id.0];

                // Runs may have their own size; the line's ascent and descent cover them all.
                let run_font_size = run.font_size.unwrap_or(font_size);
                let font_metrics = font.metrics();
                let font_scale = f32::from(run_font_size) / font_metrics.units_per_em as f32;
                max_ascent = max_ascent.max(font_metrics.ascent * font_scale);
                max_descent = max_descent.max(-font_metrics.descent * font_scale);

                let font_size = if break_ligature {
                    px(f32::from(run_font_size).next_up())
                } else {
                    run_font_size
                };
                unsafe {
                    string.set_attribute(
                        cf_range,
                        kCTFontAttributeName,
                        &font.native_font().clone_with_font_size(font_size.into()),
                    );
                }
                break_ligature = !break_ligature;
            }
        }
        // Retrieve the glyphs from the shaped line, converting UTF16 offsets to UTF8 offsets.
        let line = CTLine::new_with_attributed_string(string.as_concrete_TypeRef());
        let glyph_runs = line.glyph_runs();
        let mut runs = <Vec<ShapedRun>>::with_capacity(glyph_runs.len() as usize);
        let mut ix_converter = StringIndexConverter::new(text);
        // The size of the font run covering a UTF-8 offset (`None`: the line's size).
        let run_font_size = |utf8_ix: usize| {
            let mut end = 0;
            for run in font_runs {
                end += run.len;
                if utf8_ix < end {
                    return run.font_size;
                }
            }
            None
        };
        for run in glyph_runs.into_iter() {
            let attributes = run.attributes().unwrap();
            let font = unsafe {
                attributes
                    .get(kCTFontAttributeName)
                    .downcast::<CTFont>()
                    .unwrap()
            };
            let font_id = self.id_for_native_font(font);

            let glyph_font_size = run
                .string_indices()
                .first()
                .map(|&utf16_ix| {
                    let mut converter = StringIndexConverter::new(text);
                    converter.advance_to_utf16_ix(usize::try_from(utf16_ix).unwrap_or(0));
                    run_font_size(converter.utf8_ix)
                })
                .unwrap_or(None);
            let glyphs = match runs.last_mut() {
                Some(run) if run.font_id == font_id && run.font_size == glyph_font_size => {
                    &mut run.glyphs
                }
                _ => {
                    runs.push(ShapedRun {
                        font_id,
                        font_size: glyph_font_size,
                        glyphs: Vec::with_capacity(run.glyph_count().try_into().unwrap_or(0)),
                    });
                    &mut runs.last_mut().unwrap().glyphs
                }
            };
            for ((&glyph_id, position), &glyph_utf16_ix) in run
                .glyphs()
                .iter()
                .zip(run.positions().iter())
                .zip(run.string_indices().iter())
            {
                let glyph_utf16_ix = usize::try_from(glyph_utf16_ix).unwrap();
                if ix_converter.utf16_ix > glyph_utf16_ix {
                    // We cannot reuse current index converter, as it can only seek forward. Restart the search.
                    ix_converter = StringIndexConverter::new(text);
                }
                ix_converter.advance_to_utf16_ix(glyph_utf16_ix);
                glyphs.push(ShapedGlyph {
                    id: GlyphId(glyph_id as u32),
                    position: point(position.x as f32, position.y as f32).map(px),
                    index: ix_converter.utf8_ix,
                    is_emoji: self.is_emoji(font_id),
                });
            }
        }
        let typographic_bounds = line.get_typographic_bounds();
        LineLayout {
            runs,
            font_size,
            width: typographic_bounds.width.into(),
            ascent: max_ascent.into(),
            descent: max_descent.into(),
            len: text.len(),
        }
    }
}

#[derive(Debug, Clone)]
struct StringIndexConverter<'a> {
    text: &'a str,
    /// Index in UTF-8 bytes
    utf8_ix: usize,
    /// Index in UTF-16 code units
    utf16_ix: usize,
}

impl<'a> StringIndexConverter<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            utf8_ix: 0,
            utf16_ix: 0,
        }
    }

    fn advance_to_utf16_ix(&mut self, utf16_target: usize) {
        for (ix, c) in self.text[self.utf8_ix..].char_indices() {
            if self.utf16_ix >= utf16_target {
                self.utf8_ix += ix;
                return;
            }
            self.utf16_ix += c.len_utf16();
        }
        self.utf8_ix = self.text.len();
    }
}

fn font_kit_metrics_to_metrics(metrics: Metrics) -> FontMetrics {
    FontMetrics {
        units_per_em: metrics.units_per_em,
        ascent: metrics.ascent,
        descent: metrics.descent,
        line_gap: metrics.line_gap,
        underline_position: metrics.underline_position,
        underline_thickness: metrics.underline_thickness,
        cap_height: metrics.cap_height,
        x_height: metrics.x_height,
        bounding_box: bounds_from_rect(metrics.bounding_box),
    }
}

fn bounds_from_rect(rect: RectF) -> Bounds<f32> {
    Bounds {
        origin: point(rect.origin_x(), rect.origin_y()),
        size: size(rect.width(), rect.height()),
    }
}

fn bounds_from_rect_i(rect: RectI) -> Bounds<DevicePixels> {
    Bounds {
        origin: point(DevicePixels(rect.origin_x()), DevicePixels(rect.origin_y())),
        size: size(DevicePixels(rect.width()), DevicePixels(rect.height())),
    }
}

// impl From<Vector2I> for Size<DevicePixels> {
//     fn from(value: Vector2I) -> Self {
//         size(value.x().into(), value.y().into())
//     }
// }

// impl From<RectI> for Bounds<i32> {
//     fn from(rect: RectI) -> Self {
//         Bounds {
//             origin: point(rect.origin_x(), rect.origin_y()),
//             size: size(rect.width(), rect.height()),
//         }
//     }
// }

// impl From<Point<u32>> for Vector2I {
//     fn from(size: Point<u32>) -> Self {
//         Vector2I::new(size.x as i32, size.y as i32)
//     }
// }

fn size_from_vector2f(vec: Vector2F) -> Size<f32> {
    size(vec.x(), vec.y())
}

/// The CSS weight of a font from its Core Text weight trait, on Apple's scale
/// (`NSFontWeight*`: ultralight -0.8 is 100, regular 0 is 400, medium 0.23 is 500,
/// semibold 0.3 is 600, bold 0.4 is 700, black 0.62 is 900), interpolated between
/// those points. `None` if the trait is missing.
fn core_text_css_weight(font: &FontKitFont) -> Option<f32> {
    let traits = font.native_font().all_traits();
    let trait_value = unsafe { traits.get(kCTFontWeightTrait) }
        .downcast::<CFNumber>()?
        .to_f64()? as f32;
    Some(css_weight_from_core_text_trait(trait_value))
}

fn css_weight_from_core_text_trait(value: f32) -> f32 {
    const SCALE: [(f32, f32); 9] = [
        (-0.8, 100.),
        (-0.6, 200.),
        (-0.4, 300.),
        (0., 400.),
        (0.23, 500.),
        (0.3, 600.),
        (0.4, 700.),
        (0.56, 800.),
        (0.62, 900.),
    ];
    if value <= SCALE[0].0 {
        return SCALE[0].1;
    }
    for pair in SCALE.windows(2) {
        let ((t0, w0), (t1, w1)) = (pair[0], pair[1]);
        if value <= t1 {
            return w0 + (value - t0) / (t1 - t0) * (w1 - w0);
        }
    }
    SCALE[SCALE.len() - 1].1
}

fn fontkit_weight(value: FontWeight) -> FontkitWeight {
    FontkitWeight(value.0)
}

fn fontkit_style(style: FontStyle) -> FontkitStyle {
    match style {
        FontStyle::Normal => FontkitStyle::Normal,
        FontStyle::Italic => FontkitStyle::Italic,
        FontStyle::Oblique => FontkitStyle::Oblique,
    }
}

// Some fonts may have no attributes despite `core_text` requiring them (and panicking).
// This is the same version as `core_text` has without `expect` calls.
mod lenient_font_attributes {
    use core_foundation::{
        base::{CFRetain, CFType, TCFType},
        string::{CFString, CFStringRef},
    };
    use core_text::font_descriptor::{
        CTFontDescriptor, CTFontDescriptorCopyAttribute, kCTFontFamilyNameAttribute,
    };

    pub fn family_name(descriptor: &CTFontDescriptor) -> Option<String> {
        unsafe { get_string_attribute(descriptor, kCTFontFamilyNameAttribute) }
    }

    fn get_string_attribute(
        descriptor: &CTFontDescriptor,
        attribute: CFStringRef,
    ) -> Option<String> {
        unsafe {
            let value = CTFontDescriptorCopyAttribute(descriptor.as_concrete_TypeRef(), attribute);
            if value.is_null() {
                return None;
            }

            let value = CFType::wrap_under_create_rule(value);
            assert!(value.instance_of::<CFString>());
            let s = wrap_under_get_rule(value.as_CFTypeRef() as CFStringRef);
            Some(s.to_string())
        }
    }

    unsafe fn wrap_under_get_rule(reference: CFStringRef) -> CFString {
        unsafe {
            assert!(!reference.is_null(), "Attempted to create a NULL object.");
            let reference = CFRetain(reference as *const ::std::os::raw::c_void) as CFStringRef;
            TCFType::wrap_under_create_rule(reference)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::AppleTextSystem;
    use gpui::{
        Bounds, DevicePixels, FontFallbacks, FontFeatures, FontRun, GlyphId, PlatformTextSystem,
        RenderGlyphParams, font, point, px,
    };
    use std::sync::Arc;

    #[test]
    fn test_reload_family_with_features_and_fallbacks() {
        let fonts = AppleTextSystem::new();
        let mut font = font("Helvetica");
        let original_id = fonts.font_id(&font).unwrap();
        font.features = FontFeatures::disable_ligatures();
        font.fallbacks = Some(FontFallbacks(Arc::new(vec!["Times".into()])));
        let configured_id = fonts.font_id(&font).unwrap();
        assert_ne!(original_id, configured_id);
        assert_eq!(fonts.font_id(&font).unwrap(), configured_id);
        assert!(fonts.glyph_for_char(configured_id, 'm').is_some());
        let layout = fonts.layout_line(
            "office",
            px(16.),
            &[FontRun {
                font_id: configured_id,
                len: 6,
                font_size: None,
                letter_spacing: None,
            }],
        );
        assert_eq!(layout.len, 6);
        assert!(!layout.runs.is_empty());
    }

    #[test]
    fn test_font_id_for_postscript_name() {
        use gpui::{Font, FontWeight};
        let fonts = AppleTextSystem::new();
        // A family's face resolves to the id family matching gave it.
        let helvetica = fonts.font_id(&font("Helvetica")).unwrap();
        assert_eq!(
            fonts.font_id_for_postscript_name("Helvetica"),
            Some(helvetica)
        );
        // A named instance of the variable system font is that instance: its advances match
        // the face CSS weight matching picks.
        let bold = fonts.font_id_for_postscript_name(".SFNS-Bold").unwrap();
        let matched = fonts
            .font_id(&Font {
                weight: FontWeight::BOLD,
                ..font(".SystemUIFont")
            })
            .unwrap();
        let regular = fonts.font_id_for_postscript_name(".SFNS-Regular").unwrap();
        let h = fonts.glyph_for_char(bold, 'H').unwrap();
        let advance = |id| fonts.advance(id, h).unwrap().width;
        assert_eq!(advance(bold), advance(matched));
        assert!(advance(bold) > advance(regular));
        // Faces family matching skips (no 'm' glyph) are reachable by name.
        let emoji = fonts
            .font_id_for_postscript_name("AppleColorEmoji")
            .unwrap();
        assert!(fonts.0.read().is_emoji(emoji));
        assert_eq!(
            fonts.font_id_for_postscript_name("AppleColorEmoji"),
            Some(emoji)
        );
        // System UI faces Core Text opens only through its UI font APIs: the UI emoji font and
        // fallback fonts of the UI font's cascade.
        for name in [
            ".AppleColorEmojiUI",
            ".SFHebrew-Regular",
            ".SFArabic-Bold",
            ".SFNSMono-Regular",
        ] {
            let id = fonts.font_id_for_postscript_name(name).unwrap();
            assert_eq!(
                fonts
                    .0
                    .read()
                    .postscript_names_by_font_id
                    .get(&id)
                    .map(String::as_str),
                Some(name)
            );
        }
        // Core Text substitutes a font for unknown names; the lookup does not.
        assert_eq!(
            fonts.font_id_for_postscript_name("NoSuchFont-Regular"),
            None
        );
        assert_eq!(
            fonts.font_id_for_postscript_name(".NoSuchFont-Regular"),
            None
        );
    }

    #[test]
    fn test_css_weights_select_the_matching_system_face() {
        use gpui::{Font, FontWeight};
        let fonts = AppleTextSystem::new();
        for (weight, face) in [
            (100., "Ultralight"),
            (200., "Thin"),
            (300., "Light"),
            (400., "Regular"),
            (500., "Medium"),
            (600., "Semibold"),
            (700., "Bold"),
            (900., "Black"),
        ] {
            let font = Font {
                weight: FontWeight(weight),
                ..font(".SystemUIFont")
            };
            let id = fonts.font_id(&font).unwrap();
            let name = fonts.0.read().fonts[id.0].full_name();
            assert!(name.ends_with(face), "weight {weight}: {name}");
        }
        assert_eq!(super::css_weight_from_core_text_trait(0.23), 500.);
        assert_eq!(super::css_weight_from_core_text_trait(-1.), 100.);
        assert_eq!(
            super::css_weight_from_core_text_trait(0.1),
            400. + 0.1 / 0.23 * 100.
        );
    }

    #[test]
    fn test_runs_shape_at_their_own_font_size() {
        let fonts = AppleTextSystem::new();
        let font_id = fonts.font_id(&font("Helvetica")).unwrap();
        let line = "abab";
        let uniform = fonts.layout_line(line, px(16.), &[FontRun::new(4, font_id)]);
        let mixed = fonts.layout_line(
            line,
            px(16.),
            &[
                FontRun::new(2, font_id),
                FontRun {
                    font_size: Some(px(32.)),
                    ..FontRun::new(2, font_id)
                },
            ],
        );
        let sizes: Vec<_> = mixed.runs.iter().map(|run| run.font_size).collect();
        assert_eq!(sizes, vec![None, Some(px(32.))]);
        // The 32 px "ab" is twice as wide as at 16 px.
        let half = uniform.runs[0].glyphs[2].position.x;
        assert!(
            (mixed.width - half * 3.).abs() < px(0.5),
            "{:?} vs {:?}",
            mixed.width,
            half * 3.
        );
        // The line's ascent and descent cover the larger run.
        assert!((mixed.ascent - uniform.ascent * 2.).abs() < px(0.01));
        assert!((mixed.descent - uniform.descent * 2.).abs() < px(0.01));
        assert_eq!(mixed.run_font_size(&mixed.runs[0]), px(16.));
        assert_eq!(mixed.run_font_size(&mixed.runs[1]), px(32.));
    }

    #[test]
    fn test_layout_line_bom_char() {
        let fonts = AppleTextSystem::new();
        let font_id = fonts.font_id(&font("Helvetica")).unwrap();
        let line = "\u{feff}";
        let mut style = FontRun {
            font_id,
            len: line.len(),
            font_size: None,
            letter_spacing: None,
        };

        let layout = fonts.layout_line(line, px(16.), &[style]);
        assert_eq!(layout.len, line.len());
        assert!(layout.runs.is_empty());

        let line = "a\u{feff}b";
        style.len = line.len();
        let layout = fonts.layout_line(line, px(16.), &[style]);
        assert_eq!(layout.len, line.len());
        assert_eq!(layout.runs.len(), 1);
        assert_eq!(layout.runs[0].glyphs.len(), 2);
        assert_eq!(layout.runs[0].glyphs[0].id, GlyphId(68u32)); // a
        // There's no glyph for \u{feff}
        assert_eq!(layout.runs[0].glyphs[1].id, GlyphId(69u32)); // b

        let line = "\u{feff}ab";
        let font_runs = &[
            FontRun {
                len: "\u{feff}".len(),
                font_id,
                font_size: None,
                letter_spacing: None,
            },
            FontRun {
                len: "ab".len(),
                font_id,
                font_size: None,
                letter_spacing: None,
            },
        ];
        let layout = fonts.layout_line(line, px(16.), font_runs);
        assert_eq!(layout.len, line.len());
        assert_eq!(layout.runs.len(), 1);
        assert_eq!(layout.runs[0].glyphs.len(), 2);
        // There's no glyph for \u{feff}
        assert_eq!(layout.runs[0].glyphs[0].id, GlyphId(68u32)); // a
        assert_eq!(layout.runs[0].glyphs[1].id, GlyphId(69u32)); // b
    }

    #[test]
    fn test_layout_line_zwnj_insertion() {
        let fonts = AppleTextSystem::new();
        let font_id = fonts.font_id(&font("Helvetica")).unwrap();

        let text = "hello world";
        let font_runs = &[
            FontRun {
                font_id,
                len: 5,
                font_size: None,
                letter_spacing: None,
            }, // "hello"
            FontRun {
                font_id,
                len: 6,
                font_size: None,
                letter_spacing: None,
            }, // " world"
        ];

        let layout = fonts.layout_line(text, px(16.), font_runs);
        assert_eq!(layout.len, text.len());

        for run in &layout.runs {
            for glyph in &run.glyphs {
                assert!(
                    glyph.index < text.len(),
                    "Glyph index {} is out of bounds for text length {}",
                    glyph.index,
                    text.len()
                );
            }
        }

        // Test with different font runs - should not insert ZWNJ
        let font_id2 = fonts.font_id(&font("Times")).unwrap_or(font_id);
        let font_runs_different = &[
            FontRun {
                font_id,
                len: 5,
                font_size: None,
                letter_spacing: None,
            }, // "hello"
            // " world"
            FontRun {
                font_id: font_id2,
                len: 6,
                font_size: None,
                letter_spacing: None,
            },
        ];

        let layout2 = fonts.layout_line(text, px(16.), font_runs_different);
        assert_eq!(layout2.len, text.len());

        for run in &layout2.runs {
            for glyph in &run.glyphs {
                assert!(
                    glyph.index < text.len(),
                    "Glyph index {} is out of bounds for text length {}",
                    glyph.index,
                    text.len()
                );
            }
        }
    }

    #[test]
    fn test_layout_line_zwnj_edge_cases() {
        let fonts = AppleTextSystem::new();
        let font_id = fonts.font_id(&font("Helvetica")).unwrap();

        let text = "hello";
        let font_runs = &[FontRun {
            font_id,
            len: 5,
            font_size: None,
            letter_spacing: None,
        }];
        let layout = fonts.layout_line(text, px(16.), font_runs);
        assert_eq!(layout.len, text.len());

        let text = "abc";
        let font_runs = &[
            FontRun {
                font_id,
                len: 1,
                font_size: None,
                letter_spacing: None,
            }, // "a"
            FontRun {
                font_id,
                len: 1,
                font_size: None,
                letter_spacing: None,
            }, // "b"
            FontRun {
                font_id,
                len: 1,
                font_size: None,
                letter_spacing: None,
            }, // "c"
        ];
        let layout = fonts.layout_line(text, px(16.), font_runs);
        assert_eq!(layout.len, text.len());

        for run in &layout.runs {
            for glyph in &run.glyphs {
                assert!(
                    glyph.index < text.len(),
                    "Glyph index {} is out of bounds for text length {}",
                    glyph.index,
                    text.len()
                );
            }
        }

        // Test with empty text
        let text = "";
        let font_runs = &[];
        let layout = fonts.layout_line(text, px(16.), font_runs);
        assert_eq!(layout.len, 0);
        assert!(layout.runs.is_empty());
    }

    #[test]
    fn test_raster_bounds_contain_glyph_at_every_font_size() {
        let fonts = AppleTextSystem::new();
        // The system font has optical sizes: its outlines differ between point sizes.
        let font_id = fonts.font_id(&font(".SystemUIFont")).unwrap();
        let margin = 16;
        for ch in ['o', 'e', 't', 'n', 'g', 'W'] {
            let glyph_id = fonts.glyph_for_char(font_id, ch).unwrap();
            for font_size in [9., 12., 16., 24., 48., 96.] {
                for scale_factor in [1., 2., 8., 32.] {
                    for subpixel_variant in [point(0, 0), point(3, 0)] {
                        let params = RenderGlyphParams {
                            font_id,
                            glyph_id,
                            font_size: px(font_size),
                            subpixel_variant,
                            scale_factor,
                            is_emoji: false,
                            subpixel_rendering: false,
                            dilation: 0,
                        };
                        let bounds = fonts.glyph_raster_bounds(&params).unwrap();
                        let (size, _) = fonts.rasterize_glyph(&params, bounds).unwrap();
                        // Rasterize again with room to spare and look for ink outside the
                        // part of the canvas the raster bounds cover.
                        let (canvas_size, alpha) = fonts
                            .rasterize_glyph(&params, bounds.dilate(DevicePixels(margin)))
                            .unwrap();
                        let canvas_width = canvas_size.width.0 as usize;
                        let inside = Bounds::new(
                            point(margin as usize, margin as usize),
                            gpui::size(size.width.0 as usize, size.height.0 as usize),
                        );
                        let clipped = alpha
                            .iter()
                            .enumerate()
                            .filter(|&(i, &a)| {
                                a > 0
                                    && !inside.contains(&point(i % canvas_width, i / canvas_width))
                            })
                            .count();
                        assert_eq!(
                            clipped, 0,
                            "{ch:?} at {font_size}pt, scale {scale_factor}, \
                             subpixel variant {subpixel_variant:?}: \
                             inked pixels outside the raster bounds {bounds:?}"
                        );
                    }
                }
            }
        }
    }
}
