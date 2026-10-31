//! Render pipelines kept between launches in a Metal binary archive.
//!
//! Creating a render pipeline from a library compiles its functions for this GPU. The compiler
//! looks the result up in the system's shader cache, which it opens in the process the first time:
//! about 3 MB of index and lookup tables (`MTLCompilerFSCache`, `libCoreFSCache`) that stay for
//! the process's life. A pipeline whose code is in a binary archive skips the compiler and the
//! cache. With a cache folder set ([`set_pipeline_cache_dir`]), each renderer loads its pipelines
//! from an archive there, and writes the archive again when any pipeline was missing from it: on
//! the first launch, and after an OS or GPU driver update (Metal keys archived code by both, so
//! stale code is a miss, never a wrong pipeline).
//!
//! With an archive to read, a renderer makes each pipeline the first time it draws with it
//! ([`Pipeline::state`]): a page that draws quads and glyphs never makes the other pipelines,
//! which cost about 40 KB each even from the archive. Without one, it compiles them all at once,
//! as the archive is written from them.

use std::{
    cell::{Cell, OnceCell, RefCell},
    hash::{DefaultHasher, Hash, Hasher},
    path::{Path, PathBuf},
    ptr,
    rc::Rc,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{MTLPipelineOption, RenderPipelineDescriptor, RenderPipelineDescriptorRef};
use objc2::{msg_send, rc::Retained, runtime::AnyObject};
use objc2_foundation::{NSString, NSURL};

static CACHE_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps renderers' compiled pipelines in `dir` between launches (module docs). Set it once,
/// before the first window or headless renderer; later calls are ignored and return `false`.
pub fn set_pipeline_cache_dir(dir: PathBuf) -> bool {
    CACHE_DIR.set(dir).is_ok()
}

/// One renderer's pipelines while it builds them: [`Self::build`] each, then [`Self::finish`].
pub(crate) struct PipelineCache {
    shared: Rc<Shared>,
}

/// What a renderer's pipelines share: the device, the archive and every pipeline's descriptor.
struct Shared {
    device: metal::Device,
    /// The archive file, when a cache folder is set.
    path: Option<PathBuf>,
    /// The archive read from `path`, when it exists and Metal accepts it; read again after a
    /// deferred pipeline's miss rewrites it.
    archive: RefCell<Option<metal::BinaryArchive>>,
    /// Every pipeline built, to write a new archive from.
    descriptors: RefCell<Vec<RenderPipelineDescriptor>>,
    missed: Cell<bool>,
    /// A pipeline made after construction missed the archive, and the archive was written again.
    rewritten: Cell<bool>,
}

/// A renderer's pipeline: made when built ([`PipelineCache::build`]) without an archive to read,
/// else when first drawn with ([`Self::state`]).
pub(crate) struct Pipeline {
    state: OnceCell<metal::RenderPipelineState>,
    /// The descriptor to make it from, until it is made.
    descriptor: Option<RenderPipelineDescriptor>,
    shared: Rc<Shared>,
}

impl Pipeline {
    /// The pipeline state, made from the archive the first time (compiled on a miss, which writes
    /// the archive again).
    pub(crate) fn state(&self) -> &metal::RenderPipelineState {
        self.state.get_or_init(|| {
            let descriptor = self.descriptor.as_deref().expect("a deferred pipeline keeps its descriptor");
            let shared = &self.shared;
            if let Some(state) = shared.archived(descriptor) {
                return state;
            }
            let state = shared.compile(descriptor);
            if !shared.rewritten.replace(true) {
                // The new archive holds every pipeline's code: the others read it, not the stale one.
                shared.write();
                if let Some(path) = &shared.path {
                    *shared.archive.borrow_mut() = open_archive(&shared.device, path);
                }
            }
            state
        })
    }

    /// Whether the pipeline state was made (tests).
    #[cfg(test)]
    pub(crate) fn is_made(&self) -> bool {
        self.state.get().is_some()
    }
}

impl PipelineCache {
    /// The cache for pipelines built from a library made from `library` (its bytes or source).
    pub(crate) fn new(device: &metal::DeviceRef, library: &[u8]) -> Self {
        let path = CACHE_DIR.get().map(|dir| {
            let mut hasher = DefaultHasher::new();
            library.hash(&mut hasher);
            device.name().hash(&mut hasher);
            dir.join(format!("gpui-pipelines-{:016x}.metalar", hasher.finish()))
        });
        Self::at(device, path)
    }

    /// The cache whose archive is the file at `path` (`None`: no archive).
    fn at(device: &metal::DeviceRef, path: Option<PathBuf>) -> Self {
        let archive = path.as_deref().and_then(|path| open_archive(device, path));
        Self {
            shared: Rc::new(Shared {
                device: device.to_owned(),
                path,
                archive: RefCell::new(archive),
                descriptors: RefCell::default(),
                missed: Cell::new(false),
                rewritten: Cell::new(false),
            }),
        }
    }

    /// The pipeline for `descriptor`: deferred to its first use when there is an archive to read
    /// it from, else compiled now.
    pub(crate) fn build(&self, descriptor: &RenderPipelineDescriptorRef) -> Pipeline {
        let shared = &self.shared;
        if shared.path.is_some() {
            shared.descriptors.borrow_mut().push(descriptor.to_owned());
        }
        let pipeline = Pipeline { state: OnceCell::new(), descriptor: None, shared: shared.clone() };
        if shared.archive.borrow().is_some() {
            return Pipeline { descriptor: Some(descriptor.to_owned()), ..pipeline };
        }
        shared.missed.set(true);
        let _ = pipeline.state.set(shared.compile(descriptor));
        pipeline
    }

    /// Writes the archive when a pipeline was missing from it. Failures are logged: the next
    /// launch compiles again.
    pub(crate) fn finish(self) {
        if self.shared.missed.get() {
            self.shared.write();
        }
    }
}

impl Shared {
    /// The pipeline from the archive, when it holds the pipeline's code.
    fn archived(&self, descriptor: &RenderPipelineDescriptorRef) -> Option<metal::RenderPipelineState> {
        let archive = self.archive.borrow();
        let archive = archive.as_ref()?;
        descriptor.set_binary_archives(&[archive]);
        let found = archived_pipeline(&self.device, descriptor);
        descriptor.set_binary_archives(&[]);
        found
    }

    fn compile(&self, descriptor: &RenderPipelineDescriptorRef) -> metal::RenderPipelineState {
        self.device
            .new_render_pipeline_state(descriptor)
            .expect("could not create render pipeline state")
    }

    /// Writes the archive from every pipeline's descriptor. Failures are logged.
    fn write(&self) {
        let Some(path) = &self.path else { return };
        if let Err(error) = write_archive(&self.device, &self.descriptors.borrow(), path) {
            log::warn!("writing the pipeline archive {}: {error}", path.display());
        }
    }
}

/// The archive at `path`, when the file exists and Metal accepts it.
fn open_archive(device: &metal::DeviceRef, path: &Path) -> Option<metal::BinaryArchive> {
    if !path.is_file() {
        return None;
    }
    let descriptor = metal::BinaryArchiveDescriptor::new();
    let url = file_url(path);
    // SAFETY: an NSURL is what `setUrl:` takes; `url` outlives the call.
    descriptor.set_url(unsafe { metal::URLRef::from_ptr(Retained::as_ptr(&url).cast_mut().cast()) });
    device
        .new_binary_archive_with_descriptor(&descriptor)
        .map_err(|error| log::warn!("reading the pipeline archive {}: {error}", path.display()))
        .ok()
}

/// The pipeline for `descriptor` only if its binary archives hold its code.
fn archived_pipeline(
    device: &metal::DeviceRef,
    descriptor: &RenderPipelineDescriptorRef,
) -> Option<metal::RenderPipelineState> {
    let device: *mut AnyObject = device.as_ptr().cast();
    let descriptor: *mut AnyObject = descriptor.as_ptr().cast();
    let mut error: *mut AnyObject = ptr::null_mut();
    // SAFETY: `newRenderPipelineStateWithDescriptor:options:reflection:error:` with no reflection
    // out-pointer; it returns a +1 pipeline state, or nil with `error` set (autoreleased).
    let state: *mut AnyObject = unsafe {
        msg_send![
            device,
            newRenderPipelineStateWithDescriptor: descriptor,
            options: MTLPipelineOption::FailOnBinaryArchiveMiss.bits(),
            reflection: ptr::null_mut::<*mut AnyObject>(),
            error: &mut error
        ]
    };
    // SAFETY: a non-nil result is an owned MTLRenderPipelineState.
    (!state.is_null()).then(|| unsafe { metal::RenderPipelineState::from_ptr(state.cast()) })
}

/// A new archive of `descriptors`' code, written next to `path` and moved over it, so a reader
/// never sees a partial file.
fn write_archive(
    device: &metal::DeviceRef,
    descriptors: &[RenderPipelineDescriptor],
    path: &Path,
) -> Result<(), String> {
    let archive =
        device.new_binary_archive_with_descriptor(&metal::BinaryArchiveDescriptor::new())?;
    for descriptor in descriptors {
        archive.add_render_pipeline_functions_with_descriptor(descriptor)?;
    }
    let dir = path.parent().ok_or("no parent folder")?;
    std::fs::create_dir_all(dir).map_err(|error| error.to_string())?;
    // Unique per write: renderers in other processes, or on other threads, may write it too.
    static WRITES: AtomicU64 = AtomicU64::new(0);
    let write = WRITES.fetch_add(1, Ordering::Relaxed);
    let partial = path.with_extension(format!("partial-{}-{write}", std::process::id()));
    let url = file_url(&partial);
    // SAFETY: an NSURL is what `serializeToURL:error:` takes; `url` outlives the call.
    archive.serialize_to_url(unsafe {
        metal::URLRef::from_ptr(Retained::as_ptr(&url).cast_mut().cast())
    })?;
    std::fs::rename(&partial, path).map_err(|error| {
        let _ = std::fs::remove_file(&partial);
        error.to_string()
    })
}

fn file_url(path: &Path) -> Retained<NSURL> {
    NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHADERS: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        vertex float4 v(uint id [[vertex_id]]) { return float4(float(id), 0.0, 0.0, 1.0); }
        fragment float4 red() { return float4(1.0, 0.0, 0.0, 1.0); }
        fragment float4 blue() { return float4(0.0, 0.0, 1.0, 1.0); }
    "#;

    fn descriptor(library: &metal::LibraryRef, fragment: &str) -> RenderPipelineDescriptor {
        let descriptor = RenderPipelineDescriptor::new();
        descriptor.set_vertex_function(Some(&library.get_function("v", None).unwrap()));
        descriptor.set_fragment_function(Some(&library.get_function(fragment, None).unwrap()));
        descriptor
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        descriptor
    }

    #[test]
    fn pipelines_in_the_archive_are_made_on_first_use() {
        let Some(device) = metal::Device::system_default() else { return };
        let library = device
            .new_library_with_source(SHADERS, &metal::CompileOptions::new())
            .unwrap();
        let dir = std::env::temp_dir().join(format!("gpui-pipelines-test-{}", std::process::id()));
        let path = dir.join("test.metalar");
        let _ = std::fs::remove_file(&path);

        // No archive yet: compiled at once, and the archive written from them.
        let first = PipelineCache::at(&device, Some(path.clone()));
        let red = first.build(&descriptor(&library, "red"));
        assert!(red.is_made());
        first.finish();
        assert!(path.is_file());
        let written = std::fs::metadata(&path).unwrap().modified().unwrap();

        // With the archive: made when first drawn with, from the archive (no rewrite).
        let second = PipelineCache::at(&device, Some(path.clone()));
        let red = second.build(&descriptor(&library, "red"));
        assert!(!red.is_made());
        let _ = red.state();
        assert!(red.is_made());
        assert!(!second.shared.rewritten.get());

        // A pipeline the archive lacks compiles on first use and writes the archive again.
        let blue = second.build(&descriptor(&library, "blue"));
        assert!(!blue.is_made());
        let _ = blue.state();
        assert!(second.shared.rewritten.get());
        // The rewritten archive answers the rest of this renderer's pipelines.
        let blue_again = second.build(&descriptor(&library, "blue"));
        assert!(second.shared.archived(blue_again.descriptor.as_deref().unwrap()).is_some());
        second.finish();
        assert!(std::fs::metadata(&path).unwrap().modified().unwrap() >= written);
        let third = PipelineCache::at(&device, Some(path.clone()));
        let blue = third.build(&descriptor(&library, "blue"));
        assert!(third.shared.archived(blue.descriptor.as_deref().unwrap()).is_some(), "the rewritten archive holds it");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
