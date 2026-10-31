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

use std::{
    cell::{Cell, RefCell},
    hash::{DefaultHasher, Hash, Hasher},
    path::{Path, PathBuf},
    ptr,
    sync::OnceLock,
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
pub(crate) struct PipelineCache<'a> {
    device: &'a metal::DeviceRef,
    /// The archive file, when a cache folder is set.
    path: Option<PathBuf>,
    /// The archive read from `path`, when it exists and Metal accepts it.
    archive: Option<metal::BinaryArchive>,
    /// Every pipeline built, to write a new archive from.
    descriptors: RefCell<Vec<RenderPipelineDescriptor>>,
    missed: Cell<bool>,
}

impl<'a> PipelineCache<'a> {
    /// The cache for pipelines built from a library made from `library` (its bytes or source).
    pub(crate) fn new(device: &'a metal::DeviceRef, library: &[u8]) -> Self {
        let path = CACHE_DIR.get().map(|dir| {
            let mut hasher = DefaultHasher::new();
            library.hash(&mut hasher);
            device.name().hash(&mut hasher);
            dir.join(format!("gpui-pipelines-{:016x}.metalar", hasher.finish()))
        });
        let archive = path
            .as_deref()
            .filter(|path| path.is_file())
            .and_then(|path| {
                let descriptor = metal::BinaryArchiveDescriptor::new();
                let url = file_url(path);
                // SAFETY: an NSURL is what `setUrl:` takes; `url` outlives the call.
                descriptor.set_url(unsafe {
                    metal::URLRef::from_ptr(Retained::as_ptr(&url).cast_mut().cast())
                });
                device
                    .new_binary_archive_with_descriptor(&descriptor)
                    .map_err(|error| {
                        log::warn!("reading the pipeline archive {}: {error}", path.display())
                    })
                    .ok()
            });
        Self {
            device,
            path,
            archive,
            descriptors: RefCell::default(),
            missed: Cell::new(false),
        }
    }

    /// The pipeline for `descriptor`: from the archive when it has the pipeline's code, else
    /// compiled.
    pub(crate) fn build(
        &self,
        descriptor: &RenderPipelineDescriptorRef,
    ) -> metal::RenderPipelineState {
        if self.path.is_some() {
            self.descriptors.borrow_mut().push(descriptor.to_owned());
        }
        if let Some(archive) = &self.archive {
            descriptor.set_binary_archives(&[archive]);
            let found = archived_pipeline(self.device, descriptor);
            descriptor.set_binary_archives(&[]);
            if let Some(state) = found {
                return state;
            }
        }
        self.missed.set(true);
        self.device
            .new_render_pipeline_state(descriptor)
            .expect("could not create render pipeline state")
    }

    /// Writes the archive when a pipeline was missing from it. Failures are logged: the next
    /// launch compiles again.
    pub(crate) fn finish(self) {
        let Some(path) = self.path.filter(|_| self.missed.get()) else {
            return;
        };
        if let Err(error) = write_archive(self.device, &self.descriptors.into_inner(), &path) {
            log::warn!("writing the pipeline archive {}: {error}", path.display());
        }
    }
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
    let partial = path.with_extension(format!("partial-{}", std::process::id()));
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
