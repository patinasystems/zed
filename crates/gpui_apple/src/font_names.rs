//! The PostScript names in font files, read from the files' `name` and `fvar` tables with plain
//! reads (a few kilobytes a face), not by Core Text opening each file: asking Core Text for a
//! file's faces (`CTFontManagerCreateFontDescriptorsFromURL`) maps the whole file, and indexing
//! every available font that way mapped about 107 MB and allocated about 22 MB mid-session.

use std::{fs::File, os::unix::fs::FileExt, path::Path, path::PathBuf};

/// PostScript names with the file each is in: the names in one buffer, sorted for lookup.
#[derive(Default)]
pub(crate) struct FontNames {
    names: String,
    /// Name `i` is `names[ends[i - 1]..ends[i]]`.
    ends: Vec<u32>,
    /// Name `i`'s file, an index into `paths`.
    file: Vec<u32>,
    paths: Vec<PathBuf>,
    /// The names' indices sorted by name, the first file's first for a name in several files.
    order: Vec<u32>,
}

impl FontNames {
    /// Indexes the faces in `paths` (each file once). `faces` reads a file's names when
    /// [`sfnt_names`] cannot (no sfnt, or variable instances without their own names).
    pub(crate) fn index(
        paths: impl IntoIterator<Item = PathBuf>,
        mut faces: impl FnMut(&Path) -> Vec<String>,
    ) -> Self {
        let mut this = Self::default();
        let mut seen = std::collections::HashSet::new();
        let mut scratch = Vec::new();
        for path in paths {
            if !seen.insert(path.clone()) {
                continue;
            }
            if sfnt_names(&path, &mut scratch).is_none() {
                scratch = faces(&path);
            }
            if scratch.is_empty() {
                continue;
            }
            let file = this.paths.len() as u32;
            for name in &scratch {
                this.names.push_str(name);
                this.ends.push(this.names.len() as u32);
                this.file.push(file);
            }
            this.paths.push(path);
        }
        let mut order: Vec<u32> = (0..this.ends.len() as u32).collect();
        // Stable: among equal names, the earlier file stays first.
        order.sort_by(|&a, &b| this.name(a).cmp(this.name(b)));
        this.order = order;
        this
    }

    fn name(&self, i: u32) -> &str {
        let i = i as usize;
        let start = if i == 0 { 0 } else { self.ends[i - 1] as usize };
        &self.names[start..self.ends[i] as usize]
    }

    /// The file holding the face named `name`.
    pub(crate) fn get(&self, name: &str) -> Option<&Path> {
        let at = self.order.partition_point(|&i| self.name(i) < name);
        let &i = self.order.get(at)?;
        (self.name(i) == name).then(|| self.paths[self.file[i as usize] as usize].as_path())
    }

    pub(crate) fn len(&self) -> usize {
        self.ends.len()
    }
}

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn read(file: &File, at: u64, len: usize) -> Option<Vec<u8>> {
    let mut buf = vec![0; len];
    file.read_exact_at(&mut buf, at).ok()?;
    Some(buf)
}

/// The PostScript names of the faces in the sfnt file at `path` (a font or a collection) into
/// `out`, as Core Text names them: each face's name ID 6, then its variable instances' own
/// names. `None` when the file is no sfnt, or a face has a variable instance without a
/// PostScript name ID (Core Text makes up that name).
pub(crate) fn sfnt_names(path: &Path, out: &mut Vec<String>) -> Option<()> {
    out.clear();
    let file = File::open(path).ok()?;
    let head = read(&file, 0, 12)?;
    let faces: Vec<u32> = match &head[..4] {
        b"ttcf" => {
            let count = u32_at(&head, 8)? as usize;
            if count == 0 || count > 4096 {
                return None;
            }
            let offsets = read(&file, 12, count * 4)?;
            (0..count).map(|i| u32_at(&offsets, i * 4)).collect::<Option<_>>()?
        }
        [0, 1, 0, 0] | b"OTTO" | b"true" => vec![0],
        _ => return None,
    };
    for face in faces {
        face_names(&file, face as u64, out)?;
    }
    Some(())
}

fn face_names(file: &File, at: u64, out: &mut Vec<String>) -> Option<()> {
    let head = read(file, at, 12)?;
    let tables = u16_at(&head, 4)? as usize;
    let directory = read(file, at + 12, tables * 16)?;
    let table = |tag: &[u8; 4]| -> Option<Vec<u8>> {
        let i = (0..tables).find(|i| directory.get(i * 16..i * 16 + 4) == Some(tag))?;
        let (offset, len) = (u32_at(&directory, i * 16 + 8)?, u32_at(&directory, i * 16 + 12)?);
        read(file, offset as u64, len as usize)
    };
    let name = table(b"name")?;
    let postscript = name_string(&name, 6)?;
    if !out.contains(&postscript) {
        out.push(postscript);
    }
    if let Some(fvar) = table(b"fvar") {
        let axes_at = u16_at(&fvar, 4)? as usize;
        let (axes, axis_size) = (u16_at(&fvar, 8)? as usize, u16_at(&fvar, 10)? as usize);
        let (instances, instance_size) = (u16_at(&fvar, 12)? as usize, u16_at(&fvar, 14)? as usize);
        let first = axes_at + axes * axis_size;
        for i in 0..instances {
            // An instance's PostScript name ID follows its coordinates, when its record has room.
            if instance_size < axes * 4 + 6 {
                return None;
            }
            let id = u16_at(&fvar, first + i * instance_size + 4 + axes * 4)?;
            if id == 0xffff {
                return None;
            }
            let instance = name_string(&name, id)?;
            if !out.contains(&instance) {
                out.push(instance);
            }
        }
    }
    Some(())
}

/// Name `id` of a `name` table: the Windows Unicode (UTF-16BE) record, else the Macintosh
/// Roman one (PostScript names are ASCII).
fn name_string(name: &[u8], id: u16) -> Option<String> {
    let (count, strings) = (u16_at(name, 2)? as usize, u16_at(name, 4)? as usize);
    let mut mac = None;
    for r in 0..count {
        let at = 6 + r * 12;
        if u16_at(name, at + 6)? != id {
            continue;
        }
        let (platform, encoding) = (u16_at(name, at)?, u16_at(name, at + 2)?);
        let (len, offset) = (u16_at(name, at + 8)? as usize, u16_at(name, at + 10)? as usize);
        let bytes = name.get(strings + offset..strings + offset + len)?;
        match (platform, encoding) {
            (3, 0 | 1) | (0, _) => {
                let units: Vec<u16> =
                    bytes.chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
                return String::from_utf16(&units).ok();
            }
            (1, 0) if mac.is_none() => mac = Some(bytes.iter().map(|&b| b as char).collect()),
            _ => {}
        }
    }
    mac
}
