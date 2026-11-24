//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! in GPU-local memory (`D3D12_HEAP_TYPE_DEFAULT`) on first use, and the
//! ranges each submission reads (vertex, index and constant data, buffer
//! views) are brought up to date from title memory before it runs via pooled
//! `UPLOAD` staging copies: the 64 KiB pages the title wrote since they were
//! last copied ([`crate::watch`]: write watching, or read-only guards on
//! physical-page views), all of it where neither works. A title GPU address
//! inside the buffer is the same offset in its copy. Ranges shaders write
//! are read back through pooled `READBACK` staging and merged into title
//! memory after the submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// One staged upload copy to record onto the submission's upload list.
pub(crate) struct UploadCopy {
    pub(crate) shadow: Arc<Shadow>,
    pub(crate) src: ID3D12Resource,
    pub(crate) src_offset: u64,
    pub(crate) dst_offset: u64,
    pub(crate) len: u64,
}

/// A title buffer's host copy in `D3D12_HEAP_TYPE_DEFAULT` memory.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; mutable state is mutex-guarded.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

/// Coalesces overlapping or adjacent `[from, to)` byte ranges in place.
pub(crate) fn merge_ranges(ranges: &mut Vec<(usize, usize)>) {
    ranges.retain(|&(a, b)| a < b);
    if ranges.len() <= 1 {
        return;
    }
    ranges.sort_unstable();
    let mut out = 0;
    for i in 1..ranges.len() {
        let (a, b) = ranges[i];
        if a <= ranges[out].1 {
            ranges[out].1 = ranges[out].1.max(b);
        } else {
            out += 1;
            ranges[out] = (a, b);
        }
    }
    ranges.truncate(out + 1);
}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    pub(crate) fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (
            from - start,
            (address.saturating_add(len)).clamp(from, end) - start,
        )
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: records that a submitted
    /// list's shaders may write `[address, address + len)` so later refreshes
    /// in the same submission do not overwrite it.
    pub(crate) fn mark_pending(&self, address: usize, len: usize) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address.saturating_add(len)));
    }

    /// After the submission completes and readbacks are merged: clears all
    /// pending shader-write ranges.
    pub(crate) fn clear_pending(&self) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// After the submission ran: compares the readback snapshots taken before
    /// (`old`) and after (`after`) the submission for buffer offset range
    /// `[from, from + old.len())`, and copies only the bytes the GPU modified
    /// back into title memory (unmapped pages left alone). Preserves any
    /// concurrent CPU writes to other bytes on the same page; pages without
    /// concurrent CPU writes are marked current so our own write-back is not
    /// re-uploaded on the next submission. Returns the bytes written.
    pub(crate) fn copy_back_range(&self, from: usize, old: &[u8], after: &[u8]) -> usize {
        debug_assert_eq!(old.len(), after.len());
        let len = old.len().min(after.len());
        if len == 0 {
            return 0;
        }
        let start = self.start as usize;
        let to = from + len;
        let mut written = 0;
        // By 64 KiB piece: compare after vs before, touching title memory
        // only where the GPU actually wrote.
        const PIECE: usize = 64 << 10;
        let mut title = vec![0u8; PIECE];
        let mut synced = Vec::new();
        let mut at = from;
        while at < to {
            let n = (PIECE - (start + at) % PIECE).min(to - at);
            let rel = at - from;
            let host = &after[rel..rel + n];
            let prev = &old[rel..rel + n];
            let title = &mut title[..n];
            if host != prev && crate::host::read_into((start + at) as *const u8, title) {
                let mut changed = false;
                let mut cpu_diff = false;
                for ((t, &h), &o) in title.iter_mut().zip(host).zip(prev) {
                    if h != o {
                        if *t != h {
                            *t = h;
                            changed = true;
                        }
                    } else if *t != h {
                        cpu_diff = true;
                    }
                }
                if changed {
                    crate::watch::writing(start + at, n);
                    crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                    written += n;
                }
                // Only mark the page synced if title memory and VRAM now match
                // byte for byte; if the title concurrently wrote other bytes on
                // this page, leave it dirty so the next refresh uploads them.
                if !cpu_diff {
                    synced.push((start + at) / PAGE);
                }
            }
            at += n;
        }
        let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
        if first < last
            && !synced.is_empty()
            && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
            && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
        {
            let base = start / PAGE;
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                copied[page - base] = now;
            }
        }
        written
    }

    /// At submission: stages what changed of `[address, address + len)` since
    /// it was last copied (all of it if the memory is not write-watched) into
    /// upload memory allocated via `alloc`, pushing GPU copy descriptors onto
    /// `out`. Returns the bytes staged.
    pub(crate) fn update(
        self: &Arc<Self>,
        address: usize,
        len: usize,
        alloc: &mut impl FnMut(usize) -> Option<(ID3D12Resource, u64, *mut u8)>,
        out: &mut Vec<UploadCopy>,
    ) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), address.saturating_add(len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a, alloc, out);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to, alloc, out);
            }
            return bytes;
        }
        self.update_free(from, to, alloc, out)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(
        self: &Arc<Self>,
        from: usize,
        to: usize,
        alloc: &mut impl FnMut(usize) -> Option<(ID3D12Resource, u64, *mut u8)>,
        out: &mut Vec<UploadCopy>,
    ) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written, alloc, out);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.stage(from, to, alloc, out)
    }

    /// Stages the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(
        self: &Arc<Self>,
        from: usize,
        to: usize,
        now: u64,
        written: &[(usize, u64)],
        alloc: &mut impl FnMut(usize) -> Option<(ID3D12Resource, u64, *mut u8)>,
        out: &mut Vec<UploadCopy>,
    ) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.stage((first * PAGE).max(from), (page * PAGE).min(to), alloc, out);
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.stage(
                (first * PAGE).max(from),
                ((last + 1) * PAGE).min(to),
                alloc,
                out,
            );
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into upload
    /// staging memory and records the GPU copy into this default-heap buffer.
    fn stage(
        self: &Arc<Self>,
        from: usize,
        to: usize,
        alloc: &mut impl FnMut(usize) -> Option<(ID3D12Resource, u64, *mut u8)>,
        out: &mut Vec<UploadCopy>,
    ) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        let len = to.saturating_sub(from);
        if len == 0 {
            return 0;
        }
        let Some((src, src_offset, dst_ptr)) = alloc(len) else {
            return 0;
        };
        crate::host::copy_mapped(dst_ptr, from as *const u8, len, false);
        out.push(UploadCopy {
            shadow: self.clone(),
            src,
            src_offset,
            dst_offset: (from - start) as u64,
            len: len as u64,
        });
        len
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // GPU-local default heap so shaders read/write VRAM at full bandwidth.
        let buffer = host()?.default_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory. Such
/// memory gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let buffer = host()?.default_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D12::{D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    struct WatchedAlloc {
        ptr: *mut u8,
        len: usize,
        resource_id: usize,
    }

    impl WatchedAlloc {
        fn new(len: usize, resource_id: usize) -> Self {
            // SAFETY: fresh write-watched virtual allocation, freed in Drop.
            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            }
            .cast::<u8>();
            assert!(!ptr.is_null());
            register(resource_id, ptr as u64, len as u64);
            Self {
                ptr,
                len,
                resource_id,
            }
        }

        fn addr(&self) -> u64 {
            self.ptr as u64
        }

        fn bytes_mut(&mut self) -> &mut [u8] {
            // SAFETY: exclusive access to our committed allocation.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    impl Drop for WatchedAlloc {
        fn drop(&mut self) {
            unregister(self.resource_id, self.addr());
            // SAFETY: frees the allocation obtained from VirtualAlloc.
            unsafe { VirtualFree(self.ptr.cast(), 0, MEM_RELEASE) };
        }
    }

    #[test]
    fn merge_ranges_coalesces_overlapping_and_adjacent() {
        let mut empty = Vec::new();
        merge_ranges(&mut empty);
        assert!(empty.is_empty());

        let mut ranges = vec![
            (64, 64),
            (200, 100),
            (128, 256),
            (0, 64),
            (64, 96),
            (96, 128),
            (512, 600),
            (550, 700),
        ];
        merge_ranges(&mut ranges);
        assert_eq!(ranges, vec![(0, 256), (512, 700)]);
    }

    #[test]
    fn default_vram_shadow_uploads_only_dirty_pages() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let mut mem = WatchedAlloc::new(2 * PAGE, 0x5000_0001);
        // Touch every 4 KiB page of the allocation before initial submission so
        // Windows demand-zero page faults don't fire during the first read.
        mem.bytes_mut().fill(0);
        mem.bytes_mut()[0..4].copy_from_slice(&0x1122_3344u32.to_le_bytes());
        mem.bytes_mut()[PAGE..PAGE + 4].copy_from_slice(&0x5566_7788u32.to_le_bytes());

        let shadow = find(mem.addr(), (2 * PAGE) as u64).expect("default shadow");

        // Verify the shadow buffer lives in GPU-local DEFAULT VRAM.
        let mut heap = D3D12_HEAP_PROPERTIES::default();
        // SAFETY: live D3D12 resource.
        unsafe { shadow.buffer.GetHeapProperties(Some(&mut heap), None) }.expect("heap props");
        assert_eq!(heap.Type, D3D12_HEAP_TYPE_DEFAULT);

        let (rb, rb_ptr) = host
            .readback_buffer((2 * PAGE) as u64)
            .expect("readback buffer");

        // Submission 1: initial upload copies both pages into DEFAULT VRAM.
        let list = host.new_list().expect("new list");
        shadow.refresh(&list, mem.addr(), (2 * PAGE) as u64);
        // SAFETY: list is open; buffers stay alive through execute.
        unsafe {
            list.list
                .CopyBufferRegion(&rb, 0, &shadow.buffer, 0, (2 * PAGE) as u64);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: plain device-removed check.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        // SAFETY: fence completed; readback buffer is mapped for 2 * PAGE bytes.
        let got = unsafe { std::slice::from_raw_parts(rb_ptr, 2 * PAGE) };
        assert_eq!(&got[0..4], &0x1122_3344u32.to_le_bytes());
        assert_eq!(&got[PAGE..PAGE + 4], &0x5566_7788u32.to_le_bytes());

        // Unchanged check: zero dirty pages -> zero bytes staged.
        assert_eq!(
            shadow.watched.load(Ordering::Relaxed),
            1,
            "write-watched allocation should be detected as watched"
        );
        let mut uploads = Vec::new();
        let staged = shadow.update(
            mem.addr() as usize,
            2 * PAGE,
            &mut |n| {
                let (b, p) = host.upload_buffer(n as u64)?;
                Some((b, 0, p))
            },
            &mut uploads,
        );
        assert_eq!(
            staged,
            0,
            "copied={:?}",
            shadow.copied.lock().unwrap_or_else(|e| e.into_inner())
        );
        assert!(uploads.is_empty());

        // Dirty only page 1: only page 1 is staged at dst_offset == PAGE.
        mem.bytes_mut()[PAGE..PAGE + 4].copy_from_slice(&0x99AA_BBCCu32.to_le_bytes());
        let mut uploads = Vec::new();
        let staged = shadow.update(
            mem.addr() as usize,
            2 * PAGE,
            &mut |n| {
                let (b, p) = host.upload_buffer(n as u64)?;
                Some((b, 0, p))
            },
            &mut uploads,
        );
        assert_eq!(staged, PAGE);
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].dst_offset, PAGE as u64);
        assert_eq!(uploads[0].len, PAGE as u64);

        // Submission 2: write page 1 again and verify GPU sees new page 1 + original page 0.
        mem.bytes_mut()[PAGE..PAGE + 4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        list.reset();
        shadow.refresh(&list, mem.addr(), (2 * PAGE) as u64);
        // SAFETY: list is open; buffers stay alive through execute.
        unsafe {
            list.list
                .CopyBufferRegion(&rb, 0, &shadow.buffer, 0, (2 * PAGE) as u64);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: plain device-removed check.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        // SAFETY: fence completed; readback buffer is mapped for 2 * PAGE bytes.
        let got = unsafe { std::slice::from_raw_parts(rb_ptr, 2 * PAGE) };
        assert_eq!(&got[0..4], &0x1122_3344u32.to_le_bytes());
        assert_eq!(&got[PAGE..PAGE + 4], &0xDEAD_BEEFu32.to_le_bytes());
    }

    #[test]
    fn gpu_writes_readback_merges_and_preserves_concurrent_cpu_writes() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let mut mem = WatchedAlloc::new(PAGE, 0x5000_0002);
        mem.bytes_mut().fill(0);
        // Indirect draw record at offset 0: vertex count = u32::MAX (clamped to 1<<24 by GPU),
        // instances = 1, start_vertex = 0, start_instance = 0; sentinel at offset 16.
        let initial: [u32; 5] = [u32::MAX, 1, 0, 0, 0x1234_5678];
        for (i, word) in initial.iter().enumerate() {
            mem.bytes_mut()[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
        }

        let shadow = find(mem.addr(), PAGE as u64).expect("default shadow");
        let list = host.new_list().expect("new list");
        shadow.refresh(&list, mem.addr(), PAGE as u64);
        shadow.gpu_writes(&list, mem.addr(), 64);
        assert!(crate::indirect::bound(
            &list,
            &shadow.buffer,
            shadow.gpu(mem.addr()),
            (0, 16, 1),
        ));
        crate::indirect::done(&list, &shadow.buffer);
        list.close();
        host.execute(&[&list]);
        // SAFETY: plain device-removed check.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        // GPU clamped word 0 to 1<<24 and readback merged it into title memory;
        // word 4 (not written by GPU) stayed untouched.
        let clamped = u32::from_le_bytes(mem.bytes_mut()[0..4].try_into().unwrap());
        let sentinel = u32::from_le_bytes(mem.bytes_mut()[16..20].try_into().unwrap());
        assert_eq!(clamped, 1 << 24);
        assert_eq!(sentinel, 0x1234_5678);

        // Flush write-watch state so the page starts clean.
        let mut uploads = Vec::new();
        shadow.update(
            mem.addr() as usize,
            PAGE,
            &mut |n| {
                let (b, p) = host.upload_buffer(n as u64)?;
                Some((b, 0, p))
            },
            &mut uploads,
        );
        uploads.clear();
        assert_eq!(
            shadow.update(
                mem.addr() as usize,
                PAGE,
                &mut |n| {
                    let (b, p) = host.upload_buffer(n as u64)?;
                    Some((b, 0, p))
                },
                &mut uploads,
            ),
            0
        );

        // Case 1: GPU-only write back on a whole page marks the page synced so our
        // own write-back is not re-uploaded on the next submission.
        let old = mem.bytes_mut().to_vec();
        let mut after = old.clone();
        after[0..4].copy_from_slice(&0x1357_2468u32.to_le_bytes());
        assert_eq!(shadow.copy_back_range(0, &old, &after), PAGE);
        assert_eq!(
            u32::from_le_bytes(mem.bytes_mut()[0..4].try_into().unwrap()),
            0x1357_2468
        );
        uploads.clear();
        assert_eq!(
            shadow.update(
                mem.addr() as usize,
                PAGE,
                &mut |n| {
                    let (b, p) = host.upload_buffer(n as u64)?;
                    Some((b, 0, p))
                },
                &mut uploads,
            ),
            0
        );

        // Case 2: GPU write on [0..4) plus concurrent CPU write on [32..36) of the
        // same page merges GPU bytes, preserves CPU bytes, and leaves the page dirty
        // so the next upload pushes the merged page back to DEFAULT VRAM.
        let old = mem.bytes_mut().to_vec();
        let mut after = old.clone();
        after[0..4].copy_from_slice(&0xCAFE_BABEu32.to_le_bytes());
        mem.bytes_mut()[32..36].copy_from_slice(&0xFACE_FEEDu32.to_le_bytes());

        assert_eq!(shadow.copy_back_range(0, &old, &after), PAGE);
        assert_eq!(
            u32::from_le_bytes(mem.bytes_mut()[0..4].try_into().unwrap()),
            0xCAFE_BABE
        );
        assert_eq!(
            u32::from_le_bytes(mem.bytes_mut()[32..36].try_into().unwrap()),
            0xFACE_FEED
        );

        uploads.clear();
        let staged = shadow.update(
            mem.addr() as usize,
            PAGE,
            &mut |n| {
                let (b, p) = host.upload_buffer(n as u64)?;
                Some((b, 0, p))
            },
            &mut uploads,
        );
        assert_eq!(staged, PAGE);
        assert_eq!(uploads.len(), 1);
    }
}
