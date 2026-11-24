//! Title memory the host GPU uses in place (unified memory). On the console
//! CPU and GPU share one memory; here a run of the title's committed private
//! memory is opened as a host heap (`OpenExistingHeapFromAddress1`) and a
//! buffer is placed over all of it, so draws read and shaders write the
//! title's own bytes: no upload before a submission, no read-back after it,
//! no write watching ([`crate::shadow`] keeps that model for what cannot be
//! opened).
//!
//! What can be opened, measured under Wine with vkd3d-proton 3.2 on RADV
//! (RX 9060 XT, 2026-11-24; the probe is in the decision record):
//!
//! - committed private memory (`VirtualAlloc`, write-watched or not), any
//!   4 KiB-aligned part of an allocation, up to 4095 MiB per heap; opening
//!   256 MiB takes under 1 ms, 4095 MiB about 50 ms. Both directions are
//!   coherent: CPU stores reach the next submission, shader stores are in
//!   title memory when the fence signals. The card reads it at about
//!   28 GB/s (PCIe), not at its own memory's speed.
//! - **not** views of sections (the physical-page pool): vkd3d-proton cannot
//!   import them, says so only in its log, and gives an ordinary allocation
//!   instead, which never sees the title's bytes. Only `MEM_PRIVATE` memory
//!   is opened.
//! - **Freeing memory under an open heap loses the device** at the next
//!   submission (the kernel driver cannot pin its pages any more). The
//!   title's `VirtualFree` of such memory is therefore deferred
//!   ([`title_free`]) until every list that used the heap has been reset and
//!   the heap released ([`drain`]); until then the memory stays committed.
//!
//! The same design (title graphics memory as host heaps, the console's GPU
//! addresses being CPU addresses) is what XCL describes for its unified
//! memory; this is our own implementation of it for VIctory's objects.
//!
//! Off unless `VICTORY_UMA=1`: on a card with its own memory it saves little
//! (play47: the costly copies are of big buffers, which must stay in card
//! memory, and of buffers shaders write, which must stay copies).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ID3D12Device13, ID3D12Heap, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_DECOMMIT, MEM_PRIVATE, MEM_RELEASE, MEMORY_BASIC_INFORMATION, VirtualFree,
    VirtualQuery,
};

use crate::host::host;

/// Largest heap the host opens (exactly 4 GiB is refused).
const MAX_HEAP: usize = (4096 << 20) - (64 << 10);
/// Host page.
const PAGE: usize = 4 << 10;

/// Whether title memory is used in place (`VICTORY_UMA`, on unless `0`).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_UMA").is_ok_and(|v| v.trim() == "1");
        tracing::info!(on, "unified memory: title buffers in private memory used in place");
        on
    })
}

/// Largest title buffer used in place (`VICTORY_UMA_MAX_KB`, 4 MiB): the
/// card reads title memory over PCIe at about 28 GB/s, an order of
/// magnitude under its own memory, so big buffers the GPU reads many times
/// a frame (geometry, meshlets) stay copies in card memory; what is used in
/// place is small data the CPU rewrites often (constants, per-draw data).
pub(crate) fn max_buffer() -> u64 {
    static MAX: OnceLock<u64> = OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("VICTORY_UMA_MAX_KB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(4096) << 10
    })
}

/// Whether memory outside buffer objects is used in place
/// (`VICTORY_UMA_LOOSE=1`; off: such reads are often large).
pub(crate) fn loose() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_UMA_LOOSE").is_ok_and(|v| v.trim() == "1"))
}

/// A run of committed title memory opened as a host heap, with a buffer
/// over all of it.
pub(crate) struct Region {
    /// Title address of the first byte, and the length.
    pub(crate) start: usize,
    pub(crate) len: usize,
    /// The buffer (dropped before the heap).
    buffer: Option<ID3D12Resource>,
    heap: Option<ID3D12Heap>,
    /// Its host GPU address.
    pub(crate) gpu: u64,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Region {}
// SAFETY: as above.
unsafe impl Sync for Region {}

impl Region {
    pub(crate) fn buffer(&self) -> &ID3D12Resource {
        self.buffer.as_ref().expect("a live region's buffer")
    }

    pub(crate) fn end(&self) -> usize {
        self.start + self.len
    }

    /// Public references to the buffer besides ours.
    fn others(&self) -> u32 {
        let Some(b) = &self.buffer else { return 0 };
        // SAFETY: AddRef and Release on a live object, in a pair: what
        // Release returns is the count without our extra one.
        unsafe {
            let raw = b.as_raw();
            (b.vtable().base__.base__.base__.base__.AddRef)(raw);
            (b.vtable().base__.base__.base__.base__.Release)(raw).saturating_sub(1)
        }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        self.buffer = None;
        self.heap = None;
    }
}

/// Open regions by start address.
static REGIONS: Mutex<BTreeMap<usize, Arc<Region>>> = Mutex::new(BTreeMap::new());

/// Title frees waiting for their regions to be released: the regions, and
/// the free to do afterwards (address, size, kind).
struct Grave {
    regions: Vec<Arc<Region>>,
    free: (usize, usize, u32),
}

static GRAVES: Mutex<Vec<Grave>> = Mutex::new(Vec::new());

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn query(at: usize) -> Option<MEMORY_BASIC_INFORMATION> {
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: a query of our own process' address space.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) };
    (n != 0).then_some(info)
}

/// The committed private run around `[from, to)`: every page of it
/// committed, in one allocation. `None` if a page of the range is not.
fn committed_run(from: usize, to: usize) -> Option<(usize, usize)> {
    let first = query(from)?;
    let ok = |i: &MEMORY_BASIC_INFORMATION| i.State == MEM_COMMIT && i.Type == MEM_PRIVATE;
    if !ok(&first) {
        return None;
    }
    let base = first.AllocationBase as usize;
    // Forward from the allocation's start: the committed run holding `from`.
    let (mut at, mut run) = (base, None::<usize>);
    loop {
        let info = query(at)?;
        if info.AllocationBase as usize != base || info.RegionSize == 0 {
            break;
        }
        let end = at + info.RegionSize;
        if ok(&info) {
            run.get_or_insert(at);
        } else if end <= from {
            run = None;
        } else {
            // A gap after `from`: the run ends here.
            break;
        }
        at = end;
    }
    let (start, end) = (run?, at);
    (start <= from && to <= end).then_some((start, end))
}

/// The region holding `[address, address + len)`, opened on first use.
/// `None` when it is off or the memory cannot be opened (not committed
/// private memory, or the host refuses).
pub(crate) fn region(address: u64, len: u64) -> Option<Arc<Region>> {
    if !enabled() {
        return None;
    }
    let (from, to) = (address as usize, address.checked_add(len.max(1))? as usize);
    {
        let regions = lock(&REGIONS);
        if let Some(r) = regions
            .range(..=from)
            .rev()
            .take(16)
            .map(|(_, r)| r)
            .find(|r| r.end() >= to)
        {
            return Some(r.clone());
        }
    }
    let (run_start, run_end) = committed_run(from & !(PAGE - 1), to.next_multiple_of(PAGE))?;
    // Only the range's own 64 KiB pages, not the run around it: the host
    // pins an opened heap's pages, and a write-watch reset anywhere in it
    // (the copy path's, for other buffers and textures nearby) makes the
    // next submission pin the whole heap again; with the title's 1-2 GiB
    // runs opened whole that cost seconds per frame (play46).
    const GRAIN: usize = 64 << 10;
    let start = (from & !(GRAIN - 1)).max(run_start);
    let end = to.next_multiple_of(GRAIN).min(run_end);
    if end - start > MAX_HEAP || to > end {
        return None;
    }
    let region = open(start, end - start)?;
    let region = Arc::new(region);
    lock(&REGIONS).insert(start, region.clone());
    Some(region)
}

fn open(start: usize, len: usize) -> Option<Region> {
    let host = host()?;
    let device: ID3D12Device13 = host.device.cast().ok()?;
    let started = std::time::Instant::now();
    // SAFETY: committed private memory of the title (checked page by page
    // above); it stays committed while the heap is open (`title_free`).
    let heap: ID3D12Heap = match unsafe { device.OpenExistingHeapFromAddress1(start as *const _, len) } {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!(start = format_args!("{start:#x}"), len, %e, "unified memory: the host did not open title memory");
            return None;
        }
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: len as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: a buffer over the whole heap.
    if let Err(e) = unsafe {
        host.device
            .CreatePlacedResource(&heap, 0, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
    } {
        tracing::warn!(start = format_args!("{start:#x}"), len, %e, "unified memory: no buffer in the opened heap");
        return None;
    }
    let buffer = buffer?;
    crate::host::name(&buffer, || format!("title memory in place {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        kb = len >> 10,
        ms = format_args!("{:.1}", started.elapsed().as_secs_f64() * 1e3),
        "unified memory: title memory opened in place"
    );
    Some(Region { start, len, buffer: Some(buffer), heap: Some(heap), gpu })
}

/// The title's `VirtualFree`: when it would free or decommit memory under
/// an open region, the regions are closed to new users and the free waits
/// for their release ([`drain`]); returns whether it was taken over (the
/// title is told it succeeded).
pub fn title_free(address: usize, size: usize, kind: u32) -> bool {
    if !enabled() || address == 0 {
        return false;
    }
    let (from, to) = if kind & MEM_RELEASE != 0 {
        // The whole allocation (size is 0).
        let Some(info) = query(address) else { return false };
        let base = info.AllocationBase as usize;
        let mut at = base;
        while let Some(i) = query(at).filter(|i| i.AllocationBase as usize == base && i.RegionSize > 0) {
            at += i.RegionSize;
        }
        (base, at)
    } else if kind & MEM_DECOMMIT != 0 {
        (address & !(PAGE - 1), (address + size).next_multiple_of(PAGE))
    } else {
        return false;
    };
    let closed: Vec<Arc<Region>> = {
        let mut regions = lock(&REGIONS);
        let keys: Vec<usize> = regions
            .iter()
            .filter(|(_, r)| r.start < to && r.end() > from)
            .map(|(k, _)| *k)
            .collect();
        keys.into_iter().filter_map(|k| regions.remove(&k)).collect()
    };
    if closed.is_empty() {
        return false;
    }
    crate::shadow::forget_regions(&closed);
    tracing::debug!(
        address = format_args!("{address:#x}"),
        size,
        kind = format_args!("{kind:#x}"),
        regions = closed.len(),
        "unified memory: a title free waits for its regions"
    );
    lock(&GRAVES).push(Grave { regions: closed, free: (address, size, kind) });
    drain();
    true
}

/// The title commits `[address, address + size)` again: a decommit of it
/// still waiting is dropped (the pages stay committed) and the range is
/// zeroed, as fresh pages are.
pub fn title_commit(address: usize, size: usize) {
    if address == 0 || size == 0 {
        return;
    }
    let mut graves = lock(&GRAVES);
    let mut zero = false;
    for g in graves.iter_mut() {
        let (a, s, kind) = g.free;
        if kind & MEM_DECOMMIT != 0 && a < address + size && a + s > address {
            g.free.2 = 0;
            zero = true;
        }
    }
    drop(graves);
    if zero {
        // SAFETY: the pages are committed (their decommit was dropped) and
        // the title owns them.
        unsafe { std::ptr::write_bytes(address as *mut u8, 0, size) };
    }
}

/// Releases regions no list uses any more and does the frees that waited
/// for them. Called once per frame and on each deferred free.
pub(crate) fn drain() {
    let mut graves = lock(&GRAVES);
    if graves.is_empty() {
        return;
    }
    graves.retain_mut(|g| {
        let free = g.regions.iter().all(|r| Arc::strong_count(r) == 1 && r.others() == 0);
        if !free {
            return true;
        }
        g.regions.clear();
        let (address, size, kind) = g.free;
        if kind != 0 {
            // SAFETY: the title's own free, of memory no host object maps
            // any more.
            let ok = unsafe { VirtualFree(address as *mut _, size, kind) };
            if ok == 0 {
                tracing::warn!(address = format_args!("{address:#x}"), size, kind, "unified memory: a deferred title free failed");
            }
        }
        false
    });
}
