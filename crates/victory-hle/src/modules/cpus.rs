//! Where the title's threads run.
//!
//! A console title places its threads on the console's processors with
//! fixed masks: seven cores for the title, each with two logical processors
//! (`2k` and `2k + 1`), in two clusters (cores 0-3 and 4-6). Passed through,
//! those masks kept the title on the host's first 14 logical processors
//! whatever the host has: on a 12-core host with SMT, 10 of 24 never ran
//! title code, and a pin meant for one core of its own landed on the second
//! thread of a core the others already used. Each console core is given its
//! share of the host's cores instead, with all their logical processors: a
//! cluster becomes about half the host, the whole console all of it.

#![allow(non_snake_case)]

use crate::HleModule;
use crate::hle_export;
use std::ffi::c_void;

/// Cores the console gives a title (logical processors `0..14`).
const CONSOLE_CORES: usize = 7;

/// `GROUP_AFFINITY`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GroupAffinity {
    mask: usize,
    group: u16,
    reserved: [u16; 3],
}

/// The host mask for console mask `console`, given the host's cores (each
/// the mask of its logical processors, in order); `None` if `console` names
/// no console processor.
fn host_mask(console: usize, cores: &[usize]) -> Option<usize> {
    let n = cores.len();
    if n == 0 || console & ((1 << (2 * CONSOLE_CORES)) - 1) == 0 {
        return None;
    }
    let mut out = 0;
    for c in 0..CONSOLE_CORES {
        if console >> (2 * c) & 0b11 == 0 {
            continue;
        }
        // At least one host core each, on hosts with fewer than seven.
        let first = (c * n / CONSOLE_CORES).min(n - 1);
        let end = ((c + 1) * n / CONSOLE_CORES).clamp(first + 1, n);
        out |= cores[first..end].iter().fold(0, |m, &core| m | core);
    }
    Some(out)
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "cpus",
        dlls: &[
            "kernel32.dll",
            "kernelbase.dll",
            "api-ms-win-core-processtopology-*",
        ],
        exports: vec![
            hle_export!(SetThreadGroupAffinity),
            hle_export!(GetThreadGroupAffinity),
        ],
    }
}

#[cfg(windows)]
mod host {
    use super::GroupAffinity;
    use std::ffi::c_void;
    use std::sync::{Mutex, OnceLock};
    use windows_sys::Win32::System::SystemInformation::{
        GROUP_AFFINITY, GetLogicalProcessorInformation, RelationProcessorCore,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::{
        GetThreadGroupAffinity, GetThreadId, SetThreadGroupAffinity,
    };

    /// The host's cores (group 0), each the mask of its logical processors,
    /// by their first logical processor.
    pub(super) fn cores() -> &'static [usize] {
        static CORES: OnceLock<Vec<usize>> = OnceLock::new();
        CORES.get_or_init(|| {
            let mut len = 0u32;
            // SAFETY: a size query first, then a buffer of that size.
            unsafe { GetLogicalProcessorInformation(std::ptr::null_mut(), &mut len) };
            let n = len as usize / size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION>();
            // SAFETY: plain data.
            let mut info: Vec<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> =
                vec![unsafe { std::mem::zeroed() }; n];
            // SAFETY: as above.
            if n == 0 || unsafe { GetLogicalProcessorInformation(info.as_mut_ptr(), &mut len) } == 0
            {
                return Vec::new();
            }
            let mut cores: Vec<usize> = info
                .iter()
                .filter(|i| i.Relationship == RelationProcessorCore && i.ProcessorMask != 0)
                .map(|i| i.ProcessorMask)
                .collect();
            cores.sort_unstable_by_key(|m| m.trailing_zeros());
            tracing::info!(
                cores = cores.len(),
                "host cores for the title's processor masks"
            );
            cores
        })
    }

    /// The console mask the title last set on each thread (by id): what it
    /// reads back.
    static SET: Mutex<Vec<(u32, usize)>> = Mutex::new(Vec::new());

    pub(super) fn set(
        thread: *mut c_void,
        affinity: &GroupAffinity,
        previous: *mut GroupAffinity,
    ) -> i32 {
        let console = affinity.mask;
        let mut mine = *affinity;
        if let Some(mask) = super::host_mask(console, cores()) {
            mine.mask = mask;
        }
        tracing::debug!(
            console = format_args!("{console:#x}"),
            host = format_args!("{:#x}", mine.mask),
            "title thread affinity"
        );
        // SAFETY: the title's thread handle; our copy of its description in
        // the host's layout; its out pointer (null is allowed).
        let ok = unsafe {
            SetThreadGroupAffinity(
                thread,
                (&raw const mine).cast::<GROUP_AFFINITY>(),
                previous.cast::<GROUP_AFFINITY>(),
            )
        };
        if ok != 0 {
            // SAFETY: the title's thread handle.
            let id = unsafe { GetThreadId(thread) };
            let mut set = SET.lock().unwrap_or_else(|e| e.into_inner());
            set.retain(|&(t, _)| t != id);
            set.push((id, console));
        }
        ok
    }

    pub(super) fn get(thread: *mut c_void, out: *mut GroupAffinity) -> i32 {
        // SAFETY: the title's thread handle and out pointer.
        let ok = unsafe { GetThreadGroupAffinity(thread, out.cast::<GROUP_AFFINITY>()) };
        if ok != 0 && !out.is_null() {
            // SAFETY: as above.
            let id = unsafe { GetThreadId(thread) };
            if let Some(&(_, console)) = SET
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .find(|&&(t, _)| t == id)
            {
                // SAFETY: filled in by the host just now.
                unsafe { (*out).mask = console };
            }
        }
        ok
    }
}

/// `SetThreadGroupAffinity`: the console mask, on the host's cores.
#[cfg(windows)]
pub extern "system" fn SetThreadGroupAffinity(
    thread: *mut c_void,
    affinity: *const GroupAffinity,
    previous: *mut GroupAffinity,
) -> i32 {
    // SAFETY: the title passes a description (null is refused).
    let Some(affinity) = (unsafe { affinity.as_ref() }) else {
        return 0;
    };
    host::set(thread, affinity, previous)
}

/// `GetThreadGroupAffinity`: the console mask the title set.
#[cfg(windows)]
pub extern "system" fn GetThreadGroupAffinity(thread: *mut c_void, out: *mut GroupAffinity) -> i32 {
    host::get(thread, out)
}

#[cfg(not(windows))]
pub extern "system" fn SetThreadGroupAffinity(
    _: *mut c_void,
    _: *const GroupAffinity,
    _: *mut GroupAffinity,
) -> i32 {
    0
}

#[cfg(not(windows))]
pub extern "system" fn GetThreadGroupAffinity(_: *mut c_void, _: *mut GroupAffinity) -> i32 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 12-core host with SMT, numbered as Linux does (core k: k, k + 12).
    fn twelve() -> Vec<usize> {
        (0..12).map(|k| 1 << k | 1 << (k + 12)).collect()
    }

    #[test]
    fn console_clusters_become_host_halves() {
        let cores = twelve();
        let first_six = (0..6).fold(0, |m, k| m | cores[k]);
        let last_six = (6..12).fold(0, |m, k| m | cores[k]);
        assert_eq!(host_mask(0xff, &cores), Some(first_six));
        assert_eq!(host_mask(0x1f00, &cores), Some(last_six));
        assert_eq!(host_mask(0x3fff, &cores), Some(first_six | last_six));
    }

    #[test]
    fn one_console_processor_gets_its_cores_share() {
        let cores = twelve();
        assert_eq!(host_mask(1 << 13, &cores), Some(cores[10] | cores[11]));
        assert_eq!(host_mask(1 << 0, &cores), Some(cores[0]));
    }

    #[test]
    fn small_hosts_give_every_console_core_one() {
        let cores: Vec<usize> = (0..4).map(|k| 1 << k).collect();
        for c in 0..CONSOLE_CORES {
            assert_ne!(host_mask(1 << (2 * c), &cores), Some(0));
        }
    }

    #[test]
    fn masks_beyond_the_console_are_left_alone() {
        assert_eq!(host_mask(1 << 20, &twelve()), None);
    }
}
