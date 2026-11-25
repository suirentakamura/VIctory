//! A clock for timing hot paths.
//!
//! `Instant::now` asks `QueryPerformanceCounter`, which Wine (11.x) answers
//! with a system call every time: `RtlQueryPerformanceCounter` has no
//! time-stamp-counter bypass, so each reading saves and restores the whole
//! register state (`xsavec`), a few hundred nanoseconds. The submission
//! path times each copy check (hundreds of thousands per world frame), and
//! those readings took more of a frame than the checks. Where the
//! processor's time-stamp counter runs at a constant rate (invariant TSC),
//! it is read instead: one instruction, scaled by a rate measured once
//! against `Instant`.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// A moment on this clock.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Stamp(u64);

/// Nanoseconds per counter tick, or `None` where the counter is not
/// invariant (ticks are then nanoseconds since [`origin`]).
fn rate() -> Option<f64> {
    static RATE: OnceLock<Option<f64>> = OnceLock::new();
    *RATE.get_or_init(|| {
        if !invariant() {
            return None;
        }
        // About 2 ms against `Instant` (Wine's runs at 10 MHz): within a
        // few parts in 10^5, plenty for statistics and budgets.
        let (t0, c0) = (Instant::now(), tsc());
        let mut t1 = Instant::now();
        while t1 - t0 < Duration::from_millis(2) {
            std::hint::spin_loop();
            t1 = Instant::now();
        }
        let c1 = tsc();
        let ticks = c1.wrapping_sub(c0);
        (ticks > 0).then(|| (t1 - t0).as_nanos() as f64 / ticks as f64)
    })
}

/// Where tick counts start when the counter is not used.
fn origin() -> Instant {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    *ORIGIN.get_or_init(Instant::now)
}

/// Whether the processor says its time-stamp counter runs at a constant
/// rate in every power state (CPUID 8000_0007h, EDX bit 8).
#[cfg(target_arch = "x86_64")]
fn invariant() -> bool {
    use std::arch::x86_64::__cpuid;
    let top = __cpuid(0x8000_0000).eax;
    top >= 0x8000_0007 && __cpuid(0x8000_0007).edx & (1 << 8) != 0
}

#[cfg(not(target_arch = "x86_64"))]
fn invariant() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn tsc() -> u64 {
    // SAFETY: RDTSC is present on every x86-64 processor.
    unsafe { std::arch::x86_64::_rdtsc() }
}

#[cfg(not(target_arch = "x86_64"))]
fn tsc() -> u64 {
    0
}

/// Now.
pub(crate) fn now() -> Stamp {
    match rate() {
        Some(_) => Stamp(tsc()),
        None => Stamp(origin().elapsed().as_nanos() as u64),
    }
}

impl Stamp {
    /// Nanoseconds since this moment.
    pub(crate) fn nanos(self) -> u64 {
        let now = now().0;
        let ticks = now.saturating_sub(self.0);
        match rate() {
            Some(rate) => (ticks as f64 * rate) as u64,
            None => ticks,
        }
    }

    /// Microseconds since this moment.
    pub(crate) fn micros(self) -> u64 {
        self.nanos() / 1000
    }

    /// Time since this moment.
    pub(crate) fn elapsed(self) -> Duration {
        Duration::from_nanos(self.nanos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clock_agrees_with_instant() {
        let (stamp, instant) = (now(), Instant::now());
        std::thread::sleep(Duration::from_millis(30));
        let (ours, theirs) = (stamp.nanos() as f64, instant.elapsed().as_nanos() as f64);
        assert!(
            (ours - theirs).abs() < theirs * 0.02 + 1e6,
            "{ours} ns against {theirs} ns"
        );
    }
}
