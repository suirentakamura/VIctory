//! A few worker threads for the memory work around a submission (copying
//! back what shaders wrote, comparing pages): the title's own threads are
//! mostly idle while the GPU runs, and one thread comparing and copying
//! 64 KiB pages moves about 2 GB/s, a fraction of what the memory gives.
//!
//! The workers are made once, on first use: a thread costs a wineserver
//! round trip under Wine. `VICTORY_COPY_THREADS=<n>` sets how many threads
//! share the work, the caller included (default: a third of the logical
//! processors, at most 8); `0` or `1` runs everything on the caller, as
//! before.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

/// The work handed out: a call per index below `n`.
#[derive(Clone, Copy)]
struct Job {
    /// The caller's closure, alive until every worker is done with it
    /// ([`for_each`] waits for that before it returns).
    f: *const (dyn Fn(usize) + Sync),
    n: usize,
}

// SAFETY: the closure is `Sync`, and outlives every use (see `Job::f`).
unsafe impl Send for Job {}

#[derive(Default)]
struct State {
    /// Bumped for each job, so that a worker takes each job once.
    round: u64,
    job: Option<Job>,
    /// Workers still on the current job.
    busy: usize,
    /// A worker's call panicked in the current job.
    panicked: bool,
}

struct Pool {
    state: Mutex<State>,
    /// Signalled when a job is posted.
    posted: Condvar,
    /// Signalled when a worker is done with a job.
    finished: Condvar,
    /// The next index to take in the current job.
    next: AtomicUsize,
    /// Workers made (each takes every job).
    workers: AtomicUsize,
    /// Held by the caller of a job: one job at a time (a second caller runs
    /// its work itself).
    running: Mutex<()>,
}

/// Threads that share the work, the caller included.
fn threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
        std::env::var("VICTORY_COPY_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or((logical / 3).clamp(1, 8))
            .clamp(1, 64)
    })
}

fn pool() -> Option<&'static Pool> {
    static POOL: OnceLock<Option<&'static Pool>> = OnceLock::new();
    *POOL.get_or_init(|| {
        let wanted = threads() - 1;
        if wanted == 0 {
            return None;
        }
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            state: Mutex::default(),
            posted: Condvar::new(),
            finished: Condvar::new(),
            next: AtomicUsize::new(0),
            workers: AtomicUsize::new(0),
            running: Mutex::new(()),
        }));
        for k in 0..wanted {
            let spawned = std::thread::Builder::new()
                .name(format!("victory copy worker {k}"))
                .spawn(move || work(pool));
            if spawned.is_ok() {
                pool.workers.fetch_add(1, Ordering::Relaxed);
            }
        }
        // With no worker made, the caller does all the work.
        (pool.workers.load(Ordering::Relaxed) > 0).then_some(pool)
    })
}

/// A worker: takes each job posted and calls its closure for the indices
/// it claims.
fn work(pool: &'static Pool) {
    let mut seen = 0u64;
    loop {
        let job = {
            let mut s = pool.state.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if s.round != seen
                    && let Some(job) = s.job
                {
                    seen = s.round;
                    break job;
                }
                s = pool.posted.wait(s).unwrap_or_else(|e| e.into_inner());
            }
        };
        let ok =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| claim(pool, job))).is_ok();
        let mut s = pool.state.lock().unwrap_or_else(|e| e.into_inner());
        s.panicked |= !ok;
        s.busy -= 1;
        if s.busy == 0 {
            pool.finished.notify_all();
        }
    }
}

/// Calls the job's closure for indices claimed one at a time until none
/// is left.
fn claim(pool: &Pool, job: Job) {
    // SAFETY: the caller of `for_each` keeps the closure alive until every
    // worker has left this function (it waits for `busy` to reach zero).
    let f = unsafe { &*job.f };
    loop {
        let i = pool.next.fetch_add(1, Ordering::Relaxed);
        if i >= job.n {
            return;
        }
        f(i);
    }
}

/// Calls `f(i)` for every `i` below `n`, on the workers and the calling
/// thread, and returns when all calls have returned. The order of the calls
/// is unspecified; `f` must not call `for_each` itself (it would run its
/// work on the one thread, which is correct but serial).
pub(crate) fn for_each(n: usize, f: &(dyn Fn(usize) + Sync)) {
    let serial = || (0..n).for_each(f);
    // A handful of calls is not worth waking the workers for.
    if n < 4 {
        return serial();
    }
    let Some(pool) = pool() else {
        return serial();
    };
    // Another job is running (another thread's): this one runs here.
    let Ok(_running) = pool.running.try_lock() else {
        return serial();
    };
    pool.next.store(0, Ordering::Relaxed);
    // SAFETY: only the lifetime is erased; the closure outlives the job,
    // since this function waits below until no worker is in it.
    let f_static: *const (dyn Fn(usize) + Sync) = unsafe {
        std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f)
    };
    {
        let mut s = pool.state.lock().unwrap_or_else(|e| e.into_inner());
        s.round += 1;
        s.job = Some(Job { f: f_static, n });
        s.busy = pool.workers.load(Ordering::Relaxed);
        s.panicked = false;
        pool.posted.notify_all();
    }
    let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        claim(pool, Job { f: f_static, n });
    }));
    let panicked = {
        let mut s = pool.state.lock().unwrap_or_else(|e| e.into_inner());
        while s.busy > 0 {
            s = pool.finished.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        s.job = None;
        s.panicked
    };
    if let Err(e) = ok {
        std::panic::resume_unwind(e);
    }
    assert!(!panicked, "a copy worker's call panicked");
}

#[cfg(test)]
mod tests {
    use super::for_each;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn every_index_is_called_once() {
        for n in [0usize, 1, 2, 7, 1000] {
            let hits: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
            for_each(n, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                hits.iter().all(|h| h.load(Ordering::Relaxed) == 1),
                "n = {n}"
            );
        }
    }

    #[test]
    fn jobs_from_several_threads_all_finish() {
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    for _ in 0..50 {
                        let sum = AtomicU64::new(0);
                        for_each(100, &|i| {
                            sum.fetch_add(i as u64, Ordering::Relaxed);
                        });
                        assert_eq!(sum.load(Ordering::Relaxed), 4950);
                    }
                });
            }
        });
    }
}
