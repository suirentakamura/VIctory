//! Diagnostics: draws and dispatches that read what only dropped work writes.
//!
//! A draw or dispatch the host cannot run (its shader does not translate, its
//! pipeline is refused or skipped) leaves what it would have written as the
//! title last saw it. A later pass that reads that, or waits for a value in
//! it, works on stale data or never finishes, which on some hosts is a
//! device hang far from the dropped work. This module keeps what dropped work
//! would have written (its unordered-access views and render targets), forgets
//! a range again when work that runs writes it, and logs once per pair
//! a reader that runs reading one of them, with the shader that was dropped.

use std::collections::{HashSet, VecDeque};
use std::sync::Mutex;

use crate::descriptor::View;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Target {
    /// Title memory, `start..end`.
    Buffer(u64, u64),
    /// A title texture object.
    Texture(usize),
}

impl Target {
    fn overlaps(self, other: Target) -> bool {
        match (self, other) {
            (Target::Buffer(a, b), Target::Buffer(c, d)) => a < d && c < b,
            (Target::Texture(a), Target::Texture(b)) => a == b,
            _ => false,
        }
    }

    fn describe(self) -> String {
        match self {
            Target::Buffer(a, b) => format!("buffer {a:#x}+{:#x}", b - a),
            Target::Texture(t) => {
                let desc = crate::resource::host_resource(t as crate::object::This).map(|r| {
                    // SAFETY: a live resource.
                    let d = unsafe { r.GetDesc() };
                    format!(" {}x{} f{}", d.Width, d.Height, d.Format.0)
                });
                format!("texture {t:#x}{}", desc.unwrap_or_default())
            }
        }
    }
}

struct Write {
    what: Target,
    writer: String,
    frame: u64,
}

const MAX_WRITES: usize = 8192;

static WRITES: Mutex<VecDeque<Write>> = Mutex::new(VecDeque::new());
static SEEN: Mutex<Option<HashSet<(String, String)>>> = Mutex::new(None);

thread_local! {
    /// The pipeline the draw or dispatch being bound on this thread runs.
    static READER: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// Whether the detector runs (`VICTORY_PRODUCERS=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PRODUCERS").is_some_and(|v| v == "1"))
}

/// The pipeline a later [`read`] or [`live_write`] on this thread belongs to.
pub(crate) fn set_reader(name: &str) {
    if on() {
        READER.with(|r| {
            let mut r = r.borrow_mut();
            r.clear();
            r.push_str(name);
        });
    }
}

fn target_of(view: &View) -> Option<Target> {
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    if let Some((address, len)) = crate::descriptor::host::buffer_range(view)
        && len > 0
    {
        return Some(Target::Buffer(address, address + len));
    }
    match *view {
        View::Shader(r, _) | View::Unordered(r, _, _) => Some(Target::Texture(r)),
        _ => None,
    }
}

fn record(what: Target, writer: &str) {
    let frame = crate::queue::presented();
    let mut writes = WRITES.lock().unwrap_or_else(|e| e.into_inner());
    // What the frame before last dropped is history: the title has moved on.
    writes.retain(|w| w.frame + 2 > frame);
    if writes.len() >= MAX_WRITES {
        writes.pop_front();
    }
    if !writes.iter().any(|w| w.what == what && w.writer == writer) {
        writes.push_back(Write {
            what,
            writer: writer.to_owned(),
            frame,
        });
    }
}

/// Dropped work `writer` would have written through `view`.
pub(crate) fn dropped_write(writer: &str, view: &View) {
    if on()
        && let Some(what) = target_of(view)
    {
        record(what, writer);
    }
}

/// Dropped draw `writer` would have rendered to title texture `resource`.
pub(crate) fn dropped_target(writer: &str, resource: usize) {
    if on() && resource != 0 {
        record(Target::Texture(resource), writer);
    }
}

/// Work that runs writes through `view`: what it writes is produced.
pub(crate) fn live_write(view: &View) {
    if !on() {
        return;
    }
    if let Some(what) = target_of(view) {
        clear_written(&mut WRITES.lock().unwrap_or_else(|e| e.into_inner()), what);
    }
}

fn clear_written(writes: &mut VecDeque<Write>, what: Target) {
    // Process only the original entries so each fragment keeps its place in the history.
    for _ in 0..writes.len() {
        let mut write = writes.pop_front().unwrap();
        if !write.what.overlaps(what) {
            writes.push_back(write);
            continue;
        }
        let (Target::Buffer(start, end), Target::Buffer(live_start, live_end)) = (write.what, what)
        else {
            continue;
        };

        let left = (start < live_start).then_some(Target::Buffer(start, live_start));
        let right = (live_end < end).then_some(Target::Buffer(live_end, end));
        match (left, right) {
            (Some(what), Some(tail)) => {
                let tail = Write {
                    what: tail,
                    writer: write.writer.clone(),
                    frame: write.frame,
                };
                write.what = what;
                writes.push_back(write);
                writes.push_back(tail);
            }
            (Some(what), None) | (None, Some(what)) => {
                write.what = what;
                writes.push_back(write);
            }
            (None, None) => {}
        }
    }

    let excess = writes.len().saturating_sub(MAX_WRITES);
    writes.drain(..excess);
}

/// A draw that runs renders to title texture `resource`.
pub(crate) fn live_target(resource: usize) {
    if on() && resource != 0 {
        WRITES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|w| w.what != Target::Texture(resource));
    }
}

/// Work that runs reads through `view`: logs it once per (reader, dropped
/// writer) if only dropped work wrote that.
pub(crate) fn read(view: &View) {
    if !on() {
        return;
    }
    let Some(what) = target_of(view) else { return };
    let hits: Vec<(String, Target)> = WRITES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|w| w.what.overlaps(what))
        .map(|w| (w.writer.clone(), w.what))
        .collect();
    if hits.is_empty() {
        return;
    }
    let reader = READER.with(|r| r.borrow().clone());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let seen = seen.get_or_insert_with(HashSet::new);
    for (writer, written) in hits {
        if seen.insert((reader.clone(), writer.clone())) {
            tracing::warn!(
                reader = %reader,
                dropped = %writer,
                what = %written.describe(),
                frame = crate::queue::presented(),
                "reads what only dropped work writes"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn missing(what: Target) -> Write {
        Write {
            what,
            writer: "synthetic dispatch".to_owned(),
            frame: 9,
        }
    }

    #[test]
    fn a_live_buffer_write_only_clears_its_own_range() {
        let readers = [(8, 10), (30, 32), (48, 50)];
        for (live_start, live_end, expected) in [
            (0, 8, [true, true, true]),
            (0, 24, [false, true, true]),
            (24, 40, [true, false, true]),
            (40, 80, [true, true, false]),
            (8, 72, [false, false, false]),
        ] {
            let mut writes =
                VecDeque::from([missing(Target::Buffer(8, 72)), missing(Target::Texture(1))]);
            clear_written(&mut writes, Target::Buffer(live_start, live_end));
            let actual = readers.map(|(start, end)| {
                writes
                    .iter()
                    .any(|write| write.what.overlaps(Target::Buffer(start, end)))
            });
            assert_eq!(actual, expected, "live write {live_start}..{live_end}");
            assert!(writes.iter().any(|write| write.what == Target::Texture(1)));
            assert!(
                writes
                    .iter()
                    .all(|write| write.writer == "synthetic dispatch" && write.frame == 9)
            );
        }
    }

    #[test]
    fn splitting_ranges_keeps_the_newest_frames_when_full() {
        let mut writes: VecDeque<_> = (0..MAX_WRITES)
            .map(|frame| Write {
                frame: frame as u64,
                ..missing(Target::Buffer(8, 72))
            })
            .collect();

        clear_written(&mut writes, Target::Buffer(24, 40));

        assert_eq!(writes.len(), MAX_WRITES);
        assert_eq!(writes.front().unwrap().frame, (MAX_WRITES / 2) as u64);
        assert!(writes[MAX_WRITES - 2].what == Target::Buffer(8, 24));
        assert!(writes.back().unwrap().what == Target::Buffer(40, 72));
        assert!(
            writes
                .iter()
                .all(|write| write.frame >= (MAX_WRITES / 2) as u64)
        );
    }
}
