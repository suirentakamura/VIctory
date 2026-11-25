//! Hash maps for the submission path's own keys (addresses, page and slot
//! numbers): std's default hasher (SipHash-1-3, keyed against flooding)
//! took a fifth of the submitting thread in the world, for keys no one
//! outside chooses. This one multiplies and rotates, word by word (the
//! scheme of rustc's `FxHasher`).

use std::hash::{BuildHasherDefault, Hasher};

/// The hasher.
#[derive(Default, Clone, Copy)]
pub(crate) struct Fast(u64);

/// An odd constant with well-spread bits (as rustc's).
const K: u64 = 0xf135_7aea_2e62_a9c5;

impl Fast {
    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = self.0.wrapping_add(word).wrapping_mul(K);
    }
}

impl Hasher for Fast {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes(c.try_into().expect("eight bytes")));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut word = [0u8; 8];
            word[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(word) ^ (rest.len() as u64) << 56);
        }
    }

    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.add(u64::from(n));
    }

    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.add(u64::from(n));
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }

    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        // The table takes its control bits from the top: the product's best
        // bits are its high ones, rotated to cover the low ones too.
        self.0.rotate_left(26)
    }
}

pub(crate) type State = BuildHasherDefault<Fast>;
pub(crate) type Map<K, V> = std::collections::HashMap<K, V, State>;
pub(crate) type Set<K> = std::collections::HashSet<K, State>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_numbers_spread_over_the_table() {
        let set: Set<usize> = (0..4096).map(|p| p * 65536).collect();
        assert_eq!(set.len(), 4096);
        let buckets: std::collections::HashSet<u64> = (0..4096usize)
            .map(|p| {
                let mut h = Fast::default();
                h.write_usize(p * 65536);
                h.finish() >> 57
            })
            .collect();
        // All 128 top-seven-bit tags are used.
        assert_eq!(buckets.len(), 128);
    }
}
