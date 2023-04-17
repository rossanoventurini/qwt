//! This module implements a quad vector to store a sequence with values from [0..3],
//! i.e., two bits symbols.
//!
//! This implementation uses a vector of `u128`. Each `u128` stores (up to) 64
//! symbols. The upper 64 bits of each `u128` store the first bit of the symbols while
//! the lower 64 bits store the second bit of each symbol.
//! This is very convenient for computing `rank` and `select` queries within a word.
//! Indeed, we take the upper and lower parts and, via boolean operation), we obtain
//! a 64-bit word with a 1 in any position containing a given symbol.
//! This way, a popcount operation counts the number of occurrences of that symbol
//! in the word.
//! Note that using `u128` instead of `u64` is convenient here because the popcount
//! operates on 64 bits. So, the use of `u128` fully uses every popcount.

use crate::{AccessQuad, RankQuad, SpaceUsage}; // Traits

use num_traits::int::PrimInt;
use num_traits::AsPrimitive;

use serde::{Deserialize, Serialize};

// A quad vector is made of `DataLine`s. Each line consists of
// four u128, so each `DataLine` is 512 bits and fits in a cache line.
// This way, it is easier to force the alignment to 64 bytes.
//
// We support `access`, `rank`, and `select queries for each line.
#[derive(Copy, Clone, Default, Eq, PartialEq, Serialize, Deserialize, Debug)]
#[repr(C, align(64))]
struct DataLine {
    words: [u128; 4],
}

impl DataLine {
    const REPEATEDSYMB: [u128; 4] = [
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF0000000000000000,
        0x0000000000000000FFFFFFFFFFFFFFFF,
        0x00000000000000000000000000000000,
    ];

    // Return a u64 where each bit is `1` if the corresponding symbol
    // in the `word` is equal to the `symbol`, `0` otherwise.
    #[inline(always)]
    fn normalize(word: u128, symbol: u8) -> u64 {
        let word = word ^ Self::REPEATEDSYMB[symbol as usize];
        let word_high = (word >> 64) as u64;
        let word_low = word as u64;

        word_high & word_low
    }
}

impl AccessQuad for DataLine {
    #[inline(always)]
    fn get(&self, i: usize) -> Option<u8> {
        assert!(i < 256);
        // SAFETY: bounds already checked
        Some(unsafe { self.get_unchecked(i) })
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> u8 {
        let word_id = i >> 6;
        let shift = i & 63;

        let word = unsafe { *self.words.get_unchecked(word_id) };

        ((word >> (64 + shift - 1)) & 2 | (word >> shift) & 1) as u8
    }
}

impl RankQuad for DataLine {
    #[inline(always)]
    fn rank(&self, symbol: u8, i: usize) -> Option<usize> {
        assert!(symbol < 4);
        assert!(i <= 256);

        // SAFETY: checks above guarantee correctness
        Some(unsafe { self.rank_unchecked(symbol, i) })
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, symbol: u8, i: usize) -> usize {
        debug_assert!(symbol <= 3, "Only the four symbols in [0, 3] are possible.");
        debug_assert!(i <= 256, "Only positions up to 256 are possible");

        let last_word = i / 64;
        let offset = i & 63; // offset within the last word

        let mask_full = u64::MAX;
        let mask_offset = if offset > 0 { (1_u64 << offset) - 1 } else { 0 };
        let mask_zero = 0;

        let mut rank = 0;
        for (i, &word) in self.words.iter().enumerate() {
            let mask = if i < last_word { mask_full } else { mask_zero };
            let mask = if i == last_word { mask_offset } else { mask };
            rank += (Self::normalize(word, symbol) & mask).count_ones() as usize;
        }

        rank
    }
}

// The trait SelectQuad is not implemented because RSSupport needs to it by hand :-)

impl SpaceUsage for DataLine {
    fn space_usage_bytes(&self) -> usize {
        64
    }
}

#[derive(Clone, Default, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct QVector {
    data: Box<[DataLine]>,
    position: usize,
}

impl QVector {
    /// Check if the vector is empty.
    ///
    /// # Examples
    /// ```
    /// use qwt::QVector;
    ///
    /// let qv = QVector::default();
    /// assert!(qv.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.position == 0
    }

    /// Return the number of symbols in the quaternary vector.
    ///
    /// # Examples
    /// ```
    /// use qwt::QVector;
    ///
    /// let qv: QVector = [0, 1, 2, 3].into_iter().cycle().take(10).collect();
    /// assert_eq!(qv.len(), 10);
    /// ```
    pub fn len(&self) -> usize {
        self.position >> 1
    }

    /// Return an iterator over the values in the quad vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::QVector;
    ///
    /// let qv: QVector = [0, 1, 2, 3].into_iter().cycle().take(100).collect();;
    ///
    /// for (i, v) in qv.iter().enumerate() {
    ///    assert_eq!((i%4) as u8, v);
    /// }
    /// ```
    pub fn iter(&self) -> QVectorIterator<&QVector> {
        QVectorIterator { i: 0, qv: self }
    }
}

impl AccessQuad for QVector {
    /// Access the `i`th value in the quaternary vector.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    ///
    /// # Examples
    /// ```
    /// use qwt::{QVector, AccessQuad};
    ///
    /// let qv: QVector = [0, 1, 2, 3].into_iter().cycle().take(10).collect();
    /// unsafe {
    ///     assert_eq!(qv.get_unchecked(8), 0);
    /// }
    /// ```
    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> u8 {
        debug_assert!(i < self.position / 2);

        let line = i >> 8;
        let offset_in_line = i & 255;
        let line = self.data.get_unchecked(line);

        line.get_unchecked(offset_in_line)
    }

    /// Access the `i`th value in the quaternary vector
    /// or `None` if out of bounds.
    ///
    /// # Examples
    /// ```
    /// use qwt::QVector;
    /// use qwt::AccessQuad;
    ///
    /// let qv: QVector = [0, 1, 2, 3].into_iter().cycle().take(10).collect();
    ///
    /// assert_eq!(qv.get(8), Some(0));
    /// assert_eq!(qv.get(10), None);
    /// ```
    #[inline(always)]
    fn get(&self, i: usize) -> Option<u8> {
        if i >= self.position >> 1 {
            return None;
        }
        // SAFETY: Check before guarantees to be not out of bound
        unsafe { Some(self.get_unchecked(i)) }
    }
}

impl SpaceUsage for QVector {
    fn space_usage_bytes(&self) -> usize {
        self.data.space_usage_bytes() + 8
    }
}

impl AsRef<QVector> for QVector {
    fn as_ref(&self) -> &QVector {
        self
    }
}

pub struct QVectorIterator<QV: AsRef<QVector>> {
    i: usize,
    qv: QV,
}

impl<QV: AsRef<QVector>> Iterator for QVectorIterator<QV> {
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
        // TODO: this may be faster without calling get.
        let qv = self.qv.as_ref();
        self.i += 1;
        qv.get(self.i - 1)
    }
}

impl IntoIterator for QVector {
    type IntoIter = QVectorIterator<QVector>;
    type Item = u8;

    fn into_iter(self) -> Self::IntoIter {
        QVectorIterator { i: 0, qv: self }
    }
}

impl<'a> IntoIterator for &'a QVector {
    type IntoIter = QVectorIterator<&'a QVector>;
    type Item = u8;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> FromIterator<T> for QVector
where
    T: PrimInt + AsPrimitive<u8>,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut qvb = QVectorBuilder::default();
        qvb.extend(iter);
        qvb.build()
    }
}

/// Builder struct to build a `qvector` by pushing symbol by symbol.
/// The main reasons for this builder are
/// - we want to force `qvector` to be immutable. So, we don't want any method that
///   could change it;
/// - we want to save space when symbols are produced one after the other and store
///   them using 2 bits each.
#[derive(Clone, Default, Eq, PartialEq)]
pub struct QVectorBuilder {
    data: Vec<DataLine>,
    position: usize,
}

impl QVectorBuilder {
    const MASK: u128 = 3;
    const N_BITS_WORD: usize = 128 * 4;

    /// Build the `qvector`.
    pub fn build(self) -> QVector {
        QVector {
            data: self.data.into_boxed_slice(),
            position: self.position,
        }
    }

    /// Create a new empty dynamic quad vector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty dynamic quad vector with the capacity of `n` quad symbols.
    ///
    /// # Panics
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    pub fn with_capacity(n: usize) -> Self {
        let capacity = (2 * n + Self::N_BITS_WORD - 1) / Self::N_BITS_WORD;
        Self {
            data: Vec::with_capacity(capacity),
            position: 0,
        }
    }

    /// Appends the (last 2 bits of the) value `v` at the end
    /// of the quad vector.
    ///
    /// It does not check if the value `v` fits is actually in [0..3].
    /// The value is truncated to the two least significant bits.
    ///
    /// # Panics
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    pub fn push(&mut self, v: u8) {
        let offset_in_line = (self.position / 2) & 255;
        if offset_in_line == 0 {
            // no more space in the current line
            self.data.push(DataLine::default());
        }

        let word_id = offset_in_line >> 6;
        let cur_shift = offset_in_line & 63;

        let v = (v as u128) & Self::MASK;

        let last_line = self.data.last_mut().unwrap();

        last_line.words[word_id] |= (v >> 1) << (64 + cur_shift) | ((v & 1) << cur_shift);

        self.position += 2;
    }
}

impl<T> FromIterator<T> for QVectorBuilder
where
    T: PrimInt + AsPrimitive<u8>,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut qvb = QVectorBuilder::default();
        qvb.extend(iter);
        qvb
    }
}

impl<T> Extend<T> for QVectorBuilder
where
    T: PrimInt + AsPrimitive<u8>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for value in iter {
            //debug_assert!((0..4).contains(&value));
            self.push(value.as_());
        }
    }
}

pub mod rs_qvector;

#[cfg(test)]
mod tests;
