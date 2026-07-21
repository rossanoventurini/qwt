//! This module implements a quad vector to store a sequence with values in the range [0..3],
//! i.e., two bits symbols.
//!
//! This implementation uses a vector of `DataLine`. Each `DataLine` is an array of four `u128`
//! and stores (up to) 256 symbols. As each `DataLine` is 512 bits it fits in a cache line.
//! The use of `DataLine` in our setting is particulary conveninet because a vector of `DataLine` is aligned.
//! This way, we load just one cache line everytime we access a `DataLine`.

use crate::{AccessQuad, RankQuad}; // Traits
use mem_dbg::{MemDbg, MemSize};
use num_traits::int::PrimInt;
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};

// A quad vector is made of `DataLine`s. Each line consists of
// four u128, so each `DataLine` is 512 bits and fits in a cache line.
// This way, it is easier to force the alignment to 64 bytes.
//
// We support `access`, `rank`, and `select queries for each line.
/// One cache-line of 256 two-bit symbols (512 bits).
/// POD layout for zero-copy I/O; fields are crate-private (Group B).
#[derive(Copy, Clone, Default, Eq, PartialEq, Serialize, MemSize, MemDbg, Deserialize, Debug)]
#[repr(C, align(64))]
pub struct DataLine {
    pub(crate) words: [u128; 4],
}



impl DataLine {
    const MASK: u128 = 3;

    const REPEATEDSYMB: [u128; 2] = [
        u128::MAX, // !bit repeated
        0,
    ];

    /// Bitmasks of positions holding `symbol` in this line.
    ///
    /// Returns two `u128` words covering symbols 0..128 and 128..256.
    #[inline(always)]
    pub fn normalize(&self, symbol: u8) -> (u128, u128) {
        let mask_high = Self::REPEATEDSYMB[(symbol >> 1) as usize];
        let mask_low = Self::REPEATEDSYMB[(symbol & 1) as usize];

        let word_high_0 = self.words[0] ^ mask_high;
        let word_low_0 = self.words[2] ^ mask_low;
        let word_high_1 = self.words[1] ^ mask_high;
        let word_low_1 = self.words[3] ^ mask_low;

        (word_high_0 & word_low_0, word_high_1 & word_low_1)
    }

    // Set the position `i` to `symbol`
    #[inline]
    fn set_symbol(&mut self, symbol: u8, i: u8) {
        // The higher bit is placed in the first two words,
        // the lower bit is placed in the second two words.

        let word_id_high = i >> 7;
        let word_id_low = word_id_high + 2;
        let cur_shift = i & 127;

        let symbol = (symbol as u128) & Self::MASK;

        self.words[word_id_high as usize] |= (symbol >> 1) << cur_shift;
        self.words[word_id_low as usize] |= (symbol & 1) << cur_shift;
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
        let word_id_high = i >> 7;
        let word_id_low = word_id_high + 2;
        let cur_shift = i & 127;

        let word_high = unsafe { *self.words.get_unchecked(word_id_high) };
        let word_low = unsafe { *self.words.get_unchecked(word_id_low) };

        ((word_high >> (cur_shift) & 1) << 1 | (word_low >> cur_shift) & 1) as u8
    }
}

impl RankQuad for DataLine {
    #[inline(always)]
    fn rank(&self, symbol: u8, i: usize) -> Option<usize> {
        if symbol >= 4 || i > 256 {
            return None;
        }

        // SAFETY: checks above guarantee correctness
        Some(unsafe { self.rank_unchecked(symbol, i) })
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, symbol: u8, i: usize) -> usize {
        debug_assert!(symbol <= 3, "Only the four symbols in [0, 3] are possible.");
        debug_assert!(i <= 256, "Only positions up to 256 are possible");

        let (word_0, word_1) = self.normalize(symbol);

        let last_word = i >> 7;
        let offset = i & 127; // offset within the last word

        let mask_full = u128::MAX;
        let mask_offset = (1_u128 << offset) - 1;

        let mask = if last_word == 0 {
            mask_offset
        } else {
            mask_full
        };
        let mut rank = (word_0 & mask).count_ones();

        let mask = if last_word == 1 {
            mask_offset
        } else {
            mask_full * (last_word == 2) as u128
        };

        rank += (word_1 & mask).count_ones();

        rank as usize
    }

    /// Rank of all four symbols `0..3` up to position `i` (excluded) within this line.
    ///
    /// Loads each data word once and partitions bits by (high, low) pair.
    /// Used by range-distinct iteration to avoid four independent
    /// `rank_unchecked` passes over the same cache line.
    #[inline(always)]
    unsafe fn rank_all_unchecked(&self, i: usize) -> [usize; 4] {
        debug_assert!(i <= 256, "Only positions up to 256 are possible");

        let last_word = i >> 7;
        let offset = i & 127;
        let mask_full = u128::MAX;
        // offset==0 → empty mask; (1<<0)-1 == 0. For offset in 1..127, (1<<offset)-1.
        // When i is a multiple of 128 with last_word>0, the previous word uses mask_full
        // and this word is not included — matching rank_unchecked.
        let mask_offset = if offset == 0 {
            0u128
        } else {
            (1_u128 << offset) - 1
        };

        let mask0 = if last_word == 0 {
            mask_offset
        } else {
            mask_full
        };
        let mask1 = if last_word == 1 {
            mask_offset
        } else {
            mask_full * (last_word == 2) as u128
        };

        let h0 = self.words[0] & mask0;
        let l0 = self.words[2] & mask0;
        let nh0 = mask0 ^ h0;
        let nl0 = mask0 ^ l0;

        let h1 = self.words[1] & mask1;
        let l1 = self.words[3] & mask1;
        let nh1 = mask1 ^ h1;
        let nl1 = mask1 ^ l1;

        // symbol = (high << 1) | low
        [
            ((nh0 & nl0).count_ones() + (nh1 & nl1).count_ones()) as usize, // 0
            ((nh0 & l0).count_ones() + (nh1 & l1).count_ones()) as usize,   // 1
            ((h0 & nl0).count_ones() + (h1 & nl1).count_ones()) as usize,   // 2
            ((h0 & l0).count_ones() + (h1 & l1).count_ones()) as usize,     // 3
        ]
    }
}

// The trait SelectQuad is not implemented because RSSupport needs to it by hand :-)

#[derive(Clone, Default, Eq, PartialEq, Serialize, MemSize, MemDbg, Deserialize, Debug)]
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
    /// let qv: QVector = [0, 1, 2, 3].into_iter().cycle().take(100).collect();
    ///
    /// for (i, v) in qv.iter().enumerate() {
    ///     assert_eq!((i % 4) as u8, v);
    /// }
    /// ```
    pub fn iter(&self) -> QVectorIterator<&QVector> {
        QVectorIterator { i: 0, qv: self }
    }

    /// Bit-cursor (`2 * len()`). Exposed for zero-copy / mmap flatten.
    #[inline]
    pub(crate) fn position_bits(&self) -> usize {
        self.position
    }

    /// Raw data lines. Exposed for zero-copy / mmap flatten.
    #[inline]
    pub(crate) fn data_lines(&self) -> &[DataLine] {
        &self.data
    }

    /// Build a `QVector` from raw data lines and a bit cursor.
    ///
    /// `position` is the bit cursor (`2 * len()`). This is the inverse of
    /// [`data_lines`](Self::data_lines) + [`position_bits`](Self::position_bits)
    /// and is the assembly path used by zero-copy I/O.
    ///
    /// # Panics
    /// - if `position` is odd (must be a multiple of 2 bits per symbol)
    /// - if `position > data.len() * 512`
    #[must_use]
    pub fn from_raw_parts(data: Box<[DataLine]>, position: usize) -> Self {
        assert!(position % 2 == 0, "position must be a multiple of 2");
        assert!(
            position <= data.len() * 512,
            "position exceeds data capacity"
        );
        Self { data, position }
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
    /// use qwt::{AccessQuad, QVector};
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
        let pos_in_last_line = i & 255;
        let line = self.data.get_unchecked(line);

        line.get_unchecked(pos_in_last_line)
    }

    /// Access the `i`th value in the quaternary vector
    /// or `None` if `i` is out of bounds.
    ///
    /// # Examples
    /// ```
    /// use qwt::{AccessQuad, QVector};
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
#[derive(Clone, Default, Eq, MemSize, MemDbg, PartialEq)]
pub struct QVectorBuilder {
    data: Vec<DataLine>,
    position: usize,
}

impl QVectorBuilder {
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
        let capacity = (2 * n).div_ceil(Self::N_BITS_WORD);
        Self {
            data: Vec::with_capacity(capacity),
            position: 0,
        }
    }

    /// Appends the (last 2 bits of the) value `symbol` at the end
    /// of the quad vector.
    ///
    /// It does not check if the value `symbol` fits is actually in [0..3].
    /// The value is truncated to the two least significant bits.
    ///
    /// # Panics
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    pub fn push(&mut self, symbol: u8) {
        let pos_in_last_line = (self.position / 2) & 255;
        if pos_in_last_line == 0 {
            // no more space in the current line
            self.data.push(DataLine::default());
        }

        self.data
            .last_mut()
            .unwrap()
            .set_symbol(symbol, pos_in_last_line as u8);

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
            // debug_assert!((0..4).contains(&value));
            self.push(value.as_());
        }
    }
}

pub mod rs_qvector;

#[cfg(test)]
mod tests;
