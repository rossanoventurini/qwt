//! This module provides implementations for both mutable and immutable bit vectors.
//!
//! The mutable bit vector offers operations to [`AccessBin`], append, and modify bits at arbitrary positions.
//!
//! The immutable bit vector allows access to bits and can be extended to support [`RankBin`] and [`SelectBin`] queries.
//!
//! For both data structures, it is possible to iterate over bits or positions of bits set either to zero or one.

use crate::{utils::select_in_word, AccessBin, RankBin, SelectBin, SpaceUsage};

use serde::{Deserialize, Serialize};

pub mod rs_narrow;
pub mod rs_wide;

#[derive(Copy, Clone, Default, Eq, PartialEq, Serialize, Deserialize, Debug)]
#[repr(C, align(64))]
struct DataLine {
    words: [u64; 8],
}

impl DataLine {
    //set symbol to position `i`
    #[inline]
    fn set_symbol(&mut self, symbol: u64, i: usize) {
        assert!(i < 512);

        let mask: u64 = 1 << (i % 64);
        self.words[i >> 6] ^= self.words[i >> 6] & mask; //zero out the position
        self.words[i >> 6] ^= (symbol & 1) << (i % 64); //set position to symbol
    }

    #[inline]
    fn get_word(&self, i: usize) -> u64 {
        assert!(i < 8);
        self.words[i]
    }

    #[inline]
    fn n_ones(&self) -> usize {
        self.words
            .iter()
            .fold(0, |a, x| a + x.count_ones() as usize)
    }

    #[inline]
    fn n_zeros(&self) -> usize {
        512 - self.n_ones()
    }
}

impl AccessBin for DataLine {
    #[inline(always)]
    fn get(&self, i: usize) -> Option<bool> {
        assert!(i < 512);
        // SAFETY: bounds already checked
        Some(unsafe { self.get_unchecked(i) })
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> bool {
        (self.words[i >> 6] >> (i % 64)) & 1 == 1
    }
}

impl RankBin for DataLine {
    #[inline(always)]
    fn rank1(&self, i: usize) -> Option<usize> {
        if i > 512 {
            return None;
        }

        Some(unsafe { self.rank1_unchecked(i) })
    }

    #[inline(always)]
    unsafe fn rank1_unchecked(&self, i: usize) -> usize {
        let mut left = i as i32;
        let mut rank = 0;
        for w in 0..8 {
            if left < 0 {
                break;
            }
            let cur_word = self.words.get_unchecked(w);
            let mask: u64 = if left > 63 {
                0xFFFFFFFFFFFFFFFF
            } else {
                (1 << left) - 1
            };
            rank += (cur_word & mask).count_ones() as usize;

            left -= 64;
        }
        rank
    }

    fn n_zeros(&self) -> usize {
        self.n_zeros()
    }
}

impl SpaceUsage for DataLine {
    fn space_usage_byte(&self) -> usize {
        8 * 8
    }
}

fn cast_to_u64_slice(data_lines: &[DataLine]) -> &[u64] {
    //WARNING: this works because DataLine is aligned
    unsafe {
        let len = data_lines.len().checked_mul(8).unwrap();
        let ptr = data_lines.as_ptr();
        let u64_ptr = ptr as *const u64;
        std::slice::from_raw_parts(u64_ptr, len)
    }
}

impl SelectBin for DataLine {
    fn select1(&self, i: usize) -> Option<usize> {
        if i >= self.n_ones() {
            return None;
        }

        Some(unsafe { self.select1_unchecked(i) })
    }

    #[inline(always)]
    unsafe fn select1_unchecked(&self, i: usize) -> usize {
        let mut off = 0;
        let mut rank = 0; //rank so far

        for w in 0..8 {
            let kp = self.words.get_unchecked(w).count_ones();
            if kp as usize > (i - rank) {
                off += select_in_word(*self.words.get_unchecked(w), (i - rank) as u64) as usize;
                break;
            } else {
                rank += kp as usize;
                off += 64;
            }
        }
        off
    }

    fn select0(&self, i: usize) -> Option<usize> {
        if i >= self.n_zeros() {
            return None;
        }

        Some(unsafe { self.select1_unchecked(i) })
    }

    #[inline(always)]
    unsafe fn select0_unchecked(&self, i: usize) -> usize {
        let mut rank = 0;
        let mut off = 0;

        for w in 0..8 {
            let word_to_select = !self.words.get_unchecked(w);
            let kp = word_to_select.count_ones();
            if kp as usize > (i - rank) {
                off += select_in_word(word_to_select, (i - rank) as u64) as usize;
                break;
            } else {
                rank += kp as usize;
                off += 64;
            }
        }
        off
    }
}

/// Implementation of an immutable bit vector.
#[derive(Default, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct BitVector {
    data: Box<[DataLine]>,
    n_bits: usize,
    n_ones: usize,
}

impl BitVector {
    /// Accesses `len` bits, with 1 <= `len` <= 64, starting at position `index`.
    ///
    /// Returns [`None`] if `index`+`len` is out of bounds,
    /// if `len` is 0, or if `len` is greater than 64.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::{BitVector, AccessBin};
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVector = v.into_iter().collect();
    /// assert_eq!(bv.get(1), Some(false));
    ///
    /// assert_eq!(bv.get_bits(1, 3), Some(0b110)); // Accesses bits from index 1 to 3
    ///
    /// // Accessing bits from index 1 to 8, which is out of bounds
    /// assert_eq!(bv.get_bits(1, 8), None);
    ///
    /// // Accessing more than 64 bits
    /// assert_eq!(bv.get_bits(0, 65), None);
    ///
    /// // Accessing more than 0 bits
    /// assert_eq!(bv.get_bits(0, 0), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn get_bits(&self, index: usize, len: usize) -> Option<u64> {
        if (len == 0) | (len > 64) | (index + len >= self.n_bits) {
            return None;
        }
        // SAFETY: safe access due to the above checks
        Some(unsafe { self.get_bits_unchecked(index, len) })
    }

    /// Accesses `len` bits, starting at position `index`, without performing bounds checking.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it does not perform bounds checking.
    /// It is the caller's responsibility to ensure that the provided `index` and `len`
    /// are within the bounds of the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::{BitVector};
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVector = v.into_iter().collect();
    ///
    /// // This is unsafe because it does not perform bounds checking
    /// unsafe {
    ///     assert_eq!(bv.get_bits_unchecked(1, 3), 0b110);
    /// }
    /// ```
    #[must_use]
    #[inline]
    pub unsafe fn get_bits_unchecked(&self, index: usize, len: usize) -> u64 {
        BitVectorMut::get_bits_slice(cast_to_u64_slice(&self.data), index, len)
    }

    /// Gets a whole 64-bit word from the bit vector at index `i` in the underlying vector of u64.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVector;
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVector = v.into_iter().collect();
    ///
    /// // Get the 64-bit word at index 0
    /// let word = bv.get_word(0);
    /// assert_eq!(word, 0b111101);
    /// ```
    #[must_use]
    #[inline(always)]
    pub fn get_word(&self, i: usize) -> u64 {
        self.data[i >> 3].words[i % 8]
    }

    /// Returns a non-consuming iterator over positions of bits set to 1 in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVector;
    ///
    /// let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
    /// let bv: BitVector = vv.iter().copied().collect();
    ///
    /// ```
    #[must_use]
    pub fn ones(&self) -> BitVectorBitPositionsIter<true> {
        BitVectorBitPositionsIter::new(cast_to_u64_slice(&self.data), self.n_bits)
    }

    /// Returns a non-consuming iterator over positions of bits set to 1 in the bit vector, starting at a specified bit position.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVector;
    ///
    /// let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
    /// let bv: BitVector = vv.iter().copied().collect();
    ///
    /// let v: Vec<usize> = bv.ones_with_pos(2).collect();
    /// assert_eq!(v, vec![63, 128, 129, 254, 1026]);
    /// ```
    #[must_use]
    pub fn ones_with_pos(&self, pos: usize) -> BitVectorBitPositionsIter<true> {
        BitVectorBitPositionsIter::with_pos(cast_to_u64_slice(&self.data), self.n_bits, pos)
    }

    /// Returns a non-consuming iterator over positions of bits set to 0 in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVector;
    /// use qwt::perf_and_test_utils::negate_vector;
    ///
    /// let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
    /// let bv: BitVector = vv.iter().copied().collect();
    ///
    /// let v: Vec<usize> = bv.zeros().collect();
    /// assert_eq!(v, negate_vector(&vv));
    /// ```
    #[must_use]
    pub fn zeros(&self) -> BitVectorBitPositionsIter<false> {
        BitVectorBitPositionsIter::new(cast_to_u64_slice(&self.data), self.n_bits)
    }

    /// Returns a non-consuming iterator over positions of bits set to 0 in the bit vector, starting at a specified bit position.
    #[must_use]
    pub fn zeros_with_pos(&self, pos: usize) -> BitVectorBitPositionsIter<false> {
        BitVectorBitPositionsIter::with_pos(cast_to_u64_slice(&self.data), self.n_bits, pos)
    }

    /// Returns a non-consuming iterator over bits of the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVector;
    ///
    /// let v = vec![0,2,3,5];
    /// let bv: BitVector = v.into_iter().collect();
    ///
    /// let mut iter = bv.iter();
    /// assert_eq!(iter.next(), Some(true)); // First bit is true
    /// assert_eq!(iter.next(), Some(false)); // Second bit is false
    /// assert_eq!(iter.next(), Some(true)); // Third bit is true
    /// assert_eq!(iter.next(), Some(true)); // Fourth bit is true
    /// assert_eq!(iter.next(), Some(false)); // Fifth bit is false
    /// assert_eq!(iter.next(), Some(true)); // Sixth bit is true
    /// assert_eq!(iter.next(), None); // End of the iterator
    /// ```
    pub fn iter(&self) -> BitVectorIter {
        BitVectorIter {
            data: cast_to_u64_slice(&self.data),
            n_bits: self.n_bits,
            i: 0,
        }
    }

    /// Checks if the bit vector is empty.
    ///
    /// # Returns
    ///
    /// Returns `true` if the bit vector is empty, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVector;
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVector = v.into_iter().collect();
    ///
    /// assert!(!bv.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_bits == 0
    }

    /// Returns the number of bits in the bit vector.
    ///
    /// # Returns
    ///
    /// The number of bits in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVector;
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVector = v.into_iter().collect();
    ///
    /// assert_eq!(bv.len(), 6);
    /// ```
    pub fn len(&self) -> usize {
        self.n_bits
    }

    /// Counts the number of ones (bits set to 1) in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVector;
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVector = v.into_iter().collect();
    ///
    /// assert_eq!(bv.count_ones(), 5);
    /// ```
    pub fn count_ones(&self) -> usize {
        self.n_ones
    }

    /// Counts the number of zeros (bits set to 0) in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVector;
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVector = v.into_iter().collect();
    ///
    /// assert_eq!(bv.count_zeros(), 1);
    /// ```
    #[inline]
    #[must_use]
    pub fn count_zeros(&self) -> usize {
        self.len() - self.n_ones
    }
}

impl AccessBin for BitVector {
    /// Returns the bit at the given position `index`,
    /// or [`None`] if `index` is out of bounds.
    ///
    /// # Examples
    /// ```
    /// use qwt::{BitVector, AccessBin};
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVector = v.into_iter().collect();
    ///
    /// assert_eq!(bv.get(5), Some(true));
    /// assert_eq!(bv.get(1), Some(false));
    /// assert_eq!(bv.get(10), None);
    /// ```
    #[must_use]
    #[inline(always)]
    fn get(&self, index: usize) -> Option<bool> {
        if index >= self.len() {
            return None;
        }
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Returns the bit at position `index`.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    ///
    /// # Examples
    /// ```
    /// use qwt::{BitVector, AccessBin};
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVector = v.into_iter().collect();
    ///
    /// assert_eq!(unsafe{bv.get_unchecked(5)}, true);
    /// ```
    #[must_use]
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> bool {
        BitVectorMut::get_bit_slice(cast_to_u64_slice(&self.data), index)
    }
}

impl SpaceUsage for BitVector {
    /// Returns the space usage in bytes.
    #[must_use]
    fn space_usage_byte(&self) -> usize {
        self.data.space_usage_byte() + 8 + 8
    }
}

/// Creates a `BitVector` from an iterator over `bool` values.
///
/// # Examples
///
/// ```
/// use qwt::{AccessBin, BitVector};
///
/// // Create a bit vector from an iterator over bool values
/// let bv: BitVector = vec![true, false, true].into_iter().collect();
///
/// assert_eq!(bv.len(), 3);
/// assert_eq!(bv.get(1), Some(false));
/// ```
impl FromIterator<bool> for BitVector {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = bool>,
    {
        let mut bv = BitVectorMut::default();
        bv.extend(iter);

        bv.into()
    }
}

// it contains all the type of num_traits::int::PrimInt without bool
pub trait MyPrimInt: TryInto<usize> {}

macro_rules! impl_my_prim_int {
    ($($t:ty),*) => {
        $(impl MyPrimInt for $t {
        })*
    }
}

impl_my_prim_int![i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, u128, i128];

/// Creates a `BitVector` from an iterator over non-negative integer values.
///
/// # Panics
/// Panics if any value of the sequence cannot be converted to usize.
///
/// # Examples
///
/// ```
/// use qwt::{AccessBin, BitVector};
///
/// // Create a bit vector from an iterator over usize values
/// let bv: BitVector = vec![0, 1, 3, 5].into_iter().collect();
///
/// assert_eq!(bv.len(), 6);
/// assert_eq!(bv.get(3), Some(true));
/// ```
impl<V> FromIterator<V> for BitVector
where
    V: MyPrimInt,
    <V as TryInto<usize>>::Error: std::fmt::Debug,
{
    #[must_use]
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = V>,
        <V as TryInto<usize>>::Error: std::fmt::Debug,
    {
        let mut bv = BitVectorMut::default();
        bv.extend(
            iter.into_iter()
                .map(|x| x.try_into().expect("Cannot a value convert to usize")),
        );

        bv.into()
    }
}

/// Implements conversion from mutable `BitVectorMut` to immutable `BitVector`.
///
/// This conversion consumes the original `BitVectorMut` and creates an immutable version.
///
/// # Examples
///
/// ```
/// use qwt::{BitVector, BitVectorMut, AccessBin};
///
/// let mut bvm = BitVectorMut::new();
/// bvm.push(true);
/// bvm.push(false);
///
/// // Convert mutable BitVectorMut to immutable BitVector
/// let bv: BitVector = bvm.into();
///
/// assert_eq!(bv.get(0), Some(true));
/// ```
impl From<BitVectorMut> for BitVector {
    fn from(bvm: BitVectorMut) -> Self {
        Self {
            data: bvm.data.into_boxed_slice(),
            n_bits: bvm.n_bits,
            n_ones: bvm.n_ones,
        }
    }
}

/// Implements conversion from immutable `BitVector` to mutable `BitVectorMut`.
///
/// This conversion takes ownership of the original `BitVector` and creates a mutable version.
///
/// # Examples
///
/// ```
/// use qwt::{BitVector, BitVectorMut, AccessBin};
///
/// let v = vec![0,2,3,4,5];
/// let mut bv: BitVector = v.into_iter().collect();
///
/// // Convert immutable BitVector to mutable BitVectorMut
/// let mut bvm: BitVectorMut = bv.into();
///
/// assert_eq!(bvm.get(0), Some(true));
/// assert_eq!(bvm.len(), 6);
/// bvm.push(true);
/// assert_eq!(bvm.len(), 7);
/// ```
impl From<BitVector> for BitVectorMut {
    fn from(bv: BitVector) -> Self {
        Self {
            data: bv.data.into(),
            n_bits: bv.n_bits,
            n_ones: bv.n_ones,
        }
    }
}

impl AsRef<BitVector> for BitVector {
    fn as_ref(&self) -> &BitVector {
        self
    }
}

impl AsRef<BitVectorMut> for BitVectorMut {
    fn as_ref(&self) -> &BitVectorMut {
        self
    }
}

pub struct BitVectorBitPositionsIter<'a, const BIT: bool> {
    data: &'a [u64],
    n_bits: usize,
    cur_position: usize,
    cur_word_pos: usize,
    cur_word: u64,
}

impl<'a, const BIT: bool> BitVectorBitPositionsIter<'a, BIT> {
    #[must_use]
    #[inline(always)]
    pub fn new(data: &'a [u64], n_bits: usize) -> Self {
        BitVectorBitPositionsIter {
            data,
            n_bits,
            cur_position: 0,
            cur_word_pos: 0, // points the the next word to read
            cur_word: 0,     // last word we read
        }
    }

    #[must_use]
    #[inline(always)]
    pub fn with_pos(data: &'a [u64], n_bits: usize, pos: usize) -> Self {
        let cur_word_pos = pos >> 6;
        let cur_word = if cur_word_pos < data.len() {
            if BIT {
                data[cur_word_pos]
            } else {
                // for zeros, just negate the word and report the positions of bit set to one!
                !data[cur_word_pos]
            }
        } else {
            0
        };
        let l = pos % 64;

        let cur_word = cur_word >> l;

        dbg!(pos, l);

        BitVectorBitPositionsIter {
            data,
            n_bits,
            cur_position: pos,
            cur_word_pos: cur_word_pos + 1,
            cur_word,
        }
    }
}

/// Iterator over the positions of bits set to BIT (false for zeros,
/// true for ones) in the bit vector.
impl<'a, const BIT: bool> Iterator for BitVectorBitPositionsIter<'a, BIT> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_position >= self.n_bits {
            return None;
        }

        while self.cur_word == 0 {
            if self.cur_word_pos < self.data.len() {
                if BIT {
                    self.cur_word = self.data[self.cur_word_pos];
                } else {
                    // for zeros, just negate the word and report the positions of bit set to one!
                    self.cur_word = !self.data[self.cur_word_pos];
                }
                self.cur_position = self.cur_word_pos << 6;
            } else {
                return None;
            }
            self.cur_word_pos += 1;
        }
        let l = self.cur_word.trailing_zeros() as usize;
        self.cur_position += l;
        let pos = self.cur_position;

        self.cur_word = if l >= 63 { 0 } else { self.cur_word >> (l + 1) };

        self.cur_position += 1;
        if pos >= self.n_bits {
            None
        } else {
            Some(pos)
        }
    }
}

pub struct BitVectorIter<'a> {
    data: &'a [u64],
    n_bits: usize,
    i: usize,
}

/// An owning iterator over the bits of a [`BitVector`]
pub struct BitVectorIntoIter {
    bv: BitVector,
    i: usize,
}

impl ExactSizeIterator for BitVectorIntoIter {
    fn len(&self) -> usize {
        self.bv.n_bits - self.i
    }
}

impl Iterator for BitVectorIntoIter {
    type Item = bool;
    fn next(&mut self) -> Option<Self::Item> {
        self.i += 1;
        self.bv.get(self.i - 1)
    }
}

impl IntoIterator for BitVector {
    type IntoIter = BitVectorIntoIter;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter {
        BitVectorIntoIter { bv: self, i: 0 }
    }
}

impl IntoIterator for BitVectorMut {
    type IntoIter = BitVectorIntoIter;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter {
        BitVectorIntoIter {
            bv: self.into(), // just use the same iterator of immutable bitvector
            i: 0,
        }
    }
}

impl<'a> IntoIterator for &'a BitVector {
    type IntoIter = BitVectorIter<'a>;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> Iterator for BitVectorIter<'a> {
    type Item = bool;
    fn next(&mut self) -> Option<Self::Item> {
        // TODO: this may be faster without calling get.
        if self.i < self.n_bits {
            self.i += 1;
            Some(unsafe { BitVectorMut::get_bit_slice(self.data, self.i - 1) })
        } else {
            None
        }
    }
}

impl<'a> ExactSizeIterator for BitVectorIter<'a> {
    fn len(&self) -> usize {
        self.n_bits - self.i
    }
}

/// Implementation of a mutable bit vector.
#[derive(Default, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct BitVectorMut {
    data: Vec<DataLine>,
    n_bits: usize,
    n_ones: usize,
}

impl BitVectorMut {
    /// Creates a new empty bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVectorMut;
    ///
    /// let bv = BitVectorMut::new();
    /// assert_eq!(bv.len(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty bit vector with at least a capacity of `n_bits`.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVectorMut;
    ///
    /// let bv = BitVectorMut::new();
    /// assert_eq!(bv.len(), 0);
    /// ```
    #[must_use]
    pub fn with_capacity(n_bits: usize) -> Self {
        let capacity = (n_bits + 63) / 64;
        Self {
            data: Vec::with_capacity(capacity),
            ..Self::default()
        }
    }

    /// Creates a bit vector with `n_bits` set to 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVectorMut;
    ///
    /// let bv = BitVectorMut::with_zeros(5);
    /// assert_eq!(bv.len(), 5);
    /// assert_eq!(bv.count_ones(), 0);
    /// ```
    #[must_use]
    pub fn with_zeros(n_bits: usize) -> Self {
        let mut bv = Self::with_capacity(n_bits);
        bv.extend_with_zeros(n_bits);
        bv.shrink_to_fit();
        bv
    }

    /// Pushes a `bit` at the end of the bit vector.
    ///
    /// # Panics
    ///
    /// Panics if the size of the bit vector exceeds `usize::MAX` bits.
    ///
    /// # Example
    ///
    /// ```
    /// use qwt::{BitVectorMut, AccessBin};
    ///
    /// let mut bv = BitVectorMut::new();
    /// bv.push(true);
    /// bv.push(false);
    /// bv.push(true);
    ///
    /// assert_eq!(bv.len(), 3);
    /// assert_eq!(bv.get(0), Some(true));
    /// assert_eq!(bv.count_ones(), 2);
    /// ```
    #[inline]
    pub fn push(&mut self, bit: bool) {
        let pos_in_line = self.n_bits % 512;
        if pos_in_line == 0 {
            self.data.push(DataLine::default());
        }
        if bit {
            // push a 1
            if let Some(last) = self.data.last_mut() {
                last.set_symbol(1, pos_in_line);
            }
            self.n_ones += 1;
        }
        self.n_bits += 1;
    }

    /// Appends `len` bits at the end of the bit vector by taking
    /// the least significant `len` bits in the u64 value `bits`.
    ///
    /// # Panics
    ///
    /// Panics if `len` is larger than 64 or if a bit of position
    /// larger than `len` is set in `bits`.
    ///
    /// Panics if the size of the bit vector exceeds `usize::MAX` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVectorMut;
    ///
    /// let mut bv = BitVectorMut::with_capacity(7);
    /// bv.append_bits(0b101, 3);  // appends 101
    /// bv.append_bits(0b0110, 4); // appends 0110  
    ///
    ///         
    /// assert_eq!(bv.len(), 7);
    /// assert_eq!(bv.get_bits(0, 3), Some(5));
    /// ```
    #[inline]
    pub fn append_bits(&mut self, bits: u64, len: usize) {
        assert!(len == 64 || (bits >> len) == 0);
        assert!(len <= 64);

        if len == 0 {
            return;
        }

        // self.n_ones += bits.count_ones() as usize; taken care in push

        // let pos_in_line: usize = self.n_bits & 511;
        // self.n_bits += len; taken care in push

        for i in 0..len {
            self.push((bits >> i) & 1 == 1);
        }

        // if pos_in_word == 0 {
        //     self.data.push(bits);
        // } else if let Some(last) = self.data.last_mut() {
        //     *last |= bits << pos_in_word;
        //     if len > 64 - pos_in_word {
        //         self.data.push(bits >> (64 - pos_in_word));
        //     }
        // }
    }

    /// Extends the bit vector by adding `n` bits set to 0.
    ///
    /// # Panics
    ///
    /// Panics if the size of the bit vector exceeds `usize::MAX` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::{BitVectorMut, AccessBin};
    ///
    /// let mut bv = BitVectorMut::with_capacity(10);
    /// bv.extend_with_zeros(10);
    /// assert_eq!(bv.len(), 10);
    /// assert_eq!(bv.get(8), Some(false));
    /// ```
    #[inline]
    pub fn extend_with_zeros(&mut self, n: usize) {
        self.n_bits += n;
        let new_size = (self.n_bits + 511) / 512;
        self.data.resize_with(new_size, Default::default);
    }

    /// Sets the bit at the given position `index` to `bit`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::{BitVectorMut, AccessBin};
    ///
    /// let mut bv = BitVectorMut::with_capacity(2);
    /// bv.push(true);
    /// bv.push(false);
    ///
    /// bv.set(1, true);
    /// assert_eq!(bv.get(1), Some(true));
    ///
    /// // This will panic because index is out of bounds
    /// // bv.set(10, false);
    /// ```
    #[inline]
    pub fn set(&mut self, index: usize, bit: bool) {
        assert!(index < self.n_bits);

        // SAFETY: check above guarantees we are within the bound
        unsafe {
            if bit && !self.get_unchecked(index) {
                self.n_ones += 1;
            }
            if !bit && self.get_unchecked(index) {
                self.n_ones -= 1;
            }
        }

        let dl = index >> 9;
        let pos_in_dl = index & 511;
        self.data[dl].set_symbol(bit as u64, pos_in_dl);
    }

    /// Accesses `len` bits, with 1 <= `len` <= 64, starting at position `index`.
    ///
    /// Returns [`None`] if `index`+`len` is out of bounds,
    /// if `len` is 0, or if `len` is greater than 64.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::{BitVectorMut, AccessBin};
    ///
    /// let mut bv = BitVectorMut::with_capacity(6);
    /// bv.append_bits(0b111101, 6); // Appends 111101 note that we append bits from right to left
    /// assert_eq!(bv.get(1), Some(false));
    ///
    /// assert_eq!(bv.get_bits(1, 3), Some(0b110)); // Accesses bits from index 1 to 3
    ///
    /// // Accessing bits from index 1 to 8, which is out of bounds
    /// assert_eq!(bv.get_bits(1, 8), None);
    ///
    /// // Accessing more than 64 bits
    /// assert_eq!(bv.get_bits(0, 65), None);
    ///
    /// // Accessing more than 0 bits
    /// assert_eq!(bv.get_bits(0, 0), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn get_bits(&self, index: usize, len: usize) -> Option<u64> {
        if (len == 0) | (len > 64) | (index + len >= self.n_bits) {
            return None;
        }
        // SAFETY: safe access due to the above checks
        Some(unsafe { self.get_bits_unchecked(index, len) })
    }

    /// Accesses `len` bits, starting at position `index`, without performing bounds checking.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it does not perform bounds checking.
    /// It is the caller's responsibility to ensure that the provided `index` and `len`
    /// are within the bounds of the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVectorMut;
    ///
    /// let mut bv = BitVectorMut::with_capacity(6);
    /// bv.append_bits(0b111101, 6); // Appends 101101
    ///
    /// // This is unsafe because it does not perform bounds checking
    /// unsafe {
    ///     assert_eq!(bv.get_bits_unchecked(1, 3), 0b110);
    /// }
    /// ```
    #[must_use]
    #[inline]
    pub unsafe fn get_bits_unchecked(&self, index: usize, len: usize) -> u64 {
        Self::get_bits_slice(cast_to_u64_slice(&self.data), index, len)
    }

    // Private function to decode bits at a given index on a slice. The function does not
    // check bounds.
    #[inline]
    unsafe fn get_bits_slice(data: &[u64], index: usize, len: usize) -> u64 {
        let block = index >> 6;
        let shift = index & 63;

        let mask = if len == 64 {
            std::u64::MAX
        } else {
            (1_u64 << len) - 1
        };

        if shift + len <= 64 {
            return data[block] >> shift & mask;
        }

        (data[block] >> shift) | (data[block + 1] << (64 - shift) & mask)
    }

    // Private function to decode a bit at a given index on a slice. The function does not
    // check bounds.
    #[inline]
    #[must_use]
    unsafe fn get_bit_slice(data: &[u64], index: usize) -> bool {
        let word = index >> 6;
        let pos_in_word = index & 63;

        data[word] >> pos_in_word & 1_u64 == 1
    }

    /// Sets `len` bits, with 1 <= `len` <= 64,
    /// starting at position `index` to the `len` least
    /// significant bits in `bits`.
    ///
    /// # Panics
    ///
    /// Panics if `index`+`len` is out of bounds,
    /// `len` is greater than 64, or if the most significant bit in `bits`
    /// is at a position larger than or equal to `len`.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVectorMut;
    ///
    /// let mut bv = BitVectorMut::with_zeros(5);
    /// bv.set_bits(0, 3, 0b101); // Sets bits 0 to 2 to 101
    /// assert_eq!(bv.get_bits(0, 3), Some(0b101));
    /// ```
    #[inline]
    pub fn set_bits(&mut self, index: usize, len: usize, bits: u64) {
        assert!(index + len <= self.n_bits);
        // check there are no spurious bits
        assert!(len == 64 || (bits >> len) == 0);
        assert!(len <= 64);

        if len == 0 {
            return;
        }

        self.n_ones += bits.count_ones() as usize;

        // let mask = if len == 64 {
        //     std::u64::MAX
        // } else {
        //     (1_u64 << len) - 1
        // };
        // let word = index >> 6;
        // let pos_in_word = index & 63;

        // self.data[word] &= !(mask << pos_in_word);
        // self.data[word] |= bits << pos_in_word;

        // let stored = 64 - pos_in_word;
        // if stored < len {
        //     self.data[word + 1] &= !(mask >> stored);
        //     self.data[word + 1] |= bits >> stored;
        // }

        for i in 0..len {
            self.data[(index + i) >> 9].set_symbol((bits >> i) & 1, (index + i) % 512)
        }
    }

    /// Gets a whole 64-bit word from the bit vector at index `i` in the underlying vector of u64.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVectorMut;
    ///
    /// let mut bv = BitVectorMut::new();
    /// bv.append_bits(0b111101, 64);
    ///
    /// // Get the 64-bit word at index 0
    /// let word = bv.get_word(0);
    /// assert_eq!(word, 0b111101);
    /// ```
    #[must_use]
    #[inline(always)]
    pub fn get_word(&self, i: usize) -> u64 {
        self.data[i >> 3].words[i % 8]
    }

    /// Returns a non-consuming iterator over positions of bits set to 1 in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVectorMut;
    ///
    /// let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
    /// let bv: BitVectorMut = vv.iter().copied().collect();
    ///
    /// let v: Vec<usize> = bv.ones().collect();
    /// assert_eq!(v, vv);
    /// ```
    #[must_use]
    pub fn ones(&self) -> BitVectorBitPositionsIter<true> {
        BitVectorBitPositionsIter::new(cast_to_u64_slice(&self.data), self.n_bits)
    }

    /// Returns a non-consuming iterator over positions of bits set to 1 in the bit vector, starting at a specified bit position.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVectorMut;
    ///
    /// let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
    /// let bv: BitVectorMut = vv.iter().copied().collect();
    ///
    /// let v: Vec<usize> = bv.ones_with_pos(2).collect();
    /// assert_eq!(v, vec![63, 128, 129, 254, 1026]);
    /// ```
    #[must_use]
    pub fn ones_with_pos(&self, pos: usize) -> BitVectorBitPositionsIter<true> {
        BitVectorBitPositionsIter::with_pos(cast_to_u64_slice(&self.data), self.n_bits, pos)
    }

    /// Returns a non-consuming iterator over positions of bits set to 0 in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVectorMut;
    /// use qwt::perf_and_test_utils::negate_vector;
    ///
    /// let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
    /// let bv: BitVectorMut = vv.iter().copied().collect();
    ///
    /// let v: Vec<usize> = bv.zeros().collect();
    /// assert_eq!(v, negate_vector(&vv));
    /// ```
    #[must_use]
    pub fn zeros(&self) -> BitVectorBitPositionsIter<false> {
        BitVectorBitPositionsIter::new(cast_to_u64_slice(&self.data), self.n_bits)
    }

    /// Returns a non-consuming iterator over positions of bits set to 0 in the bit vector, starting at a specified bit position.
    #[must_use]
    pub fn zeros_with_pos(&self, pos: usize) -> BitVectorBitPositionsIter<false> {
        BitVectorBitPositionsIter::with_pos(cast_to_u64_slice(&self.data), self.n_bits, pos)
    }

    /// Returns a non-consuming iterator over bits of the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVectorMut;
    ///
    /// let mut bv = BitVectorMut::with_capacity(6);
    /// bv.append_bits(0b101101, 6); // Appends 101101
    ///
    /// let mut iter = bv.iter();
    /// assert_eq!(iter.next(), Some(true)); // First bit is true
    /// assert_eq!(iter.next(), Some(false)); // Second bit is false
    /// assert_eq!(iter.next(), Some(true)); // Third bit is true
    /// assert_eq!(iter.next(), Some(true)); // Fourth bit is true
    /// assert_eq!(iter.next(), Some(false)); // Fifth bit is false
    /// assert_eq!(iter.next(), Some(true)); // Sixth bit is true
    /// assert_eq!(iter.next(), None); // End of the iterator
    /// ```
    pub fn iter(&self) -> BitVectorIter {
        BitVectorIter {
            data: cast_to_u64_slice(&self.data),
            n_bits: self.n_bits,
            i: 0,
        }
    }

    /// Shrinks the underlying vector of 64-bit words to fit the actual size of the bit vector.
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }

    /// Checks if the bit vector is empty.
    ///
    /// # Returns
    ///
    /// Returns `true` if the bit vector is empty, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVectorMut;
    ///
    /// let mut bv = BitVectorMut::new();
    ///
    /// assert!(bv.is_empty());
    ///
    /// bv.push(true);
    ///
    /// assert!(!bv.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_bits == 0
    }

    /// Returns the number of bits in the bit vector.
    ///
    /// # Returns
    ///
    /// The number of bits in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVectorMut;
    ///
    /// let mut bv = BitVectorMut::new();
    ///
    /// assert_eq!(bv.len(), 0);
    ///
    /// bv.push(true);
    ///
    /// assert_eq!(bv.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.n_bits
    }

    /// Counts the number of ones (bits set to 1) in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVectorMut;
    ///
    /// let mut bv = BitVectorMut::with_capacity(6);
    /// bv.push(true);
    /// bv.push(false);
    /// bv.push(true);
    ///
    /// assert_eq!(bv.count_ones(), 2);
    /// ```
    pub fn count_ones(&self) -> usize {
        self.n_ones
    }

    /// Counts the number of zeros (bits set to 0) in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::BitVectorMut;
    ///
    /// let mut bv = BitVectorMut::with_capacity(6);
    /// bv.push(true);
    /// bv.push(false);
    /// bv.push(true);
    ///
    /// assert_eq!(bv.count_zeros(), 1);
    /// ```
    #[inline]
    #[must_use]
    pub fn count_zeros(&self) -> usize {
        self.len() - self.n_ones
    }
}

impl AccessBin for BitVectorMut {
    /// Returns the bit at the given position `index`,
    /// or [`None`] if `index` is out of bounds.
    ///
    /// # Examples
    /// ```
    /// use qwt::{BitVectorMut, AccessBin};
    ///
    /// let mut bv = BitVectorMut::with_capacity(10);
    /// bv.extend_with_zeros(10);
    /// assert_eq!(bv.get(8), Some(false));
    /// assert_eq!(bv.get(10), None);
    /// ```
    #[must_use]
    #[inline(always)]
    fn get(&self, index: usize) -> Option<bool> {
        if index >= self.len() {
            return None;
        }
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Returns the bit at position `index`.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    ///
    /// # Examples
    /// ```
    /// use qwt::{BitVectorMut, AccessBin};
    ///
    /// let mut bv = BitVectorMut::with_capacity(10);
    /// bv.extend_with_zeros(10);
    /// assert_eq!(unsafe{bv.get_unchecked(8)}, false);
    /// ```
    #[must_use]
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> bool {
        Self::get_bit_slice(cast_to_u64_slice(&self.data), index)
    }
}

impl SpaceUsage for BitVectorMut {
    /// Returns the space usage in bytes.
    #[must_use]
    fn space_usage_byte(&self) -> usize {
        self.data.space_usage_byte() + 8 + 8
    }
}

impl Extend<bool> for BitVectorMut {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = bool>,
    {
        for bit in iter {
            self.push(bit);
        }
    }

    /* Nigthly
        fn extend_one(&mut self, item: bool) {
            self.push(item);
        }
        fn extend_reserve(&mut self, additional: usize) {
            self.data.reserve
        }
    */
}

/// Implements creating a `BitVectorMut` from an iterator over `bool` values.
///
/// # Examples
///
/// ```
/// use qwt::{AccessBin, BitVectorMut};
///
/// // Create a bit vector from an iterator over bool values
/// let bv: BitVectorMut = vec![true, false, true].into_iter().collect();
/// assert_eq!(bv.len(), 3);
/// assert_eq!(bv.get(1), Some(false));
/// ```
impl FromIterator<bool> for BitVectorMut {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = bool>,
    {
        let mut bv = BitVectorMut::default();
        bv.extend(iter);
        bv.shrink_to_fit();

        bv
    }
}

/// Implements creating a `BitVectorMut` from an iterator over `usize` values.
///
/// # Examples
///
/// ```
/// use qwt::{AccessBin, BitVectorMut};
///
/// // Create a bit vector from an iterator over usize values
/// let bv: BitVectorMut = vec![0, 1, 3, 5].into_iter().collect();
/// assert_eq!(bv.len(), 6);
/// assert_eq!(bv.get(3), Some(true));
/// ```
impl FromIterator<usize> for BitVectorMut {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = usize>,
    {
        let mut bv = BitVectorMut::default();
        bv.extend(iter);
        bv.shrink_to_fit();

        bv
    }
}

/// Extends a `BitVectorMut` with an iterator over `usize` values.
///
/// # Examples
///
/// ```
/// use qwt::{BitVectorMut, AccessBin};
///
/// let mut bv = BitVectorMut::new();
///
/// // Extending the bit vector with a range of positions
/// bv.extend(0..5);
/// assert_eq!(bv.len(), 5);
/// assert_eq!(bv.get(3), Some(true));
/// ```
impl Extend<usize> for BitVectorMut {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = usize>,
    {
        for pos in iter {
            if pos >= self.n_bits {
                self.extend_with_zeros(pos + 1 - self.n_bits);
            }
            self.set(pos, true);
        }
    }
}

impl std::fmt::Debug for BitVector {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let data_str: Vec<String> = self.data.iter().map(|x| format!("{:?}", x)).collect();
        write!(
            fmt,
            "BitVector {{ n_bits:{:?}, data:{:?}}}",
            self.n_bits, data_str
        )
    }
}

impl std::fmt::Debug for BitVectorMut {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let data_str: Vec<String> = self.data.iter().map(|x| format!("{:?}", x)).collect();
        write!(
            fmt,
            "BitVectorMut {{ n_bits:{:?}, data:{:?}}}",
            self.n_bits, data_str
        )
    }
}

#[cfg(test)]
mod tests;
