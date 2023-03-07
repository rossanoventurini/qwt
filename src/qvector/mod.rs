//! This module implements a quad vector to store a sequence with values from [0..3], i.e., two bits symbols.
//!
//! This implementation uses a vector of `u128`. Each `u128` stores (up to) 64
//! symbols. The upper 64 bits of each `u128` store the first bit of the symbols while
//! the lower 64 bits store the second bit of each symbol.
//! This is very convenient for computing `rank` and `select` queries within a word.
//! Indeed, we take the upper and lower parts and, via boolean operation), we obtain
//! a 64 bit word with a 1 in any position containing a given symbol.
//! This way, a popcount operation counts the number of occurrences of that symbol
//! in the word.
//! Note that using `u128` instead of `u64` is convenient here because the popcount
//! operates on 64 bits. So, the use of `u128` fully uses every popcount.

use crate::{AccessUnsigned, SpaceUsage}; // Traits

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Default, Eq, PartialEq)]
pub struct QVector {
    data: Box<[u128]>,
    position: usize,
}

impl QVector {
    /// Checks if the vector is empty.
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

    /// Returns the number of symbols in the quaternary vector.
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

    /* TODO: More tests are needed to understand if align is worth.
    /// Aligns data in the `qvector` to 64 bytes.
    ///
    /// Todo: make this safe by checking invariants.
    ///
    /// # Safety
    /// See Safety of [Vec::Vec::from_raw_parts](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.from_raw_parts).
    pub unsafe fn align_to_64(&mut self) {
        use crate::utils::get_64byte_aligned_vector;

        let mut v = get_64byte_aligned_vector::<u128>(self.data.len());
        for &word in self.data.iter() {
            v.push(word);
        }
        self.data = v.into_boxed_slice();
    }*/

    /// Gets access to the internal data.
    pub fn get_data(&self) -> &[u128] {
        &self.data
    }

    /// Returns an iterator over the values in the quad vector.
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

impl AccessUnsigned for QVector {
    type Item = u8;

    /// Accesses the `i`th value in the quaternary vector.
    /// The caller must guarantee that the position `i` is
    /// valid.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    ///
    /// # Examples
    /// ```
    /// use qwt::{QVector, AccessUnsigned};
    ///
    /// let qv: QVector = [0, 1, 2, 3].into_iter().cycle().take(10).collect();
    /// unsafe {
    ///     assert_eq!(qv.get_unchecked(8), 0);
    /// }
    /// ```
    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item {
        debug_assert!(i < self.position / 2);

        let block = i >> 6;
        let shift = i & 63;
        // SAFETY: Caller has to guarantee that `block` is a valid index.
        let word = *self.data.get_unchecked(block);

        ((word >> (64 + shift - 1)) & 2 | (word >> shift) & 1) as u8
    }

    /// Accesses the `i`th value in the quaternary vector
    /// or `None` if out of bounds.
    ///
    /// # Examples
    /// ```
    /// use qwt::QVector;
    /// use qwt::AccessUnsigned;
    ///
    /// let qv: QVector = [0, 1, 2, 3].into_iter().cycle().take(10).collect();
    ///
    /// assert_eq!(qv.get(8), Some(0));
    /// assert_eq!(qv.get(10), None);
    /// ```
    #[inline(always)]
    fn get(&self, i: usize) -> Option<Self::Item> {
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

macro_rules! impl_from_iterator_qvector {
    ($($t:ty),*) => {
        $(impl FromIterator<$t> for QVector {
                fn from_iter<T>(iter: T) -> Self
                where
                    T: IntoIterator<Item = $t>,
                {
                    let mut qvb = QVectorBuilder::default();
                    qvb.extend(iter);
                    qvb.build()
                }
            })*
    }
}

impl_from_iterator_qvector![i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize];

impl std::fmt::Debug for QVector {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let data_str: Vec<String> = self.data.iter().map(|x| format!("{:b}", x)).collect();
        write!(
            fmt,
            "QVector {{ position:{:?}, data:{:?}}}",
            self.position, data_str,
        )
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
    data: Vec<u128>,
    position: usize,
}

impl QVectorBuilder {
    const MASK: u128 = 3;
    const N_BITS_WORD: usize = 128;

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
        let cur_shift = (self.position / 2) % 64;
        let v = (v as u128) & Self::MASK;
        if cur_shift == 0 {
            self.data.push(0);
        }

        let last = self.data.last_mut().unwrap();
        *last |= (v >> 1) << (64 + cur_shift) | ((v & 1) << cur_shift);

        self.position += 2;
    }
}

macro_rules! impl_from_iterator_qvector_builder {
    ($($t:ty),*) => {
        $(impl FromIterator<$t> for QVectorBuilder {
                fn from_iter<T>(iter: T) -> Self
                where
                    T: IntoIterator<Item = $t>,
                {
                    let mut qvb = QVectorBuilder::default();
                    qvb.extend(iter);
                    qvb
                }
            }

        impl Extend<$t> for QVectorBuilder {
            fn extend<T>(&mut self, iter: T)
            where
                T: IntoIterator<Item = $t>
            {
                for value in iter {
                    debug_assert!((0..4).contains(&value));
                    self.push(value as u8);
                }
            }
        })*
    }
}

impl_from_iterator_qvector_builder![i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize];

#[cfg(test)]
mod tests;
