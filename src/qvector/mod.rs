//! A quaternary vector stores a vector with values from the range [0..4].
//!
//! This implementation uses a vector of `u128` and split the two bits of each
//! symbol. The upper 64 bits of each u128 store the first bit of the symbols while
//! the lower 64 bits store the second bit of each symbol.
//! This is very convenient for computing rank and select queries within a word as,
//! differently from representation that uses u64, we compute the operation on
//! 64 symbols at once (instead of just 32 symbols).
//! Rank queries are 10% faster.

use crate::utils::get_64byte_aligned_vector;
use crate::{AccessUnsigned, SpaceUsage}; // Traits

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Default, Eq, PartialEq)]
pub struct QVector {
    data: Vec<u128>,
    position: usize,
}

impl QVector {
    const MASK: u128 = 3;
    const N_BITS_WORD: usize = 128;

    /// Create a new empty quaternary vector.
    ///
    /// # Examples
    /// ```
    /// use qwt::QVector;
    ///
    /// let qv = QVector::new();
    ///
    /// assert_eq!(qv.is_empty(), true);
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty quaternary vector with the capacity of `n` quaternary symbols.
    ///
    /// # Panics
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    /// ```
    /// use qwt::QVector;
    ///
    /// let qv = QVector::with_capacity(10);
    ///
    /// assert_eq!(qv.is_empty(), true);
    /// ```
    pub fn with_capacity(n: usize) -> Self {
        let capacity = (2 * n + Self::N_BITS_WORD - 1) / Self::N_BITS_WORD;
        Self {
            data: Vec::with_capacity(capacity),
            position: 0,
        }
    }

    /// Creates an empty vector with the capacity of `n` quaternary symbols..
    ///
    /// The data is 64-byte aligned so that every block is aligned to a
    /// cache line.
    /// However, this makes the vector not suitable too small sizes as
    /// the space usage is a multiple of 64 bytes.
    ///
    /// # Panics
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    pub fn with_capacity_align64(n: usize) -> Self {
        let capacity = (2 * n + Self::N_BITS_WORD - 1) / Self::N_BITS_WORD;
        let v;
        unsafe {
            v = get_64byte_aligned_vector::<u128>(capacity);
        }
        Self {
            data: v,
            position: 0,
        }
    }

    /// Checks if the vector is empty.
    ///
    /// # Examples
    /// ```
    /// use qwt::QVector;
    ///
    /// let qv = QVector::new();
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
    /// let qv = QVector::new();
    /// assert_eq!(qv.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.position >> 1
    }

    /// Appends the (last 2 bits of the) value `v` at the end
    /// of the quaternary vector.
    ///
    /// The does not check if the value `v` fits within the current width.
    /// The value is truncated to the two least significant bits.
    ///
    /// # Panics
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    /// ```
    /// use qwt::{QVector, AccessUnsigned};
    ///
    /// let mut qv = QVector::new();
    ///
    /// for i in 0..4 {
    ///     qv.push(i);
    /// }
    ///
    /// assert_eq!(qv.len(), 4);
    ///
    /// qv.push(11); // 1011 in bynary
    ///
    /// assert_eq!(qv.get(4), Some(3));
    /// ```
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

    /// Aligns data in the quatvector to 64 bytes.
    ///
    /// Todo: make this safe by checking invariants.
    ///
    /// # Safety
    /// See Safety of [Vec::Vec::from_raw_parts](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.from_raw_parts).
    pub unsafe fn align_to_64(&mut self) {
        let mut v = get_64byte_aligned_vector::<u128>(self.data.len());
        for &word in &self.data {
            v.push(word);
        }
        self.data = v;
    }

    /// Shrinks the vector to fits the data (rounded to the closest u128).
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }

    /// Gets access to the internal data.
    pub fn get_data(&self) -> &[u128] {
        &self.data
    }

    /// Returns an iterator over the values in the quaternary vector.
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
    pub fn iter(&self) -> QVectorIter {
        QVectorIter { qv: self, i: 0 }
    }
}

impl AccessUnsigned for QVector {
    type Item = u8;

    /// Accesses the `i`th value in the quaternary vector.
    /// The caller must guarantee that the position `i`  is
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
        // SAFETY:
        // Caller has to guarantee that block is a valid index.
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
        // Safety: Check before guarantees to be not out of bound
        unsafe { Some(self.get_unchecked(i)) }
    }
}

impl SpaceUsage for QVector {
    fn space_usage_bytes(&self) -> usize {
        self.data.space_usage_bytes() + 8
    }
}

pub struct QVectorIter<'a> {
    qv: &'a QVector,
    i: usize,
}

/*
impl IntoIterator for QVector {
    type IntoIter = QVectorIter<'a>;
    type Item = u64;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
*/

impl<'a> IntoIterator for &'a QVector {
    type IntoIter = QVectorIter<'a>;
    type Item = u8;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> Iterator for QVectorIter<'a> {
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
        // TODO: this may be faster without calling get.
        self.i += 1;
        self.qv.get(self.i - 1)
    }
}

macro_rules! impl_from_iterator_quat_vector {
    ($($t:ty),*) => {
        $(impl FromIterator<$t> for QVector {
                fn from_iter<T>(iter: T) -> Self
                where
                    T: IntoIterator<Item = $t>,
                {
                    let mut qv = QVector::default();
                    qv.extend(iter);
                    qv.shrink_to_fit();
                    qv
                }
            }

        impl Extend<$t> for QVector {
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

impl_from_iterator_quat_vector![i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize];

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

#[cfg(test)]
mod tests;
