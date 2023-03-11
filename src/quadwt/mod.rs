//! This module implements a Quad Wavelet Tree to support access, rank, and select
//! queries on a vector of unsigned integers.
//!
//! This data structure supports three operations:
//! - `get(i)` accesses the `i`-th symbols of the indexed sequence;
//! - `rank(s, i)` counts the number of occurrences of symbol `s` up to position `i` excluded;
//! - `select(s, i)` returns the position of the `i`-th occurrence of symbol `s`.
//!
//! We can index vectors of length up to 2^{43} symbols.

use crate::utils::{msb, stable_partition_of_4};
use crate::{AccessUnsigned, RankUnsigned, SelectUnsigned, SpaceUsage, SymbolsStats};
use crate::{QVector, QVectorBuilder}; // Traits

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

// Traits bound
use num_traits::{AsPrimitive, PrimInt, Unsigned};
use std::ops::{Shl, Shr};

/// Alias for the trait bounds to be satisfied by a data structure
/// to support `rank` and `select` queries at each level of the wavelet tree.
/// We need an alias to avoid repeating a lot of bounds here and there.
pub trait RSforWT:
    From<QVector>
    + AccessUnsigned<Item = u8>
    + RankUnsigned
    + SelectUnsigned
    + SymbolsStats
    + SpaceUsage
    + Default
{
}
// Generic implementation for any T
impl<T> RSforWT for T where
    T: From<QVector>
        + AccessUnsigned<Item = u8>
        + RankUnsigned
        + SelectUnsigned
        + SymbolsStats
        + SpaceUsage
        + Default
{
}

/// Alias for the trait bounds of the type T to be indexable in the
/// wavelet tree.
pub trait WTIndexable:
    Unsigned + PrimInt + Ord + Shr<usize> + Shl<usize> + AsPrimitive<u8>
{
}

impl<T> WTIndexable for T
where
    T: Unsigned + PrimInt + Ord + Shr<usize> + Shl<usize> + AsPrimitive<u8>,
    u8: AsPrimitive<T>,
{
}

/// The generic RS is the data structure we use to index a quaternary sequence
/// to support Rank, Select and Access queries.
#[derive(Default, Clone, Serialize, Deserialize, PartialEq, Debug)]
pub struct QWaveletTree<T, RS> {
    n: usize,        // The length of the represented sequence
    n_levels: usize, // The number of levels of the wavelet matrix
    sigma: T, // The largest symbol in the sequence. *NOTE*: It's not +1 because it may overflow
    qvs: Vec<RS>, // A quad vector for each level
}

impl<T, RS> QWaveletTree<T, RS>
where
    T: WTIndexable,
    u8: AsPrimitive<T>,
    RS: RSforWT,
{
    /// Builds the wavelet tree of the `sequence` of unsigned integers.
    /// The input sequence is **destroyed**.
    ///
    /// The alphabet size `sigma` is the largest value in the `sequence`.
    /// Both space usage and query time of a QWaveletTree depend on
    /// $$\lfloor\log_2 (\sigma-1)\rfloor + 1$$ (i.e., the length of the
    /// binary representation of values in the sequence).
    /// For this reason, it may be convenient for both space usage and query time to
    /// remap the alphabet to form a consecutive range [0, d], where d is
    /// the number of distinct values in `sequence`.
    ///
    /// ## Panics
    /// Panics if the sequence is longer than the largest possible length.
    /// The largest possible length is 2^{43} symbols.
    ///
    /// # Examples
    /// ```
    /// use qwt::QWT256;
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWT256::from(data);
    ///
    /// assert_eq!(qwt.len(), 8);
    /// ```
    pub fn new(sequence: &mut [T]) -> Self {
        if sequence.is_empty() {
            return Self {
                n: 0,
                n_levels: 0,
                sigma: T::zero(),
                qvs: vec![RS::default()],
            };
        }
        let sigma = *sequence.iter().max().unwrap();
        let log_sigma = msb(sigma) + 1;
        let n_levels = ((log_sigma + 1) / 2) as usize; // TODO: if log_sigma is odd, the FIRST level should be a binary vector!

        let mut qvs = Vec::<RS>::with_capacity(n_levels);

        let mut shift = 2 * (n_levels - 1);

        for _level in 0..n_levels {
            let mut cur_qv = QVectorBuilder::with_capacity(sequence.len());
            for &symbol in sequence.iter() {
                let two_bits: u8 = (symbol >> shift).as_() & 3; // take the last 2 bits
                cur_qv.push(two_bits);
            }

            qvs.push(RS::from(cur_qv.build()));
            stable_partition_of_4(sequence, shift);

            if shift >= 2 {
                shift -= 2;
            }
        }

        qvs.shrink_to_fit();

        Self {
            n: sequence.len(),
            n_levels,
            sigma,
            qvs,
        }
    }

    /// Returns the length of the indexed sequence.
    ///
    /// # Examples
    /// ```
    /// use qwt::QWT256;
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWT256::from(data);
    ///
    /// assert_eq!(qwt.len(), 8);
    /// ```
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns the largest value in the sequence. Note: it is not +1 because it may overflow.
    pub fn sigma(&self) -> T {
        self.sigma
    }

    /// Checks if the indexed sequence is empty.
    ///
    /// # Examples
    /// ```
    /// use qwt::QWT256;
    ///
    /// let qwt = QWT256::<u8>::default();
    ///
    /// assert_eq!(qwt.is_empty(), true);
    /// ```
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Returns the number of levels in the wavelet tree.
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }

    /// Returns an iterator over the values in the wavelet tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::QWT256;
    ///
    /// let data: Vec<u8> = (0..10u8).into_iter().cycle().take(100).collect();
    ///
    /// let qwt = QWT256::from(data);
    ///
    /// for (i, v) in qwt.iter().enumerate() {
    ///    assert_eq!((i%10) as u8, v);
    /// }
    /// ```
    pub fn iter(&self) -> QWTIterator<T, RS, &QWaveletTree<T, RS>> {
        QWTIterator {
            i: 0,
            qwt: self,
            _phantom: PhantomData,
        }
    }
}

impl<T, RS> RankUnsigned for QWaveletTree<T, RS>
where
    T: WTIndexable,
    u8: AsPrimitive<T>,
    RS: RSforWT,
{
    /// Returns rank of `symbol` up to position `i` **excluded**.
    /// `None`, is returned if `i` is out of bound or if `symbol`
    /// is not valid (i.e., it is greater than or equal to the alphabet size).
    ///
    /// # Examples
    /// ```
    /// use qwt::{QWT256, RankUnsigned};
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWT256::from(data);
    ///
    /// assert_eq!(qwt.rank(1, 2), Some(1));
    /// assert_eq!(qwt.rank(3, 8), Some(1));
    /// assert_eq!(qwt.rank(1, 0), Some(0));
    /// assert_eq!(qwt.rank(1, 9), None);     // too large position
    /// assert_eq!(qwt.rank(6, 1), None);     // too large symbol
    /// ```
    #[inline(always)]
    fn rank(&self, symbol: Self::Item, i: usize) -> Option<usize> {
        if i > self.n || symbol > self.sigma {
            return None;
        }

        // Safety: Check above guarantees we are not out of bound
        Some(unsafe { self.rank_unchecked(symbol, i) })
    }

    /// Returns rank of `symbol` up to position `i` **excluded**.
    ///
    /// # Safety
    /// Calling this method with a position `i` larger than the size of the sequence
    /// of with invalid symbol is undefined behavior.
    ///
    /// # Examples
    /// ```
    /// use qwt::{QWT256, RankUnsigned};
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWT256::from(data);
    ///
    /// unsafe {
    ///     assert_eq!(qwt.rank_unchecked(1, 2), 1);
    /// }
    /// ```
    #[inline(always)]
    unsafe fn rank_unchecked(&self, symbol: Self::Item, i: usize) -> usize {
        let mut shift: i64 = (2 * (self.n_levels - 1)) as i64;
        let mut cur_i = i;
        let mut cur_p = 0;

        for level in 0..self.n_levels - 1 {
            let two_bits: u8 = (symbol >> shift as usize).as_() & 3;

            // Safety: Here we are sure that two_bits is a symbol in [0..3]
            let offset = unsafe { self.qvs[level].occs_smaller_unchecked(two_bits) };
            cur_p = self.qvs[level].rank_unchecked(two_bits, cur_p) + offset;
            cur_i = self.qvs[level].rank_unchecked(two_bits, cur_i) + offset;

            shift -= 2;
        }

        let two_bits: u8 = (symbol >> shift as usize).as_() & 3;

        cur_i = self.qvs[self.n_levels - 1].rank_unchecked(two_bits, cur_i);
        cur_p = self.qvs[self.n_levels - 1].rank_unchecked(two_bits, cur_p);

        cur_i - cur_p
    }
}

impl<T, RS> AccessUnsigned for QWaveletTree<T, RS>
where
    T: WTIndexable,
    u8: AsPrimitive<T>,
    RS: RSforWT,
{
    type Item = T;

    /// Returns the `i`-th symbol of the indexed sequence, `None` is returned if `i` is out of bound.
    ///
    /// # Examples
    /// ```
    /// use qwt::{QWT256, AccessUnsigned};
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWT256::from(data);
    ///
    /// assert_eq!(qwt.get(2), Some(1));
    /// assert_eq!(qwt.get(3), Some(0));
    /// assert_eq!(qwt.get(8), None);
    /// ```
    ///
    /// ```
    /// use qwt::{QWT256, AccessUnsigned, RankUnsigned, SelectUnsigned};
    ///
    /// let data = vec![1u32, 0, 1, 0, 2, 1000000, 5, 3];
    /// let qwt = QWT256::from(data);
    ///
    /// assert_eq!(qwt.get(2), Some(1));
    /// assert_eq!(qwt.get(5), Some(1000000));
    /// assert_eq!(qwt.get(8), None);
    /// ```

    #[inline(always)]
    fn get(&self, i: usize) -> Option<Self::Item> {
        if i >= self.n {
            return None;
        }
        // Safety: check before guarantees we are not out of bound
        Some(unsafe { self.get_unchecked(i) })
    }

    /// Returns the `i`-th symbol of the indexed sequence.
    ///    
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    ///
    /// # Examples
    /// ```
    /// use qwt::{QWT256, AccessUnsigned};
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWT256::from(data);
    ///
    /// unsafe {
    ///     assert_eq!(qwt.get_unchecked(2), 1);
    ///     assert_eq!(qwt.get_unchecked(3), 0);
    /// }
    /// ```
    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item {
        let mut result = T::zero();

        let mut cur_i = i;
        for level in 0..self.n_levels - 1 {
            // The last rank can be saved. The improvement is just ~3%. Indeed, most of the cost is for the cache miss for data access that we pay anyway
            let symbol = self.qvs[level].get_unchecked(cur_i);
            result = (result << 2) | symbol.as_();

            // Safety: Here we are sure that symbol is in [0..3]
            let offset = unsafe { self.qvs[level].occs_smaller_unchecked(symbol) };
            cur_i = self.qvs[level].rank_unchecked(symbol, cur_i) + offset;
        }

        let symbol = self.qvs[self.n_levels - 1].get_unchecked(cur_i);
        (result << 2) | symbol.as_()
    }
}

impl<T, RS> SelectUnsigned for QWaveletTree<T, RS>
where
    T: WTIndexable + std::fmt::Debug,
    u8: AsPrimitive<T>,
    RS: RSforWT,
{
    /// Returns the position of the `i`-th occurrence of symbol `symbol`, `None` is
    /// returned if i is 0 or if there is no such occurrence for the symbol or if
    /// `symbol` is not valid (i.e., it is greater than or equal to the alphabet size).
    ///
    /// # Examples
    /// ```
    /// use qwt::{QWT256, SelectUnsigned};
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWT256::from(data);
    ///
    /// assert_eq!(qwt.select(1, 1), Some(0));
    /// assert_eq!(qwt.select(0, 2), Some(3));
    /// assert_eq!(qwt.select(1, 0), None);
    /// assert_eq!(qwt.select(6, 1), None);
    /// ```    
    #[inline(always)]
    fn select(&self, symbol: Self::Item, i: usize) -> Option<usize> {
        if i == 0 || symbol > self.sigma {
            return None;
        }

        let mut path_off = Vec::with_capacity(self.n_levels);
        let mut rank_path_off = Vec::with_capacity(self.n_levels);

        let mut b = 0;
        let mut shift: i64 = 2 * (self.n_levels - 1) as i64;

        for level in 0..self.n_levels {
            path_off.push(b);

            let two_bits = (symbol >> shift as usize).as_() & 3;

            let rank_b = self.qvs[level].rank(two_bits, b)?;

            // Safety: we are sure the symbol `two_bits` is in [0..3]
            b = rank_b + unsafe { self.qvs[level].occs_smaller_unchecked(two_bits) };
            shift -= 2;

            rank_path_off.push(rank_b);
        }

        shift = 0;
        let mut result = i;
        for level in (0..self.n_levels).rev() {
            b = path_off[level];
            let rank_b = rank_path_off[level];
            let two_bits = (symbol >> shift as usize).as_() & 3;

            result = self.qvs[level].select(two_bits, rank_b + result)? - b + 1;
            shift += 2;
        }

        Some(result - 1)
    }

    /// Returns the position of the `i`-th occurrence of symbol `symbol`.
    ///
    /// # Safety
    /// Calling this method with a value of `i` which is larger than the number of
    /// occurrences of the `symbol` or if the `symbol` is not valid is undefined behavior.
    ///
    /// In the current implementation there is no reason to prefer this unsafe select
    /// over the safe one.
    #[inline(always)]
    unsafe fn select_unchecked(&self, symbol: Self::Item, i: usize) -> usize {
        self.select(symbol, i).unwrap()
    }
}

impl<T, RS> SpaceUsage for QWaveletTree<T, RS>
where
    T: WTIndexable,
    u8: AsPrimitive<T>,
    RS: RSforWT,
{
    /// Gives the space usage in bytes of the struct.
    fn space_usage_bytes(&self) -> usize {
        8 + 8
            + self
                .qvs
                .iter()
                .fold(0, |acc, ds| acc + ds.space_usage_bytes())
    }
}

impl<T, RS> AsRef<QWaveletTree<T, RS>> for QWaveletTree<T, RS>
where
    T: WTIndexable,
    u8: AsPrimitive<T>,
    RS: From<QVector>
        + AccessUnsigned<Item = u8>
        + RankUnsigned
        + SelectUnsigned
        + SymbolsStats
        + SpaceUsage
        + Default,
{
    fn as_ref(&self) -> &QWaveletTree<T, RS> {
        self
    }
}

// This is a naive implementation of an iterator for WT.
// We could do better by storing more information and
// avoid rank operations!
#[derive(Debug, PartialEq)]
pub struct QWTIterator<T, RS, Q: AsRef<QWaveletTree<T, RS>>>
where
    T: WTIndexable,
    u8: AsPrimitive<T>,
    RS: RSforWT,
{
    i: usize,
    qwt: Q,
    _phantom: PhantomData<(T, RS)>,
}

impl<T, RS, Q: AsRef<QWaveletTree<T, RS>>> Iterator for QWTIterator<T, RS, Q>
where
    T: WTIndexable,
    u8: AsPrimitive<T>,
    RS: RSforWT,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        // TODO: this may be faster without calling get.
        let qwt = self.qwt.as_ref();
        self.i += 1;
        qwt.get(self.i - 1)
    }
}

impl<T, RS> IntoIterator for QWaveletTree<T, RS>
where
    T: WTIndexable,
    u8: AsPrimitive<T>,
    RS: RSforWT,
{
    type IntoIter = QWTIterator<T, RS, QWaveletTree<T, RS>>;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        QWTIterator {
            i: 0,
            qwt: self,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T, RS> IntoIterator for &'a QWaveletTree<T, RS>
where
    T: WTIndexable,
    u8: AsPrimitive<T>,
    RS: RSforWT,
{
    type IntoIter = QWTIterator<T, RS, &'a QWaveletTree<T, RS>>;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, RS> FromIterator<T> for QWaveletTree<T, RS>
where
    T: WTIndexable,
    u8: AsPrimitive<T>,
    RS: RSforWT,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        QWaveletTree::new(&mut iter.into_iter().collect::<Vec<T>>())
    }
}

impl<T, RS> From<Vec<T>> for QWaveletTree<T, RS>
where
    T: WTIndexable,
    u8: AsPrimitive<T>,
    RS: RSforWT,
{
    fn from(mut v: Vec<T>) -> Self {
        QWaveletTree::new(&mut v[..])
    }
}

#[cfg(test)]
mod tests;
