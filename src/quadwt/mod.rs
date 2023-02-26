//! This module implements a Quad Wavelet Tree to support access, rank, and select //! queries on a vector of unsigned integers.
//!
//! This data structure supports three operations:
//! - `get(i)` accesses the `i`-th symbols of the indexed sequence;
//! - `rank(s, i)` counts the number of occurrences of symbol `s` up to position `i` excluded;
//! - `select(s, i)` returns the position of the `i`-th occurrence of symbol `s`.
//!
//! We can index vectors of length up to (2^{44})/log sigma symbols.

use crate::utils::{msb, stable_partition_of_4};
use crate::QVector;
use crate::{AccessUnsigned, RankUnsigned, SelectUnsigned, SpaceUsage}; // Traits

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

// Traits bound
use num_traits::{AsPrimitive, PrimInt, Unsigned};
use std::ops::{Shl, Shr};

/// The generic RS is the data structure we use to index a quaternary sequence
/// to support Rank, Select and Access queries.
#[derive(Default, Clone, Serialize, Deserialize, PartialEq, Debug)]
pub struct QWaveletTree<T, RS>
where
    T: Unsigned + PrimInt + Ord + Shr<usize> + Shl<usize> + AsPrimitive<u8>,
    u8: AsPrimitive<T>,
    RS: From<QVector>
        + AccessUnsigned<Item = u8>
        + RankUnsigned
        + SelectUnsigned
        + SpaceUsage
        + Default,
{
    n: usize,        // length of the represented sequence
    n_levels: usize, // number of levels of the wavelet matrix
    qv: RS, // quaternary vector storing the entire wavelet matrix, one level after the other.
    rank: Vec<u64>, // for each level l, for each quaternary symbol s = 0..3, we store rank(s, l * size)
    count: Vec<u64>, // for each level l, for each quaternary symbol s = 0..3, we store how many symbols < s there are at level l.
    item_type: PhantomData<T>,
}

impl<T, RS> QWaveletTree<T, RS>
where
    T: Unsigned + PrimInt + Ord + Shr<usize> + Shl<usize> + AsPrimitive<u8>,
    u8: AsPrimitive<T>,
    RS: From<QVector>
        + AccessUnsigned<Item = u8>
        + RankUnsigned
        + SelectUnsigned
        + SpaceUsage
        + Default,
{
    /// Builds the wavelet matrix of the ```sequence``` of unsigned integers.
    /// The input sequence is **destroyed**.
    ///
    /// The alphabet size ```sigma``` is the largest value in the ```sequence```.
    /// Both space usage and query time of a QWaveletTree depends on $$\lfloor\log_2 (\sigma-1)\rfloor + 1$$ (i.e., the length of the binary representation of values in the sequence).
    /// For this reason, it may be convenient for both space usage and query time to
    /// remap the alphabet to form a consecutive range [0, d],  where d is the number of
    ///  distinct values in ```sequence```.
    ///
    /// ## Panics
    /// Panics if the sequence is longer than the largest possible length.
    /// The largest possible length is (2^{44})/log sigma symbols.
    ///
    /// # Examples
    /// ```
    /// use qwt::QWaveletTreeP256;
    ///
    /// let mut data: [u8; 8] = [1, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWaveletTreeP256::new(&mut data);
    ///
    /// assert_eq!(qwt.len(), 8);
    /// ```
    pub fn new(sequence: &mut [T]) -> Self {
        if sequence.is_empty() {
            return Self {
                n: 0,
                n_levels: 0,
                qv: RS::default(),
                rank: Vec::new(),
                count: Vec::new(),
                item_type: PhantomData,
            };
        }
        let sigma = *sequence.iter().max().unwrap();
        let log_sigma = msb(sigma) + 1;
        let n_levels = ((log_sigma + 1) / 2) as usize; // TODO: if log_sigma is odd, the FIRST level should be a binary vector!

        let mut qv = QVector::with_capacity(sequence.len() * n_levels);
        let mut shift = 2 * (n_levels - 1);

        for _level in 0..n_levels {
            for &symbol in sequence.iter() {
                let two_bits: u8 = (symbol >> shift).as_() & 3; // take the last 2 bits
                qv.push(two_bits);
            }

            stable_partition_of_4(sequence, shift);

            if shift >= 2 {
                shift -= 2;
            }
        }

        let mut rank = Vec::with_capacity(n_levels);
        let mut count = Vec::with_capacity(n_levels);

        let qv = RS::from(qv);
        let size = sequence.len();
        for level in 0..n_levels {
            for symbol in 0..4u8 {
                rank.push(qv.rank(symbol, level * size).unwrap() as u64);
                let mut cnt = 0;
                for s in 0..symbol {
                    let rank_level =
                        qv.rank(s, (level + 1) * size).unwrap() - qv.rank(s, level * size).unwrap();
                    cnt += rank_level;
                }
                count.push(cnt as u64);
            }
        }

        /*
        qv.shrink_to_fit();
        unsafe {
            qv.align_to_64();
        }
        */

        Self {
            n: sequence.len(),
            n_levels,
            qv,
            rank,
            count,
            item_type: PhantomData,
        }
    }

    /// Returns the length of the indexed sequence.
    ///
    /// # Examples
    /// ```
    /// use qwt::QWaveletTreeP256;
    ///
    /// let mut data: [u8; 8] = [1, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWaveletTreeP256::new(&mut data);
    ///
    /// assert_eq!(qwt.len(), 8);
    /// ```
    pub fn len(&self) -> usize {
        self.n
    }

    /// Checks if the indexed sequence is empty.
    ///
    /// # Examples
    /// ```
    /// use qwt::QWaveletTreeP256;
    ///
    /// let qwt = QWaveletTreeP256::default();
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
}

impl<T, RS> RankUnsigned for QWaveletTree<T, RS>
where
    T: Unsigned + PrimInt + Ord + Shr<usize> + Shl<usize> + AsPrimitive<u8>,
    u8: AsPrimitive<T>,
    RS: From<QVector>
        + AccessUnsigned<Item = u8>
        + RankUnsigned
        + SelectUnsigned
        + SpaceUsage
        + Default,
{
    /// Returns rank of `symbol` up to position `i` **excluded**.
    /// `None`, is returned if `i` is out of bound.
    ///
    /// # Examples
    /// ```
    /// use qwt::{QWaveletTreeP256, RankUnsigned};
    ///
    /// let mut data: [u8; 8] = [1, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWaveletTreeP256::new(&mut data);
    ///
    /// assert_eq!(qwt.rank(1, 2), Some(1));
    /// assert_eq!(qwt.rank(3, 8), Some(1));
    /// assert_eq!(qwt.rank(1, 0), Some(0));
    /// assert_eq!(qwt.rank(1, 9), None);
    /// ```
    #[inline(always)]
    fn rank(&self, symbol: Self::Item, i: usize) -> Option<usize> {
        if i > self.n {
            return None;
        }
        // Safety: Check above guarantees we are not out of bound
        Some(unsafe { self.rank_unchecked(symbol, i) })
    }

    /// Returns rank of `symbol` up to position `i` **excluded**.
    ///
    /// # Safety
    /// Calling this method with a position `i` larger than the size of the sequence
    /// is undefined behavior.
    ///
    /// # Examples
    /// ```
    /// use qwt::{QWaveletTreeP256, RankUnsigned};
    ///
    /// let mut data: [u8; 8] = [1, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWaveletTreeP256::new(&mut data);
    ///
    /// unsafe {
    ///     assert_eq!(qwt.rank_unchecked(1, 2), 1);
    /// }
    /// ```
    #[inline(always)]
    unsafe fn rank_unchecked(&self, symbol: Self::Item, i: usize) -> usize {
        let mut b = 0; // b is always the beginning of the level
        let mut shift: i64 = (2 * (self.n_levels - 1)) as i64;
        let mut curr_i = i;

        for level in 0..self.n_levels {
            let two_bits: u8 = (symbol >> shift as usize).as_() & 3;
            let j = (level * 4) + two_bits as usize;

            // # occurrences of 's' in the interval [0,b)
            let rank_b = self.qv.rank_unchecked(two_bits, b);

            // # occurrences of 'symbol' in the interval [b,i)
            curr_i = self.qv.rank_unchecked(two_bits, b + curr_i) - rank_b;

            b = (level + 1) * self.n + (rank_b - self.rank[j] as usize) + (self.count[j] as usize);
            shift -= 2;
        }

        curr_i
    }
}

impl<T, RS> AccessUnsigned for QWaveletTree<T, RS>
where
    T: Unsigned + PrimInt + Ord + Shr<usize> + Shl<usize> + AsPrimitive<u8>,
    u8: AsPrimitive<T>,
    RS: From<QVector>
        + AccessUnsigned<Item = u8>
        + RankUnsigned
        + SelectUnsigned
        + SpaceUsage
        + Default,
{
    type Item = T;

    /// Returns the `i`-th symbol of the indexed sequence, `None` is returned if `i` is out of bound.
    ///
    /// # Examples
    /// ```
    /// use qwt::{QWaveletTreeP256, AccessUnsigned};
    ///
    /// let mut data: [u8; 8] = [1, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWaveletTreeP256::new(&mut data);
    ///
    /// assert_eq!(qwt.get(2), Some(1));
    /// assert_eq!(qwt.get(3), Some(0));
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
    /// use qwt::{QWaveletTreeP256, AccessUnsigned};
    ///
    /// let mut data: [u8; 8] = [1, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWaveletTreeP256::new(&mut data);
    ///
    /// unsafe {
    ///     assert_eq!(qwt.get_unchecked(2), 1);
    ///     assert_eq!(qwt.get_unchecked(3), 0);
    /// }
    /// ```
    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item {
        let mut result = T::zero();

        let mut curr_i = i;
        for level in 0..self.n_levels - 1 {
            // last rank can be saved. ~3% improvement. Indeed, most of the cost is for the cache miss for data access that we pay anyway
            let symbol = self.qv.get_unchecked(curr_i);

            result = (result << 2) | symbol.as_();

            let rank = self.qv.rank_unchecked(symbol, curr_i);

            let j = (level * 4) + symbol as usize;

            curr_i =
                (level + 1) * self.len() + (rank - self.rank[j] as usize) + self.count[j] as usize;
        }

        let symbol = self.qv.get_unchecked(curr_i);
        (result << 2) | symbol.as_()
    }
}

impl<T, RS> SelectUnsigned for QWaveletTree<T, RS>
where
    T: Unsigned + PrimInt + Ord + Shr<usize> + Shl<usize> + AsPrimitive<u8> + std::fmt::Debug,
    u8: AsPrimitive<T>,
    RS: From<QVector>
        + AccessUnsigned<Item = u8>
        + RankUnsigned
        + SelectUnsigned
        + SpaceUsage
        + Default,
{
    /// Returns the position of the `i`-th occurrence of symbol `symbol`, `None` is returned if i is 0 or if there is no such occurrence for the symbol.
    ///
    /// # Examples
    /// ```
    /// use qwt::{QWaveletTreeP256, SelectUnsigned};
    ///
    /// let mut data: [u8; 8] = [1, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWaveletTreeP256::new(&mut data);
    ///
    /// assert_eq!(qwt.select(1, 1), Some(0));
    /// assert_eq!(qwt.select(0, 2), Some(3));
    /// assert_eq!(qwt.select(1, 0), None);
    /// ```    
    #[inline(always)]
    fn select(&self, symbol: Self::Item, i: usize) -> Option<usize> {
        if i == 0 {
            return None;
        }

        let mut path_off = Vec::with_capacity(self.n_levels);
        let mut rank_path_off = Vec::with_capacity(self.n_levels);

        let mut b = 0;
        let mut shift: i64 = 2 * (self.n_levels - 1) as i64;

        for level in 0..self.n_levels {
            path_off.push(b);

            let two_bits = (symbol >> shift as usize).as_() & 3;
            let j = (level * 4) + two_bits as usize;

            let rank_b = self.qv.rank(two_bits, b)?;

            b = (level + 1) * self.n + (rank_b - self.rank[j] as usize) + (self.count[j] as usize);
            shift -= 2;

            rank_path_off.push(rank_b);
        }

        shift = 0;
        let mut result = i;
        for level in (0..self.n_levels).rev() {
            b = path_off[level];
            let rank_b = rank_path_off[level];
            let two_bits = (symbol >> shift as usize).as_() & 3;

            result = self.qv.select(two_bits, rank_b + result)? - b + 1;
            shift += 2;
        }

        Some(result - 1)
    }

    /// Returns the position of the `i`-th occurrence of symbol `symbol`.
    ///
    /// # Safety
    /// Calling this method with a value of `i` which is larger than the number of
    /// occurrences of the `symbol` is undefined behavior.
    /// In the current implementation there is no reason to prefer this unsafe select
    /// over the safe one.
    #[inline(always)]
    unsafe fn select_unchecked(&self, symbol: Self::Item, i: usize) -> usize {
        self.select(symbol, i).unwrap()
    }
}

impl<T, RS> SpaceUsage for QWaveletTree<T, RS>
where
    T: Unsigned + PrimInt + Ord + Shr<usize> + Shl<usize> + AsPrimitive<u8>,
    u8: AsPrimitive<T>,
    RS: From<QVector>
        + AccessUnsigned<Item = u8>
        + RankUnsigned
        + SelectUnsigned
        + SpaceUsage
        + Default,
{
    /// Gives the space usage in bytes of the struct.
    fn space_usage_bytes(&self) -> usize {
        8 + 8
            + self.qv.space_usage_bytes()
            + self.rank.space_usage_bytes()
            + self.count.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests;
