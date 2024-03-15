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
use crate::{AccessUnsigned, RankUnsigned, SelectUnsigned, SpaceUsage, WTSupport};
use crate::{QVector, QVectorBuilder}; // Traits

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

// Traits bound
use num_traits::{AsPrimitive, PrimInt, Unsigned};
use std::ops::{Shl, Shr};

mod prefetch_support;
use crate::quadwt::prefetch_support::PrefetchSupport;

/// Alias for the trait bounds to be satisfied by a data structure
/// to support `rank` and `select` queries at each level of the wavelet tree.
/// We need an alias to avoid repeating a lot of bounds here and there.
pub trait RSforWT: From<QVector> + WTSupport + SpaceUsage + Default {}

// Generic implementation for any T
impl<T> RSforWT for T where T: From<QVector> + WTSupport + SpaceUsage + Default {}

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

/// The generic RS is the data structure we use to index a quaternary
/// sequence to support `access, `rank`, and `select` queries.
///
/// The const generic `PREFETCH_DATA` specifies if the wavelet tree
/// is augmented with extra data to support a deeper level of prefetching.
/// This is needed only for sequences such that data about superblocks and
/// blocks do not fit in L3 cache.
#[derive(Default, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct QWaveletTree<T, RS, const WITH_PREFETCH_SUPPORT: bool = false> {
    n: usize,        // The length of the represented sequence
    n_levels: usize, // The number of levels of the wavelet matrix
    sigma: T, // The largest symbol in the sequence. *NOTE*: It's not +1 because it may overflow
    qvs: Vec<RS>, // A quad vector for each level
    prefetch_support: Option<Vec<PrefetchSupport>>,
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
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
                prefetch_support: None,
            };
        }
        let sigma = *sequence.iter().max().unwrap();
        let log_sigma = msb(sigma) + 1; // Note that sigma equals the largest symbol, so it's already "alphabet_size - 1"
        let n_levels = ((log_sigma + 1) / 2) as usize; // TODO: if log_sigma is odd, the FIRST level should be a binary vector!

        let mut prefetch_support = Vec::with_capacity(n_levels); // used only if WITH_PREFETCH_SUPPORT

        let mut qvs = Vec::<RS>::with_capacity(n_levels);

        let mut shift = 2 * (n_levels - 1);

        for _level in 0..n_levels {
            let mut cur_qv = QVectorBuilder::with_capacity(sequence.len());
            for &symbol in sequence.iter() {
                let two_bits: u8 = (symbol >> shift).as_() & 3; // take the last 2 bits
                cur_qv.push(two_bits);
            }

            let qv = cur_qv.build();

            if WITH_PREFETCH_SUPPORT {
                let pfs = PrefetchSupport::new(&qv, 11); // 11 -> sample_rate = 2048
                prefetch_support.push(pfs);
            }
            qvs.push(RS::from(qv));

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
            prefetch_support: if WITH_PREFETCH_SUPPORT {
                Some(prefetch_support)
            } else {
                None
            },
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
    pub fn iter(
        &self,
    ) -> QWTIterator<T, RS, &QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>, WITH_PREFETCH_SUPPORT>
    {
        QWTIterator {
            i: 0,
            qwt: self,
            _phantom: PhantomData,
        }
    }

    #[inline]
    unsafe fn rank_prefetch_superblocks_unchecked(&self, symbol: T, i: usize) -> usize {
        if !WITH_PREFETCH_SUPPORT {
            return 0;
        }

        if let Some(ref prefetch_support) = self.prefetch_support {
            let mut shift: i64 = (2 * (self.n_levels - 1)) as i64;
            let mut range = 0..i;

            //let mut real_range = 0..i;

            self.qvs[0].prefetch_data(range.end);
            self.qvs[0].prefetch_info(range.start);
            self.qvs[0].prefetch_info(range.end);

            #[allow(clippy::needless_range_loop)]
            for level in 0..self.n_levels - 1 {
                let two_bits: u8 = (symbol >> shift as usize).as_() & 3;

                // SAFETY: Here we are sure that two_bits is a symbol in [0..3]
                let offset = self.qvs[level].occs_smaller_unchecked(two_bits);

                let rank_start =
                    prefetch_support[level].approx_rank_unchecked(two_bits, range.start);
                let rank_end = prefetch_support[level].approx_rank_unchecked(two_bits, range.end);

                range = (rank_start + offset)..(rank_end + offset);
                self.qvs[level + 1].prefetch_info(range.start);
                self.qvs[level + 1].prefetch_info(range.start + 2048);

                self.qvs[level + 1].prefetch_info(range.end);
                self.qvs[level + 1].prefetch_info(range.end + 2048);
                if level > 0 {
                    self.qvs[level + 1].prefetch_info(range.start + 2 * 2048);
                    self.qvs[level + 1].prefetch_info(range.end + 2 * 2048);
                    self.qvs[level + 1].prefetch_info(range.end + 3 * 2048);
                }
                // self.qvs[level + 1].prefetch_info(range.end + 4 * 2048);

                // // CHECK!
                // let rank_start = self.qvs[level].rank_unchecked(two_bits, real_range.start);
                // let rank_end = self.qvs[level].rank_unchecked(two_bits, real_range.end);

                // real_range = (rank_start + offset)..(rank_end + offset);

                // //if range.start > real_range.start || range.end > real_range.end {
                // //     println!("Happen this");
                // // }

                // if range.start / 2048 != real_range.start / 2048
                //     && range.start / 2048 + 1 != real_range.start / 2048
                //     && range.start / 2048 + 2 != real_range.start / 2048
                // {
                //     println!("Level: {}", level);
                //     println!("Real range.start: {:?}", real_range);
                //     println!("Appr range.start: {:?}", range);
                //     println!("real_range.start / 2048:   {}", real_range.start / 2048);
                //     println!("approx range.start / 2048: {}\n", range.start / 2048);
                // }

                // if range.end / 2048 != real_range.end / 2048
                //     && range.end / 2048 + 1 != real_range.end / 2048
                //     && range.end / 2048 + 2 != real_range.end / 2048
                //     && range.end / 2048 + 3 != real_range.end / 2048
                // {
                //     println!("Level: {}", level);
                //     println!("Real range.end: {:?}", real_range);
                //     println!("Appr range.end: {:?}", range);
                //     println!("real_range.end / 2048:   {}", real_range.end / 2048);
                //     println!("approx range.end / 2048: {}\n", range.end / 2048);
                // }

                shift -= 2;
            }

            return range.end - range.start;
        }

        0
    }

    /// Returns rank of `symbol` up to position `i` **excluded**.
    /// `None`, is returned if `i` is out of bound or if `symbol`
    /// is not valid (i.e., it is greater than or equal to the alphabet size).
    ///
    /// Differently from `rank` function, it runs a first phase
    /// in which it estimates the positions in the wavelet tree
    /// needed by rank queries and prefetches these data.
    /// It is faster than the original rank whenever the superblock/block
    /// counters fit in L3 cache but the sequence is larger.
    ///
    /// # Examples
    /// ```
    /// use qwt::{QWT256, RankUnsigned};
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWT256::from(data);
    ///
    /// assert_eq!(qwt.rank_prefetch(1, 2), Some(1));
    /// assert_eq!(qwt.rank_prefetch(3, 8), Some(1));
    /// assert_eq!(qwt.rank_prefetch(1, 0), Some(0));
    /// assert_eq!(qwt.rank_prefetch(1, 9), None);     // too large position
    /// assert_eq!(qwt.rank_prefetch(6, 1), None);     // too large symbol
    /// ```
    #[inline(always)]
    pub fn rank_prefetch(&self, symbol: T, i: usize) -> Option<usize> {
        if i > self.n || symbol > self.sigma {
            return None;
        }

        // SAFETY: Check the above guarantees we are not out of bound
        Some(unsafe { self.rank_prefetch_unchecked(symbol, i) })
    }

    #[inline(always)]
    /// Returns rank of `symbol` up to position `i` **excluded**.
    /// Differently from `rank_unchecked`, it runs a first phase
    /// in which it estimates the positions in the wavelet tree
    /// needed by rank queries and prefetches these data.
    /// It is faster than the original rank whenever the superblock/block
    /// counters fit in L3 cache but the sequence is larger.
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
    ///     assert_eq!(qwt.rank_prefetch_unchecked(1, 2), 1);
    /// }
    /// ```
    pub unsafe fn rank_prefetch_unchecked(&self, symbol: T, i: usize) -> usize {
        if WITH_PREFETCH_SUPPORT {
            let _ = self.rank_prefetch_superblocks_unchecked(symbol, i);
        }

        let mut range = 0..i;
        let mut shift: i64 = (2 * (self.n_levels - 1)) as i64;

        const BLOCK_SIZE: usize = 256; // TODO: fix me!

        //let mut real_range = 0..i;

        self.qvs[0].prefetch_data(range.start);
        self.qvs[0].prefetch_data(range.end);
        for level in 0..self.n_levels - 1 {
            let two_bits: u8 = (symbol >> shift as usize).as_() & 3;

            // SAFETY: Here we are sure that two_bits is a symbol in [0..3]
            let offset = self.qvs[level].occs_smaller_unchecked(two_bits);

            let rank_start = self.qvs[level].rank_block_unchecked(two_bits, range.start);
            let rank_end = self.qvs[level].rank_block_unchecked(two_bits, range.end);

            range = (rank_start + offset)..(rank_end + offset);

            // The estimated position can be off by BLOCK_SIZE for every level

            self.qvs[level + 1].prefetch_data(range.start);
            self.qvs[level + 1].prefetch_data(range.start + BLOCK_SIZE);

            self.qvs[level + 1].prefetch_data(range.end);
            self.qvs[level + 1].prefetch_data(range.end + BLOCK_SIZE);
            for i in 0..level {
                self.qvs[level + 1].prefetch_data(range.end + 2 * BLOCK_SIZE + i * BLOCK_SIZE);
            }

            // // CHECK!
            // let rank_start = self.qvs[level].rank_unchecked(two_bits, real_range.start);
            // let rank_end = self.qvs[level].rank_unchecked(two_bits, real_range.end);

            // real_range = (rank_start + offset)..(rank_end + offset);

            // //if range.start > real_range.start || range.end > real_range.end {
            // //     // THIS NEVER HAPPEN!
            // // }

            // if range.start / 256 != real_range.start / 256
            //     && range.start / 256 + 1 != real_range.start / 256
            // {
            //     println!("Level: {}", level);
            //     println!("Real range.start: {:?}", real_range);
            //     println!("Appr range.start: {:?}", range);
            //     println!("real_range.start / 256:   {}", real_range.start / 256);
            //     println!("approx range.start / 256: {}\n", range.start / 256);
            // }

            // if !(range.end / 256 <= real_range.end / 256
            //     && range.end / 256 + level + 1 >= real_range.end / 256)
            // {
            //     println!("{}", range.end / 256 <= real_range.end / 256);
            //     println!("{}", range.end / 256 + level + 1 >= real_range.end / 256);
            //     println!("{}", range.end / 256 + level >= real_range.end / 256);
            //     println!("{}", range.end / 256 <= real_range.end / 256);
            //     println!("Level: {}", level);
            //     println!("Real range.end: {:?}", real_range);
            //     println!("Appr range.end: {:?}", range);
            //     println!("real_range.end / 256:   {}", real_range.end / 256);
            //     println!("approx range.end / 256: {}\n", range.end / 256);
            // }

            shift -= 2;
        }
        self.rank_unchecked(symbol, i)
    }
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> RankUnsigned
    for QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
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

        // SAFETY: Check above guarantees we are not out of bound
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

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> AccessUnsigned
    for QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
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
        // SAFETY: check before guarantees we are not out of bound
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

            self.qvs[level].prefetch_info(cur_i); // Compiler is not able to infer that later it is needed for the rank query. Access is roughly 33% slower for large files without this.
            let symbol = self.qvs[level].get_unchecked(cur_i);
            result = (result << 2) | symbol.as_();

            // SAFETY: Here we are sure that symbol is in [0..3]
            let offset = unsafe { self.qvs[level].occs_smaller_unchecked(symbol) };
            cur_i = self.qvs[level].rank_unchecked(symbol, cur_i) + offset;
        }

        let symbol = self.qvs[self.n_levels - 1].get_unchecked(cur_i);
        (result << 2) | symbol.as_()
    }
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> SelectUnsigned
    for QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
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
    /// assert_eq!(qwt.select(1, 1), Some(2));
    /// assert_eq!(qwt.select(0, 1), Some(3));
    /// assert_eq!(qwt.select(0, 2), None);
    /// assert_eq!(qwt.select(1, 0), Some(0));
    /// assert_eq!(qwt.select(6, 1), None);
    /// ```    
    #[inline(always)]
    fn select(&self, symbol: Self::Item, i: usize) -> Option<usize> {
        if symbol > self.sigma {
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

            result = self.qvs[level].select(two_bits, rank_b + result)? - b;
            shift += 2;
        }

        Some(result)
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

impl<T, RS: SpaceUsage, const WITH_PREFETCH_SUPPORT: bool> SpaceUsage
    for QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
{
    /// Gives the space usage in bytes of the struct.
    fn space_usage_byte(&self) -> usize {
        let space_prefetch_support: usize = self
            .prefetch_support
            .iter()
            .flatten()
            .map(|ps| ps.space_usage_byte())
            .sum();

        8 + 8
            + self
                .qvs
                .iter()
                .fold(0, |acc, ds| acc + ds.space_usage_byte())
            + space_prefetch_support
    }
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> AsRef<QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>>
    for QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
{
    fn as_ref(&self) -> &QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT> {
        self
    }
}

// This is a naive implementation of an iterator for WT.
// We could do better by storing more information and
// avoid rank operations!
#[derive(Debug, PartialEq)]
pub struct QWTIterator<
    T,
    RS,
    Q: AsRef<QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>>,
    const WITH_PREFETCH_SUPPORT: bool = false,
> {
    i: usize,
    qwt: Q,
    _phantom: PhantomData<(T, RS)>,
}

impl<
        T,
        RS,
        Q: AsRef<QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>>,
        const WITH_PREFETCH_SUPPORT: bool,
    > Iterator for QWTIterator<T, RS, Q, WITH_PREFETCH_SUPPORT>
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

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> IntoIterator
    for QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    u8: AsPrimitive<T>,
    RS: RSforWT,
{
    type IntoIter =
        QWTIterator<T, RS, QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>, WITH_PREFETCH_SUPPORT>;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        QWTIterator {
            i: 0,
            qwt: self,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T, RS, const WITH_PREFETCH_SUPPORT: bool> IntoIterator
    for &'a QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    u8: AsPrimitive<T>,
    RS: RSforWT,
{
    type IntoIter =
        QWTIterator<T, RS, &'a QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>, WITH_PREFETCH_SUPPORT>;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> FromIterator<T>
    for QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
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

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> From<Vec<T>>
    for QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
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
