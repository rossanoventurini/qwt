//! # Quad Wavelet Tree
//!
//! This module implements a Quad Wavelet Tree, providing efficient implementations of [`AccessUnsigned`], [`RankUnsigned`], and [`SelectUnsigned`] for a vector of unsigned integers.
//!
//! The Quad Wavelet Tree indexes a sequence of unsigned integers and facilitates the following three operations:
//!
//! - `get(i)`: Accesses the `i`-th symbol of the indexed sequence.
//! - `rank(s, i)`: Counts the number of occurrences of symbol `s` up to position `i`, excluding `i`.
//! - `select(s, i)`: Returns the position of the `i+1`-th occurrence of symbol `s`.
//!
//! We have four aliases types for a Quad Wavelet Tree: `QWT256<T>`, `QWT512<T>`, `QWT256Pfs<T>`, and `QWT512Pfs<T>`. The generic type `T` is the type of the indexed unsigned integer values.
//! The values 256 and 512 are the employed block sizes in the internal representation.
//! A block size of 256 has a faster query time at the cost of slightly larger space overhead.
//! The prefix `Pfs` indicates that the wavelet tree uses additional data structures to speed up the `rank` queries with prefetching. Refer to the paper for more details.
//!
//! ## Performance
//!
//! All operations run in $$\Theta(\log \sigma)$$ time, where $$\sigma$$ is the alphabet size, i.e., one plus the largest symbol in the sequence. The space usage is $$n \log \sigma + o(n \log \sigma )$$ bits.
//!
//! To optimize query time and space usage, it's advisable to compact the alphabet and remove "holes," if any.
//!
//! ## Limitations
//!
//! This data structure can efficiently index vectors of lengths up to 2^{43} symbols.
//!
//! ## Examples
//!
//! ```rust
//! use qwt::{QWT256Pfs,AccessUnsigned, RankUnsigned, SelectUnsigned};
//!
//! // Example usage of Quad Wavelet Tree
//! // Constructing a Qwt256Pfs for u32 integers
//! let qwt = QWT256Pfs::from(vec![1_u32, 2, 3, 4, 5, 6, 7, 8]);
//!
//! // Querying operations
//! assert_eq!(qwt.get(3), Some(4));  // Accesses the 3rd symbol (0-indexed), should return 4
//! assert_eq!(qwt.rank(3, 7), Some(1));  // Counts the occurrences of symbol 3 up to position 7, should return 1
//! assert_eq!(qwt.select(3, 0), Some(2));  // Finds the position of the 1st occurrence of symbol 3, should return Some(2)
//! ```

use crate::utils::{msb, stable_partition_of_4};
use crate::{AccessUnsigned, RankUnsigned, SelectUnsigned, SpaceUsage, WTIterator, WTSupport};
use crate::{QVector, QVectorBuilder}; // Traits

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

// Traits bound
use num_traits::{AsPrimitive, PrimInt, Unsigned};
use std::ops::{Shl, Shr};

pub mod huffqwt;

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
    Unsigned + PrimInt + Ord + Shr<usize> + Shl<usize> + AsPrimitive<usize>
{
}

impl<T> WTIndexable for T
where
    T: Unsigned + PrimInt + Ord + Shr<usize> + Shl<usize> + AsPrimitive<usize> + AsPrimitive<u8>,
    u8: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
}

/// The generic RS is the data structure we use to index a quaternary
/// sequence to support `access, `rank`, and `select` queries.
///
/// The const generic `PREFETCH_DATA` specifies if the wavelet tree
/// is augmented with extra data to support a deeper level of prefetching.
/// This extra informationa are needed only for sequences such that data
/// about superblocks and blocks do not fit in L3 cache.
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
    usize: AsPrimitive<T>,
    RS: RSforWT,
{
    /// Builds the wavelet tree of the `sequence` of unsigned integers.
    /// The input `sequence` will be **destroyed**.
    ///
    /// The alphabet size `sigma` is the largest value in the `sequence`.
    /// Both space usage and query time of a QWaveletTree depend on
    /// $$\lfloor\log_2 (\sigma-1)\rfloor + 1$$ (i.e., the length of the
    /// binary representation of values in the sequence).
    /// For this reason, it may be convenient for both space usage and query time to
    /// remap the alphabet to form a consecutive range [0, d], where d is
    /// the number of distinct values in `sequence`.
    ///
    /// ## Panics
    /// Panics if the sequence is longer than the largest possible length.
    /// The largest possible length is 2^{43} symbols.
    ///
    /// # Examples
    /// ```
    /// use qwt::QWT256;
    ///
    /// let mut data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWT256::new(&mut data);
    ///
    /// assert_eq!(qwt.len(), 8);
    /// ```
    #[must_use]
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
                let two_bits: u8 = ((symbol >> shift).as_() & 3) as u8; // take the last 2 bits
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
    ///
    /// ```
    /// use qwt::QWT256;
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWT256::from(data);
    ///
    /// assert_eq!(qwt.len(), 8);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns the largest value in the sequence, or `None` if the sequence is empty.
    ///
    /// Note: that for us sigma is the largest value and not the largest value  plus 1 as common
    /// because the latter may overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::QWT256;
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = QWT256::from(data);
    ///
    /// assert_eq!(qwt.sigma(), Some(5));
    /// ```
    #[must_use]
    pub fn sigma(&self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            Some(self.sigma)
        }
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
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    /// Returns the number of levels in the wavelet tree.
    ///
    /// The number of levels represents the depth of the wavelet tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::QWT256;
    ///
    /// let data = vec![1u8, 0, 1, 0, 255, 4, 5, 3];
    ///
    /// let qwt = QWT256::from(data);
    ///
    /// assert_eq!(qwt.n_levels(), 4);
    /// ```
    #[must_use]
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
    /// let qwt = QWT256::from(data.clone());
    ///
    /// assert_eq!(qwt.iter().collect::<Vec<_>>(), data);
    ///
    /// assert_eq!(qwt.iter().rev().collect::<Vec<_>>(), data.into_iter().rev().collect::<Vec<_>>());
    /// ```
    pub fn iter(
        &self,
    ) -> WTIterator<
        T,
        QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>,
        &QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>,
    > {
        WTIterator {
            i: 0,
            end: self.len(),
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

            self.qvs[0].prefetch_data(range.end);
            self.qvs[0].prefetch_info(range.start);
            self.qvs[0].prefetch_info(range.end);

            #[allow(clippy::needless_range_loop)]
            for level in 0..self.n_levels - 1 {
                let two_bits: u8 = ((symbol >> shift as usize).as_() & 3) as u8;

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

    /// Returns the rank of `symbol` up to position `i` **excluded**.
    ///
    /// `None` is returned if `i` is out of bound or if `symbol` is not valid
    /// (i.e., it is greater than or equal to the alphabet size).
    ///
    /// Differently from the `rank` function, `rank_prefetch` runs a first phase
    /// in which it estimates the positions in the wavelet tree needed by rank queries
    /// and prefetches these data. It is faster than the original `rank` function whenever
    /// the superblock/block counters fit in L3 cache but the sequence is larger.
    ///
    /// # Examples
    ///
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
    /// assert_eq!(qwt.rank_prefetch(1, 9), None);  // Too large position
    /// assert_eq!(qwt.rank_prefetch(6, 1), None);  // Too large symbol
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn rank_prefetch(&self, symbol: T, i: usize) -> Option<usize> {
        if i > self.n || symbol > self.sigma {
            return None;
        }

        // SAFETY: Check the above guarantees we are not out of bound
        Some(unsafe { self.rank_prefetch_unchecked(symbol, i) })
    }

    /// Returns the rank of `symbol` up to position `i` **excluded**.
    ///
    /// Differently from the `rank_unchecked` function, `rank_prefetch` runs a first phase
    /// in which it estimates the positions in the wavelet tree needed by rank queries
    /// and prefetches these data. It is faster than the original `rank` function whenever
    /// the superblock/block counters fit in L3 cache but the sequence is larger.
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
    #[must_use]
    #[inline(always)]
    pub unsafe fn rank_prefetch_unchecked(&self, symbol: T, i: usize) -> usize {
        if WITH_PREFETCH_SUPPORT {
            let _ = self.rank_prefetch_superblocks_unchecked(symbol, i);
        }

        let mut range = 0..i;
        let mut shift: i64 = (2 * (self.n_levels - 1)) as i64;

        const BLOCK_SIZE: usize = 256; // TODO: fix me!

        self.qvs[0].prefetch_data(range.start);
        self.qvs[0].prefetch_data(range.end);
        for level in 0..self.n_levels - 1 {
            let two_bits: u8 = ((symbol >> shift as usize).as_() & 3) as u8;

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
    usize: AsPrimitive<T>,
    RS: RSforWT,
{
    /// Returns the rank of `symbol` up to position `i` **excluded**.
    ///
    /// `None` is returned if `i` is out of bound or if `symbol` is not valid
    /// (i.e., it is greater than or equal to the alphabet size).
    ///
    /// # Examples
    ///
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
    /// assert_eq!(qwt.rank(1, 9), None);  // Too large position
    /// assert_eq!(qwt.rank(6, 1), None);  // Too large symbol
    /// ```
    #[must_use]
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
    ///
    /// Calling this method with a position `i` larger than the size of the sequence
    /// or with an invalid symbol is undefined behavior.
    ///
    /// Users must ensure that the position `i` is within the bounds of the sequence
    /// and that the symbol is valid.
    ///
    /// # Examples
    ///
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
    #[must_use]
    #[inline(always)]
    unsafe fn rank_unchecked(&self, symbol: Self::Item, i: usize) -> usize {
        let mut shift: i64 = (2 * (self.n_levels - 1)) as i64;
        let mut cur_i = i;
        let mut cur_p = 0;

        for level in 0..self.n_levels - 1 {
            let two_bits: u8 = ((symbol >> shift as usize).as_() & 3) as u8;

            // Safety: Here we are sure that two_bits is a symbol in [0..3]
            let offset = unsafe { self.qvs[level].occs_smaller_unchecked(two_bits) };
            cur_p = self.qvs[level].rank_unchecked(two_bits, cur_p) + offset;
            cur_i = self.qvs[level].rank_unchecked(two_bits, cur_i) + offset;

            shift -= 2;
        }

        let two_bits: u8 = ((symbol >> shift as usize).as_() & 3) as u8;

        cur_i = self.qvs[self.n_levels - 1].rank_unchecked(two_bits, cur_i);
        cur_p = self.qvs[self.n_levels - 1].rank_unchecked(two_bits, cur_p);

        cur_i - cur_p
    }
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> AccessUnsigned
    for QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    RS: RSforWT,
{
    type Item = T;

    /// Returns the `i`-th symbol of the indexed sequence.
    ///
    /// `None` is returned if `i` is out of bound.
    ///
    /// # Examples
    ///
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
    #[must_use]
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
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    ///
    /// Users must ensure that the index `i` is within the bounds of the sequence.
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
    #[must_use]
    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item {
        let mut result = T::zero();

        let mut cur_i = i;
        for level in 0..self.n_levels - 1 {
            // The last rank can be saved. The improvement is just ~3%. Indeed, most of the cost is for the cache miss for data access that we pay anyway

            self.qvs[level].prefetch_info(cur_i); // Compiler is not able to infer that later it is needed for the rank query. Access is roughly 33% slower for large files without this.
            let symbol = self.qvs[level].get_unchecked(cur_i);
            result = (result << 2) | (symbol as usize).as_();

            // SAFETY: Here we are sure that symbol is in [0..3]
            let offset = unsafe { self.qvs[level].occs_smaller_unchecked(symbol) };
            cur_i = self.qvs[level].rank_unchecked(symbol, cur_i) + offset;
        }

        let symbol = self.qvs[self.n_levels - 1].get_unchecked(cur_i);
        (result << 2) | (symbol as usize).as_()
    }
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> SelectUnsigned
    for QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    RS: RSforWT,
{
    /// Returns the position of the `i+1`-th occurrence of symbol `symbol`.
    ///
    /// `None` is returned if the is no (i+1)th such occurrence for the symbol
    /// or if `symbol` is not valid (i.e., it is greater than or equal to the alphabet size).
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
    /// assert_eq!(qwt.select(5, 0), Some(6));
    /// assert_eq!(qwt.select(6, 1), None);
    /// ```    
    #[must_use]
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

            let rank_b = self.qvs[level].rank(two_bits as u8, b)?;

            // Safety: we are sure the symbol `two_bits` is in [0..3]
            b = rank_b + unsafe { self.qvs[level].occs_smaller_unchecked(two_bits as u8) };
            shift -= 2;

            rank_path_off.push(rank_b);
        }

        shift = 0;
        let mut result = i;
        for level in (0..self.n_levels).rev() {
            b = path_off[level];
            let rank_b = rank_path_off[level];
            let two_bits = (symbol >> shift as usize).as_() & 3;

            result = self.qvs[level].select(two_bits as u8, rank_b + result)? - b;
            shift += 2;
        }

        Some(result)
    }

    /// Returns the position of the `i+1`-th occurrence of symbol `symbol`.
    ///
    /// # Safety
    ///
    /// Calling this method with a value of `i` larger than the number of occurrences
    /// of the `symbol`, or if the `symbol` is not valid, is undefined behavior.
    ///
    /// In the current implementation, there is no efficiency reason to prefer this
    /// unsafe `select` over the safe one.
    #[must_use]
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

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> IntoIterator
    for QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    RS: RSforWT,
{
    type IntoIter = WTIterator<
        T,
        QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>,
        QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>,
    >;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        WTIterator {
            i: 0,
            end: self.len(),
            qwt: self,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T, RS, const WITH_PREFETCH_SUPPORT: bool> IntoIterator
    for &'a QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    RS: RSforWT,
{
    type IntoIter = WTIterator<
        T,
        QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>,
        &'a QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>,
    >;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> FromIterator<T>
    for QWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
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
    usize: AsPrimitive<T>,
    RS: RSforWT,
{
    fn from(mut v: Vec<T>) -> Self {
        QWaveletTree::new(&mut v[..])
    }
}

#[cfg(test)]
mod tests;
