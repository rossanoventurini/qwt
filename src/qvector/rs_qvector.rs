//! This module provides support for `rank` and `select` queries on a quad vector.

use super::{QVector, QVectorIterator};

use crate::utils::{prefetch_read_NTA, select_in_word_u128};

use num_traits::int::PrimInt;
use num_traits::{AsPrimitive, Unsigned};

use serde::{Deserialize, Serialize};

// Traits
use crate::{AccessQuad, RankQuad, SelectQuad, SpaceUsage, WTSupport};

/// Alternative representations to support Rank/Select queries at the level of blocks
mod rs_support_plain;
use crate::qvector::rs_qvector::rs_support_plain::RSSupportPlain;

/// Possible specializations which provide different space/time trade-offs.
pub type RSQVector256 = RSQVector<RSSupportPlain<256>>;
pub type RSQVector512 = RSQVector<RSSupportPlain<512>>;

/// The generic `S` is the data structure used to provide rank/select
/// support at the level of blocks.
#[derive(Default, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct RSQVector<S> {
    qv: QVector,
    rs_support: S,
    n_occs_smaller: [usize; 5], // for each symbol c, store the number of occurrences of in qv of symbols smaller than c. We store 5 (instead of 4) counters so we can use them to compute also the number of occurrences of each symbol without branches.
}

impl<S> RSQVector<S> {
    /// Returns an iterator over the values in the quad vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::RSQVector256;
    ///
    /// let rsqv: RSQVector256 = (0..10_u64).into_iter().map(|x| x % 4).collect();
    ///
    /// for (i, v) in rsqv.iter().enumerate() {
    ///    assert_eq!((i%4) as u8, v);
    /// }
    /// ```
    pub fn iter(&self) -> QVectorIterator<&QVector> {
        self.qv.iter()
    }
}

impl<S: SpaceUsage> SpaceUsage for RSQVector<S> {
    /// Gives the space usage in bytes of the data structure.
    fn space_usage_byte(&self) -> usize {
        self.qv.space_usage_byte() + self.rs_support.space_usage_byte() + 5 * 8
    }
}

impl<S: RSSupport> From<QVector> for RSQVector<S> {
    /// Converts a given quad vector `qv` into a `RSQVector` with support
    /// for `rank` and `select` queries.
    ///
    /// # Examples
    /// ```
    /// use qwt::RSQVector256;
    ///
    /// let rsqv: RSQVector256 = (0..10_u64).into_iter().map(|x| x % 4).collect();
    ///
    /// assert_eq!(rsqv.is_empty(), false);
    /// assert_eq!(rsqv.len(), 10);
    /// ```
    fn from(qv: QVector) -> Self {
        let rank_support = S::new(&qv);
        let mut n_occs_smaller = [0; 5];
        for c in qv.iter() {
            n_occs_smaller[c as usize] += 1;
        }

        let mut prev = n_occs_smaller[0];
        n_occs_smaller[0] = 0;
        for i in 1..5 {
            let tmp = n_occs_smaller[i];
            n_occs_smaller[i] = n_occs_smaller[i - 1] + prev;
            prev = tmp;
        }

        Self {
            qv,
            rs_support: rank_support,
            n_occs_smaller,
        }
    }
}

impl<S: RSSupport> RSQVector<S> {
    /// Creates a quad vector with support for `RankQuad` and `SelectQuad`
    /// queries for a sequence of integers in the range [0, 3].
    ///
    /// # Panics
    /// Panics if the vector is longer than the largest possible length 2^{43}-1 symbols.
    /// If you need longer vectors, consider using a vector of `RSQVector`s,
    /// each of length smaller than 2^{43} symbols.
    /// This limit is due to two different reasons:
    ///     - The number of occurrences of a symbol up to the beginning of its superblock is stored in 44 bits.
    ///     - A select sample stores a superblock id. The size of a superblock is at least 2048 symbols, thus a superblock id for a sequence of length 2^{43}-1 fits in 32 bits.
    ///
    /// # Examples
    /// ```
    /// use qwt::RSQVector256;
    ///
    /// let v: Vec<u64> = (0..10).into_iter().map(|x| x % 4).collect();
    /// let rsqv = RSQVector256::new(&v);
    ///
    /// assert_eq!(rsqv.is_empty(), false);
    /// assert_eq!(rsqv.len(), 10);
    /// ```
    pub fn new<T>(v: &[T]) -> Self
    where
        T: Unsigned + Copy,
        QVector: FromIterator<T>,
    {
        let qv: QVector = v.iter().copied().collect();
        Self::from(qv)
    }

    #[inline]
    fn select_intra_block(&self, symbol: u8, i: usize, pos: usize) -> usize {
        let line_id = pos >> 8;
        let mut result = 0;
        let mut i = i - 1;

        for j in 0..if S::BLOCK_SIZE == 256 { 1 } else { 2 } {
            // May need two iterations for blocks of size 512
            let (word_0, word_1) =
                unsafe { self.qv.data.get_unchecked(line_id + j).normalize(symbol) };

            let cnt_0 = word_0.count_ones() as usize;
            if cnt_0 > i {
                let p = select_in_word_u128(word_0, i as u64) as usize;
                return result + p;
            } else {
                i -= cnt_0;
                result += 128;
            }

            let cnt_1 = word_1.count_ones() as usize;
            if cnt_1 > i {
                return result + select_in_word_u128(word_1, i as u64) as usize;
            } else {
                i -= cnt_1;
                result += 128;
            }
        }
        0
    }

    #[inline]
    fn rank_intra_block(&self, symbol: u8, i: usize) -> usize {
        debug_assert!(
            symbol <= 3,
            "RSQVector indexes only four symbols in [0, 3]."
        );

        debug_assert!(
            S::BLOCK_SIZE == 256 || S::BLOCK_SIZE == 512,
            "RSQVector supports only blocks of size 256 or 512."
        );

        if S::BLOCK_SIZE == 256 {
            let data_line_id = i >> 8;
            let offset = i & 255;

            let rank = if let Some(d) = self.qv.data.get(data_line_id) {
                unsafe { d.rank_unchecked(symbol, offset) }
            } else {
                0
            };

            return rank;
            // dbg!(self.qv.data.len());

            // return unsafe {
            //     self.qv
            //         .data //[data_line_id]
            //         .get_unchecked(data_line_id)
            //         .rank_unchecked(symbol, offset)
            // };
        }

        if S::BLOCK_SIZE == 512 {
            let block_id = i >> 9;
            let offset_in_block = i & 511;

            let offset_in_first_block = if offset_in_block <= 256 {
                offset_in_block
            } else {
                256
            };

            let mut rank = if let Some(d) = self.qv.data.get(block_id * 2) {
                unsafe { d.rank_unchecked(symbol, offset_in_first_block) }
            } else {
                0
            };

            // let mut rank = unsafe {
            //     self.qv
            //         .data
            //         .get_unchecked(block_id * 2)
            //         .rank_unchecked(symbol, offset_in_first_block)
            // };

            if offset_in_block > 256 {
                rank += if let Some(d) = self.qv.data.get(block_id * 2 + 1) {
                    unsafe { d.rank_unchecked(symbol, offset_in_block - 256) }
                } else {
                    0
                };

                // rank += unsafe {
                //     self.qv
                //         .data
                //         .get_unchecked(block_id * 2 + 1)
                //         .rank_unchecked(symbol, offset_in_block - 256)
                // };
            }

            return rank;
        }

        0
    }

    // Returns the number of symbols in the quad vector.
    pub fn len(&self) -> usize {
        self.qv.len()
    }

    /// Checks if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.qv.len() == 0
    }
}

impl<S> AccessQuad for RSQVector<S> {
    /// Accesses the `i`-th value in the quad vector.
    /// The caller must guarantee that the position `i` is valid.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    ///
    /// # Examples
    /// ```
    /// use qwt::{RSQVector256, AccessQuad};
    ///
    /// let rsqv: RSQVector256 = (0..10_u64).into_iter().map(|x| x % 4).collect();
    ///
    /// assert_eq!(unsafe { rsqv.get_unchecked(0) }, 0);
    /// assert_eq!(unsafe { rsqv.get_unchecked(1) }, 1);
    /// ```
    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> u8 {
        self.qv.get_unchecked(i)
    }

    /// Accesses the `i`th value in the quad vector
    /// or `None` if out of bounds.
    ///
    /// # Examples
    /// ```
    /// use qwt::{RSQVector256, AccessQuad};
    ///
    /// let rsqv: RSQVector256 = (0..10_u64).into_iter().map(|x| x % 4).collect();
    ///
    /// assert_eq!(rsqv.get(0), Some(0));
    /// assert_eq!(rsqv.get(1), Some(1));
    /// assert_eq!(rsqv.get(10), None);
    /// ```
    #[inline]
    fn get(&self, i: usize) -> Option<u8> {
        self.qv.get(i)
    }
}

impl<S: RSSupport> RankQuad for RSQVector<S> {
    /// Returns rank of `symbol` up to position `i` **excluded**.
    /// Returns `None` if out of bounds.
    ///
    /// # Examples
    /// ```
    /// use qwt::{RSQVector256, RankQuad};
    ///
    /// let rsqv: RSQVector256 = (0..10_u64).into_iter().map(|x| x % 4).collect();
    ///
    /// assert_eq!(rsqv.rank(0, 0), Some(0));
    /// assert_eq!(rsqv.rank(0, 1), Some(1));
    /// assert_eq!(rsqv.rank(0, 2), Some(1));
    /// assert_eq!(rsqv.rank(0, 10), Some(3));
    /// assert_eq!(rsqv.rank(0, 11), None);
    /// ```
    #[inline(always)]
    fn rank(&self, symbol: u8, i: usize) -> Option<usize> {
        if i > self.qv.len() {
            return None;
        }
        // Safety: The check above guarantees we are not out of bound
        Some(unsafe { self.rank_unchecked(symbol, i) })
    }

    /// Returns rank of `symbol` up to position `i` **excluded**.
    ///
    /// # Safety
    /// Calling this method with a position `i` larger than the length of the vector
    /// is undefined behavior.
    #[inline(always)]
    unsafe fn rank_unchecked(&self, symbol: u8, i: usize) -> usize {
        debug_assert!(symbol <= 3);
        self.rs_support.rank_block(symbol, i) + self.rank_intra_block(symbol, i)
    }
}

impl<S: RSSupport> SelectQuad for RSQVector<S> {
    /// Returns the position of the `i+1`th occurrence of `symbol`, meaning
    /// the position `pos` such that `rank(symbol, pos) = i`
    /// Returns `None` if i is not valid, i.e., if i is larger than
    /// the number of occurrences of `symbol`, or if `symbol` is not in [0..3].
    ///
    /// # Examples
    /// ```
    /// use qwt::{RSQVector256, SelectQuad};
    ///
    /// let rsqv: RSQVector256 = (0..10_u64).into_iter().map(|x| x % 4).collect();
    ///
    /// assert_eq!(rsqv.select(0, 0), Some(0));
    /// assert_eq!(rsqv.select(0, 1), Some(4));
    /// assert_eq!(rsqv.select(0, 2), Some(8));
    /// assert_eq!(rsqv.select(0, 3), None);
    /// ```
    #[inline]
    fn select(&self, symbol: u8, i: usize) -> Option<usize> {
        if symbol > 3 || unsafe { self.occs_unchecked(symbol) } <= i {
            return None;
        }

        let (mut pos, rank) = self.rs_support.select_block(symbol, i + 1);

        // if rank == i {
        //     return Some(pos);
        // }

        pos += self.select_intra_block(symbol, i - rank + 1, pos);

        Some(pos)
    }

    /// Returns the position of the `i+1`th occurrence of `symbol`.
    ///
    /// # Safety
    /// Calling this method with a value of `i` which is larger than the number of
    /// occurrences of the `symbol` or if `symbol is larger than 3 is  
    /// undefined behavior.
    ///
    /// In the current implementation there is no reason to prefer this unsafe select
    /// over the safe one.
    #[inline]
    unsafe fn select_unchecked(&self, symbol: u8, i: usize) -> usize {
        debug_assert!(symbol <= 3);
        debug_assert!(i > 0);
        debug_assert!(self.occs(symbol) <= Some(i));

        self.select(symbol, i).unwrap()
    }
}

impl<S: RSSupport> WTSupport for RSQVector<S> {
    /// Returns the number of occurrences of `symbol` in the indexed sequence,
    /// `None` if `symbol` is not in [0..3].  
    #[inline(always)]
    fn occs(&self, symbol: u8) -> Option<usize> {
        if symbol > 3 {
            return None;
        }

        Some(unsafe { self.occs_unchecked(symbol) })
    }

    /// Returns the number of occurrences of `symbol` in the indexed sequence.
    ///
    /// # Safety
    /// Calling this method if the `symbol` is not in [0..3] is undefined behavior.
    #[inline(always)]
    unsafe fn occs_unchecked(&self, symbol: u8) -> usize {
        debug_assert!(symbol <= 3, "Symbols are in [0, 3].");

        self.n_occs_smaller[(symbol + 1) as usize] - self.n_occs_smaller[symbol as usize]
    }

    /// Returns the number of occurrences of all the symbols smaller than the input
    /// `symbol`, `None` if `symbol` is not in [0..3].
    #[inline(always)]
    fn occs_smaller(&self, symbol: u8) -> Option<usize> {
        if symbol > 3 {
            return None;
        }
        Some(unsafe { self.occs_smaller_unchecked(symbol) })
    }

    /// Returns the number of occurrences of all the symbols smaller than the input
    /// `symbol` in the indexed sequence.
    ///
    /// # Safety
    /// Calling this method if the `symbol` is not in [0..3] is undefined behavior.
    #[inline(always)]
    unsafe fn occs_smaller_unchecked(&self, symbol: u8) -> usize {
        debug_assert!(symbol <= 3, "Symbols are in [0, 3].");

        self.n_occs_smaller[symbol as usize]
    }

    /// Returns the rank of `symbol` up to the block that contains the position
    /// `i`.
    ///
    /// # Safety
    /// Calling this method if the `symbol` is larger than 3 of
    /// if the position `i` is out of bound is undefined behavior.
    #[inline(always)]
    unsafe fn rank_block_unchecked(&self, symbol: u8, i: usize) -> usize {
        self.rs_support.rank_block(symbol, i)
    }

    /// Prefetches counters of the superblock and blocks containing the position `pos`.
    #[inline(always)]
    fn prefetch_info(&self, pos: usize) {
        self.rs_support.prefetch(pos)
    }

    /// Prefetches data containing the position `pos`.
    #[inline(always)]
    fn prefetch_data(&self, pos: usize) {
        let line_id = pos >> 8;

        prefetch_read_NTA(&self.qv.data, line_id);
        if S::BLOCK_SIZE == 512 {
            prefetch_read_NTA(&self.qv.data, if line_id > 0 { line_id - 1 } else { 0 });
        }
    }
}

impl<S> AsRef<RSQVector<S>> for RSQVector<S> {
    fn as_ref(&self) -> &RSQVector<S> {
        self
    }
}

impl<S> IntoIterator for RSQVector<S> {
    type IntoIter = QVectorIterator<QVector>;
    type Item = u8;

    fn into_iter(self) -> Self::IntoIter {
        self.qv.into_iter()
    }
}

impl<'a, S> IntoIterator for &'a RSQVector<S> {
    type IntoIter = QVectorIterator<&'a QVector>;
    type Item = u8;

    fn into_iter(self) -> Self::IntoIter {
        self.qv.iter()
    }
}

/// This trait should be implemented by any data structure that
/// provides `rank` and `select` support on blocks.
pub trait RSSupport {
    const BLOCK_SIZE: usize;

    fn new(qv: &QVector) -> Self;

    /// Returns the number of occurrences of `SYMBOL` up to the beginning
    /// of the block that contains position `i`.
    ///
    /// We use a const generic to have a specialized method for each symbol.
    fn rank_block(&self, symbol: u8, i: usize) -> usize;

    /// Returns a pair `(position, rank)` where the position is the beginning of the block
    /// that contains the `i`th occurrence of `symbol`, and `rank` is the number of
    /// occurrences of `symbol` up to the beginning of this block.
    fn select_block(&self, symbol: u8, i: usize) -> (usize, usize);

    fn prefetch(&self, pos: usize);
}

impl<T, S: RSSupport> FromIterator<T> for RSQVector<S>
where
    T: PrimInt + AsPrimitive<u8>,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self::from(QVector::from_iter(iter))
    }
}

#[cfg(test)]
#[generic_tests::define]
mod tests {
    use super::*;
    use std::iter;

    #[test]
    fn test_just_one_data_line<D>()
    where
        D: From<QVector> + AccessQuad + RankQuad + SelectQuad + WTSupport,
    {
        let qv: QVector = [0].into_iter().cycle().take(256).collect(); // a full DataLine
        let rsqv = D::from(qv);

        assert_eq!(rsqv.rank(0, 256), Some(256));
        assert_eq!(rsqv.rank(1, 256), Some(0));
        assert_eq!(rsqv.rank(2, 256), Some(0));
        assert_eq!(rsqv.rank(3, 256), Some(0));
    }

    #[test]
    fn test_small<D>()
    where
        D: From<QVector> + AccessQuad + RankQuad + SelectQuad + WTSupport,
    {
        let qv: QVector = [0, 1, 2, 3].into_iter().cycle().take(10000).collect();
        let rsqv = D::from(qv.clone());

        // test occs and occs_smaller
        for c in 0..4 {
            assert_eq!(rsqv.occs(c), Some(10000 / 4));
        }
        assert_eq!(rsqv.occs(4), None);

        for c in 0..4 {
            assert_eq!(rsqv.occs_smaller(c), Some((10000 / 4) * (c as usize)));
        }

        // test get on just created qvector
        for (i, c) in qv.iter().enumerate() {
            assert_eq!(rsqv.get(i), Some(c));
        }

        for i in 0..qv.len() {
            let r = rsqv.rank(0, i);
            let extra = if i % 4 > 0 { 1 } else { 0 };
            assert_eq!(r, Some(i / 4 + extra));
        }

        for symbol in 0..4 {
            let r = rsqv.rank(symbol, qv.len());
            let cnt = qv.iter().filter(|x| *x == symbol).count();
            assert_eq!(r, Some(cnt));
            let r = rsqv.rank(symbol, qv.len() + 1);
            assert_eq!(r, None);
        }

        for (i, c) in qv.iter().enumerate() {
            let rank = rsqv.rank(c, i).unwrap();
            let s = rsqv.select(c, rank).unwrap();
            // println!("symbol: {} | rank: {} | selected: {}", c, rank, s);
            assert_eq!(s, i);
        }
    }

    #[test]
    fn test_boundaries<D>()
    where
        D: From<QVector> + AccessQuad + RankQuad + SelectQuad,
    {
        for n in [
            100,
            255,
            256,
            257,
            511,
            512,
            513,
            1024,
            1025,
            2047,
            2048,
            2049,
            256 * 8 - 1,
            256 * 8,
            256 * 8 + 1,
            512 * 8,
            512 * 8 + 1,
        ] {
            // tests blocks and superblocks boundaries
            for symbol in 0..3u8 {
                let qv: QVector = iter::repeat(symbol).take(n).collect();
                let rsqv = D::from(qv.clone());
                for i in 0..qv.len() + 1 {
                    if i < qv.len() {
                        assert_eq!(rsqv.get(i), Some(symbol));
                    }
                    assert_eq!(rsqv.rank(symbol, i), Some(i));
                    assert_eq!(rsqv.rank(3, i), Some(0));
                }
            }
        }
    }

    #[test]
    fn test_from_wt<D>()
    where
        D: From<QVector> + AccessQuad + RankQuad + SelectQuad,
    {
        // a bug in wt gives a test like this
        let n = 1025 * 2;
        let qv: QVector = (0..n).map(|x| if x < n / 2 { 0 } else { 1 }).collect();
        let rsqv = D::from(qv);

        for i in 0..n {
            if i < n / 2 {
                assert_eq!(rsqv.get(i), Some(0));
                assert_eq!(rsqv.rank(0, i), Some(i));
                assert_eq!(rsqv.rank(1, i), Some(0));
            } else {
                assert_eq!(rsqv.get(i), Some(1));
                assert_eq!(rsqv.rank(0, i), Some(n / 2));
                assert_eq!(rsqv.rank(1, i), Some(i - n / 2));
            }
        }
    }

    #[instantiate_tests(<RSQVector256>)]
    mod testp256 {}

    #[instantiate_tests(<RSQVector512>)]
    mod testp512 {}
}
