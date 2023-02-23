use crate::utils::select_in_word;
use crate::QVector;

use serde::{Deserialize, Serialize};

// Traits
use crate::{AccessUnsigned, RankUnsigned, SelectUnsigned, SpaceUsage};
use num_traits::Unsigned;

/// Alternative representations to support Rank/Select queries at the level of blocks
mod rs_support_plain;
use crate::rs_qvector::rs_support_plain::RSSupportPlain;

/// Possible specializations which provide different space/time trade-offs.
pub type RSQVectorP256 = RSQVector<RSSupportPlain<256>>;
pub type RSQVectorP512 = RSQVector<RSSupportPlain<512>>;

/// The generic S is the data structure used to
/// provide rank/select support at the level of blocks.
#[derive(Default, Clone, Serialize, Deserialize, PartialEq, Debug)]
pub struct RSQVector<S: RSSupport + SpaceUsage> {
    qv: QVector,
    rs_support: S,
}

impl<S: RSSupport + SpaceUsage> SpaceUsage for RSQVector<S> {
    /// Gives the space usage in bytes of the data structure.
    fn space_usage_bytes(&self) -> usize {
        self.qv.space_usage_bytes() + self.rs_support.space_usage_bytes()
    }
}

impl<S: RSSupport + SpaceUsage> From<QVector> for RSQVector<S> {
    /// Converts a given quaternary vector `qv` to an `RSQVector` with support for `Rank`
    /// and `Select` queries.
    ///
    /// # Examples
    /// ```
    /// use qwt::{RSQVectorP256, QVector};
    ///
    /// let qv: QVector = (0..10_u64).into_iter().map(|x| x % 4).collect();
    /// let rsqv = RSQVectorP256::from(qv);
    ///
    /// assert_eq!(rsqv.is_empty(), false);
    /// assert_eq!(rsqv.len(), 10);
    /// ```
    fn from(qv: QVector) -> Self {
        let rank_support = S::new(&qv);

        let mut me = Self {
            qv,
            rs_support: rank_support,
        };

        me.shrink_to_fit();

        me
    }
}

impl<S: RSSupport + SpaceUsage> RSQVector<S> {
    /// Creates a quaternary vector with support for `RankUnsigned` and `SelectUnsigned`
    /// queries for a sequence of unsigned integers in the range [0, 3].
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
    /// use qwt::RSQVectorP256;
    ///
    /// let v: Vec<u64> = (0..10).into_iter().map(|x| x % 4).collect();
    /// let rsqv = RSQVectorP256::new(&v);
    ///
    /// assert_eq!(rsqv.is_empty(), false);
    /// assert_eq!(rsqv.len(), 10);
    /// ```
    pub fn new<T>(v: &[T]) -> Self
    where
        T: Unsigned + Copy,
        QVector: FromIterator<T>,
    {
        let mut qv: QVector = v.iter().copied().collect();
        unsafe {
            qv.align_to_64();
        }
        Self::from(qv)
    }

    #[inline(always)]
    fn rank_word<const SYMBOL: u8>(word: u128, mask: u64) -> usize {
        assert!(
            SYMBOL <= 3,
            "RSQVector indexes only four symbols in [0, 3]."
        );
        let word_high = (word >> 64) as u64;
        let word_low = word as u64;

        let r = match SYMBOL {
            0 => (!word_high & !word_low) & mask,
            1 => (!word_high & word_low) & mask,
            2 => (word_high & !word_low) & mask,
            _ => (word_high & word_low) & mask, // only 3 is possible here!
        };
        r.count_ones() as usize
    }

    #[inline(always)]
    fn select_intra_block<const SYMBOL: u8>(&self, i: usize, first_word: usize) -> usize {
        assert!(
            SYMBOL <= 3,
            "RSQVector indexes only four symbols in [0, 3]."
        );

        let mut cnt = 0;
        let mut prev_cnt;

        let data = self.qv.get_data();

        for k in 0..S::BLOCK_SIZE * 2 / 128 {
            prev_cnt = cnt;
            let word = unsafe { *data.get_unchecked(first_word + k) };
            let word_high = (word >> 64) as u64;
            let word_low = word as u64;

            let word = match SYMBOL {
                0 => !word_high & !word_low,
                1 => !word_high & word_low,
                2 => word_high & !word_low,
                _ => word_high & word_low, // only 3 is possible here!
            };

            cnt += word.count_ones() as usize;

            if cnt >= i {
                // previous word is the target
                let residual = i - prev_cnt;
                let mut result = k * 64;
                result += if residual == 0 {
                    0
                } else {
                    select_in_word(word, (residual - 1) as u64) as usize
                    // -1 because select_in_words starts counting occurrences from 0
                };
                return result;
            }
        }

        0
    }

    #[inline(always)]
    fn rank_intra_block<const SYMBOL: u8>(&self, i: usize) -> usize {
        assert!(
            SYMBOL <= 3,
            "RSQVector indexes only four symbols in [0, 3]."
        );

        let mut rank = 0;
        let data = self.qv.get_data();
        let first_word = i / S::BLOCK_SIZE * (S::BLOCK_SIZE * 2 / 128);
        let offset = i % S::BLOCK_SIZE; // offset (in symbols) within the block
        let last_word = first_word + offset / 64;

        /*  // Prefetching the second cache-line for the larger block size is not very useful.
        if S::BLOCK_SIZE >= 512 {
            // if a block covers two cache-lines, it could be convenient to h the second cache-line.
            unsafe {
                let p = data.as_ptr().add(last_word);
                _mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0);
            }
        } */

        for i in first_word..last_word {
            let word = unsafe { *data.get_unchecked(i) };
            rank += Self::rank_word::<SYMBOL>(word, u64::MAX);
        }

        let offset = offset % 64; // offset within the last word
        if offset > 0 {
            let mask = (1_u64 << offset) - 1;
            let word = unsafe { *data.get_unchecked(last_word) };
            rank += Self::rank_word::<SYMBOL>(word, mask);
        }

        rank
    }

    // Returns the number of symbols in the quaternary vector.
    pub fn len(&self) -> usize {
        self.qv.len()
    }

    /// Checks if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.qv.len() == 0
    }

    /// Shrinks the capacity of the allocated vector to fit the real length.
    #[inline(always)]
    pub fn shrink_to_fit(&mut self) {
        self.qv.shrink_to_fit();
        self.rs_support.shrink_to_fit();
    }

    /// Alligns data to 64-byte.
    ///
    /// Todo: make this safe by checking invariants.
    ///
    /// # Safety
    /// See Safety of [Vec::Vec::from_raw_parts](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.from_raw_parts).
    #[inline(always)]
    pub unsafe fn align_to_64(&mut self) {
        self.qv.align_to_64();
    }
}

impl<S: RSSupport + SpaceUsage> AccessUnsigned for RSQVector<S> {
    type Item = u8;

    /// Accesses the `i`-th value in the quad vector.
    /// The caller must guarantee that the position `i` is valid.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    ///
    /// # Examples
    /// ```
    /// use qwt::{RSQVectorP256, QVector, AccessUnsigned};
    ///
    /// let qv: QVector = (0..10_u64).into_iter().map(|x| x % 4).collect();
    /// let rsqv = RSQVectorP256::from(qv);
    ///
    /// assert_eq!(unsafe { rsqv.get_unchecked(0) }, 0);
    /// assert_eq!(unsafe { rsqv.get_unchecked(1) }, 1);
    /// ```
    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item {
        self.qv.get_unchecked(i)
    }

    /// Accesses the `i`th value in the quaternary vector
    /// or `None` if out of bounds.
    ///
    /// # Examples
    /// ```
    /// use qwt::{RSQVectorP256, QVector, AccessUnsigned};
    ///
    /// let qv: QVector = (0..10_u64).into_iter().map(|x| x % 4).collect();
    /// let rsqv = RSQVectorP256::from(qv);
    ///
    /// assert_eq!(rsqv.get(0), Some(0));
    /// assert_eq!(rsqv.get(1), Some(1));
    /// assert_eq!(rsqv.get(10), None);
    /// ```
    #[inline(always)]
    fn get(&self, i: usize) -> Option<Self::Item> {
        self.qv.get(i)
    }
}

impl<S: RSSupport + SpaceUsage> RankUnsigned for RSQVector<S> {
    /// Returns rank of `symbol` up to position `i` **excluded**.
    /// Returns `None` if out of bound.
    ///
    /// # Examples
    /// ```
    /// use qwt::{RSQVectorP256, QVector, RankUnsigned};
    ///
    /// let qv: QVector = (0..10_u64).into_iter().map(|x| x % 4).collect();
    /// let rsqv = RSQVectorP256::from(qv);
    ///
    /// assert_eq!(rsqv.rank(0, 0), Some(0));
    /// assert_eq!(rsqv.rank(0, 1), Some(1));
    /// assert_eq!(rsqv.rank(0, 2), Some(1));
    /// assert_eq!(rsqv.rank(0, 10), Some(3));
    /// assert_eq!(rsqv.rank(0, 11), None);
    /// ```
    #[inline(always)]
    fn rank(&self, symbol: Self::Item, i: usize) -> Option<usize> {
        if i > self.qv.len() {
            return None;
        }
        // Safety: Check above guarantees we are not out of bound
        Some(unsafe { self.rank_unchecked(symbol, i) })
    }

    /// Returns rank of `symbol` up to position `i` **excluded**.
    ///
    /// # Safety
    /// Calling this method with a position `i` larger than the length of the vector
    /// is undefined behavior.
    #[inline(always)]
    unsafe fn rank_unchecked(&self, symbol: Self::Item, i: usize) -> usize {
        debug_assert!(symbol <= 3);

        match symbol {
            0 => self.rs_support.rank_block::<0>(i) + self.rank_intra_block::<0>(i),
            1 => self.rs_support.rank_block::<1>(i) + self.rank_intra_block::<1>(i),
            2 => self.rs_support.rank_block::<2>(i) + self.rank_intra_block::<2>(i),
            _ => self.rs_support.rank_block::<3>(i) + self.rank_intra_block::<3>(i),
        }
    }
}

impl<S: RSSupport + SpaceUsage> SelectUnsigned for RSQVector<S> {
    /// Returns the position of the `i`th occurrence of `symbol`.
    /// Returns `None` if i is not valid, i.e., if i == 0 or i is larger than
    /// the number of occurrences of `symbol`.
    ///
    /// # Examples
    /// ```
    /// use qwt::{RSQVectorP256, QVector, SelectUnsigned};
    ///
    /// let qv: QVector = (0..10_u64).into_iter().map(|x| x % 4).collect();
    /// let rsqv = RSQVectorP256::from(qv);
    ///
    /// assert_eq!(rsqv.select(0, 1), Some(0));
    /// assert_eq!(rsqv.select(0, 2), Some(4));
    /// assert_eq!(rsqv.select(0, 3), Some(8));
    /// assert_eq!(rsqv.select(0, 0), None);
    /// assert_eq!(rsqv.select(0, 4), None);
    /// ```
    #[inline(always)]
    fn select(&self, symbol: Self::Item, i: usize) -> Option<usize> {
        debug_assert!(symbol <= 3);

        if i == 0 || self.rs_support.n_occs(symbol) < i {
            return None;
        }

        let (mut pos, rank) = self.rs_support.select_block(symbol, i);

        if rank == i {
            return Some(pos);
        }

        let first_word = pos * 2 / 128;

        pos += match symbol {
            0 => self.select_intra_block::<0>(i - rank, first_word),
            1 => self.select_intra_block::<1>(i - rank, first_word),
            2 => self.select_intra_block::<2>(i - rank, first_word),
            _ => self.select_intra_block::<3>(i - rank, first_word),
        };

        Some(pos)
    }

    /// Returns the position of the `i`th occurrence of `symbol`.
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

/// This trait should be implemented by any data structure that
/// provides rank/select support on blocks.
pub trait RSSupport {
    const BLOCK_SIZE: usize;

    fn new(qv: &QVector) -> Self;

    /// Returns the number of occurrences of `SYMBOL` up to the beginning
    /// of the block that contains position `i`.
    ///
    /// We use a const generic to have a specialized method for each symbol.
    fn rank_block<const SYMBOL: u8>(&self, i: usize) -> usize;

    /// Returns a pair `(position, rank)` where the position is the beginning of the block
    /// that contains the `i`th occurrence of `symbol`, and `rank` is the number of
    /// occurrences of `symbol` up to the beginning of this block.
    fn select_block(&self, symbol: u8, i: usize) -> (usize, usize);

    /// Returns the number of occurrences of `SYMBOL` in the whole sequence.
    fn n_occs(&self, symbol: u8) -> usize;

    /// Shrinks to fit.
    fn shrink_to_fit(&mut self);

    /// Length of the indexed sequence.
    fn len(&self) -> usize;

    /// Check if the indexed sequence is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
#[generic_tests::define]
mod tests {
    use super::*;
    use std::iter;

    #[test]
    fn test_small<D>()
    where
        D: From<QVector> + AccessUnsigned<Item = u8> + RankUnsigned + SelectUnsigned,
    {
        let qv: QVector = [0, 1, 2, 3].into_iter().cycle().take(10000).collect();
        let rsqv = D::from(qv.clone());

        // test get on just created quadvector
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
            let rank = rsqv.rank(c, i + 1).unwrap();
            let s = rsqv.select(c, rank).unwrap();

            assert_eq!(s, i);
        }
    }

    #[test]
    fn test_boundaries<D>()
    where
        D: From<QVector> + AccessUnsigned<Item = u8> + RankUnsigned + SelectUnsigned,
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
                    dbg!(i, symbol, n);
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
        D: From<QVector> + AccessUnsigned<Item = u8> + RankUnsigned + SelectUnsigned,
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

    #[instantiate_tests(<RSQVectorP256>)]
    mod testp256 {}

    #[instantiate_tests(<RSQVectorP512>)]
    mod testp512 {}
}
