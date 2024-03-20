//! The module implements [`DArray`], a data structure that provides efficient
//! `select1` and `select0` queries on a binary vector, supporting the [`SelectBin`] trait ([`SelectBin::select0`] and [`SelectBin::select1`] queries).
//! The rank queries are not supported.
//!
//! In many applications of this data structure, the binary vector is the characteristic
//! vector of a strictly increasing sequence.
//!
//! The `select1(i)` query returns the position of the (i+1)-th occurrence of a bit
//! set to 1 in the binary vector. For example, if the binary vector is 010000101,
//! `select1(0)` = 1, `select1(1)` = 6, and `select1(2)` = 6.
//! Similarly, the `select0(i)` query returns the position of the (i+1)-th zero
//! in the binary vector.
//! If we are representing a strictly increasing sequence S, `select1(i)` gives
//! the (i+1)th element of the sequence, i.e., S\[i\].
//!
//! ## Example
//! A [`DArray`] is built from a strictly increasening sequence of `usize`.
//! A boolean const generic is used to specify the need for `select0` query support.
//! Without this support, the query [`SelectBin::select0`] will panic.
//!
//! ```
//! use qwt::DArray;
//! use qwt::{SpaceUsage, SelectBin, RankBin};
//!
//! let vv: Vec<usize> = vec![0, 12, 33, 42, 55, 61, 1000];
//! let da: DArray<false> = vv.into_iter().collect();
//!
//! assert_eq!(da.select1(1), Some(12));
//! ```
//!
//! ## Technical details
//! [`DArray`] has been introduced in *D. Okanohara and K. Sadakane. Practical entropy-compressed Rank/Select dictionary. In Proceedings of the Workshop on Algorithm Engineering and Experiments (ALENEX), 2007* ([link](https://arxiv.org/abs/cs/0610001)).
//! This Rust implementation is inspired by the C++ implementation by
//! [Giuseppe Ottaviano](https://github.com/ot/succinct/blob/master/darray.hpp).
//!
//! Efficient queries are obtained by storing additional information on top of the binary vector.
//! The binary vector is split into blocks of variable size. The end of a block is marked every
//! `BLOCK_SIZE` = 1024-th occurrence of 1. Each block has two cases:
//!
//! 1. The block is *dense* if it is at most `MAX_IN_BLOCK_DISTACE` = 1 << 16 bits long.
//! 2. The block is *sparse* otherwise.
//!
//! For case 1, occurrences of 1 are further split into subblocks of size
//! `SUBBLOCK_SIZE` = 32 bits each. The position of the first 1 of each block is stored
//! in a vector (called *subblock_inventory*) using 16 bits each.
//! In case 2, the position of all ones is explicitly written in a vector called *overflow_positions*.
//! The vector *block_inventory* stores a pointer to *subblock_inventory* for blocks of the first kind
//! and a pointer to *overflow_positions* for the other kind of blocks. Positive or negative integers
//! are used to distinguish the two cases.
//!
//! A `select1(i)` query is solved as follows: First, compute b=i/`BLOCK_SIZE`, i.e., the block of
//! the i-th occurrence of 1, and access *`block_inventory[b]`*. If the block is dense, access
//! the position of the first one in its block and start a linear scan from that position
//! looking for the i-th occurrence of 1. If the block is sparse, the answer is stored in vector
//! *overflow_positions*.
//!
// TODO! //! Space overhead is TODO!
//!
//! These three vectors are stored in a private struct `Inventories`.
//! The const generic BITS in this struct allows us to build and store these vectors to support
//! `select0` as well.

use crate::bitvector::{BitVectorBitPositionsIter, BitVectorIter};
use crate::utils::select_in_word;
use crate::BitVector;
use crate::{AccessBin, SelectBin, SpaceUsage};

use serde::{Deserialize, Serialize};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_popcnt64;

const BLOCK_SIZE: usize = 1024;
const SUBBLOCK_SIZE: usize = 32;
const MAX_IN_BLOCK_DISTACE: usize = 1 << 16;

/// Const generic SELECT0_SUPPORT may optionally add
/// extra data structures to support fast `select0` queries,
/// which otherwise are not supported.

#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DArray<const SELECT0_SUPPORT: bool = false> {
    bv: BitVector,
    ones_inventories: Inventories<true>,
    zeroes_inventories: Option<Inventories<false>>,
}

// Helper struct for DArray that stores
// statistics, counters and overflow positions for bits
// set either to 0 or 1
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Inventories<const BIT: bool> {
    n_sets: usize, // number of bits set to BIT
    block_inventory: Box<[i64]>,
    subblock_inventory: Box<[u16]>,
    overflow_positions: Box<[usize]>,
}

/// Const generic BIT specifies if we are computing statistics
/// for zeroes (BIT=false) or for ones (BIT=true).
impl<const BIT: bool> Inventories<BIT> {
    fn new(bv: &BitVector) -> Self {
        let mut block_inventory = Vec::new();
        let mut subblock_inventory = Vec::new();
        let mut overflow_positions = Vec::new();

        let mut curr_block_positions = Vec::with_capacity(BLOCK_SIZE);

        // FIXME: Need to duplicate the code because
        // let mut iter_positions: BitVectorBitPositionsIter = if !BIT {bv.zeroes(0)} else {bv.ones(0)};
        // doesn't compile.

        let mut n_sets = 0;

        if !BIT {
            for curr_pos in bv.zeros() {
                curr_block_positions.push(curr_pos);
                if curr_block_positions.len() == BLOCK_SIZE {
                    Self::flush_block(
                        &curr_block_positions,
                        &mut block_inventory,
                        &mut subblock_inventory,
                        &mut overflow_positions,
                    );
                    curr_block_positions.clear()
                }
                n_sets += 1;
            }
        } else {
            for curr_pos in bv.ones() {
                curr_block_positions.push(curr_pos);
                if curr_block_positions.len() == BLOCK_SIZE {
                    Self::flush_block(
                        &curr_block_positions,
                        &mut block_inventory,
                        &mut subblock_inventory,
                        &mut overflow_positions,
                    );
                    curr_block_positions.clear()
                }
                n_sets += 1;
            }
        }

        Self::flush_block(
            &curr_block_positions,
            &mut block_inventory,
            &mut subblock_inventory,
            &mut overflow_positions,
        );

        Self {
            n_sets,
            block_inventory: block_inventory.into_boxed_slice(),
            subblock_inventory: subblock_inventory.into_boxed_slice(),
            overflow_positions: overflow_positions.into_boxed_slice(),
        }
    }

    fn flush_block(
        curr_positions: &[usize],
        block_inventory: &mut Vec<i64>,
        subblock_inventory: &mut Vec<u16>,
        overflow_positions: &mut Vec<usize>,
    ) {
        if curr_positions.is_empty() {
            return;
        }
        if curr_positions.last().unwrap() - curr_positions.first().unwrap() < MAX_IN_BLOCK_DISTACE {
            let v = *curr_positions.first().unwrap();
            block_inventory.push(v as i64);
            for i in (0..curr_positions.len()).step_by(SUBBLOCK_SIZE) {
                let dist = (curr_positions[i] - v) as u16;
                subblock_inventory.push(dist);
            }
        } else {
            let v: i64 = (-(overflow_positions.len() as i64)) - 1;
            block_inventory.push(v);
            overflow_positions.extend(curr_positions.iter());
            subblock_inventory.extend(std::iter::repeat(u16::MAX).take(curr_positions.len()));
        }
    }
}

/// Const genetic SELECT0_SUPPORT
impl<const SELECT0_SUPPORT: bool> DArray<SELECT0_SUPPORT> {
    /// Creates a [`DArray`] from a [`BitVector`].
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::{DArray, SelectBin};
    ///
    /// let v = vec![0,2,3,4,5];
    /// let da: DArray::<true> = v.into_iter().collect(); // <true> to support the select0 query
    ///
    /// assert_eq!(da.select1(2), Some(3));
    /// assert_eq!(da.select0(0), Some(1));
    /// ```
    #[must_use]
    pub fn new(bv: BitVector) -> Self {
        let ones_inventories = Inventories::new(&bv);
        let zeroes_inventories = if SELECT0_SUPPORT {
            Some(Inventories::new(&bv))
        } else {
            None
        };

        DArray {
            bv,
            ones_inventories,
            zeroes_inventories,
        }
    }

    /// Returns a non-consuming iterator over positions of bits set to 1 in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::DArray;
    ///
    /// let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
    /// let da: DArray = vv.iter().copied().collect();
    ///
    /// let v: Vec<usize> = da.ones().collect();
    /// assert_eq!(v, vv);
    /// ```
    #[must_use]
    pub fn ones(&self) -> BitVectorBitPositionsIter<true> {
        self.bv.ones()
    }

    /// Returns a non-consuming iterator over positions of bits set to 1 in the bit vector, starting at a specified bit position.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::DArray;
    ///
    /// let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
    /// let da: DArray = vv.iter().copied().collect();
    ///
    /// let v: Vec<usize> = da.ones_with_pos(2).collect();
    /// assert_eq!(v, vec![63, 128, 129, 254, 1026]);
    /// ```
    #[must_use]
    pub fn ones_with_pos(&self, pos: usize) -> BitVectorBitPositionsIter<true> {
        self.bv.ones_with_pos(pos)
    }

    /// Returns a non-consuming iterator over positions of bits set to 0 in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::DArray;
    /// use qwt::perf_and_test_utils::negate_vector;
    ///
    /// let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
    /// let da: DArray = vv.iter().copied().collect();
    ///
    /// let v: Vec<usize> = da.zeros().collect();
    /// assert_eq!(v, negate_vector(&vv));
    /// ```
    #[must_use]
    pub fn zeros(&self) -> BitVectorBitPositionsIter<false> {
        self.bv.zeros()
    }

    /// Returns a non-consuming iterator over positions of bits set to 0 in the bit vector, starting at a specified bit position.
    #[must_use]
    pub fn zeros_with_pos(&self, pos: usize) -> BitVectorBitPositionsIter<false> {
        self.bv.zeros_with_pos(pos)
    }

    /// Returns a non-consuming iterator over bits of the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::DArray;
    ///
    /// let v = vec![0,2,3,5];
    /// let da: DArray = v.into_iter().collect();
    ///
    /// let mut iter = da.iter();
    /// assert_eq!(iter.next(), Some(true)); // First bit is true
    /// assert_eq!(iter.next(), Some(false)); // Second bit is false
    /// assert_eq!(iter.next(), Some(true)); // Third bit is true
    /// assert_eq!(iter.next(), Some(true)); // Fourth bit is true
    /// assert_eq!(iter.next(), Some(false)); // Fifth bit is false
    /// assert_eq!(iter.next(), Some(true)); // Sixth bit is true
    /// assert_eq!(iter.next(), None); // End of the iterator
    /// ```
    pub fn iter(&self) -> BitVectorIter {
        self.bv.iter()
    }

    /// Returns the number of ones in the [`DArray`].
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::{DArray, SelectBin};
    ///
    /// let v = vec![0, 2, 3, 4, 5];
    /// let da: DArray<true> = v.into_iter().collect(); // <true> to support the select0 query
    ///
    /// assert_eq!(da.count_ones(), 5);
    /// ```
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.ones_inventories.n_sets
    }

    /// Returns the number of zeros in the [`DArray`].
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::{DArray, SelectBin};
    ///
    /// let v = vec![0, 2, 3, 4, 5];
    /// let da: DArray<true> = v.into_iter().collect(); // <true> to support the select0 query
    ///
    /// assert_eq!(da.count_zeros(), 1);
    /// ```
    #[must_use]
    pub fn count_zeros(&self) -> usize {
        self.bv.len() - self.ones_inventories.n_sets
    }

    /// Returns the number of elements in the [`DArray`].
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::{DArray, SelectBin};
    ///
    /// let v = vec![0, 2, 3, 4, 5];
    /// let da: DArray<true> = v.into_iter().collect(); // <true> to support the select0 query
    ///
    /// assert_eq!(da.len(), 6);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.bv.len()
    }

    /// Checks if the [`DArray`] is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::{DArray, SelectBin};
    ///
    /// let v: Vec<usize> = vec![];
    /// let da: DArray<true> = v.into_iter().collect(); // <true> to support the select0 query
    ///
    /// assert!(da.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bv.len() == 0
    }

    // Private generic select query, which solves either select0 and select1.
    #[inline(always)]
    fn select<const BIT: bool>(&self, i: usize, inventories: &Inventories<BIT>) -> Option<usize> {
        if i >= inventories.n_sets {
            return None;
        }
        let block = i / BLOCK_SIZE;
        let block_pos = inventories.block_inventory[block];

        if block_pos < 0 {
            // block is sparse
            let overflow_pos: usize = (-block_pos - 1) as usize;
            let idx = overflow_pos + (i & (BLOCK_SIZE - 1));
            return Some(inventories.overflow_positions[idx]);
        }
        let subblock = i / SUBBLOCK_SIZE;
        let start_pos = (block_pos as usize) + (inventories.subblock_inventory[subblock] as usize);
        let mut reminder = i & (SUBBLOCK_SIZE - 1);

        if reminder == 0 {
            return Some(start_pos);
        }

        let mut word_idx = start_pos >> 6;
        let word_shift = start_pos & 63;
        let mut word = if !BIT {
            !self.bv.get_word(word_idx) & (std::u64::MAX << word_shift) // if select0, negate the current word!
        } else {
            self.bv.get_word(word_idx) & (std::u64::MAX << word_shift)
        };

        loop {
            let popcnt;
            #[cfg(not(target_arch = "x86_64"))]
            {
                popcnt = word.count_ones() as usize;
            }
            #[cfg(target_arch = "x86_64")]
            {
                unsafe {
                    popcnt = _popcnt64(word as i64) as usize;
                }
            }
            if reminder < popcnt {
                break;
            }
            reminder -= popcnt;
            word_idx += 1;
            word = self.bv.get_word(word_idx);
            if !BIT {
                word = !word; // if select0, negate the current word!
            }
        }
        let select_intra = select_in_word(word, reminder as u64) as usize;

        Some((word_idx << 6) + select_intra)
    }
}

impl<const SELECT0_SUPPORT: bool> AccessBin for DArray<SELECT0_SUPPORT> {
    /// Returns the bit at the given position `i`, or [`None`] if `i` is out of bounds.
    ///
    /// # Examples
    /// ```
    /// use qwt::{DArray, AccessBin};
    ///
    /// let v = vec![0,2,3,4,5];
    /// let da: DArray = v.into_iter().collect();;
    ///
    /// assert_eq!(da.get(5), Some(true));
    /// assert_eq!(da.get(1), Some(false));
    /// assert_eq!(da.get(10), None);
    /// ```
    #[must_use]
    #[inline(always)]
    fn get(&self, i: usize) -> Option<bool> {
        self.bv.get(i)
    }

    /// Returns the bit at position `i`.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    ///
    /// # Examples
    /// ```
    /// use qwt::{DArray, AccessBin};
    ///
    /// let v = vec![0,2,3,4,5];
    /// let da: DArray = v.into_iter().collect();;
    /// assert_eq!(unsafe{da.get_unchecked(8)}, false);
    /// ```
    #[must_use]
    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> bool {
        self.bv.get_unchecked(i)
    }
}

impl<const SELECT0_SUPPORT: bool> SelectBin for DArray<SELECT0_SUPPORT> {
    /// Answers a `select1` query.
    ///
    /// The query `select1(i)` returns the position of the (i+1)-th
    /// occurrence of 1 in the binary vector.
    ///
    /// # Examples
    /// ```
    /// use qwt::{DArray, SelectBin};
    ///
    /// let v: Vec<usize> = vec![0, 12, 33, 42, 55, 61, 1000];
    /// let da: DArray = v.into_iter().collect();
    ///
    /// assert_eq!(da.select1(1), Some(12));
    /// ```
    ///
    /// # Panics
    /// It panics if [`DArray`] is built without support for `select0`query.
    #[must_use]
    #[inline(always)]
    fn select1(&self, i: usize) -> Option<usize> {
        self.select(i, &self.ones_inventories)
    }

    /// Answers a `select1` query without checking for bounds.
    ///
    /// The query `select1(i)` returns the position of the (i+1)-th
    /// occurrence of 1 in the binary vector.
    ///
    /// # Examples
    /// ```
    /// use qwt::{DArray, SelectBin};
    ///
    /// let v: Vec<usize> = vec![0, 12, 33, 42, 55, 61, 1000];
    /// let da: DArray = v.into_iter().collect();
    ///
    /// assert_eq!(unsafe{da.select1_unchecked(1)}, 12);
    /// ```
    #[must_use]
    #[inline(always)]
    unsafe fn select1_unchecked(&self, i: usize) -> usize {
        self.select(i, &self.ones_inventories).unwrap()
    }

    /// Answers a `select0` query.
    ///
    /// The query `select0(i)` returns the position of the (i+1)-th
    /// occurrence of 0 in the binary vector.
    ///
    /// # Examples
    /// ```
    /// use qwt::{DArray, SelectBin};
    ///
    /// let v: Vec<usize> = vec![0, 12, 33, 42, 55, 61, 1000];
    /// let da: DArray<true> = v.into_iter().collect();
    ///
    /// assert_eq!(da.select0(1), Some(2));
    /// assert_eq!(da.select0(11), Some(13));
    /// ```
    ///
    /// # Panics
    /// It panics if [`DArray`] is built without support for `select0`query.
    #[must_use]
    #[inline(always)]
    fn select0(&self, i: usize) -> Option<usize> {
        assert!(SELECT0_SUPPORT);

        self.select(i, self.zeroes_inventories.as_ref().unwrap())
    }

    /// Answers a `select0` query without checkin bounds.
    ///
    /// The query `select0(i)` returns the position of the (i+1)-th
    /// occurrence of 0 in the binary vector.
    ///
    /// # Examples
    /// ```
    /// use qwt::{DArray, SelectBin};
    ///
    /// let v: Vec<usize> = vec![0, 12, 33, 42, 55, 61, 1000];
    /// let da: DArray<true> = v.into_iter().collect();
    ///
    /// assert_eq!(unsafe{da.select0_unchecked(1)}, 2);
    /// assert_eq!(unsafe{da.select0_unchecked(11)}, 13);
    /// ```
    #[inline(always)]
    unsafe fn select0_unchecked(&self, i: usize) -> usize {
        assert!(SELECT0_SUPPORT);

        self.select(i, self.zeroes_inventories.as_ref().unwrap())
            .unwrap()
    }
}

/// Creates a [`DArray`] from an iterator over `bool` values.
///
/// # Examples
///
/// ```
/// use qwt::{AccessBin, DArray};
///
/// // Create a bit vector from an iterator over bool values
/// let da: DArray = vec![true, false, true].into_iter().collect();
///
/// assert_eq!(da.len(), 3);
/// assert_eq!(da.get(1), Some(false));
/// ```
impl<const SELECT0_SUPPORT: bool> FromIterator<bool> for DArray<SELECT0_SUPPORT> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = bool>,
    {
        DArray::<SELECT0_SUPPORT>::new(BitVector::from_iter(iter))
    }
}

/// Create a [`DArray`] from an iterator over strictly increasing sequence of integer values.
///
/// # Panics
/// Panics if the sequence is not stricly increasing or if any value of the sequence cannot be converted to usize.
///
/// # Examples
///
/// ```
/// use qwt::{DArray, AccessBin};
///
/// // Create a [`DArray`] from an iterator over strictly increasing sequence of non-negative integer values.
/// let da: DArray = vec![0, 1, 3, 5].into_iter().collect();
///
/// assert_eq!(da.len(), 6);
/// assert_eq!(da.get(3), Some(true));
/// ```
impl<V, const SELECT0_SUPPORT: bool> FromIterator<V> for DArray<SELECT0_SUPPORT>
where
    V: crate::bitvector::MyPrimInt + PartialOrd,
    <V as TryInto<usize>>::Error: std::fmt::Debug,
{
    #[must_use]
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = V>,
        <V as TryInto<usize>>::Error: std::fmt::Debug,
    {
        let data: Vec<_> = iter.into_iter().collect(); // TODO: we are doing this only to check sortedness. use the iterator directly without allocating a vector

        assert!(
            data.windows(2).all(|w| w[0] < w[1]),
            "Sequence must be strictly increasing"
        );

        DArray::<SELECT0_SUPPORT>::new(BitVector::from_iter(data))
    }
}

impl<const SELECT0_SUPPORT: bool> SpaceUsage for DArray<SELECT0_SUPPORT> {
    /// Returns the space usage of the data structure in bytes.
    fn space_usage_byte(&self) -> usize {
        let mut space = self.bv.space_usage_byte() + self.ones_inventories.space_usage_byte();

        if let Some(p) = self.zeroes_inventories.as_ref() {
            space += p.space_usage_byte();
        }
        space
    }
}

impl<const BIT: bool> SpaceUsage for Inventories<BIT> {
    fn space_usage_byte(&self) -> usize {
        self.n_sets.space_usage_byte()
            + self.block_inventory.space_usage_byte()
            + self.subblock_inventory.space_usage_byte()
            + self.overflow_positions.space_usage_byte()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_and_test_utils::{gen_strictly_increasing_sequence, negate_vector};

    #[test]
    fn test_select1() {
        let bv = BitVector::default();
        let v: Vec<usize> = bv.ones().collect();
        assert!(v.is_empty());

        let vv: Vec<usize> = vec![0, 12, 33, 42, 55, 61, 62, 63, 128, 129, 254, 1026];
        let da: DArray<false> = vv.iter().copied().collect();

        for (i, &sel) in vv.iter().enumerate() {
            let res = da.select1(i);
            assert_eq!(res.unwrap(), sel);
        }
        let res = da.select1(vv.len());
        assert_eq!(res, None);

        let vv = gen_strictly_increasing_sequence(1024 * 4, 1 << 20);
        let da: DArray<false> = vv.iter().copied().collect();

        for (i, &sel) in vv.iter().enumerate() {
            let res = da.select1(i);
            assert_eq!(res.unwrap(), sel);
        }
    }

    #[test]
    fn test_select0() {
        let bv = BitVector::default();
        let v: Vec<usize> = bv.ones().collect();
        assert!(v.is_empty());

        let vv: Vec<usize> = vec![0, 12, 33, 42, 55, 61, 62, 63, 128, 129, 254, 1026];
        let da: DArray<true> = vv.iter().copied().collect();

        for (i, &sel) in negate_vector(&vv).iter().enumerate() {
            let res = da.select0(i);
            assert_eq!(res.unwrap(), sel);
        }

        let vv = gen_strictly_increasing_sequence(1024 * 4, 1 << 20);
        let da: DArray<true> = vv.iter().copied().collect();

        for (i, &sel) in negate_vector(&vv).iter().enumerate() {
            let res = da.select0(i);
            assert_eq!(res.unwrap(), sel);
        }
    }
}
