//! Implements data structure to support `rank` and `select` queries on a binary vector
//! with (small) 64-bit blocks.
//!
//! This implementation is inspired by the C++ implementation by [Giuseppe Ottaviano](https://github.com/ot/succinct/blob/master/rs_bit_vector.cpp).

use crate::{AccessBin, BitVector, RankBin, SpaceUsage};

use serde::{Deserialize, Serialize};

const BLOCK_SIZE: usize = 8; // in 64bit words

// SELECT NOT IMPLEMENTED YET
// const SELECT_ONES_PER_HINT: usize = 64 * BLOCK_SIZE * 2; // must be > block_size * 64
// const SELECT_ZEROS_PER_HINT: usize = SELECT_ONES_PER_HINT;

#[derive(Clone, Default, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct RSNarrow {
    bv: BitVector,
    block_rank_pairs: Vec<u64>,
}

impl RSNarrow {
    pub fn new(bv: BitVector) -> Self {
        let mut block_rank_pairs = Vec::new();
        let mut next_rank: u64 = 0;
        let mut cur_subrank: u64 = 0;
        let mut subranks: u64 = 0;
        block_rank_pairs.push(0);

        // We split data into blocks of BLOCK_SIZE = 8 words each.
        // for each block stores BLOCK_SIZE-1=7 9bit entries
        // with the number of ones from the beginning of the block
        // and a 64-bit entry with the number of ones from the
        // beginning of the bit vector.
        for (b, &word) in bv.data.iter().enumerate() {
            let word_pop = word.count_ones() as u64;
            let shift = b % BLOCK_SIZE;

            if shift >= 1 {
                subranks <<= 9;
                subranks |= cur_subrank;
            }

            next_rank += word_pop;
            cur_subrank += word_pop;

            if shift == BLOCK_SIZE - 1 {
                block_rank_pairs.push(subranks);
                block_rank_pairs.push(next_rank);
                subranks = 0;
                cur_subrank = 0;
            }
        }

        let left = BLOCK_SIZE - (bv.data.len() % BLOCK_SIZE);
        for _ in 0..left {
            subranks <<= 9;
            subranks |= cur_subrank;
        }
        block_rank_pairs.push(subranks);

        if bv.data.len() % BLOCK_SIZE > 0 {
            block_rank_pairs.push(next_rank);
            block_rank_pairs.push(0);
        }

        block_rank_pairs.shrink_to_fit();

        Self {
            bv,
            block_rank_pairs,
        }
    }

    /// Returns the number of bits set to 1 in the bitvector.
    #[inline(always)]
    pub fn n_ones(&self) -> usize {
        self.rank1(self.bv.len() - 1).unwrap() + self.bv.get(self.bv.len() - 1).unwrap() as usize
    }

    /// Returns the number of bits set to 0 in the bitvector.
    #[inline(always)]
    pub fn n_zeros(&self) -> usize {
        self.bv.len() - self.n_ones()
    }

    #[inline(always)]
    fn block_rank(&self, block: usize) -> usize {
        self.block_rank_pairs[block * 2] as usize
    }

    #[inline(always)]
    fn sub_block_ranks(&self, block: usize) -> usize {
        self.block_rank_pairs[block * 2 + 1] as usize
    }

    #[inline(always)]
    fn sub_block_rank(&self, sub_block: usize) -> usize {
        let mut result = 0;
        let block = sub_block / BLOCK_SIZE;
        result += self.block_rank(block);
        let left = sub_block % BLOCK_SIZE;
        result += self.sub_block_ranks(block) >> ((7 - left) * 9) & 0x1FF;
        result
    }
}

impl AccessBin for RSNarrow {
    /// Returns the bit at the given position `i`,
    /// or [`None`] if `i` is out of bounds.
    #[inline(always)]
    fn get(&self, i: usize) -> Option<bool> {
        if i >= self.bv.len() {
            return None;
        }
        Some(unsafe { self.get_unchecked(i) })
    }

    /// Returns the bit at the given position `i`.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> bool {
        self.bv.get_unchecked(i)
    }
}

impl RankBin for RSNarrow {
    #[inline(always)]
    fn rank1(&self, i: usize) -> Option<usize> {
        if self.bv.is_empty() || i > self.bv.len() {
            return None;
        }

        Some(unsafe { self.rank1_unchecked(i) })
    }

    #[inline(always)]
    unsafe fn rank1_unchecked(&self, i: usize) -> usize {
        if i == 0 {
            return 0;
        }
        let i = i - 1;

        let sub_block = i >> 6;
        let mut result = self.sub_block_rank(sub_block);
        let sub_left = (i & 63) as u32 + 1;

        result += if sub_left == 0 {
            0
        } else {
            unsafe {
                (*self.bv.data.get_unchecked(sub_block))
                    .wrapping_shl(64 - sub_left)
                    .count_ones() as usize
            }
        };

        result
    }
}

impl SpaceUsage for RSNarrow {
    /// Gives the space usage in bytes of the data structure.
    fn space_usage_byte(&self) -> usize {
        self.bv.space_usage_byte() + self.block_rank_pairs.space_usage_byte()
    }
}

#[cfg(test)]
mod tests;
