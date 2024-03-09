//! Implements data structure to support `rank` and `select` queries on a binary vector
//! with (small) 64-bit blocks.
//!
//! This implementation is inspired by [this paper] (https://link.springer.com/chapter/10.1007/978-3-031-20643-6_19)
use super::*;

use crate::{utils::select_in_word, AccessBin, RankBin, SelectBin};

use serde::{Deserialize, Serialize};

//superblock is 44 bits, blocks are BLOCK_SIZE-1 * 12 bits each
const BLOCK_SIZE: usize = 8; // 8 64bit words for each block

const SUPERBLOCK_SIZE: usize = 8 * BLOCK_SIZE; // 8 blocks for each superblock (this is the size in u64 words)

// SELECT NOT IMPLEMENTED YET
// const SELECT_ONES_PER_HINT: usize = 64 * BLOCK_SIZE * 2; // must be > block_size * 64
// const SELECT_ZEROS_PER_HINT: usize = SELECT_ONES_PER_HINT;

#[derive(Clone, Default, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct RSBitVector {
    bv: BitVector,
    superblock_metadata: Vec<u128>, // in each u128 we store the pair (superblock, <7 blocks>) like so |L1  |L2|L2|L2|L2|L2|L2|L2|
}

impl RSBitVector {
    pub fn new(bv: BitVector) -> Self {
        let mut superblock_metadata = Vec::new();
        let mut total_rank: u128 = 0;
        let mut cur_metadata: u128 = 0;
        let mut word_pop: u128 = 0;

        for (b, &word) in bv.data.iter().enumerate() {
            if b % SUPERBLOCK_SIZE == 0 {
                //we are at the start of new superblock, so we push the rank so far
                total_rank += word_pop;
                word_pop = 0;

                cur_metadata = 0;
                cur_metadata |= total_rank;
                // cur_metadata <<= 128 - 44;
                println!("new superblock! added metadata, total rank: {}", total_rank);
                // println!("metadata so far: {:0>128b}", cur_metadata);
            } else if b % BLOCK_SIZE == 0 {
                //we ignore the frist block beacuse it would be 0

                //if we are not at the start of a new superblock, we are at the start of a block
                cur_metadata <<= 12;
                cur_metadata |= word_pop;

                println!("new block! added metadata count");
                // println!("metadata so far: {:0>128b}", cur_metadata);
            }

            word_pop += word.count_ones() as u128;

            if (b + 1) % SUPERBLOCK_SIZE == 0 {
                //next round we reset the metadata so we push it now
                superblock_metadata.push(cur_metadata);
                println!("Pushed superblock!");
            }
        }

        //we flush the remainder in total rank
        total_rank += word_pop;

        let left: usize = (bv.data.len() % SUPERBLOCK_SIZE);
        println!("LEFT CALCULATION: {} / {}", left, SUPERBLOCK_SIZE);

        if left != 0 {
            for i in left..SUPERBLOCK_SIZE {
                if i % BLOCK_SIZE == 0 {
                    cur_metadata <<= 12;
                    cur_metadata |= word_pop;
                    println!("new block! added metadata count");
                }
            }

            superblock_metadata.push(cur_metadata);
            println!("Pushed superblock!");
        }

        //we push last superblock containing only the last total_rank
        cur_metadata = 0;
        cur_metadata |= total_rank;
        cur_metadata <<= 128 - 44;
        superblock_metadata.push(cur_metadata);
        println!("Pushed LAST superblock!");

        superblock_metadata.shrink_to_fit();

        Self {
            bv,
            superblock_metadata,
        }
    }

    //     /// Returns the number of bits set to 1 in the bitvector.
    //     #[inline(always)]
    //     pub fn n_ones(&self) -> usize {
    //         self.rank1(self.bv.len() - 1).unwrap() + self.bv.get(self.bv.len() - 1).unwrap() as usize
    //     }

    //     /// Returns the number of bits set to 0 in the bitvector.
    //     #[inline(always)]
    //     pub fn n_zeros(&self) -> usize {
    //         self.bv.len() - self.n_ones()
    //     }

    #[inline(always)]
    fn superblock_rank(&self, block: usize) -> usize {
        (self.superblock_metadata[block] >> 128 - 44) as usize
    }

    ///returns the total rank up to `sub_block`
    #[inline(always)]
    fn sub_block_rank(&self, sub_block: usize) -> usize {
        let mut result = 0;
        let superblock = sub_block / (SUPERBLOCK_SIZE / BLOCK_SIZE);
        result += self.superblock_rank(superblock);
        //println!("SUPERBLOCK: {:0>128b}", self.superblock_metadata[superblock]);
        let left = sub_block % (SUPERBLOCK_SIZE / BLOCK_SIZE);
        //println!("subblock {}: {:0>12b}", left, ((self.superblock_metadata[superblock] >> ((7 - left) * 12)) & 0x111111111111));
        // println!("sub_block: {sub_block} | superblock {superblock} | subblock {left}");

        if left != 0 {
            result += ((self.superblock_metadata[superblock] >> ((7 - left) * 12)) & 0b111111111111)
                as usize;
        }
        result
    }

    //     #[inline(always)]
    //     /// Returns a pair `(position, rank)` where the position is the index of the word containing the first `1` having rank `i`
    //     /// and `rank` is the number of occurrences of `symbol` up to the beginning of this block.
    //     ///
    //     /// The caller must guarantee that `i` is not zero or greater than the length of the indexed sequence.
    //     fn select1_subblock(&self, i: usize) -> (usize, usize) {
    //         let mut position = 0;
    //         let mut rank;

    //         println!("block rank pairs len = {}", self.block_rank_pairs.len());
    //         let n_blocks = self.block_rank_pairs.len() / 2;

    //         for j in 0..n_blocks {
    //             println!("{}: {}", j, self.block_rank(j));
    //             if self.block_rank(j) > i {
    //                 position = j - 1;
    //                 break;
    //             }
    //         }
    //         rank = self.block_rank(position);
    //         println!("selected block {} with rank {}", position, rank);
    //         //position is now superblock

    //         //now we examine sub_blocks
    //         position = position * BLOCK_SIZE;

    //         println!("now sub_blocks");
    //         for j in 0..BLOCK_SIZE {
    //             println!("{}: {}", j, self.sub_block_rank(position + j));
    //             if self.sub_block_rank(position + j) > i {
    //                 position += j - 1;
    //                 break;
    //             }
    //         }
    //         rank = self.sub_block_rank(position);

    //         (position, rank)
    //     }

    //     #[inline(always)]
    //     /// Returns a pair `(position, rank)` where the position is the index of the word containing the first `1` having rank `i`
    //     /// and `rank` is the number of occurrences of `symbol` up to the beginning of this block.
    //     ///
    //     /// The caller must guarantee that `i` is not zero or greater than the length of the indexed sequence.
    //     fn select0_subblock(&self, i: usize) -> (usize, usize) {
    //         let mut position = 0;
    //         let mut rank;

    //         println!("block rank pairs len = {}", self.block_rank_pairs.len());
    //         let n_blocks = self.block_rank_pairs.len() / 2;

    //         let max_rank_for_block = BLOCK_SIZE * 64;

    //         for j in 0..n_blocks {
    //             println!("{}: {}", j, self.block_rank(j));
    //             let rank0 = j * max_rank_for_block - self.block_rank(j);
    //             if rank0 > i {
    //                 position = j - 1;
    //                 break;
    //             }
    //         }
    //         rank = position * max_rank_for_block - self.block_rank(position);
    //         println!("selected block {} with rank0 {}", position, rank);
    //         //position is now superblock

    //         //now we examine sub_blocks
    //         position = position * BLOCK_SIZE;

    //         let max_rank_for_subblock = 64;
    //         println!("now sub_blocks");
    //         for j in 0..BLOCK_SIZE {
    //             println!("{}: {}", j, self.sub_block_rank(position + j));
    //             let rank0 = max_rank_for_subblock * (position + j) - self.sub_block_rank(position + j);
    //             if rank0 > i {
    //                 position += j - 1;
    //                 break;
    //             }
    //         }
    //         rank = max_rank_for_subblock * position - self.sub_block_rank(position);

    //         (position, rank)
    //     }
}

impl AccessBin for RSBitVector {
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

impl RankBin for RSBitVector {
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

// impl SelectBin for RSNarrow {
//     fn select1(&self, i: usize) -> Option<usize> {
//         if i == 0 {
//             return None;
//         }

//         Some(self.select1_unchecked(i))
//     }

//     fn select1_unchecked(&self, i: usize) -> usize {
//         //block_rank_pairs layout
//         //|superblock0|block0|superblock1|block1...

//         let (block, rank) = self.select1_subblock(i);
//         println!("selected block {}, rank {}", block, rank);

//         block * 64 + select_in_word(self.bv.get_word(block), (i - rank) as u64) as usize
//     }

//     fn select0(&self, i: usize) -> Option<usize> {
//         if i == 0 {
//             return None;
//         }

//         Some(self.select0_unchecked(i))
//     }

//     fn select0_unchecked(&self, i: usize) -> usize {
//         //block_rank_pairs layout
//         //|superblock0|blocks0|superblock1|blocks1...

//         let (block, rank) = self.select0_subblock(i);
//         println!("selected block {}, rank {}", block, rank);

//         let word_to_select = !self.bv.get_word(block);
//         block * 64 + select_in_word(word_to_select, (i - rank) as u64) as usize
//     }
// }

impl SpaceUsage for RSBitVector {
    /// Gives the space usage in bytes of the data structure.
    fn space_usage_byte(&self) -> usize {
        self.bv.space_usage_byte() + self.superblock_metadata.space_usage_byte()
    }
}

#[cfg(test)]
mod tests;
