//! Implements data structure to support `rank` and `select` queries on a binary vector
//! with (small) 64-bit blocks.
//!
//! This implementation is inspired by [this paper] (https://link.springer.com/chapter/10.1007/978-3-031-20643-6_19)
use super::*;

use crate::{AccessBin, RankBin, SelectBin};

use serde::{Deserialize, Serialize};

//superblock is 44 bits, blocks are BLOCK_SIZE-1 * 12 bits each
const BLOCK_SIZE: usize = 8; // 8 64bit words for each block
const SUPERBLOCK_SIZE: usize = 8 * BLOCK_SIZE; // 8 blocks for each superblock (this is the size in u64 words)

// SELECT NOT IMPLEMENTED YET
const SELECT_ONES_PER_HINT: usize = 64 * SUPERBLOCK_SIZE * 2; // must be > superblock_size * 64
const SELECT_ZEROS_PER_HINT: usize = SELECT_ONES_PER_HINT;

#[derive(Clone, Default, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct RSBitVector {
    bv: BitVector,
    superblock_metadata: Vec<u128>, // in each u128 we store the pair (superblock, <7 blocks>) like so |L1  |L2|L2|L2|L2|L2|L2|L2|
    select_samples: [Box<[u32]>; 2],
}

impl RSBitVector {
    pub fn new(bv: BitVector) -> Self {
        let mut superblock_metadata = Vec::new();
        let mut total_rank: u128 = 0;
        let mut cur_metadata: u128 = 0;
        let mut word_pop: u128 = 0;
        let mut zeros_so_far: u128 = 0;
        let mut select_samples: [Vec<u32>; 2] = [Vec::new(), Vec::new()];

        let mut cur_hint_0 = 0;
        let mut cur_hint_1 = 0;

        select_samples[0].push(0);
        select_samples[1].push(0);

        for (b, &dl) in bv.data.iter().enumerate() {
            // println!("dataline numero {}", b);
            if b % 8 == 0 {
                //we are at the start of new superblock, so we push the rank so far
                total_rank += word_pop;
                word_pop = 0;

                cur_metadata = 0;
                cur_metadata |= total_rank;
                // cur_metadata <<= 128 - 44;
                // println!("new superblock! added metadata, total rank: {}", total_rank);
                // println!("metadata so far: {:0>128b}", cur_metadata);
            } else {
                //we ignore the frist block beacuse it would be 0

                //if we are not at the start of a new superblock, we are at the start of a block
                cur_metadata <<= 12;
                cur_metadata |= word_pop;

                // println!("new block! added metadata count");
                // println!("metadata so far: {:0>128b}", cur_metadata);
            }

            word_pop += dl.n_ones() as u128;

            if (total_rank + word_pop) / SELECT_ONES_PER_HINT as u128 > cur_hint_1 {
                //we insert a new hint for 0
                select_samples[1].push((b / 8) as u32);
                cur_hint_1 += 1;
                // println!("NUOVO HINT 1");
            }

            zeros_so_far += dl.n_zeros() as u128;
            if (zeros_so_far / SELECT_ZEROS_PER_HINT as u128) > cur_hint_0 {
                //we insert a new hint for 0
                select_samples[0].push((b / 8) as u32);
                cur_hint_0 += 1;
                // println!("NUOVO HINT 0");
            }

            if (b + 1) % 8 == 0 {
                //next round we reset the metadata so we push it now
                superblock_metadata.push(cur_metadata);
                // println!("Pushed superblock!");
            }
        }

        //we flush the remainder in total rank
        total_rank += word_pop;

        let left: usize = bv.data.len() % 8;
        // println!("BLOCKS LEFT CALCULATION: {} / {}", left, 8);

        if left != 0 {
            for _ in left..8 {
                cur_metadata <<= 12;
                cur_metadata |= word_pop;
                // println!("new block! added metadata count");
            }

            superblock_metadata.push(cur_metadata);
            // println!("Pushed superblock!");
        }

        //we push last superblock containing only the last total_rank
        cur_metadata = 0;
        cur_metadata |= total_rank;
        cur_metadata <<= 128 - 44;
        superblock_metadata.push(cur_metadata);
        // println!("Pushed LAST superblock!");

        superblock_metadata.shrink_to_fit();

        Self {
            bv,
            superblock_metadata,
            select_samples: select_samples
                .into_iter()
                .map(|x| x.into_boxed_slice())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }

        //if we want to use a slice
        // let slice = superblock_metadata.into_boxed_slice();

        // Self {
        //     bv,
        //     superblock_metadata: slice,
        // }
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
    fn superblock_rank(&self, block: usize) -> usize {
        (self.superblock_metadata[block] >> (128 - 44)) as usize
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

    #[inline(always)]
    /// Returns a pair `(position, rank)` where the position is the index of the word containing the first `1` having rank1 `i`
    /// and `rank` is the number of occurrences of `symbol` up to the beginning of this block.
    ///
    /// The caller must guarantee that `i` is not zero or greater than the length of the indexed sequence.
    fn select1_subblock(&self, i: usize) -> (usize, usize) {
        let mut position = 0;

        let n_blocks = self.superblock_metadata.len();

        let hint = i / SELECT_ONES_PER_HINT;
        let hint_start = self.select_samples[1][hint] as usize;
        // let hint_end = self.select_samples[1][hint+1] as usize;

        // println!("HINT START: {}", hint_start);

        for j in hint_start..n_blocks {
            // println!("{}: {}", j, self.superblock_rank(j));
            if self.superblock_rank(j) > i {
                position = j - 1;
                break;
            }
        }
        // println!("selected superblock {} with rank {}", position, self.superblock_rank(position););
        //position is now superblock

        //now we examine sub_blocks
        position *= SUPERBLOCK_SIZE / BLOCK_SIZE;

        // println!("now sub_blocks");
        for j in 0..(SUPERBLOCK_SIZE / BLOCK_SIZE) {
            // println!("{}: {}", j, self.sub_block_rank(position + j));
            if self.sub_block_rank(position + j) > i {
                position += j - 1;
                break;
            }
            //if we didn't stop before
            if j == 7 {
                position += j;
            }
        }
        let rank = self.sub_block_rank(position);

        (position, rank)
    }

    #[inline(always)]
    /// Returns a pair `(position, rank)` where the position is the index of the word containing the first `0` having rank0 `i`
    /// and `rank` is the number of occurrences of `symbol` up to the beginning of this block.
    ///
    /// The caller must guarantee that `i` is not zero or greater than the length of the indexed sequence.
    fn select0_subblock(&self, i: usize) -> (usize, usize) {
        let mut position = 0;

        let n_blocks = self.superblock_metadata.len();

        let hint = i / SELECT_ZEROS_PER_HINT;
        let hint_start = self.select_samples[0][hint] as usize;

        let max_rank_for_block = SUPERBLOCK_SIZE * 64;

        for j in hint_start..n_blocks {
            // println!("{}: {}", j, self.superblock_rank(j));
            let rank0 = j * max_rank_for_block - self.superblock_rank(j);
            if rank0 > i {
                position = j - 1;
                break;
            }
        }
        // println!("selected block {} with rank0 {}", position, position * max_rank_for_block - self.superblock_rank(position));
        //position is now superblock

        //now we examine sub_blocks
        position *= SUPERBLOCK_SIZE / BLOCK_SIZE;

        let max_rank_for_subblock = BLOCK_SIZE * 64;
        // println!("now sub_blocks");
        for j in 0..(SUPERBLOCK_SIZE / BLOCK_SIZE) {
            // println!("iterazione {}", j);
            // println!(
            //     "{}: {}",
            //     j,
            //     max_rank_for_subblock * (position + j) - self.sub_block_rank(position + j)
            // );
            let rank0 = max_rank_for_subblock * (position + j) - self.sub_block_rank(position + j);
            if rank0 > i {
                position += j - 1;
                break;
            }
            if j == 7 {
                position += j;
            }
        }
        let rank = max_rank_for_subblock * position - self.sub_block_rank(position);

        (position, rank)
    }
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

        let sub_block = i >> 9;
        let mut result = self.sub_block_rank(sub_block);
        let sub_left = (i & 511) as i32 + 1;

        // sub_block *= BLOCK_SIZE; //we will handle single words from now on

        result += if sub_left == 0 {
            0
        } else {
            // let mut remainder = 0;
            // for _ in 0..BLOCK_SIZE {
            //     if sub_left <= 64 {
            //         unsafe {
            //             remainder += (*self.bv.data.get_unchecked(sub_block))
            //                 .wrapping_shl(64 - sub_left)
            //                 .count_ones() as usize;
            //         }
            //         break;
            //     } else {
            //         unsafe {
            //             remainder += (*self.bv.data.get_unchecked(sub_block)).count_ones() as usize;
            //         }
            //         sub_left -= 64;
            //     }
            //     sub_block += 1;
            // }

            //this will become `remainder = *self.bv.data[sub_block].rank1(sub_left)`
            // while sub_left > 0 {
            //     let cur_word = *self.bv.data.get_unchecked(sub_block);
            //     let x = if sub_left > 64 { 64 } else { sub_left };
            //     remainder += (cur_word & ((1u128 << x) - 1) as u64).count_ones() as usize;

            //     sub_left -= 64;
            //     sub_block += 1;
            // }
            // remainder

            self.bv.data[sub_block].rank1(sub_left as usize).unwrap()
        };

        result
    }
}

impl SelectBin for RSBitVector {
    #[inline(always)]
    /// # Examples
    /// ```
    /// use qwt::{BitVector, RSBitVector, SelectBin};
    ///
    ///
    /// let vv: Vec<usize> = vec![3, 5, 8, 128, 129, 513, 1000, 1024, 1025];
    /// let bv: BitVector = vv.iter().copied().collect();
    /// let rs = RSBitVector::new(bv);
    ///
    /// assert_eq!(rs.select1(0), Some(3));
    /// assert_eq!(rs.select1(1), Some(5));
    /// assert_eq!(rs.select1(2), Some(8));
    /// assert_eq!(rs.select1(3), Some(128));
    /// assert_eq!(rs.select1(4), Some(129));
    /// assert_eq!(rs.select1(5), Some(513));
    /// assert_eq!(rs.select1(8), Some(1025));
    /// assert_eq!(rs.select1(9), None);
    ///
    /// ```
    fn select1(&self, i: usize) -> Option<usize> {
        if i >= self.n_ones() {
            return None;
        }

        Some(unsafe { self.select1_unchecked(i) })
    }

    #[inline(always)]
    unsafe fn select1_unchecked(&self, i: usize) -> usize {
        let (block, rank) = self.select1_subblock(i);
        // println!("selected subblock {}, rank {}", block, rank);

        // self.bv[block].select(i-rank)
        let off = self.bv.data[block].select1_unchecked(i - rank);

        // block *= BLOCK_SIZE; // actual word in the bitvector

        // for _ in 0..BLOCK_SIZE {
        //     let kp = self.bv.get_word(block).count_ones();
        //     if kp as usize > (i - rank) {
        //         off = select_in_word(self.bv.get_word(block), (i - rank) as u64) as usize;
        //         break;
        //     } else {
        //         rank += kp as usize;
        //     }
        //     block += 1;
        // }

        block * 512 + off
    }

    /// # Examples
    /// ```
    /// use qwt::{BitVector, RSBitVector, SelectBin};
    /// use qwt::perf_and_test_utils::negate_vector;
    ///
    /// let vv: Vec<usize> = vec![3, 5, 8, 128, 129, 513, 1000, 1024, 1025];
    /// let bv: BitVector = vv.iter().copied().collect();
    /// let rs = RSBitVector::new(bv);
    /// let zeros_vector = negate_vector(&vv);
    ///
    /// assert_eq!(rs.select0(0), Some(0));
    /// assert_eq!(rs.select0(1), Some(1));
    /// assert_eq!(rs.select0(2), Some(2));
    /// assert_eq!(rs.select0(3), Some(4));
    /// assert_eq!(rs.select0(124), Some(127));
    /// assert_eq!(rs.select0(125), Some(130));
    /// assert_eq!(rs.select0(1016), Some(1023));
    /// assert_eq!(rs.select0(1017), None);
    ///
    /// ```
    #[inline(always)]
    fn select0(&self, i: usize) -> Option<usize> {
        if i >= self.n_zeros() {
            return None;
        }

        Some(unsafe { self.select0_unchecked(i) })
    }

    #[inline(always)]
    unsafe fn select0_unchecked(&self, i: usize) -> usize {
        let (block, rank) = self.select0_subblock(i);
        // println!("selected block {}, rank {}", block, rank);

        let off = self.bv.data[block].select0_unchecked(i - rank);

        // block *= BLOCK_SIZE;

        // for _ in 0..BLOCK_SIZE {
        //     let word_to_select = !self.bv.get_word(block);
        //     let kp = word_to_select.count_ones();
        //     if kp as usize > (i - rank) {
        //         off = select_in_word(word_to_select, (i - rank) as u64) as usize;
        //         break;
        //     } else {
        //         rank += kp as usize;
        //     }
        //     block += 1;
        // }

        block * 512 + off
    }
}

impl SpaceUsage for RSBitVector {
    /// Gives the space usage in bytes of the data structure.
    fn space_usage_byte(&self) -> usize {
        self.bv.space_usage_byte() + self.superblock_metadata.space_usage_byte()
    }
}

#[cfg(test)]
mod tests;
