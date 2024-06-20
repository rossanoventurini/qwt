//! Implements data structure to support `rank` and `select` queries on a binary vector with 512-bit blocks.
//!
//! This implementation is inspired by [this paper by Florian Kurpicz] (https://link.springer.com/chapter/10.1007/978-3-031-20643-6_19)
use crate::{
    utils::prefetch_read_NTA, AccessBin, BinWTSupport, BitVector, RankBin, SelectBin, SpaceUsage,
};

use serde::{Deserialize, Serialize};

//superblock is 44 bits, blocks are (BLOCK_SIZE-1) * 12 bits each
const BLOCK_SIZE: usize = 8; // 8 64bit words for each block
const SUPERBLOCK_SIZE: usize = 8 * BLOCK_SIZE; // 8 blocks for each superblock (this is the size in u64 words)

const SELECT_ONES_PER_HINT: usize = 64 * SUPERBLOCK_SIZE * 2; // must be > superblock_size * 64
const SELECT_ZEROS_PER_HINT: usize = SELECT_ONES_PER_HINT;

#[derive(Clone, Default, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct RSWide {
    bv: BitVector,
    superblock_metadata: Box<[u128]>, // in each u128 we store the pair (superblock, <7 blocks>) like so |L1  |L2|L2|L2|L2|L2|L2|L2|
    select_samples: [Box<[usize]>; 2],
    n_zeros: usize,
}

impl RSWide {
    pub fn new(bv: BitVector) -> Self {
        let mut superblock_metadata = Vec::new();
        let mut total_rank: u128 = 0;
        let mut cur_metadata: u128 = 0;
        let mut word_pop: u128 = 0;
        let mut zeros_so_far: u128 = 0;
        let mut select_samples: [Vec<usize>; 2] = [Vec::new(), Vec::new()];

        let mut cur_hint_0 = 0;
        let mut cur_hint_1 = 0;

        select_samples[0].push(0);
        select_samples[1].push(0);

        for (b, &dl) in bv.data.iter().enumerate() {
            if b % 8 == 0 {
                //we are at the start of new superblock, so we push the rank so far
                total_rank += word_pop;
                word_pop = 0;

                cur_metadata = 0;
                cur_metadata |= total_rank;
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
                //we insert a new hint for 1
                select_samples[1].push(b / 8);
                cur_hint_1 += 1;
            }

            zeros_so_far += dl.n_zeros() as u128;
            if (zeros_so_far / SELECT_ZEROS_PER_HINT as u128) > cur_hint_0 {
                //we insert a new hint for 0
                select_samples[0].push(b / 8);
                cur_hint_0 += 1;
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

        //guard at the end
        select_samples[0].push(superblock_metadata.len() - 1);
        select_samples[1].push(superblock_metadata.len() - 1);

        let n_zeros = bv.len() - total_rank as usize;

        Self {
            bv,
            superblock_metadata: superblock_metadata.into_boxed_slice(),
            select_samples: select_samples
                .into_iter()
                .map(|x| x.into_boxed_slice())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            n_zeros,
        }
    }

    /// Returns the number of bits set to 1 in the bitvector.
    #[inline(always)]
    pub fn n_ones(&self) -> usize {
        self.bv.len() - self.n_zeros()
    }

    /// Returns the number of bits set to 0 in the bitvector.
    #[inline(always)]
    pub fn n_zeros(&self) -> usize {
        self.n_zeros
    }

    /// Returns the number of bits in the bitvector.
    #[inline(always)]
    pub fn bv_len(&self) -> usize {
        self.bv.len()
    }

    #[inline(always)]
    fn superblock_rank(&self, block: usize) -> usize {
        (self.superblock_metadata[block] >> (128 - 44)) as usize
    }

    ///returns the total rank1 up to `sub_block`
    #[inline(always)]
    fn sub_block_rank(&self, sub_block: usize) -> usize {
        let mut result = 0;
        let superblock = sub_block / (SUPERBLOCK_SIZE / BLOCK_SIZE);
        result += self.superblock_rank(superblock);
        let left = sub_block % (SUPERBLOCK_SIZE / BLOCK_SIZE);

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
    /// The caller must guarantee that `i` is less than the length of the indexed sequence.
    fn select1_subblock(&self, i: usize) -> (usize, usize) {
        let mut position;

        let hint = i / SELECT_ONES_PER_HINT;
        let mut hint_start = self.select_samples[1][hint];
        let hint_end = 1 + self.select_samples[1][hint + 1];

        while hint_start < hint_end {
            if self.superblock_rank(hint_start) > i {
                break;
            }
            hint_start += 1;
        }
        position = hint_start - 1;
        // println!("selected superblock {} with rank {}", position, self.superblock_rank(position););

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
    /// The caller must guarantee that `i` is less than the length of the indexed sequence.
    fn select0_subblock(&self, i: usize) -> (usize, usize) {
        let mut position;

        let hint = i / SELECT_ZEROS_PER_HINT;
        let mut hint_start = self.select_samples[0][hint];
        let hint_end = 1 + self.select_samples[0][hint + 1];

        let max_rank_for_block = SUPERBLOCK_SIZE * 64;

        while hint_start < hint_end {
            if max_rank_for_block * hint_start - self.superblock_rank(hint_start) > i {
                break;
            }
            hint_start += 1;
        }
        position = hint_start - 1;
        // println!("selected block {} with rank0 {}", position, position * max_rank_for_block - self.superblock_rank(position));

        //now we examine sub_blocks
        position *= SUPERBLOCK_SIZE / BLOCK_SIZE;

        let max_rank_for_subblock = BLOCK_SIZE * 64;
        // println!("now sub_blocks");
        for j in 0..(SUPERBLOCK_SIZE / BLOCK_SIZE) {
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

impl BinWTSupport for RSWide {
    fn prefetch_info(&self, pos: usize) {
        prefetch_read_NTA(&self.superblock_metadata, pos / 512)
    }

    fn prefetch_data(&self, pos: usize) {
        prefetch_read_NTA(&self.bv.data, pos / 512)
    }
}

impl AccessBin for RSWide {
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

impl RankBin for RSWide {
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

        result += if sub_left == 0 {
            0
        } else {
            self.bv.data[sub_block].rank1(sub_left as usize).unwrap()
        };

        result
    }

    fn n_zeros(&self) -> usize {
        self.n_zeros()
    }
}

impl SelectBin for RSWide {
    #[inline(always)]
    /// Returns the position `pos` such that the element is `1` and rank1(pos) = i.
    /// Returns `None` if the data structure has no such element (i >= maximum rank1)
    /// # Examples
    /// ```
    /// use qwt::{BitVector, RSWide, SelectBin};
    ///
    ///
    /// let vv: Vec<usize> = vec![3, 5, 8, 128, 129, 513, 1000, 1024, 1025];
    /// let bv: BitVector = vv.iter().copied().collect();
    /// let rs = RSWide::new(bv);
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
    /// Returns the position `pos` such that the element is `1` and rank1(pos) = i.
    ///
    /// # Safety
    /// This method doesn't check that such element exists
    /// Calling this method with an i >= maximum rank1 is undefined behaviour.
    unsafe fn select1_unchecked(&self, i: usize) -> usize {
        let (block, rank) = self.select1_subblock(i);
        // println!("selected subblock {}, rank {}", block, rank);

        let off = self.bv.data[block].select1_unchecked(i - rank);

        block * 512 + off
    }

    #[inline(always)]
    /// Returns the position `pos` such that the element is `0` and rank0(pos) = i.
    /// Returns `None` if the data structure has no such element (i >= maximum rank0)
    /// # Examples
    /// ```
    /// use qwt::{BitVector, RSWide, SelectBin};
    /// use qwt::perf_and_test_utils::negate_vector;
    ///
    /// let vv: Vec<usize> = vec![3, 5, 8, 128, 129, 513, 1000, 1024, 1025];
    /// let bv: BitVector = vv.iter().copied().collect();
    /// let rs = RSWide::new(bv);
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
    fn select0(&self, i: usize) -> Option<usize> {
        if i >= self.n_zeros() {
            return None;
        }

        Some(unsafe { self.select0_unchecked(i) })
    }

    #[inline(always)]
    /// Returns the position `pos` such that the element is `0` and rank0(pos) = i.
    ///
    /// # Safety
    /// This method doesnt check that such element exists
    /// Calling this method with an `i >= maximum rank0` is undefined behaviour.
    unsafe fn select0_unchecked(&self, i: usize) -> usize {
        let (block, rank) = self.select0_subblock(i);
        // println!("selected block {}, rank {}", block, rank);

        let off = self.bv.data[block].select0_unchecked(i - rank);

        block * 512 + off
    }
}

impl SpaceUsage for RSWide {
    /// Gives the space usage in bytes of the data structure.
    fn space_usage_byte(&self) -> usize {
        self.bv.space_usage_byte()
            + self.superblock_metadata.space_usage_byte()
            + self.select_samples[0].space_usage_byte()
            + self.select_samples[1].space_usage_byte()
    }
}

impl From<BitVector> for RSWide {
    fn from(bv: BitVector) -> Self {
        RSWide::new(bv)
    }
}

#[cfg(test)]
mod tests;
