//! A data structure for rank support provides an encoding scheme for counters of
//! symbols of a quad sequence up to the beginning of blocks of a fixed size
//! `Self::BLOCK_SIZE`.

use crate::qvector::rs_qvector::RSSupport;
use crate::utils::prefetch_read_NTA;
use crate::QVector;
use crate::{AccessQuad, SpaceUsage}; // Traits

use serde::{Deserialize, Serialize};

/// The generic const `B_SIZE` specifies the number of symbols in each block.
/// The possible values are 256 (default) and 512.
/// The space overhead for 256 is 12.5% while 512 halves this
/// space overhead (6.25%) at the cost of (slightly) increasing the query time.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct RSSupportPlain<const B_SIZE: usize = 256> {
    superblocks: Box<[SuperblockPlain]>,
    select_samples: [Box<[u32]>; 4],
}

impl<const B_SIZE: usize> SpaceUsage for RSSupportPlain<B_SIZE> {
    /// Gives the space usage in bytes of the struct.
    fn space_usage_byte(&self) -> usize {
        let mut select_space = 0;
        for c in 0..4 {
            select_space += self.select_samples[c].space_usage_byte();
        }
        self.superblocks.space_usage_byte() + select_space
    }
}

impl<const B_SIZE: usize> RSSupport for RSSupportPlain<B_SIZE> {
    const BLOCK_SIZE: usize = B_SIZE;

    fn new(qv: &QVector) -> Self {
        assert!(qv.len() < (1 << 43));

        assert!(
            (Self::BLOCK_SIZE == 256) | (Self::BLOCK_SIZE == 512),
            "Block size is either 256 or 512 symbols."
        );

        // A counter for each symbol
        //    - from the beginning of the sequence
        //    - from the beginning of the superblock
        let mut superblock_counters = [0; 4];
        let mut block_counters = [0; 4];
        let mut occs = [0; 4];

        // Sample superblock ids for each symbol at every SELECT_SAMPLES occurrence
        let mut select_samples: [Vec<u32>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        // Number of symbols in each superblock
        let superblock_size: usize = Self::BLOCKS_IN_SUPERBLOCK * Self::BLOCK_SIZE;
        let n_superblocks = (qv.len() + superblock_size) / superblock_size;
        let mut superblocks = Vec::<SuperblockPlain>::with_capacity(n_superblocks);

        for i in 0..qv.len() + 1 {
            // Need position qv.len() to make last superblock if needed

            if i % superblock_size == 0 {
                superblocks.push(SuperblockPlain::new(&superblock_counters));
                block_counters = [0; 4]; // reset block counters
            }

            if i % Self::BLOCK_SIZE == 0 {
                // Start a new block and add occs in the block to its counter
                let block_id = (i / Self::BLOCK_SIZE) % Self::BLOCKS_IN_SUPERBLOCK;

                superblocks
                    .last_mut()
                    .unwrap()
                    .set_block_counters(block_id, &block_counters);

                for symbol in 0..4u8 {
                    // just check if everything is ok
                    debug_assert_eq!(
                        block_counters[symbol as usize],
                        superblocks
                            .last()
                            .unwrap()
                            .get_block_counter(symbol, block_id)
                    );
                }
            }

            if i < qv.len() {
                // Safety: We are sure to be not out of bound
                let symbol = unsafe { qv.get_unchecked(i) as usize };

                if occs[symbol] % Self::SELECT_NUM_SAMPLES == 0 {
                    // we store a superblock id in a u32. Make sure it fits.
                    debug_assert!(Self::superblock_index(i) <= u32::MAX as usize);
                    debug_assert!(Self::superblock_index(i) < superblocks.len());
                    select_samples[symbol].push(Self::superblock_index(i) as u32);
                }

                superblock_counters[symbol] += 1;
                block_counters[symbol] += 1;
                occs[symbol] += 1;
            }
        }

        // Fill the next blocks with max occurrences. This is a sentinel for `select` query algorithm.
        let next_block_id = (qv.len() / Self::BLOCK_SIZE) % Self::BLOCKS_IN_SUPERBLOCK + 1;

        if next_block_id < Self::BLOCKS_IN_SUPERBLOCK {
            superblocks
                .last_mut()
                .unwrap()
                .set_block_counters(next_block_id, &block_counters);
        }

        // Add a sentinel for select_samples
        for sample in &mut select_samples {
            if sample.is_empty() {
                // always sample at least once
                sample.push(0);
            }
            sample.push(superblocks.len() as u32 - 1); // sentinel
        }

        Self {
            superblocks: superblocks.into_boxed_slice(),
            select_samples: select_samples
                .into_iter()
                .map(|sample| sample.into_boxed_slice())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(), // all this just to convert Vecs into boxed slices (and because we cannot collect into an array directly)
        }
    }

    /// Returns the number of occurrences of `symbol` up to the beginning
    /// of the block that contains position `i`.
    #[inline(always)]
    fn rank_block(&self, symbol: u8, i: usize) -> usize {
        debug_assert!(symbol <= 3, "Symbols are in [0, 3].");

        let superblock_index = Self::superblock_index(i);
        let block_index = Self::block_index(i);

        unsafe {
            self.superblocks
                .get_unchecked(superblock_index)
                .get_rank(symbol, block_index & 7)
        }
    }

    /// Returns a pair `(position, rank)` where the position is the beginning of the block
    /// that contains the `i`th occurrence of `symbol`, and `rank` is the number of
    /// occurrences of `symbol` up to the beginning of this block.
    ///
    /// The caller must guarantee that `i` is not zero or greater than the length of the indexed sequence.
    #[inline(always)]
    fn select_block(&self, symbol: u8, i: usize) -> (usize, usize) {
        let sampled_i = (i - 1) / Self::SELECT_NUM_SAMPLES;

        let mut first_sblock_id = self.select_samples[symbol as usize][sampled_i] as usize;
        let last_sblock_id = 1 + self.select_samples[symbol as usize][sampled_i + 1] as usize; // dont worry we have a sentinel

        let step = f64::sqrt((last_sblock_id - first_sblock_id) as f64) as usize + 1;

        while first_sblock_id < last_sblock_id {
            if self.superblocks[first_sblock_id].get_superblock_counter(symbol) >= i {
                break;
            }
            first_sblock_id += step;
        }

        first_sblock_id -= step;

        while first_sblock_id < last_sblock_id {
            if self.superblocks[first_sblock_id].get_superblock_counter(symbol) >= i {
                break;
            }
            first_sblock_id += 1;
        }

        first_sblock_id -= 1;

        let mut position = first_sblock_id * Self::BLOCK_SIZE * Self::BLOCKS_IN_SUPERBLOCK; // i.e., superblocksize
        let mut rank = self.superblocks[first_sblock_id].get_superblock_counter(symbol);

        // we have a sentinel block at the end. No way we can go too far.
        let (block_id, block_rank) =
            self.superblocks[first_sblock_id].block_predecessor(symbol, i - rank);

        position += block_id * Self::BLOCK_SIZE;
        rank += block_rank;

        (position, rank)
    }

    #[inline(always)]
    fn prefetch(&self, pos: usize) {
        let superblock_index = Self::superblock_index(pos);

        prefetch_read_NTA(&self.superblocks, superblock_index);
    }
}

impl<const B_SIZE: usize> RSSupportPlain<B_SIZE> {
    const SELECT_NUM_SAMPLES: usize = 1 << 13;
    const BLOCKS_IN_SUPERBLOCK: usize = 8; // Number of blocks in each superblock

    #[inline(always)]
    fn superblock_index(i: usize) -> usize {
        i / (Self::BLOCK_SIZE * Self::BLOCKS_IN_SUPERBLOCK)
    }

    #[inline(always)]
    fn block_index(i: usize) -> usize {
        i / Self::BLOCK_SIZE
    }
}

/// Stores counters for a superblock and its blocks.
/// We use a u128 for each of the 4 symbols.
/// A u128 is subdivided as follows:
/// - First 44 bits to store superblock counters
/// - Next 84 to store counters for 7 (out of 8) blocks (the first one is excluded)
#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize, PartialEq)]
#[repr(C, align(64))]
struct SuperblockPlain {
    counters: [u128; 4],
}

impl SpaceUsage for SuperblockPlain {
    /// Gives the space usage in bytes of the struct.
    fn space_usage_byte(&self) -> usize {
        4 * 128 / 8
    }
}

impl SuperblockPlain {
    const BLOCKS_IN_SUPERBLOCK: usize = 8; // Number of blocks in each superblock

    /// Creates a new superblock initialized with the number of occurrences
    /// of the four symbols from the beginning of the text.
    fn new(sbc: &[usize; 4]) -> Self {
        let mut counters = [0u128; 4];
        for symbol in 0..4 {
            counters[symbol] = (sbc[symbol] as u128) << 84;
        }

        Self { counters }
    }

    #[inline(always)]
    fn get_rank(&self, symbol: u8, block_id: usize) -> usize {
        let data = unsafe { *self.counters.get_unchecked(symbol as usize) };
        let sb = (data >> 84) as usize;

        // We avoid a branch here. We want b be 0 if block_id is 0, real counter extracted from data otherwise
        let not_first = (block_id > 0) as usize;
        let b = ((data >> ((block_id - not_first) * 12)) as usize & 0b111111111111) * not_first;

        sb + b
    }

    fn get_superblock_counter(&self, symbol: u8) -> usize {
        (unsafe { *self.counters.get_unchecked(symbol as usize) } >> 84) as usize
    }

    fn set_block_counters(&mut self, block_id: usize, counters: &[usize; 4]) {
        assert!(block_id < 8);
        for &counter in counters.iter() {
            //assert!(counters[i] < SUPERBLOCK_SIZE);
            assert!(counter < (1 << 12));
        }
        if block_id == 0 {
            return;
        }

        for (symbol, &counter) in counters.iter().enumerate() {
            self.counters[symbol] |= (counter as u128) << ((block_id - 1) * 12);
        }
    }

    fn get_block_counter(&self, symbol: u8, block_id: usize) -> usize {
        debug_assert!(block_id < Self::BLOCKS_IN_SUPERBLOCK);

        if block_id == 0 {
            0
        } else {
            (self.counters[symbol as usize] >> ((block_id - 1) * 12) & 0b111111111111) as usize
        }
    }

    /// Returns the largest block id for `symbol` in this superblock
    /// such that its counter is smaller than `target` value.
    ///
    /// # TODO
    /// The loop is not (auto)vectorized but we know we are just searching for the
    /// predecessor x of a 12bit value in the last 84 bits of a u128.
    #[inline(always)]
    pub fn block_predecessor(&self, symbol: u8, target: usize) -> (usize, usize) {
        let mut cnt = self.counters[symbol as usize];
        let mut prev_cnt = 0;

        for block_id in 1..Self::BLOCKS_IN_SUPERBLOCK {
            let curr_cnt = (cnt & 0b111111111111) as usize;
            if curr_cnt >= target {
                return (block_id - 1, prev_cnt);
            }
            cnt >>= 12;
            prev_cnt = curr_cnt;
        }

        (Self::BLOCKS_IN_SUPERBLOCK - 1, prev_cnt)
    }
}
