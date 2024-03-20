use crate::{BitVectorMut, QVector, RSNarrow, RankBin, SpaceUsage};

use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct PrefetchSupport {
    samples: Vec<RSNarrow>,
    sample_rate_shift: usize, // it's the log_2 of sample_rate, which must be a power of 2
}

impl PrefetchSupport {
    /// We conceptually split the quad vector `qv` into blocks of size
    /// `sample_rate`, where `sample_rate` is 2^`sample_rate_shift`. For each symbol, we have a
    /// bit vector with `qv.len()/sample_rate` bits.
    /// The ith bit in the bit vector of symbol `c` is `1` if and only if the
    /// ith block contains an occurrence of which is a multiple of `sample_rate`, `0´ otherwise.
    pub fn new(qv: &QVector, sample_rate_shift: usize) -> Self {
        let mut bvs = [
            BitVectorMut::default(),
            BitVectorMut::default(),
            BitVectorMut::default(),
            BitVectorMut::default(),
        ];
        let mut counters = [0; 4];
        let mut bits = [false; 4];

        let sample_rate = 1 << sample_rate_shift;

        for (i, symbol) in qv.iter().enumerate() {
            let symbol = symbol as usize;
            counters[symbol] += 1;
            if counters[symbol] % sample_rate == 0 {
                bits[symbol] = true;
            }

            if i % sample_rate == 0 || i == qv.len() - 1 {
                for i in 0..4 {
                    bvs[i].push(bits[i]);
                }
                bits = [false, false, false, false];
            }
        }

        Self {
            samples: bvs
                .into_iter()
                .map(|bvm| RSNarrow::new(bvm.into()))
                .collect(),
            sample_rate_shift,
        }
    }

    #[inline]
    pub unsafe fn approx_rank_unchecked(&self, symbol: u8, i: usize) -> usize {
        let block_id = i >> self.sample_rate_shift;
        let sample_rate = 1 << self.sample_rate_shift;

        self.samples
            .get_unchecked(symbol as usize)
            .rank1(block_id + 1)
            .unwrap()
            * sample_rate
    }
}

impl SpaceUsage for PrefetchSupport {
    fn space_usage_byte(&self) -> usize {
        self.samples.iter().map(|bv| bv.space_usage_byte()).sum()
    }
}
