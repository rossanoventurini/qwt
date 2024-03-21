use std::{collections::HashMap, hash::Hash};

use minimum_redundancy::{BitsPerFragment, Code, Coding};
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};

use crate::{AccessBin, BitVector, QVector, RankBin, SelectBin, SpaceUsage, WTIndexable, WTSupport};

pub trait BinWTSupport: AccessBin + RankBin + SelectBin {}
impl<T> BinWTSupport for T where T: AccessBin + RankBin + SelectBin {}

pub trait BinRSforWT: From<BitVector> + BinWTSupport + SpaceUsage + Default {}
impl<T> BinRSforWT for T where T: From<BitVector> + BinWTSupport + SpaceUsage + Default{}

pub trait RSforWT: From<QVector> + WTSupport + SpaceUsage + Default {}
// Generic implementation for any T
impl<T> RSforWT for T where T: From<QVector> + WTSupport + SpaceUsage + Default {}

#[derive(Default, Clone, PartialEq, Debug)] // TODO: implement Serialize, Deserialize
pub struct HuffQWaveletTree<T, BRS, RS, const WITH_PREFETCH_SUPPORT: bool = false> {
    n: usize,        // The length of the represented sequence
    n_levels: usize, // The number of levels of the wavelet matrix
    sigma: u32,      // The longest code in the sequence. *NOTE*: It's not +1 because it may overflow
    codes: Vec<(T, Code)>,
    bvs: Vec<BRS>,   // A bit vector for each final level
    qvs: Vec<RS>,    // A quad vector for each level
    // prefetch_support: Option<Vec<PrefetchSupport>>,
}

impl<T, BRS, RS, const WITH_PREFETCH_SUPPORT: bool> HuffQWaveletTree<T, BRS, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable + Hash,
    u8: AsPrimitive<T>,
    BRS: BinRSforWT,
    RS: RSforWT,
{
    pub fn new(sequence: &mut [T]) -> Self {
        if sequence.is_empty() {
            return Self {
                n: 0,
                n_levels: 0,
                sigma: 0,
                codes: Vec::default(),
                bvs: vec![BRS::default()],
                qvs: vec![RS::default()],
                // prefetch_support: None,
            };
        }

        //count symbol frequences
        let freqs: HashMap<T, usize> = sequence.iter().fold(HashMap::new(), |mut map, &c| {
            *map.entry(c).or_insert(0) += 1;
            map
        });

        let codes = Coding::from_frequencies(BitsPerFragment(1), freqs).codes_for_values().into_iter().collect::<Vec<_>>();

        let n_levels = 0;
        let sigma = codes.iter().map(|x| x.1.len).max().expect("error while finding sigma");
        let bvs = vec![BRS::default()];
        let qvs = vec![RS::default()];



        Self {
            n: sequence.len(),
            n_levels,
            sigma,
            codes,
            bvs,
            qvs,
            // prefetch_support: if WITH_PREFETCH_SUPPORT {
            //     Some(prefetch_support)
            // } else {
            //     None
            // },
        }
    }
}

#[cfg(test)]
mod tests;