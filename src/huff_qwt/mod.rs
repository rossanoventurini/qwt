use std::{collections::HashMap, fmt::Debug, hash::Hash, marker::PhantomData, vec};

use minimum_redundancy::{BitsPerFragment, Coding, Frequencies};
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};

use crate::{
    utils::stable_partition_of_4_with_codes, AccessBin, AccessUnsigned, BitVector, QVector,
    QVectorBuilder, RankBin, SelectBin, SpaceUsage, WTIndexable, WTSupport,
};

pub trait HWTIndexable: WTIndexable + Hash + Debug {}
impl<T> HWTIndexable for T where T: WTIndexable + Hash + Debug {} //helper to inlcude debug and hash for now

pub trait BinWTSupport: AccessBin + RankBin + SelectBin {}
impl<T> BinWTSupport for T where T: AccessBin + RankBin + SelectBin {}

pub trait BinRSforWT: From<BitVector> + BinWTSupport + SpaceUsage + Default {}
impl<T> BinRSforWT for T where T: From<BitVector> + BinWTSupport + SpaceUsage + Default {}

pub trait RSforWT: From<QVector> + WTSupport + SpaceUsage + Default {}
// Generic implementation for any T
impl<T> RSforWT for T where T: From<QVector> + WTSupport + SpaceUsage + Default {}

#[derive(Default, Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct PrefixCode {
    pub content: u32,
    pub len: u32,
}

#[derive(Default, Clone, PartialEq, Debug)] // TODO: implement Serialize, Deserialize
pub struct HuffQWaveletTree<T, RS, const WITH_PREFETCH_SUPPORT: bool = false> {
    n: usize,        // The length of the represented sequence
    n_levels: usize, // The number of levels of the wavelet matrix
    codes: Vec<PrefixCode>,
    qvs: Vec<RS>, // A quad vector for each level
    lens: Vec<usize>,
    phantom_data: PhantomData<T>, // prefetch_support: Option<Vec<PrefetchSupport>>,
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: HWTIndexable,
    u8: AsPrimitive<T>,
    RS: RSforWT,
{
    /// Builds the compressed wavelet tree of the `sequence` of unsigned integers.
    /// The input `sequence`` will be **destroyed**.
    ///
    /// Both space usage and query time of a QWaveletTree depend on the length
    /// of the compressed representation of the symbols.
    ///
    /// ## Panics
    /// Panics if the sequence is longer than the largest possible length.
    /// The largest possible length is 2^{43} symbols.
    ///
    /// # Examples
    /// ```
    /// use qwt::HQWT512;
    ///
    /// let mut data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = HQWT512::new(&mut data);
    ///
    /// assert_eq!(qwt.len(), 8);
    /// ```
    pub fn new(sequence: &mut [T]) -> Self {
        if sequence.is_empty() {
            return Self {
                n: 0,
                n_levels: 0,
                codes: Vec::default(),
                qvs: vec![RS::default()],
                lens: vec![0],
                // prefetch_support: None,
                phantom_data: PhantomData,
            };
        }

        //count symbol frequences
        let freqs = sequence.iter().fold(HashMap::new(), |mut map, &c| {
            *map.entry(c.as_()).or_insert(0u32) += 1;
            map
        });

        println!("entropy: {}", Frequencies::entropy(&freqs));

        //we get the codes and we fill the uninteresting bits with 1 (useful for partitioning later)
        let codes = Coding::from_frequencies(BitsPerFragment(2), freqs)
            .codes_for_values_array()
            .iter()
            .map(|&x| PrefixCode {
                len: x.len * 2, //convert fragments -> bits
                content: x.content,
            })
            .collect::<Vec<_>>();

        let max_len = codes
            .iter()
            .map(|x| x.len)
            .max()
            .expect("error while finding max code length") as usize;
        let n_levels = max_len / 2;

        let mut qvs = Vec::with_capacity(n_levels);
        let mut lens = Vec::with_capacity(n_levels);

        let mut shift = 0;

        for _level in 0..n_levels {
            let mut cur_qv = QVectorBuilder::new();

            for s in sequence.iter() {
                let cur_code = codes
                    .get(s.as_() as usize)
                    .expect("some error occurred during code translation while building huffqwt");
                //different paths if it goes in qv of bv
                if cur_code.len <= shift {
                    //we finished handling this symbol in an upper level
                    continue;
                }

                if cur_code.len - shift >= 2 {
                    //we put in a qvector
                    let qv_symbol = (cur_code.content >> (cur_code.len - shift - 2)) & 3;
                    cur_qv.push(qv_symbol as u8);
                }
            }

            shift += 2;

            let qv = cur_qv.build();
            lens.push(qv.len());
            qvs.push(RS::from(qv));

            stable_partition_of_4_with_codes(sequence, shift as usize, &codes);
        }

        qvs.shrink_to_fit();

        Self {
            n: sequence.len(),
            n_levels,
            codes: codes.into_iter().collect::<Vec<_>>(),
            qvs,
            lens,
            // prefetch_support: if WITH_PREFETCH_SUPPORT {
            //     Some(prefetch_support)
            // } else {
            //     None
            // },
            phantom_data: PhantomData,
        }
    }

    /// Returns the length of the indexed sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::HQWT256;
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = HQWT256::from(data);
    ///
    /// assert_eq!(qwt.len(), 8);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Checks if the indexed sequence is empty.
    ///
    /// # Examples
    /// ```
    /// use qwt::HQWT256;
    ///
    /// let qwt = HQWT256::<u8>::default();
    ///
    /// assert_eq!(qwt.is_empty(), true);
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Returns the number of levels in the wavelet tree.
    ///
    /// The number of levels represents the depth of the wavelet tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::HQWT256;
    ///
    /// let data = vec![1u8, 0, 1, 0, 255, 4, 5, 3];
    ///
    /// let qwt = HQWT256::from(data);
    ///
    /// assert_eq!(qwt.n_levels(), 2);
    /// ```
    #[must_use]
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> AccessUnsigned
    for HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    u8: AsPrimitive<T>,
    RS: RSforWT,
{
    type Item = T;

    fn get(&self, i: usize) -> Option<Self::Item> {
        if i >= self.n {
            return None;
        }

        Some(unsafe { self.get_unchecked(i) })
    }

    unsafe fn get_unchecked(&self, i: usize) -> Self::Item {
        let mut cur_i = i;
        let mut result: u32 = 0;

        let mut shift = 0;

        for level in 0..self.n_levels {
            // println!(
            //     "[level {}] cur_i: {}, self.lens[level]: {}",
            //     level, cur_i, self.lens[level]
            // );

            if cur_i >= self.lens[level] {
                break;
            }
            shift += 2;

            let symbol = self.qvs[level].get_unchecked(cur_i);
            result = (result << 2) | symbol as u32;

            let offset = unsafe { self.qvs[level].occs_smaller_unchecked(symbol) };
            cur_i = self.qvs[level].rank_unchecked(symbol, cur_i) + offset;
        }

        // println!("found result: {}", result);
        // println!("found shift: {}", shift);

        T::from(
            self.codes
                .iter()
                .position(|x| x.len == shift && x.content == result)
                .expect("could not translate symbol"),
        )
        .unwrap()
    }
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> From<Vec<T>>
    for HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: HWTIndexable,
    u8: AsPrimitive<T>,
    RS: RSforWT,
{
    fn from(mut v: Vec<T>) -> Self {
        HuffQWaveletTree::new(&mut v[..])
    }
}

#[cfg(test)]
mod tests;
