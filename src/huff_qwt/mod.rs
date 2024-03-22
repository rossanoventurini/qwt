use std::{collections::HashMap, fmt::Debug, hash::Hash};

use minimum_redundancy::{BitsPerFragment, Code, Coding};
use num_traits::AsPrimitive;

use crate::{
    utils::{stable_partition_of_2_with_codes, stable_partition_of_4_with_codes},
    AccessBin, AccessUnsigned, BitVector, BitVectorMut, QVector, QVectorBuilder, RankBin,
    SelectBin, SpaceUsage, WTIndexable, WTSupport,
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

#[derive(Default, Clone, PartialEq, Debug)] // TODO: implement Serialize, Deserialize
pub struct HuffQWaveletTree<T, BRS, RS, const WITH_PREFETCH_SUPPORT: bool = false> {
    n: usize,        // The length of the represented sequence
    n_levels: usize, // The number of levels of the wavelet matrix
    codes: Vec<(T, Code)>,
    bvs: Vec<BRS>, // A bit vector for each final level
    qvs: Vec<RS>,  // A quad vector for each level
    lens: Vec<[usize; 2]>, //len of qv and bv for each level
                   // prefetch_support: Option<Vec<PrefetchSupport>>,
}

impl<T, BRS, RS, const WITH_PREFETCH_SUPPORT: bool>
    HuffQWaveletTree<T, BRS, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable + Hash + Debug,
    u8: AsPrimitive<T>,
    BRS: BinRSforWT,
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
                bvs: vec![BRS::default()],
                qvs: vec![RS::default()],
                lens: Vec::default(), // prefetch_support: None,
            };
        }

        //count symbol frequences
        let freqs: HashMap<T, usize> = sequence.iter().fold(HashMap::new(), |mut map, &c| {
            *map.entry(c).or_insert(0) += 1;
            map
        });

        //we get the codes and we fill the uninteresting bits with 1 (useful for partitioning later)
        let codes: HashMap<T, Code> = Coding::from_frequencies(BitsPerFragment(1), freqs)
            .codes_for_values()
            .into_iter()
            // .map(|mut c| {  // fill with ones
            //     c.1.content |= std::u32::MAX << c.1.len;
            //     c
            // })
            .collect();

        println!("{:?}", codes);

        println!("INITIAL SEQ: ");
        println!("{:?}", sequence);

        //the first level has to be a bitvector because we need to index the first time in the correct order (for a correct get)
        //we cant start partitioning into qvs and bvs straight from the first level
        let max_len = codes
            .iter()
            .map(|x| x.1.len)
            .max()
            .expect("error while finding max code length") as usize;
        let n_levels = 1 + (max_len - 1) / 2 + (max_len - 1) % 2; //if max len is odd the last level is a bv

        let mut bvs = Vec::with_capacity(n_levels);
        let mut qvs = Vec::with_capacity(n_levels);
        let mut lens = Vec::with_capacity(n_levels);

        let mut shift = 1;
        //first level is separated
        let cur_qv = QVectorBuilder::new(); //first level qv is empty because we need to index by bv
        let mut cur_bv = BitVectorMut::new();
        for s in sequence.iter() {
            let cur_code = codes.get(s).expect("could not tanslate symbol into code");
            cur_bv.push((cur_code.content >> (cur_code.len - shift)) & 1 == 1);
        }

        let qv = cur_qv.build();
        let cur_len = [qv.len(), cur_bv.len()];
        lens.push(cur_len);
        qvs.push(RS::from(qv));
        bvs.push(BRS::from(cur_bv.iter().collect()));

        stable_partition_of_2_with_codes(sequence, shift as usize, &codes);
        println!("partition now: ");
        println!("{:?}", sequence);
        //now we partitioned the codes and we can index them correctly

        for _level in 1..n_levels {
            let mut cur_qv = QVectorBuilder::new();
            let mut cur_bv = BitVectorMut::new();

            for s in sequence.iter() {
                let cur_code = codes
                    .get(s)
                    .expect("some error occurred during code translation while building huffqwt");
                //different paths if it goes in qv of bv
                if cur_code.len < shift {
                    //we finished handling this symbol in an upper level
                    continue;
                }

                if cur_code.len - shift >= 2 {
                    //we put in a qvector
                    let qv_symbol = (cur_code.content >> (cur_code.len - shift - 2)) & 3;
                    cur_qv.push(qv_symbol as u8);
                } else {
                    //we are at a bitvector leaf
                    let bv_symbol = cur_code.content & 1;
                    cur_bv.push(bv_symbol == 1);
                }
            }

            shift += 2;

            let qv = cur_qv.build();
            let cur_len = [qv.len(), cur_bv.len()];
            lens.push(cur_len);
            qvs.push(RS::from(qv));
            bvs.push(BRS::from(cur_bv.iter().collect()));

            stable_partition_of_4_with_codes(sequence, shift as usize, &codes);
            println!("partition now: ");
            println!("{:?}", sequence);
            shift += 2;
        }

        qvs.shrink_to_fit();
        bvs.shrink_to_fit();
        lens.shrink_to_fit();

        // println!("{:?}", sequence);

        Self {
            n: sequence.len(),
            n_levels,
            codes: codes.into_iter().collect::<Vec<_>>(),
            bvs,
            qvs,
            lens,
            // prefetch_support: if WITH_PREFETCH_SUPPORT {
            //     Some(prefetch_support)
            // } else {
            //     None
            // },
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

impl<T, BRS, RS, const WITH_PREFETCH_SUPPORT: bool> AccessUnsigned
    for HuffQWaveletTree<T, BRS, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    u8: AsPrimitive<T>,
    RS: RSforWT,
    BRS: BinRSforWT,
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
            //we check which bv we need to lookup
            if cur_i < self.lens[level][0] {
                //we can access qv
                shift += 2;

                let symbol = self.qvs[level].get_unchecked(cur_i);
                result = (result << 2) | symbol as u32;

                let offset = unsafe { self.qvs[level].occs_smaller_unchecked(symbol) };
                cur_i = self.qvs[level].rank_unchecked(symbol, cur_i) + offset;
            } else if cur_i - self.lens[level][0] < self.lens[level][1] {
                //we can access bv
                shift += 1;

                let symbol = self.bvs[level].get_unchecked(cur_i - self.lens[level][0]);
                result = (result << 1) | symbol as u32;

                let offset = if symbol { self.bvs[level].n_zeros() } else { 0 };

                cur_i = if symbol {
                    self.bvs[level].rank1_unchecked(cur_i)
                } else {
                    self.bvs[level].rank0_unchecked(cur_i)
                };
                cur_i += offset;
            } else {
                //we finished contructing the result in the upper level
                break;
            }
        }

        println!("found result: {}", result);
        println!("found shift: {}", shift);

        self.codes
            .iter()
            .find(|x| x.1.content == result && x.1.len == shift)
            .expect("could not translate symbol")
            .0
    }
}

impl<T, BRS, RS, const WITH_PREFETCH_SUPPORT: bool> From<Vec<T>>
    for HuffQWaveletTree<T, BRS, RS, WITH_PREFETCH_SUPPORT>
where
    T: HWTIndexable,
    u8: AsPrimitive<T>,
    BRS: BinRSforWT,
    RS: RSforWT,
{
    fn from(mut v: Vec<T>) -> Self {
        HuffQWaveletTree::new(&mut v[..])
    }
}

#[cfg(test)]
mod tests;
