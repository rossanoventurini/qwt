use std::{collections::HashMap, hash::Hash};

use minimum_redundancy::{BitsPerFragment, Code, Coding};
use num_traits::AsPrimitive;

use crate::{utils::stable_partition_of_4_with_codes, AccessBin, BitVector, BitVectorMut, QVector, QVectorBuilder, RSWide, RankBin, SelectBin, SpaceUsage, WTIndexable, WTSupport};

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
    /// Builds the compressed wavelet tree of the `sequence` of unsigned integers.
    /// The input `sequence`` will be **destroyed**.
    ///
    /// Both space usage and query time of a QWaveletTree depend on the length
    /// of the compressed representation of the symbol.
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
    /// println!("{:?}", qwt);
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
                // prefetch_support: None,
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
            .map(|mut c| {
                c.1.content |= std::u32::MAX << c.1.len;
                c
            }).collect();

        let max_len = codes.iter().map(|x| x.1.len).max().expect("error while finding max code length") as usize;
        let n_levels = max_len / 2 + max_len % 2 + 1; //if max len is odd the last level is a bv
        
        let mut bvs = Vec::with_capacity(n_levels);
        let mut qvs = Vec::with_capacity(n_levels);


        for level in 0..n_levels {
            let mut cur_qv = QVectorBuilder::new();
            let mut cur_bv = BitVectorMut::new();

            for s in sequence.iter(){ 
                let cur_code = codes.get(s).expect("some error occurred during code translation while building huffqwt");
                //different paths if it goes in qv of bv
                if cur_code.len <= level as u32 * 2 {
                    //we finished handling this symbol in an upper level
                    continue;
                } 

                if cur_code.len - level as u32>= 2 {
                    //we put in a qvector
                    let qv_symbol = (cur_code.content >> (level * 2)) & 3;
                    cur_qv.push(qv_symbol as u8);
                }else{
                    //we are at a bitvector leaf
                    let bv_symbol = (cur_code.content >> (level * 2)) & 1;
                    cur_bv.push(bv_symbol == 1);
                }
            }

            let qv = cur_qv.build();
            qvs.push(RS::from(qv));
            bvs.push(BRS::from(cur_bv.iter().collect()));

            stable_partition_of_4_with_codes(sequence, level * 2, &codes);
        }

        qvs.shrink_to_fit();
        bvs.shrink_to_fit();

        Self {
            n: sequence.len(),
            n_levels,
            codes: codes.into_iter().collect::<Vec<_>>(),
            bvs,
            qvs,
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
    /// use qwt::HuffQWaveletTree;
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = HuffQWaveletTree::from(data);
    ///
    /// assert_eq!(qwt.len(), 8);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

}

impl<T, BRS, RS, const WITH_PREFETCH_SUPPORT: bool> From<Vec<T>>
    for HuffQWaveletTree<T, BRS, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable + Hash,
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