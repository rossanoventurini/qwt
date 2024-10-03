use std::{collections::HashMap, marker::PhantomData};

use minimum_redundancy::{BitsPerFragment, Coding};
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};

use crate::{
    quadwt::huffqwt::PrefixCode,
    utils::{msb, stable_partition_of_2, stable_partition_of_2_with_codes},
    AccessUnsigned, BinWTSupport, BitVector, BitVectorMut, RankUnsigned, SelectUnsigned,
    SpaceUsage, WTIndexable, WTIterator,
};

pub trait BinRSforWT: From<BitVector> + BinWTSupport + SpaceUsage + Default {}
impl<T> BinRSforWT for T where T: From<BitVector> + BinWTSupport + SpaceUsage + Default {}

#[derive(Default, Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct WaveletTree<T, BRS, const COMPRESSED: bool = false> {
    n: usize,                                 // The length of the represented sequence
    n_levels: usize,                          // The number of levels of the wavelet matrix
    sigma: Option<T>,                         // Sigma used only if no compressed
    codes_encode: Option<Vec<PrefixCode>>,    // Lookup table for encoding
    codes_decode: Option<Vec<Vec<(u32, T)>>>, // Lookup table for decoding symbols
    bvs: Vec<BRS>,                            // Each level uses either a quad or bit vector
    lens: Vec<usize>,                         // Length of each vector
    phantom_data: PhantomData<T>,
}

struct LenInfo(usize, u32); //symbol, len

#[allow(clippy::identity_op)]
fn craft_wm_codes(freq: &mut HashMap<usize, u32>, sigma: usize) -> Vec<PrefixCode> {
    // count size of the alphabet
    let alph_size = freq.iter().count();

    let mut f = freq
        .iter()
        .map(|(&k, &v)| LenInfo(k, v))
        .collect::<Vec<_>>();

    f.sort_by_key(|x| x.1);

    let mut c = vec![0; alph_size];
    let mut assignments = vec![PrefixCode { content: 0, len: 0 }; sigma + 1];
    let mut m = 1; //how many codes we have so far
    let mut l = 0;

    for j in 0..alph_size {
        // println!("f[{}]: ({}, {})", j, f[j].0, f[j].1);

        while f[j].1 > l {
            for r in j..m {
                c[(m - j) * 1 + r] = c[r];
                c[r] |= 1 << l;
            }
            m = 2 * m - j;
            l += 1;
        }

        //the codes are stored in lexicographic order of their reverse codes,
        //now we get the actual one we need by reversing it
        let mut reversed_code = 0;
        for t in 0..l {
            reversed_code |= ((c[j] >> t) & 1) << (l - t - 1);
        }

        assignments[f[j].0 as usize] = PrefixCode {
            content: reversed_code,
            len: l,
        };
    }

    assignments
}

impl<T, BRS, const COMPRESSED: bool> WaveletTree<T, BRS, COMPRESSED>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    BRS: BinRSforWT,
{
    /// Builds a binary wavelet tree of the `sequence` of unsigned integers.
    /// The input `sequence` will be **destroyed**.
    /// If `[COMPRESSED == true]` the wavelet tree will be compressed, meaning the symbols
    /// will be represented using huffman coding.
    ///
    ///
    /// Both space usage and query time of a QWaveletTree depend on the length
    /// of the representation of the symbols.
    ///
    /// ## Panics
    /// Panics if the sequence is longer than the largest possible length.
    /// The largest possible length is 2^{43} symbols.
    ///
    /// # Examples
    /// ```
    /// use qwt::WT;
    ///
    /// let mut data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let wt = WT::new(&mut data);
    ///
    /// assert_eq!(wt.len(), 8);
    /// ```
    pub fn new(sequence: &mut [T]) -> Self {
        if sequence.is_empty() {
            return Self {
                n: 0,
                n_levels: 0,
                sigma: None,
                codes_encode: None,
                codes_decode: None,
                bvs: vec![],
                lens: vec![],
                phantom_data: PhantomData,
            };
        }

        let mut codes_encode = None;
        let mut codes_decode = None;
        let n_levels;
        let sig;
        let sigma = *sequence.iter().max().unwrap();

        if COMPRESSED {
            //we craft the codes

            //count symbol frequences
            let freqs = sequence.iter().fold(HashMap::new(), |mut map, &c| {
                *map.entry(c.as_()).or_insert(0u32) += 1;
                map
            });

            let mut lengths = Coding::from_frequencies(BitsPerFragment(1), freqs).code_lengths();

            let codes = craft_wm_codes(&mut lengths, sigma.as_());

            let max_len = codes
                .iter()
                .map(|x| x.len)
                .max()
                .expect("error while finding max code length") as usize;

            n_levels = max_len;

            let mut decoder = vec![Vec::default(); max_len + 1];
            for (i, c) in codes.iter().enumerate() {
                if c.len != 0 {
                    decoder[c.len as usize].push((c.content, i.as_()));
                }
            }

            //sort codes to make it easier to search
            for v in decoder.iter_mut() {
                v.sort_by_key(|(x, _)| *x)
            }

            codes_decode = Some(decoder);
            codes_encode = Some(codes);
            sig = None;
        } else {
            let log_sigma = msb(sigma) + 1; // Note that sigma equals the largest symbol, so it's already "alphabet_size - 1"
            n_levels = log_sigma as usize;
            sig = Some(sigma);
        }

        //populate bvs
        let mut bvs = Vec::with_capacity(n_levels);
        let mut lens = Vec::with_capacity(n_levels);

        let mut shift = 1;

        for _level in 0..n_levels {
            let mut cur_bv = BitVectorMut::new();

            for &s in sequence.iter() {
                if COMPRESSED {
                    let cur_code = codes_encode.as_ref().unwrap().get(s.as_() as usize).expect(
                        "some error occurred during code translation while building huffqwt",
                    );

                    if cur_code.len >= shift {
                        let symbol = ((cur_code.content >> (cur_code.len - shift)) & 1) == 1;
                        cur_bv.push(symbol);
                    }
                } else {
                    let symbol = ((s >> (n_levels - shift as usize)).as_() & 1) == 1;
                    cur_bv.push(symbol);
                }
            }

            let bv = BitVector::from(cur_bv);

            lens.push(bv.len());
            bvs.push(BRS::from(bv));

            if COMPRESSED {
                stable_partition_of_2_with_codes(
                    sequence,
                    shift as usize,
                    codes_encode.as_ref().unwrap(),
                );
            } else {
                stable_partition_of_2(sequence, n_levels - shift as usize);
            }

            shift += 1;
        }

        bvs.shrink_to_fit();

        Self {
            n: sequence.len(),
            n_levels,
            sigma: sig,
            codes_encode,
            codes_decode,
            bvs,
            lens,
            phantom_data: PhantomData,
        }
    }

    /// Returns the length of the indexed sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::WT;
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = WT::from(data);
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
    /// use qwt::WT;
    ///
    /// let wt = WT::<u8>::default();
    ///
    /// assert_eq!(wt.is_empty(), true);
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
    /// use qwt::{WT, HWT};
    ///
    /// let data = vec![1u8, 0, 1, 0, 255, 4, 5, 3];
    ///
    /// let wt = WT::from(data.clone());
    /// assert_eq!(wt.n_levels(), 8);
    ///
    /// let hwt = HWT::from(data.clone());
    /// assert_eq!(hwt.n_levels(), 3);
    /// ```
    #[must_use]
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }

    /// Returns an iterator over the values in the wavelet tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::WT;
    ///
    /// let data: Vec<u8> = (0..10u8).into_iter().cycle().take(100).collect();
    ///
    /// let wt = WT::from(data.clone());
    ///
    /// assert_eq!(wt.iter().collect::<Vec<_>>(), data);
    ///
    /// assert_eq!(wt.iter().rev().collect::<Vec<_>>(), data.into_iter().rev().collect::<Vec<_>>());
    /// ```
    pub fn iter(
        &self,
    ) -> WTIterator<T, WaveletTree<T, BRS, COMPRESSED>, &WaveletTree<T, BRS, COMPRESSED>> {
        WTIterator {
            i: 0,
            end: self.len(),
            qwt: self,
            _phantom: PhantomData,
        }
    }
}

impl<T, BRS, const COMPRESSED: bool> AccessUnsigned for WaveletTree<T, BRS, COMPRESSED>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    BRS: BinRSforWT,
{
    type Item = T;

    #[must_use]
    #[inline(always)]
    fn get(&self, i: usize) -> Option<Self::Item> {
        if i >= self.n {
            return None;
        }

        Some(unsafe { self.get_unchecked(i) })
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item {
        let mut cur_i = i;
        let mut result: u32 = 0;

        let mut shift = 0;

        for level in 0..self.n_levels {
            if COMPRESSED && cur_i >= self.lens[level] {
                break;
            }

            let symbol = self.bvs[level].get_unchecked(cur_i);
            result = (result << 1) | symbol as u32;

            let tmp = self.bvs[level].rank1_unchecked(cur_i);

            cur_i = if symbol {
                tmp + self.bvs[level].n_zeros()
            } else {
                cur_i - tmp
            };
            shift += 1;
        }

        if COMPRESSED {
            let idx = self.codes_decode.as_ref().unwrap()[shift]
                .binary_search_by_key(&result, |(x, _)| *x)
                .expect("could not translate symbol");

            T::from(self.codes_decode.as_ref().unwrap()[shift][idx].1).unwrap()
        } else {
            T::from(result).unwrap()
        }
    }
}

impl<T, BRS, const COMPRESSED: bool> RankUnsigned for WaveletTree<T, BRS, COMPRESSED>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    BRS: BinRSforWT,
{
    #[inline(always)]
    fn rank(&self, symbol: Self::Item, i: usize) -> Option<usize> {
        if i > self.n {
            return None;
        }

        if !COMPRESSED && symbol > *self.sigma.as_ref().unwrap() {
            return None;
        }

        if COMPRESSED
            && (symbol.as_() >= self.codes_encode.as_ref().unwrap().len()
                || self.codes_encode.as_ref().unwrap()[symbol.as_() as usize].len == 0)
        {
            return None;
        }

        Some(unsafe { self.rank_unchecked(symbol, i) })
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, symbol: Self::Item, i: usize) -> usize {
        let mut cur_i = i;
        let mut cur_p = 0;

        let symbol_len;
        let repr;

        if COMPRESSED {
            let code = &self.codes_encode.as_ref().unwrap()[symbol.as_() as usize];
            symbol_len = code.len as usize;
            repr = code.content;
        } else {
            repr = symbol.as_() as u32;
            symbol_len = self.n_levels;
        }

        for level in 0..symbol_len {
            let bit = ((repr >> (symbol_len - level - 1)) & 1) == 1;

            let offset = self.bvs[level].n_zeros();

            let tmp_p = self.bvs[level].rank1_unchecked(cur_p);
            let tmp_i = self.bvs[level].rank1_unchecked(cur_i);

            cur_p = if bit { tmp_p + offset } else { cur_p - tmp_p };

            cur_i = if bit { tmp_i + offset } else { cur_i - tmp_i };
        }

        cur_i - cur_p
    }
}

impl<T, BRS, const COMPRESSED: bool> SelectUnsigned for WaveletTree<T, BRS, COMPRESSED>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    BRS: BinRSforWT,
{
    #[inline(always)]
    fn select(&self, symbol: Self::Item, i: usize) -> Option<usize> {
        if COMPRESSED && self.codes_encode.as_ref().unwrap()[symbol.as_() as usize].len == 0 {
            return None;
        }

        let symbol_len;
        let repr;

        if COMPRESSED {
            let code = &self.codes_encode.as_ref().unwrap()[symbol.as_() as usize];
            symbol_len = code.len as usize;
            repr = code.content;
        } else {
            repr = symbol.as_() as u32;
            symbol_len = self.n_levels;
        }
        let mut b = 0;

        let mut path_off = Vec::with_capacity(symbol_len);
        let mut rank_path_off = Vec::with_capacity(symbol_len);

        for level in 0..symbol_len {
            path_off.push(b);

            let bit = ((repr >> (symbol_len - level - 1)) & 1) == 1;

            let rank_b = if bit {
                self.bvs[level].rank1(b)
            } else {
                self.bvs[level].rank0(b)
            }?;

            b = rank_b + if bit { self.bvs[level].n_zeros() } else { 0 };

            rank_path_off.push(rank_b);
        }

        let mut result = i;
        for level in (0..symbol_len).rev() {
            b = path_off[level];
            let rank_b = rank_path_off[level];
            let bit = ((repr >> (symbol_len - level - 1)) & 1) == 1;

            result = if bit {
                self.bvs[level].select1(rank_b + result)
            } else {
                self.bvs[level].select0(rank_b + result)
            }? - b;
        }

        Some(result)
    }

    #[inline(always)]
    unsafe fn select_unchecked(&self, symbol: Self::Item, i: usize) -> usize {
        self.select(symbol, i).unwrap()
    }
}

impl<T, BRS, const COMPRESSED: bool> From<Vec<T>> for WaveletTree<T, BRS, COMPRESSED>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    BRS: BinRSforWT,
{
    fn from(mut v: Vec<T>) -> Self {
        WaveletTree::new(&mut v[..])
    }
}

impl<T, BRS, const COMPRESSED: bool> AsRef<WaveletTree<T, BRS, COMPRESSED>>
    for WaveletTree<T, BRS, COMPRESSED>
{
    fn as_ref(&self) -> &WaveletTree<T, BRS, COMPRESSED> {
        self
    }
}

impl<T, BRS, const COMPRESSED: bool> IntoIterator for WaveletTree<T, BRS, COMPRESSED>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    BRS: BinRSforWT,
{
    type IntoIter = WTIterator<T, WaveletTree<T, BRS, COMPRESSED>, WaveletTree<T, BRS, COMPRESSED>>;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        WTIterator {
            i: 0,
            end: self.len(),
            qwt: self,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T, BRS, const COMPRESSED: bool> IntoIterator for &'a WaveletTree<T, BRS, COMPRESSED>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    BRS: BinRSforWT,
{
    type IntoIter =
        WTIterator<T, WaveletTree<T, BRS, COMPRESSED>, &'a WaveletTree<T, BRS, COMPRESSED>>;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, BRS, const COMPRESSED: bool> FromIterator<T> for WaveletTree<T, BRS, COMPRESSED>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    BRS: BinRSforWT,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        WaveletTree::new(&mut iter.into_iter().collect::<Vec<T>>())
    }
}

impl<T, BRS: SpaceUsage, const COMPRESSED: bool> SpaceUsage for WaveletTree<T, BRS, COMPRESSED> {
    /// Gives the space usage in bytes of the struct.
    fn space_usage_byte(&self) -> usize {
        let coding_overhead = if COMPRESSED {
            256 * 8 // 256 + 2 * sizeof(u32) codes_encode
            + self.codes_decode //codes_decode
                .iter()
                .fold(0, |a, v| a + v.len() * (4+1))
        } else {
            0
        };

        8 + 8
            + coding_overhead
            + self.lens.len() * 8
            + self
                .bvs
                .iter()
                .fold(0, |acc, ds| acc + ds.space_usage_byte())
    }
}

#[cfg(test)]
mod tests;
