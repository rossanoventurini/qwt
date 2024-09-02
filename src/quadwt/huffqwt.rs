use std::{collections::HashMap, fmt::Debug, marker::PhantomData, vec};

use minimum_redundancy::{BitsPerFragment, Coding};
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};

use crate::{
    utils::stable_partition_of_4_with_codes, AccessUnsigned, QVectorBuilder, RankUnsigned,
    SelectUnsigned, SpaceUsage, WTIndexable, WTIterator,
};

use super::{prefetch_support::PrefetchSupport, RSforWT};

#[derive(Default, Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct PrefixCode {
    pub content: u32,
    pub len: u32,
}

/// Implements a compressed wavelet tree on quad vectors.
/// It doesn't achieve maximum compression, but the queries are faster
#[derive(Default, Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct HuffQWaveletTree<T, RS, const WITH_PREFETCH_SUPPORT: bool = false> {
    n: usize,                          // The length of the represented sequence
    n_levels: usize,                   // The number of levels of the wavelet matrix
    codes_encode: Vec<PrefixCode>,     // Lookup table for encoding
    codes_decode: Vec<Vec<(u32, u8)>>, // Lookup table for decoding symbols
    qvs: Vec<RS>,                      // A quad vector for each level
    lens: Vec<usize>,                  // Length of each qv
    phantom_data: PhantomData<T>,
    prefetch_support: Option<Vec<PrefetchSupport>>,
}

struct LenInfo(usize, u32); //symbol, len

#[allow(clippy::identity_op)]
fn craft_wm_codes(freq: &mut HashMap<usize, u32>, sigma: usize) -> Vec<PrefixCode> {
    // count size of the alphabet
    let alph_size = freq.iter().count();

    let mut f = freq
        .iter()
        .map(|(&k, &v)| LenInfo(k, v * 2)) // each fragment is 2 bits
        .collect::<Vec<_>>();

    f.sort_by_key(|x| x.1);

    let mut c = vec![0; alph_size * 4];
    let mut assignments = vec![PrefixCode { content: 0, len: 0 }; sigma + 1];
    let mut m = 1; //how many codes we have so far
    let mut l = 0;

    for j in 0..alph_size {
        // println!("f[{}]: ({}, {})", j, f[j].0, f[j].1);

        while f[j].1 > l {
            for r in j..m {
                c[(m - j) * 3 + r] = c[r];
                c[(m - j) * 2 + r] = c[r] | 1 << l;
                c[(m - j) * 1 + r] = c[r] | 2 << l;
                c[r] |= 3 << l;
            }
            m = 4 * m - 3 * j;
            l += 2;
        }

        //the codes are stored in lexicographic order of their reverse codes,
        //now we get the actual one we need by reversing it
        let mut reversed_code = 0;
        for t in (0..l).step_by(2) {
            reversed_code |= ((c[j] >> t) & 3) << (l - t - 2);
        }

        assignments[f[j].0 as usize] = PrefixCode {
            content: reversed_code,
            len: l,
        };
    }

    assignments
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
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
                codes_encode: Vec::default(),
                codes_decode: Vec::default(),
                qvs: vec![RS::default()],
                lens: vec![0],
                prefetch_support: None,
                phantom_data: PhantomData,
            };
        }

        let sigma = *sequence.iter().max().unwrap();
        //count symbol frequences
        let freqs = sequence.iter().fold(HashMap::new(), |mut map, &c| {
            *map.entry(c.as_()).or_insert(0usize) += 1;
            map
        });

        // println!("entropy: {}", Frequencies::entropy(&freqs));

        // println!("freqs: {:?}", &freqs);

        // let tot_occs = sequence.len();
        // println!("total occurrences: {}", tot_occs);

        let mut lengths =
            Coding::from_frequencies_cloned(BitsPerFragment(2), &freqs).code_lengths();

        // println!("lengths: {:?}", &lengths);

        // let mut awpl = 0;

        // for (&k, &v) in lengths.iter() {
        //     // println!("{} {} {}", &k, &v, freqs[&k]);
        //     awpl += v as usize * 2 * freqs[&k];
        // }

        // println!("awpl in bits: {}", awpl as f64 / tot_occs as f64);

        let codes = craft_wm_codes(&mut lengths, sigma.as_());

        let max_len = codes
            .iter()
            .map(|x| x.len)
            .max()
            .expect("error while finding max code length") as usize;
        let n_levels = max_len / 2; //we handle 2 bits for each level

        let mut codes_decode = vec![Vec::default(); max_len + 1];
        for (i, c) in codes.iter().enumerate() {
            if c.len != 0 {
                codes_decode[c.len as usize].push((c.content, i as u8));
            }
        }

        //sort codes to make it easier to search
        for v in codes_decode.iter_mut() {
            v.sort_by_key(|(x, _)| *x)
        }

        let mut prefetch_support = Vec::with_capacity(n_levels); // used only if WITH_PREFETCH_SUPPORT

        let mut qvs = Vec::with_capacity(n_levels);
        let mut lens = Vec::with_capacity(n_levels);

        let mut shift = 2;

        for _level in 0..n_levels {
            let mut cur_qv = QVectorBuilder::new();

            for &s in sequence.iter() {
                let cur_code = codes
                    .get(s.as_() as usize)
                    .expect("some error occurred during code translation while building huffqwt");

                if cur_code.len >= shift {
                    //we put in a qvector
                    let qv_symbol = (cur_code.content >> (cur_code.len - shift)) & 3;
                    cur_qv.push(qv_symbol as u8);
                }
            }

            let qv = cur_qv.build();
            let cur_qv_len = qv.len();
            // println!("{:?}", &sequence[0..cur_qv_len]);

            if WITH_PREFETCH_SUPPORT {
                let pfs = PrefetchSupport::new(&qv, 11); // 11 -> sample_rate = 2048
                prefetch_support.push(pfs);
            }

            lens.push(cur_qv_len);
            qvs.push(RS::from(qv));

            stable_partition_of_4_with_codes(sequence, shift as usize, &codes);
            shift += 2;
        }

        qvs.shrink_to_fit();

        // println!("lens of qwt: {:?}", &lens);

        Self {
            n: sequence.len(),
            n_levels,
            codes_encode: codes.into_iter().collect::<Vec<_>>(),
            codes_decode,
            qvs,
            lens,
            prefetch_support: if WITH_PREFETCH_SUPPORT {
                Some(prefetch_support)
            } else {
                None
            },
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

    /// Returns an iterator over the values in the wavelet tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::HQWT256;
    ///
    /// let data: Vec<u8> = (0..10u8).into_iter().cycle().take(100).collect();
    ///
    /// let qwt = HQWT256::from(data.clone());
    ///
    /// assert_eq!(qwt.iter().collect::<Vec<_>>(), data);
    ///
    /// assert_eq!(qwt.iter().rev().collect::<Vec<_>>(), data.into_iter().rev().collect::<Vec<_>>());
    /// ```
    pub fn iter(
        &self,
    ) -> WTIterator<
        T,
        HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>,
        &HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>,
    > {
        WTIterator {
            i: 0,
            end: self.len(),
            qwt: self,
            _phantom: PhantomData,
        }
    }

    #[inline]
    unsafe fn rank_prefetch_superblocks_unchecked(
        &self,
        repr: u32,
        code_len: u32,
        i: usize,
    ) -> usize {
        if !WITH_PREFETCH_SUPPORT {
            return 0;
        }

        if let Some(ref prefetch_support) = self.prefetch_support {
            let mut shift: i64 = code_len as i64 - 2;

            let mut range = 0..i;

            // //CHECK
            // let mut real_range = range.start..range.end;

            let mut level = 0;

            self.qvs[0].prefetch_data(range.end);
            self.qvs[0].prefetch_info(range.start);
            self.qvs[0].prefetch_info(range.end);

            while shift >= 2 {
                let two_bits: u8 = (repr >> shift as usize) as u8 & 3;

                // SAFETY: Here we are sure that two_bits is a symbol in [0..3]
                let offset = self.qvs[level].occs_smaller_unchecked(two_bits);

                let rank_start =
                    prefetch_support[level].approx_rank_unchecked(two_bits, range.start);
                let rank_end = prefetch_support[level].approx_rank_unchecked(two_bits, range.end);

                range = (rank_start + offset)..(rank_end + offset);
                self.qvs[level + 1].prefetch_info(range.start);
                self.qvs[level + 1].prefetch_info(range.start + 2048);

                self.qvs[level + 1].prefetch_info(range.end);
                self.qvs[level + 1].prefetch_info(range.end + 2048);
                if level > 0 {
                    self.qvs[level + 1].prefetch_info(range.start + 2 * 2048);
                    self.qvs[level + 1].prefetch_info(range.end + 2 * 2048);
                    self.qvs[level + 1].prefetch_info(range.end + 3 * 2048);
                }
                // self.qvs[level + 1].prefetch_info(range.end + 4 * 2048);

                // // CHECK!
                // let rank_start = self.qvs[level].rank_unchecked(two_bits, real_range.start);
                // let rank_end = self.qvs[level].rank_unchecked(two_bits, real_range.end);

                // real_range = (rank_start + offset)..(rank_end + offset);

                // //if range.start > real_range.start || range.end > real_range.end {
                // //     println!("Happen this");
                // // }

                // if range.start / 2048 != real_range.start / 2048
                //     && range.start / 2048 + 1 != real_range.start / 2048
                //     && range.start / 2048 + 2 != real_range.start / 2048
                // {
                //     println!("Level: {}", level);
                //     println!("Real range.start: {:?}", real_range);
                //     println!("Appr range.start: {:?}", range);
                //     println!("real_range.start / 2048:   {}", real_range.start / 2048);
                //     println!("approx range.start / 2048: {}\n", range.start / 2048);
                // }

                // if range.end / 2048 != real_range.end / 2048
                //     && range.end / 2048 + 1 != real_range.end / 2048
                //     && range.end / 2048 + 2 != real_range.end / 2048
                //     && range.end / 2048 + 3 != real_range.end / 2048
                // {
                //     println!("Level: {}", level);
                //     println!("Real range.end: {:?}", real_range);
                //     println!("Appr range.end: {:?}", range);
                //     println!("real_range.end / 2048:   {}", real_range.end / 2048);
                //     println!("approx range.end / 2048: {}\n", range.end / 2048);
                // }

                level += 1;
                shift -= 2;
            }

            return range.end - range.start;
        }

        0
    }

    /// Returns the rank of `symbol` up to position `i` **excluded**.
    ///
    /// `None` is returned if `i` is out of bound or if `symbol` is not valid
    /// (i.e., there is no occurence of the symbol in the original sequence).
    ///
    /// Differently from the `rank` function, `rank_prefetch` runs a first phase
    /// in which it estimates the positions in the wavelet tree needed by rank queries
    /// and prefetches these data. It is faster than the original `rank` function whenever
    /// the superblock/block counters fit in L3 cache but the sequence is larger.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::{HQWT256, RankUnsigned};
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = HQWT256::from(data);
    ///
    /// assert_eq!(qwt.rank_prefetch(1, 2), Some(1));
    /// assert_eq!(qwt.rank_prefetch(3, 8), Some(1));
    /// assert_eq!(qwt.rank_prefetch(1, 0), Some(0));
    /// assert_eq!(qwt.rank_prefetch(1, 9), None);  // Too large position
    /// assert_eq!(qwt.rank_prefetch(6, 1), None);  // Too large symbol
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn rank_prefetch(&self, symbol: T, i: usize) -> Option<usize> {
        if i > self.n
            || symbol.as_() >= self.codes_encode.len()
            || self.codes_encode[symbol.as_() as usize].len == 0
        {
            return None;
        }

        // SAFETY: Check the above guarantees we are not out of bound
        Some(unsafe { self.rank_prefetch_unchecked(symbol, i) })
    }

    /// Returns the rank of `symbol` up to position `i` **excluded**.
    ///
    /// Differently from the `rank_unchecked` function, `rank_prefetch` runs a first phase
    /// in which it estimates the positions in the wavelet tree needed by rank queries
    /// and prefetches these data. It is faster than the original `rank` function whenever
    /// the superblock/block counters fit in L3 cache but the sequence is larger.
    ///
    /// # Safety
    /// Calling this method with a position `i` larger than the size of the sequence
    /// of with invalid symbol is undefined behavior.
    ///
    /// # Examples
    /// ```
    /// use qwt::{HQWT256, RankUnsigned};
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = HQWT256::from(data);
    ///
    /// unsafe {
    ///     assert_eq!(qwt.rank_prefetch_unchecked(1, 2), 1);
    /// }
    /// ```
    #[must_use]
    #[inline(always)]
    pub unsafe fn rank_prefetch_unchecked(&self, symbol: T, i: usize) -> usize {
        //we get the code on which we rank
        let code = &self.codes_encode[symbol.as_() as usize];

        if WITH_PREFETCH_SUPPORT {
            let _ = self.rank_prefetch_superblocks_unchecked(code.content, code.len, i);
        }

        let mut range = 0..i;

        // //CHECK
        // let mut real_range = range.start..range.end;

        let mut shift: i64 = code.len as i64 - 2;
        let repr = code.content;

        const BLOCK_SIZE: usize = 256; // TODO: fix me!

        let mut level = 0;

        self.qvs[0].prefetch_data(range.start);
        self.qvs[0].prefetch_data(range.end);
        while shift >= 2 {
            let two_bits: u8 = (repr >> shift as usize) as u8 & 3;

            // SAFETY: Here we are sure that two_bits is a symbol in [0..3]
            let offset = self.qvs[level].occs_smaller_unchecked(two_bits);

            let rank_start = self.qvs[level].rank_block_unchecked(two_bits, range.start);
            let rank_end = self.qvs[level].rank_block_unchecked(two_bits, range.end);

            range = (rank_start + offset)..(rank_end + offset);

            // The estimated position can be off by BLOCK_SIZE for every level

            self.qvs[level + 1].prefetch_data(range.start);
            self.qvs[level + 1].prefetch_data(range.start + BLOCK_SIZE);

            self.qvs[level + 1].prefetch_data(range.end);
            self.qvs[level + 1].prefetch_data(range.end + BLOCK_SIZE);
            for i in 0..level {
                self.qvs[level + 1].prefetch_data(range.end + 2 * BLOCK_SIZE + i * BLOCK_SIZE);
            }

            // println!("Level: {} | two_bits {}", level, two_bits);
            // println!("prefetch range start: {}", range.start >> 8);
            // println!(
            //     "prefetch range end: {} .. {}",
            //     range.end >> 8,
            //     last_prefetch >> 8
            // );

            // // CHECK!
            // let rank_start = self.qvs[level].rank_unchecked(two_bits, real_range.start);
            // let rank_end = self.qvs[level].rank_unchecked(two_bits, real_range.end);

            // real_range = (rank_start + offset)..(rank_end + offset);

            // //if range.start > real_range.start || range.end > real_range.end {
            // //     // THIS NEVER HAPPEN!
            // // }

            // if range.start / 256 != real_range.start / 256
            //     && range.start / 256 + 1 != real_range.start / 256
            // {
            //     println!("Level: {}", level);
            //     println!("Real range.start: {:?}", real_range);
            //     println!("Appr range.start: {:?}", range);
            //     println!("real_range.start / 256:   {}", real_range.start / 256);
            //     println!("approx range.start / 256: {}\n", range.start / 256);
            // }

            // if !(range.end / 256 <= real_range.end / 256
            //     && range.end / 256 + level + 1 >= real_range.end / 256)
            // {
            //     println!("{}", range.end / 256 <= real_range.end / 256);
            //     println!("{}", range.end / 256 + level + 1 >= real_range.end / 256);
            //     println!("{}", range.end / 256 + level >= real_range.end / 256);
            //     println!("{}", range.end / 256 <= real_range.end / 256);
            //     println!("Level: {}", level);
            //     println!("Real range.end: {:?}", real_range);
            //     println!("Appr range.end: {:?}", range);
            //     println!("real_range.end / 256:   {}", real_range.end / 256);
            //     println!("approx range.end / 256: {}\n", range.end / 256);
            // }

            level += 1;
            shift -= 2;
        }
        self.rank_unchecked(symbol, i)
    }
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> AccessUnsigned
    for HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    RS: RSforWT,
{
    type Item = T;

    /// Returns the `i`-th symbol of the indexed sequence.
    ///
    /// `None` is returned if `i` is out of bound.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::{HQWT256, AccessUnsigned};
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = HQWT256::from(data);
    ///
    /// assert_eq!(qwt.get(2), Some(1));
    /// assert_eq!(qwt.get(3), Some(0));
    /// assert_eq!(qwt.get(8), None);
    /// ```
    #[must_use]
    #[inline(always)]
    fn get(&self, i: usize) -> Option<Self::Item> {
        if i >= self.n {
            return None;
        }

        Some(unsafe { self.get_unchecked(i) })
    }

    /// Returns the `i`-th symbol of the indexed sequence.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    ///
    /// Users must ensure that the index `i` is within the bounds of the sequence.
    ///
    /// # Examples
    /// ```
    /// use qwt::{HQWT256, AccessUnsigned};
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = HQWT256::from(data);
    ///
    /// unsafe {
    ///     assert_eq!(qwt.get_unchecked(2), 1);
    ///     assert_eq!(qwt.get_unchecked(3), 0);
    /// }
    /// ```
    #[must_use]
    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item {
        let mut cur_i = i;
        let mut result: u32 = 0;

        let mut shift = 0;

        for level in 0..self.n_levels {
            if cur_i >= self.lens[level] {
                break;
            }

            self.qvs[level].prefetch_info(cur_i);
            let symbol = self.qvs[level].get_unchecked(cur_i);
            result = (result << 2) | symbol as u32;

            let offset = unsafe { self.qvs[level].occs_smaller_unchecked(symbol) };
            cur_i = self.qvs[level].rank_unchecked(symbol, cur_i) + offset;

            shift += 2;
        }

        // println!("found result len:{}, repr:{}", shift, result);

        //find the symbol
        let idx = self.codes_decode[shift]
            .binary_search_by_key(&result, |(x, _)| *x)
            .expect("could not translate symbol");

        T::from(self.codes_decode[shift][idx].1).unwrap()
    }
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> From<Vec<T>>
    for HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    RS: RSforWT,
{
    fn from(mut v: Vec<T>) -> Self {
        HuffQWaveletTree::new(&mut v[..])
    }
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> RankUnsigned
    for HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    RS: RSforWT,
{
    /// Returns the rank of `symbol` up to position `i` **excluded**.
    ///
    /// `None` is returned if `i` is out of bound or if `symbol` is not valid
    /// (i.e., there is no occurence of the symbol in the original sequence).
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::{HQWT256, RankUnsigned};
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = HQWT256::from(data);
    ///
    /// assert_eq!(qwt.rank(6, 1), None);  // Too large symbol
    /// assert_eq!(qwt.rank(1, 2), Some(1));
    /// assert_eq!(qwt.rank(3, 8), Some(1));
    /// assert_eq!(qwt.rank(1, 0), Some(0));
    /// assert_eq!(qwt.rank(1, 9), None);  // Too large position
    /// ```
    #[must_use]
    #[inline(always)]
    fn rank(&self, symbol: Self::Item, i: usize) -> Option<usize> {
        if i > self.n
            || symbol.as_() >= self.codes_encode.len()
            || self.codes_encode[symbol.as_()].len == 0
        {
            return None;
        }

        // SAFETY: Check above guarantees we are not out of bound
        Some(unsafe { self.rank_unchecked(symbol, i) })
    }

    /// Returns rank of `symbol` up to position `i` **excluded**.
    ///
    /// # Safety
    ///
    /// Calling this method with a position `i` larger than the size of the sequence
    /// or with an invalid symbol is undefined behavior.
    ///
    /// Users must ensure that the position `i` is within the bounds of the sequence
    /// and that the symbol is valid.
    ///
    /// # Examples
    ///
    /// ```
    /// use qwt::{HQWT256, RankUnsigned};
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = HQWT256::from(data);
    ///
    /// unsafe {
    ///     assert_eq!(qwt.rank_unchecked(1, 2), 1);
    /// }
    /// ```
    #[must_use]
    #[inline(always)]
    unsafe fn rank_unchecked(&self, symbol: Self::Item, i: usize) -> usize {
        let mut cur_i = i;
        let mut cur_p = 0;

        let code = &self.codes_encode[symbol.as_() as usize];
        let mut shift: i64 = code.len as i64 - 2;
        let repr = code.content;
        let mut level = 0;

        while shift >= 0 {
            let two_bits = ((repr >> shift as usize) & 3) as u8;

            let offset = unsafe { self.qvs[level].occs_smaller_unchecked(two_bits) };
            cur_p = self.qvs[level].rank_unchecked(two_bits, cur_p) + offset;
            cur_i = self.qvs[level].rank_unchecked(two_bits, cur_i) + offset;

            level += 1;
            shift -= 2;
        }

        cur_i - cur_p
    }
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> SelectUnsigned
    for HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    RS: RSforWT,
{
    /// Returns the position of the `i+1`-th occurrence of symbol `symbol`.
    ///
    /// `None` is returned if the is no (i+1)th such occurrence for the symbol
    /// or if `symbol` is not valid (i.e., it is greater than or equal to the alphabet size).
    ///
    /// # Examples
    /// ```
    /// use qwt::{HQWT256, SelectUnsigned};
    ///
    /// let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    ///
    /// let qwt = HQWT256::from(data);
    ///
    /// assert_eq!(qwt.select(1, 1), Some(2));
    /// assert_eq!(qwt.select(0, 1), Some(3));
    /// assert_eq!(qwt.select(0, 2), None);
    /// assert_eq!(qwt.select(1, 0), Some(0));
    /// assert_eq!(qwt.select(5, 0), Some(6));
    /// assert_eq!(qwt.select(6, 1), None);
    /// ```    
    #[must_use]
    #[inline(always)]
    fn select(&self, symbol: Self::Item, i: usize) -> Option<usize> {
        if symbol.as_() >= self.codes_encode.len()
            || self.codes_encode[symbol.as_() as usize].len == 0
        {
            return None;
        }

        let mut path_off = Vec::with_capacity(self.n_levels);
        let mut rank_path_off = Vec::with_capacity(self.n_levels);

        let code = &self.codes_encode[symbol.as_() as usize];
        let mut shift: i64 = code.len as i64 - 2;
        let repr = code.content;

        let mut b = 0;

        let mut level = 0;
        while shift >= 0 {
            path_off.push(b);

            let two_bits = ((repr >> shift as usize) & 3) as u8;

            let rank_b = self.qvs[level].rank(two_bits, b)?;

            b = rank_b + unsafe { self.qvs[level].occs_smaller_unchecked(two_bits) };
            rank_path_off.push(rank_b);

            level += 1;
            shift -= 2;
        }

        shift = 0;
        let mut result = i;
        for level in (0..level).rev() {
            b = path_off[level];
            let rank_b = rank_path_off[level];
            let two_bits = ((repr >> shift as usize) & 3) as u8;

            result = self.qvs[level].select(two_bits, rank_b + result)? - b;
            shift += 2;
        }

        Some(result)
    }

    /// Returns the position of the `i+1`-th occurrence of symbol `symbol`.
    ///
    /// # Safety
    ///
    /// Calling this method with a value of `i` larger than the number of occurrences
    /// of the `symbol`, or if the `symbol` is not valid, is undefined behavior.
    ///
    /// In the current implementation, there is no efficiency reason to prefer this
    /// unsafe `select` over the safe one.
    #[must_use]
    #[inline(always)]
    unsafe fn select_unchecked(&self, symbol: Self::Item, i: usize) -> usize {
        self.select(symbol, i).unwrap()
    }
}

impl<T, RS: SpaceUsage, const WITH_PREFETCH_SUPPORT: bool> SpaceUsage
    for HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
{
    /// Gives the space usage in bytes of the struct.
    fn space_usage_byte(&self) -> usize {
        let space_prefetch_support: usize = self
            .prefetch_support
            .iter()
            .flatten()
            .map(|ps| ps.space_usage_byte())
            .sum();

        8 + 8
            + 256 * 8  // 256 + 2 * sizeof(u32) codes_encode
            + self.codes_decode //codes_decode
                .iter()
                .fold(0, |a, v| a + v.len() * (4+1))
            + self.lens.len() * 8
            + self
                .qvs
                .iter()
                .fold(0, |acc, ds| acc + ds.space_usage_byte())
        + space_prefetch_support
    }
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> AsRef<HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>>
    for HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
{
    fn as_ref(&self) -> &HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT> {
        self
    }
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> IntoIterator
    for HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    RS: RSforWT,
{
    type IntoIter = WTIterator<
        T,
        HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>,
        HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>,
    >;
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

impl<'a, T, RS, const WITH_PREFETCH_SUPPORT: bool> IntoIterator
    for &'a HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    RS: RSforWT,
{
    type IntoIter = WTIterator<
        T,
        HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>,
        &'a HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>,
    >;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, RS, const WITH_PREFETCH_SUPPORT: bool> FromIterator<T>
    for HuffQWaveletTree<T, RS, WITH_PREFETCH_SUPPORT>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    RS: RSforWT,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        HuffQWaveletTree::new(&mut iter.into_iter().collect::<Vec<T>>())
    }
}

#[cfg(test)]
mod tests;
