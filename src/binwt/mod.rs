use std::{
    collections::HashMap,
    marker::PhantomData,
    ops::{Bound, Range, RangeBounds},
};

use mem_dbg::{MemDbg, MemSize};
use minimum_redundancy::{BitsPerFragment, Coding};
use num_traits::AsPrimitive;
use serde::{Deserialize, Deserializer, Serialize};

use crate::{
    quadwt::huffqwt::PrefixCode,
    utils::{msb, stable_partition_of_2, stable_partition_of_2_with_codes},
    AccessBin, AccessUnsigned, BinWTSupport, BitVector, BitVectorMut, OccsRangeUnsigned,
    RankUnsigned, SelectUnsigned, WTIndexable, WTIterator,
};

pub trait BinRSforWT: From<BitVector> + BinWTSupport + MemSize + MemDbg + Default {}
impl<T> BinRSforWT for T where T: From<BitVector> + BinWTSupport + MemSize + MemDbg + Default {}

#[derive(Default, Clone, PartialEq, Serialize, MemSize, MemDbg, Debug)]
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

#[derive(Deserialize)]
struct WaveletTreeSerde<T, BRS> {
    n: usize,
    n_levels: usize,
    sigma: Option<T>,
    codes_encode: Option<Vec<PrefixCode>>,
    codes_decode: Option<Vec<Vec<(u32, T)>>>,
    bvs: Vec<BRS>,
    lens: Vec<usize>,
    phantom_data: PhantomData<T>,
}

impl<'de, T, BRS, const COMPRESSED: bool> Deserialize<'de> for WaveletTree<T, BRS, COMPRESSED>
where
    T: WTIndexable + Deserialize<'de>,
    usize: AsPrimitive<T>,
    BRS: BinRSforWT + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let decoded = WaveletTreeSerde::<T, BRS>::deserialize(deserializer)?;
        let _ = decoded.phantom_data;
        let tree = Self {
            n: decoded.n,
            n_levels: decoded.n_levels,
            sigma: decoded.sigma,
            codes_encode: decoded.codes_encode,
            codes_decode: decoded.codes_decode,
            bvs: decoded.bvs,
            lens: decoded.lens,
            phantom_data: PhantomData,
        };
        tree.validate_deserialized()
            .map_err(serde::de::Error::custom)?;
        Ok(tree)
    }
}

struct LenInfo(usize, u32); // symbol, len

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

        assignments[f[j].0] = PrefixCode {
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
    fn validate_deserialized(&self) -> Result<(), &'static str> {
        if self.bvs.len() != self.n_levels || self.lens.len() != self.n_levels {
            return Err("wavelet-tree level metadata has inconsistent lengths");
        }
        if self.n == 0 {
            if self.n_levels != 0
                || !self.bvs.is_empty()
                || !self.lens.is_empty()
                || self.sigma.is_some()
                || self.codes_encode.is_some()
                || self.codes_decode.is_some()
            {
                return Err("empty wavelet tree contains non-empty metadata");
            }
            return Ok(());
        }

        let type_levels = std::mem::size_of::<T>()
            .checked_mul(u8::BITS as usize)
            .ok_or("wavelet-tree type width overflow")?;
        if self.n_levels == 0 || self.n_levels > type_levels {
            return Err("wavelet-tree level count exceeds the symbol type width");
        }
        if COMPRESSED && self.n_levels > u32::BITS as usize {
            return Err("compressed wavelet-tree code exceeds 32 bits");
        }

        for (level, (bv, &len)) in self.bvs.iter().zip(&self.lens).enumerate() {
            if len > self.n {
                return Err("wavelet-tree level exceeds the sequence length");
            }
            if (len > 0 && AccessBin::get(bv, len - 1).is_none())
                || AccessBin::get(bv, len).is_some()
            {
                return Err("wavelet-tree level length disagrees with its bitvector");
            }
            if !COMPRESSED && len != self.n {
                return Err("plain wavelet-tree levels must match the sequence length");
            }
            if COMPRESSED && level > 0 && len > self.lens[level - 1] {
                return Err("compressed wavelet-tree level lengths must be non-increasing");
            }
        }
        if COMPRESSED && self.lens.first().copied() != Some(self.n) {
            return Err("compressed wavelet-tree first level must match the sequence length");
        }

        if COMPRESSED {
            if self.sigma.is_some() {
                return Err("compressed wavelet tree must not contain plain sigma metadata");
            }
            let encode = self
                .codes_encode
                .as_ref()
                .ok_or("compressed wavelet tree is missing its encoder")?;
            let decode = self
                .codes_decode
                .as_ref()
                .ok_or("compressed wavelet tree is missing its decoder")?;
            if decode.len() != self.n_levels + 1 {
                return Err("compressed wavelet-tree decoder has the wrong depth");
            }
            for row in decode {
                if !row.windows(2).all(|pair| pair[0].0 < pair[1].0) {
                    return Err("compressed wavelet-tree decoder rows must be strictly sorted");
                }
            }
            for (symbol, code) in encode.iter().enumerate() {
                if code.len == 0 {
                    continue;
                }
                let code_len = code.len as usize;
                if code_len > self.n_levels
                    || (code_len < u32::BITS as usize && code.content >= (1u32 << code_len))
                {
                    return Err("compressed wavelet-tree encoder contains an invalid code");
                }
                let decoded_symbol = decode[code_len]
                    .binary_search_by_key(&code.content, |(content, _)| *content)
                    .ok()
                    .and_then(|index| decode[code_len].get(index))
                    .map(|(_, value)| value.as_());
                if decoded_symbol != Some(symbol) {
                    return Err("compressed wavelet-tree encode/decode tables disagree");
                }
            }
            for (code_len, row) in decode.iter().enumerate().skip(1) {
                for &(content, symbol) in row {
                    let Some(code) = encode.get(symbol.as_()) else {
                        return Err("compressed wavelet-tree decoder symbol is out of range");
                    };
                    if code.len as usize != code_len || code.content != content {
                        return Err("compressed wavelet-tree decode/encode tables disagree");
                    }
                }
            }
        } else {
            if self.codes_encode.is_some() || self.codes_decode.is_some() {
                return Err("plain wavelet tree contains compressed-code metadata");
            }
            let sigma = self
                .sigma
                .ok_or("plain wavelet tree is missing sigma metadata")?;
            if self.n_levels != msb(sigma) as usize + 1 {
                return Err("plain wavelet-tree sigma disagrees with its level count");
            }
        }

        let mut actual_sigma = 0usize;
        for index in 0..self.n {
            let symbol = AccessUnsigned::get(self, index)
                .ok_or("wavelet-tree payload cannot decode every declared position")?;
            actual_sigma = actual_sigma.max(symbol.as_());
        }
        if !COMPRESSED && self.sigma.is_none_or(|sigma| sigma.as_() != actual_sigma) {
            return Err("plain wavelet-tree sigma disagrees with its encoded payload");
        }
        Ok(())
    }

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
                    let cur_code = codes_encode.as_ref().unwrap().get(s.as_()).expect(
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

    #[inline(always)]
    fn get(&self, i: usize) -> Option<Self::Item> {
        if i >= self.n {
            return None;
        }
        let mut cur_i = i;
        let mut result: u32 = 0;
        let mut shift = 0;
        for level in 0..self.n_levels {
            if COMPRESSED && cur_i >= *self.lens.get(level)? {
                break;
            }
            let bv = self.bvs.get(level)?;
            let symbol = bv.get(cur_i)?;
            result = (result << 1) | u32::from(symbol);
            let ones = bv.rank1(cur_i)?;
            cur_i = if symbol {
                ones.checked_add(bv.count_zeros())?
            } else {
                cur_i.checked_sub(ones)?
            };
            shift += 1;
        }
        if COMPRESSED {
            let leaves = self.codes_decode.as_ref()?.get(shift)?;
            let index = leaves
                .binary_search_by_key(&result, |(code, _)| *code)
                .ok()?;
            Some(leaves.get(index)?.1)
        } else {
            T::from(result)
        }
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
                tmp + self.bvs[level].count_zeros()
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

impl<T, BRS, const COMPRESSED: bool> OccsRangeUnsigned for WaveletTree<T, BRS, COMPRESSED>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    BRS: BinRSforWT,
{
    type Iter<'a>
        = OccsRangeIter<'a, T, BRS, COMPRESSED>
    where
        Self: 'a;

    /// Returns an iterator over the number of occurrences of symbols in the provided range having at least
    /// one occurrence. Symbols that do not appear in the range are not yielded.
    ///
    /// Guaranteed to iterate in lexicographic symbol order if the tree is not compressed. If compressed, it
    /// will iterate in an undefined order.
    ///
    /// Returns `None` if the provided range is out-of-bounds.
    fn occs_range<R: RangeBounds<usize>>(&self, range: R) -> Option<Self::Iter<'_>> {
        let start = match range.start_bound() {
            Bound::Included(start) => *start,
            Bound::Excluded(start) => start.checked_add(1)?,
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(end) => end.checked_add(1)?,
            Bound::Excluded(end) => *end,
            Bound::Unbounded => self.n,
        };

        if end > self.n || start > end {
            return None;
        }

        Some(unsafe { self.occs_range_unchecked(start..end) })
    }

    /// Returns an iterator over the number of occurrences of symbols in the provided range having at least
    /// one occurrence. Symbols that do not appear in the range are not yielded.
    ///
    /// Guaranteed to iterate in lexicographic symbol order if the tree is not compressed. If compressed, it
    /// will iterate in an undefined order.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds range is undefined behavior.
    unsafe fn occs_range_unchecked(&self, range: Range<usize>) -> Self::Iter<'_> {
        if range.start == range.end {
            let stack = vec![];
            return OccsRangeIter { tree: self, stack };
        }

        let mut stack = Vec::with_capacity(self.n_levels + 1);

        stack.push(OccsRangeFrame {
            range,
            level: 0,
            bit_path: 0,
        });

        OccsRangeIter { tree: self, stack }
    }
}

pub struct OccsRangeIter<'a, T, BRS, const COMPRESSED: bool> {
    tree: &'a WaveletTree<T, BRS, COMPRESSED>,
    stack: Vec<OccsRangeFrame>,
}

struct OccsRangeFrame {
    range: Range<usize>,
    level: usize,
    bit_path: usize,
}

impl<'a, T, BRS, const COMPRESSED: bool> Iterator for OccsRangeIter<'a, T, BRS, COMPRESSED>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    BRS: BinRSforWT,
{
    type Item = (T, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(cur) = self.stack.pop() {
            // have we reached the bottom of the tree? huffman tree needs slightly different logic to check
            if COMPRESSED {
                let leaves = self.tree.codes_decode.as_ref()?.get(cur.level)?;

                if let Ok(idx) = leaves.binary_search_by_key(&(cur.bit_path as u32), |(c, _)| *c) {
                    let leaf = leaves.get(idx)?;

                    // prefix free property means we don't descend further here
                    return Some((leaf.1, cur.range.end - cur.range.start));
                }
            } else if cur.level == self.tree.n_levels {
                return Some((cur.bit_path.as_(), cur.range.end - cur.range.start));
            }

            let bv = self.tree.bvs.get(cur.level)?;

            // right child (pushing it first makes non-huffman iterate in lexicographic symbol order)

            let r_lo = bv.rank1(cur.range.start)?;
            let r_hi = bv.rank1(cur.range.end)?;

            if r_hi > r_lo {
                let offset = bv.count_zeros();

                let frame = OccsRangeFrame {
                    range: offset + r_lo..offset + r_hi,
                    level: cur.level + 1,
                    bit_path: (cur.bit_path << 1) | 1,
                };

                self.stack.push(frame);
            }

            // left child (we can derive the range from ^ rank calls)

            let l_lo = cur.range.start - r_lo;
            let l_hi = cur.range.end - r_hi;

            if l_hi > l_lo {
                let frame = OccsRangeFrame {
                    range: l_lo..l_hi,
                    level: cur.level + 1,
                    bit_path: (cur.bit_path << 1), // | 0
                };

                self.stack.push(frame);
            }
        }

        None
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

        let (symbol_len, repr) = if COMPRESSED {
            let code = self.codes_encode.as_ref()?.get(symbol.as_())?;
            if code.len == 0 {
                return None;
            }
            (code.len as usize, code.content)
        } else {
            if symbol > *self.sigma.as_ref()? {
                return None;
            }
            (self.n_levels, symbol.as_() as u32)
        };
        let mut cur_i = i;
        let mut cur_p = 0usize;
        for level in 0..symbol_len {
            let bit = ((repr >> (symbol_len - level - 1)) & 1) == 1;
            let bv = self.bvs.get(level)?;
            let offset = bv.count_zeros();
            let tmp_p = bv.rank1(cur_p)?;
            let tmp_i = bv.rank1(cur_i)?;
            cur_p = if bit {
                tmp_p.checked_add(offset)?
            } else {
                cur_p.checked_sub(tmp_p)?
            };
            cur_i = if bit {
                tmp_i.checked_add(offset)?
            } else {
                cur_i.checked_sub(tmp_i)?
            };
        }
        cur_i.checked_sub(cur_p)
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, symbol: Self::Item, i: usize) -> usize {
        let mut cur_i = i;
        let mut cur_p = 0;

        let symbol_len;
        let repr;

        if COMPRESSED {
            let code = &self.codes_encode.as_ref().unwrap()[symbol.as_()];
            symbol_len = code.len as usize;
            repr = code.content;
        } else {
            repr = symbol.as_() as u32;
            symbol_len = self.n_levels;
        }

        for level in 0..symbol_len {
            let bit = ((repr >> (symbol_len - level - 1)) & 1) == 1;

            let offset = self.bvs[level].count_zeros();

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
        let (symbol_len, repr) = if COMPRESSED {
            let code = self.codes_encode.as_ref()?.get(symbol.as_())?;
            if code.len == 0 {
                return None;
            }
            (code.len as usize, code.content)
        } else {
            if symbol > *self.sigma.as_ref()? {
                return None;
            }
            (self.n_levels, symbol.as_() as u32)
        };
        let mut b = 0;

        let mut path_off = Vec::with_capacity(symbol_len);
        let mut rank_path_off = Vec::with_capacity(symbol_len);

        for level in 0..symbol_len {
            path_off.push(b);

            let bit = ((repr >> (symbol_len - level - 1)) & 1) == 1;
            let bv = self.bvs.get(level)?;

            let rank_b = if bit { bv.rank1(b) } else { bv.rank0(b) }?;

            b = rank_b.checked_add(if bit { bv.count_zeros() } else { 0 })?;

            rank_path_off.push(rank_b);
        }

        let mut result = i;
        for level in (0..symbol_len).rev() {
            b = path_off[level];
            let rank_b = rank_path_off[level];
            let bit = ((repr >> (symbol_len - level - 1)) & 1) == 1;
            let bv = self.bvs.get(level)?;

            let selected = if bit {
                bv.select1(rank_b.checked_add(result)?)
            } else {
                bv.select0(rank_b.checked_add(result)?)
            }?;
            result = selected.checked_sub(b)?;
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

#[cfg(test)]
mod tests;
