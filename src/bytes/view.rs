//! Zero-copy borrowed views over `QWTB` / `HQWB` containers.
//!
//! The caller owns the underlying bytes (mmap, `AlignedBuf`, …). These views
//! cast POD slices in place and never allocate for level payloads. Huffman
//! code tables are small and are decoded into owned side allocations.

use super::level::{HqwtLevelDir, LevelDir, RSQVectorView};
use super::{
    align_up, checked_region, ensure_le, LayoutError, FLAG_B512, FLAG_PREFETCH, FORMAT_VERSION,
    HEADER_SIZE, HQWB_MAGIC, HQWT_LEVEL_DIR_SIZE, LEVEL_DIR_SIZE, QWTB_MAGIC,
};
use crate::quadwt::huffqwt::PrefixCode;
use crate::quadwt::WTIndexable;
use crate::{AccessUnsigned, RankUnsigned, SelectUnsigned};
use num_traits::AsPrimitive;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::Range;


// ── Plain QWT view ──────────────────────────────────────────────────────────

/// Borrowing plain quad wavelet tree over a `QWTB` blob.
///
/// Implements [`AccessUnsigned`], [`RankUnsigned`], and [`SelectUnsigned`].
/// The underlying `&[u8]` must outlive the view and its absolute base address
/// must be 64-byte aligned so POD casts of `DataLine` / `SuperblockPlain`
/// succeed (format offsets are already 64-aligned).
#[derive(Clone, Debug)]
pub struct QwtView<'a, T, const B_SIZE: usize = 256> {
    n: usize,
    n_levels: usize,
    sigma: T,
    levels: Vec<RSQVectorView<'a, B_SIZE>>,
    _marker: PhantomData<&'a [u8]>,
}

impl<'a, T, const B_SIZE: usize> QwtView<'a, T, B_SIZE>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    u8: AsPrimitive<T>,
{
    /// Open a zero-copy view over a `QWTB` blob.
    ///
    /// # Errors
    /// Layout / magic / version / alignment failures — see [`LayoutError`].
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, LayoutError> {
        ensure_le()?;
        if bytes.len() < HEADER_SIZE {
            return Err(LayoutError::Truncated);
        }
        if &bytes[0..4] != QWTB_MAGIC {
            return Err(LayoutError::BadMagic);
        }
        let mut o = 4;
        let version = get_u16(bytes, &mut o);
        if version != FORMAT_VERSION {
            return Err(LayoutError::BadVersion);
        }
        let flags = bytes[o];
        o += 1;
        if flags & FLAG_PREFETCH != 0 {
            return Err(LayoutError::PrefetchNotSupported);
        }
        let want_b512 = flags & FLAG_B512 != 0;
        if (B_SIZE == 512) != want_b512 {
            return Err(LayoutError::Inconsistent {
                detail: "B_SIZE flag does not match requested type",
            });
        }
        let t_width = bytes[o];
        o += 1;
        if t_width as usize != size_of::<T>() {
            return Err(LayoutError::Inconsistent {
                detail: "t_width does not match T",
            });
        }
        let n = get_u64(bytes, &mut o) as usize;
        let sigma_u = get_u64(bytes, &mut o) as usize;
        let sigma: T = sigma_u.as_();
        let n_levels = get_u16(bytes, &mut o) as usize;
        let _ = o;

        let dir_end = HEADER_SIZE + n_levels * LEVEL_DIR_SIZE;
        if bytes.len() < dir_end {
            return Err(LayoutError::Truncated);
        }

        let mut levels = Vec::with_capacity(n_levels);
        for li in 0..n_levels {
            let dir_off = HEADER_SIZE + li * LEVEL_DIR_SIZE;
            let dir = LevelDir::read(&bytes[dir_off..dir_off + LEVEL_DIR_SIZE])?;
            levels.push(RSQVectorView::from_dir(bytes, &dir)?);
        }

        Ok(Self {
            n,
            n_levels,
            sigma,
            levels,
            _marker: PhantomData,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    #[inline]
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }

    #[inline]
    pub fn sigma_raw(&self) -> T {
        self.sigma
    }

    #[inline]
    pub fn levels(&self) -> &[RSQVectorView<'a, B_SIZE>] {
        &self.levels
    }

    /// Returns the symbol at position `i` together with `rank(symbol, i + 1)`
    /// (occurrences of that symbol in `0..=i`) in a single tree descent.
    ///
    /// Equivalent to `(get(i), rank(get(i), i + 1))` but shares the wavelet
    /// path for both queries. Returns `None` if `i` is out of bounds.
    #[inline(always)]
    #[must_use]
    pub fn get_and_rank(&self, i: usize) -> Option<(T, usize)> {
        if i >= self.n || self.n_levels == 0 {
            return None;
        }
        // SAFETY: bounds checked above
        Some(unsafe { self.get_and_rank_unchecked(i) })
    }

    /// Returns the symbol at position `i` together with `rank(symbol, i + 1)`
    /// in a single tree descent.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline(always)]
    #[must_use]
    pub unsafe fn get_and_rank_unchecked(&self, i: usize) -> (T, usize) {
        let mut result = T::zero();
        let mut cur_i = i;
        let mut cur_p = 0usize;

        for level in 0..self.n_levels - 1 {
            let lv = &self.levels[level];
            let symbol = lv.get_unchecked(cur_i);
            result = (result << 2) | (symbol as usize).as_();
            let offset = lv.occs_smaller_unchecked(symbol);
            cur_p = lv.rank_unchecked(symbol, cur_p) + offset;
            cur_i = lv.rank_unchecked(symbol, cur_i) + offset;
        }

        let last = self.n_levels - 1;
        let lv = &self.levels[last];
        let symbol = lv.get_unchecked(cur_i);
        let result = (result << 2) | (symbol as usize).as_();
        // rank(c, i) on the last level (no occs offset), then +1 → rank(c, i+1)
        let r_i = lv.rank_unchecked(symbol, cur_i);
        let r_p = lv.rank_unchecked(symbol, cur_p);
        let rank_inclusive = r_i - r_p + 1;
        (result, rank_inclusive)
    }

    /// Bulk-extract symbols in a contiguous position range as an **ascending
    /// multiset** (sorted by symbol value, with multiplicity).
    ///
    /// This is **not** position order: the output is algebraically equal to
    /// sorting the multiset `{ get(i) | i ∈ range }`, not to iterating
    /// `get` over the range in index order.
    ///
    /// Implementation walks the wavelet tree level-by-level, partitioning each
    /// contiguous range into at most four child ranges (one per 2-bit digit)
    /// via `rank_all` + `occs_smaller`. Singleton ranges short-circuit to a single
    /// remaining-path descent. Empty / out-of-bounds ranges return an empty
    /// vector.
    #[inline]
    #[must_use]
    pub fn extract_range(&self, range: Range<usize>) -> Vec<T> {
        if range.start >= range.end || range.end > self.n || self.n_levels == 0 {
            return Vec::new();
        }
        let n = range.end - range.start;
        let mut out = vec![T::zero(); n];
        // SAFETY: range is in-bounds on the top level; child ranges stay valid
        // by wavelet matrix invariants.
        unsafe {
            self.extract_range_sorted_rec(0, range.start, range.end, T::zero(), &mut out);
        }
        out
    }

    /// Bulk-extract **ascending distinct** symbols for a contiguous position
    /// range (one entry per unique value, sorted).
    ///
    /// Same expand-tree as [`Self::extract_range`], but leaf digit runs emit a
    /// single symbol instead of `count` copies — no intermediate multiset and
    /// no post-dedup pass. Algebraically equal to sort+dedup of per-row
    /// [`AccessUnsigned::get`] over the range.
    ///
    /// Empty / out-of-bounds ranges return an empty vector.
    #[inline]
    #[must_use]
    pub fn extract_range_distinct(&self, range: Range<usize>) -> Vec<T> {
        let mut out = Vec::new();
        self.extract_range_distinct_into(range, &mut out);
        out
    }

    /// Like [`Self::extract_range_distinct`], writing into `out` (cleared first).
    ///
    /// Prefer this when the caller owns a reusable buffer.
    #[inline]
    pub fn extract_range_distinct_into(&self, range: Range<usize>, out: &mut Vec<T>) {
        out.clear();
        if range.start >= range.end || range.end > self.n || self.n_levels == 0 {
            return;
        }
        out.reserve(range.end - range.start);
        // SAFETY: range is in-bounds on the top level.
        unsafe {
            self.extract_range_distinct_rec(0, range.start, range.end, T::zero(), out);
        }
    }

    /// Finish one remaining position starting at `level` with path `prefix`.
    ///
    /// Used when a child range has `count == 1` so we avoid expanding a 4-way
    /// tree for a singleton.
    #[inline]
    unsafe fn extract_singleton_from(&self, mut level: usize, mut pos: usize, mut prefix: T) -> T {
        while level + 1 < self.n_levels {
            let lv = &self.levels[level];
            let dig = lv.get_unchecked(pos);
            prefix = (prefix << 2) | (dig as usize).as_();
            let offset = lv.occs_smaller_unchecked(dig);
            pos = lv.rank_unchecked(dig, pos) + offset;
            level += 1;
        }
        let dig = self.levels[level].get_unchecked(pos);
        (prefix << 2) | (dig as usize).as_()
    }

    /// Recursive sorted multiset fill: write ascending symbols into `out`.
    ///
    /// # Safety
    /// `start..end` must be a valid contiguous range at `level` of the wavelet
    /// matrix; `out.len() == end - start`.
    unsafe fn extract_range_sorted_rec(
        &self,
        level: usize,
        start: usize,
        end: usize,
        prefix: T,
        out: &mut [T],
    ) {
        debug_assert_eq!(out.len(), end - start);
        if start >= end {
            return;
        }
        // Singleton short-circuit: one position → single descent.
        if end - start == 1 {
            out[0] = self.extract_singleton_from(level, start, prefix);
            return;
        }

        let lv = &self.levels[level];
        let last = level + 1 == self.n_levels;

        // Partition [start, end) by 2-bit digit via rank_all at both ends.
        // SAFETY: start/end valid at this level by caller contract.
        let ranks_s = lv.rank_all_unchecked(start);
        let ranks_e = lv.rank_all_unchecked(end);
        let counts = [
            ranks_e[0] - ranks_s[0],
            ranks_e[1] - ranks_s[1],
            ranks_e[2] - ranks_s[2],
            ranks_e[3] - ranks_s[3],
        ];

        if last {
            // Fill runs of (prefix<<2)|b by digit order — already ascending.
            let mut off = 0usize;
            for (b, &cnt) in counts.iter().enumerate() {
                if cnt == 0 {
                    continue;
                }
                let sym: T = (prefix << 2) | b.as_();
                out[off..off + cnt].fill(sym);
                off += cnt;
            }
            debug_assert_eq!(off, out.len());
            return;
        }

        // Partition out into digit-order child slices (no intermediate Vecs).
        let mut off = 0usize;
        for (b, &cnt) in counts.iter().enumerate() {
            if cnt == 0 {
                continue;
            }
            // SAFETY: b in 0..4
            let offset = lv.occs_smaller_unchecked(b as u8);
            let child_start = offset + ranks_s[b];
            let child_end = child_start + cnt;
            let child_prefix: T = (prefix << 2) | b.as_();
            self.extract_range_sorted_rec(
                level + 1,
                child_start,
                child_end,
                child_prefix,
                &mut out[off..off + cnt],
            );
            off += cnt;
        }
        debug_assert_eq!(off, out.len());
    }

    /// Recursive ascending-distinct emit into `out`.
    ///
    /// # Safety
    /// `start..end` must be a valid contiguous range at `level`.
    unsafe fn extract_range_distinct_rec(
        &self,
        level: usize,
        start: usize,
        end: usize,
        prefix: T,
        out: &mut Vec<T>,
    ) {
        if start >= end {
            return;
        }
        if end - start == 1 {
            out.push(self.extract_singleton_from(level, start, prefix));
            return;
        }

        let lv = &self.levels[level];
        let last = level + 1 == self.n_levels;

        // SAFETY: start/end valid at this level by caller contract.
        let ranks_s = lv.rank_all_unchecked(start);
        let ranks_e = lv.rank_all_unchecked(end);
        let counts = [
            ranks_e[0] - ranks_s[0],
            ranks_e[1] - ranks_s[1],
            ranks_e[2] - ranks_s[2],
            ranks_e[3] - ranks_s[3],
        ];

        if last {
            for (b, &cnt) in counts.iter().enumerate() {
                if cnt != 0 {
                    let sym: T = (prefix << 2) | b.as_();
                    out.push(sym);
                }
            }
            return;
        }

        for (b, &cnt) in counts.iter().enumerate() {
            if cnt == 0 {
                continue;
            }
            let offset = lv.occs_smaller_unchecked(b as u8);
            let child_start = offset + ranks_s[b];
            let child_end = child_start + cnt;
            let child_prefix: T = (prefix << 2) | b.as_();
            self.extract_range_distinct_rec(
                level + 1,
                child_start,
                child_end,
                child_prefix,
                out,
            );
        }
    }
}

impl<'a, T, const B_SIZE: usize> AccessUnsigned for QwtView<'a, T, B_SIZE>

where
    T: WTIndexable,
    usize: AsPrimitive<T>,
{
    type Item = T;

    #[inline(always)]
    fn get(&self, i: usize) -> Option<Self::Item> {
        if i >= self.n || self.n_levels == 0 {
            return None;
        }
        Some(unsafe { self.get_unchecked(i) })
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item {
        debug_assert!(self.n_levels > 0);
        let mut result = T::zero();
        let mut cur_i = i;
        for level in 0..self.n_levels - 1 {
            let lv = &self.levels[level];
            let symbol = lv.get_unchecked(cur_i);
            result = (result << 2) | (symbol as usize).as_();
            let offset = lv.occs_smaller_unchecked(symbol);
            cur_i = lv.rank_unchecked(symbol, cur_i) + offset;
        }
        let lv = &self.levels[self.n_levels - 1];
        let symbol = lv.get_unchecked(cur_i);
        (result << 2) | (symbol as usize).as_()
    }
}

impl<'a, T, const B_SIZE: usize> RankUnsigned for QwtView<'a, T, B_SIZE>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
{
    #[inline(always)]
    fn rank(&self, symbol: Self::Item, i: usize) -> Option<usize> {
        if self.n_levels == 0 {
            return if i <= self.n { Some(0) } else { None };
        }
        if i > self.n || symbol > self.sigma {
            return None;
        }
        Some(unsafe { self.rank_unchecked(symbol, i) })
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, symbol: Self::Item, i: usize) -> usize {
        debug_assert!(self.n_levels > 0);
        let mut shift: i64 = (2 * (self.n_levels - 1)) as i64;
        let mut cur_i = i;
        let mut cur_p = 0usize;

        for level in 0..self.n_levels - 1 {
            let two_bits: u8 = ((symbol >> shift as usize).as_() & 3) as u8;
            let lv = &self.levels[level];
            let offset = lv.occs_smaller_unchecked(two_bits);
            cur_p = lv.rank_unchecked(two_bits, cur_p) + offset;
            cur_i = lv.rank_unchecked(two_bits, cur_i) + offset;
            shift -= 2;
        }

        let two_bits: u8 = ((symbol >> shift as usize).as_() & 3) as u8;
        let lv = &self.levels[self.n_levels - 1];
        cur_i = lv.rank_unchecked(two_bits, cur_i);
        cur_p = lv.rank_unchecked(two_bits, cur_p);
        cur_i - cur_p
    }
}

impl<'a, T, const B_SIZE: usize> SelectUnsigned for QwtView<'a, T, B_SIZE>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
{
    #[inline(always)]
    fn select(&self, symbol: Self::Item, i: usize) -> Option<usize> {
        if self.n_levels == 0 || symbol > self.sigma {
            return None;
        }

        let mut path_off = Vec::with_capacity(self.n_levels);
        let mut rank_path_off = Vec::with_capacity(self.n_levels);

        let mut b = 0usize;
        let mut shift: i64 = 2 * (self.n_levels - 1) as i64;

        for level in 0..self.n_levels {
            path_off.push(b);
            let two_bits = (symbol >> shift as usize).as_() & 3;
            let lv = &self.levels[level];
            if b > lv.len() {
                return None;
            }
            let rank_b = unsafe { lv.rank_unchecked(two_bits as u8, b) };
            b = rank_b + unsafe { lv.occs_smaller_unchecked(two_bits as u8) };
            shift -= 2;
            rank_path_off.push(rank_b);
        }

        shift = 0;
        let mut result = i;
        for level in (0..self.n_levels).rev() {
            b = path_off[level];
            let rank_b = rank_path_off[level];
            let two_bits = (symbol >> shift as usize).as_() & 3;
            let lv = &self.levels[level];
            result = lv.select(two_bits as u8, rank_b + result)? - b;
            shift += 2;
        }

        Some(result)
    }

    #[inline(always)]
    unsafe fn select_unchecked(&self, symbol: Self::Item, i: usize) -> usize {
        self.select(symbol, i).unwrap()
    }
}

// ── Huffman QWT view ────────────────────────────────────────────────────────

/// Borrowing Huffman quad wavelet tree over an `HQWB` blob.
///
/// Level RSQ payloads are borrowed; Huffman code tables are owned (small).
#[derive(Clone, Debug)]
pub struct HqwtView<'a, T, const B_SIZE: usize = 256> {
    n: usize,
    n_levels: usize,
    codes_encode: Vec<PrefixCode>,
    codes_decode: Vec<Vec<(u32, T)>>,
    levels: Vec<RSQVectorView<'a, B_SIZE>>,
    /// Per-level sequence length (HQWT uneven levels).
    lens: Vec<usize>,
    _marker: PhantomData<&'a [u8]>,
}

impl<'a, T, const B_SIZE: usize> HqwtView<'a, T, B_SIZE>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    u8: AsPrimitive<T>,
{
    /// Open a zero-copy view over an `HQWB` blob.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, LayoutError> {
        ensure_le()?;
        if bytes.len() < HEADER_SIZE {
            return Err(LayoutError::Truncated);
        }
        if &bytes[0..4] != HQWB_MAGIC {
            return Err(LayoutError::BadMagic);
        }
        let mut o = 4;
        let version = get_u16(bytes, &mut o);
        if version != FORMAT_VERSION {
            return Err(LayoutError::BadVersion);
        }
        let flags = bytes[o];
        o += 1;
        if flags & FLAG_PREFETCH != 0 {
            return Err(LayoutError::PrefetchNotSupported);
        }
        let want_b512 = flags & FLAG_B512 != 0;
        if (B_SIZE == 512) != want_b512 {
            return Err(LayoutError::Inconsistent {
                detail: "B_SIZE flag does not match requested type",
            });
        }
        let t_width = bytes[o];
        o += 1;
        if t_width as usize != size_of::<T>() {
            return Err(LayoutError::Inconsistent {
                detail: "t_width does not match T",
            });
        }
        let n = get_u64(bytes, &mut o) as usize;
        let n_levels = get_u16(bytes, &mut o) as usize;
        let encode_len = get_u16(bytes, &mut o) as usize;
        let decode_n_buckets = get_u16(bytes, &mut o) as usize;
        let _ = o;

        let dir_end = HEADER_SIZE + n_levels * HQWT_LEVEL_DIR_SIZE;
        if bytes.len() < dir_end {
            return Err(LayoutError::Truncated);
        }

        // Code tables (owned) — after directory, 8-aligned.
        let mut p = align_up(dir_end, 8);
        let encode_bytes = checked_region(p, encode_len, 8, bytes.len())?;
        let mut codes_encode = Vec::with_capacity(encode_len);
        for i in 0..encode_len {
            let base = p + i * 8;
            let content = u32::from_le_bytes(bytes[base..base + 4].try_into().unwrap());
            let len = u32::from_le_bytes(bytes[base + 4..base + 8].try_into().unwrap());
            codes_encode.push(PrefixCode { content, len });
        }
        p += encode_bytes;

        let mut codes_decode: Vec<Vec<(u32, T)>> = Vec::with_capacity(decode_n_buckets);
        for _ in 0..decode_n_buckets {
            let _ = checked_region(p, 1, 4, bytes.len())?;
            let n_entries = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            let _ = checked_region(p, n_entries, 12, bytes.len())?;
            let mut bucket = Vec::with_capacity(n_entries);
            for _ in 0..n_entries {
                let content = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
                p += 4;
                let sym_u = u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap()) as usize;
                p += 8;
                bucket.push((content, sym_u.as_()));
            }
            codes_decode.push(bucket);
        }

        let mut levels = Vec::with_capacity(n_levels);
        let mut lens = Vec::with_capacity(n_levels);
        for li in 0..n_levels {
            let dir_off = HEADER_SIZE + li * HQWT_LEVEL_DIR_SIZE;
            let dir = HqwtLevelDir::read(&bytes[dir_off..dir_off + HQWT_LEVEL_DIR_SIZE])?;
            levels.push(RSQVectorView::from_dir(bytes, &dir.plain)?);
            lens.push(dir.level_len as usize);
        }

        Ok(Self {
            n,
            n_levels,
            codes_encode,
            codes_decode,
            levels,
            lens,
            _marker: PhantomData,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    #[inline]
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }

    #[inline]
    pub fn levels(&self) -> &[RSQVectorView<'a, B_SIZE>] {
        &self.levels
    }

    #[inline]
    pub fn codes_encode(&self) -> &[PrefixCode] {
        &self.codes_encode
    }

    #[inline]
    pub fn codes_decode(&self) -> &[Vec<(u32, T)>] {
        &self.codes_decode
    }

    #[inline]
    pub fn level_lens(&self) -> &[usize] {
        &self.lens
    }
}

impl<'a, T, const B_SIZE: usize> AccessUnsigned for HqwtView<'a, T, B_SIZE>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
{
    type Item = T;

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
        let mut shift = 0usize;

        for level in 0..self.n_levels {
            if cur_i >= self.lens[level] {
                break;
            }
            let lv = &self.levels[level];
            let symbol = lv.get_unchecked(cur_i);
            result = (result << 2) | symbol as u32;
            let offset = lv.occs_smaller_unchecked(symbol);
            cur_i = lv.rank_unchecked(symbol, cur_i) + offset;
            shift += 2;
        }

        let idx = self.codes_decode[shift]
            .binary_search_by_key(&result, |(x, _)| *x)
            .expect("could not translate symbol");
        T::from(self.codes_decode[shift][idx].1).unwrap()
    }
}

impl<'a, T, const B_SIZE: usize> RankUnsigned for HqwtView<'a, T, B_SIZE>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
{
    #[inline(always)]
    fn rank(&self, symbol: Self::Item, i: usize) -> Option<usize> {
        if i > self.n
            || symbol.as_() >= self.codes_encode.len()
            || self.codes_encode[symbol.as_()].len == 0
        {
            return None;
        }
        Some(unsafe { self.rank_unchecked(symbol, i) })
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, symbol: Self::Item, i: usize) -> usize {
        let mut cur_i = i;
        let mut cur_p = 0usize;
        let code = &self.codes_encode[symbol.as_()];
        let mut shift: i64 = code.len as i64 - 2;
        let repr = code.content;
        let mut level = 0usize;

        while shift >= 0 {
            let two_bits = ((repr >> shift as usize) & 3) as u8;
            let lv = &self.levels[level];
            let offset = lv.occs_smaller_unchecked(two_bits);
            cur_p = lv.rank_unchecked(two_bits, cur_p) + offset;
            cur_i = lv.rank_unchecked(two_bits, cur_i) + offset;
            level += 1;
            shift -= 2;
        }
        cur_i - cur_p
    }
}

impl<'a, T, const B_SIZE: usize> SelectUnsigned for HqwtView<'a, T, B_SIZE>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
{
    #[inline(always)]
    fn select(&self, symbol: Self::Item, i: usize) -> Option<usize> {
        if symbol.as_() >= self.codes_encode.len() || self.codes_encode[symbol.as_()].len == 0 {
            return None;
        }

        let mut path_off = Vec::with_capacity(self.n_levels);
        let mut rank_path_off = Vec::with_capacity(self.n_levels);

        let code = &self.codes_encode[symbol.as_()];
        let mut shift: i64 = code.len as i64 - 2;
        let repr = code.content;
        let mut b = 0usize;
        let mut level = 0usize;

        while shift >= 0 {
            path_off.push(b);
            let two_bits = ((repr >> shift as usize) & 3) as u8;
            let lv = &self.levels[level];
            if b > lv.len() {
                return None;
            }
            let rank_b = unsafe { lv.rank_unchecked(two_bits, b) };
            b = rank_b + unsafe { lv.occs_smaller_unchecked(two_bits) };
            rank_path_off.push(rank_b);
            level += 1;
            shift -= 2;
        }

        shift = 0;
        let mut result = i;
        for lvl in (0..level).rev() {
            b = path_off[lvl];
            let rank_b = rank_path_off[lvl];
            let two_bits = ((repr >> shift as usize) & 3) as u8;
            let lv = &self.levels[lvl];
            result = lv.select(two_bits, rank_b + result)? - b;
            shift += 2;
        }
        Some(result)
    }

    #[inline(always)]
    unsafe fn select_unchecked(&self, symbol: Self::Item, i: usize) -> usize {
        self.select(symbol, i).unwrap()
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn get_u16(buf: &[u8], o: &mut usize) -> u16 {
    let v = u16::from_le_bytes(buf[*o..*o + 2].try_into().unwrap());
    *o += 2;
    v
}

#[inline]
fn get_u64(buf: &[u8], o: &mut usize) -> u64 {
    let v = u64::from_le_bytes(buf[*o..*o + 8].try_into().unwrap());
    *o += 8;
    v
}

/// Owned byte buffer whose usable slice is 64-byte aligned.
///
/// Zero-copy views require absolute alignment of POD arrays. A plain
/// `Vec<u8>` from `to_bytes` is typically only 8-aligned; wrap it here
/// before opening a view (mmap'd pages are naturally page-aligned).
pub struct AlignedBytes {
    buf: Vec<u8>,
    start: usize,
    len: usize,
}

impl AlignedBytes {
    /// Copy `src` into an over-allocated buffer and expose a 64-aligned window.
    pub fn from_slice(src: &[u8]) -> Self {
        if src.is_empty() {
            return Self {
                buf: Vec::new(),
                start: 0,
                len: 0,
            };
        }
        // Over-allocate so a 64-aligned subslice of length `src.len()` fits.
        let mut buf = vec![0u8; src.len() + 63];
        let base = buf.as_ptr() as usize;
        let start = align_up(base, 64) - base;
        debug_assert!(start + src.len() <= buf.len());
        buf[start..start + src.len()].copy_from_slice(src);
        // Confirm alignment still holds (Vec must not have reallocated).
        debug_assert_eq!((buf.as_ptr() as usize + start) % 64, 0);
        Self {
            buf,
            start,
            len: src.len(),
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.buf[self.start..self.start + self.len]
    }
}

impl std::ops::Deref for AlignedBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytes::{hqwt256_to_bytes, qwt256_to_bytes};
    use crate::{HQWT256, QWT256};

    #[test]
    fn qwt_view_matches_owned() {
        let data: Vec<u32> = (0..1000).map(|x| (x * 3) % 128).collect();
        let original = QWT256::from(data.clone());
        let bytes = qwt256_to_bytes(&original).unwrap();
        let aligned = AlignedBytes::from_slice(&bytes);
        let view: QwtView<'_, u32, 256> = QwtView::from_bytes(aligned.as_slice()).unwrap();

        assert_eq!(view.len(), original.len());
        assert_eq!(view.n_levels(), original.n_levels());
        assert_eq!(view.sigma_raw(), original.sigma_raw());

        for i in 0..data.len() {
            assert_eq!(view.get(i), original.get(i), "get@{i}");
        }
        for &sym in &[0u32, 1, 5, 64, 127] {
            for i in (0..=data.len()).step_by(31) {
                assert_eq!(view.rank(sym, i), original.rank(sym, i), "rank({sym},{i})");
            }
            if let Some(cnt) = original.rank(sym, data.len()) {
                for k in 0..cnt.min(3) {
                    assert_eq!(
                        view.select(sym, k),
                        original.select(sym, k),
                        "select({sym},{k})"
                    );
                }
            }
        }
    }

    #[test]
    fn qwt_view_empty() {
        let mut empty: Vec<u32> = vec![];
        let original = QWT256::new(&mut empty);
        let bytes = qwt256_to_bytes(&original).unwrap();
        let aligned = AlignedBytes::from_slice(&bytes);
        let view: QwtView<'_, u32, 256> = QwtView::from_bytes(aligned.as_slice()).unwrap();
        assert_eq!(view.len(), 0);
        assert!(view.is_empty());
        assert_eq!(view.n_levels(), 0);
        assert_eq!(view.get(0), None);
    }

    #[test]
    fn qwt_view_rejects_unaligned_base() {
        // Build a blob, then force the slice start to be 1-mod-64 so cast fails
        // when there is at least one level with POD payload.
        let original = QWT256::from(vec![1u32, 2, 3, 4, 5, 6, 7, 8]);
        let bytes = qwt256_to_bytes(&original).unwrap();
        // Pad so we can take an unaligned window that still has the full blob.
        let mut padded = vec![0u8; 1];
        padded.extend_from_slice(&bytes);
        // padded[1..] has base = padded.as_ptr()+1. If that is not 64-aligned
        // (almost always), from_bytes should fail with Misaligned when casting.
        let slice = &padded[1..];
        let addr = slice.as_ptr() as usize;
        if !addr.is_multiple_of(64) && original.n_levels() > 0 {
            let err = QwtView::<u32, 256>::from_bytes(slice).unwrap_err();
            assert_eq!(err, LayoutError::Misaligned);
        }
    }

    #[test]
    fn qwt_view_rejects_non_monotone_child_offsets() {
        let original = QWT256::from((0..1000).map(|x| (x * 3) % 64).collect::<Vec<u32>>());
        let mut bytes = qwt256_to_bytes(&original).unwrap();
        // First LevelDir.n_occs_smaller[0] must be zero.
        bytes[HEADER_SIZE + 88..HEADER_SIZE + 96].copy_from_slice(&1u64.to_le_bytes());
        let aligned = AlignedBytes::from_slice(&bytes);
        let error = QwtView::<u32, 256>::from_bytes(aligned.as_slice()).unwrap_err();
        assert!(matches!(error, LayoutError::Inconsistent { .. }));
    }

    #[test]
    fn qwt_view_rejects_missing_select_sentinels() {
        let original = QWT256::from((0..1000).map(|x| (x * 3) % 64).collect::<Vec<u32>>());
        let mut bytes = qwt256_to_bytes(&original).unwrap();
        // First LevelDir.n_sel[0] must include the initial sample and sentinel.
        bytes[HEADER_SIZE + 72..HEADER_SIZE + 76].copy_from_slice(&1u32.to_le_bytes());
        let aligned = AlignedBytes::from_slice(&bytes);
        let error = QwtView::<u32, 256>::from_bytes(aligned.as_slice()).unwrap_err();
        assert!(matches!(error, LayoutError::Inconsistent { .. }));
    }

    #[test]
    fn qwt_view_rejects_out_of_bounds_rank_metadata() {
        let original = QWT256::from((0..1000).map(|x| (x * 3) % 64).collect::<Vec<u32>>());
        let mut bytes = qwt256_to_bytes(&original).unwrap();
        let sb_field = HEADER_SIZE + 24;
        let sb_offset =
            u64::from_le_bytes(bytes[sb_field..sb_field + 8].try_into().unwrap()) as usize;
        bytes[sb_offset..sb_offset + 16].fill(0xff);
        let aligned = AlignedBytes::from_slice(&bytes);
        let error = QwtView::<u32, 256>::from_bytes(aligned.as_slice()).unwrap_err();
        assert!(matches!(error, LayoutError::Inconsistent { .. }));
    }

    /// Multi-superblock image where a rare symbol first appears late.
    #[test]
    fn qwt_view_accepts_late_debuting_symbol() {
        let mut data = vec![1u32; 100_000];
        data.push(7);
        let original = QWT256::from(data.clone());
        let bytes = qwt256_to_bytes(&original).unwrap();
        let aligned = AlignedBytes::from_slice(&bytes);
        let view: QwtView<'_, u32, 256> = QwtView::from_bytes(aligned.as_slice()).unwrap();

        assert_eq!(view.len(), original.len());
        for i in (0..data.len()).step_by(997) {
            assert_eq!(view.get(i), original.get(i), "get@{i}");
        }
        for &sym in &[1u32, 7] {
            for i in (0..=data.len()).step_by(4099) {
                assert_eq!(view.rank(sym, i), original.rank(sym, i), "rank({sym},{i})");
            }
            if let Some(cnt) = original.rank(sym, data.len()) {
                for k in 0..cnt.min(5) {
                    assert_eq!(
                        view.select(sym, k),
                        original.select(sym, k),
                        "select({sym},{k})"
                    );
                }
            }
        }
    }

    /// Select sample pointing at the wrong superblock is rejected.
    #[test]
    fn qwt_view_rejects_corrupt_select_sample() {
        let data: Vec<u32> = (0..50_000).map(|x| (x % 4) as u32).collect();
        let original = QWT256::from(data);
        let mut bytes = qwt256_to_bytes(&original).unwrap();
        let n_sel0 =
            u32::from_le_bytes(bytes[HEADER_SIZE + 72..HEADER_SIZE + 76].try_into().unwrap())
                as usize;
        assert!(n_sel0 >= 2);
        let off_sel0 =
            u64::from_le_bytes(bytes[HEADER_SIZE + 40..HEADER_SIZE + 48].try_into().unwrap())
                as usize;
        let n_sb =
            u32::from_le_bytes(bytes[HEADER_SIZE + 32..HEADER_SIZE + 36].try_into().unwrap());
        assert!(n_sb > 1);
        let last = n_sb - 1;
        bytes[off_sel0..off_sel0 + 4].copy_from_slice(&last.to_le_bytes());
        let aligned = AlignedBytes::from_slice(&bytes);
        let error = QwtView::<u32, 256>::from_bytes(aligned.as_slice()).unwrap_err();
        assert!(matches!(error, LayoutError::Inconsistent { .. }));
    }


    #[test]
    fn hqwt_view_matches_owned() {

        let data: Vec<u32> = (0..500).map(|x| (x * 7) % 64).collect();
        let original = HQWT256::from(data.clone());
        let bytes = hqwt256_to_bytes(&original).unwrap();
        let aligned = AlignedBytes::from_slice(&bytes);
        let view: HqwtView<'_, u32, 256> = HqwtView::from_bytes(aligned.as_slice()).unwrap();

        assert_eq!(view.len(), original.len());
        assert_eq!(view.n_levels(), original.n_levels());

        for i in 0..data.len() {
            assert_eq!(view.get(i), original.get(i), "get@{i}");
        }
        for &sym in &[0u32, 1, 7, 31, 63] {
            for i in (0..=data.len()).step_by(29) {
                assert_eq!(view.rank(sym, i), original.rank(sym, i), "rank({sym},{i})");
            }
            if let Some(cnt) = original.rank(sym, data.len()) {
                for k in 0..cnt.min(3) {
                    assert_eq!(
                        view.select(sym, k),
                        original.select(sym, k),
                        "select({sym},{k})"
                    );
                }
            }
        }
    }

    #[test]
    fn hqwt_view_empty() {
        let mut empty: Vec<u32> = vec![];
        let original = HQWT256::new(&mut empty);
        let bytes = hqwt256_to_bytes(&original).unwrap();
        let aligned = AlignedBytes::from_slice(&bytes);
        let view: HqwtView<'_, u32, 256> = HqwtView::from_bytes(aligned.as_slice()).unwrap();
        assert_eq!(view.len(), 0);
        assert!(view.is_empty());
    }

    #[test]
    fn hqwt_view_bad_magic() {
        let bytes = hqwt256_to_bytes(&HQWT256::from(vec![1u32, 2, 3])).unwrap();
        // Corrupt magic in a fresh aligned copy.
        let mut raw = bytes.clone();
        raw[0] = b'X';
        let aligned = AlignedBytes::from_slice(&raw);
        let err = HqwtView::<u32, 256>::from_bytes(aligned.as_slice()).unwrap_err();
        assert_eq!(err, LayoutError::BadMagic);
    }

    // ── extract / get_and_rank differential (view ↔ heap) ────────────────

    #[test]
    fn qwt_view_extract_matches_owned() {
        let data: Vec<u32> = (0..500).map(|x| (x * 7) % 64).collect();
        let original = QWT256::from(data.clone());
        let bytes = qwt256_to_bytes(&original).unwrap();
        let aligned = AlignedBytes::from_slice(&bytes);
        let view: QwtView<'_, u32, 256> = QwtView::from_bytes(aligned.as_slice()).unwrap();
        let n = data.len();

        // full range
        assert_eq!(view.extract_range(0..n), original.extract_range(0..n));
        assert_eq!(
            view.extract_range_distinct(0..n),
            original.extract_range_distinct(0..n)
        );

        // empty / oob
        assert!(view.extract_range(0..0).is_empty());
        assert!(view.extract_range(n..n).is_empty());
        assert!(view.extract_range(0..n + 1).is_empty());
        let reversed = std::ops::Range { start: 3, end: 2 };
        assert!(view.extract_range(reversed).is_empty());

        // subrange sweep
        for start in (0..=n).step_by(37) {
            for end in (start..=n).step_by(41) {
                let got = view.extract_range(start..end);
                let exp = original.extract_range(start..end);
                assert_eq!(got, exp, "multiset mismatch for {start}..{end}");

                let got_d = view.extract_range_distinct(start..end);
                let exp_d = original.extract_range_distinct(start..end);
                assert_eq!(got_d, exp_d, "distinct mismatch for {start}..{end}");

                // into API
                let mut buf = vec![99u32; 3];
                view.extract_range_distinct_into(start..end, &mut buf);
                assert_eq!(buf, exp_d);

                // also equals sort / sort+dedup of per-row get
                let mut per_row: Vec<_> = (start..end).map(|i| view.get(i).unwrap()).collect();
                per_row.sort_unstable();
                assert_eq!(got, per_row);
                let mut deduped = per_row.clone();
                deduped.dedup();
                assert_eq!(got_d, deduped);
            }
        }

        // singleton
        if n > 0 {
            assert_eq!(view.extract_range(0..1), original.extract_range(0..1));
            assert_eq!(
                view.extract_range_distinct(0..1),
                original.extract_range_distinct(0..1)
            );
        }
    }

    #[test]
    fn qwt_view_get_and_rank_matches_owned() {
        let data: Vec<u32> = (0..300).map(|x| (x * 11) % 128).collect();
        let original = QWT256::from(data.clone());
        let bytes = qwt256_to_bytes(&original).unwrap();
        let aligned = AlignedBytes::from_slice(&bytes);
        let view: QwtView<'_, u32, 256> = QwtView::from_bytes(aligned.as_slice()).unwrap();

        assert_eq!(view.get_and_rank(data.len()), None);
        assert_eq!(original.get_and_rank(data.len()), None);

        for i in 0..data.len() {
            let v = view.get_and_rank(i).unwrap();
            let o = original.get_and_rank(i).unwrap();
            assert_eq!(v, o, "get_and_rank@{i}");
            // also matches get + rank(get, i+1)
            let sym = view.get(i).unwrap();
            assert_eq!(v.0, sym);
            assert_eq!(v.1, view.rank(sym, i + 1).unwrap());
        }
    }

    #[test]
    fn qwt_view_extract_empty_tree() {
        let mut empty: Vec<u32> = vec![];
        let original = QWT256::new(&mut empty);
        let bytes = qwt256_to_bytes(&original).unwrap();
        let aligned = AlignedBytes::from_slice(&bytes);
        let view: QwtView<'_, u32, 256> = QwtView::from_bytes(aligned.as_slice()).unwrap();
        assert!(view.extract_range(0..0).is_empty());
        assert!(view.extract_range_distinct(0..0).is_empty());
        assert_eq!(view.get_and_rank(0), None);
    }

    #[test]
    fn qwt_view_extract_large_alphabet() {
        let data: Vec<u32> = (0..800).map(|x| (x * 13) % 4000).collect();
        let original = QWT256::from(data.clone());
        let bytes = qwt256_to_bytes(&original).unwrap();
        let aligned = AlignedBytes::from_slice(&bytes);
        let view: QwtView<'_, u32, 256> = QwtView::from_bytes(aligned.as_slice()).unwrap();
        let n = data.len();

        for &(start, end) in &[(0, n), (0, 1), (n / 3, 2 * n / 3), (n - 1, n), (10, 50)] {
            assert_eq!(
                view.extract_range(start..end),
                original.extract_range(start..end),
                "multiset {start}..{end}"
            );
            assert_eq!(
                view.extract_range_distinct(start..end),
                original.extract_range_distinct(start..end),
                "distinct {start}..{end}"
            );
        }

        for i in (0..n).step_by(23) {
            assert_eq!(
                view.get_and_rank(i),
                original.get_and_rank(i),
                "get_and_rank@{i}"
            );
        }
    }

}
