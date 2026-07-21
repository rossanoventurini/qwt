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
        if addr % 64 != 0 && original.n_levels() > 0 {
            let err = QwtView::<u32, 256>::from_bytes(slice).unwrap_err();
            assert_eq!(err, LayoutError::Misaligned);
        }
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
}
