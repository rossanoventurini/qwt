//! `QWTB` container: owned serialize / deserialize for plain [`QWaveletTree`].
//!
//! Layout (little-endian):
//!
//! ```text
//! [0..4)   magic b"QWTB"
//! [4..6)   version u16 = 1
//! [6]      flags u8 (bit0 = B_SIZE=512, bit1 = prefetch — rejected)
//! [7]      t_width u8 = size_of::<T>()
//! [8..16)  n u64
//! [16..24) sigma_raw u64 (zero-extended)
//! [24..26) n_levels u16
//! [26..32) reserved 0
//! [32..)   level directory: n_levels × 128 B
//!          then 64-aligned POD payloads per level
//! ```

use super::{
    align_up, checked_region, copy_pod_slice, ensure_le, LayoutError, FLAG_B512, FLAG_PREFETCH,
    FORMAT_VERSION, HEADER_SIZE, LEVEL_DIR_SIZE, QWTB_MAGIC,
};

use crate::quadwt::{QWaveletTree, RSforWT, WTIndexable};
use crate::qvector::rs_qvector::{RSQVector, RSSupportPlain, SuperblockPlain};
use crate::qvector::{DataLine, QVector};
use num_traits::AsPrimitive;
use std::mem::size_of;

/// One level directory entry (128 bytes, packed LE).

#[derive(Clone, Debug)]
struct LevelDir {
    off_data: u64,
    n_datalines: u32,
    position_bits: u64,
    off_superblocks: u64,
    n_superblocks: u32,
    off_sel: [u64; 4],
    n_sel: [u32; 4],
    n_occs_smaller: [u64; 5],
}

impl LevelDir {
    fn write(&self, out: &mut [u8]) {
        debug_assert_eq!(out.len(), LEVEL_DIR_SIZE);
        let mut o = 0;
        put_u64(out, &mut o, self.off_data);
        put_u32(out, &mut o, self.n_datalines);
        put_u32(out, &mut o, 0); // pad0
        put_u64(out, &mut o, self.position_bits);
        put_u64(out, &mut o, self.off_superblocks);
        put_u32(out, &mut o, self.n_superblocks);
        put_u32(out, &mut o, 0); // pad1
        for i in 0..4 {
            put_u64(out, &mut o, self.off_sel[i]);
        }
        for i in 0..4 {
            put_u32(out, &mut o, self.n_sel[i]);
        }
        for i in 0..5 {
            put_u64(out, &mut o, self.n_occs_smaller[i]);
        }
        debug_assert_eq!(o, LEVEL_DIR_SIZE);
    }

    fn read(buf: &[u8]) -> Result<Self, LayoutError> {
        if buf.len() < LEVEL_DIR_SIZE {
            return Err(LayoutError::Truncated);
        }
        let mut o = 0;
        let off_data = get_u64(buf, &mut o);
        let n_datalines = get_u32(buf, &mut o);
        let _pad0 = get_u32(buf, &mut o);
        let position_bits = get_u64(buf, &mut o);
        let off_superblocks = get_u64(buf, &mut o);
        let n_superblocks = get_u32(buf, &mut o);
        let _pad1 = get_u32(buf, &mut o);
        let mut off_sel = [0u64; 4];
        for slot in &mut off_sel {
            *slot = get_u64(buf, &mut o);
        }
        let mut n_sel = [0u32; 4];
        for slot in &mut n_sel {
            *slot = get_u32(buf, &mut o);
        }
        let mut n_occs_smaller = [0u64; 5];
        for slot in &mut n_occs_smaller {
            *slot = get_u64(buf, &mut o);
        }
        Ok(Self {
            off_data,
            n_datalines,
            position_bits,
            off_superblocks,
            n_superblocks,
            off_sel,
            n_sel,
            n_occs_smaller,
        })
    }
}

#[inline]
fn put_u16(out: &mut [u8], o: &mut usize, v: u16) {
    out[*o..*o + 2].copy_from_slice(&v.to_le_bytes());
    *o += 2;
}
#[inline]
fn put_u32(out: &mut [u8], o: &mut usize, v: u32) {
    out[*o..*o + 4].copy_from_slice(&v.to_le_bytes());
    *o += 4;
}
#[inline]
fn put_u64(out: &mut [u8], o: &mut usize, v: u64) {
    out[*o..*o + 8].copy_from_slice(&v.to_le_bytes());
    *o += 8;
}
#[inline]
fn get_u16(buf: &[u8], o: &mut usize) -> u16 {
    let v = u16::from_le_bytes(buf[*o..*o + 2].try_into().unwrap());
    *o += 2;
    v
}
#[inline]
fn get_u32(buf: &[u8], o: &mut usize) -> u32 {
    let v = u32::from_le_bytes(buf[*o..*o + 4].try_into().unwrap());
    *o += 4;
    v
}
#[inline]
fn get_u64(buf: &[u8], o: &mut usize) -> u64 {
    let v = u64::from_le_bytes(buf[*o..*o + 8].try_into().unwrap());
    *o += 8;
    v
}

/// Serialize a plain [`QWT256`](crate::QWT256) into a `QWTB` blob.
///
/// The result is a self-contained little-endian byte vector. The source buffer
/// for a later [`qwt256_from_bytes`] call need **not** be 64-byte aligned
/// (the owned path copies POD payloads onto the heap).
///
/// # Errors
/// Returns [`LayoutError::NotLittleEndian`] on big-endian hosts.
pub fn qwt256_to_bytes<T>(
    tree: &QWaveletTree<T, RSQVector<RSSupportPlain<256>>, false>,
) -> Result<Vec<u8>, LayoutError>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
{
    to_bytes_generic::<T, 256>(tree)
}

/// Serialize a plain [`QWT512`](crate::QWT512) into a `QWTB` blob.
///
/// See [`qwt256_to_bytes`] for details; the only difference is the block size
/// flag (`FLAG_B512`).
pub fn qwt512_to_bytes<T>(
    tree: &QWaveletTree<T, RSQVector<RSSupportPlain<512>>, false>,
) -> Result<Vec<u8>, LayoutError>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
{
    to_bytes_generic::<T, 512>(tree)
}

fn to_bytes_generic<T, const B: usize>(
    tree: &QWaveletTree<T, RSQVector<RSSupportPlain<B>>, false>,
) -> Result<Vec<u8>, LayoutError>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
{
    ensure_le()?;

    let n_levels = tree.n_levels();
    let dir_bytes = n_levels * LEVEL_DIR_SIZE;
    let header_and_dir = HEADER_SIZE + dir_bytes;

    // First pass: measure payload sizes and plan offsets.
    let mut dirs = Vec::with_capacity(n_levels);
    let mut cursor = align_up(header_and_dir, 64);

    // Only the first n_levels entries are real; empty trees keep a default qv
    // in the vec with n_levels == 0.
    for level in tree.levels().iter().take(n_levels) {
        let qv = level.qvector();
        let rs = level.rs_support();
        let n_datalines = qv.data_lines().len() as u32;
        let position_bits = qv.position_bits() as u64;
        let n_superblocks = rs.superblocks().len() as u32;
        let n_sel: [u32; 4] = std::array::from_fn(|s| rs.select_samples(s).len() as u32);
        let n_occs = level.n_occs_smaller();
        let n_occs_smaller: [u64; 5] = std::array::from_fn(|i| n_occs[i] as u64);

        cursor = align_up(cursor, 64);
        let off_data = cursor as u64;
        cursor += n_datalines as usize * size_of::<DataLine>();

        cursor = align_up(cursor, 64);
        let off_superblocks = cursor as u64;
        cursor += n_superblocks as usize * size_of::<SuperblockPlain>();

        // select samples: 4-byte aligned is fine
        cursor = align_up(cursor, 4);
        let mut off_sel = [0u64; 4];
        for s in 0..4 {
            off_sel[s] = cursor as u64;
            cursor += n_sel[s] as usize * size_of::<u32>();
        }

        dirs.push(LevelDir {
            off_data,
            n_datalines,
            position_bits,
            off_superblocks,
            n_superblocks,
            off_sel,
            n_sel,
            n_occs_smaller,
        });
    }

    let mut out = vec![0u8; cursor];

    // Header
    out[0..4].copy_from_slice(QWTB_MAGIC);
    let mut o = 4;
    put_u16(&mut out, &mut o, FORMAT_VERSION);
    let flags: u8 = if B == 512 { FLAG_B512 } else { 0 };
    out[o] = flags;
    o += 1;
    out[o] = size_of::<T>() as u8;
    o += 1;
    put_u64(&mut out, &mut o, tree.len() as u64);
    // sigma_raw zero-extended to u64
    let sigma_u: usize = tree.sigma_raw().as_();
    put_u64(&mut out, &mut o, sigma_u as u64);
    put_u16(&mut out, &mut o, n_levels as u16);
    // reserved already zero
    debug_assert_eq!(o, 26);
    o = HEADER_SIZE;

    // Directory
    for dir in &dirs {
        dir.write(&mut out[o..o + LEVEL_DIR_SIZE]);
        o += LEVEL_DIR_SIZE;
    }

    // Payloads
    for (level, dir) in tree.levels().iter().zip(dirs.iter()) {
        let qv = level.qvector();
        let rs = level.rs_support();

        let off = dir.off_data as usize;
        let nbytes = dir.n_datalines as usize * size_of::<DataLine>();
        // SAFETY / layout: DataLine is repr(C, align(64)); we write raw bytes.
        let src =
            unsafe { std::slice::from_raw_parts(qv.data_lines().as_ptr() as *const u8, nbytes) };
        out[off..off + nbytes].copy_from_slice(src);

        let off = dir.off_superblocks as usize;
        let nbytes = dir.n_superblocks as usize * size_of::<SuperblockPlain>();
        let src =
            unsafe { std::slice::from_raw_parts(rs.superblocks().as_ptr() as *const u8, nbytes) };
        out[off..off + nbytes].copy_from_slice(src);

        for s in 0..4 {
            let off = dir.off_sel[s] as usize;
            let n = dir.n_sel[s] as usize;
            let samples = rs.select_samples(s);
            debug_assert_eq!(samples.len(), n);
            for (i, &v) in samples.iter().enumerate() {
                out[off + i * 4..off + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
    }

    Ok(out)
}

/// Deserialize a `QWTB` blob into an owned [`QWT256`](crate::QWT256).
///
/// Copies POD payloads onto the heap, so `bytes` need not be 64-byte aligned.
/// For a zero-copy alternative see [`crate::bytes::QwtView`].
///
/// # Errors
/// See [`LayoutError`] — magic/version/flags/alignment/truncation failures.
pub fn qwt256_from_bytes<T>(
    bytes: &[u8],
) -> Result<QWaveletTree<T, RSQVector<RSSupportPlain<256>>, false>, LayoutError>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    u8: AsPrimitive<T>,
{
    from_bytes_generic::<T, 256>(bytes)
}

/// Deserialize a `QWTB` blob into an owned [`QWT512`](crate::QWT512).
///
/// See [`qwt256_from_bytes`].
pub fn qwt512_from_bytes<T>(
    bytes: &[u8],
) -> Result<QWaveletTree<T, RSQVector<RSSupportPlain<512>>, false>, LayoutError>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    u8: AsPrimitive<T>,
{
    from_bytes_generic::<T, 512>(bytes)
}

fn from_bytes_generic<T, const B: usize>(
    bytes: &[u8],
) -> Result<QWaveletTree<T, RSQVector<RSSupportPlain<B>>, false>, LayoutError>
where
    T: WTIndexable,
    usize: AsPrimitive<T>,
    u8: AsPrimitive<T>,
{
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
    if (B == 512) != want_b512 {
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

    let mut qvs = Vec::with_capacity(n_levels);
    for li in 0..n_levels {
        let dir_off = HEADER_SIZE + li * LEVEL_DIR_SIZE;
        let dir = LevelDir::read(&bytes[dir_off..dir_off + LEVEL_DIR_SIZE])?;

        // Data lines — offsets must be 64-aligned in the format; the source
        // buffer itself need not be (owned path copies into aligned heap).
        if !(dir.off_data as usize).is_multiple_of(64) {
            return Err(LayoutError::Misaligned);
        }
        let data_off = dir.off_data as usize;
        let n_datalines = dir.n_datalines as usize;
        let _ = checked_region(data_off, n_datalines, size_of::<DataLine>(), bytes.len())?;
        let lines = copy_pod_slice::<DataLine>(&bytes[data_off..], n_datalines)?;
        let qv = QVector::from_raw_parts(lines, dir.position_bits as usize);

        // Superblocks
        if !(dir.off_superblocks as usize).is_multiple_of(64) {
            return Err(LayoutError::Misaligned);
        }
        let sb_off = dir.off_superblocks as usize;
        let n_superblocks = dir.n_superblocks as usize;
        let _ = checked_region(
            sb_off,
            n_superblocks,
            size_of::<SuperblockPlain>(),
            bytes.len(),
        )?;
        let superblocks = copy_pod_slice::<SuperblockPlain>(&bytes[sb_off..], n_superblocks)?;

        // Select samples
        let mut select_samples: [Box<[u32]>; 4] = Default::default();
        for (s, sample_slot) in select_samples.iter_mut().enumerate() {
            let n_sel = dir.n_sel[s] as usize;
            let off = dir.off_sel[s] as usize;
            let _ = checked_region(off, n_sel, size_of::<u32>(), bytes.len())?;
            let mut v = Vec::with_capacity(n_sel);
            for i in 0..n_sel {
                let b = off + i * 4;
                v.push(u32::from_le_bytes(bytes[b..b + 4].try_into().unwrap()));
            }
            *sample_slot = v.into_boxed_slice();
        }

        let rs = RSSupportPlain::<B>::from_parts(superblocks, select_samples);
        let n_occs: [usize; 5] = std::array::from_fn(|i| dir.n_occs_smaller[i] as usize);
        qvs.push(RSQVector::from_parts(qv, rs, n_occs));
    }

    // Empty-tree convention in qwt::new uses one default level; from_parts
    // accepts n_levels==0 with empty qvs, or the serde-empty shape.
    if n == 0 && qvs.is_empty() {
        qvs.push(RSQVector::default());
    }

    QWaveletTree::from_parts(n, n_levels, sigma, qvs)
}

// Silence "RSforWT unused" when only used via bound on QWaveletTree methods.
const _: fn() = || {
    fn assert_rsforwt<T: RSforWT>() {}
    let _ = assert_rsforwt::<RSQVector<RSSupportPlain<256>>>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AccessUnsigned, RankUnsigned, SelectUnsigned, QWT256};

    #[test]
    fn qwtb_roundtrip_get_rank_select() {
        let data: Vec<u32> = (0..1000).map(|x| (x * 3) % 128).collect();
        let original = QWT256::from(data.clone());
        let bytes = qwt256_to_bytes(&original).unwrap();
        assert_eq!(&bytes[0..4], QWTB_MAGIC);
        let rebuilt: QWT256<u32> = qwt256_from_bytes(&bytes).unwrap();

        assert_eq!(original.len(), rebuilt.len());
        assert_eq!(original.n_levels(), rebuilt.n_levels());
        assert_eq!(original.sigma_raw(), rebuilt.sigma_raw());

        for i in 0..data.len() {
            assert_eq!(original.get(i), rebuilt.get(i), "get@{i}");
        }
        for &sym in &[0u32, 1, 5, 64, 127] {
            for i in (0..=data.len()).step_by(31) {
                assert_eq!(
                    original.rank(sym, i),
                    rebuilt.rank(sym, i),
                    "rank({sym},{i})"
                );
            }
            if let Some(cnt) = original.rank(sym, data.len()) {
                for k in 0..cnt.min(3) {
                    assert_eq!(
                        original.select(sym, k),
                        rebuilt.select(sym, k),
                        "select({sym},{k})"
                    );
                }
            }
        }
    }

    #[test]
    fn qwtb_empty_tree() {
        // Empty trees use n_levels=0 with one default qv sentinel.
        let mut empty: Vec<u32> = vec![];
        let original = QWT256::new(&mut empty);
        let bytes = qwt256_to_bytes(&original).unwrap();
        let rebuilt: QWT256<u32> = qwt256_from_bytes(&bytes).unwrap();
        assert_eq!(rebuilt.len(), 0);
        assert!(rebuilt.is_empty());
        assert_eq!(rebuilt.n_levels(), 0);
    }

    #[test]
    fn qwtb_bad_magic() {
        let mut bytes = qwt256_to_bytes(&QWT256::from(vec![1u32, 2, 3])).unwrap();
        bytes[0] = b'X';
        let err = qwt256_from_bytes::<u32>(&bytes).unwrap_err();
        assert_eq!(err, LayoutError::BadMagic);
    }

    #[test]
    fn qwtb_truncated() {
        let err = qwt256_from_bytes::<u32>(&[0u8; 8]).unwrap_err();
        assert!(matches!(
            err,
            LayoutError::BadMagic | LayoutError::Truncated
        ));
    }
}
