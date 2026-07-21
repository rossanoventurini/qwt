//! Borrowed per-level rank/select views over QWTB/HQWB POD payloads.
//!
//! These views are the zero-copy counterpart of [`crate::RSQVector`]: they
//! borrow `DataLine` / `SuperblockPlain` / select-sample slices that live in a
//! caller-owned byte buffer (typically an mmap) and implement the same
//! level-local get / rank / select algorithms.

use super::{cast_slice, LayoutError, LEVEL_DIR_SIZE};
use crate::qvector::rs_qvector::SuperblockPlain;
use crate::qvector::DataLine;
use crate::utils::select_in_word_u128;
use crate::{AccessQuad, RankQuad};
use std::mem::size_of;

/// Select sampling stride used by `RSSupportPlain` (every 2^13 occurrences).
const SELECT_NUM_SAMPLES: usize = 1 << 13;
/// Blocks per superblock (matches `RSSupportPlain`).
const BLOCKS_IN_SUPERBLOCK: usize = 8;

/// One plain-QWT level directory entry (128 bytes, packed LE).
///
/// Shared by the owned `from_bytes` path and zero-copy views.
#[derive(Clone, Debug)]
pub(crate) struct LevelDir {
    pub off_data: u64,
    pub n_datalines: u32,
    pub position_bits: u64,
    pub off_superblocks: u64,
    pub n_superblocks: u32,
    pub off_sel: [u64; 4],
    pub n_sel: [u32; 4],
    pub n_occs_smaller: [u64; 5],
}

impl LevelDir {
    pub(crate) fn read(buf: &[u8]) -> Result<Self, LayoutError> {
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
        for i in 0..4 {
            off_sel[i] = get_u64(buf, &mut o);
        }
        let mut n_sel = [0u32; 4];
        for i in 0..4 {
            n_sel[i] = get_u32(buf, &mut o);
        }
        let mut n_occs_smaller = [0u64; 5];
        for i in 0..5 {
            n_occs_smaller[i] = get_u64(buf, &mut o);
        }
        debug_assert_eq!(o, LEVEL_DIR_SIZE);
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

/// HQWT level directory = plain dir + `level_len` u64 (136 B).
#[derive(Clone, Debug)]
pub(crate) struct HqwtLevelDir {
    pub plain: LevelDir,
    pub level_len: u64,
}

impl HqwtLevelDir {
    pub(crate) fn read(buf: &[u8]) -> Result<Self, LayoutError> {
        if buf.len() < LEVEL_DIR_SIZE + 8 {
            return Err(LayoutError::Truncated);
        }
        let plain = LevelDir::read(&buf[..LEVEL_DIR_SIZE])?;
        let mut o = LEVEL_DIR_SIZE;
        let level_len = get_u64(buf, &mut o);
        Ok(Self { plain, level_len })
    }
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

/// Borrowing view of one RS-augmented quad-vector level.
///
/// `B_SIZE` is the block size (256 or 512), matching `RSSupportPlain<B_SIZE>`.
#[derive(Clone, Copy, Debug)]
pub struct RSQVectorView<'a, const B_SIZE: usize = 256> {
    data: &'a [DataLine],
    superblocks: &'a [SuperblockPlain],
    select_samples: [&'a [u32]; 4],
    /// Bit cursor (`2 * n_symbols`).
    position_bits: usize,
    /// Wavelet-matrix child offsets (5 counters).
    n_occs_smaller: [usize; 5],
}

impl<'a, const B_SIZE: usize> RSQVectorView<'a, B_SIZE> {
    /// Open a level from a full container blob and a parsed directory entry.
    ///
    /// Requires the **absolute** addresses of data/superblock slices to be
    /// 64-byte aligned (format offsets are 64-aligned; the base buffer must
    /// also be 64-aligned for zero-copy casting).
    pub(crate) fn from_dir(bytes: &'a [u8], dir: &LevelDir) -> Result<Self, LayoutError> {
        debug_assert!(B_SIZE == 256 || B_SIZE == 512);

        if dir.off_data as usize % 64 != 0 || dir.off_superblocks as usize % 64 != 0 {
            return Err(LayoutError::Misaligned);
        }

        let data_off = dir.off_data as usize;
        let n_data = dir.n_datalines as usize;
        let data_bytes = n_data
            .checked_mul(size_of::<DataLine>())
            .ok_or(LayoutError::Truncated)?;
        if data_off.checked_add(data_bytes).ok_or(LayoutError::Truncated)? > bytes.len() {
            return Err(LayoutError::Truncated);
        }
        let data = cast_slice::<DataLine>(&bytes[data_off..data_off + data_bytes])?;

        let sb_off = dir.off_superblocks as usize;
        let n_sb = dir.n_superblocks as usize;
        let sb_bytes = n_sb
            .checked_mul(size_of::<SuperblockPlain>())
            .ok_or(LayoutError::Truncated)?;
        if sb_off.checked_add(sb_bytes).ok_or(LayoutError::Truncated)? > bytes.len() {
            return Err(LayoutError::Truncated);
        }
        let superblocks = cast_slice::<SuperblockPlain>(&bytes[sb_off..sb_off + sb_bytes])?;

        let mut select_samples: [&'a [u32]; 4] = [&[]; 4];
        for s in 0..4 {
            let n_sel = dir.n_sel[s] as usize;
            let off = dir.off_sel[s] as usize;
            let need = n_sel
                .checked_mul(size_of::<u32>())
                .ok_or(LayoutError::Truncated)?;
            if off.checked_add(need).ok_or(LayoutError::Truncated)? > bytes.len() {
                return Err(LayoutError::Truncated);
            }
            // u32 LE samples: 4-byte aligned offsets in a 64-aligned base → cast ok.
            select_samples[s] = cast_slice::<u32>(&bytes[off..off + need])?;
        }

        let n_occs_smaller: [usize; 5] = std::array::from_fn(|i| dir.n_occs_smaller[i] as usize);
        let position_bits = dir.position_bits as usize;
        if position_bits % 2 != 0 {
            return Err(LayoutError::Inconsistent {
                detail: "position_bits must be even",
            });
        }
        if position_bits > n_data * 512 {
            return Err(LayoutError::Inconsistent {
                detail: "position_bits exceeds data capacity",
            });
        }

        Ok(Self {
            data,
            superblocks,
            select_samples,
            position_bits,
            n_occs_smaller,
        })
    }

    /// Number of 2-bit symbols on this level.
    #[inline]
    pub fn len(&self) -> usize {
        self.position_bits >> 1
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.position_bits == 0
    }

    /// Bit cursor (`2 * len()`).
    #[inline]
    pub fn position_bits(&self) -> usize {
        self.position_bits
    }

    /// Borrowed data lines (POD).
    #[inline]
    pub fn data_lines(&self) -> &'a [DataLine] {
        self.data
    }

    /// Borrowed superblock counters (POD).
    #[inline]
    pub fn superblocks(&self) -> &'a [SuperblockPlain] {
        self.superblocks
    }

    /// Borrowed select samples for symbol `s ∈ 0..4`.
    #[inline]
    pub fn select_samples(&self, s: usize) -> &'a [u32] {
        self.select_samples[s]
    }

    /// Wavelet-matrix child offsets (`n_occs_smaller`).
    #[inline]
    pub fn n_occs_smaller(&self) -> [usize; 5] {
        self.n_occs_smaller
    }


    /// Occurrences of all symbols strictly smaller than `symbol` (0..3).
    #[inline(always)]
    pub unsafe fn occs_smaller_unchecked(&self, symbol: u8) -> usize {
        debug_assert!(symbol <= 3);
        self.n_occs_smaller[symbol as usize]
    }

    /// Total occurrences of `symbol` (0..3) on this level.
    #[inline(always)]
    pub unsafe fn occs_unchecked(&self, symbol: u8) -> usize {
        debug_assert!(symbol <= 3);
        self.n_occs_smaller[(symbol + 1) as usize] - self.n_occs_smaller[symbol as usize]
    }

    /// 2-bit symbol at position `i`.
    ///
    /// # Safety
    /// `i` must be `< self.len()`.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, i: usize) -> u8 {
        debug_assert!(i < self.len());
        let line = i >> 8;
        let pos = i & 255;
        self.data.get_unchecked(line).get_unchecked(pos)
    }

    /// Rank of 2-bit `symbol` up to `i` (excluded).
    ///
    /// # Safety
    /// `symbol ≤ 3` and `i ≤ self.len()`.
    #[inline(always)]
    pub unsafe fn rank_unchecked(&self, symbol: u8, i: usize) -> usize {
        debug_assert!(symbol <= 3);
        self.rank_block(symbol, i) + self.rank_intra_block(symbol, i)
    }

    /// Ranks of all four symbols up to `i` (excluded).
    ///
    /// # Safety
    /// `i ≤ self.len()`.
    #[inline(always)]
    pub unsafe fn rank_all_unchecked(&self, i: usize) -> [usize; 4] {
        let block = i / B_SIZE;
        let sb_idx = block / BLOCKS_IN_SUPERBLOCK;
        let block_in_sb = block % BLOCKS_IN_SUPERBLOCK;
        let block_ranks = if sb_idx < self.superblocks.len() {
            self.superblocks
                .get_unchecked(sb_idx)
                .get_rank_all(block_in_sb)
        } else {
            [0; 4]
        };
        let intra = self.rank_intra_block_all(i);
        [
            block_ranks[0] + intra[0],
            block_ranks[1] + intra[1],
            block_ranks[2] + intra[2],
            block_ranks[3] + intra[3],
        ]
    }

    /// Select: position of the `i`-th occurrence of `symbol` (0-based).
    pub fn select(&self, symbol: u8, i: usize) -> Option<usize> {
        if symbol > 3 {
            return None;
        }
        let total = unsafe { self.occs_unchecked(symbol) };
        if total <= i {
            return None;
        }
        let (mut pos, rank) = self.select_block(symbol, i + 1);
        pos += self.select_intra_block(symbol, i - rank + 1, pos);
        Some(pos)
    }

    #[inline(always)]
    fn rank_block(&self, symbol: u8, i: usize) -> usize {
        let superblock_index = i / (B_SIZE * BLOCKS_IN_SUPERBLOCK);
        let block_index = (i / B_SIZE) % BLOCKS_IN_SUPERBLOCK;
        if superblock_index >= self.superblocks.len() {
            return 0;
        }
        unsafe {
            self.superblocks
                .get_unchecked(superblock_index)
                .get_rank(symbol, block_index)
        }
    }

    #[inline(always)]
    fn rank_intra_block(&self, symbol: u8, i: usize) -> usize {
        if B_SIZE == 256 {
            let data_line_id = i >> 8;
            let offset = i & 255;
            return if let Some(d) = self.data.get(data_line_id) {
                unsafe { d.rank_unchecked(symbol, offset) }
            } else {
                0
            };
        }
        // B_SIZE == 512
        let block_id = i >> 9;
        let offset_in_block = i & 511;
        let offset_in_first = if offset_in_block <= 256 {
            offset_in_block
        } else {
            256
        };
        let mut rank = if let Some(d) = self.data.get(block_id * 2) {
            unsafe { d.rank_unchecked(symbol, offset_in_first) }
        } else {
            0
        };
        if offset_in_block > 256 {
            rank += if let Some(d) = self.data.get(block_id * 2 + 1) {
                unsafe { d.rank_unchecked(symbol, offset_in_block - 256) }
            } else {
                0
            };
        }
        rank
    }

    #[inline(always)]
    fn rank_intra_block_all(&self, i: usize) -> [usize; 4] {
        if B_SIZE == 256 {
            let data_line_id = i >> 8;
            let offset = i & 255;
            return if let Some(d) = self.data.get(data_line_id) {
                unsafe { d.rank_all_unchecked(offset) }
            } else {
                [0; 4]
            };
        }
        let block_id = i >> 9;
        let offset_in_block = i & 511;
        let offset_in_first = if offset_in_block <= 256 {
            offset_in_block
        } else {
            256
        };
        let mut ranks = if let Some(d) = self.data.get(block_id * 2) {
            unsafe { d.rank_all_unchecked(offset_in_first) }
        } else {
            [0; 4]
        };
        if offset_in_block > 256 {
            let second = if let Some(d) = self.data.get(block_id * 2 + 1) {
                unsafe { d.rank_all_unchecked(offset_in_block - 256) }
            } else {
                [0; 4]
            };
            for s in 0..4 {
                ranks[s] += second[s];
            }
        }
        ranks
    }

    /// `(block_start_pos, rank_at_block_start)` for the 1-based `i`-th occurrence.
    #[inline]
    fn select_block(&self, symbol: u8, i: usize) -> (usize, usize) {
        let samples = self.select_samples[symbol as usize];
        debug_assert!(!samples.is_empty(), "select samples always have sentinel");
        let sampled_i = (i - 1) / SELECT_NUM_SAMPLES;
        let sampled_i = sampled_i.min(samples.len().saturating_sub(2));
        let mut first_sblock_id = samples[sampled_i] as usize;
        let last_sblock_id = 1 + samples[sampled_i + 1] as usize;

        let step = ((last_sblock_id - first_sblock_id) as f64).sqrt() as usize + 1;

        while first_sblock_id < last_sblock_id {
            if self.superblocks[first_sblock_id].get_superblock_counter(symbol) >= i {
                break;
            }
            first_sblock_id += step;
        }
        first_sblock_id = first_sblock_id.saturating_sub(step);

        while first_sblock_id < last_sblock_id {
            if self.superblocks[first_sblock_id].get_superblock_counter(symbol) >= i {
                break;
            }
            first_sblock_id += 1;
        }
        first_sblock_id = first_sblock_id.saturating_sub(1);

        let mut position = first_sblock_id * B_SIZE * BLOCKS_IN_SUPERBLOCK;
        let mut rank = self.superblocks[first_sblock_id].get_superblock_counter(symbol);

        let (block_id, block_rank) =
            self.superblocks[first_sblock_id].block_predecessor(symbol, i - rank);
        position += block_id * B_SIZE;
        rank += block_rank;
        (position, rank)
    }

    /// Offset within the block at `pos` of the 1-based residual occurrence `i`.
    #[inline]
    fn select_intra_block(&self, symbol: u8, i: usize, pos: usize) -> usize {
        let line_id = pos >> 8;
        let mut rem = i - 1;

        // B_SIZE 256 → one DataLine; 512 may need two.
        let n_lines = if B_SIZE == 256 { 1 } else { 2 };
        let mut result = 0usize;
        for j in 0..n_lines {
            let (word_0, word_1) = unsafe {
                self.data
                    .get_unchecked(line_id + j)
                    .normalize(symbol)
            };
            let cnt_0 = word_0.count_ones() as usize;
            if cnt_0 > rem {
                return result + select_in_word_u128(word_0, rem as u64) as usize;
            }
            rem -= cnt_0;
            result += 128;
            let cnt_1 = word_1.count_ones() as usize;
            if cnt_1 > rem {
                return result + select_in_word_u128(word_1, rem as u64) as usize;
            }
            rem -= cnt_1;
            result += 128;
        }
        0
    }
}

impl<const B_SIZE: usize> AccessQuad for RSQVectorView<'_, B_SIZE> {
    #[inline]
    fn get(&self, i: usize) -> Option<u8> {
        if i >= self.len() {
            return None;
        }
        Some(unsafe { self.get_unchecked(i) })
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> u8 {
        RSQVectorView::get_unchecked(self, i)
    }
}

impl<const B_SIZE: usize> RankQuad for RSQVectorView<'_, B_SIZE> {
    #[inline]
    fn rank(&self, symbol: u8, i: usize) -> Option<usize> {
        if symbol > 3 || i > self.len() {
            return None;
        }
        Some(unsafe { self.rank_unchecked(symbol, i) })
    }

    #[inline]
    unsafe fn rank_unchecked(&self, symbol: u8, i: usize) -> usize {
        RSQVectorView::rank_unchecked(self, symbol, i)
    }

    #[inline]
    unsafe fn rank_all_unchecked(&self, i: usize) -> [usize; 4] {
        RSQVectorView::rank_all_unchecked(self, i)
    }
}
