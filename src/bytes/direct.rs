//! Budget-aware file-backed construction for canonical QWTB sections.
//!
//! The first production consumer is Nova Ring's C_o column. Input is a
//! replayable little-endian `u32` sequence. Construction emits one wavelet
//! level at a time and stable-partitions the original symbols through bounded
//! scratch files, so no `Vec<u32>` or heap QWT scales with the row count.

use super::qwtb::LevelDir;
use super::{align_up, ensure_le, FORMAT_VERSION, HEADER_SIZE, LEVEL_DIR_SIZE, QWTB_MAGIC};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) const DIRECT_BLOCK_ROWS: u64 = 256;
pub(super) const DIRECT_SUPERBLOCK_ROWS: u64 = 2_048;
pub(super) const DIRECT_SELECT_SAMPLE_ROWS: u64 = 8_192;
const PIECE_ROWS: u64 = 64;
/// Staged rows per input block, bounded so a block stays cache-resident even
/// when the budget would allow megabytes.
const MAX_BLOCK_ROWS: usize = 16_384;
const DIRECT_STREAMS: u64 = 6;
const MIN_STREAM_BUFFER_BYTES: u64 = 3 * 1_024;
const MIN_AUX_BUFFER_BYTES: u64 = 4 * 1_024;
const MAX_AUX_BUFFER_BYTES: u64 = 32 * 1_024;
const AUX_BUFFER_BUDGET_DIVISOR: u64 = 8;
static DIRECT_BUILD_ID: AtomicU64 = AtomicU64::new(0);

/// Limits enforced by the file-backed QWTB writer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectQwtBuildBudget {
    /// Total bytes available to the active input, the partition outputs, the
    /// packed-level output, and the rank/select auxiliary-file buffers.
    pub buffer_bytes: u64,
    /// Maximum bytes simultaneously occupied by source, partition, and
    /// per-level support files. The final QWTB output is not scratch.
    pub scratch_bytes: u64,
}

/// Measured resource use returned by a successful direct build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectQwtBuildStats {
    pub rows: u64,
    pub levels: u16,
    pub output_bytes: u64,
    pub peak_scratch_bytes: u64,
}

/// Conservative byte ceiling for the final canonical QWTB output.
///
/// The direct writer excludes its destination from `scratch_bytes`. Outer
/// container builders use this bound to reserve space while QWTB is still an
/// intermediate staged component.
pub fn qwt256_u32_output_upper_bound(rows: u64, sigma: u32) -> Result<u64, DirectQwtBuildError> {
    validate_row_capacity(rows, "QWTB")?;
    let levels = if rows == 0 {
        0
    } else {
        ((u32::BITS - sigma.leading_zeros()).max(1).div_ceil(2)) as u16
    };
    container_output_upper_bound(rows, levels, LEVEL_DIR_SIZE as u64, 0)
}

/// Conservative scratch ceiling for direct construction of a canonical
/// QWT256/u32 `QWTB` section.
///
/// Container builders use this before starting a larger atomic build so a
/// nested QWT cannot discover an insufficient scratch allowance after its
/// surrounding representation has already been staged.
pub fn qwt256_u32_scratch_upper_bound(rows: u64, sigma: u32) -> Result<u64, DirectQwtBuildError> {
    validate_row_capacity(rows, "QWTB")?;
    let levels = if rows == 0 {
        0
    } else {
        ((u32::BITS - sigma.leading_zeros()).max(1).div_ceil(2)) as u16
    };
    scratch_upper_bound(rows, levels)
}

/// Failures specific to direct QWTB construction.
#[derive(Debug)]
pub enum DirectQwtBuildError {
    Io(io::Error),
    InvalidInput(&'static str),
    BufferBudget { required: u64, limit: u64 },
    ScratchBudget { required: u64, limit: u64 },
    Overflow(&'static str),
}

impl fmt::Display for DirectQwtBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O: {error}"),
            Self::InvalidInput(detail) => write!(f, "invalid direct-QWT input: {detail}"),
            Self::BufferBudget { required, limit } => write!(
                f,
                "direct-QWT buffer budget requires {required} bytes, limit is {limit}"
            ),
            Self::ScratchBudget { required, limit } => write!(
                f,
                "direct-QWT scratch budget requires {required} bytes, limit is {limit}"
            ),
            Self::Overflow(detail) => write!(f, "direct-QWT size overflow: {detail}"),
        }
    }
}

impl std::error::Error for DirectQwtBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for DirectQwtBuildError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

pub(super) struct DirectWorkspace {
    pub(super) path: PathBuf,
}

impl DirectWorkspace {
    pub(super) fn create(parent: &Path) -> Result<(Self, u64), DirectQwtBuildError> {
        std::fs::create_dir_all(parent)?;
        loop {
            let id = DIRECT_BUILD_ID.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!("qwt-direct-{}-{id}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok((Self { path }, id)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
    }
}

impl Drop for DirectWorkspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

pub(super) struct OutputGuard {
    pub(super) path: PathBuf,
    pub(super) keep: bool,
}

impl Drop for OutputGuard {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

pub(super) struct SourceSet {
    pub(super) paths: Vec<PathBuf>,
    pub(super) owned: bool,
}

impl SourceSet {
    pub(super) fn remove_owned(&self) {
        if self.owned {
            for path in &self.paths {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

/// Block staging for the input stream of one direct level pass.
///
/// The three arrays together occupy exactly the per-stream buffer that a
/// `BufReader` over the source would have used: `4 + lanes` bytes per staged
/// row (raw little-endian symbols plus one byte per lane).
pub(super) struct BlockScratch {
    pub(super) symbols: Vec<u8>,
    pub(super) digits: Vec<u8>,
    pub(super) buckets: Vec<u8>,
}

impl BlockScratch {
    /// `lanes` is 1 when only digits are staged and 2 when per-row bucket
    /// ids are staged as well.
    pub(super) fn new(stream_bytes: usize, lanes: usize) -> Self {
        let rows = (stream_bytes / (4 + lanes)).clamp(1, MAX_BLOCK_ROWS);
        Self {
            symbols: vec![0; rows * 4],
            digits: vec![0; rows],
            buckets: vec![0; if lanes > 1 { rows } else { 0 }],
        }
    }

    pub(super) fn rows(&self) -> usize {
        self.digits.len()
    }
}

/// Writes each row of `symbols` to the bucket named by its id, one write per
/// run of equal ids. `ids[i]` selects the writer for the four bytes at
/// `symbols[4 * i..]`.
pub(super) fn write_bucket_runs(
    writers: &mut [BufWriter<File>],
    ids: &[u8],
    symbols: &[u8],
) -> io::Result<()> {
    debug_assert_eq!(ids.len() * 4, symbols.len());
    let mut start = 0;
    while start < ids.len() {
        let id = ids[start];
        let mut end = start + 1;
        while end < ids.len() && ids[end] == id {
            end += 1;
        }
        writers[usize::from(id)].write_all(&symbols[start * 4..end * 4])?;
        start = end;
    }
    Ok(())
}

/// Heap-buffer split for one direct build.
///
/// The main streams (input, partition outputs, packed level data) share the
/// budget evenly. The rank-superblock and select-sample writers receive a
/// separate allowance, `budget / 8` clamped to the auxiliary minimum and
/// maximum, split half to the superblock file and an eighth to each of the
/// four select files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DirectBuffers {
    pub(super) stream: usize,
    pub(super) superblocks: usize,
    pub(super) select: usize,
}

impl DirectBuffers {
    /// Smallest buffer budget that can run `streams` main streams plus the
    /// auxiliary writers.
    pub(super) fn min_bytes(streams: u64) -> Result<u64, DirectQwtBuildError> {
        streams
            .checked_mul(MIN_STREAM_BUFFER_BYTES)
            .and_then(|bytes| bytes.checked_add(MIN_AUX_BUFFER_BYTES))
            .ok_or(DirectQwtBuildError::Overflow("minimum buffer bytes"))
    }

    pub(super) fn plan(buffer_bytes: u64, streams: u64) -> Result<Self, DirectQwtBuildError> {
        let required = Self::min_bytes(streams)?;
        if buffer_bytes < required {
            return Err(DirectQwtBuildError::BufferBudget {
                required,
                limit: buffer_bytes,
            });
        }
        let aux = (buffer_bytes / AUX_BUFFER_BUDGET_DIVISOR)
            .clamp(MIN_AUX_BUFFER_BYTES, MAX_AUX_BUFFER_BYTES);
        let convert = |bytes: u64| {
            usize::try_from(bytes).map_err(|_| DirectQwtBuildError::Overflow("buffer bytes"))
        };
        Ok(Self {
            stream: convert((buffer_bytes - aux) / streams)?,
            superblocks: convert(aux / 2)?,
            select: convert(aux / 8)?,
        })
    }
}

pub(super) struct LevelFiles {
    pub(super) data: PathBuf,
    pub(super) superblocks: PathBuf,
    pub(super) select: [PathBuf; 4],
}

pub(super) struct LevelSummary {
    pub(super) n_datalines: u32,
    pub(super) position_bits: u64,
    pub(super) n_superblocks: u32,
    pub(super) n_sel: [u32; 4],
    pub(super) n_occs_smaller: [u64; 5],
    pub(super) aux_bytes: u64,
}

pub(super) struct LevelEncoder {
    data: BufWriter<File>,
    superblocks: BufWriter<File>,
    select: [BufWriter<File>; 4],
    words: [u128; 4],
    line_rows: u16,
    n_datalines: u32,
    global: [u64; 4],
    block: [u64; 4],
    superblock_words: [u128; 4],
    n_superblocks: u32,
    n_sel: [u32; 4],
}

impl LevelEncoder {
    pub(super) fn new(
        files: &LevelFiles,
        buffers: DirectBuffers,
    ) -> Result<Self, DirectQwtBuildError> {
        let data = BufWriter::with_capacity(buffers.stream, File::create(&files.data)?);
        let superblocks =
            BufWriter::with_capacity(buffers.superblocks, File::create(&files.superblocks)?);
        let select = [
            BufWriter::with_capacity(buffers.select, File::create(&files.select[0])?),
            BufWriter::with_capacity(buffers.select, File::create(&files.select[1])?),
            BufWriter::with_capacity(buffers.select, File::create(&files.select[2])?),
            BufWriter::with_capacity(buffers.select, File::create(&files.select[3])?),
        ];
        Ok(Self {
            data,
            superblocks,
            select,
            words: [0; 4],
            line_rows: 0,
            n_datalines: 0,
            global: [0; 4],
            block: [0; 4],
            superblock_words: [0; 4],
            n_superblocks: 1,
            n_sel: [0; 4],
        })
    }

    /// Appends one digit at absolute `row`. Rows must arrive sequentially from
    /// zero; `push_block` is the production path and this is its scalar oracle.
    #[cfg(test)]
    pub(super) fn push(&mut self, digit: u8, row: u64) -> Result<(), DirectQwtBuildError> {
        if row > 0 && row.is_multiple_of(DIRECT_SUPERBLOCK_ROWS) {
            self.flush_superblock()?;
            self.superblock_words =
                std::array::from_fn(|symbol| (self.global[symbol] as u128) << 84);
            self.block = [0; 4];
            self.n_superblocks = self
                .n_superblocks
                .checked_add(1)
                .ok_or(DirectQwtBuildError::Overflow("superblock count"))?;
        }
        if row.is_multiple_of(DIRECT_BLOCK_ROWS) {
            self.set_block_counters(((row / DIRECT_BLOCK_ROWS) % 8) as usize)?;
        }

        let symbol = usize::from(digit);
        if self.global[symbol].is_multiple_of(DIRECT_SELECT_SAMPLE_ROWS) {
            let superblock = u32::try_from(row / DIRECT_SUPERBLOCK_ROWS)
                .map_err(|_| DirectQwtBuildError::Overflow("select superblock id"))?;
            self.select[symbol].write_all(&superblock.to_le_bytes())?;
            self.n_sel[symbol] = self.n_sel[symbol]
                .checked_add(1)
                .ok_or(DirectQwtBuildError::Overflow("select sample count"))?;
        }
        self.global[symbol] += 1;
        self.block[symbol] += 1;

        let line_pos = self.line_rows as u8;
        let high_word = usize::from(line_pos >> 7);
        let low_word = high_word + 2;
        let shift = line_pos & 127;
        self.words[high_word] |= u128::from(digit >> 1) << shift;
        self.words[low_word] |= u128::from(digit & 1) << shift;
        self.line_rows += 1;
        if self.line_rows == DIRECT_BLOCK_ROWS as u16 {
            self.flush_line()?;
        }
        Ok(())
    }

    /// Appends `digits` (each in `0..4`) starting at absolute `start_row`.
    ///
    /// Produces exactly the bytes of calling `push` once per digit. Digits are
    /// processed in pieces that never cross a 64-row boundary, so every piece
    /// lies inside one rank block, superblock, data-line word, and select
    /// sample interval. Each piece is transposed into two bit planes whose
    /// popcounts give the per-symbol counts without per-digit branching.
    pub(super) fn push_block(
        &mut self,
        digits: &[u8],
        start_row: u64,
    ) -> Result<(), DirectQwtBuildError> {
        let mut row = start_row;
        let mut rest = digits;
        while !rest.is_empty() {
            let take = rest.len().min((PIECE_ROWS - row % PIECE_ROWS) as usize);
            let (piece, tail) = rest.split_at(take);
            self.push_piece(piece, row)?;
            row += take as u64;
            rest = tail;
        }
        Ok(())
    }

    fn push_piece(&mut self, piece: &[u8], row: u64) -> Result<(), DirectQwtBuildError> {
        debug_assert!(!piece.is_empty() && piece.len() as u64 <= PIECE_ROWS - row % PIECE_ROWS);
        debug_assert_eq!(u64::from(self.line_rows), row % DIRECT_BLOCK_ROWS);
        if row > 0 && row.is_multiple_of(DIRECT_SUPERBLOCK_ROWS) {
            self.flush_superblock()?;
            self.superblock_words =
                std::array::from_fn(|symbol| (self.global[symbol] as u128) << 84);
            self.block = [0; 4];
            self.n_superblocks = self
                .n_superblocks
                .checked_add(1)
                .ok_or(DirectQwtBuildError::Overflow("superblock count"))?;
        }
        if row.is_multiple_of(DIRECT_BLOCK_ROWS) {
            self.set_block_counters(((row / DIRECT_BLOCK_ROWS) % 8) as usize)?;
        }

        let mut high = 0u64;
        let mut low = 0u64;
        for (index, &digit) in piece.iter().enumerate() {
            debug_assert!(digit < 4);
            high |= u64::from((digit >> 1) & 1) << index;
            low |= u64::from(digit & 1) << index;
        }
        let rows = piece.len() as u64;
        let c3 = u64::from((high & low).count_ones());
        let c2 = u64::from((high & !low).count_ones());
        let c1 = u64::from((!high & low).count_ones());
        let counts = [rows - c1 - c2 - c3, c1, c2, c3];

        // A piece holds at most 64 rows, so each symbol crosses at most one
        // select-sample threshold, and every row shares one superblock id.
        for (symbol, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let first = self.global[symbol];
            if (first + count - 1) / DIRECT_SELECT_SAMPLE_ROWS
                != first.wrapping_sub(1) / DIRECT_SELECT_SAMPLE_ROWS
            {
                let superblock = u32::try_from(row / DIRECT_SUPERBLOCK_ROWS)
                    .map_err(|_| DirectQwtBuildError::Overflow("select superblock id"))?;
                self.select[symbol].write_all(&superblock.to_le_bytes())?;
                self.n_sel[symbol] = self.n_sel[symbol]
                    .checked_add(1)
                    .ok_or(DirectQwtBuildError::Overflow("select sample count"))?;
            }
            self.global[symbol] = first + count;
            self.block[symbol] += count;
        }

        let line_pos = self.line_rows;
        let high_word = usize::from(line_pos >> 7);
        let shift = line_pos & 127;
        self.words[high_word] |= u128::from(high) << shift;
        self.words[high_word + 2] |= u128::from(low) << shift;
        self.line_rows += piece.len() as u16;
        if self.line_rows == DIRECT_BLOCK_ROWS as u16 {
            self.flush_line()?;
        }
        Ok(())
    }

    pub(super) fn finish(
        mut self,
        rows: u64,
        files: &LevelFiles,
    ) -> Result<LevelSummary, DirectQwtBuildError> {
        if self.line_rows != 0 {
            self.flush_line()?;
        }

        if rows.is_multiple_of(DIRECT_SUPERBLOCK_ROWS) {
            self.flush_superblock()?;
            self.superblock_words =
                std::array::from_fn(|symbol| (self.global[symbol] as u128) << 84);
            self.block = [0; 4];
            self.n_superblocks = self
                .n_superblocks
                .checked_add(1)
                .ok_or(DirectQwtBuildError::Overflow("terminal superblock count"))?;
        } else if rows.is_multiple_of(DIRECT_BLOCK_ROWS) {
            self.set_block_counters(((rows / DIRECT_BLOCK_ROWS) % 8) as usize)?;
        }
        let next_block = ((rows / DIRECT_BLOCK_ROWS) % 8 + 1) as usize;
        if next_block < 8 {
            self.set_block_counters(next_block)?;
        }
        self.flush_superblock()?;

        for symbol in 0..4 {
            if self.n_sel[symbol] == 0 {
                self.select[symbol].write_all(&0u32.to_le_bytes())?;
                self.n_sel[symbol] += 1;
            }
            let sentinel = self.n_superblocks - 1;
            self.select[symbol].write_all(&sentinel.to_le_bytes())?;
            self.n_sel[symbol] += 1;
        }
        // Level files are transient scratch: they are read back through the
        // page cache and removed, so syncing them only costs a device flush.
        // Flushing here, rather than in BufWriter's Drop, surfaces write
        // errors and makes the lengths read below complete.
        self.data.flush()?;
        self.superblocks.flush()?;
        for select in &mut self.select {
            select.flush()?;
        }

        let mut cumulative = [0u64; 5];
        for symbol in 0..4 {
            cumulative[symbol + 1] = cumulative[symbol] + self.global[symbol];
        }
        let mut aux_bytes = file_len(&files.data)?
            .checked_add(file_len(&files.superblocks)?)
            .ok_or(DirectQwtBuildError::Overflow("level auxiliary bytes"))?;
        for path in &files.select {
            aux_bytes = aux_bytes
                .checked_add(file_len(path)?)
                .ok_or(DirectQwtBuildError::Overflow("level auxiliary bytes"))?;
        }
        Ok(LevelSummary {
            n_datalines: self.n_datalines,
            position_bits: rows
                .checked_mul(2)
                .ok_or(DirectQwtBuildError::Overflow("level position bits"))?,
            n_superblocks: self.n_superblocks,
            n_sel: self.n_sel,
            n_occs_smaller: cumulative,
            aux_bytes,
        })
    }

    fn flush_line(&mut self) -> Result<(), DirectQwtBuildError> {
        for word in self.words {
            self.data.write_all(&word.to_le_bytes())?;
        }
        self.words = [0; 4];
        self.line_rows = 0;
        self.n_datalines = self
            .n_datalines
            .checked_add(1)
            .ok_or(DirectQwtBuildError::Overflow("data-line count"))?;
        Ok(())
    }

    fn flush_superblock(&mut self) -> Result<(), DirectQwtBuildError> {
        for word in self.superblock_words {
            self.superblocks.write_all(&word.to_le_bytes())?;
        }
        Ok(())
    }

    fn set_block_counters(&mut self, block_id: usize) -> Result<(), DirectQwtBuildError> {
        if block_id == 0 {
            return Ok(());
        }
        if block_id >= 8 || self.block.iter().any(|count| *count >= (1 << 12)) {
            return Err(DirectQwtBuildError::InvalidInput(
                "rank-support block counter is out of range",
            ));
        }
        let shift = (block_id - 1) * 12;
        for symbol in 0..4 {
            self.superblock_words[symbol] |= (self.block[symbol] as u128) << shift;
        }
        Ok(())
    }
}

/// Builds one canonical QWT256/u32 `QWTB` section directly from a replay file.
///
/// `source` must contain exactly `rows` little-endian `u32` values and `sigma`
/// must equal the maximum value for non-empty input. The output is installed
/// only after it is synced completely; all intermediate files are removed on
/// success and failure.
pub fn write_qwt256_u32_direct(
    source: &Path,
    rows: u64,
    sigma: u32,
    output: &Path,
    scratch: &Path,
    budget: DirectQwtBuildBudget,
) -> Result<DirectQwtBuildStats, DirectQwtBuildError> {
    ensure_le().map_err(|_| DirectQwtBuildError::InvalidInput("big-endian host"))?;
    if source == output {
        return Err(DirectQwtBuildError::InvalidInput(
            "source and output paths must differ",
        ));
    }
    validate_row_capacity(rows, "QWTB")?;
    let source_bytes = rows
        .checked_mul(4)
        .ok_or(DirectQwtBuildError::Overflow("source bytes"))?;
    if file_len(source)? != source_bytes {
        return Err(DirectQwtBuildError::InvalidInput(
            "source length does not equal rows * 4",
        ));
    }
    let buffers = DirectBuffers::plan(budget.buffer_bytes, DIRECT_STREAMS)?;
    let levels = if rows == 0 {
        0
    } else {
        ((u32::BITS - sigma.leading_zeros()).max(1).div_ceil(2)) as u16
    };
    let scratch_required = qwt256_u32_scratch_upper_bound(rows, sigma)?;
    if scratch_required > budget.scratch_bytes {
        return Err(DirectQwtBuildError::ScratchBudget {
            required: scratch_required,
            limit: budget.scratch_bytes,
        });
    }

    let (workspace, build_id) = DirectWorkspace::create(scratch)?;
    let mut tmp_os = output.as_os_str().to_os_string();
    tmp_os.push(format!(".qwt-direct-{}-{build_id}.tmp", std::process::id()));
    let tmp_path = PathBuf::from(tmp_os);
    let mut output_guard = OutputGuard {
        path: tmp_path.clone(),
        keep: false,
    };
    let mut out = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&tmp_path)?;

    let directory_bytes = usize::from(levels)
        .checked_mul(LEVEL_DIR_SIZE)
        .and_then(|bytes| HEADER_SIZE.checked_add(bytes))
        .ok_or(DirectQwtBuildError::Overflow("QWTB directory bytes"))?;
    let payload_start = align_up(directory_bytes, 64);
    out.set_len(
        u64::try_from(payload_start)
            .map_err(|_| DirectQwtBuildError::Overflow("QWTB payload start"))?,
    )?;
    out.seek(SeekFrom::Start(payload_start as u64))?;
    let mut cursor = payload_start as u64;
    let mut dirs = Vec::with_capacity(usize::from(levels));
    let mut current = SourceSet {
        paths: vec![source.to_path_buf()],
        owned: false,
    };
    let per_stream_buffer = buffers.stream;
    let copy_buffer_len = usize::try_from(budget.buffer_bytes.min(1024 * 1024))
        .map_err(|_| DirectQwtBuildError::Overflow("copy buffer bytes"))?;
    let mut peak_scratch = if levels == 0 { 0 } else { source_bytes };
    let mut actual_sigma = 0u32;
    let mut blocks = BlockScratch::new(per_stream_buffer, 1);

    for level in 0..levels {
        let shift = 2 * (usize::from(levels - level - 1));
        let needs_partition = level + 1 < levels;
        let level_files = LevelFiles {
            data: workspace.path.join(format!("level-{level}-data.bin")),
            superblocks: workspace
                .path
                .join(format!("level-{level}-superblocks.bin")),
            select: std::array::from_fn(|symbol| {
                workspace
                    .path
                    .join(format!("level-{level}-select-{symbol}.bin"))
            }),
        };
        let mut encoder = LevelEncoder::new(&level_files, buffers)?;
        let bucket_paths: [PathBuf; 4] = std::array::from_fn(|symbol| {
            workspace
                .path
                .join(format!("level-{level}-bucket-{symbol}.u32"))
        });
        let mut buckets = if needs_partition {
            Some([
                BufWriter::with_capacity(per_stream_buffer, File::create(&bucket_paths[0])?),
                BufWriter::with_capacity(per_stream_buffer, File::create(&bucket_paths[1])?),
                BufWriter::with_capacity(per_stream_buffer, File::create(&bucket_paths[2])?),
                BufWriter::with_capacity(per_stream_buffer, File::create(&bucket_paths[3])?),
            ])
        } else {
            None
        };
        let mut seen = 0u64;
        for path in &current.paths {
            let mut file = File::open(path)?;
            let path_rows = file.metadata()?.len() / 4;
            let mut remaining = path_rows;
            while remaining > 0 {
                let block = remaining.min(blocks.rows() as u64) as usize;
                let symbols = &mut blocks.symbols[..block * 4];
                file.read_exact(symbols)?;
                remaining -= block as u64;
                let digits = &mut blocks.digits[..block];
                let mut block_max = 0u32;
                for (digit, encoded) in digits.iter_mut().zip(symbols.as_chunks::<4>().0) {
                    let symbol = u32::from_le_bytes(*encoded);
                    block_max = block_max.max(symbol);
                    *digit = ((symbol >> shift) & 3) as u8;
                }
                if block_max > sigma {
                    return Err(DirectQwtBuildError::InvalidInput(
                        "source symbol exceeds declared sigma",
                    ));
                }
                if level == 0 {
                    actual_sigma = actual_sigma.max(block_max);
                }
                encoder.push_block(digits, seen)?;
                if let Some(bucket_writers) = buckets.as_mut() {
                    write_bucket_runs(bucket_writers, digits, symbols)?;
                }
                seen += block as u64;
            }
        }
        if seen != rows {
            return Err(DirectQwtBuildError::InvalidInput(
                "partitioned source row count changed between levels",
            ));
        }
        if level == 0 && rows != 0 && actual_sigma != sigma {
            return Err(DirectQwtBuildError::InvalidInput(
                "declared sigma is not the maximum source symbol",
            ));
        }
        if let Some(bucket_writers) = buckets.as_mut() {
            for writer in bucket_writers {
                writer.flush()?;
            }
        }
        drop(buckets);

        let summary = encoder.finish(rows, &level_files)?;
        let partition_bytes = if needs_partition { source_bytes } else { 0 };
        let level_peak = source_bytes
            .checked_add(partition_bytes)
            .and_then(|bytes| bytes.checked_add(summary.aux_bytes))
            .ok_or(DirectQwtBuildError::Overflow("observed scratch bytes"))?;
        peak_scratch = peak_scratch.max(level_peak);
        if peak_scratch > budget.scratch_bytes {
            return Err(DirectQwtBuildError::ScratchBudget {
                required: peak_scratch,
                limit: budget.scratch_bytes,
            });
        }

        let mut copy_buffer = vec![0u8; copy_buffer_len];
        cursor = align_output(&mut out, cursor, 64)?;
        let off_data = cursor;
        cursor = append_file(&mut out, &level_files.data, cursor, &mut copy_buffer)?;
        cursor = align_output(&mut out, cursor, 64)?;
        let off_superblocks = cursor;
        cursor = append_file(&mut out, &level_files.superblocks, cursor, &mut copy_buffer)?;
        cursor = align_output(&mut out, cursor, 4)?;
        let mut off_sel = [0u64; 4];
        for (symbol, offset) in off_sel.iter_mut().enumerate() {
            *offset = cursor;
            cursor = append_file(
                &mut out,
                &level_files.select[symbol],
                cursor,
                &mut copy_buffer,
            )?;
        }
        dirs.push(LevelDir {
            off_data,
            n_datalines: summary.n_datalines,
            position_bits: summary.position_bits,
            off_superblocks,
            n_superblocks: summary.n_superblocks,
            off_sel,
            n_sel: summary.n_sel,
            n_occs_smaller: summary.n_occs_smaller,
        });
        drop(copy_buffer);

        current.remove_owned();
        current = if needs_partition {
            SourceSet {
                paths: bucket_paths.into_iter().collect(),
                owned: true,
            }
        } else {
            SourceSet {
                paths: Vec::new(),
                owned: false,
            }
        };
        remove_level_files(&level_files);
    }
    current.remove_owned();

    write_header_and_directory(&mut out, rows, sigma, levels, &dirs)?;
    out.set_len(cursor)?;
    out.sync_all()?;
    drop(out);
    std::fs::rename(&tmp_path, output)?;
    output_guard.keep = true;
    Ok(DirectQwtBuildStats {
        rows,
        levels,
        output_bytes: cursor,
        peak_scratch_bytes: peak_scratch,
    })
}

pub(super) fn container_output_upper_bound(
    rows: u64,
    levels: u16,
    directory_entry_bytes: u64,
    metadata_bytes: u64,
) -> Result<u64, DirectQwtBuildError> {
    let directory = u64::from(levels)
        .checked_mul(directory_entry_bytes)
        .and_then(|bytes| bytes.checked_add(HEADER_SIZE as u64))
        .ok_or(DirectQwtBuildError::Overflow("output directory bytes"))?;
    let payload_start = align_up_u64(
        directory
            .checked_add(metadata_bytes)
            .ok_or(DirectQwtBuildError::Overflow("output metadata bytes"))?,
        64,
    )?;
    let data = rows
        .div_ceil(DIRECT_BLOCK_ROWS)
        .checked_mul(64)
        .ok_or(DirectQwtBuildError::Overflow("output data-line bytes"))?;
    let superblocks = rows
        .checked_add(DIRECT_SUPERBLOCK_ROWS)
        .ok_or(DirectQwtBuildError::Overflow("output superblock rows"))?
        / DIRECT_SUPERBLOCK_ROWS;
    let superblocks = superblocks
        .checked_mul(64)
        .ok_or(DirectQwtBuildError::Overflow("output superblock bytes"))?;
    let samples = rows
        .div_ceil(DIRECT_SELECT_SAMPLE_ROWS)
        .checked_add(8)
        .and_then(|entries| entries.checked_mul(4))
        .ok_or(DirectQwtBuildError::Overflow("output select bytes"))?;
    let per_level = data
        .checked_add(superblocks)
        .and_then(|bytes| bytes.checked_add(samples))
        .and_then(|bytes| bytes.checked_add(128))
        .ok_or(DirectQwtBuildError::Overflow("output level bytes"))?;
    u64::from(levels)
        .checked_mul(per_level)
        .and_then(|bytes| bytes.checked_add(payload_start))
        .ok_or(DirectQwtBuildError::Overflow("output container bytes"))
}

pub(super) fn validate_row_capacity(
    rows: u64,
    format: &'static str,
) -> Result<(), DirectQwtBuildError> {
    if rows >= (1u64 << 43) {
        return Err(DirectQwtBuildError::InvalidInput(
            "row count exceeds QWT256 rank/select capacity",
        ));
    }
    if rows.div_ceil(DIRECT_BLOCK_ROWS) > u64::from(u32::MAX) {
        return Err(DirectQwtBuildError::InvalidInput(match format {
            "HQWB" => "row count exceeds HQWB data-line capacity",
            _ => "row count exceeds QWTB data-line capacity",
        }));
    }
    Ok(())
}

fn align_up_u64(value: u64, alignment: u64) -> Result<u64, DirectQwtBuildError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or(DirectQwtBuildError::Overflow("output alignment"))
}

fn scratch_upper_bound(rows: u64, levels: u16) -> Result<u64, DirectQwtBuildError> {
    if levels == 0 {
        return Ok(0);
    }
    let source = rows
        .checked_mul(4)
        .ok_or(DirectQwtBuildError::Overflow("scratch source bytes"))?;
    let partition = if levels > 1 { source } else { 0 };
    let data = rows
        .div_ceil(DIRECT_BLOCK_ROWS)
        .checked_mul(64)
        .ok_or(DirectQwtBuildError::Overflow("scratch data-line bytes"))?;
    let superblocks = rows
        .checked_add(DIRECT_SUPERBLOCK_ROWS)
        .ok_or(DirectQwtBuildError::Overflow("scratch superblock rows"))?
        / DIRECT_SUPERBLOCK_ROWS;
    let superblocks = superblocks
        .checked_mul(64)
        .ok_or(DirectQwtBuildError::Overflow("scratch superblock bytes"))?;
    let samples = rows
        .div_ceil(DIRECT_SELECT_SAMPLE_ROWS)
        .checked_add(8)
        .and_then(|entries| entries.checked_mul(4))
        .ok_or(DirectQwtBuildError::Overflow("scratch select bytes"))?;
    source
        .checked_add(partition)
        .and_then(|bytes| bytes.checked_add(data))
        .and_then(|bytes| bytes.checked_add(superblocks))
        .and_then(|bytes| bytes.checked_add(samples))
        .ok_or(DirectQwtBuildError::Overflow("scratch upper bound"))
}

fn write_header_and_directory(
    out: &mut File,
    rows: u64,
    sigma: u32,
    levels: u16,
    dirs: &[LevelDir],
) -> Result<(), DirectQwtBuildError> {
    let directory_bytes = usize::from(levels)
        .checked_mul(LEVEL_DIR_SIZE)
        .and_then(|bytes| HEADER_SIZE.checked_add(bytes))
        .ok_or(DirectQwtBuildError::Overflow("header directory bytes"))?;
    let mut header = vec![0u8; directory_bytes];
    header[0..4].copy_from_slice(QWTB_MAGIC);
    header[4..6].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    header[6] = 0;
    header[7] = 4;
    header[8..16].copy_from_slice(&rows.to_le_bytes());
    header[16..24].copy_from_slice(&u64::from(sigma).to_le_bytes());
    header[24..26].copy_from_slice(&levels.to_le_bytes());
    for (level, dir) in dirs.iter().enumerate() {
        let start = HEADER_SIZE + level * LEVEL_DIR_SIZE;
        dir.write(&mut header[start..start + LEVEL_DIR_SIZE]);
    }
    out.seek(SeekFrom::Start(0))?;
    out.write_all(&header)?;
    Ok(())
}

pub(super) fn align_output(
    out: &mut File,
    cursor: u64,
    align: usize,
) -> Result<u64, DirectQwtBuildError> {
    let cursor_usize =
        usize::try_from(cursor).map_err(|_| DirectQwtBuildError::Overflow("output cursor"))?;
    let aligned = u64::try_from(align_up(cursor_usize, align))
        .map_err(|_| DirectQwtBuildError::Overflow("aligned output cursor"))?;
    if aligned > cursor {
        out.set_len(aligned)?;
    }
    out.seek(SeekFrom::Start(aligned))?;
    Ok(aligned)
}

pub(super) fn append_file(
    out: &mut File,
    path: &Path,
    mut cursor: u64,
    buffer: &mut [u8],
) -> Result<u64, DirectQwtBuildError> {
    let mut input = File::open(path)?;
    loop {
        let read = input.read(buffer)?;
        if read == 0 {
            break;
        }
        out.write_all(&buffer[..read])?;
        cursor = cursor
            .checked_add(read as u64)
            .ok_or(DirectQwtBuildError::Overflow("output payload bytes"))?;
    }
    Ok(cursor)
}

pub(super) fn file_len(path: &Path) -> Result<u64, DirectQwtBuildError> {
    Ok(std::fs::metadata(path)?.len())
}

pub(super) fn remove_level_files(files: &LevelFiles) {
    let _ = std::fs::remove_file(&files.data);
    let _ = std::fs::remove_file(&files.superblocks);
    for path in &files.select {
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytes::{qwt256_to_bytes, QwtView};
    use crate::{AccessUnsigned, RankUnsigned, SelectUnsigned, QWT256};

    fn test_dir(name: &str) -> PathBuf {
        let id = DIRECT_BUILD_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "qwt-direct-test-{name}-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_source(path: &Path, values: &[u32]) {
        let mut file = File::create(path).unwrap();
        for value in values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        file.sync_all().unwrap();
    }

    fn min_buffer_bytes() -> u64 {
        DirectBuffers::min_bytes(DIRECT_STREAMS).unwrap()
    }

    fn budget_with_buffer(rows: u64, buffer_bytes: u64) -> DirectQwtBuildBudget {
        DirectQwtBuildBudget {
            buffer_bytes,
            scratch_bytes: rows.saturating_mul(16).saturating_add(1 << 20),
        }
    }

    fn budget(rows: u64) -> DirectQwtBuildBudget {
        budget_with_buffer(rows, min_buffer_bytes())
    }

    fn assert_matches_owned(name: &str, values: &[u32], buffer_bytes: u64) {
        let dir = test_dir(name);
        let source = dir.join("source.u32");
        let output = dir.join("output.qwtb");
        let scratch = dir.join("scratch");
        write_source(&source, values);
        let rows = values.len() as u64;
        let sigma = values.iter().copied().max().unwrap_or(0);
        let stats = write_qwt256_u32_direct(
            &source,
            rows,
            sigma,
            &output,
            &scratch,
            budget_with_buffer(rows, buffer_bytes),
        )
        .unwrap();
        let direct = std::fs::read(&output).unwrap();
        assert_eq!(
            direct,
            qwt256_to_bytes(&QWT256::from(values.to_vec())).unwrap()
        );
        assert_eq!(stats.output_bytes, direct.len() as u64);
        assert!(stats.output_bytes <= qwt256_u32_output_upper_bound(rows, sigma).unwrap());
        assert!(stats.peak_scratch_bytes <= qwt256_u32_scratch_upper_bound(rows, sigma).unwrap());
        assert!(std::fs::read_dir(&scratch).unwrap().next().is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn direct_qwtb_matches_owned_bytes_and_queries() {
        let values: Vec<u32> = (0..25_013)
            .map(|row| ((row * 17 + row / 11) % 1_003) as u32)
            .collect();
        let dir = test_dir("parity");
        let source = dir.join("source.u32");
        let output = dir.join("output.qwtb");
        let scratch = dir.join("scratch");
        write_source(&source, &values);
        let sigma = *values.iter().max().unwrap();
        let stats = write_qwt256_u32_direct(
            &source,
            values.len() as u64,
            sigma,
            &output,
            &scratch,
            budget(values.len() as u64),
        )
        .unwrap();
        let direct = std::fs::read(&output).unwrap();
        let owned = qwt256_to_bytes(&QWT256::from(values.clone())).unwrap();
        assert_eq!(direct, owned);
        assert_eq!(stats.output_bytes, direct.len() as u64);
        assert!(
            stats.output_bytes
                <= qwt256_u32_output_upper_bound(values.len() as u64, sigma).unwrap()
        );
        assert!(stats.peak_scratch_bytes <= budget(values.len() as u64).scratch_bytes);

        let view = QwtView::<u32, 256>::from_bytes(&direct).unwrap();
        for &row in &[0usize, 1, 255, 256, 8_192, values.len() - 1] {
            assert_eq!(view.get(row), Some(values[row]));
        }
        let owned_tree = QWT256::from(values.clone());
        for &symbol in &[0u32, 1, 17, 500, sigma] {
            for &end in &[0usize, 1, 257, 10_000, values.len()] {
                assert_eq!(view.rank(symbol, end), owned_tree.rank(symbol, end));
            }
            let count = owned_tree.rank(symbol, values.len()).unwrap_or(0);
            for occurrence in 0..count.min(3) {
                assert_eq!(
                    view.select(symbol, occurrence),
                    owned_tree.select(symbol, occurrence)
                );
            }
        }
        for &(start, end) in &[(0usize, values.len()), (17, 1_337), (8_191, 16_777)] {
            assert_eq!(
                view.extract_range_distinct(start..end),
                owned_tree.extract_range_distinct(start..end)
            );
        }
        assert!(std::fs::read_dir(&scratch).unwrap().next().is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn direct_qwtb_empty_matches_owned() {
        let dir = test_dir("empty");
        let source = dir.join("source.u32");
        let output = dir.join("output.qwtb");
        let scratch = dir.join("scratch");
        write_source(&source, &[]);
        let stats = write_qwt256_u32_direct(&source, 0, 0, &output, &scratch, budget(0)).unwrap();
        let direct = std::fs::read(&output).unwrap();
        let owned = qwt256_to_bytes(&QWT256::from(Vec::<u32>::new())).unwrap();
        assert_eq!(direct, owned);
        assert_eq!(stats.peak_scratch_bytes, 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn direct_qwtb_matches_owned_at_level_and_block_boundaries() {
        for (case, values) in [
            ("zero", vec![0u32; 8_193]),
            (
                "one-level",
                (0..2_049).map(|row| (row % 4) as u32).collect(),
            ),
            (
                "two-level",
                (0..8_193).map(|row| (row % 5) as u32).collect(),
            ),
        ] {
            let dir = test_dir(case);
            let source = dir.join("source.u32");
            let output = dir.join("output.qwtb");
            let scratch = dir.join("scratch");
            write_source(&source, &values);
            let sigma = *values.iter().max().unwrap();
            write_qwt256_u32_direct(
                &source,
                values.len() as u64,
                sigma,
                &output,
                &scratch,
                budget(values.len() as u64),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(output).unwrap(),
                qwt256_to_bytes(&QWT256::from(values)).unwrap()
            );
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn direct_qwtb_enforces_budget_and_cleans_scratch() {
        let values: Vec<u32> = (0..10_000).map(|row| (row % 257) as u32).collect();
        let dir = test_dir("budget");
        let source = dir.join("source.u32");
        let output = dir.join("output.qwtb");
        let scratch = dir.join("scratch");
        write_source(&source, &values);
        let buffer_error = write_qwt256_u32_direct(
            &source,
            values.len() as u64,
            256,
            &output,
            &scratch,
            DirectQwtBuildBudget {
                buffer_bytes: min_buffer_bytes() - 1,
                scratch_bytes: u64::MAX,
            },
        )
        .unwrap_err();
        assert!(matches!(
            buffer_error,
            DirectQwtBuildError::BufferBudget { required, limit }
                if required == min_buffer_bytes() && limit == min_buffer_bytes() - 1
        ));
        assert!(!output.exists());
        let error = write_qwt256_u32_direct(
            &source,
            values.len() as u64,
            256,
            &output,
            &scratch,
            DirectQwtBuildBudget {
                buffer_bytes: min_buffer_bytes(),
                scratch_bytes: 1,
            },
        )
        .unwrap_err();
        assert!(matches!(error, DirectQwtBuildError::ScratchBudget { .. }));
        assert!(!output.exists());
        assert!(!scratch.exists() || std::fs::read_dir(&scratch).unwrap().next().is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn public_scratch_upper_bound_matches_enforced_requirement() {
        let values: Vec<u32> = (0..10_000).map(|row| (row % 257) as u32).collect();
        let rows = values.len() as u64;
        let sigma = 256;
        let required = qwt256_u32_scratch_upper_bound(rows, sigma).unwrap();
        assert_eq!(qwt256_u32_scratch_upper_bound(0, sigma).unwrap(), 0);
        let dir = test_dir("public-scratch-bound");
        let source = dir.join("source.u32");
        let output = dir.join("output.qwtb");
        let scratch = dir.join("scratch");
        write_source(&source, &values);
        let error = write_qwt256_u32_direct(
            &source,
            rows,
            sigma,
            &output,
            &scratch,
            DirectQwtBuildBudget {
                buffer_bytes: budget(rows).buffer_bytes,
                scratch_bytes: required - 1,
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            DirectQwtBuildError::ScratchBudget { required: r, .. } if r == required
        ));
        write_qwt256_u32_direct(
            &source,
            rows,
            sigma,
            &output,
            &scratch,
            DirectQwtBuildBudget {
                buffer_bytes: budget(rows).buffer_bytes,
                scratch_bytes: required,
            },
        )
        .unwrap();
        assert_eq!(
            std::fs::read(&output).unwrap(),
            qwt256_to_bytes(&QWT256::from(values)).unwrap()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn direct_buffer_plan_stays_within_budget_and_meets_minimums() {
        let minimum = min_buffer_bytes();
        for buffer_bytes in [
            minimum,
            24 * 1_024,
            minimum + 1,
            64 * 1_024,
            1 << 20,
            37_000_000,
            u64::MAX / 2,
        ] {
            let plan = DirectBuffers::plan(buffer_bytes, DIRECT_STREAMS).unwrap();
            assert!(plan.stream as u64 >= MIN_STREAM_BUFFER_BYTES);
            assert!(plan.superblocks > 0 && plan.select > 0);
            let heap = DIRECT_STREAMS * plan.stream as u64
                + plan.superblocks as u64
                + 4 * plan.select as u64;
            assert!(heap <= buffer_bytes, "{heap} > {buffer_bytes}");
        }
        assert!(matches!(
            DirectBuffers::plan(minimum - 1, DIRECT_STREAMS),
            Err(DirectQwtBuildError::BufferBudget { required, .. }) if required == minimum
        ));
    }

    #[test]
    fn direct_qwtb_accepts_exact_minimum_and_small_callers_with_parity() {
        let values: Vec<u32> = (0..12_345)
            .map(|row| ((row * 31 + row / 7) % 777) as u32)
            .collect();
        assert_matches_owned("minimum", &values, min_buffer_bytes());
        assert_matches_owned("small-budget", &values, 24 * 1_024);
        assert_matches_owned("large-budget", &values, 8 << 20);
    }

    #[test]
    fn direct_qwtb_buffered_aux_files_match_owned_across_many_boundaries() {
        let values: Vec<u32> = (0..300_007u64)
            .map(|row| {
                let spread = (row.wrapping_mul(2_654_435_761) >> 7) % 70_001;
                if row % 3 == 0 {
                    (row % 5) as u32
                } else {
                    spread as u32
                }
            })
            .collect();
        assert_matches_owned("multi-level-min", &values, min_buffer_bytes());
        assert_matches_owned("multi-level-24k", &values, 24 * 1_024);
        assert_matches_owned("multi-level-1m", &values, 1 << 20);
    }

    #[test]
    fn direct_qwtb_rejects_unrepresentable_data_line_count_up_front() {
        let dir = test_dir("row-capacity");
        let error = write_qwt256_u32_direct(
            &dir.join("missing-source.u32"),
            u64::from(u32::MAX) * DIRECT_BLOCK_ROWS + 1,
            0,
            &dir.join("output.qwtb"),
            &dir.join("scratch"),
            DirectQwtBuildBudget {
                buffer_bytes: u64::MAX,
                scratch_bytes: u64::MAX,
            },
        )
        .unwrap_err();
        assert!(matches!(error, DirectQwtBuildError::InvalidInput(_)));
        assert!(!dir.join("output.qwtb").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn level_files(dir: &Path) -> LevelFiles {
        std::fs::create_dir_all(dir).unwrap();
        LevelFiles {
            data: dir.join("data.bin"),
            superblocks: dir.join("superblocks.bin"),
            select: std::array::from_fn(|symbol| dir.join(format!("select-{symbol}.bin"))),
        }
    }

    fn level_bytes(files: &LevelFiles) -> Vec<Vec<u8>> {
        [&files.data, &files.superblocks]
            .into_iter()
            .chain(&files.select)
            .map(|path| std::fs::read(path).unwrap())
            .collect()
    }

    fn digit_stream(rows: usize, kind: &str, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        (0..rows)
            .map(|row| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let draw = (state >> 20) % 100;
                match kind {
                    "uniform" => (state >> 33) as u8 & 3,
                    "skewed" => match draw {
                        0..=93 => 0,
                        94..=96 => 1,
                        97 => 2,
                        _ => 3,
                    },
                    "runs" => ((row / 97) % 4) as u8,
                    _ => 2,
                }
            })
            .collect()
    }

    fn assert_batch_matches_scalar(name: &str, digits: &[u8], chunks: &[usize]) {
        let dir = test_dir(name);
        let buffers = DirectBuffers::plan(min_buffer_bytes(), DIRECT_STREAMS).unwrap();
        let scalar_files = level_files(&dir.join("scalar"));
        let batch_files = level_files(&dir.join("batch"));

        let mut scalar = LevelEncoder::new(&scalar_files, buffers).unwrap();
        for (row, &digit) in digits.iter().enumerate() {
            scalar.push(digit, row as u64).unwrap();
        }
        let scalar = scalar.finish(digits.len() as u64, &scalar_files).unwrap();

        let mut batch = LevelEncoder::new(&batch_files, buffers).unwrap();
        let mut row = 0usize;
        let mut turn = 0usize;
        while row < digits.len() {
            let take = chunks[turn % chunks.len()].min(digits.len() - row);
            batch
                .push_block(&digits[row..row + take], row as u64)
                .unwrap();
            row += take;
            turn += 1;
        }
        let batch = batch.finish(digits.len() as u64, &batch_files).unwrap();

        assert_eq!(
            level_bytes(&scalar_files),
            level_bytes(&batch_files),
            "{name}"
        );
        assert_eq!(scalar.n_datalines, batch.n_datalines, "{name}");
        assert_eq!(scalar.position_bits, batch.position_bits, "{name}");
        assert_eq!(scalar.n_superblocks, batch.n_superblocks, "{name}");
        assert_eq!(scalar.n_sel, batch.n_sel, "{name}");
        assert_eq!(scalar.n_occs_smaller, batch.n_occs_smaller, "{name}");
        assert_eq!(scalar.aux_bytes, batch.aux_bytes, "{name}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn level_encoder_batches_match_scalar_across_boundaries() {
        let lengths = [
            0usize, 1, 63, 64, 65, 255, 256, 257, 511, 512, 513, 2_047, 2_048, 2_049, 8_191, 8_192,
            8_193, 16_385, 70_001,
        ];
        for kind in ["uniform", "skewed", "runs", "constant"] {
            for &rows in &lengths {
                let digits = digit_stream(rows, kind, 0x9e37_79b9 + rows as u64);
                assert_batch_matches_scalar(
                    &format!("whole-{kind}-{rows}"),
                    &digits,
                    &[usize::MAX],
                );
            }
        }
    }

    #[test]
    fn level_encoder_batches_match_scalar_for_misaligned_chunks() {
        for kind in ["uniform", "skewed"] {
            for (index, chunks) in [
                vec![1usize],
                vec![7, 64, 3],
                vec![63, 65],
                vec![255, 1, 257],
                vec![600],
                vec![2_048, 1],
                vec![8_191, 2, 8_193],
            ]
            .iter()
            .enumerate()
            {
                let digits = digit_stream(25_013, kind, 17 + index as u64);
                assert_batch_matches_scalar(&format!("chunks-{kind}-{index}"), &digits, chunks);
            }
        }
    }

    #[test]
    fn block_scratch_stays_within_the_stream_buffer() {
        for stream in [1usize, 5, 6, 3 * 1_024, 24 * 1_024, 1 << 20, 37_000_001] {
            for lanes in [1usize, 2] {
                let blocks = BlockScratch::new(stream, lanes);
                let bytes = blocks.symbols.len() + blocks.digits.len() + blocks.buckets.len();
                assert!(blocks.rows() >= 1);
                assert_eq!(blocks.symbols.len(), blocks.rows() * 4);
                if stream >= 6 {
                    assert!(bytes <= stream, "{bytes} > {stream} with {lanes} lanes");
                }
            }
        }
    }

    #[test]
    fn direct_qwtb_matches_owned_when_source_is_not_a_block_multiple() {
        // The minimum budget stages 614 rows per block.
        let block_rows = BlockScratch::new(
            DirectBuffers::plan(min_buffer_bytes(), DIRECT_STREAMS)
                .unwrap()
                .stream,
            1,
        )
        .rows();
        for rows in [
            1,
            block_rows - 1,
            block_rows,
            block_rows + 1,
            2 * block_rows - 1,
            2 * block_rows + 1,
            7 * block_rows + 13,
        ] {
            let values: Vec<u32> = (0..rows)
                .map(|row| ((row * 29 + row / 5) % 3_001) as u32)
                .collect();
            assert_matches_owned(
                &format!("partial-block-{rows}"),
                &values,
                min_buffer_bytes(),
            );
            assert_matches_owned(&format!("partial-block-24k-{rows}"), &values, 24 * 1_024);
        }
    }

    #[test]
    fn direct_qwtb_rejects_symbols_above_sigma_at_any_block_position() {
        let block_rows = BlockScratch::new(
            DirectBuffers::plan(min_buffer_bytes(), DIRECT_STREAMS)
                .unwrap()
                .stream,
            1,
        )
        .rows();
        let rows = 3 * block_rows + 5;
        for bad_row in [
            0,
            block_rows / 2,
            block_rows - 1,
            block_rows,
            2 * block_rows - 1,
            rows - 1,
        ] {
            let mut values: Vec<u32> = (0..rows).map(|row| (row % 400) as u32).collect();
            values[rows / 2] = 499;
            values[bad_row] = 500;
            let dir = test_dir("sigma-error");
            let source = dir.join("source.u32");
            let output = dir.join("output.qwtb");
            let scratch = dir.join("scratch");
            write_source(&source, &values);
            let error = write_qwt256_u32_direct(
                &source,
                rows as u64,
                499,
                &output,
                &scratch,
                budget(rows as u64),
            )
            .unwrap_err();
            assert!(
                matches!(
                    error,
                    DirectQwtBuildError::InvalidInput("source symbol exceeds declared sigma")
                ),
                "row {bad_row}: {error}"
            );
            assert!(!output.exists());
            assert!(std::fs::read_dir(&scratch).unwrap().next().is_none());
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn direct_qwtb_rejects_a_sigma_that_is_not_the_maximum() {
        let values: Vec<u32> = (0..5_000).map(|row| (row % 300) as u32).collect();
        let dir = test_dir("sigma-not-max");
        let source = dir.join("source.u32");
        write_source(&source, &values);
        let error = write_qwt256_u32_direct(
            &source,
            values.len() as u64,
            400,
            &dir.join("output.qwtb"),
            &dir.join("scratch"),
            budget(values.len() as u64),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            DirectQwtBuildError::InvalidInput("declared sigma is not the maximum source symbol")
        ));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
