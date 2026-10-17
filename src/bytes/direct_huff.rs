//! Budget-aware file-backed construction for canonical `HQWB` sections.
//!
//! Input symbols are already locally densified to `[0, alphabet_size)`. The
//! builder keeps only schema-sized Huffman metadata on the heap, emits one
//! rank/select level at a time, and stable-partitions the row stream through
//! five bounded scratch files (four code digits plus completed leaves).

use super::direct::{
    align_output, append_file, container_output_upper_bound, file_len, remove_level_files,
    validate_row_capacity, write_bucket_runs, BlockScratch, DirectBuffers, DirectQwtBuildBudget,
    DirectQwtBuildError, DirectQwtBuildStats, DirectWorkspace, LevelEncoder, LevelFiles,
    OutputGuard, SourceSet, DIRECT_BLOCK_ROWS, DIRECT_SELECT_SAMPLE_ROWS, DIRECT_SUPERBLOCK_ROWS,
};
use super::hqwb::LevelDir;
use super::{align_up, ensure_le, FORMAT_VERSION, HEADER_SIZE, HQWB_MAGIC, HQWT_LEVEL_DIR_SIZE};
use crate::quadwt::huffqwt::{huffman_codes_from_frequencies, PrefixCode};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const DIRECT_HUFF_STREAMS: u64 = 7;
const HUFF_METADATA_BYTES_PER_SYMBOL: u64 = 128;
const HUFF_METADATA_FIXED_BYTES: u64 = 4 * 1_024;

/// Minimum heap-buffer allowance required by the direct Huffman QWT writer
/// for an alphabet of `alphabet_size` symbols.
pub fn hqwt256_u32_min_buffer_bytes(alphabet_size: u32) -> Result<u64, DirectQwtBuildError> {
    let metadata_bytes = u64::from(alphabet_size)
        .checked_mul(HUFF_METADATA_BYTES_PER_SYMBOL)
        .and_then(|bytes| bytes.checked_add(HUFF_METADATA_FIXED_BYTES))
        .ok_or(DirectQwtBuildError::Overflow(
            "Huffman metadata buffer bytes",
        ))?;
    let stream_bytes = DirectBuffers::min_bytes(DIRECT_HUFF_STREAMS)?;
    metadata_bytes
        .checked_add(stream_bytes)
        .ok_or(DirectQwtBuildError::Overflow(
            "minimum Huffman buffer bytes",
        ))
}

/// Conservative byte ceiling for the final canonical HQWB output.
///
/// This deliberately assumes the maximum 16 quad levels permitted by the
/// u32 Huffman code representation. It lets an outer staged-container budget
/// account for HQWB while the direct writer still treats its destination as
/// final output.
pub fn hqwt256_u32_output_upper_bound(
    rows: u64,
    alphabet_size: u32,
) -> Result<u64, DirectQwtBuildError> {
    validate_row_capacity(rows, "HQWB")?;
    if (rows == 0) != (alphabet_size == 0) {
        return Err(DirectQwtBuildError::InvalidInput(
            "empty input and zero alphabet size must agree",
        ));
    }
    if alphabet_size > u32::from(u16::MAX) {
        return Err(DirectQwtBuildError::InvalidInput(
            "local Huffman alphabet exceeds HQWB u16 code-table capacity",
        ));
    }
    let levels = if rows == 0 { 0 } else { 16 };
    // encode: 8 bytes/symbol; decode: 12 bytes/symbol plus at most 33
    // bucket lengths; 7 bytes cover the directory-to-encode alignment.
    let metadata = u64::from(alphabet_size)
        .checked_mul(20)
        .and_then(|bytes| bytes.checked_add(33 * 4 + 7))
        .ok_or(DirectQwtBuildError::Overflow("HQWB output metadata bytes"))?;
    container_output_upper_bound(rows, levels, HQWT_LEVEL_DIR_SIZE as u64, metadata)
}

/// Builds one canonical HQWT256/u32 `HQWB` section directly from a replay file.
///
/// `source` must contain exactly `rows` little-endian local `u32` values. For a
/// non-empty source every value in `[0, alphabet_size)` must occur at least
/// once; this is the contract produced by column-local densification. The
/// output is installed only after sync, and all intermediate files are removed
/// on success or failure.
pub fn write_hqwt256_u32_direct(
    source: &Path,
    rows: u64,
    alphabet_size: u32,
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
    validate_row_capacity(rows, "HQWB")?;
    if (rows == 0) != (alphabet_size == 0) {
        return Err(DirectQwtBuildError::InvalidInput(
            "empty input and zero alphabet size must agree",
        ));
    }
    if alphabet_size > u32::from(u16::MAX) {
        return Err(DirectQwtBuildError::InvalidInput(
            "local Huffman alphabet exceeds HQWB u16 code-table capacity",
        ));
    }
    let source_bytes = rows
        .checked_mul(4)
        .ok_or(DirectQwtBuildError::Overflow("source bytes"))?;
    if file_len(source)? != source_bytes {
        return Err(DirectQwtBuildError::InvalidInput(
            "source length does not equal rows * 4",
        ));
    }

    let metadata_bytes = u64::from(alphabet_size)
        .checked_mul(HUFF_METADATA_BYTES_PER_SYMBOL)
        .and_then(|bytes| bytes.checked_add(HUFF_METADATA_FIXED_BYTES))
        .ok_or(DirectQwtBuildError::Overflow(
            "Huffman metadata buffer bytes",
        ))?;
    let required_buffers = hqwt256_u32_min_buffer_bytes(alphabet_size)?;
    if budget.buffer_bytes < required_buffers {
        return Err(DirectQwtBuildError::BufferBudget {
            required: required_buffers,
            limit: budget.buffer_bytes,
        });
    }
    let stream_budget = budget.buffer_bytes - metadata_bytes;
    let buffers = DirectBuffers::plan(stream_budget, DIRECT_HUFF_STREAMS)?;
    let per_stream_buffer = buffers.stream;

    let alphabet_len = usize::try_from(alphabet_size)
        .map_err(|_| DirectQwtBuildError::Overflow("Huffman alphabet size"))?;
    let mut frequencies = vec![0usize; alphabet_len];
    let mut blocks = BlockScratch::new(per_stream_buffer, 2);
    if rows != 0 {
        let mut file = File::open(source)?;
        let mut remaining = rows;
        while remaining > 0 {
            let block = remaining.min(blocks.rows() as u64) as usize;
            let symbols = &mut blocks.symbols[..block * 4];
            file.read_exact(symbols)?;
            remaining -= block as u64;
            for encoded in symbols.as_chunks::<4>().0 {
                let symbol = u32::from_le_bytes(*encoded);
                let frequency = frequencies.get_mut(symbol as usize).ok_or(
                    DirectQwtBuildError::InvalidInput("source symbol exceeds local alphabet"),
                )?;
                *frequency = frequency
                    .checked_add(1)
                    .ok_or(DirectQwtBuildError::Overflow("Huffman symbol frequency"))?;
            }
        }
        let mut trailing = [0u8; 1];
        if file.read(&mut trailing)? != 0 {
            return Err(DirectQwtBuildError::InvalidInput(
                "source contains trailing rows",
            ));
        }
        if frequencies.contains(&0) {
            return Err(DirectQwtBuildError::InvalidInput(
                "local Huffman alphabet contains an unused symbol",
            ));
        }
    }

    let codes = if rows == 0 {
        Vec::new()
    } else {
        huffman_codes_from_frequencies(&frequencies)
    };
    let max_code_len = codes.iter().map(|code| code.len).max().unwrap_or(0);
    if !max_code_len.is_multiple_of(2) {
        return Err(DirectQwtBuildError::InvalidInput(
            "Huffman code length is not a multiple of two",
        ));
    }
    if max_code_len > u32::BITS {
        return Err(DirectQwtBuildError::InvalidInput(
            "Huffman code exceeds the HQWB u32 representation",
        ));
    }
    let levels = u16::try_from(max_code_len / 2)
        .map_err(|_| DirectQwtBuildError::Overflow("HQWB level count"))?;
    let mut decode = if rows == 0 {
        Vec::new()
    } else {
        vec![Vec::<(u32, u32)>::new(); max_code_len as usize + 1]
    };
    for (symbol, code) in codes.iter().enumerate() {
        if code.len != 0 {
            decode[code.len as usize].push((code.content, symbol as u32));
        }
    }
    for bucket in &mut decode {
        bucket.sort_unstable_by_key(|(content, _)| *content);
    }
    let decode_buckets = u16::try_from(decode.len())
        .map_err(|_| DirectQwtBuildError::Overflow("HQWB decode bucket count"))?;

    let scratch_required = scratch_upper_bound(rows, levels)?;
    if scratch_required > budget.scratch_bytes {
        return Err(DirectQwtBuildError::ScratchBudget {
            required: scratch_required,
            limit: budget.scratch_bytes,
        });
    }

    let (workspace, build_id) = DirectWorkspace::create(scratch)?;
    let mut tmp_os = output.as_os_str().to_os_string();
    tmp_os.push(format!(
        ".hqwt-direct-{}-{build_id}.tmp",
        std::process::id()
    ));
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
        .checked_mul(HQWT_LEVEL_DIR_SIZE)
        .and_then(|bytes| HEADER_SIZE.checked_add(bytes))
        .ok_or(DirectQwtBuildError::Overflow("HQWB directory bytes"))?;
    let off_encode = align_up(directory_bytes, 8);
    let encode_bytes = codes
        .len()
        .checked_mul(8)
        .ok_or(DirectQwtBuildError::Overflow("HQWB encode-table bytes"))?;
    let off_decode = off_encode
        .checked_add(encode_bytes)
        .ok_or(DirectQwtBuildError::Overflow("HQWB decode-table offset"))?;
    let decode_bytes = decode.iter().try_fold(0usize, |total, bucket| {
        bucket
            .len()
            .checked_mul(12)
            .and_then(|bytes| bytes.checked_add(4))
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or(DirectQwtBuildError::Overflow("HQWB decode-table bytes"))
    })?;
    let code_table_end = off_decode
        .checked_add(decode_bytes)
        .ok_or(DirectQwtBuildError::Overflow("HQWB code-table end"))?;
    out.set_len(
        u64::try_from(code_table_end)
            .map_err(|_| DirectQwtBuildError::Overflow("HQWB code-table end"))?,
    )?;
    write_code_tables(&mut out, off_encode, off_decode, &codes, &decode)?;

    let mut cursor = code_table_end as u64;
    let mut dirs = Vec::with_capacity(usize::from(levels));
    let mut current = SourceSet {
        paths: vec![source.to_path_buf()],
        owned: false,
    };
    let copy_buffer_len = usize::try_from(stream_budget.min(1024 * 1024))
        .map_err(|_| DirectQwtBuildError::Overflow("copy buffer bytes"))?;
    let mut peak_scratch = if levels == 0 { 0 } else { source_bytes };

    for level in 0..levels {
        let shift = 2 * (u32::from(level) + 1);
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
        let bucket_paths: [PathBuf; 5] = std::array::from_fn(|bucket| {
            workspace
                .path
                .join(format!("level-{level}-bucket-{bucket}.u32"))
        });
        let mut buckets = if needs_partition {
            Some([
                BufWriter::with_capacity(per_stream_buffer, File::create(&bucket_paths[0])?),
                BufWriter::with_capacity(per_stream_buffer, File::create(&bucket_paths[1])?),
                BufWriter::with_capacity(per_stream_buffer, File::create(&bucket_paths[2])?),
                BufWriter::with_capacity(per_stream_buffer, File::create(&bucket_paths[3])?),
                BufWriter::with_capacity(per_stream_buffer, File::create(&bucket_paths[4])?),
            ])
        } else {
            None
        };
        let mut seen = 0u64;
        let mut level_rows = 0u64;
        for path in &current.paths {
            let mut file = File::open(path)?;
            let path_bytes = file.metadata()?.len();
            if !path_bytes.is_multiple_of(4) {
                return Err(DirectQwtBuildError::InvalidInput(
                    "partition source contains a partial symbol",
                ));
            }
            let mut remaining = path_bytes / 4;
            while remaining > 0 {
                let block = remaining.min(blocks.rows() as u64) as usize;
                let symbols = &mut blocks.symbols[..block * 4];
                file.read_exact(symbols)?;
                remaining -= block as u64;
                let ids = &mut blocks.buckets[..block];
                let digits = &mut blocks.digits[..block];
                let mut staged = 0;
                for (id, encoded) in ids.iter_mut().zip(symbols.as_chunks::<4>().0) {
                    let symbol = u32::from_le_bytes(*encoded);
                    let code =
                        codes
                            .get(symbol as usize)
                            .ok_or(DirectQwtBuildError::InvalidInput(
                                "partition symbol exceeds local alphabet",
                            ))?;
                    let digit = if code.len >= shift {
                        ((code.content >> (code.len - shift)) & 3) as u8
                    } else {
                        0
                    };
                    digits[staged] = digit;
                    staged += usize::from(code.len >= shift);
                    *id = if code.len <= shift { 4 } else { digit };
                }
                encoder.push_block(&digits[..staged], level_rows)?;
                level_rows += staged as u64;
                if let Some(bucket_writers) = buckets.as_mut() {
                    write_bucket_runs(bucket_writers, ids, symbols)?;
                }
                seen += block as u64;
            }
        }
        if seen != rows {
            return Err(DirectQwtBuildError::InvalidInput(
                "partitioned source row count changed between levels",
            ));
        }
        if let Some(bucket_writers) = buckets.as_mut() {
            for writer in bucket_writers {
                writer.flush()?;
            }
        }
        drop(buckets);

        let summary = encoder.finish(level_rows, &level_files)?;
        let partition_bytes = if needs_partition { source_bytes } else { 0 };
        let level_peak = source_bytes
            .checked_add(partition_bytes)
            .and_then(|bytes| bytes.checked_add(summary.aux_bytes))
            .ok_or(DirectQwtBuildError::Overflow("observed HQWB scratch bytes"))?;
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
            level_len: level_rows,
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

    write_header_and_directory(
        &mut out,
        rows,
        levels,
        alphabet_size as u16,
        decode_buckets,
        &dirs,
    )?;
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

fn scratch_upper_bound(rows: u64, levels: u16) -> Result<u64, DirectQwtBuildError> {
    if levels == 0 {
        return Ok(0);
    }
    let source = rows
        .checked_mul(4)
        .ok_or(DirectQwtBuildError::Overflow("HQWB scratch source bytes"))?;
    let partition = if levels > 1 { source } else { 0 };
    let data =
        rows.div_ceil(DIRECT_BLOCK_ROWS)
            .checked_mul(64)
            .ok_or(DirectQwtBuildError::Overflow(
                "HQWB scratch data-line bytes",
            ))?;
    let superblocks =
        rows.checked_add(DIRECT_SUPERBLOCK_ROWS)
            .ok_or(DirectQwtBuildError::Overflow(
                "HQWB scratch superblock rows",
            ))?
            / DIRECT_SUPERBLOCK_ROWS;
    let superblocks = superblocks
        .checked_mul(64)
        .ok_or(DirectQwtBuildError::Overflow(
            "HQWB scratch superblock bytes",
        ))?;
    let samples = rows
        .div_ceil(DIRECT_SELECT_SAMPLE_ROWS)
        .checked_add(8)
        .and_then(|entries| entries.checked_mul(4))
        .ok_or(DirectQwtBuildError::Overflow("HQWB scratch select bytes"))?;
    source
        .checked_add(partition)
        .and_then(|bytes| bytes.checked_add(data))
        .and_then(|bytes| bytes.checked_add(superblocks))
        .and_then(|bytes| bytes.checked_add(samples))
        .ok_or(DirectQwtBuildError::Overflow("HQWB scratch upper bound"))
}

fn write_code_tables(
    out: &mut File,
    off_encode: usize,
    off_decode: usize,
    encode: &[PrefixCode],
    decode: &[Vec<(u32, u32)>],
) -> Result<(), DirectQwtBuildError> {
    out.seek(SeekFrom::Start(off_encode as u64))?;
    for code in encode {
        out.write_all(&code.content.to_le_bytes())?;
        out.write_all(&code.len.to_le_bytes())?;
    }
    out.seek(SeekFrom::Start(off_decode as u64))?;
    for bucket in decode {
        let entries = u32::try_from(bucket.len())
            .map_err(|_| DirectQwtBuildError::Overflow("HQWB decode bucket entries"))?;
        out.write_all(&entries.to_le_bytes())?;
        for &(content, symbol) in bucket {
            out.write_all(&content.to_le_bytes())?;
            out.write_all(&u64::from(symbol).to_le_bytes())?;
        }
    }
    Ok(())
}

fn write_header_and_directory(
    out: &mut File,
    rows: u64,
    levels: u16,
    encode_len: u16,
    decode_buckets: u16,
    dirs: &[LevelDir],
) -> Result<(), DirectQwtBuildError> {
    let directory_bytes = usize::from(levels)
        .checked_mul(HQWT_LEVEL_DIR_SIZE)
        .and_then(|bytes| HEADER_SIZE.checked_add(bytes))
        .ok_or(DirectQwtBuildError::Overflow("HQWB header directory bytes"))?;
    let mut header = vec![0u8; directory_bytes];
    header[0..4].copy_from_slice(HQWB_MAGIC);
    header[4..6].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    header[6] = 0;
    header[7] = 4;
    header[8..16].copy_from_slice(&rows.to_le_bytes());
    header[16..18].copy_from_slice(&levels.to_le_bytes());
    header[18..20].copy_from_slice(&encode_len.to_le_bytes());
    header[20..22].copy_from_slice(&decode_buckets.to_le_bytes());
    for (level, dir) in dirs.iter().enumerate() {
        let start = HEADER_SIZE + level * HQWT_LEVEL_DIR_SIZE;
        dir.write(&mut header[start..start + HQWT_LEVEL_DIR_SIZE]);
    }
    out.seek(SeekFrom::Start(0))?;
    out.write_all(&header)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytes::{hqwt256_to_bytes, AlignedBytes, HqwtView};
    use crate::{AccessUnsigned, RankUnsigned, SelectUnsigned, HQWT256};
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_ID: AtomicU64 = AtomicU64::new(0);

    fn test_dir(name: &str) -> PathBuf {
        let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "hqwt-direct-test-{name}-{}-{id}",
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

    fn budget(rows: u64, alphabet_size: u32) -> DirectQwtBuildBudget {
        DirectQwtBuildBudget {
            buffer_bytes: u64::from(alphabet_size)
                .saturating_mul(HUFF_METADATA_BYTES_PER_SYMBOL)
                .saturating_add(1 << 20),
            scratch_bytes: rows.saturating_mul(16).saturating_add(1 << 20),
        }
    }

    fn assert_exact(name: &str, values: Vec<u32>) {
        let alphabet_size = values.iter().copied().max().map_or(0, |value| value + 1);
        assert_exact_with_buffer(name, values, budget(0, alphabet_size).buffer_bytes);
    }

    fn assert_exact_with_buffer(name: &str, values: Vec<u32>, buffer_bytes: u64) {
        let root = test_dir(name);
        let source = root.join("source.u32");
        let output = root.join("output.hqwb");
        let scratch = root.join("scratch");
        write_source(&source, &values);
        let alphabet_size = values.iter().copied().max().map_or(0, |value| value + 1);
        let budget = |rows, _| DirectQwtBuildBudget {
            buffer_bytes,
            ..budget(rows, alphabet_size)
        };
        let expected = hqwt256_to_bytes(&HQWT256::from(values.clone())).unwrap();
        let stats = write_hqwt256_u32_direct(
            &source,
            values.len() as u64,
            alphabet_size,
            &output,
            &scratch,
            budget(values.len() as u64, alphabet_size),
        )
        .unwrap();
        let actual = std::fs::read(&output).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(stats.output_bytes, actual.len() as u64);
        assert!(
            stats.output_bytes
                <= hqwt256_u32_output_upper_bound(values.len() as u64, alphabet_size).unwrap()
        );
        assert!(
            stats.peak_scratch_bytes <= budget(values.len() as u64, alphabet_size).scratch_bytes
        );
        assert!(std::fs::read_dir(&scratch).unwrap().next().is_none());

        if values.len() > 10_000 {
            std::fs::remove_dir_all(root).unwrap();
            return;
        }

        let aligned = AlignedBytes::from_slice(&actual);
        let view = HqwtView::<u32, 256>::from_bytes(&aligned).unwrap();
        for (position, &value) in values.iter().enumerate() {
            assert_eq!(view.get(position), Some(value));
        }
        for symbol in 0..alphabet_size {
            for position in (0..=values.len()).step_by(37) {
                let expected_rank = values[..position]
                    .iter()
                    .filter(|&&value| value == symbol)
                    .count();
                assert_eq!(view.rank(symbol, position), Some(expected_rank));
            }
            let positions = values
                .iter()
                .enumerate()
                .filter_map(|(position, &value)| (value == symbol).then_some(position));
            for (occurrence, position) in positions.enumerate().take(4) {
                assert_eq!(view.select(symbol, occurrence), Some(position));
            }
        }

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_bytes_empty_and_single_symbol() {
        assert_exact("empty", Vec::new());
        assert_exact("single", vec![0; 8_193]);
    }

    #[test]
    fn exact_bytes_skewed_multilevel_and_support_boundaries() {
        let mut values = (0..8_193)
            .map(|row| match row % 17 {
                0..=9 => 0,
                10..=12 => 1,
                13..=14 => 2,
                15 => 3,
                _ => 4,
            })
            .collect::<Vec<_>>();
        values.extend(0..17);
        assert_exact("skewed", values);
    }

    fn skewed_values(rows: u64, alphabet: u64) -> Vec<u32> {
        let mut values = (0..rows)
            .map(|row| {
                let hash = row.wrapping_mul(2_654_435_761) >> 5;
                match row % 8 {
                    0..=3 => (hash % 3) as u32,
                    4..=5 => (hash % 40) as u32,
                    _ => (hash % alphabet) as u32,
                }
            })
            .collect::<Vec<_>>();
        values.extend(0..alphabet as u32);
        values
    }

    #[test]
    fn exact_bytes_at_minimum_buffer_budget() {
        let values = skewed_values(20_011, 300);
        let minimum = hqwt256_u32_min_buffer_bytes(300).unwrap();
        assert_exact_with_buffer("exact-minimum", values, minimum);
    }

    #[test]
    fn exact_bytes_with_buffered_aux_files_across_many_boundaries() {
        let values = skewed_values(300_007, 1_500);
        let minimum = hqwt256_u32_min_buffer_bytes(1_500).unwrap();
        assert_exact_with_buffer("multi-boundary-min", values.clone(), minimum);
        assert_exact_with_buffer("multi-boundary-large", values, 8 << 20);
    }

    #[test]
    fn rejects_non_dense_local_alphabet_and_cleans_scratch() {
        let root = test_dir("non-dense");
        let source = root.join("source.u32");
        let output = root.join("output.hqwb");
        let scratch = root.join("scratch");
        write_source(&source, &[0, 2, 0, 2]);
        let error =
            write_hqwt256_u32_direct(&source, 4, 3, &output, &scratch, budget(4, 3)).unwrap_err();
        assert!(error.to_string().contains("unused symbol"));
        assert!(!output.exists());
        assert!(!scratch.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn enforces_buffer_and_scratch_budgets() {
        let root = test_dir("budgets");
        let source = root.join("source.u32");
        let output = root.join("output.hqwb");
        let scratch = root.join("scratch");
        let values = (0..10_000).map(|row| (row % 5) as u32).collect::<Vec<_>>();
        write_source(&source, &values);
        let required_buffer = HUFF_METADATA_FIXED_BYTES
            + 5 * HUFF_METADATA_BYTES_PER_SYMBOL
            + DirectBuffers::min_bytes(DIRECT_HUFF_STREAMS).unwrap();
        assert_eq!(hqwt256_u32_min_buffer_bytes(5).unwrap(), required_buffer);
        let buffer_error = write_hqwt256_u32_direct(
            &source,
            values.len() as u64,
            5,
            &output,
            &scratch,
            DirectQwtBuildBudget {
                buffer_bytes: required_buffer - 1,
                scratch_bytes: u64::MAX,
            },
        )
        .unwrap_err();
        assert!(matches!(
            buffer_error,
            DirectQwtBuildError::BufferBudget { required, .. } if required == required_buffer
        ));
        let scratch_error = write_hqwt256_u32_direct(
            &source,
            values.len() as u64,
            5,
            &output,
            &scratch,
            DirectQwtBuildBudget {
                buffer_bytes: required_buffer,
                scratch_bytes: 1,
            },
        )
        .unwrap_err();
        assert!(matches!(
            scratch_error,
            DirectQwtBuildError::ScratchBudget { .. }
        ));
        assert!(!output.exists());
        assert!(!scratch.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_bytes_when_source_is_not_a_block_multiple() {
        // The minimum budget stages 512 rows per block with two byte lanes.
        for rows in [1usize, 511, 512, 513, 1_023, 1_025, 5_000] {
            let mut values = skewed_values(rows as u64, 40);
            values.truncate(rows.max(40));
            let minimum = hqwt256_u32_min_buffer_bytes(40).unwrap();
            assert_exact_with_buffer(&format!("partial-block-{rows}"), values, minimum);
        }
    }

    #[test]
    fn rejects_symbols_outside_the_alphabet_at_any_block_position() {
        let rows = 2_000usize;
        for bad_row in [0usize, 255, 511, 512, 1_023, 1_999] {
            let mut values: Vec<u32> = (0..rows).map(|row| (row % 20) as u32).collect();
            values[bad_row] = 20;
            let root = test_dir("alphabet-error");
            let source = root.join("source.u32");
            let output = root.join("output.hqwb");
            let scratch = root.join("scratch");
            write_source(&source, &values);
            let error = write_hqwt256_u32_direct(
                &source,
                rows as u64,
                20,
                &output,
                &scratch,
                DirectQwtBuildBudget {
                    buffer_bytes: hqwt256_u32_min_buffer_bytes(20).unwrap(),
                    scratch_bytes: 1 << 24,
                },
            )
            .unwrap_err();
            assert!(
                matches!(
                    error,
                    DirectQwtBuildError::InvalidInput("source symbol exceeds local alphabet")
                ),
                "row {bad_row}: {error}"
            );
            assert!(!output.exists());
            std::fs::remove_dir_all(root).unwrap();
        }
    }
}
