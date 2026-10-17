//! Times the file-backed direct builders (`write_qwt256_u32_direct` and
//! `write_hqwt256_u32_direct`) over a synthetic `u32` source file.
//!
//! ```text
//! cargo run --release --example perf_direct_builders -- \
//!     [--rows N] [--sigma S] [--huff-alphabet A] [--runs R] [--buffer-bytes B] [--dir PATH]
//! ```
//!
//! The plain input is uniform over `[0, sigma]`; the Huffman input is skewed
//! (roughly geometric) over `[0, huff-alphabet)`. Both are generated from a
//! fixed seed, so every run and every build of the crate sees the same bytes.
//! Each line reports min/median/max wall time over `runs` builds and an
//! FNV-1a hash of the output, which must be identical across code versions
//! that claim byte-identical output.

use qwt::bytes::{write_hqwt256_u32_direct, write_qwt256_u32_direct, DirectQwtBuildBudget};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    rows: u64,
    sigma: u32,
    huff_alphabet: u32,
    runs: usize,
    buffer_bytes: u64,
    dir: PathBuf,
}

fn parse_args() -> Args {
    let mut args = Args {
        rows: 20_000_000,
        sigma: 4_999_999,
        huff_alphabet: 4_096,
        runs: 5,
        buffer_bytes: 32 << 20,
        dir: std::env::temp_dir().join("qwt-perf-direct-builders"),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .unwrap_or_else(|| panic!("missing value for {flag}"));
        match flag.as_str() {
            "--rows" => args.rows = value.parse().unwrap(),
            "--sigma" => args.sigma = value.parse().unwrap(),
            "--huff-alphabet" => args.huff_alphabet = value.parse().unwrap(),
            "--runs" => args.runs = value.parse().unwrap(),
            "--buffer-bytes" => args.buffer_bytes = value.parse().unwrap(),
            "--dir" => args.dir = PathBuf::from(value),
            _ => panic!("unknown flag {flag}"),
        }
    }
    assert!(args.rows > 0 && args.runs > 0 && args.huff_alphabet > 1);
    args
}

/// SplitMix64: small, fast, and reproducible without extra dependencies.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Writes `rows` little-endian `u32`s produced by `symbol` and returns the
/// largest value written.
fn write_source(path: &Path, rows: u64, mut symbol: impl FnMut(u64) -> u32) -> u32 {
    let mut out = BufWriter::with_capacity(1 << 20, File::create(path).unwrap());
    let mut max = 0;
    for row in 0..rows {
        let value = symbol(row);
        max = max.max(value);
        out.write_all(&value.to_le_bytes()).unwrap();
    }
    out.flush().unwrap();
    max
}

fn fnv1a(path: &Path) -> u64 {
    let mut file = File::open(path).unwrap();
    let mut buf = vec![0u8; 1 << 20];
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    loop {
        let n = file.read(&mut buf).unwrap();
        if n == 0 {
            return hash;
        }
        for &byte in &buf[..n] {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
}

fn time_builder(
    name: &str,
    args: &Args,
    source: &Path,
    build: impl Fn(&Path, &Path, &Path, DirectQwtBuildBudget) -> (u16, u64, u64),
) {
    let output = args.dir.join(format!("{name}.out"));
    let scratch = args.dir.join(format!("{name}.scratch"));
    let budget = DirectQwtBuildBudget {
        buffer_bytes: args.buffer_bytes,
        scratch_bytes: u64::MAX,
    };
    let mut times = Vec::with_capacity(args.runs);
    let mut summary = (0, 0, 0);
    for _ in 0..args.runs {
        let _ = fs::remove_file(&output);
        let _ = fs::remove_dir_all(&scratch);
        let start = Instant::now();
        summary = build(source, &output, &scratch, budget);
        times.push(start.elapsed().as_secs_f64());
    }
    times.sort_by(f64::total_cmp);
    let (levels, output_bytes, peak_scratch_bytes) = summary;
    println!(
        "{name}: [rows: {}, levels: {levels}, runs: {}, min_s: {:.3}, median_s: {:.3}, max_s: {:.3}, output_bytes: {output_bytes}, peak_scratch_bytes: {peak_scratch_bytes}, output_fnv1a: {:016x}]",
        args.rows,
        args.runs,
        times[0],
        times[times.len() / 2],
        times[times.len() - 1],
        fnv1a(&output),
    );
    let _ = fs::remove_file(&output);
    let _ = fs::remove_dir_all(&scratch);
}

fn main() {
    let args = parse_args();
    fs::create_dir_all(&args.dir).unwrap();

    let plain_source = args.dir.join("plain.u32");
    let mut rng = SplitMix64(0x5EED_0001);
    let span = u64::from(args.sigma) + 1;
    let sigma = write_source(&plain_source, args.rows, |_| (rng.next() % span) as u32);
    time_builder(
        "qwt256_direct",
        &args,
        &plain_source,
        |src, out, scratch, budget| {
            let stats =
                write_qwt256_u32_direct(src, args.rows, sigma, out, scratch, budget).unwrap();
            (stats.levels, stats.output_bytes, stats.peak_scratch_bytes)
        },
    );
    fs::remove_file(&plain_source).unwrap();

    let huff_source = args.dir.join("huff.u32");
    let mut rng = SplitMix64(0x5EED_0002);
    let alphabet = args.huff_alphabet;
    // Every symbol appears at least once (the builder expects a dense
    // alphabet); the rest are geometric, so code lengths vary widely.
    write_source(&huff_source, args.rows, |row| {
        if row < u64::from(alphabet) {
            row as u32
        } else {
            (rng.next().trailing_zeros() * 97 + (rng.next() % 97) as u32) % alphabet
        }
    });
    time_builder(
        "hqwt256_direct",
        &args,
        &huff_source,
        |src, out, scratch, budget| {
            let stats =
                write_hqwt256_u32_direct(src, args.rows, alphabet, out, scratch, budget).unwrap();
            (stats.levels, stats.output_bytes, stats.peak_scratch_bytes)
        },
    );
    fs::remove_file(&huff_source).unwrap();
}
