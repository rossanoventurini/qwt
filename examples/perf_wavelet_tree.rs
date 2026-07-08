use std::{fs, path::Path};

use clap::Parser;
use mem_dbg::{MemSize, SizeFlags};
use qwt::{
    perf_and_test_utils::{
        gen_queries, gen_range_queries, gen_rank_queries, gen_select_queries, type_of,
        TimingQueries,
    },
    quadwt::RSforWT,
    utils::{msb, text_remap},
    AccessUnsigned, HQWT256Pfs, HQWT512Pfs, HuffQWaveletTree, OccsRangeUnsigned, QWT256Pfs,
    QWT512Pfs, QWaveletTree, RankUnsigned, SelectUnsigned, HQWT256, HQWT512, HWT, QWT256, QWT512,
    WT,
};
use serde::{Deserialize, Serialize};

const N_RUNS: usize = 10;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Spacify the input filename (The Index is saved with the extension ".qwt")
    #[clap(short, long, value_parser)]
    input_file: String,
    /// Specify the number of randomly generated queries to run
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10000000)]
    n_queries: usize,
    /// Check the correctness by running a slow test
    #[arg(short, long)]
    test_correctness: bool,
    /// Run occs_range queries
    #[arg(short, long)]
    occs_range: bool,
    /// Run rank queries
    #[arg(short, long)]
    rank: bool,
    /// Run get queries
    #[arg(short, long)]
    access: bool,
    /// Run select queries
    #[arg(short, long)]
    select: bool,
    #[arg(long)]
    wt: bool,
    #[arg(long)]
    hwt: bool,
    #[arg(long)]
    qwt256: bool,
    #[arg(long)]
    qwt256pfs: bool,
    #[arg(long)]
    qwt512: bool,
    #[arg(long)]
    qwt512pfs: bool,
    #[arg(long)]
    hqwt256: bool,
    #[arg(long)]
    hqwt256pfs: bool,
    #[arg(long)]
    hqwt512: bool,
    #[arg(long)]
    hqwt512pfs: bool,
    #[arg(long)]
    all_structs: bool,
}

pub fn load_or_build_and_save_qwt<DS>(
    output_filename: &str,
    text: &[<DS as AccessUnsigned>::Item],
) -> DS
where
    DS: Serialize
        + for<'a> Deserialize<'a>
        + From<Vec<<DS as AccessUnsigned>::Item>>
        + AccessUnsigned,
    <DS as AccessUnsigned>::Item: Clone,
{
    let ds: DS;
    let path = Path::new(&output_filename);
    if path.exists() {
        println!(
            "The data structure already exists. Filename: {}. I'm going to load it ...",
            output_filename
        );
        let serialized = fs::read(path).unwrap();
        println!("Serialized size: {:?} bytes", serialized.len());
        ds = bincode::deserialize::<DS>(&serialized).unwrap();
    } else {
        let mut t = TimingQueries::new(1, 1); // measure building time
        t.start();
        ds = DS::from(text.to_owned());
        t.stop();
        let (t_min, _, _) = t.get();
        println!("Construction time {:?} millisecs", t_min / 1000000);

        let serialized = bincode::serialize(&ds).unwrap();
        println!("Serialized size: {:?} bytes", serialized.len());
        fs::write(path, serialized).unwrap();
    }

    ds
}

fn test_correctness<
    T: AccessUnsigned<Item = u8> + RankUnsigned<Item = u8> + SelectUnsigned<Item = u8> + MemSize,
>(
    ds: &T,
    sequence: &[u8],
) {
    print!("\nTesting correctness... ");
    for (i, &symbol) in sequence.iter().enumerate() {
        assert_eq!(ds.get(i), Some(symbol));
        let rank = ds.rank(symbol, i).unwrap();
        let s = ds.select(symbol, rank).unwrap();
        assert_eq!(s, i);
    }
    println!("Everything is ok!\n");
}

fn test_occs_range_naive_latency<T: RankUnsigned<Item = u8> + MemSize>(
    ds: &T,
    n: usize,
    queries: &[(usize, usize)],
    file: String,
    sigma: u8,
) {
    let mut t = TimingQueries::new(N_RUNS, queries.len());

    for _ in 0..N_RUNS {
        t.start();
        for &(sp, ep) in queries.iter() {
            let it = (0..sigma).map(|s| {
                let lo = unsafe { ds.rank_unchecked(s, sp) };
                let hi = unsafe { ds.rank_unchecked(s, ep) };
                (s, hi - lo)
            });

            for out in it {
                std::hint::black_box(out);
            }
        }
        t.stop()
    }

    let (t_min, t_max, t_avg) = t.get();
    println!(
    "RESULT algo={} exp=occs_range_naive_latency input={} n={} logn={:?} min_time_ns={} max_time_ns={} avg_time_ns={} space_in_bytes={} space_in_mib={:.2} n_queries={} n_runs={}",
        type_of(&ds).chars().filter(|c| !c.is_whitespace()).collect::<String>(),
    file,
        n,
        msb(n),
        t_min,
        t_max,
        t_avg,
        ds.mem_size(SizeFlags::default()),
        ds.mem_size(SizeFlags::default()) as f64 / (1024.0 * 1024.0),
        queries.len(),
        N_RUNS
    );
}

fn test_occs_range_latency<T: OccsRangeUnsigned<Item = u8> + MemSize>(
    ds: &T,
    n: usize,
    queries: &[(usize, usize)],
    file: String,
) {
    let mut t = TimingQueries::new(N_RUNS, queries.len());

    for _ in 0..N_RUNS {
        t.start();
        for &(sp, ep) in queries.iter() {
            for out in unsafe { ds.occs_range_unchecked(sp..ep) } {
                std::hint::black_box(out);
            }
        }
        t.stop()
    }

    let (t_min, t_max, t_avg) = t.get();
    println!(
    "RESULT algo={} exp=occs_range_latency input={} n={} logn={:?} min_time_ns={} max_time_ns={} avg_time_ns={} space_in_bytes={} space_in_mib={:.2} n_queries={} n_runs={}",
        type_of(&ds).chars().filter(|c| !c.is_whitespace()).collect::<String>(),
    file,
        n,
        msb(n),
        t_min,
        t_max,
        t_avg,
        ds.mem_size(SizeFlags::default()),
        ds.mem_size(SizeFlags::default()) as f64 / (1024.0 * 1024.0),
        queries.len(),
        N_RUNS
    );
}

fn test_rank_latency<T: RankUnsigned<Item = u8> + MemSize>(
    ds: &T,
    n: usize,
    queries: &[(usize, u8)],
    file: String,
) {
    let mut result = 0;
    let mut t = TimingQueries::new(N_RUNS, queries.len());

    for _ in 0..N_RUNS {
        t.start();
        for &(pos, symbol) in queries.iter() {
            let i = (pos + result) % n;
            result = unsafe { ds.rank_unchecked(symbol, i) };
        }
        t.stop()
    }

    let (t_min, t_max, t_avg) = t.get();
    println!(
    "RESULT algo={} exp=rank_latency input={} n={} logn={:?} min_time_ns={} max_time_ns={} avg_time_ns={} space_in_bytes={} space_in_mib={:.2} n_queries={} n_runs={}",
        type_of(&ds).chars().filter(|c| !c.is_whitespace()).collect::<String>(),
    file,
        n,
        msb(n),
        t_min,
        t_max,
        t_avg,
        ds.mem_size(SizeFlags::default()),
        ds.mem_size(SizeFlags::default()) as f64 / (1024.0 * 1024.0),
        queries.len(),
        N_RUNS
    );

    println!("Result: {}", result);
}

fn test_rank_prefetch_latency<RS, const WITH_PREFETCH_SUPPORT: bool>(
    ds: &HuffQWaveletTree<u8, RS, WITH_PREFETCH_SUPPORT>,
    n: usize,
    queries: &[(usize, u8)],
    file: String,
) where
    RS: RSforWT + MemSize,
    Vec<RS>: MemSize,
{
    let mut result = 0;
    let mut t = TimingQueries::new(N_RUNS, queries.len());

    for _ in 0..N_RUNS {
        t.start();
        for &(pos, symbol) in queries.iter() {
            let i = (pos + result) % n;
            result = unsafe { ds.rank_prefetch_unchecked(symbol, i) };
        }
        t.stop()
    }

    let (t_min, t_max, t_avg) = t.get();
    println!(
    "RESULT algo={} exp=rank_prefetch_latency input={} n={} logn={:?} min_time_ns={} max_time_ns={} avg_time_ns={} space_in_bytes={} space_in_mib={:.2} n_queries={} n_runs={}",
        type_of(&ds).chars().filter(|c| !c.is_whitespace()).collect::<String>(),
    file,
        n,
        msb(n),
        t_min,
        t_max,
        t_avg,
        ds.mem_size(SizeFlags::default()),
        ds.mem_size(SizeFlags::default()) as f64 / (1024.0 * 1024.0),
        queries.len(),
        N_RUNS
    );

    println!("Result: {}", result);
}

fn test_rank_prefetch_latency_qwt<RS, const WITH_PREFETCH_SUPPORT: bool>(
    ds: &QWaveletTree<u8, RS, WITH_PREFETCH_SUPPORT>,
    n: usize,
    queries: &[(usize, u8)],
    file: String,
) where
    RS: RSforWT + MemSize,
    Vec<RS>: MemSize,
{
    let mut result = 0;
    let mut t = TimingQueries::new(N_RUNS, queries.len());

    for _ in 0..N_RUNS {
        t.start();
        for &(pos, symbol) in queries.iter() {
            let i = (pos + result) % n;
            result = unsafe { ds.rank_prefetch_unchecked(symbol, i) };
        }
        t.stop()
    }

    let (t_min, t_max, t_avg) = t.get();
    println!(
    "RESULT algo={} exp=rank_prefetch_latency input={} n={} logn={:?} min_time_ns={} max_time_ns={} avg_time_ns={} space_in_bytes={} space_in_mib={:.2} n_queries={} n_runs={}",
        type_of(&ds).chars().filter(|c| !c.is_whitespace()).collect::<String>(),
    file,
        n,
        msb(n),
        t_min,
        t_max,
        t_avg,
        ds.mem_size(SizeFlags::default()),
        ds.mem_size(SizeFlags::default()) as f64 / (1024.0 * 1024.0),
        queries.len(),
        N_RUNS
    );

    println!("Result: {}", result);
}

fn test_select_latency<T: SelectUnsigned<Item = u8> + MemSize>(
    ds: &T,
    n: usize,
    queries: &[(usize, u8)],
    file: String,
) {
    let mut t = TimingQueries::new(N_RUNS, queries.len());

    let mut result = 0;
    for _ in 0..N_RUNS {
        t.start();
        for &(pos, symbol) in queries.iter() {
            let i = pos - 1 + result % 2;
            let i = std::cmp::max(1, i);
            result = unsafe { ds.select_unchecked(symbol, i - 1) };
        }
        t.stop()
    }

    let (t_min, t_max, t_avg) = t.get();
    println!(
	"RESULT algo={} exp=select_latency input={} n={} logn={:?} min_time_ns={} max_time_ns={} avg_time_ns={} space_in_bytes={} space_in_mib={:.2} n_queries={} n_runs={}",
        type_of(&ds).chars().filter(|c| !c.is_whitespace()).collect::<String>(),
	file,
        n,
        msb(n),
        t_min,
        t_max,
        t_avg,
        ds.mem_size(SizeFlags::default()),
        ds.mem_size(SizeFlags::default()) as f64 / (1024.0 * 1024.0),
        queries.len(),
        N_RUNS
    );

    println!("Result: {}", result);
}

fn test_access_latency<T: AccessUnsigned<Item = u8> + MemSize>(
    ds: &T,
    n: usize,
    queries: &[usize],
    file: String,
) {
    let mut t = TimingQueries::new(N_RUNS, queries.len());
    let mut result: u8 = 0;
    for _ in 0..N_RUNS {
        t.start();
        for &pos in queries.iter() {
            let i = (pos * ((result as usize) + 42)) % n;
            result = unsafe { ds.get_unchecked(i) };
        }
        t.stop()
    }

    let (t_min, t_max, t_avg) = t.get();
    println!(
	"RESULT algo={} exp=access_latency input={} n={} logn={:?} min_time_ns={} max_time_ns={} avg_time_ns={} space_in_bytes={} space_in_mib={:.2} n_queries={} n_runs={}",
        type_of(&ds).chars().filter(|c| !c.is_whitespace()).collect::<String>(),
	file,
        n,
        msb(n),
        t_min,
        t_max,
        t_avg,
        ds.mem_size(SizeFlags::default()),
        ds.mem_size(SizeFlags::default()) as f64 / (1024.0 * 1024.0),
        queries.len(),
        N_RUNS
    );
    println!("Result: {result}");
}

#[allow(dead_code)]
fn test_access_throughput<T: AccessUnsigned<Item = u8> + MemSize>(
    ds: &T,
    n: usize,
    queries: &[usize],
    file: String,
) {
    let mut t = TimingQueries::new(N_RUNS, queries.len());
    let mut result: u8 = 0;
    for _ in 0..N_RUNS {
        t.start();
        for &pos in queries.iter() {
            result += unsafe { ds.get_unchecked(pos) };
        }
        t.stop()
    }

    let (t_min, t_max, t_avg) = t.get();
    println!(
	"RESULT algo={} exp=access_throughput input={} n={} logn={:?} min_throughput_ms={} max_throughput_ms={} avg_throughput_ms={} space_in_bytes={} space_in_mib={:.2} n_queries={} n_runs={}",
        type_of(&ds).chars().filter(|c| !c.is_whitespace()).collect::<String>(),
	file,
        n,
        msb(n),
        (queries.len() as f64) / (((t_max * queries.len() as u128) as f64) / 1000.0 / 1000.0),
        (queries.len() as f64) / (((t_min * queries.len() as u128) as f64) / 1000.0 / 1000.0),
        (queries.len() as f64) / (((t_avg * queries.len() as u128) as f64) / 1000.0 / 1000.0),
        ds.mem_size(SizeFlags::default()),
        ds.mem_size(SizeFlags::default()) as f64 / (1024.0 * 1024.0),
        queries.len(),
        N_RUNS
    );
    println!("Result: {result}");
}

#[allow(dead_code)]
fn test_select_throughput<T: SelectUnsigned<Item = u8> + MemSize>(
    ds: &T,
    n: usize,
    queries: &[(usize, u8)],
    file: String,
) {
    let mut t = TimingQueries::new(N_RUNS, queries.len());

    let mut result = 0;
    for _ in 0..N_RUNS {
        t.start();
        for &(pos, symbol) in queries.iter() {
            result += unsafe { ds.select_unchecked(symbol, pos - 1) };
        }
        t.stop()
    }

    let (t_min, t_max, t_avg) = t.get();
    println!(
	"RESULT algo={} exp=select_throughput input={} n={} logn={:?} min_throughput_ms={} max_throughput_ms={} avg_throughput_ms={} space_in_bytes={} space_in_mib={:.2} n_queries={} n_runs={}",
        type_of(&ds).chars().filter(|c| !c.is_whitespace()).collect::<String>(),
	file,
        n,
        msb(n),
        (queries.len() as f64) / (((t_max * queries.len() as u128) as f64) / 1000.0 / 1000.0),
        (queries.len() as f64) / (((t_min * queries.len() as u128) as f64) / 1000.0 / 1000.0),
        (queries.len() as f64) / (((t_avg * queries.len() as u128) as f64) / 1000.0 / 1000.0),
        ds.mem_size(SizeFlags::default()),
        ds.mem_size(SizeFlags::default()) as f64 / (1024.0 * 1024.0),
        queries.len(),
        N_RUNS
    );

    println!("Result: {}", result);
}

fn main() {
    let args = Args::parse();
    let input_filename = args.input_file;
    let mut text = std::fs::read(&input_filename).expect("Cannot read the input file.");

    let sigma = text_remap(&mut text);
    println!("sigma: {}", sigma);
    let n = text.len();
    println!("Text length: {:?}", n);

    // Generate queries
    let range_queries = gen_range_queries(args.n_queries, n);
    let rank_queries = gen_rank_queries(args.n_queries, &text);
    let access_queries = gen_queries(args.n_queries, n);
    let select_queries = gen_select_queries(args.n_queries, &text);

    macro_rules! test_ds {
        ($($t: ty: $e:ident: $rank_f:ident), *) => {
            $({
                if args.$e || args.all_structs {
                    let output_filename = input_filename.clone() + "." + stringify!($e) + ".wt";
                    let ds = load_or_build_and_save_qwt::<$t>(&output_filename, &text);

                    if args.test_correctness {
                        test_correctness(&ds, &text);
                    }

                    if args.occs_range {
                        test_occs_range_latency(&ds, n, &range_queries, input_filename.clone());
                        test_occs_range_naive_latency(&ds, n, &range_queries, input_filename.clone(), sigma as u8);
                    }

                    if args.rank {
                        $rank_f(&ds, n, &rank_queries, input_filename.clone());
                        // test_rank_throughput(&ds, n, &rank_queries, input_filename.clone());
                    }

                    if args.access {
                        test_access_latency(&ds, n, &access_queries, input_filename.clone());
                        // test_access_throughput(&ds, n, &access_queries, input_filename.clone());
                    }

                    if args.select {
                        test_select_latency(&ds, n, &select_queries, input_filename.clone());
                        // test_select_throughput(&ds, n, &select_queries, input_filename.clone());
                    }
                }
            })*
        };
    }

    test_ds!(
        WT<_>: wt: test_rank_latency,
        HWT<_>: hwt: test_rank_latency,
        QWT256<_>: qwt256: test_rank_latency,
        QWT512<_>: qwt512: test_rank_latency,
        HQWT256<_>: hqwt256: test_rank_latency,
        HQWT512<_>: hqwt512: test_rank_latency,
        QWT256Pfs<_>: qwt256pfs: test_rank_prefetch_latency_qwt,
        QWT512Pfs<_>: qwt512pfs: test_rank_prefetch_latency_qwt,
        HQWT256Pfs<_>: hqwt256pfs: test_rank_prefetch_latency,
        HQWT512Pfs<_>: hqwt512pfs: test_rank_prefetch_latency
    );
}
