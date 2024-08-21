use qwt::perf_and_test_utils::{
    gen_queries, gen_rank_queries, gen_select_queries, load_or_build_and_save_qwt, type_of,
    TimingQueries,
};
use qwt::quadwt::RSforWT;
use qwt::utils::msb;
use qwt::utils::text_remap;
use qwt::{AccessUnsigned, QWaveletTree, RankUnsigned, SelectUnsigned, SpaceUsage};
use qwt::{QWT256Pfs, QWT512Pfs, QWT256, QWT512};

use clap::Parser;

const N_RUNS: usize = 10;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Input filename
    #[clap(short, long, value_parser)]
    input_file: String,
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10000000)]
    n_queries: usize,
    #[arg(short, long)]
    test_correctness: bool,
    #[arg(short, long)]
    rank: bool,
    #[arg(short, long)]
    access: bool,
    #[arg(short, long)]
    select: bool,
    #[arg(short, long)]
    rank_prefetch: bool,
}

fn test_rank_latency<T: RankUnsigned<Item = u8> + SpaceUsage>(
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
        ds.space_usage_byte(),
        ds.space_usage_MiB(),
        queries.len(),
        N_RUNS
    );

    println!("Result: {}", result);
}

fn test_rank_throughput<T: RankUnsigned<Item = u8> + SpaceUsage>(
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
            result += unsafe { ds.rank_unchecked(symbol, pos) };
        }
        t.stop()
    }

    let (t_min, t_max, t_avg) = t.get();
    println!(
    "RESULT algo={} exp=rank_throughput input={} n={} logn={:?} min_throughput_ms={} max_throughput_ms={} avg_throughput_ms={} space_in_bytes={} space_in_mib={:.2} n_queries={} n_runs={}",
        type_of(&ds).chars().filter(|c| !c.is_whitespace()).collect::<String>(),
    file,
        n,
        msb(n),
        (queries.len() as f64) / (((t_max * queries.len() as u128) as f64) / 1000.0 / 1000.0),
        (queries.len() as f64) / (((t_min * queries.len() as u128) as f64) / 1000.0 / 1000.0),
        (queries.len() as f64) / (((t_avg * queries.len() as u128) as f64) / 1000.0 / 1000.0),
        ds.space_usage_byte(),
        ds.space_usage_MiB(),
        queries.len(),
        N_RUNS
    );

    println!("Result: {}", result);
}

fn test_rank_prefetch_latency<RS, const WITH_PREFETCH_SUPPORT: bool>(
    ds: &QWaveletTree<u8, RS, WITH_PREFETCH_SUPPORT>,
    n: usize,
    queries: &[(usize, u8)],
    file: String,
) where
    RS: SpaceUsage,
    RS: RSforWT,
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
        ds.space_usage_byte(),
        ds.space_usage_MiB(),
        queries.len(),
        N_RUNS
    );

    println!("Result: {}", result);
}

fn test_rank_prefetch_throughput<RS, const WITH_PREFETCH_SUPPORT: bool>(
    ds: &QWaveletTree<u8, RS, WITH_PREFETCH_SUPPORT>,
    n: usize,
    queries: &[(usize, u8)],
    file: String,
) where
    RS: SpaceUsage,
    RS: RSforWT,
{
    let mut result = 0;
    let mut t = TimingQueries::new(N_RUNS, queries.len());

    for _ in 0..N_RUNS {
        t.start();
        for &(pos, symbol) in queries.iter() {
            result += unsafe { ds.rank_prefetch_unchecked(symbol, pos) };
        }
        t.stop()
    }

    let (t_min, t_max, t_avg) = t.get();
    println!(
    "RESULT algo={} exp=rank_prefetch_throughput input={} n={} logn={:?} min_throughput_ms={} max_throughput_ms={} avg_throughput_ms={} space_in_bytes={} space_in_mib={:.2} n_queries={} n_runs={}",
        type_of(&ds).chars().filter(|c| !c.is_whitespace()).collect::<String>(),
    file,
        n,
        msb(n),
        (queries.len() as f64) / (((t_max * queries.len() as u128) as f64) / 1000.0 / 1000.0),
        (queries.len() as f64) / (((t_min * queries.len() as u128) as f64) / 1000.0 / 1000.0),
        (queries.len() as f64) / (((t_avg * queries.len() as u128) as f64) / 1000.0 / 1000.0),
        ds.space_usage_byte(),
        ds.space_usage_MiB(),
        queries.len(),
        N_RUNS
    );

    println!("Result: {}", result);
}

fn test_access_latency<T: AccessUnsigned<Item = u8> + SpaceUsage>(
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
        ds.space_usage_byte(),
        ds.space_usage_MiB(),
        queries.len(),
        N_RUNS
    );
    println!("Result: {result}");
}

fn test_access_throughput<T: AccessUnsigned<Item = u8> + SpaceUsage>(
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
        ds.space_usage_byte(),
        ds.space_usage_MiB(),
        queries.len(),
        N_RUNS
    );
    println!("Result: {result}");
}

fn test_select_latency<T: SelectUnsigned<Item = u8> + SpaceUsage>(
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
        ds.space_usage_byte(),
        ds.space_usage_MiB(),
        queries.len(),
        N_RUNS
    );

    println!("Result: {}", result);
}

fn test_select_throughput<T: SelectUnsigned<Item = u8> + SpaceUsage>(
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
            result += unsafe { ds.select_unchecked(symbol, pos) };
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
        ds.space_usage_byte(),
        ds.space_usage_MiB(),
        queries.len(),
        N_RUNS
    );

    println!("Result: {}", result);
}

fn test_correctness<
    T: AccessUnsigned<Item = u8> + RankUnsigned<Item = u8> + SelectUnsigned<Item = u8> + SpaceUsage,
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

fn main() {
    let args = Args::parse();
    let input_filename = args.input_file;
    let mut text = std::fs::read(&input_filename).expect("Cannot read the input file.");

    let sigma = text_remap(&mut text);
    let n = text.len();
    println!("Text length: {:?}", n);
    println!("Alphabet size: {sigma}");

    // Generate queries
    let rank_queries = gen_rank_queries(args.n_queries, &text);
    let access_queries = gen_queries(args.n_queries, n);
    let select_queries = gen_select_queries(args.n_queries, &text);

    let output_filename = input_filename.clone() + ".256.qwt";
    let ds = load_or_build_and_save_qwt::<QWT256<_>>(&output_filename, &text);

    if args.test_correctness {
        test_correctness(&ds, &text);
    }

    if args.rank {
        test_rank_latency(&ds, n, &rank_queries, input_filename.clone());
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

    if args.rank_prefetch {
        test_rank_prefetch_latency(&ds, n, &rank_queries, input_filename.clone());
        // test_rank_prefetch_throughput(&ds, n, &rank_queries, input_filename.clone());
    }
    // TODO: make this a macro!

    let output_filename = input_filename.clone() + ".256Pfs.qwt";
    let ds = load_or_build_and_save_qwt::<QWT256Pfs<_>>(&output_filename, &text);

    if args.test_correctness {
        test_correctness(&ds, &text);
    }

    if args.rank {
        test_rank_prefetch_latency(&ds, n, &rank_queries, input_filename.clone());
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

    if args.rank_prefetch {
        test_rank_prefetch_latency(&ds, n, &rank_queries, input_filename.clone());
        // test_rank_prefetch_throughput(&ds, n, &rank_queries, input_filename.clone());
    }

    let output_filename = input_filename.clone() + ".512.qwt";
    let ds = load_or_build_and_save_qwt::<QWT512<_>>(&output_filename, &text);

    if args.test_correctness {
        test_correctness(&ds, &text);
    }

    if args.rank {
        test_rank_latency(&ds, n, &rank_queries, input_filename.clone());
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

    if args.rank_prefetch {
        test_rank_prefetch_latency(&ds, n, &rank_queries, input_filename.clone());
        // test_rank_prefetch_throughput(&ds, n, &rank_queries, input_filename.clone());
    }

    let output_filename = input_filename.clone() + ".512Pfs.qwt";
    let ds = load_or_build_and_save_qwt::<QWT512Pfs<_>>(&output_filename, &text);

    if args.test_correctness {
        test_correctness(&ds, &text);
    }

    if args.rank {
        test_rank_prefetch_latency(&ds, n, &rank_queries, input_filename.clone());
        test_rank_throughput(&ds, n, &rank_queries, input_filename.clone());
    }

    if args.access {
        test_access_latency(&ds, n, &access_queries, input_filename.clone());
        test_access_throughput(&ds, n, &access_queries, input_filename.clone());
    }

    if args.select {
        test_select_latency(&ds, n, &select_queries, input_filename.clone());
        test_select_throughput(&ds, n, &select_queries, input_filename.clone());
    }

    if args.rank_prefetch {
        test_rank_prefetch_latency(&ds, n, &rank_queries, input_filename.clone());
        test_rank_prefetch_throughput(&ds, n, &rank_queries, input_filename.clone());
    }
}
