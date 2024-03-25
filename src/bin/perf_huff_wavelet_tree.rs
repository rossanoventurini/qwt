use std::{fs, path::Path};

use clap::Parser;
use qwt::{
    perf_and_test_utils::{
        gen_queries, gen_rank_queries, gen_select_queries, type_of, TimingQueries,
    },
    utils::msb,
    AccessUnsigned, RankUnsigned, SelectUnsigned, SpaceUsage, HQWT256,
};
use serde::{Deserialize, Serialize};

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
    T: AccessUnsigned<Item = u8> + RankUnsigned<Item = u8> + SelectUnsigned<Item = u8> + SpaceUsage,
>(
    ds: &T,
    sequence: &[u8],
) {
    print!("\nTesting correctness... ");
    for (i, &symbol) in sequence.iter().enumerate() {
        assert_eq!(ds.get(i), Some(symbol));
        let rank = ds.rank(symbol, i + 1).unwrap();
        let s = ds.select(symbol, rank).unwrap();
        assert_eq!(s, i);
    }
    println!("Everything is ok!\n");
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
            result = unsafe { ds.select_unchecked(symbol, i) };
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

fn main() {
    let args = Args::parse();
    let input_filename = args.input_file;
    let mut text = std::fs::read(&input_filename).expect("Cannot read the input file.");

    let n = text.len();
    println!("Text length: {:?}", n);

    // Generate queries
    let rank_queries = gen_rank_queries(args.n_queries, &text);
    let access_queries = gen_queries(args.n_queries, n);
    let select_queries = gen_select_queries(args.n_queries, &text);

    let output_filename = input_filename.clone() + ".256.hqwt";
    let ds = load_or_build_and_save_qwt::<HQWT256<_>>(&output_filename, &text);

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
}
