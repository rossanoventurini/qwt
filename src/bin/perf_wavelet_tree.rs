use qwt::perf_and_test_utils::{
    gen_queries, gen_rank_queries, gen_select_queries, type_of, TimingQueries,
};
use qwt::utils::msb;
use qwt::utils::text_remap;
use qwt::{AccessUnsigned, RankUnsigned, SelectUnsigned, SpaceUsage};
use qwt::{QWT256, QWT512};

use std::fs;
use std::path::Path;

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
}

fn test_rank_performace<T: RankUnsigned<Item = u8> + SpaceUsage>(
    ds: &T,
    n: usize,
    queries: &[(usize, u8)],
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
            "[ds_name: {}, exp: rank_latency, n: {}, logn: {:?}, min_time (ns): {}, max_time (ns): {}, avg_time (ns): {}, space (bytes): {}, space (Mbytes): {:.2}, n_queries: {}, n_runs: {}]",
            type_of(&ds),
            n,
            msb(n),
            t_min,
            t_max,
            t_avg,
            ds.space_usage_bytes(),
            ds.space_usage_mbytes(),
            queries.len(),
            N_RUNS
        );

    println!("Result: {}", result);
}

fn test_access_performace<T: AccessUnsigned<Item = u8> + SpaceUsage>(
    ds: &T,
    n: usize,
    queries: &[usize],
) {
    let mut t = TimingQueries::new(N_RUNS, queries.len());
    let mut result: u8 = 0;
    for _ in 0..N_RUNS {
        t.start();
        for &pos in queries.iter() {
            let i = (pos * (result as usize)) % n;
            result = unsafe { ds.get_unchecked(i) };
        }
        t.stop()
    }

    let (t_min, t_max, t_avg) = t.get();
    println!(
        "[ds_name: {}, exp: access_latency, n: {}, logn: {:?}, min_time (ns): {}, max_time (ns): {}, avg_time (ns): {}, space (bytes): {}, space (Mbytes): {:.2}, n_queries: {}, n_runs: {}]",
        type_of(&ds),
        n,
        msb(n),
        t_min,
        t_max,
        t_avg,
        ds.space_usage_bytes(),
        ds.space_usage_mbytes(),
        queries.len(),
        N_RUNS
    );
    println!("Result: {result}");
}

fn test_select_performace<T: SelectUnsigned<Item = u8> + SpaceUsage>(
    ds: &T,
    n: usize,
    queries: &[(usize, u8)],
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
        "[ds_name: {}, exp: select_latency, n: {}, logn: {:?}, min_time (ns): {}, max_time (ns): {}, avg_time (ns): {}, space (bytes): {}, space (Mbytes): {:.2}, n_queries: {}, n_runs: {}]",
        type_of(&ds),
        n,
        msb(n),
        t_min,
        t_max,
        t_avg,
        ds.space_usage_bytes(),
        ds.space_usage_mbytes(),
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
        let rank = ds.rank(symbol, i + 1).unwrap();
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
    let ds: QWT256<_>;
    let path = Path::new(&output_filename);
    if path.exists() {
        println!(
            "Wavelet tree already exists. Filename: {}. I'm going to read it ...",
            output_filename
        );
        let serialized = fs::read(path).unwrap();
        println!("Serialized size: {:?} bytes", serialized.len());
        ds = bincode::deserialize::<QWT256<u8>>(&serialized).unwrap();
    } else {
        let mut t = TimingQueries::new(1, 1); // measure building time
        t.start();
        ds = QWT256::from(text.clone());
        t.stop();
        let (t_min, _, _) = t.get();
        println!("Construction time {:?} millisecs", t_min / 1000000);

        let serialized = bincode::serialize(&ds).unwrap();
        println!("Serialized size: {:?} bytes", serialized.len());
        fs::write(path, serialized).unwrap();
    }

    if args.test_correctness {
        test_correctness(&ds, &text);
    }

    if args.rank {
        test_rank_performace(&ds, n, &rank_queries);
    }

    if args.access {
        test_access_performace(&ds, n, &access_queries);
    }

    if args.select {
        test_select_performace(&ds, n, &select_queries);
    }

    // TODO: make this a macro!

    let output_filename = input_filename + ".512.qwt";
    let ds: QWT512<_>;
    let path = Path::new(&output_filename);
    if path.exists() {
        println!(
            "Wavelet tree already exists. Filename: {}. I'm going to read it ...",
            output_filename
        );
        let serialized = fs::read(path).unwrap();
        println!("Serialized size: {:?} bytes", serialized.len());
        ds = bincode::deserialize::<QWT512<u8>>(&serialized).unwrap();
    } else {
        let mut t = TimingQueries::new(1, 1); // measure building time
        t.start();
        ds = QWT512::from(text.clone());
        t.stop();
        let (t_min, _, _) = t.get();
        println!("Construction time {:?} millisecs", t_min / 1000000);

        let serialized = bincode::serialize(&ds).unwrap();
        println!("Serialized size: {:?} bytes", serialized.len());
        fs::write(path, serialized).unwrap();
    }

    if args.test_correctness {
        test_correctness(&ds, &text);
    }

    if args.rank {
        test_rank_performace(&ds, n, &rank_queries);
    }

    if args.access {
        test_access_performace(&ds, n, &access_queries);
    }

    if args.select {
        test_select_performace(&ds, n, &select_queries);
    }
}
