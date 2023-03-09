use qwt::perf_and_test_utils::{
    gen_queries, gen_rank_queries, gen_select_queries, type_of, TimingQueries,
};
use qwt::utils::{msb, text_remap};
use qwt::{QWT256, QWT512};

use sucds::WaveletMatrixBuilder;
use sucds::{Searial, WaveletMatrix};

// use sdsl::int_vectors;
// use sdsl::wavelet_trees::WtInt;

use clap::Parser;

const N_RUNS: usize = 3;

// Use this just to have a common interface between qwt and sucds libraries
pub trait Operations {
    fn rank(&self, s: u8, i: usize) -> usize;
    fn select(&self, s: u8, i: usize) -> usize;
    fn get(&self, i: usize) -> u8;
    fn space_usage_bytes(&self) -> usize;
    fn space_usage_mib(&self) -> f32 {
        (self.space_usage_bytes() as f32) / ((1024 * 1024) as f32)
    }
}

impl Operations for WaveletMatrix {
    fn rank(&self, s: u8, i: usize) -> usize {
        self.rank(i, s as usize)
    }
    fn select(&self, s: u8, i: usize) -> usize {
        self.select(i - 1, s as usize)
    }
    fn get(&self, i: usize) -> u8 {
        self.get(i) as u8
    }
    fn space_usage_bytes(&self) -> usize {
        self.size_in_bytes()
    }
}

// impl<'a> Operations
//     for WtInt<'a, sdsl::bit_vectors::BitVector, sdsl::rank_supports::RankSupportV<'a, sdsl::bit_patterns::P1>, sdsl::select_supports::SelectSupportMcl<'a, sdsl::bit_patterns::P1>, sdsl::select_supports::SelectSupportMcl<'a, sdsl::bit_patterns::P0>> {
//     fn rank(&self, s: u8, i: usize) -> usize {
//         self.rank(i, s as usize)
//     }
//     fn select(&self, s: u8, i: usize) -> usize {
//         self.select(i - 1, s as usize)
//     }
//     fn get(&self, i: usize) -> u8 {
//         self.get(i) as u8
//     }
//     fn space_usage_bytes(&self) -> usize {
//         0
//     }
// }

use qwt::{AccessUnsigned, RankUnsigned, SelectUnsigned, SpaceUsage};
impl Operations for QWT256<u8> {
    fn rank(&self, s: u8, i: usize) -> usize {
        unsafe { self.rank_unchecked(s, i) }
    }
    fn select(&self, s: u8, i: usize) -> usize {
        unsafe { self.select_unchecked(s, i) }
    }
    fn get(&self, i: usize) -> u8 {
        unsafe { self.get_unchecked(i) }
    }
    fn space_usage_bytes(&self) -> usize {
        SpaceUsage::space_usage_bytes(self)
    }
}

impl Operations for QWT512<u8> {
    fn rank(&self, s: u8, i: usize) -> usize {
        unsafe { self.rank_unchecked(s, i) }
    }
    fn select(&self, s: u8, i: usize) -> usize {
        unsafe { self.select_unchecked(s, i) }
    }
    fn get(&self, i: usize) -> u8 {
        unsafe { self.get_unchecked(i) }
    }
    fn space_usage_bytes(&self) -> usize {
        SpaceUsage::space_usage_bytes(self)
    }
}

fn test_rank_performace<T: Operations>(ds: &T, n: usize, queries: &[(usize, u8)]) {
    let mut result = 0;
    let mut t = TimingQueries::new(N_RUNS, queries.len());

    for _ in 0..N_RUNS {
        t.start();
        for &(pos, symbol) in queries.iter() {
            let i = (pos + result) % n;
            result = ds.rank(symbol, i);
        }
        t.stop()
    }

    let (_, _, t_avg) = t.get();
    println!(
        "[ds_name: {}, exp: rank, avg_time (ns): {}, space (Mbytes): {:.2}]",
        type_of(&ds),
        t_avg,
        ds.space_usage_mib(),
    );

    println!("fake {}", result);
}

fn test_access_performace<T: Operations>(ds: &T, n: usize, queries: &[usize]) {
    let mut t = TimingQueries::new(N_RUNS, queries.len());
    let mut result: u8 = 0;
    for _ in 0..N_RUNS {
        t.start();
        for &pos in queries.iter() {
            let i = (pos * (result as usize)) % n;
            result = ds.get(i);
        }
        t.stop()
    }

    let (_, _, t_avg) = t.get();
    println!(
        "[ds_name: {}, exp: access, avg_time (ns): {}, space (Mbytes): {:.2}]",
        type_of(&ds),
        t_avg,
        ds.space_usage_mib(),
    );
    println!("Fake {result}");
}

fn test_select_performace<T: Operations>(ds: &T, _n: usize, queries: &[(usize, u8)]) {
    let mut t = TimingQueries::new(N_RUNS, queries.len());

    let mut result = 0;
    for _ in 0..N_RUNS {
        t.start();
        for &(pos, symbol) in queries.iter() {
            let i = pos - 1 + result % 2;
            let i = std::cmp::max(1, i);
            result = ds.select(symbol, i);
        }
        t.stop()
    }

    let (_, _, t_avg) = t.get();
    println!(
        "[ds_name: {}, exp: select, avg_time (ns): {}, space (Mbytes): {:.2}]",
        type_of(&ds),
        t_avg,
        ds.space_usage_mib(),
    );

    println!("fake {}", result);
}

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
    rank: bool,
    #[arg(short, long)]
    access: bool,
    #[arg(short, long)]
    select: bool,
}

fn main() {
    let args = Args::parse();
    let input_filename = args.input_file;
    let mut text = std::fs::read(input_filename).expect("Cannot read the input file.");
    let sigma = text_remap(&mut text);
    let n = text.len();
    let n_levels = msb(sigma) as usize;

    println!("Text length: {n}");
    println!("Alphabet size: {sigma}");
    println!("Number of queries: {}", args.n_queries);
    println!("Number of levels: {n_levels}");

    // Generate queries
    let rank_queries = gen_rank_queries(args.n_queries, &text);
    let access_queries = gen_queries(args.n_queries, n);
    let select_queries = gen_select_queries(args.n_queries, &text);

    // Building wavelet trees
    let mut wmb = WaveletMatrixBuilder::with_width(n_levels);
    text.iter().for_each(|c| wmb.push(*c as usize));
    let sucds_wm = wmb.build().unwrap();

    let qwt256 = QWT256::from(text.clone());
    let qwt512 = QWT512::from(text.clone());

    // let mut iv = sdsl::int_vectors::IntVector::<0>::new(n, 0, Some(n_levels as u8)).unwrap();
    // text.iter()
    //     .enumerate()
    //     .for_each(|(i, &c)| iv.set(i, c as usize));

    // let sdsl_wt =
    //     sdsl::wavelet_trees::WtInt::<sdsl::bit_vectors::BitVector>::from_int_vector(&iv).unwrap();

    if args.rank {
        test_rank_performace(&sucds_wm, n, &rank_queries);
        test_rank_performace(&qwt256, n, &rank_queries);
        test_rank_performace(&qwt512, n, &rank_queries);
        //test_rank_performace(&sdsl_wt, n, &rank_queries);
    }

    if args.access {
        test_access_performace(&sucds_wm, n, &access_queries);
        test_access_performace(&qwt256, n, &access_queries);
        test_access_performace(&qwt512, n, &access_queries);
        //test_access_performace(&sdsl_wt, n, &access_queries);
    }

    if args.select {
        test_select_performace(&sucds_wm, n, &select_queries);
        test_select_performace(&qwt256, n, &select_queries);
        test_select_performace(&qwt256, n, &select_queries);
        //test_access_performace(&sdsl_wt, n, &access_queries);
    }
}
