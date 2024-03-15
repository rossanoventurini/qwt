use qwt::{
    bitvector::rs_bitvector::RSBitVector,
    perf_and_test_utils::{gen_queries, gen_strictly_increasing_sequence, type_of, TimingQueries},
    RankBin, SpaceUsage,
};

const N_RUNS: usize = 5;
const N_QUERIES: usize = 10000000;

fn perf_rank1<T>(ds: &T, queries: &[usize], n: usize, logn: usize, u: usize)
where
    T: RankBin + SpaceUsage,
{
    let mut result = 0;

    let mut t = TimingQueries::new(N_RUNS, N_QUERIES);
    for _ in 0..N_RUNS {
        t.start();
        for &query in queries.iter() {
            let i = (query + result) % n;
            result = unsafe { ds.rank1_unchecked(i) };
        }
        t.stop();
    }
    let (t_min, t_max, t_avg) = t.get();
    println!(
        "RANK: [ds_name: {}, n: {}, logn: {}, bitsize: {:?}, min_time (ns): {}, max_time (ns): {}, avg_time (ns): {}, space (bytes): {}, space (Mbyte): {:.2}]",
        type_of(&ds),
        n,
        logn,
        u,
        t_min,
        t_max,
        t_avg,
        ds.space_usage_byte(),
        ds.space_usage_MiB()
    );
}

fn main() {
    for logn in 30..33 {
        let n = 1 << logn;

        let fill_factor = 2; // 1/2 full

        let v = gen_strictly_increasing_sequence(n / fill_factor, n);
        let rs = RSBitVector::new(v.iter().copied().collect());

        let queries = gen_queries(N_QUERIES, n);
        perf_rank1(&rs, &queries, n, logn, n);
    }
}
