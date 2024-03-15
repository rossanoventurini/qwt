use qwt::{
    bitvector::rs_bitvector::RSBitVector,
    perf_and_test_utils::{gen_queries, type_of, TimingQueries},
    BitVector, RankBin, SelectBin, SpaceUsage,
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

fn perf_select1<T>(ds: &T, queries: &[usize], n: usize, logn: usize, u: usize)
where
    T: SelectBin + SpaceUsage,
{
    let mut result = 0;

    let mut t = TimingQueries::new(N_RUNS, N_QUERIES);
    for _ in 0..N_RUNS {
        t.start();
        for &query in queries.iter() {
            let i = query + result % 2;
            result = ds.select1(i).unwrap();
        }
        t.stop();
    }
    let (t_min, t_max, t_avg) = t.get();
    println!(
        "SELECT1: [ds_name: {}, n: {}, logn: {}, bitsize: {:?}, min_time (ns): {}, max_time (ns): {}, avg_time (ns): {}, space (bytes): {}, space (Mbytes): {:.2}]",
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

fn perf_select0<T>(ds: &T, queries: &[usize], n: usize, logn: usize, u: usize)
where
    T: SelectBin + SpaceUsage,
{
    let mut result = 0;

    let mut t = TimingQueries::new(N_RUNS, N_QUERIES);
    for _ in 0..N_RUNS {
        t.start();
        for &query in queries.iter() {
            let i = query + result % 2;
            result = ds.select0(i).unwrap();
        }
        t.stop();
    }
    let (t_min, t_max, t_avg) = t.get();
    println!(
        "SELECT0: [ds_name: {}, n: {}, logn: {}, bitsize: {:?}, min_time (ns): {}, max_time (ns): {}, avg_time (ns): {}, space (bytes): {}, space (Mbytes): {:.2}]",
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

        let v: Vec<_> = (0..n).filter(|x| x % fill_factor == 0).collect();
        let bv = v.iter().copied().collect();
        let rs = RSBitVector::new(bv);

        let queries = gen_queries(N_QUERIES, n);
        perf_rank1(&rs, &queries, n, logn, n);

        let queries = gen_queries(N_QUERIES, n / fill_factor);
        perf_select1(&rs, &queries, n, logn, n);

        perf_select0(&rs, &queries, n, logn, n);
    }
}
