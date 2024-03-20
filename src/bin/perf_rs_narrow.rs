use qwt::{
    perf_and_test_utils::{gen_queries, type_of, TimingQueries},
    RSNarrow, RankBin, SelectBin, SpaceUsage,
};

const N_RUNS: usize = 5;
const N_QUERIES: usize = 10_000_000;

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
            result = ds
                .select1(i)
                .unwrap_or_else(|| panic!("None on select1({})", i));
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
            result = ds
                .select0(i)
                .unwrap_or_else(|| panic!("None on select0({})", i));
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
        let n: usize = 1 << logn;

        let fill_factor = 2; // 1/2 full

        let bv = (0..n).filter(|x| x % fill_factor == 0).collect();
        let rs = RSNarrow::new(bv);

        println!(
            "created new rs_narrow | n_ones: {} | n_zeros: {}",
            rs.n_ones(),
            rs.n_zeros(),
        );

        let queries = gen_queries(N_QUERIES, n);
        perf_rank1(&rs, &queries, n, logn, n);

        let queries = gen_queries(N_QUERIES, rs.n_ones() - 1);
        perf_select1(&rs, &queries, n, logn, n);

        let queries = gen_queries(N_QUERIES, rs.n_zeros() - 1);
        perf_select0(&rs, &queries, n, logn, n);
    }
}
