use qwt::perf_and_test_utils::{gen_queries, type_of, TimingQueries};
use qwt::RSQVectorP256;

// traits
use qwt::{AccessUnsigned, RankUnsigned, SelectUnsigned, SpaceUsage};

const N_RUNS: usize = 15;
const N_QUERIES: usize = 10000000;

fn perf_rank<T>(ds: &T, queries: &[usize], n: usize, logn: usize, u: usize)
where
    T: RankUnsigned<Item = u8> + SpaceUsage,
{
    let mut result = 0;

    let mut t = TimingQueries::new(N_RUNS, N_QUERIES);
    for _ in 0..N_RUNS {
        t.start();
        for &query in queries.iter() {
            let i = (query + result) % n;
            result = unsafe { ds.rank_unchecked((result % 4) as u8, i) };
        }
        t.stop();
    }
    let (t_min, t_max, t_avg) = t.get();
    println!(
        "RANK: [ds_name: {}, n: {}, logn: {}, bitsize: {:?}, min_time (ns): {}, max_time (ns): {}, avg_time (ns): {}, space (bytes): {}, space (Mbytes): {:.2}]",
        type_of(&ds),
        n,
        logn,
        u,
        t_min,
        t_max,
        t_avg,
        ds.space_usage_bytes(),
        ds.space_usage_mbytes()
    );
}

fn perf_get<T>(ds: &T, queries: &[usize], n: usize, logn: usize, u: usize)
where
    T: AccessUnsigned<Item = u8> + SpaceUsage,
{
    let mut result = 0;

    let mut t = TimingQueries::new(N_RUNS, N_QUERIES);
    for _ in 0..N_RUNS {
        t.start();
        for &query in queries.iter() {
            let i = (query + result as usize) % n;
            result = unsafe { ds.get_unchecked(i) };
        }
        t.stop();
    }
    println!("Fake {}", result);
    let (t_min, t_max, t_avg) = t.get();
    println!(
        "GET: [ds_name: {}, n: {}, logn: {}, bitsize: {:?}, min_time (ns): {}, max_time (ns): {}, avg_time (ns): {}, space (bytes): {}, space (Mbytes): {:.2}]",
        type_of(&ds),
        n,
        logn,
        u,
        t_min,
        t_max,
        t_avg,
        ds.space_usage_bytes(),
        ds.space_usage_mbytes()
    );
}

fn perf_select<T>(ds: &T, queries: &[usize], n: usize, logn: usize, u: usize)
where
    T: SelectUnsigned<Item = u8> + SpaceUsage,
{
    let mut result = 0;

    let mut t = TimingQueries::new(N_RUNS, N_QUERIES);
    for _ in 0..N_RUNS {
        t.start();
        for &query in queries.iter() {
            let mut i = query + result % 2;
            if i == 0 {
                i = 1;
            }
            result = ds.select((result % 4) as u8, i).unwrap();
        }
        t.stop();
    }
    let (t_min, t_max, t_avg) = t.get();
    println!(
        "SELECT: [ds_name: {}, n: {}, logn: {}, bitsize: {:?}, min_time (ns): {}, max_time (ns): {}, avg_time (ns): {}, space (bytes): {}, space (Mbytes): {:.2}]",
        type_of(&ds),
        n,
        logn,
        u,
        t_min,
        t_max,
        t_avg,
        ds.space_usage_bytes(),
        ds.space_usage_mbytes()
    );
}

fn main() {
    for logn in 33..34 {
        let n = 1 << logn;

        let v: Vec<u8> = [0, 1, 2, 3].into_iter().cycle().take(n).collect();
        let rsqv = RSQVectorP256::new(&v);

        let queries = gen_queries(N_QUERIES, n);
        perf_rank(&rsqv, &queries, n, logn, 2 * n);

        let queries = gen_queries(N_QUERIES, n);
        perf_get(&rsqv, &queries, n, logn, 2 * n);

        let queries = gen_queries(N_QUERIES, n / 4 - 10);
        perf_select(&rsqv, &queries, n, logn, 2 * n);
    }
}
