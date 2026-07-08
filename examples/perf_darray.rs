use qwt::perf_and_test_utils::{
    gen_queries, gen_strictly_increasing_sequence, type_of, TimingQueries,
};

use mem_dbg::{MemSize, SizeFlags};
use qwt::{DArray, SelectBin};

fn main() {
    let n_queries = 100000;
    let n_runs = 3;

    for logn in 26..29 {
        let n = 1 << logn;
        let u = 2 * n;

        let seq = gen_strictly_increasing_sequence(n, u);
        let queries = gen_queries(n_queries, n);

        let ds: DArray<false> = DArray::new(seq.iter().copied().collect());
        let mut res = 0;

        let mut t = TimingQueries::new(n_runs, n_queries);
        for _ in 0..n_runs {
            t.start();
            for &q in queries.iter() {
                res += ds.select1(q).unwrap();
            }
            t.stop();
        }

        let (t_min, t_max, t_avg) = t.get();
        println!(
        "SELECT1: [ds_name: {}, n: {}, bitsize: {:?}, min_time (ns): {}, max_time (ns): {}, avg_time (ns): {}, space (bytes): {}, space (Mbyte): {:.2}]",
        type_of(&ds),
        n,
        u,
        t_min,
        t_max,
        t_avg,
        ds.mem_size(SizeFlags::default()),
        ds.mem_size(SizeFlags::default()) as f64 / (1024.0 * 1024.0),
    );

        println!("Fake: {res}");
    }
}
