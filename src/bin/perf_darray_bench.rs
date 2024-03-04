use qwt::perf_and_test_utils::{
    gen_queries, gen_strictly_increasing_sequence, type_of, TimingQueries,
};
use qwt::{BitVector, DArray, SelectBin, SpaceUsage};

use sucds::bit_vectors::darray::DArray as sucds_DArray;
use sucds::bit_vectors::Select as sucds_Select;
use sucds::Serializable;

trait Select {
    fn select(&self, i: usize) -> usize;
    fn space_usage_byte(&self) -> usize;
    fn space_usage_mib(&self) -> f32 {
        (self.space_usage_byte() as f32) / ((1024 * 1024) as f32)
    }
}

impl Select for DArray {
    #[inline]
    fn select(&self, i: usize) -> usize {
        self.select1(i).unwrap()
    }
    fn space_usage_byte(&self) -> usize {
        SpaceUsage::space_usage_byte(self)
    }
}

impl Select for sucds_DArray {
    #[inline]
    fn select(&self, i: usize) -> usize {
        self.select1(i).unwrap()
    }
    fn space_usage_byte(&self) -> usize {
        self.size_in_bytes()
    }
}

fn run_queries<T: Select>(ds: &T, queries: &[usize], n_runs: usize, u: usize, n: usize) {
    let mut res = 0;
    let n_queries = queries.len();

    let mut t = TimingQueries::new(n_runs, n_queries);
    for _ in 0..n_runs {
        t.start();
        for &q in queries.iter() {
            res += ds.select(q);
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
    ds.space_usage_byte(),
    ds.space_usage_mib()
);

    println!("Fake: {res}");
}

fn main() {
    let n_queries = 100000;
    let n_runs = 3;

    for logn in 26..29 {
        let n = 1 << logn;
        let u = 2 * n;

        let seq = gen_strictly_increasing_sequence(n, u);
        let queries = gen_queries(n_queries, n);

        let ds: DArray<false> = DArray::new(seq.iter().copied().collect());

        run_queries(&ds, &queries, n_runs, u, n);

        let bv: BitVector = seq.iter().copied().collect();
        let v: Vec<bool> = bv.iter().collect();
        let ds = sucds_DArray::from_bits(v);

        run_queries(&ds, &queries, n_runs, u, n);
        println!();
    }
}
