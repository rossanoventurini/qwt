use qwt::perf_and_test_utils::gen_queries;
use qwt::perf_and_test_utils::gen_strictly_increasing_sequence;
use qwt::perf_and_test_utils::type_of;
use qwt::perf_and_test_utils::TimingQueries;
use qwt::BitVector;
use qwt::RSNarrow;
use qwt::RSWide;

use qwt::SpaceUsage;
use qwt::{RankBin, SelectBin};

use simple_sds::ops::Rank;
use simple_sds::ops::Select;
use simple_sds::raw_vector::AccessRaw;
use simple_sds::raw_vector::RawVector;
use sucds::bit_vectors::Rank as SucdsRank;
use sucds::bit_vectors::Rank9Sel;
use sucds::bit_vectors::Select as SucdsSelect;
use sucds::Serializable;

use simple_sds::bit_vector::BitVector as sds_BitVector;
use simple_sds::serialize::Serialize as simple_sds_Serialize;

pub trait Operations {
    fn rank(&self, i: usize) -> usize;
    fn select(&self, i: usize) -> usize;
    fn space_usage_byte(&self) -> usize;
    fn space_usage_mib(&self) -> f32 {
        (self.space_usage_byte() as f32) / ((1024 * 1024) as f32)
    }
}

impl Operations for RSWide {
    fn rank(&self, i: usize) -> usize {
        self.rank1(i).unwrap()
    }

    fn select(&self, i: usize) -> usize {
        self.select1(i).unwrap()
    }

    fn space_usage_byte(&self) -> usize {
        SpaceUsage::space_usage_byte(self)
    }
}

impl Operations for RSNarrow {
    fn rank(&self, i: usize) -> usize {
        self.rank1(i).unwrap()
    }

    fn select(&self, i: usize) -> usize {
        self.select1(i).unwrap()
    }

    fn space_usage_byte(&self) -> usize {
        SpaceUsage::space_usage_byte(self)
    }
}

impl Operations for Rank9Sel {
    fn rank(&self, i: usize) -> usize {
        SucdsRank::rank1(self, i).unwrap()
    }

    fn select(&self, i: usize) -> usize {
        SucdsSelect::select1(self, i).unwrap()
    }

    fn space_usage_byte(&self) -> usize {
        self.size_in_bytes()
    }
}

impl Operations for sds_BitVector {
    fn rank(&self, i: usize) -> usize {
        simple_sds::ops::Rank::rank(self, i)
    }

    fn select(&self, i: usize) -> usize {
        simple_sds::ops::Select::select(self, i).unwrap()
    }

    fn space_usage_byte(&self) -> usize {
        simple_sds_Serialize::size_in_bytes(self)
    }
}

fn test_rank_performace<T: Operations>(ds: &T, n: usize, queries: &[usize]) {
    let mut result = 0;
    let mut t = TimingQueries::new(N_RUNS, queries.len());

    for _ in 0..N_RUNS {
        t.start();
        for &pos in queries.iter() {
            let i = (pos + result) % n;
            result = ds.rank(i);
        }
        t.stop()
    }

    let (_, _, t_avg) = t.get();
    println!(
        "RANK: [ds_name: {}, exp: rank, avg_time (ns): {}, space (Mbytes): {:.2}]",
        type_of(&ds),
        t_avg,
        ds.space_usage_mib(),
    );

    println!("fake {}", result);
}

fn test_select_performace<T: Operations>(ds: &T, _n: usize, queries: &[usize]) {
    let mut t = TimingQueries::new(N_RUNS, queries.len());

    let mut result = 0;
    for _ in 0..N_RUNS {
        t.start();
        for &pos in queries.iter() {
            let i = pos - 1 + result % 2;
            result = ds.select(i);
        }
        t.stop()
    }

    let (_, _, t_avg) = t.get();
    println!(
        "SELECT: [ds_name: {}, exp: select, avg_time (ns): {}, space (Mbytes): {:.2}]",
        type_of(&ds),
        t_avg,
        ds.space_usage_mib(),
    );

    println!("fake {}", result);
}

const N_RUNS: usize = 3;
const N_QUERIES: usize = 100000;

fn main() {
    for logn in 23..26 {
        let n = 1 << logn;
        let u = 2 * n;

        println!("NEW ROUND OF TESTING -------------------------------------");

        let seq = gen_strictly_increasing_sequence(n, u);
        let queries = gen_queries(N_QUERIES, n);

        let rsnarrow = RSNarrow::new(seq.iter().copied().collect());
        let rswide = RSWide::new(seq.iter().copied().collect());

        let bv: BitVector = seq.iter().copied().collect();
        let rank9 = Rank9Sel::from_bits(bv.iter().collect::<Vec<_>>()).select1_hints();

        let mut rv = RawVector::with_len(u, false);
        for i in seq {
            rv.set_bit(i, true);
        }
        let mut sds = sds_BitVector::from(rv);
        sds.enable_rank();
        sds.enable_select();

        test_rank_performace(&rsnarrow, n, &queries);
        test_rank_performace(&rswide, n, &queries);
        test_rank_performace(&rank9, n, &queries);
        test_rank_performace(&sds, n, &queries);

        test_select_performace(&rsnarrow, n, &queries);
        test_select_performace(&rswide, n, &queries);
        test_select_performace(&rank9, n, &queries);
        test_select_performace(&sds, n, &queries);
    }
}
