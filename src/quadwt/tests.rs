use super::*;
use crate::perf_and_test_utils::gen_sequence;
use crate::RSQVector512;
use crate::QWT256;
use crate::{OccsRangeUnsigned, RankUnsigned};
use rand::RngExt;

#[test]
fn test_small() {
    let data: [u8; 9] = [1, 0, 1, 0, 3, 4, 5, 3, 7];
    let qwt = QWaveletTree::<_, RSQVector512>::new(&mut data.clone());

    assert_eq!(qwt.rank(1, 4), Some(2));
    assert_eq!(qwt.rank(1, 0), Some(0));
    assert_eq!(qwt.rank(8, 1), None); // too large symbol
    assert_eq!(qwt.rank(1, 9), Some(2));
    assert_eq!(qwt.rank(7, 9), Some(1));
    assert_eq!(qwt.rank(1, 10), None); // too large position
    assert_eq!(qwt.select(5, 0), Some(6));

    for (i, &v) in data.iter().enumerate() {
        let rank = qwt.rank(v, i).unwrap();
        let s = qwt.select(v, rank).unwrap();
        assert_eq!(s, i);
    }

    // test iterators
    assert!(qwt.iter().eq(data.iter().copied()));
    assert!(qwt.into_iter().eq(data.iter().copied()));

    // test from_iterator
    let qwt: QWT256<_> = (0..10_u32).cycle().take(1000).collect();

    assert_eq!(qwt.len(), 1000);
}

#[test]
fn test_occs_range() {
    let data: [u8; 9] = [1, 0, 1, 0, 3, 4, 5, 3, 7];
    let qwt = QWaveletTree::<_, RSQVector512>::new(&mut data.clone());

    // out-of-bounds ranges
    assert!(qwt.occs_range(..data.len() + 1).is_none());
    assert!(qwt.occs_range(data.len() - 1..data.len() + 1).is_none());

    // nonsense ranges (struct form avoids clippy::reversed_empty_ranges on literals)
    assert!(qwt.occs_range(std::ops::Range { start: 5, end: 4 }).is_none());
    assert!(qwt.occs_range(std::ops::Range { start: 2, end: 0 }).is_none());

    // empty ranges
    assert_eq!(0, qwt.occs_range(data.len()..).unwrap().count());
    assert_eq!(0, qwt.occs_range(..0).unwrap().count());

    // unbounded
    let occs: Vec<_> = qwt.occs_range(..).unwrap().collect();
    assert!(occs.is_sorted_by_key(|(s, _)| s));
    assert_eq!(occs, [(0, 2), (1, 2), (3, 2), (4, 1), (5, 1), (7, 1)]);

    // start bound
    let occs: Vec<_> = qwt.occs_range(3..).unwrap().collect();
    assert!(occs.is_sorted_by_key(|(s, _)| s));
    assert_eq!(occs, [(0, 1), (3, 2), (4, 1), (5, 1), (7, 1)]);

    // end bound
    let occs: Vec<_> = qwt.occs_range(..5).unwrap().collect();
    assert!(occs.is_sorted_by_key(|(s, _)| s));
    assert_eq!(occs, [(0, 2), (1, 2), (3, 1)]);

    // fully bounded
    let occs: Vec<_> = qwt.occs_range(4..7).unwrap().collect();
    assert!(occs.is_sorted_by_key(|(s, _)| s));
    assert_eq!(occs, [(3, 1), (4, 1), (5, 1)]);

    // empty data
    let data: [u8; 0] = [];
    let qwt = QWaveletTree::<_, RSQVector512>::new(&mut data.clone());
    assert_eq!(0, qwt.occs_range(..).unwrap().count());
}

/// Property-based test for occs_range:
/// 1. sum of all occurrences == range length
/// 2. each symbol's count == rank(symbol, end) - rank(symbol, start)
#[test]
fn test_occs_range_properties() {
    let mut rng = rand::rng();

    for sigma in [4, 16, 64, 256] {
        let sequence = gen_sequence(1000, sigma);
        let qwt = QWaveletTree::<_, RSQVector512>::new(&mut sequence.clone());
        let n = sequence.len();

        // Test multiple random ranges
        for _ in 0..100 {
            let a = rng.random_range(0..=n);
            let b = rng.random_range(0..=n);
            let (start, end) = if a <= b { (a, b) } else { (b, a) };

            let occs: Vec<_> = qwt.occs_range(start..end).unwrap().collect();

            // Property 1: sum of occurrences == range length
            let total: usize = occs.iter().map(|(_, count)| count).sum();
            assert_eq!(
                total,
                end - start,
                "Sum of occurrences should equal range length for range {}..{}",
                start,
                end
            );

            // Property 2: each count matches rank difference
            for (symbol, count) in &occs {
                let rank_end = qwt.rank(*symbol, end).unwrap();
                let rank_start = qwt.rank(*symbol, start).unwrap();
                assert_eq!(
                    *count,
                    rank_end - rank_start,
                    "Count mismatch for symbol {} in range {}..{}",
                    symbol,
                    start,
                    end
                );
            }

            // Property 3: symbols not in occs should have zero count
            for s in 0..sigma {
                let s = s as u8;
                let rank_end = qwt.rank(s, end).unwrap_or(0);
                let rank_start = qwt.rank(s, start).unwrap_or(0);
                let expected_count = rank_end - rank_start;

                let found_count = occs
                    .iter()
                    .find(|(sym, _)| *sym == s)
                    .map(|(_, c)| *c)
                    .unwrap_or(0);

                assert_eq!(
                    found_count, expected_count,
                    "Symbol {} should have count {} but found {} in range {}..{}",
                    s, expected_count, found_count, start, end
                );
            }
        }
    }
}

/// Test occs_range with large alphabets (σ > 256)
/// Uses u16 symbols to support larger alphabet sizes
#[test]
fn test_occs_range_large_alphabet() {
    let mut rng = rand::rng();

    for sigma in [512_u16, 1000, 4000, 16000] {
        // Generate random sequence with u16 symbols
        let sequence: Vec<u16> = (0..2000).map(|_| rng.random_range(0..sigma)).collect();
        let qwt = QWaveletTree::<_, RSQVector512>::new(&mut sequence.clone());
        let n = sequence.len();

        // Test multiple random ranges
        for _ in 0..50 {
            let a = rng.random_range(0..=n);
            let b = rng.random_range(0..=n);
            let (start, end) = if a <= b { (a, b) } else { (b, a) };

            let occs: Vec<_> = qwt.occs_range(start..end).unwrap().collect();

            // Property 1: sum of occurrences == range length
            let total: usize = occs.iter().map(|(_, count)| count).sum();
            assert_eq!(
                total,
                end - start,
                "σ={}: Sum of occurrences should equal range length for range {}..{}",
                sigma,
                start,
                end
            );

            // Property 2: each count matches rank difference
            for (symbol, count) in &occs {
                let rank_end = qwt.rank(*symbol, end).unwrap();
                let rank_start = qwt.rank(*symbol, start).unwrap();
                assert_eq!(
                    *count,
                    rank_end - rank_start,
                    "σ={}: Count mismatch for symbol {} in range {}..{}",
                    sigma,
                    symbol,
                    start,
                    end
                );
            }

            // Property 3: lexicographic ordering (for QWaveletTree)
            assert!(
                occs.is_sorted_by_key(|(s, _)| s),
                "σ={}: Results should be in lexicographic order",
                sigma
            );
        }
    }
}

#[test]
fn test_from_iterator() {
    let qwt: QWT256<_> = (0..10u32).cycle().take(100).collect();

    assert!(qwt.into_iter().eq((0..10u32).cycle().take(100)));
}

#[test]
fn test() {
    const N: usize = 1025;
    for sigma in [4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 255, 633] {
        let mut sequence: [u16; N] = [0; N];
        sequence[N - 1] = sigma - 1;
        let qwt = QWaveletTree::<_, RSQVector512>::new(&mut sequence.clone());

        for i in 0..N - 1 {
            assert_eq!(qwt.rank(0, i).unwrap(), i);
        }

        for i in 0..N {
            assert_eq!(qwt.rank(sigma - 2, i).unwrap(), 0);
        }

        for (i, &symbol) in sequence.iter().enumerate() {
            let rank = qwt.rank(symbol, i).unwrap();
            let s = qwt.select(symbol, rank).unwrap();
            assert_eq!(s, i);
        }

        // Select out of bound
        assert_eq!(qwt.select(0, N), None);
        assert_eq!(qwt.select(1, 1), None);
        assert_eq!(qwt.select(sigma - 1, 2), None);
    }

    for sigma in [4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 255, 256, 16000] {
        let mut sequence: [u16; N] = [0; N];
        sequence[N - 1] = sigma - 1;
        let qwt = QWaveletTree::<_, RSQVector512>::new(&mut sequence.clone());

        for i in 1..N - 1 {
            assert_eq!(qwt.rank(0, i).unwrap(), i);
        }

        for i in 1..N {
            assert_eq!(qwt.rank(sigma - 2, i).unwrap(), 0);
        }

        for (i, &symbol) in sequence.iter().enumerate() {
            let rank = qwt.rank(symbol, i).unwrap();
            let s = qwt.select(symbol, rank).unwrap();
            assert_eq!(s, i);
        }

        // Select out of bound
        assert_eq!(qwt.select(0, N), None);
        assert_eq!(qwt.select(1, 1), None);
        assert_eq!(qwt.select(sigma - 1, 2), None);
    }
}

#[test]
fn test_get() {
    let n = 1025;
    for sigma in [4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 255, 256] {
        let sequence = gen_sequence(n, sigma);
        let qwt = QWaveletTree::<_, RSQVector512>::new(&mut sequence.clone());
        for (i, &symbol) in sequence.iter().enumerate() {
            assert_eq!(qwt.get(i), Some(symbol));
        }
    }
}

#[test]
fn test_serialize() {
    let qwt = QWaveletTree::<_, RSQVector512>::new(&mut [0_u8; 10]);
    let s = bincode::serialize(&qwt).unwrap();

    let des_qwt = bincode::deserialize::<QWaveletTree<u8, RSQVector512>>(&s).unwrap();

    assert_eq!(des_qwt, qwt);
}


#[test]
fn test_get_and_rank() {
    let data: [u8; 9] = [1, 0, 1, 0, 3, 4, 5, 3, 7];
    let qwt = QWaveletTree::<_, RSQVector512>::new(&mut data.clone());

    assert_eq!(qwt.get_and_rank(9), None);
    for (i, &expected) in data.iter().enumerate() {
        let (sym, rank_inc) = qwt.get_and_rank(i).unwrap();
        assert_eq!(sym, expected);
        // rank(symbol, i+1) == occurrences in 0..=i
        assert_eq!(rank_inc, qwt.rank(sym, i + 1).unwrap());
        // also equals 1 + rank(symbol, i)
        assert_eq!(rank_inc, qwt.rank(sym, i).unwrap() + 1);
    }
}

#[test]
fn test_extract_range_matches_sorted_get() {
    let data: [u8; 9] = [1, 0, 1, 0, 3, 4, 5, 3, 7];
    let qwt = QWaveletTree::<_, RSQVector512>::new(&mut data.clone());

    // empty / oob
    assert!(qwt.extract_range(0..0).is_empty());
    assert!(qwt.extract_range(5..5).is_empty());
    assert!(qwt.extract_range(9..9).is_empty());
    // Reversed bounds: construct Range so clippy does not flag a literal empty range.
    let reversed = std::ops::Range { start: 3, end: 2 };
    assert!(qwt.extract_range(reversed).is_empty());
    assert!(qwt.extract_range(0..10).is_empty()); // end > n

    // full range multiset
    let multiset = qwt.extract_range(0..9);
    let mut expected: Vec<_> = data.to_vec();
    expected.sort_unstable();
    assert_eq!(multiset, expected);

    // distinct
    let distinct = qwt.extract_range_distinct(0..9);
    let mut exp_d = expected.clone();
    exp_d.dedup();
    assert_eq!(distinct, exp_d);

    // subranges
    for start in 0..=9 {
        for end in start..=9 {
            let got = qwt.extract_range(start..end);
            let mut exp: Vec<_> = data[start..end].to_vec();
            exp.sort_unstable();
            assert_eq!(got, exp, "multiset mismatch for {}..{}", start, end);

            let got_d = qwt.extract_range_distinct(start..end);
            let mut exp_d = exp.clone();
            exp_d.dedup();
            assert_eq!(got_d, exp_d, "distinct mismatch for {}..{}", start, end);

            // into API
            let mut buf = vec![99u8; 3];
            qwt.extract_range_distinct_into(start..end, &mut buf);
            assert_eq!(buf, exp_d);
        }
    }

    // singleton short-circuit path
    assert_eq!(qwt.extract_range(4..5), vec![3]);
    assert_eq!(qwt.extract_range_distinct(4..5), vec![3]);
}

/// Property: extract_range == sort(per-row get); distinct == sort+dedup
#[test]
fn test_extract_range_properties() {
    use crate::AccessUnsigned;
    let mut rng = rand::rng();

    for sigma in [4, 16, 64, 256] {
        let sequence = gen_sequence(1000, sigma);
        let qwt = QWaveletTree::<_, RSQVector512>::new(&mut sequence.clone());
        let n = sequence.len();

        for _ in 0..50 {
            let a = rng.random_range(0..=n);
            let b = rng.random_range(0..=n);
            let (start, end) = if a <= b { (a, b) } else { (b, a) };

            let got = qwt.extract_range(start..end);
            let mut exp: Vec<_> = (start..end).map(|i| qwt.get(i).unwrap()).collect();
            exp.sort_unstable();
            assert_eq!(
                got, exp,
                "σ={} multiset mismatch for {}..{}",
                sigma, start, end
            );

            let got_d = qwt.extract_range_distinct(start..end);
            let mut exp_d = exp.clone();
            exp_d.dedup();
            assert_eq!(
                got_d, exp_d,
                "σ={} distinct mismatch for {}..{}",
                sigma, start, end
            );

            // multiset length == range length
            assert_eq!(got.len(), end - start);
            // distinct is sorted unique
            assert!(got_d.windows(2).all(|w| w[0] < w[1]) || got_d.len() <= 1);
        }
    }
}

#[test]
fn test_extract_range_large_alphabet() {
    let mut rng = rand::rng();

    for sigma in [512_u16, 4000] {
        let sequence: Vec<u16> = (0..1500).map(|_| rng.random_range(0..sigma)).collect();
        let qwt = QWaveletTree::<_, RSQVector512>::new(&mut sequence.clone());
        let n = sequence.len();

        for _ in 0..30 {
            let a = rng.random_range(0..=n);
            let b = rng.random_range(0..=n);
            let (start, end) = if a <= b { (a, b) } else { (b, a) };

            let got = qwt.extract_range(start..end);
            let mut exp: Vec<_> = sequence[start..end].to_vec();
            exp.sort_unstable();
            assert_eq!(got, exp);

            let got_d = qwt.extract_range_distinct(start..end);
            let mut exp_d = exp.clone();
            exp_d.dedup();
            assert_eq!(got_d, exp_d);
        }

        // get_and_rank spot check
        for i in (0..n).step_by(17) {
            let (sym, r) = qwt.get_and_rank(i).unwrap();
            assert_eq!(sym, sequence[i]);
            assert_eq!(r, qwt.rank(sym, i + 1).unwrap());
        }
    }
}

#[test]
fn test_extract_range_empty_tree() {
    let qwt = QWaveletTree::<u8, RSQVector512>::default();
    assert!(qwt.extract_range(0..0).is_empty());
    assert!(qwt.extract_range_distinct(0..0).is_empty());
    assert_eq!(qwt.get_and_rank(0), None);
}
