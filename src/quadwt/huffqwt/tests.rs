use rand::Rng;

use crate::{
    perf_and_test_utils::{gen_sequence, TimingQueries},
    AccessUnsigned, HuffQWaveletTree, OccsRangeUnsigned, RSQVector512, RankUnsigned,
    SelectUnsigned, HQWT256,
};

#[test]
fn test_small() {
    let data: [u8; 9] = [1, 0, 1, 0, 3, 4, 5, 3, 7];
    let qwt = HuffQWaveletTree::<_, RSQVector512>::new(&mut data.clone());

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
    let qwt: HQWT256<_> = (0..10_u32).cycle().take(1000).collect();

    assert_eq!(qwt.len(), 1000);
}

#[test]
fn test_occs_range() {
    let data: [u8; 9] = [1, 0, 1, 0, 3, 4, 5, 3, 7];
    let qwt = HuffQWaveletTree::<_, RSQVector512>::new(&mut data.clone());

    // out-of-bounds ranges
    assert!(qwt.occs_range(..data.len() + 1).is_none());
    assert!(qwt.occs_range(data.len() - 1..data.len() + 1).is_none());

    // nonsense ranges
    assert!(qwt.occs_range(5..4).is_none());
    assert!(qwt.occs_range(2..0).is_none());

    // empty ranges
    assert_eq!(0, qwt.occs_range(data.len()..).unwrap().count());
    assert_eq!(0, qwt.occs_range(..0).unwrap().count());

    // unbounded
    let mut occs: Vec<_> = qwt.occs_range(..).unwrap().collect();
    occs.sort_by_key(|p| p.0);
    assert_eq!(occs, [(0, 2), (1, 2), (3, 2), (4, 1), (5, 1), (7, 1)]);

    // start bound
    let mut occs: Vec<_> = qwt.occs_range(3..).unwrap().collect();
    occs.sort_by_key(|p| p.0);
    assert_eq!(occs, [(0, 1), (3, 2), (4, 1), (5, 1), (7, 1)]);

    // end bound
    let mut occs: Vec<_> = qwt.occs_range(..5).unwrap().collect();
    occs.sort_by_key(|p| p.0);
    assert_eq!(occs, [(0, 2), (1, 2), (3, 1)]);

    // fully bounded
    let mut occs: Vec<_> = qwt.occs_range(4..7).unwrap().collect();
    occs.sort_by_key(|p| p.0);
    assert_eq!(occs, [(3, 1), (4, 1), (5, 1)]);

    // empty data
    let data: [u8; 0] = [];
    let qwt = HuffQWaveletTree::<_, RSQVector512>::new(&mut data.clone());
    assert_eq!(0, qwt.occs_range(..).unwrap().count());
}

/// Property-based test for occs_range on HuffQWaveletTree:
/// 1. sum of all occurrences == range length
/// 2. each symbol's count == rank(symbol, end) - rank(symbol, start)
#[test]
fn test_occs_range_properties() {
    let mut rng = rand::thread_rng();

    for sigma in [4, 16, 64, 256] {
        let sequence = gen_sequence(1000, sigma);
        let qwt = HuffQWaveletTree::<_, RSQVector512>::new(&mut sequence.clone());
        let n = sequence.len();

        // Test multiple random ranges
        for _ in 0..100 {
            let a = rng.gen_range(0..=n);
            let b = rng.gen_range(0..=n);
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
    let mut rng = rand::thread_rng();

    for sigma in [512_u16, 1000, 4000, 16000] {
        // Generate random sequence with u16 symbols
        let sequence: Vec<u16> = (0..2000).map(|_| rng.gen_range(0..sigma)).collect();
        let qwt = HuffQWaveletTree::<_, RSQVector512>::new(&mut sequence.clone());
        let n = sequence.len();

        // Test multiple random ranges
        for _ in 0..50 {
            let a = rng.gen_range(0..=n);
            let b = rng.gen_range(0..=n);
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

            // Note: HuffQWaveletTree does NOT guarantee lexicographic order
            // (Huffman codes produce a different traversal order)
        }
    }
}

#[test]
fn test_from_iterator() {
    let qwt: HQWT256<_> = (0..10u32).cycle().take(100).collect();

    assert!(qwt.into_iter().eq((0..10u32).cycle().take(100)));
}

#[test]
fn test() {
    const N: usize = 1025;
    for sigma in [4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 255] {
        let mut sequence: [u8; N] = [0; N];
        sequence[N - 1] = sigma - 1;
        let qwt = HuffQWaveletTree::<_, RSQVector512>::new(&mut sequence.clone());

        for i in 0..N - 1 {
            assert_eq!(qwt.rank(0, i).unwrap(), i);
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
        let qwt = HuffQWaveletTree::<_, RSQVector512>::new(&mut sequence.clone());

        for i in 1..N - 1 {
            assert_eq!(qwt.rank(0, i).unwrap(), i);
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
        let qwt = HuffQWaveletTree::<_, RSQVector512>::new(&mut sequence.clone());
        for (i, &symbol) in sequence.iter().enumerate() {
            assert_eq!(qwt.get(i), Some(symbol));
        }
    }
}

#[test]
fn test_serialize() {
    let qwt = HuffQWaveletTree::<_, RSQVector512>::new(&mut [0_u8; 10]);
    let s = bincode::serialize(&qwt).unwrap();

    let des_qwt = bincode::deserialize::<HuffQWaveletTree<u8, RSQVector512>>(&s).unwrap();

    assert_eq!(des_qwt, qwt);
}

fn gen_seq(n: usize, sigma: usize) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(0..sigma) as u64).collect()
}

#[test]
fn test_big_sigma() {
    let seq = gen_seq(100_000, 10_000);

    let qwt = HuffQWaveletTree::<u64, RSQVector512>::from(seq.clone());

    assert_eq!(seq, qwt.iter().collect::<Vec<_>>());
}

#[test]
fn test_runtinme_get() {
    let n = 100_000;
    let sigma = 140;
    let n_queries = 10_000;

    let seq = gen_seq(n, sigma);
    let queries = gen_seq(n_queries, n);

    let qwt = HQWT256::from(seq.clone());

    let mut check = 0;

    let mut timer = TimingQueries::new(1, n_queries);

    timer.start();
    for q in queries {
        check ^= qwt.get(q as usize).unwrap();
    }
    timer.stop();

    println!("check {}", check);
    println!("took {:?}", timer.get());
}

// QWT256 took (1099, 1099, 1099)
// HQWT took (1616, 1616, 1616)
