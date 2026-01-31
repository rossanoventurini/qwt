use crate::perf_and_test_utils::gen_sequence;
use crate::{AccessUnsigned, OccsRangeUnsigned, RankUnsigned, SelectUnsigned, HWT, WT};
use rand::Rng;

#[test]
fn build_test() {
    let s: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 1, 1, 1, 1, 30000];

    let wt = WT::new(&mut s.clone());
    let hwt = HWT::new(&mut s.clone());

    println!("{:?} {:?}", wt.codes_decode, wt.lens);
    println!("{:?} {:?}", hwt.codes_decode, hwt.lens);

    for (i, &c) in s.iter().enumerate() {
        assert_eq!(Some(c), wt.get(i));
        assert_eq!(Some(c), hwt.get(i));
    }
}

#[test]
fn test_small() {
    let data: [u32; 10] = [1, 0, 1, 0, 3, 4, 5, 3, 7, 64001];
    let wt = WT::new(&mut data.clone());

    assert_eq!(wt.rank(1, 4), Some(2));
    assert_eq!(wt.rank(1, 0), Some(0));
    assert_eq!(wt.rank(8, 1), Some(0));
    assert_eq!(wt.rank(1, 9), Some(2));
    assert_eq!(wt.rank(7, 9), Some(1));
    assert_eq!(wt.rank(1, 10), Some(2));
    assert_eq!(wt.rank(1, 11), None); // too large position
    assert_eq!(wt.select(5, 0), Some(6));

    for (i, &v) in data.iter().enumerate() {
        let rank = wt.rank(v, i).unwrap();
        let s = wt.select(v, rank).unwrap();
        assert_eq!(s, i);
    }
}

#[test]
fn test_occs_range() {
    let data: [u8; 9] = [1, 0, 1, 0, 3, 4, 5, 3, 7];
    let wt = WT::new(&mut data.clone());

    // out-of-bounds ranges
    assert!(wt.occs_range(..data.len() + 1).is_none());
    assert!(wt.occs_range(data.len() - 1..data.len() + 1).is_none());

    // nonsense ranges
    assert!(wt.occs_range(5..4).is_none());
    assert!(wt.occs_range(2..0).is_none());

    // empty ranges
    assert_eq!(0, wt.occs_range(data.len()..).unwrap().count());
    assert_eq!(0, wt.occs_range(..0).unwrap().count());

    // unbounded
    let occs: Vec<_> = wt.occs_range(..).unwrap().collect();
    assert!(occs.is_sorted_by_key(|(s, _)| s));
    assert_eq!(occs, [(0, 2), (1, 2), (3, 2), (4, 1), (5, 1), (7, 1)]);

    // start bound
    let occs: Vec<_> = wt.occs_range(3..).unwrap().collect();
    assert!(occs.is_sorted_by_key(|(s, _)| s));
    assert_eq!(occs, [(0, 1), (3, 2), (4, 1), (5, 1), (7, 1)]);

    // end bound
    let occs: Vec<_> = wt.occs_range(..5).unwrap().collect();
    assert!(occs.is_sorted_by_key(|(s, _)| s));
    assert_eq!(occs, [(0, 2), (1, 2), (3, 1)]);

    // fully bounded
    let occs: Vec<_> = wt.occs_range(4..7).unwrap().collect();
    assert!(occs.is_sorted_by_key(|(s, _)| s));
    assert_eq!(occs, [(3, 1), (4, 1), (5, 1)]);

    // empty data
    let data: [u8; 0] = [];
    let wt = WT::new(&mut data.clone());
    assert_eq!(0, wt.occs_range(..).unwrap().count());
}

#[test]
fn test_occs_range_compressed() {
    let data: [u8; 9] = [1, 0, 1, 0, 3, 4, 5, 3, 7];
    let wt = HWT::new(&mut data.clone());

    // out-of-bounds ranges
    assert!(wt.occs_range(..data.len() + 1).is_none());
    assert!(wt.occs_range(data.len() - 1..data.len() + 1).is_none());

    // nonsense ranges
    assert!(wt.occs_range(5..4).is_none());
    assert!(wt.occs_range(2..0).is_none());

    // empty ranges
    assert_eq!(0, wt.occs_range(data.len()..).unwrap().count());
    assert_eq!(0, wt.occs_range(..0).unwrap().count());

    // unbounded
    let mut occs: Vec<_> = wt.occs_range(..).unwrap().collect();
    occs.sort_by_key(|p| p.0);
    assert_eq!(occs, [(0, 2), (1, 2), (3, 2), (4, 1), (5, 1), (7, 1)]);

    // start bound
    let mut occs: Vec<_> = wt.occs_range(3..).unwrap().collect();
    occs.sort_by_key(|p| p.0);
    assert_eq!(occs, [(0, 1), (3, 2), (4, 1), (5, 1), (7, 1)]);

    // end bound
    let mut occs: Vec<_> = wt.occs_range(..5).unwrap().collect();
    occs.sort_by_key(|p| p.0);
    assert_eq!(occs, [(0, 2), (1, 2), (3, 1)]);

    // fully bounded
    let mut occs: Vec<_> = wt.occs_range(4..7).unwrap().collect();
    occs.sort_by_key(|p| p.0);
    assert_eq!(occs, [(3, 1), (4, 1), (5, 1)]);

    // empty data
    let data: [u8; 0] = [];
    let wt = HWT::new(&mut data.clone());
    assert_eq!(0, wt.occs_range(..).unwrap().count());
}

/// Property-based test for occs_range on binary WaveletTree:
/// 1. sum of all occurrences == range length
/// 2. each symbol's count == rank(symbol, end) - rank(symbol, start)
#[test]
fn test_occs_range_properties() {
    let mut rng = rand::thread_rng();

    for sigma in [4, 16, 64, 256] {
        let sequence = gen_sequence(1000, sigma);
        let wt = WT::new(&mut sequence.clone());
        let n = sequence.len();

        // Test multiple random ranges
        for _ in 0..100 {
            let a = rng.gen_range(0..=n);
            let b = rng.gen_range(0..=n);
            let (start, end) = if a <= b { (a, b) } else { (b, a) };

            let occs: Vec<_> = wt.occs_range(start..end).unwrap().collect();

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
                let rank_end = wt.rank(*symbol, end).unwrap();
                let rank_start = wt.rank(*symbol, start).unwrap();
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
                let rank_end = wt.rank(s, end).unwrap_or(0);
                let rank_start = wt.rank(s, start).unwrap_or(0);
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

/// Property-based test for occs_range on Huffman-compressed binary WaveletTree
#[test]
fn test_occs_range_compressed_properties() {
    let mut rng = rand::thread_rng();

    for sigma in [4, 16, 64, 256] {
        let sequence = gen_sequence(1000, sigma);
        let wt = HWT::new(&mut sequence.clone());
        let n = sequence.len();

        // Test multiple random ranges
        for _ in 0..100 {
            let a = rng.gen_range(0..=n);
            let b = rng.gen_range(0..=n);
            let (start, end) = if a <= b { (a, b) } else { (b, a) };

            let occs: Vec<_> = wt.occs_range(start..end).unwrap().collect();

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
                let rank_end = wt.rank(*symbol, end).unwrap();
                let rank_start = wt.rank(*symbol, start).unwrap();
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
                let rank_end = wt.rank(s, end).unwrap_or(0);
                let rank_start = wt.rank(s, start).unwrap_or(0);
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

/// Test occs_range with large alphabets (σ > 256) on binary WaveletTree
/// Uses u16 symbols to support larger alphabet sizes
#[test]
fn test_occs_range_large_alphabet() {
    let mut rng = rand::thread_rng();

    for sigma in [512_u16, 1000, 4000, 16000] {
        // Generate random sequence with u16 symbols
        let sequence: Vec<u16> = (0..2000).map(|_| rng.gen_range(0..sigma)).collect();
        let wt = WT::new(&mut sequence.clone());
        let n = sequence.len();

        // Test multiple random ranges
        for _ in 0..50 {
            let a = rng.gen_range(0..=n);
            let b = rng.gen_range(0..=n);
            let (start, end) = if a <= b { (a, b) } else { (b, a) };

            let occs: Vec<_> = wt.occs_range(start..end).unwrap().collect();

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
                let rank_end = wt.rank(*symbol, end).unwrap();
                let rank_start = wt.rank(*symbol, start).unwrap();
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

            // Property 3: lexicographic ordering (for non-compressed WaveletTree)
            assert!(
                occs.is_sorted_by_key(|(s, _)| s),
                "σ={}: Results should be in lexicographic order",
                sigma
            );
        }
    }
}

/// Test occs_range with large alphabets (σ > 256) on Huffman-compressed WaveletTree
#[test]
fn test_occs_range_compressed_large_alphabet() {
    let mut rng = rand::thread_rng();

    for sigma in [512_u16, 1000, 4000, 16000] {
        // Generate random sequence with u16 symbols
        let sequence: Vec<u16> = (0..2000).map(|_| rng.gen_range(0..sigma)).collect();
        let wt = HWT::new(&mut sequence.clone());
        let n = sequence.len();

        // Test multiple random ranges
        for _ in 0..50 {
            let a = rng.gen_range(0..=n);
            let b = rng.gen_range(0..=n);
            let (start, end) = if a <= b { (a, b) } else { (b, a) };

            let occs: Vec<_> = wt.occs_range(start..end).unwrap().collect();

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
                let rank_end = wt.rank(*symbol, end).unwrap();
                let rank_start = wt.rank(*symbol, start).unwrap();
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

            // Note: HuffWaveletTree does NOT guarantee lexicographic order
        }
    }
}
