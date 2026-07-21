use super::*;
use crate::perf_and_test_utils::gen_sequence;
use crate::{OccsRangeUnsigned, RSQVector512, RankUnsigned, QWT256};
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

    // nonsense ranges
    assert!(qwt
        .occs_range(std::ops::Range { start: 5, end: 4 })
        .is_none());
    assert!(qwt
        .occs_range(std::ops::Range { start: 2, end: 0 })
        .is_none());

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

// ── range_next_value ─────────────────────────────────────────────────────

fn rnv_scan_oracle(seq: &[u8], range: std::ops::Range<usize>, target: u8) -> Option<u8> {
    let mut best = None;
    for &v in &seq[range] {
        if v >= target {
            best = Some(match best {
                Some(b) if b <= v => b,
                _ => v,
            });
            if best == Some(target) {
                return Some(target);
            }
        }
    }
    best
}

#[test]
fn test_range_next_value_edge_cases() {
    let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    let qwt = QWT256::from(data.clone());
    // empty
    assert_eq!(qwt.range_next_value(0..0, 0), None);
    // singleton
    assert_eq!(qwt.range_next_value(0..1, 0), Some(1));
    assert_eq!(qwt.range_next_value(0..1, 1), Some(1));
    assert_eq!(qwt.range_next_value(0..1, 2), None);
    // full / below min / exact / gap / max / above
    assert_eq!(qwt.range_next_value(0..8, 0), Some(0));
    assert_eq!(qwt.range_next_value(0..8, 3), Some(3));
    assert_eq!(qwt.range_next_value(0..8, 5), Some(5));
    assert_eq!(qwt.range_next_value(0..8, 6), None);
    // subrange without 0
    assert_eq!(qwt.range_next_value(4..8, 0), Some(2));
    assert_eq!(qwt.range_next_value(4..8, 3), Some(3));
    assert_eq!(qwt.range_next_value(4..8, 4), Some(4));
    // vs scan oracle
    for lo in 0..=8 {
        for hi in lo..=8 {
            for t in 0..=8 {
                let n = qwt.range_next_value(lo..hi, t);
                let s = qwt.range_next_value_scan(lo..hi, t);
                let o = rnv_scan_oracle(&data, lo..hi, t);
                assert_eq!(n, s, "native vs scan {lo}..{hi} t={t}");
                assert_eq!(n, o, "native vs oracle {lo}..{hi} t={t}");
            }
        }
    }
}

#[test]
fn test_range_next_value_duplicates_and_backtrack() {
    // Many duplicates; force backtracking when target branch exists but no ≥ suffix.
    let data: Vec<u8> = vec![7, 7, 7, 1, 1, 3, 3, 5, 5, 0, 2, 4, 6, 6];
    let qwt = QWT256::from(data.clone());
    for lo in 0..=data.len() {
        for hi in lo..=data.len() {
            for t in 0..=10 {
                assert_eq!(
                    qwt.range_next_value(lo..hi, t),
                    rnv_scan_oracle(&data, lo..hi, t),
                    "lo={lo} hi={hi} t={t}"
                );
            }
        }
    }
}

#[test]
fn test_range_next_value_random_vs_scan() {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0xE57B_0001);
    for trial in 0..40 {
        let n = rng.random_range(1..400);
        let sigma = rng.random_range(2..200u32);
        let seq: Vec<u8> = (0..n).map(|_| rng.random_range(0..sigma) as u8).collect();
        let qwt = QWT256::from(seq.clone());
        for _ in 0..80 {
            let a = rng.random_range(0..=n);
            let b = rng.random_range(0..=n);
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            let t = rng.random_range(0..=(sigma + 5)) as u8;
            let native = qwt.range_next_value(lo..hi, t);
            let scan = qwt.range_next_value_scan(lo..hi, t);
            let oracle = rnv_scan_oracle(&seq, lo..hi, t);
            assert_eq!(native, scan, "trial {trial} {lo}..{hi} t={t}");
            assert_eq!(native, oracle, "trial {trial} oracle");
        }
    }
}

#[test]
fn test_range_next_value_large_alphabet_u32() {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0xE57B_0032);
    let n = 2000;
    let sigma = 50_000u32;
    let seq: Vec<u32> = (0..n).map(|_| rng.random_range(0..sigma)).collect();
    let qwt = QWT256::from(seq.clone());
    for _ in 0..200 {
        let a = rng.random_range(0..=n);
        let b = rng.random_range(0..=n);
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let t = rng.random_range(0..=sigma + 10);
        let native = qwt.range_next_value(lo..hi, t);
        // local scan oracle for u32
        let mut best = None;
        for &v in &seq[lo..hi] {
            if v >= t {
                best = Some(match best {
                    Some(b) if b <= v => b,
                    _ => v,
                });
                if best == Some(t) {
                    break;
                }
            }
        }
        assert_eq!(native, best, "{lo}..{hi} t={t}");
    }
}

// ── range_distinct_iter ──────────────────────────────────────────────────

#[test]
fn test_range_distinct_iter_matches_occs_and_rnv() {
    use crate::{OccsRangeUnsigned, QWT256};
    let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    let qwt = QWT256::from(data.clone());

    // full range vs occs_range
    let rdi: Vec<_> = qwt.range_distinct_iter(0..8).collect();
    let mut occs: Vec<_> = qwt.occs_range(0..8).unwrap().collect();
    occs.sort_by_key(|(s, _)| *s);
    assert_eq!(rdi, occs);

    // vs repeated RNV
    let mut via_rnv = Vec::new();
    let mut t = 0u8;
    while let Some(v) = qwt.range_next_value(0..8, t) {
        let c = data.iter().filter(|&&x| x == v).count();
        via_rnv.push((v, c));
        t = v.saturating_add(1);
        if t == 0 {
            break;
        }
    }
    assert_eq!(rdi, via_rnv);

    // empty / singleton / subrange
    assert_eq!(qwt.range_distinct_iter(0..0).count(), 0);
    assert_eq!(
        qwt.range_distinct_iter(0..1).collect::<Vec<_>>(),
        vec![(1, 1)]
    );
    let sub: Vec<_> = qwt.range_distinct_iter(4..8).collect();
    assert_eq!(sub, vec![(2, 1), (3, 1), (4, 1), (5, 1)]);
}

#[test]
fn test_range_distinct_iter_random_vs_occs() {
    use crate::{OccsRangeUnsigned, QWT256};
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xE57C);
    for _ in 0..40 {
        let n = rng.random_range(1..120usize);
        let sigma = rng.random_range(1..40u8);
        let data: Vec<u8> = (0..n).map(|_| rng.random_range(0..sigma)).collect();
        let qwt = QWT256::from(data.clone());
        for _ in 0..20 {
            let lo = rng.random_range(0..=n);
            let hi = rng.random_range(lo..=n);
            let rdi: Vec<_> = qwt.range_distinct_iter(lo..hi).collect();
            let mut occs: Vec<_> = qwt.occs_range(lo..hi).unwrap().collect();
            occs.sort_by_key(|(s, _)| *s);
            assert_eq!(rdi, occs, "n={n} {lo}..{hi}");
            // sum of counts == range length
            assert_eq!(rdi.iter().map(|(_, c)| *c).sum::<usize>(), hi - lo);
        }
    }
}

#[test]
fn test_range_distinct_iter_large_sigma_u32() {
    use crate::{OccsRangeUnsigned, QWT256};
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
    let n = 200usize;
    let sigma = 5000u32;
    let data: Vec<u32> = (0..n).map(|_| rng.random_range(0..sigma)).collect();
    let qwt = QWT256::from(data);
    let rdi: Vec<_> = qwt.range_distinct_iter(0..n).collect();
    let mut occs: Vec<_> = qwt.occs_range(0..n).unwrap().collect();
    occs.sort_by_key(|(s, _)| *s);
    assert_eq!(rdi, occs);
    assert_eq!(rdi.iter().map(|(_, c)| *c).sum::<usize>(), n);
}
