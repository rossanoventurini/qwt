use super::*;
use crate::perf_and_test_utils::{gen_strictly_increasing_sequence, negate_vector};

/// Tests rank1 op by querying every position of a bit set to 1 in the binary vector
/// and the next position.
pub fn test_rank1(ds: &RSWide, bv: &BitVector) {
    for (rank, pos) in bv.ones().enumerate() {
        let result = ds.rank1(pos);
        // dbg!(pos, rank);
        assert_eq!(result, Some(rank));
        let result = ds.rank1(pos + 1);
        //dbg!(pos + 1, rank);
        assert_eq!(result, Some(rank + 1));
    }
    let result = ds.rank1(bv.len() + 1);
    assert_eq!(result, None);
}

#[test]
fn test_large_random_rank() {
    let vv = gen_strictly_increasing_sequence(1024 * 4, 1 << 15);
    let bv: BitVector = vv.iter().copied().collect();
    let rs = RSWide::new(bv);

    test_rank1(&rs, &rs.bv);
}

#[test]
fn test_select1() {
    let vv: Vec<usize> = vec![
        3, 5, 8, 128, 129, 513, 1000, 1024, 1025, 4096, 7500, 7600, 7630, 7680, 8000, 8001, 10000,
    ];
    let bv: BitVector = vv.iter().copied().collect();
    let rs = RSWide::new(bv);

    for (i, &el) in vv.iter().enumerate() {
        let selected = rs.select1(i);
        assert_eq!(selected, Some(el));
    }
}

#[test]
fn test_select0() {
    let vv: Vec<usize> = vec![
        3, 5, 8, 128, 129, 513, 1000, 1024, 1025, 4096, 7500, 7600, 7630, 7680, 8000, 8001, 10000,
    ];
    let bv: BitVector = vv.iter().copied().collect();
    let rs = RSWide::new(bv);

    let zeros_vector = negate_vector(&vv);

    for (i, &el) in zeros_vector.iter().enumerate() {
        let selected = rs.select0(i);
        assert_eq!(selected, Some(el));
    }
}

#[test]
fn test_random_select1() {
    let vv: Vec<usize> = gen_strictly_increasing_sequence(10000, 1 << 22);
    let bv: BitVector = vv.iter().copied().collect();
    let rs = RSWide::new(bv);

    for (i, &el) in vv.iter().enumerate() {
        let selected = rs.select1(i);
        assert_eq!(selected, Some(el));
    }
}

#[test]
fn test_random_select0() {
    let vv: Vec<usize> = gen_strictly_increasing_sequence(10000, 1 << 22);
    let bv: BitVector = vv.iter().copied().collect();
    let rs = RSWide::new(bv);

    let zeros_vector = negate_vector(&vv);

    for (i, &el) in zeros_vector.iter().enumerate() {
        let selected = rs.select0(i);
        assert_eq!(selected, Some(el));
    }
}
