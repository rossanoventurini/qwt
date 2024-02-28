use super::*;
use crate::perf_and_test_utils::gen_strictly_increasing_sequence;

/// Tests rank1 op by querying every position of a bit set to 1 in the binary vector
/// and the next position.
pub fn test_rank1(ds: &RSNarrow, bv: &BitVector) {
    for (rank, pos) in bv.ones().enumerate() {
        let result = ds.rank1(pos);
        assert_eq!(result, Some(rank));
        let result = ds.rank1(pos + 1);
        dbg!(pos + 1, rank);
        assert_eq!(result, Some(rank + 1));
    }
    let result = ds.rank1(bv.len() + 1);
    assert_eq!(result, None);
}

// Empty bit vector
#[test]
fn test_empty() {
    let bv = BitVector::new();
    let rs = RSNarrow::new(bv);

    assert_eq!(rs.rank1(0), None);
    assert_eq!(rs.rank1(100), None);
}

// Tests a bit vector where the last bit of l2 is set
#[test]
fn test_l2_bound() {
    let vv: Vec<usize> = vec![0, 12, 33, 42, 55, 61, 62, 63, 128, 129, 254, 511];
    let bv: BitVector = vv.iter().copied().collect();
    let rs = RSNarrow::new(bv);

    test_rank1(&rs, &rs.bv);
}

#[test]
fn test_n() {
    let vv: Vec<usize> = vec![0, 12, 33, 42, 55, 61, 62, 63];
    let bv: BitVector = vv.iter().copied().collect();

    let rs = RSNarrow::new(bv);

    assert_eq!(rs.rank1(64), Some(8));
}

// Tests a bit vector where the last bit of l1 is set
#[test]
fn test_l1_bound() {
    let vv: Vec<usize> = vec![0, 12, 33, 42, 55, 61, 62, 63, 128, 129, 254, 512 * 128 - 1];
    let bv: BitVector = vv.iter().copied().collect();

    let rs = RSNarrow::new(bv);

    test_rank1(&rs, &rs.bv);
}

// Tests a random bit vector of size 1<<20. This gives 16 blocks in l1.
#[test]
fn test_large_random() {
    let vv = gen_strictly_increasing_sequence(1024 * 4, 1 << 20);
    let bv: BitVector = vv.iter().copied().collect();
    let rs = RSNarrow::new(bv);

    test_rank1(&rs, &rs.bv);
}

#[test]
fn test_select() {
    let vv: Vec<usize> = vec![3, 5, 8, 128, 129, 513];
    let bv: BitVector = vv.iter().copied().collect();
    let rs = RSNarrow::new(bv);

    println!("{:?}", rs.bv);
    println!("{:?}", rs.block_rank_pairs);


    let i = 5;
    let selected = rs.select1(i).unwrap();
    println!("select1({}) = {}", i, selected);

    let j = selected;
    println!("rank1({}) = {}", j, rs.rank1(j).unwrap());


}