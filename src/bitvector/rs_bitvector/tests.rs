use super::*;
use crate::perf_and_test_utils::{gen_strictly_increasing_sequence, negate_vector, TimingQueries};

/// Tests rank1 op by querying every position of a bit set to 1 in the binary vector
/// and the next position.
pub fn test_rank1(ds: &RSBitVector, bv: &BitVector) {
    let mut t = TimingQueries::new((bv.ones().count() * 2) + 1, 1);

    for (rank, pos) in bv.ones().enumerate() {
        t.start();
        let result = ds.rank1(pos);
        t.stop();
        // dbg!(pos, rank);
        assert_eq!(result, Some(rank));
        t.start();
        let result = ds.rank1(pos + 1);
        t.stop();
        //dbg!(pos + 1, rank);
        assert_eq!(result, Some(rank + 1));
    }
    t.start();
    let result = ds.rank1(bv.len() + 1);
    t.stop();
    assert_eq!(result, None);

    let data = t.get();
    println!("average time per query: {:?} ns", data.2);
}

#[test]
fn test_large_random_rank() {
    let vv = gen_strictly_increasing_sequence(1024 * 4, 1 << 20);
    let bv: BitVector = vv.iter().copied().collect();
    let rs = RSBitVector::new(bv);

    test_rank1(&rs, &rs.bv);
}

#[test]
fn playground() {
    let vv = gen_strictly_increasing_sequence(17000, 1 << 15);
    let bv: BitVector = vv.iter().copied().collect();
    let rs = RSBitVector::new(bv);

    println!("{:?}", rs.select1(0));
}

#[test]
fn playground2() {
    let vv: Vec<usize> = vec![4, 12, 33, 42, 64, 65, 512, 513, 620, 1030];
    let bv: BitVector = vv.iter().copied().collect();
    let rs = rs_bitvector::RSBitVector::new(bv);

    // for i in 0..rs.superblock_metadata.len(){
    //     println!("{}", rs.superblock_rank(i));
    // }

    let i = 1;
    println!("select: {:?}", rs.select1(i));
    println!("rank: {:?}", rs.rank1(rs.select1(i).unwrap()));
}

#[test]
fn playground3() {
    let vv: Vec<usize> = vec![3, 5, 8, 128, 129, 513];
    let bv: BitVector = vv.iter().copied().collect();
    let rs = rs_bitvector::RSBitVector::new(bv);

    // for i in 0..rs.superblock_metadata.len(){
    //     println!("{}", rs.superblock_rank(i));
    // }

    let i = 514;
    println!("rank1({}) | {}", i, rs.rank1(i).unwrap());
}

#[test]
fn test_select1() {
    let vv: Vec<usize> = vec![
        3, 5, 8, 128, 129, 513, 1000, 1024, 1025, 4096, 7500, 7600, 7630, 7680, 8000, 8001,
    ];
    // let vv: Vec<usize> = gen_strictly_increasing_sequence(1024 * 4, 1 << 14);
    let bv: BitVector = vv.iter().copied().collect();
    let rs = RSBitVector::new(bv);

    let mut t = TimingQueries::new(vv.len() - 2, 1);

    //NOTE: timing is higly dependent on u value

    for i in 0..vv.len() {
        t.start();
        let selected = rs.select1(i);
        t.stop();

        assert_eq!(selected, Some(vv[i]));
    }

    println!("select1 data: {:?}", t.get());
}

#[test]
fn test_select0() {
    // let vv: Vec<usize> = vec![3, 5, 8, 128, 129, 513];
    let vv = gen_strictly_increasing_sequence(1024, 1 << 20);
    let bv: BitVector = vv.iter().copied().collect();
    let rs = RSBitVector::new(bv);

    let zeros_vector = negate_vector(&vv);

    let mut t = TimingQueries::new(zeros_vector.len() - 2, 1);

    for i in 1..zeros_vector.len() {
        // println!("SELECTIAMO {}", i);
        t.start();
        let selected = rs.select0(i);
        t.stop();
        assert_eq!(selected, Some(zeros_vector[i]));
    }

    println!("select0 data: {:?}", t.get());
}
