use super::*;
use crate::perf_and_test_utils::gen_strictly_increasing_sequence;

/// Tests rank1 op by querying every position of a bit set to 1 in the binary vector
/// and the next position.
pub fn test_rank1(ds: &RSBitVector, bv: &BitVector) {
    for (rank, pos) in bv.ones().enumerate() {
        let result = ds.rank1(pos);
        dbg!(pos, rank);
        assert_eq!(result, Some(rank));
        let result = ds.rank1(pos + 1);
        dbg!(pos + 1, rank);
        assert_eq!(result, Some(rank + 1));
    }
    let result = ds.rank1(bv.len() + 1);
    assert_eq!(result, None);
}

#[test]
fn test_large_random() {
    let vv = gen_strictly_increasing_sequence(1024 * 4, 1 << 20);
    let bv: BitVector = vv.iter().copied().collect();
    let rs = RSBitVector::new(bv);

    test_rank1(&rs, &rs.bv);
}

#[test]
fn playground() {
    let vv = gen_strictly_increasing_sequence(17000, 1 << 15);
    let bv: BitVector = vv.iter().copied().collect();
    let rs = rs_bitvector::RSBitVector::new(bv);

    println!("{:?}", rs.select_samples)
}

#[test]
fn playground2() {
    let vv: Vec<usize> = vec![0, 12, 33, 42, 64, 65, 512, 513, 620, 1030];
    let bv: BitVector = vv.iter().copied().collect();
    let rs = rs_bitvector::RSBitVector::new(bv);

    // for i in 0..rs.superblock_metadata.len(){
    //     println!("{}", rs.superblock_rank(i));
    // }

    let i = 4;
    println!("rank1 {} | {}", i, rs.sub_block_rank(i));
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
    // let vv: Vec<usize> = vec![3, 5, 8, 128, 129, 513, 1000, 1024, 1025, 4096, 7500, 7600, 7630, 7680, 8000, 8001];
    let vv: Vec<usize> = gen_strictly_increasing_sequence(1024 * 4, 1 << 20);
    let bv: BitVector = vv.iter().copied().collect();
    let rs = RSBitVector::new(bv);

    // println!("{:?}", rs.bv);
    // println!("{:?}", rs.select_samples);

    for i in 1..vv.len() {
        // println!("SELECTIAMO {}", i);
        assert_eq!(rs.select1(i), Some(vv[i]));
    }

    // //wtf rabk for 9
    // let i = 12;
    // let selected = rs.select1(i);
    // println!("select1({}) = {:?}", i, selected);

    // let j = selected.unwrap();
    // println!("rank1({}) = {:?}", j, rs.rank1(j));
}

//FIX SELECT 0
#[test]
fn test_select0() {
    let vv: Vec<usize> = vec![3, 5, 8, 128, 129, 513];
    let bv: BitVector = vv.iter().copied().collect();
    let rs = RSBitVector::new(bv);

    println!("{:?}", rs.bv);
    println!("{:?}", rs.superblock_metadata);

    let i = 500;
    let selected = rs.select0(i).unwrap();
    println!("select0({}) = {}", i, selected);

    let j = selected;
    println!("rank0({}) = {}", j, rs.rank0(j).unwrap());
}
