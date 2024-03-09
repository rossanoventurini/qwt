use super::*;
use crate::perf_and_test_utils::gen_strictly_increasing_sequence;

#[test]
fn playground() {
    let vv = gen_strictly_increasing_sequence(1678, (1 << 14) + 3);
    let bv: BitVector = vv.iter().copied().collect();
    let rs = rs_bitvector::RSBitVector::new(bv);

    for i in 0..rs.superblock_metadata.len() {
        println!("superblock {} | {}", i, rs.superblock_rank(i));
    }

    for i in 0..rs.superblock_metadata.len() - 1 {
        for j in 0..8 {
            println!(
                "superblock {} | subblock {} | {}",
                i,
                j,
                rs.sub_block_rank(i * 8 + j)
            );
        }
    }
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
    println!("subblock {} | {}", i, rs.sub_block_rank(i));
}
