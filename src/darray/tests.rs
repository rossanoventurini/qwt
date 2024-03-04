use super::*;
use crate::perf_and_test_utils::{gen_strictly_increasing_sequence, negate_vector};

#[test]
fn test_select1() {
    let bv = BitVector::default();
    let v: Vec<usize> = bv.ones().collect();
    assert!(v.is_empty());

    let vv: Vec<usize> = vec![0, 12, 33, 42, 55, 61, 62, 63, 128, 129, 254, 1026];
    let bv: BitVector = vv.iter().copied().collect();

    let da: DArray<false> = DArray::new(bv);

    for (i, &sel) in vv.iter().enumerate() {
        let res = da.select1(i);
        assert_eq!(res.unwrap(), sel);
    }
    let res = da.select1(vv.len());
    assert_eq!(res, None);

    let vv = gen_strictly_increasing_sequence(1024 * 4, 1 << 20);

    let bv: BitVector = vv.iter().copied().collect();

    let da: DArray<false> = DArray::new(bv);

    for (i, &sel) in vv.iter().enumerate() {
        let res = da.select1(i);
        assert_eq!(res.unwrap(), sel);
    }
}

#[test]
fn test_select0() {
    let bv = BitVector::default();
    let v: Vec<usize> = bv.ones().collect();
    assert!(v.is_empty());

    let vv: Vec<usize> = vec![0, 12, 33, 42, 55, 61, 62, 63, 128, 129, 254, 1026];
    let bv: BitVector = vv.iter().copied().collect();

    let da: DArray<true> = DArray::new(bv);

    for (i, &sel) in negate_vector(&vv).iter().enumerate() {
        let res = da.select0(i);
        assert_eq!(res.unwrap(), sel);
    }

    let vv = gen_strictly_increasing_sequence(1024 * 4, 1 << 20);
    let bv: BitVector = vv.iter().copied().collect();

    let da: DArray<true> = DArray::new(bv);

    for (i, &sel) in negate_vector(&vv).iter().enumerate() {
        let res = da.select0(i);
        assert_eq!(res.unwrap(), sel);
    }
}
