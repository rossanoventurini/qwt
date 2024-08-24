use crate::{AccessUnsigned, RankUnsigned, SelectUnsigned, HWT, WT};

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
