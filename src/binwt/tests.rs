use crate::{AccessUnsigned, OccsRangeUnsigned, RankUnsigned, SelectUnsigned, HWT, WT};

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
