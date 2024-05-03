use crate::{AccessUnsigned, HWT, WT};

#[test]
fn build_test() {
    let s: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 1, 1, 1, 1];

    let wt = WT::new(&mut s.clone());
    let hwt = HWT::new(&mut s.clone());

    println!("{:?} {:?}", wt.codes_decode, wt.lens);
    println!("{:?} {:?}", hwt.codes_decode, hwt.lens);

    for (i, &c) in s.iter().enumerate() {
        assert_eq!(Some(c), wt.get(i));
        assert_eq!(Some(c), hwt.get(i));
    }
}
