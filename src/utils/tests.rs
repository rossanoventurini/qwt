use super::*;

use num_traits::AsPrimitive;

#[test]
fn test_stable_partition_of_4() {
    let mut v: Vec<u8> = vec![1, 2, 3, 0, 2, 2, 2, 3, 3, 0, 0, 0, 1, 3, 2, 1];

    let mut vv = v.clone();
    let shift = 0;
    stable_partition_of_4(&mut vv, shift);

    v.sort_by(|a, b| {
        // stable sorting by current 2 bits
        let a_bits: u8 = AsPrimitive::<u8>::as_(*a >> shift) & 3;
        let b_bits: u8 = AsPrimitive::<u8>::as_(*b >> shift) & 3;
        a_bits.cmp(&b_bits)
    });

    assert_eq!(vv, v);
}
