use super::*;

#[test]
fn test_select_in_word() {
    assert_eq!(select_in_word(1, 0), 0);
    assert_eq!(select_in_word(2, 0), 1);
    assert_eq!(select_in_word(63, 0), 0);
    assert_eq!(select_in_word(63, 1), 1);
    assert_eq!(select_in_word(63, 2), 2);
    assert_eq!(select_in_word(1024 - 2, 1), 2);

    let w = 0x5050505050505050_u64;
    assert_eq!(select_in_word(w, 0), 4);
    assert_eq!(select_in_word(w, 1), 6);
    assert_eq!(select_in_word(w, 2), 12);
    assert_eq!(select_in_word(w, 3), 14);
    assert_eq!(select_in_word(w, 4), 20);
    assert_eq!(select_in_word(w, 5), 22);
    assert_eq!(select_in_word(w, 6), 28);
    assert_eq!(select_in_word(w, 7), 30);
    assert_eq!(select_in_word(w, 8), 36);
    assert_eq!(select_in_word(w, 9), 38);
    assert_eq!(select_in_word(w, 10), 44);
    assert_eq!(select_in_word(w, 11), 46);
    assert_eq!(select_in_word(w, 12), 52);
    assert_eq!(select_in_word(w, 13), 54);
    assert_eq!(select_in_word(w, 14), 60);
    assert_eq!(select_in_word(w, 15), 62);
    assert_eq!(select_in_word(w, 16), 64);
}

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
