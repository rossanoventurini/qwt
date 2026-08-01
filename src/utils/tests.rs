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

#[test]
fn stable_partition_of_4_into_matches_bucket_partition() {
    let input = vec![0u16, 17, 2, 31, 16, 3, 18, 1, 30, 19, 4, 29];
    let shift = 2;
    let mut counts = [0usize; 4];
    for &symbol in &input {
        counts[((symbol >> shift) & 3) as usize] += 1;
    }

    let mut expected = input.clone();
    stable_partition_of_4(&mut expected, shift);
    let mut output = vec![0; input.len()];
    stable_partition_of_4_into(&input, shift, counts, &mut output);

    assert_eq!(output, expected);
}

#[test]
fn stable_partition_of_4_with_codes_into_matches_bucket_partition() {
    let codes = vec![
        PrefixCode { content: 0, len: 2 },
        PrefixCode { content: 1, len: 2 },
        PrefixCode {
            content: 0b1000,
            len: 4,
        },
        PrefixCode {
            content: 0b1100,
            len: 4,
        },
        PrefixCode {
            content: 0b010000,
            len: 6,
        },
        PrefixCode { content: 0, len: 6 },
    ];
    let input = vec![0u8, 2, 5, 1, 4, 3, 2, 0, 5, 4, 3, 1];
    let shift = 2;
    let mut counts = [0usize; 5];
    for &symbol in &input {
        let code = &codes[symbol as usize];
        let bucket = if code.len <= shift {
            4
        } else {
            ((code.content >> (code.len - shift)) & 3) as usize
        };
        counts[bucket] += 1;
    }

    let mut expected = input.clone();
    stable_partition_of_4_with_codes(&mut expected, shift as usize, &codes);
    let mut output = vec![0; input.len()];
    stable_partition_of_4_with_codes_into(&input, shift as usize, &codes, counts, &mut output);

    assert_eq!(output, expected);
}
