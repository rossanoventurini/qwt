use std::collections::HashMap;

use minimum_redundancy::{BitsPerFragment, Coding};

use crate::{AccessUnsigned, HQWT512};

#[test]
fn playgroud() {
    let sequence = String::from("aaaaaaaaccccbbd");

    //count symbol frequences
    let freqs: HashMap<char, usize> = sequence.chars().fold(HashMap::new(), |mut map, c| {
        *map.entry(c).or_insert(0) += 1;
        map
    });

    let huffman = Coding::from_frequencies(BitsPerFragment(1), freqs)
        .codes_for_values()
        .into_iter()
        // .map(|mut c| {
        //     c.1.content |= std::u32::MAX << c.1.len;
        //     c
        // })
        .collect::<HashMap<_, _>>();

    println!("{:?}", huffman);
}

#[test]
fn pg2() {
    let mut data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    let qwt = HQWT512::new(&mut data);

    println!("{:?}", qwt);

    unsafe {
        println!("{}", qwt.get_unchecked(6));
    }
    assert_eq!(qwt.len(), 8);
}
