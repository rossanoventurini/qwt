use std::collections::HashMap;

use minimum_redundancy::{BitsPerFragment, Coding};

#[test]
fn playgroud(){
    let sequence = String::from("aaaaaaaaccccbbd");

    //count symbol frequences
    let freqs:HashMap<char, usize> = sequence.chars().fold(HashMap::new(), |mut map, c| {
        *map.entry(c).or_insert(0) += 1;
        map
    });

    let huffman: HashMap<char, minimum_redundancy::Code> = Coding::from_frequencies(BitsPerFragment(1), freqs)
        .codes_for_values()
        .into_iter()
        .map(|mut c| {
            c.1.content |= std::u32::MAX << c.1.len;
            c
        }).collect();

    println!("{:?}", huffman);
}
