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

    let huffman = Coding::from_frequencies(BitsPerFragment(1), freqs);    

    println!("{:?}", huffman.codes_for_values());
}