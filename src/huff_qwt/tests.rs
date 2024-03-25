use std::collections::HashMap;

use minimum_redundancy::{BitsPerFragment, Coding};
use num_traits::AsPrimitive;

use crate::{perf_and_test_utils::TimingQueries, AccessUnsigned, HQWT256, HQWT512};

#[test]
fn playgroud() {
    let sequence = String::from("aaaaaaaaccccbbdef");

    //count symbol frequences
    let freqs: HashMap<char, usize> = sequence.chars().fold(HashMap::new(), |mut map, c| {
        *map.entry(c.as_()).or_insert(0) += 1;
        map
    });

    let huffman = Coding::from_frequencies(BitsPerFragment(2), freqs);
    // .codes_for_values()
    // .into_iter()
    // // .map(|mut c| {
    // //     c.1.content |= std::u32::MAX << c.1.len;
    // //     c
    // // })
    // .collect::<HashMap<_, _>>();

    let mut t = TimingQueries::new(1, 1);
    t.start();
    let mut d = huffman.decoder();
    t.stop();
    println!("decoder creation took {:?}", t.get());

    let mut t1 = TimingQueries::new(1, 1);
    let fragments: Vec<u32> = vec![0, 0, 1];
    t1.start();
    let val = d.decode(&mut fragments.into_iter());
    t1.stop();

    println!("{:?} | decoding took {:?}", val, t1.get());

    println!("{:?}", huffman.codes_for_values());
}

#[test]
fn pg2() {
    let mut data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
    let qwt = HQWT512::new(&mut data);

    println!("{:?}", qwt);

    unsafe {
        println!("{}", qwt.get_unchecked(1));
    }
    assert_eq!(qwt.len(), 8);
}

#[test]
fn playgroud3() {
    let mut sequence = String::from("aaaaaaaaccccbbdAAAABBEFawegf")
        .bytes()
        .collect::<Vec<_>>();
    let seq_check = sequence.to_vec();

    let hqwt = HQWT256::new(sequence.as_mut_slice());

    println!("hqwt levels: {:?}", hqwt.n_levels);

    for i in 0..hqwt.len() {
        println!("index {} | should be {}", i, seq_check[i]);
        assert_eq!(hqwt.get(i), Some(seq_check[i]));
    }
    assert_eq!(hqwt.get(hqwt.len()), None);
}
