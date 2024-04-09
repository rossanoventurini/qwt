use std::collections::HashMap;

use minimum_redundancy::{BitsPerFragment, Coding};
use num_traits::AsPrimitive;

use crate::{perf_and_test_utils::TimingQueries, AccessUnsigned, HQWT256, HQWT512};

use super::craft_wm_codes;

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

    // println!("{:?}", qwt);

    unsafe {
        println!("{}", qwt.get_unchecked(1));
    }
    assert_eq!(qwt.len(), 8);
}

#[test]
fn playgroud3() {
    // let mut sequence: Vec<u8> = String::from("aaaaaaaaccccbbdAAAABBEFawegf")
    //     .bytes()
    //     .collect();
    let mut sequence = vec![
        1u32, 1, 1, 1, 5, 6, 8, 34, 34, 65, 7, 8, 9, 34, 2, 45, 7, 21, 22, 23, 34, 25, 26, 3, 234,
        255, 234, 234, 234, 234, 234, 234,
    ];
    // sequence.reverse();
    let seq_check = sequence.clone();

    let hqwt = HQWT256::new(sequence.as_mut_slice());

    println!("hqwt levels: {:?}", hqwt.n_levels);

    for i in 0..hqwt.len() {
        println!(
            "-------------------\nindex {} | should be {} | {:?}",
            i, seq_check[i], hqwt.codes_encode[seq_check[i] as usize]
        );
        assert_eq!(hqwt.get(i), Some(seq_check[i]));
    }
    assert_eq!(hqwt.get(hqwt.len()), None);
}

#[test]
fn playgroud1() {
    let s = "tobeornottobethatisthequestion";

    let freqs = s.chars().fold(HashMap::new(), |mut map, c| {
        *map.entry(c as u8).or_insert(0u32) += 1;
        map
    });

    let mut lengths = Coding::from_frequencies(BitsPerFragment(2), freqs).code_lengths();
    println!("{:?}", lengths);

    let codes = craft_wm_codes(&mut lengths);

    println!(
        "{}, {:?}",
        s.chars().nth(0).unwrap(),
        codes[s.chars().nth(0).unwrap() as usize]
    );
    println!(
        "{}, {:?}",
        s.chars().nth(1).unwrap(),
        codes[s.chars().nth(1).unwrap() as usize]
    );
    println!(
        "{}, {:?}",
        s.chars().nth(2).unwrap(),
        codes[s.chars().nth(2).unwrap() as usize]
    );
    println!(
        "{}, {:?}",
        s.chars().nth(3).unwrap(),
        codes[s.chars().nth(3).unwrap() as usize]
    );
    println!(
        "{}, {:?}",
        s.chars().nth(5).unwrap(),
        codes[s.chars().nth(5).unwrap() as usize]
    );
    println!(
        "{}, {:?}",
        s.chars().nth(15).unwrap(),
        codes[s.chars().nth(15).unwrap() as usize]
    );
    println!(
        "{}, {:?}",
        s.chars().nth(17).unwrap(),
        codes[s.chars().nth(17).unwrap() as usize]
    );
    println!(
        "{}, {:?}",
        s.chars().nth(20).unwrap(),
        codes[s.chars().nth(20).unwrap() as usize]
    );
    println!(
        "{}, {:?}",
        s.chars().nth(25).unwrap(),
        codes[s.chars().nth(25).unwrap() as usize]
    );
    println!(
        "{}, {:?}",
        s.chars().nth(22).unwrap(),
        codes[s.chars().nth(22).unwrap() as usize]
    );
    println!(
        "{}, {:?}",
        s.chars().nth(23).unwrap(),
        codes[s.chars().nth(23).unwrap() as usize]
    );
}
