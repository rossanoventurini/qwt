use super::*;

#[test]
fn get_freqs_test() {
    let v = Vec::from("aaaaaaaabbbbcdx".as_bytes());

    println!("{:?}", get_freqs_from_vec(&v));
}

#[test]
fn build_huffman_tree_test() {
    let v = Vec::from("xxxxxxxxxxxxxxxxxxxxxxxxxxrehgwrth5yhwhqergw".as_bytes());

    let tree = HuffmanTree::build_from_freqs(get_freqs_from_vec(&v));

    //println!("{:?}", tree);

    println!("{:?}", tree.codebook());
}

#[test]
fn canonical_test() {
    let v = Vec::from("xxxxxxxxxxxxxxxxxxxxxxxxxxrehgwrthfybrtwy4yAQTKIyhwhqergw".as_bytes());
    let tree = HuffmanTree::build_from_freqs(get_freqs_from_vec(&v));
    let mut cb = tree.codebook();

    println!("Codebook\n {:?}", cb);

    println!("Canonical codes\n {:?}", get_canonical_code(&mut cb));
}
