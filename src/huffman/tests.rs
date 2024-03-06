use super::*;

#[test]
fn get_freqs_test(){
    let v = Vec::from("aaaaaaaabbbbcdx".as_bytes());

    println!("{:?}", get_freqs_from_vec(&v));
}

#[test]
fn build_huffman_tree_test(){
    let v = Vec::from("xxxxxxxxxxxxxxxxxxxxxxxxxxrehgwrth5yhwhqergw".as_bytes());

    let tree = HuffmanTree::build_from_freqs(get_freqs_from_vec(&v));

    //println!("{:?}", tree);

    println!("{:?}", tree.codebook());

}