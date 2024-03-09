use std::collections::{BinaryHeap, HashMap};

use std::cmp::Reverse;
use std::fmt::Debug;

///alias for a hash map containing pairs `symbol: frequency`
pub type FreqTable = HashMap<u8, u64>;

#[derive(Eq, PartialEq)]
pub struct HuffmanCode {
    symbol: u8,    //symbol encoded
    length: usize, //length of the code
    repr: u128,    //representation in bits
}

impl Debug for HuffmanCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HuffmanCode{{ symbol: {:#x}, length: {}, repr `",
            self.symbol, self.length
        )?;
        for i in (0..self.length).rev() {
            let chr = match !((self.repr & 1 << i) == 0) {
                false => "0",
                true => "1",
            };
            write!(f, "{}", chr)?;
        }
        write!(f, "`}}")
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct HuffmanTree {
    freq: u64,
    symbol: Option<u8>,
    left: Option<Box<HuffmanTree>>,
    right: Option<Box<HuffmanTree>>,
}

impl Ord for HuffmanTree {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.freq.cmp(&other.freq)
    }
}

impl PartialOrd for HuffmanTree {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl HuffmanTree {
    fn leaf(freq: u64, symbol: u8) -> Self {
        HuffmanTree {
            freq: freq,
            symbol: Some(symbol),
            left: None,
            right: None,
        }
    }

    fn internal_node(freq: u64, left: HuffmanTree, right: HuffmanTree) -> Self {
        HuffmanTree {
            freq: freq,
            symbol: None,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }

    ///builds a HuffmanTree using the given `freq_table`
    pub fn build_from_freqs(freq_table: FreqTable) -> Self {
        //build leaf nodes
        let mut q = BinaryHeap::new();
        for (sym, freq) in freq_table {
            q.push(Reverse(HuffmanTree::leaf(freq, sym)));
        }

        while q.len() > 1 {
            let left = q.pop().unwrap().0;
            let right = q.pop().unwrap().0;
            q.push(Reverse(HuffmanTree::internal_node(
                left.freq + right.freq,
                left,
                right,
            )));
        }

        q.pop().unwrap().0
    }

    pub fn codebook(&self) -> Vec<HuffmanCode> {
        fn collect(output: &mut Vec<HuffmanCode>, tree: &HuffmanTree, indent: usize, bits: u128) {
            if let Some(value) = tree.symbol {
                output.push(HuffmanCode {
                    symbol: value,
                    length: indent,
                    repr: bits,
                });
            }

            if let Some(left) = &tree.left {
                collect(output, left.as_ref(), indent + 1, bits << 1);
            }

            if let Some(right) = &tree.right {
                collect(output, right.as_ref(), indent + 1, bits << 1 | 1);
            }
        }

        let mut codebook = Vec::<HuffmanCode>::new();
        collect(&mut codebook, &self, 0, 0);
        codebook
    }
}

///Returns a frequency table of the items in `input`
pub fn get_freqs_from_vec(input: &Vec<u8>) -> FreqTable {
    input.iter().fold(FreqTable::new(), |mut map, c| {
        *map.entry(*c).or_insert(0) += 1;
        map
    })
}

pub fn get_canonical_code(input: &mut Vec<HuffmanCode>) -> Vec<HuffmanCode> {
    input.sort_by(|c1, c2| c1.length.partial_cmp(&c2.length).unwrap());

    let mut canonical_codebook = Vec::new();

    let mut repr = 0;
    let mut length = 0;

    //add canonical codes
    for code in input {
        while length < code.length {
            repr <<= 1;
            length += 1;
        }

        canonical_codebook.push(HuffmanCode {
            symbol: code.symbol,
            length: length,
            repr: !repr, //we take the negative so we have the longest codes on the right
        });

        repr += 1;
    }

    canonical_codebook
}

#[cfg(test)]
mod tests;
