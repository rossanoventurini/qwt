# QWT: Rust Quad Wavelet Tree

The [wavelet tree](https://en.wikipedia.org/wiki/Wavelet_Tree) [[1](#bib)] is a compact data structure that for a sequence $S$ of length $n$ over an alphabet of size $\sigma$ requires only $n\lceil\log \sigma \rceil (1+o(1))$ bits of space and can answer *rank* and *select* queries in $\Theta(\log \sigma)$ time.

A rank query counts the number of occurrences of a symbol up to a given position in the sequence. A select query finds the position in the sequence of a symbol with a given rank. These queries have applications in, e.g., compression, computational geometry, and pattern matching in the form of the backward search---the backbone of many compressed full-text indices.

This repository provides a very fast implementation of wavelet trees in Rust. A companion C++ implementation is available [here](https://github.com/MatteoCeregini/quad-wavelet-tree). More precisely, we implement here a variant called wavelet matrix [[2](#bib)], which gives a more elegant implementation.

The Quad Wavelet Tree (**QWT**) improves query performance by using a 4-ary tree instead of a binary tree as the basis of the wavelet tree. The 4-ary tree layout of a wavelet tree helps to halve the number of cache misses during queries and thus reduces the query latency.

An experimental evaluation shows that the quad wavelet tree improves the latency of access, rank, and select queries by a factor of $\approx$ 2 compared to other implementations of wavelet trees (e.g., the implementation in the widely used C++ Succinct Data Structure Library ([SDSL](https://github.com/simongog/sdsl-lite))). For more details, see [Benchmarks](#bench) and the paper [[3](#bib)].

## <a name="faste">Even faster rank query</a>

As previously highlighted, the Quad Wavelet Tree (QWT) enhances query performance by replacing the conventional binary tree structure in the wavelet tree with a 4-ary tree.

When working with moderately large sequences, the primary factor affecting query performance is the cost of the cache misses, which occur at each level of the wavelet tree. However, by utilizing a 4-ary tree structure, we effectively reduce the tree's height by half. Consequently, this reduction in height leads to a proportional decrease in the number of cache misses, resulting in the ~2x improvement in query time.

The **rank** queries can be further improved using a **small prediction model** designed to anticipate and pre-fetch the cache lines required for rank queries. This could give a further improvement up to a factor of 1.6 for rank query.

## <a name="bench">Benchmarks</a>
We report here a few experiments to compare our implementation with other state-of-the-art implementations.
The experiments use a single thread on a server machine with 8 Intel i9-9900KF cores with base frequencies of 3.60 GHz running Ubuntu 23.04 LTS kernel version 6.2.0-36. The code is compiled with Rust 1.73.0. Each core has a dedicated L1 cache of size 32 KiB, a dedicated L2 cache of size 256 KiB, a shared L3 cache of size 16 MiB, and 64 GiB of RAM.
A more detailed experimental evaluation (on different machines) can be found in [[3](#bib)].

The dataset, named [`Big English`](http://pages.di.unipi.it/rossano/big_english.gz), is the concatenation of all 35,750 English text files from the Gutenberg Project that are encoded in ASCII. Headers related to the project were removed, leaving only the actual text. The prefix of size 4 GiB was used. The text has an alphabet with 168 distinct symbols. Below we report details to download the dataset.


| Implementation                                  | *access* (ns) | *rank* (ns) | *select* (ns) | space (MiB) | Language |
| :-------------------------------------------- | ------------: | ----------: | ------------: | ----------: | :---------- |
| [SDSL 2.1.1](https://github.com/simongog/sdsl-lite) |          1178 |         1223 |          2900 |        6089 | C++ |
| [Pasta](https://github.com/pasta-toolbox)     |           1598 |         1729 |          2860 |       4112 | C++ |
| [Sucds 0.8.1](https://github.com/kampersanda/sucds) |           967 |         1015 |          2727 |        5376 | Rust |
| [Simple-SDS 0.3.1](https://github.com/jltsiren/simple-sds) |            933 |          1005 |          2558 |        6383 | Rust |
| Qwt256                                       |          516 |         542 |          1226 |       4616 | C++/Rust |
| Qwt256Pfs                                    |          515 |         363 |          1226 |        4626 | Rust |
| Qwt512                                       |          525 |         569 |          1196 |       4360 | C++/Rust |
| Qwt512Pfs                                    |          526 |         398 |         1197 |        4369 | Rust |

We note that the results for the rank query depend on how we generate the symbols to rank in the query set. Here for every rank query, we choose a symbol at random by following the distribution of symbols in the text, i.e., more frequent symbols are selected more frequently. All the data structures have more or less the same performance in ranking rare symbols. The reason is that the portion of the last layers for those rare symbols will likely fit in the cache.

There are four instances of our proposed wavelet trees, `Qwt256` and `Qwt512`, which are quad wavelet trees with block sizes of 256 and 512 symbols, respectively. The suffix `Pfs` in `Qwt256Pfs` and `Qwt512Pfs` indicates that they utilize additional space to store a predicting model, which can accelerate further 'rank' queries. Please refer to our full paper [[3](#bib)] for more details.

To run the experiments, we need to compile the binary executables with

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

This produces the two executables `perf_rs_quat_vector` and `perf_wavelet_tree` in `\target\release\`.

The former is used to measure the performance of QuadVectors, which are the building block of our implementation of Wavelet Trees. You can safely ignore it.

The latter measures the performance of a Quad Wavelet Tree built on a given input text.

We can now download and uncompress in the current directory the [Big English](http://pages.di.unipi.it/rossano/big_english.gz). Then, we take its prefix of length 4 GiB.

```bash
wget http://pages.di.unipi.it/rossano/big_english.gz
gunzip big_english.gz
head -c 4294967296 big_english > big_english.4GiB
```

The following command builds the wavelet trees (QWT 256 and 512 with or without prefetching support) on this input text and runs 10 million random *access*, *rank*, and *select* queries.

```bash
./target/release/perf_wavelet_tree --input-file big_english.4GiB --access --rank --select
```

We can use the flag `--test-correctness` to perform some extra tests for the correctness of the index. We can also specify the number of queries with `n_queries` (default is 10,000,000 queries).

The code measures the *latency* of the queries by forcing the input of each query to depend on the output of the previous one. This is consistent with the use of the queries in a real setting. For example, the more advanced queries supported by compressed text indexes (e.g., CSA or FM-index) decompose into several dependent queries on the underlying wavelet tree.

To repeat the comparison against other Rust libraries, please check out the branch `benchmark`.
Then, run the benchmark `perf_wt_bench` using the following command:

```bash
/target/release/perf_wt_bench --input-file big_english.4GiB --rank --select --access
```

## Examples

Run the following Cargo command in your project directory

```bash
cargo add qwt
```

to add the library.

Once the crate has been added, we can easily build a Quad Wavelet Tree with the following code. 

```rust
use qwt::QWT256;

let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];

let qwt = QWT256::from(data);

assert_eq!(qwt.len(), 8);
```

We can print the space usage of the wavelet tree with

```rust
use qwt::QWT256;
use qwt::SpaceUsage;

let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];

let qwt = QWT256::from(data);

println!("{}", qwt.space_usage_byte() );
```

A wavelet tree implements `FromIterator` and, thus, we can use `.collect()`.

```rust
use qwt::QWT256;

let qwt: QWT256<_> = (0..10_u8).into_iter().cycle().take(1000).collect();

assert_eq!(qwt.len(), 1000);
```

The data structure supports three operations:
- `get(i)` accesses the `i`-th symbols of the indexed sequence;
- `rank(c, i)` counts the number of occurrences of symbol `c` up to position `i` excluded;
- `select(c, i)` returns the position of the `i`-th occurrence of symbol `c`.

Here is an example of the three operations.

```rust
use qwt::{QWT256, AccessUnsigned, RankUnsigned, SelectUnsigned};

let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];

let qwt = QWT256::from(data);

assert_eq!(qwt.get(2), Some(1));
assert_eq!(qwt.get(3), Some(0));
assert_eq!(qwt.get(8), None);

assert_eq!(qwt.rank(1, 2), Some(1));
assert_eq!(qwt.rank(1, 0), Some(0));
assert_eq!(qwt.rank(3, 8), Some(1));
assert_eq!(qwt.rank(1, 9), None);

assert_eq!(qwt.select(1, 0), Some(0));
assert_eq!(qwt.select(0, 1), Some(3));
assert_eq!(qwt.select(4, 0), Some(5));
assert_eq!(qwt.select(1, 3), None);
```

In the following example, we use QWT to index a sequence over a larger alphabet.

```rust
use qwt::{QWT256, AccessUnsigned, RankUnsigned, SelectUnsigned};

let data = vec![1u32, 0, 1, 0, 2, 1000000, 5, 3];
let qwt = QWT256::from(data);

assert_eq!(qwt.get(2), Some(1));
assert_eq!(qwt.get(5), Some(1000000));
assert_eq!(qwt.get(8), None);
```

For more details, take a look at the [documentation](https://docs.rs/qwt/latest/qwt/).

Serialization and deserialization can be done with [`bincode`](https://docs.rs/bincode/latest/bincode/) as follows.

```rust
use std::fs;
use std::path::Path;

use qwt::{QWT256, AccessUnsigned};

let data = vec![1u8, 0, 1, 0, 2, 4, 5, 3];
let qwt = QWT256::from(data);

assert_eq!(qwt.get(2), Some(1));

// serialize
let serialized = bincode::serialize(&qwt).unwrap();

// write on file, if needed
let output_filename = "example.qwt256".to_string();
fs::write(Path::new(&output_filename), serialized).unwrap();

// read from file 
let serialized = fs::read(Path::new(&output_filename)).unwrap();

// deserialize
let qwt = bincode::deserialize::<QWT256<u8>>(&serialized).unwrap();

assert_eq!(qwt.get(2), Some(1));

```

We can index any sequence over any [num::traits::Unsigned](https://docs.rs/num/latest/num/traits/trait.Unsigned.html) integers. 
As the space usage depends on the largest value in the sequence, it could be worth remapping the values to remove "holes".

## <a name="bib">Bibliography</a>
1. Roberto Grossi, Ankur Gupta, and Jeffrey Scott Vitter. *High-order entropy-compressed text indexes.* In SODA, pages 841–850. ACM/SIAM, 2003.
2. Francisco Claude, Gonzalo Navarro, and Alberto Ordóñez Pereira. *The wavelet matrix: An efficient wavelet tree for large alphabets.* Information Systems, 47:15–32, 2015.
3. Matteo Ceregini, Florian Kurpicz, Rossano Venturini. *Faster Wavelet Trees with Quad Vectors*. Data Compression Conference (DCC), 2024.
----

Please cite the following [paper](http://arxiv.org/abs/2302.09239) if you use this code.

```bibtex
@inproceedings{QWT,
  author = {Matteo Ceregini, Florian Kurpicz, Rossano Venturini},
  title = {Faster Wavelet Trees with Quad Vectors},
  booktitle = {Data Compression Conference ({DCC})}
  publisher = {IEEE},
  year = {2024},
  doi = {10.48550/ARXIV.2302.09239},
  url = {http://arxiv.org/abs/2302.09239}
}
```
