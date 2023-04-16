//! The crate provides an efficient implementation of [Wavelet Trees](https://en.wikipedia.org/wiki/Wavelet_Tree).
//!
//! A wavelet tree [[1](#bib)] is a compact data structure that for a text of length
//! $n$ over an alphabet of size $\sigma$ requires only $n\lceil\log \sigma \rceil (1+o(1))$
//! bits of space and can answer `rank` and `select` queries in $\Theta(\log \sigma)$ time.
//!
//! Given a static sequence `S[0,n-1]`, a wavelet tree indexes the sequence `S` and
//! supports three operations:
//! - `get(i)` returns S[i];
//! - `rank(c, i)` returns the number of occurrences of the symbol `c` in the prefix S[0...i-1];
//! - `select(c, i)` returns the position in S of the `i`th occurrence of the symbol `c`.
//!  
//! Our implementation of Wavelet Tree [2] improves query performance by using a 4-ary
//! tree instead of a binary tree as the basis of the wavelet tree.
//! The 4-ary tree layout of a wavelet tree helps to halve the number of cache misses
//! during queries and thus reduces the query latency. This way we are roughly 2 times
//! faster than other existing implementations (e.g., SDSL).
//!
//! ## <a name="bib">Bibliography</a>
//! 1. Roberto Grossi, Ankur Gupta, and Jeffrey Scott Vitter. *High-order entropy-compressed text indexes.* In SODA, pages 841–850. ACM/SIAM, 2003.
//! 2. Matteo Ceregini, Florian Kurpicz, Rossano Venturini. Faster Wavelet Trees with Quad Vectors. Arxiv, 2023.

//#![feature(stdsimd)]

pub mod perf_and_test_utils;
pub mod qvector;
pub use qvector::QVector;
pub use qvector::QVectorBuilder;

pub mod utils;

pub use qvector::rs_qvector::RSQVector;
pub use qvector::rs_qvector::RSQVector256;
pub use qvector::rs_qvector::RSQVector512;

pub mod quadwt;
pub use quadwt::QWaveletTree;

pub type QWT256<T> = QWaveletTree<T, RSQVector256>;
pub type QWT512<T> = QWaveletTree<T, RSQVector512>;

use num_traits::Unsigned;

/// An interface for supporting `get` queries over an `Unsigned` alphabet.
pub trait AccessUnsigned {
    type Item: Unsigned;

    /// Returns the symbol at position `i`.
    fn get(&self, i: usize) -> Option<Self::Item>;

    /// Returns the symbol at position `i`.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item;
}

/// An interface for supporting `rank` queries over an `Unsigned` alphabet.
pub trait RankUnsigned: AccessUnsigned {
    /// Returns the number of occurrences in the indexed sequence of `symbol` up to
    /// position `i` excluded.
    fn rank(&self, symbol: Self::Item, i: usize) -> Option<usize>;

    /// Returns the number of occurrences in the indexed sequence of `symbol` up to
    /// position `i` excluded. The function does not check boundaries.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    unsafe fn rank_unchecked(&self, symbol: Self::Item, i: usize) -> usize;
}

/// An interface for supporting `select` queries over an `Unsigned` alphabet.
pub trait SelectUnsigned: AccessUnsigned {
    /// Returns the position in the indexed sequence of the `i`th occurrence of
    /// `symbol`.
    /// We start counting from 1, so that `select(symbol, 1)` refers to the first
    /// occurrence of `symbol`. `select(symbol, 0)` returns `None`.
    fn select(&self, symbol: Self::Item, i: usize) -> Option<usize>;

    /// Returns the position in the indexed sequence of the `i`th occurrence of
    /// `symbol`.
    /// We start counting from 1, so that `select(symbol, 1)` refers to the first
    /// occurrence of `symbol`.
    ///
    /// # Safety
    /// Calling this method if the `i`th occurrence of `symbol` does not exist is undefined behavior.
    unsafe fn select_unchecked(&self, symbol: Self::Item, i: usize) -> usize;
}

/// An interface for supporting `get` queries over the alphabet [0..3].
pub trait AccessQuad {
    /// Returns the symbol at position `i`.
    fn get(&self, i: usize) -> Option<u8>;

    /// Returns the symbol at position `i`.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    unsafe fn get_unchecked(&self, i: usize) -> u8;
}

pub trait RankQuad {
    /// Returns the number of occurrences in the indexed sequence of `symbol` up to
    /// position `i` excluded.
    fn rank(&self, symbol: u8, i: usize) -> Option<usize>;

    /// Returns the number of occurrences in the indexed sequence of `symbol` up to
    /// position `i` excluded.The function does not check boundaries.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index or with a symbol larger than
    /// 3 is undefined behavior.
    unsafe fn rank_unchecked(&self, symbol: u8, i: usize) -> usize;
}

pub trait SelectQuad {
    /// Returns the position in the indexed sequence of the `i`th occurrence of
    /// `symbol`.
    /// We start counting from 1, so that `select(symbol, 1)` refers to the first
    /// occurrence of `symbol`. `select(symbol, 0)` returns `None`.
    fn select(&self, symbol: u8, i: usize) -> Option<usize>;

    /// Returns the position in the indexed sequence of the `i`th occurrence of
    /// `symbol`.
    /// We start counting from 1, so that `select(symbol, 1)` refers to the first
    /// occurrence of `symbol`.
    ///
    /// # Safety
    /// Calling this method if the `i`th occurrence of `symbol` does not exist or with a symbol larger than 3 is undefined behavior.
    unsafe fn select_unchecked(&self, symbol: u8, i: usize) -> usize;
}

/// An interface to report the space usage of a data structure.
pub trait SpaceUsage {
    /// Gives the space usage of the data structure in bytes.
    fn space_usage_bytes(&self) -> usize;

    /// Gives the space usage of the data structure in Kbytes.
    fn space_usage_kbytes(&self) -> f64 {
        let bytes = self.space_usage_bytes();
        (bytes as f64) / (1024_f64)
    }

    /// Gives the space usage of the data structure in Mbytes.
    fn space_usage_mbytes(&self) -> f64 {
        let bytes = self.space_usage_bytes();
        (bytes as f64) / ((1024 * 1024) as f64)
    }

    /// Gives the space usage of the data structure in Gbytes.
    fn space_usage_gbytes(&self) -> f64 {
        let bytes = self.space_usage_bytes();
        (bytes as f64) / ((1024 * 1024 * 1024) as f64)
    }
}

/// An interface for the operations that a quad vector implementation needs
/// to provide to support Wavelet Tree queries.
pub trait WTSupport: AccessQuad + RankQuad + SelectQuad {
    /// Returns the number of occurrences of `symbol` in the indexed sequence,
    /// `None` if `symbol` is larger than the largest symbol, i.e., `symbol` is not valid.  
    fn occs(&self, symbol: u8) -> Option<usize>;

    /// Returns the number of occurrences of `symbol` in the indexed sequence.
    ///
    /// # Safety
    /// Calling this method if the `i`th occurrence of `symbol`
    /// larger than the largest symbol is undefined behavior.
    unsafe fn occs_unchecked(&self, symbol: u8) -> usize;

    /// Returns the number of occurrences of all the symbols smaller than the
    /// input `symbol`, `None` if `symbol` is larger than the largest symbol,
    /// i.e., `symbol` is not valid.  
    fn occs_smaller(&self, symbol: u8) -> Option<usize>;

    /// Returns the number of occurrences of all the symbols smaller than the input
    /// `symbol` in the indexed sequence.
    ///
    /// # Safety
    /// Calling this method if the `i`th occurrence of `symbol` larger than the
    /// largest symbol is undefined behavior.
    unsafe fn occs_smaller_unchecked(&self, symbol: u8) -> usize;

    /// Prefetches counter of superblock and blocks containing the position `pos`.
    fn prefetch_info(&self, pos: usize);

    /// Prefetches data containing the position `pos`.
    fn prefetch_data(&self, pos: usize);
}

use std::mem;
/// TODO: Improve and generalize. Incorrect if T is not a primitive type.
/// It is also error-prone to implement this for every data structure.
/// Make a macro to go over the member of a struct!
impl<T> SpaceUsage for Vec<T>
where
    T: SpaceUsage + Copy,
{
    fn space_usage_bytes(&self) -> usize {
        if !self.is_empty() {
            mem::size_of::<Self>() + self.get(0).unwrap().space_usage_bytes() * self.capacity()
        } else {
            mem::size_of::<Self>() + mem::size_of::<T>() * self.capacity()
        }
    }
}

impl<T> SpaceUsage for Box<[T]>
where
    T: SpaceUsage + Copy,
{
    fn space_usage_bytes(&self) -> usize {
        if !self.is_empty() {
            mem::size_of::<Self>() + self.get(0).unwrap().space_usage_bytes() * self.len()
        } else {
            mem::size_of::<Self>()
        }
    }
}

macro_rules! impl_space_usage {
    ($($t:ty),*) => {
        $(impl SpaceUsage for $t {
            fn space_usage_bytes(&self) -> usize {
                mem::size_of::<Self>()
            }
        })*
    }
}

impl_space_usage![bool, i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize, f32, f64];
