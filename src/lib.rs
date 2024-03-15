#![doc = include_str!("../README.md")]

pub mod perf_and_test_utils;
pub mod qvector;
pub use qvector::QVector;
pub use qvector::QVectorBuilder;

pub mod bitvector;
pub use bitvector::rs_bitvector::RSBitVector;
pub use bitvector::rs_narrow::RSNarrow;
pub use bitvector::BitVector;
pub use bitvector::BitVectorMut;

pub mod huffman;

pub mod utils;

pub use qvector::rs_qvector::RSQVector;
pub use qvector::rs_qvector::RSQVector256;
pub use qvector::rs_qvector::RSQVector512;

pub mod quadwt;
pub use quadwt::QWaveletTree;
pub use quadwt::WTIndexable;

pub type QWT256<T> = QWaveletTree<T, RSQVector256>;
pub type QWT512<T> = QWaveletTree<T, RSQVector512>;

// Quad Wavelet tree with support for prefetching
pub type QWT256Pfs<T> = QWaveletTree<T, RSQVector256, true>;
pub type QWT512Pfs<T> = QWaveletTree<T, RSQVector512, true>;

use num_traits::Unsigned;

/// A trait fro the support `get` query over an `Unsigned` alphabet.
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

/// A trait for the support of `rank` query over an `Unsigned` alphabet.
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

/// A trait for the support of ``select` query over an `Unsigned` alphabet.
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

/// A trait for the support of `get` query over the binary alphabet.
pub trait AccessBin {
    /// Returns the bit at the given position `i`,
    /// or [`None`] if ```i``` is out of bounds.
    fn get(&self, i: usize) -> Option<bool>;

    /// Returns the symbol at the given position `i`.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    unsafe fn get_unchecked(&self, i: usize) -> bool;
}

/// A trait for the support of `rank` query over the binary alphabet.
pub trait RankBin {
    /// Returns the number of zeros in the indexed sequence up to
    /// position `i` excluded.
    #[inline]
    fn rank0(&self, i: usize) -> Option<usize> {
        if let Some(k) = self.rank1(i) {
            return Some(i - k);
        }

        None
    }

    /// Returns the number of ones in the indexed sequence up to
    /// position `i` excluded.
    fn rank1(&self, i: usize) -> Option<usize>;

    /// Returns the number of ones in the indexed sequence up to
    /// position `i` excluded. `None` if the position is out of bound.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    unsafe fn rank1_unchecked(&self, i: usize) -> usize;

    /// Returns the number of zeros in the indexed sequence up to
    /// position `i` excluded.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline]
    unsafe fn rank0_unchecked(&self, i: usize) -> usize {
        i - self.rank1_unchecked(i)
    }
}

// TODO: Add SelectBin trait when select will be implemented
pub trait SelectBin {
    /// Returns the position `pos` such that the element is `1` and rank1(pos) = i.
    /// Returns `None` if the data structure has no such element i >= maximum rank1
    fn select1(&self, i: usize) -> Option<usize>;

    /// Returns the position `pos` such that the element is `1` and rank1(pos) = i.
    ///
    /// # Safety
    /// This method doesnt check that such element exists
    /// Calling this method with an i >= maximum rank1 is undefined behaviour.
    unsafe fn select1_unchecked(&self, i: usize) -> usize;

    /// Returns the position `pos` such that the element is `0` and rank0(pos) = i.
    /// Returns `None` if the data structure has no such element (i >= maximum rank0 in the struct)
    fn select0(&self, i: usize) -> Option<usize>;

    /// Returns the position `pos` such that the element is `0` and rank0(pos) = i.
    ///
    /// # Safety
    /// This method doesnt check that such element exists
    /// Calling this method with an `i >= maximum rank0` is undefined behaviour.
    unsafe fn select0_unchecked(&self, i: usize) -> usize;
}

/// A trait for the support of `get` query over the alphabet [0..3].
pub trait AccessQuad {
    /// Returns the symbol at position `i`, `None` if the position is out of bound.
    fn get(&self, i: usize) -> Option<u8>;

    /// Returns the symbol at position `i`.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    unsafe fn get_unchecked(&self, i: usize) -> u8;
}

/// A trait for the support of `rank` query over the alphabet [0..3].
pub trait RankQuad {
    /// Returns the number of occurrences in the indexed sequence of `symbol` up to
    /// position `i` excluded. `None` if the position is out of bound.
    fn rank(&self, symbol: u8, i: usize) -> Option<usize>;

    /// Returns the number of occurrences in the indexed sequence of `symbol` up to
    /// position `i` excluded. The function does not check boundaries.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index or with a symbol larger than
    /// 3 is undefined behavior.
    unsafe fn rank_unchecked(&self, symbol: u8, i: usize) -> usize;
}

/// A trait for the support of `select` query over the alphabet [0..3].
pub trait SelectQuad {
    /// Returns the position in the indexed sequence of the `i`th occurrence of `symbol`
    /// (0-indexed, mening the first occurrence is obtained using `i = 0`).
    fn select(&self, symbol: u8, i: usize) -> Option<usize>;

    /// Returns the position in the indexed sequence of the `i`th occurrence of `symbol`.
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
    fn space_usage_byte(&self) -> usize;

    /// Gives the space usage of the data structure in KiB.
    #[allow(non_snake_case)]
    fn space_usage_KiB(&self) -> f64 {
        let bytes = self.space_usage_byte();
        (bytes as f64) / (1024_f64)
    }

    /// Gives the space usage of the data structure in MiB.
    #[allow(non_snake_case)]
    fn space_usage_MiB(&self) -> f64 {
        let bytes = self.space_usage_byte();
        (bytes as f64) / ((1024 * 1024) as f64)
    }

    /// Gives the space usage of the data structure in GiB.
    #[allow(non_snake_case)]
    fn space_usage_GiB(&self) -> f64 {
        let bytes = self.space_usage_byte();
        (bytes as f64) / ((1024 * 1024 * 1024) as f64)
    }
}

/// A trait for the operations that a quad vector implementation needs
/// to provide to be used in a Quad Wavelet Tree.
pub trait WTSupport: AccessQuad + RankQuad + SelectQuad {
    /// Returns the number of occurrences of `symbol` in the indexed sequence,
    /// `None` if `symbol` is larger than 3, i.e., `symbol` is not valid.  
    fn occs(&self, symbol: u8) -> Option<usize>;

    /// Returns the number of occurrences of `symbol` in the indexed sequence.
    ///
    /// # Safety
    /// Calling this method if the `symbol` is larger than 3 (i.e., `symbol` is not valid)
    /// is undefined behavior.
    unsafe fn occs_unchecked(&self, symbol: u8) -> usize;

    /// Returns the number of occurrences of all the symbols smaller than the
    /// input `symbol`, `None` if `symbol` is larger than 3,
    /// i.e., `symbol` is not valid.  
    fn occs_smaller(&self, symbol: u8) -> Option<usize>;

    /// Returns the rank of `symbol` up to the block that contains the position
    /// `i`.
    ///
    /// # Safety
    /// Calling this method if the `symbol` is larger than 3 of
    /// if the position `i` is out of bound is undefined behavior.
    unsafe fn rank_block_unchecked(&self, symbol: u8, i: usize) -> usize;

    /// Returns the number of occurrences of all the symbols smaller than the input
    /// `symbol` in the indexed sequence.
    ///
    /// # Safety
    /// Calling this method if the `symbol` is larger than 3 is undefined behavior.
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
    fn space_usage_byte(&self) -> usize {
        if !self.is_empty() {
            mem::size_of::<Self>() + self.get(0).unwrap().space_usage_byte() * self.capacity()
        } else {
            mem::size_of::<Self>() + mem::size_of::<T>() * self.capacity()
        }
    }
}

impl<T> SpaceUsage for Box<[T]>
where
    T: SpaceUsage + Copy,
{
    fn space_usage_byte(&self) -> usize {
        if !self.is_empty() {
            mem::size_of::<Self>() + self.get(0).unwrap().space_usage_byte() * self.len()
        } else {
            mem::size_of::<Self>()
        }
    }
}

macro_rules! impl_space_usage {
    ($($t:ty),*) => {
        $(impl SpaceUsage for $t {
            fn space_usage_byte(&self) -> usize {
                mem::size_of::<Self>()
            }
        })*
    }
}

impl_space_usage![bool, i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize, f32, f64];
