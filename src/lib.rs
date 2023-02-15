pub mod perf_and_test_utils;
pub mod qvector;
pub use qvector::QVector;

pub mod rs_qvector;
pub use rs_qvector::RSQVector;
pub use rs_qvector::RSQVectorP256;
pub use rs_qvector::RSQVectorP512;

pub mod quadwt;
pub use quadwt::QWaveletTree;
pub mod utils;

pub type QWaveletTreeP256 = QWaveletTree<u8, RSQVectorP256>;
pub type QWaveletTreeP512 = QWaveletTree<u8, RSQVectorP512>;

use num_traits::Unsigned;

/// An interface for supporting access queries over an `Unsigned` alphabet.
/// The operation `get(i)` returns the symbol at position `i`.
pub trait AccessUnsigned {
    type Item: Unsigned;
    fn get(&self, i: usize) -> Option<Self::Item>;
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item;
}

/// An interface for supporting rank queries over an `Unsigned` alphabet.
/// The operation `rank(symbol, i)` returns the number of
/// occurrences in the indexed sequence of `symbol` up to
/// position `i` excluded.
pub trait RankUnsigned: AccessUnsigned {
    fn rank(&self, symbol: Self::Item, i: usize) -> Option<usize>;
    unsafe fn rank_unchecked(&self, symbol: Self::Item, i: usize) -> usize;
}

/// An interface for supporting select queries over an `Unsigned` alphabet.
/// The operation `select(symbol, i)` returns the position in the indexed
/// sequence of the `i`th occurrence of `symbol`.  
/// We start counting from 1, so that `select(symbol, 1)` refers to the first
/// occurrence of `symbol`. `select(symbol, 0)` should be `None`.
pub trait SelectUnsigned: AccessUnsigned {
    fn select(&self, symbol: Self::Item, i: usize) -> Option<usize>;
    unsafe fn select_unchecked(&self, symbol: Self::Item, i: usize) -> usize;
}

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

use std::mem;
/// TODO: Improve and generalize. Incorrect if T is not primitive type
/// It is also error prone to implement this for every data structure.
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
