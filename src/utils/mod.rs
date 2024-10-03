//! The module provides low-level utilities to perform
//! bitwise operations, aligned allocation, and so on.
use num_traits::{AsPrimitive, PrimInt, Unsigned};
use std::collections::{HashMap, HashSet};
use std::ops::Shr;

#[allow(non_snake_case)]
pub fn prefetch_read_NTA<T>(data: &[T], offset: usize) {
    let _p = unsafe { data.as_ptr().add(offset) as *const i8 };

    #[cfg(all(feature = "prefetch", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{_mm_prefetch, _MM_HINT_NTA};

        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_NTA};

        unsafe {
            _mm_prefetch(_p, _MM_HINT_NTA);
        }
    }

    #[cfg(all(feature = "prefetch", target_arch = "aarch64"))]
    {
        use core::arch::aarch64::{_prefetch, _PREFETCH_LOCALITY0, _PREFETCH_READ};

        unsafe {
            _prefetch(_p, _PREFETCH_READ, _PREFETCH_LOCALITY0);
        }
    }
}

// Required by select64
const K_SELECT_IN_BYTE: [u8; 2048] = [
    8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    8, 8, 8, 1, 8, 2, 2, 1, 8, 3, 3, 1, 3, 2, 2, 1, 8, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    8, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    8, 6, 6, 1, 6, 2, 2, 1, 6, 3, 3, 1, 3, 2, 2, 1, 6, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    8, 7, 7, 1, 7, 2, 2, 1, 7, 3, 3, 1, 3, 2, 2, 1, 7, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    7, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    7, 6, 6, 1, 6, 2, 2, 1, 6, 3, 3, 1, 3, 2, 2, 1, 6, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    8, 8, 8, 8, 8, 8, 8, 2, 8, 8, 8, 3, 8, 3, 3, 2, 8, 8, 8, 4, 8, 4, 4, 2, 8, 4, 4, 3, 4, 3, 3, 2,
    8, 8, 8, 5, 8, 5, 5, 2, 8, 5, 5, 3, 5, 3, 3, 2, 8, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2,
    8, 8, 8, 6, 8, 6, 6, 2, 8, 6, 6, 3, 6, 3, 3, 2, 8, 6, 6, 4, 6, 4, 4, 2, 6, 4, 4, 3, 4, 3, 3, 2,
    8, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3, 2, 6, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2,
    8, 8, 8, 7, 8, 7, 7, 2, 8, 7, 7, 3, 7, 3, 3, 2, 8, 7, 7, 4, 7, 4, 4, 2, 7, 4, 4, 3, 4, 3, 3, 2,
    8, 7, 7, 5, 7, 5, 5, 2, 7, 5, 5, 3, 5, 3, 3, 2, 7, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2,
    8, 7, 7, 6, 7, 6, 6, 2, 7, 6, 6, 3, 6, 3, 3, 2, 7, 6, 6, 4, 6, 4, 4, 2, 6, 4, 4, 3, 4, 3, 3, 2,
    7, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3, 2, 6, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 4, 8, 4, 4, 3,
    8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 5, 8, 5, 5, 3, 8, 8, 8, 5, 8, 5, 5, 4, 8, 5, 5, 4, 5, 4, 4, 3,
    8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 3, 8, 8, 8, 6, 8, 6, 6, 4, 8, 6, 6, 4, 6, 4, 4, 3,
    8, 8, 8, 6, 8, 6, 6, 5, 8, 6, 6, 5, 6, 5, 5, 3, 8, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3,
    8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 3, 8, 8, 8, 7, 8, 7, 7, 4, 8, 7, 7, 4, 7, 4, 4, 3,
    8, 8, 8, 7, 8, 7, 7, 5, 8, 7, 7, 5, 7, 5, 5, 3, 8, 7, 7, 5, 7, 5, 5, 4, 7, 5, 5, 4, 5, 4, 4, 3,
    8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 3, 8, 7, 7, 6, 7, 6, 6, 4, 7, 6, 6, 4, 6, 4, 4, 3,
    8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 3, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 5, 8, 5, 5, 4,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 4,
    8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 5, 8, 8, 8, 6, 8, 6, 6, 5, 8, 6, 6, 5, 6, 5, 5, 4,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 4,
    8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 5, 8, 8, 8, 7, 8, 7, 7, 5, 8, 7, 7, 5, 7, 5, 5, 4,
    8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 4,
    8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 5, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 5,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 5,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6,
    8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 5,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7,
];

/// Computes `select(1, k)` operation on a 64-bit word.
///
/// It computes the zero-based position of the (k+1)-th
/// bit set in the word.
///
/// For example, if the word is ```(10101010)^4```, select for k = 0
/// returns 1 and select for k=1 returns 3.
///
/// # Examples
///
/// ```
/// use qwt::utils::select_in_word;
///
/// unsafe {
///     let word = 2863311530; // word = (10101010)^4
///     let res = select_in_word(word, 0);
///     assert_eq!(res, 1_u32);
///
///     let res = select_in_word(word, 1);
///     assert_eq!(res, 3_u32);
/// }
/// ```
/// # Notes
///
/// The current implementation uses the broadword selection algorithm
/// by Vigna \[1\], improved by Gog and Petri \[2\] and Vigna \[3\].
/// Facebook's Folly implementation \[4\].
/// - \[1\] Sebastiano Vigna. Broadword Implementation of Rank/Select Queries. WEA, 200
/// - \[2\] Simon Gog, Matthias Petri. Optimized succinct data structures for massive data. Softw. Pract. Exper., 2014
/// - \[3\] Sebastiano Vigna. MG4J 5.2.1. <http://mg4j.di.unimi.it/>
/// - \[4\] Facebook Folly library: <https://github.com/facebook/folly>

#[inline(always)]
pub fn select_in_word(word: u64, k: u64) -> u32 {
    // use core::arch::x86_64::_pdep_u64;
    // let mask = std::u64::MAX << k;
    // return unsafe{ _pdep_u64(mask, word).trailing_zeros() };
    let k_ones_step4 = 0x1111111111111111_u64;
    let k_ones_step8 = 0x0101010101010101_u64;
    let k_lambdas_step8 = 0x8080808080808080_u64;

    let mut s = word;
    s = s - ((s & (0xA * k_ones_step4)) >> 1);
    s = (s & (0x3 * k_ones_step4)) + ((s >> 2) & (0x3 * k_ones_step4));
    s = (s + (s >> 4)) & (0xF * k_ones_step8);
    let byte_sums = s.wrapping_mul(k_ones_step8);
    // byte_sums contains 8 bytes. byte_sums[j] is the number of bits in word set to 1 up to (and including) jth byte. These are values in [0, 64]
    let k_step8 = k * k_ones_step8;

    // geq_k_step8 contains 8 bytes, the jth byte geq_k_step8[j] == 128 iff byte_sums[j] <= k
    let geq_k_step8 = ((k_step8 | k_lambdas_step8) - byte_sums) & k_lambdas_step8;

    let place = geq_k_step8.count_ones() * 8;

    if place == 64 {
        return 64;
    }
    let byte_rank = k - (((byte_sums << 8) >> place) & 0xFF_u64);

    place + K_SELECT_IN_BYTE[(((word >> place) & 0xFF_u64) | (byte_rank << 8)) as usize] as u32
}

#[inline(always)]
pub fn select_in_word_u128(word: u128, k: u64) -> u32 {
    let first = word as u64;

    let kp = first.count_ones();
    if kp as u64 > k {
        select_in_word(first, k)
    } else {
        64 + select_in_word((word >> 64) as u64, k - kp as u64)
    }
}

/// Compute popcnt for a slice of N words.
#[inline(always)]
pub fn popcnt_wide<const N: usize>(data: &[u64]) -> usize {
    let mut res: usize = 0;
    for &word in data.iter().take(N) {
        res += word.count_ones() as usize;
    }
    res
}

use std::mem;

use crate::quadwt::huffqwt::PrefixCode;

#[repr(C, align(64))]
struct AlignToSixtyFour([u8; 64]);

/// Returns a 64-byte aligned vector of T with at least the
/// given capacity.
///
/// Todo: make this safe by checking invariants.
///
/// # Safety
/// See Safety of [Vec::Vec::from_raw_parts](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.from_raw_parts).
pub unsafe fn get_64byte_aligned_vector<T>(capacity: usize) -> Vec<T> {
    assert!(mem::size_of::<T>() <= mem::size_of::<AlignToSixtyFour>());
    assert!(mem::size_of::<AlignToSixtyFour>() % mem::size_of::<T>() == 0); // must divide otherwise fro raw parts below doesnt work

    let n_units = (capacity * mem::size_of::<T>() + mem::size_of::<AlignToSixtyFour>() - 1)
        / mem::size_of::<AlignToSixtyFour>();
    let mut aligned: Vec<AlignToSixtyFour> = Vec::with_capacity(n_units);

    let ptr = aligned.as_mut_ptr();
    let len_units = aligned.len();
    let cap_units = aligned.capacity();

    mem::forget(aligned);

    Vec::from_raw_parts(
        ptr as *mut T,
        len_units * mem::size_of::<AlignToSixtyFour>() / mem::size_of::<T>(),
        cap_units * mem::size_of::<AlignToSixtyFour>() / mem::size_of::<T>(),
    )
}

/// Computes the position of the most significant bit in v.
pub fn msb<T>(v: T) -> u32
where
    T: PrimInt,
{
    if v == T::zero() {
        return 0;
    }
    (std::mem::size_of::<T>() * 8 - 1) as u32 - v.leading_zeros()
}

/// Remaps the alphabet of the input text with a new alphabet of consecutive symbols.
/// New alphabet preserves the original relative order among symbols.
/// Returns the alphabet size.
pub fn text_remap(input: &mut [u8]) -> usize {
    let mut unique: Vec<u8> = input
        .iter_mut()
        .map(|c| *c)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    unique.sort();
    let remap = unique
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c, i))
        .collect::<HashMap<_, _>>();

    for c in input.iter_mut() {
        *c = *remap.get(c).unwrap() as u8;
    }

    remap.len()
}

/// Utility function to partition values in `sequence` by the two bits
/// that we obtain by shifting `shift` bits to the right.
/// This is used by the construction of WaveletTree.
pub fn stable_partition_of_4<T>(sequence: &mut [T], shift: usize)
where
    T: Unsigned + PrimInt + Ord + Shr<usize> + AsPrimitive<usize>,
    usize: AsPrimitive<T>,
{
    let mut vecs = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    for &a in sequence.iter() {
        let two_bits: usize = (a.as_() >> shift) & 3;
        vecs[two_bits as usize].push(a);
    }

    let mut pos = 0;
    for i in 0..4 {
        sequence[pos..pos + vecs[i].len()].copy_from_slice(&(vecs[i][..]));
        pos += vecs[i].len()
    }
}

pub fn stable_partition_of_4_with_codes<T>(sequence: &mut [T], shift: usize, codes: &[PrefixCode])
where
    T: Unsigned + PrimInt + Ord + Shr<usize> + AsPrimitive<usize>,
    usize: AsPrimitive<T>,
{
    let mut vecs = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()]; // the fifth one contains symbols we dont want to partition (leaf at un upper level)

    for &a in sequence.iter() {
        let code = &codes[a.as_() as usize];
        if code.len <= shift as u32 {
            //we dont care about this symbol (already taken care of)
            vecs[4].push(a);
        } else {
            //we partition as normal
            let two_bits = (code.content >> (code.len - shift as u32)) & 3;
            vecs[two_bits as usize].push(a);
        }
    }

    let mut pos = 0;
    for i in 0..5 {
        sequence[pos..pos + vecs[i].len()].copy_from_slice(&(vecs[i][..]));
        pos += vecs[i].len()
    }
}

pub fn stable_partition_of_2_with_codes<T>(sequence: &mut [T], shift: usize, codes: &[PrefixCode])
where
    T: Unsigned + PrimInt + Ord + Shr<usize> + AsPrimitive<usize>,
    usize: AsPrimitive<T>,
{
    let mut vecs = [Vec::new(), Vec::new(), Vec::new()];

    for &a in sequence.iter() {
        let code = &codes[a.as_() as usize];
        if code.len <= shift as u32 {
            //we dont care about this symbol (already taken care of)
            vecs[2].push(a);
        } else {
            let bit = (code.content >> (code.len - shift as u32)) & 1;
            vecs[bit as usize].push(a);
        }
    }

    let mut pos = 0;
    for i in 0..3 {
        sequence[pos..pos + vecs[i].len()].copy_from_slice(&(vecs[i][..]));
        pos += vecs[i].len()
    }
}

pub fn stable_partition_of_2<T>(sequence: &mut [T], shift: usize)
where
    T: Unsigned + PrimInt + Ord + Shr<usize> + AsPrimitive<usize>,
    usize: AsPrimitive<T>,
{
    let mut vecs = [Vec::new(), Vec::new()];

    for &a in sequence.iter() {
        let bit = (a >> shift).as_() & 1;
        vecs[bit as usize].push(a);
    }

    let mut pos = 0;
    for i in 0..2 {
        sequence[pos..pos + vecs[i].len()].copy_from_slice(&(vecs[i][..]));
        pos += vecs[i].len()
    }
}

#[cfg(test)]
mod tests;
