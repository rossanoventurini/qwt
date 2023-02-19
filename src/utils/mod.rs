//! The module provides low-level utilities to perform
//! bitwise operations, aligned allocation, and so on.
use core::arch::x86_64::_pdep_u64;

use num_traits::{AsPrimitive, PrimInt, Unsigned};
use std::collections::{HashMap, HashSet};
use std::ops::Shr;

/// Computes select_1(k) on a 64-bit word.
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
#[inline(always)]
pub unsafe fn select_in_word(word: u64, k: u64) -> u32 {
    let mask = std::u64::MAX << k;
    _pdep_u64(mask, word).trailing_zeros() 
}

#[inline(always)]
pub unsafe fn select_in_word_u128(word: u128, k: u64) -> u32 {
    let first = word as u64;

    let kp = first.count_ones();

    if kp as u64 > k {
        select_in_word(first, k)
    } else {
        kp + select_in_word((word >> 64) as u64, k - kp as u64)
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

#[repr(C, align(64))]
struct AlignToSixtyFour([u8; 64]);

/// Returns a 64-byte aligned vector of T with at least the
/// given capacity.
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
    T: Unsigned + PrimInt + Ord + Shr<usize> + AsPrimitive<u8>,
    u8: AsPrimitive<T>,
{
    let mut vecs = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    for &a in sequence.iter() {
        let two_bits = (a >> shift).as_() & 3;
        vecs[two_bits as usize].push(a);
    }

    let mut pos = 0;
    for i in 0..4 {
        sequence[pos..pos + vecs[i].len()].copy_from_slice(&(vecs[i][..]));
        pos += vecs[i].len()
    }
}

#[cfg(test)]
mod tests;