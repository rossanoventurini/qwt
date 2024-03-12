//! The module implements a bit vector with operations to access,
//! to append bits, and to modify bits at arbitrary positions.
//!
//! It is possible to iterate over bits or positions of bits set
//! either to zero or one.

use crate::{AccessBin, SpaceUsage};

use serde::{Deserialize, Serialize};

pub mod rs_bitvector;
pub mod rs_narrow;

// vettore dataline 512 bit allineato
// struct DataLine {
//     words: [u64; 8],
// }

#[derive(Default, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct BitVector {
    //data: Box<[DataLine]>
    data: Vec<u64>,
    position: usize,
}

impl BitVector {
    /// Creates a new empty binary vector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty binary vector with at least a capacity of `n_bits`.
    pub fn with_capacity(n_bits: usize) -> Self {
        let capacity = (n_bits + 63) / 64;
        Self {
            data: Vec::with_capacity(capacity),
            ..Self::default()
        }
    }

    /// Creates a binary vector with `n_bits` set to 0.
    pub fn with_zeros(n_bits: usize) -> Self {
        let mut bv = Self::with_capacity(n_bits);
        bv.extend_with_zeros(n_bits);
        bv.shrink_to_fit();
        bv
    }

    /// Pushes a `bit` at the end of the binary vector.
    ///
    /// # Example
    /// ```
    /// use qwt::{BitVector, AccessBin};
    ///
    /// let mut bv = BitVector::new();
    /// bv.push(true);
    /// bv.push(false);
    /// bv.push(true);
    /// assert_eq!(bv.len(), 3);
    /// assert_eq!(bv.get(0), Some(true));
    /// ```
    #[inline]
    pub fn push(&mut self, bit: bool) {
        let pos_in_word = self.position % 64;
        if pos_in_word == 0 {
            self.data.push(0);
        }
        if bit {
            // push a 1
            if let Some(last) = self.data.last_mut() {
                *last |= (bit as u64) << pos_in_word;
            }
        }
        self.position += 1;
    }

    /// Appends `len` bits at the end of the bit vector by taking
    /// the least significant `len` bits in the u64 value `bits`.
    ///
    /// # Panics
    /// Panics if len is larger than 64 or if a bit of position  
    /// larger than len is set in bits.
    ///
    /// # Examples
    /// ```
    /// use qwt::BitVector;
    ///
    /// let mut bv = BitVector::new();
    /// bv.append_bits(5, 3); // appends 101
    /// bv.append_bits(6, 4); // appends 0110          
    /// assert_eq!(bv.len(), 7);
    /// assert_eq!(bv.get_bits(0, 3), Some(5));
    /// ```
    #[inline]
    pub fn append_bits(&mut self, bits: u64, len: usize) {
        assert!(len == 64 || (bits >> len) == 0);
        assert!(len <= 64);
        if len == 0 {
            return;
        }
        let pos_in_word: usize = self.position & 63;
        self.position += len;

        if pos_in_word == 0 {
            self.data.push(bits);
        } else if let Some(last) = self.data.last_mut() {
            *last |= bits << pos_in_word;
            if len > 64 - pos_in_word {
                self.data.push(bits >> (64 - pos_in_word));
            }
        }
    }

    /// Extends the bit vector by adding `n` bits set to 0.
    ///
    /// # Examples
    /// ```
    /// use qwt::{BitVector, AccessBin};
    ///
    /// let mut bv = BitVector::new();
    /// bv.extend_with_zeros(10);
    /// assert_eq!(bv.len(), 10);
    /// assert_eq!(bv.get(8), Some(false));
    /// ```
    #[inline]
    pub fn extend_with_zeros(&mut self, n: usize) {
        self.position += n;
        let new_size = (self.position + 63) / 64;
        self.data.resize_with(new_size, Default::default);
    }

    /// Sets the to `bit` the given position `index`.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds.
    #[inline]
    pub fn set(&mut self, index: usize, bit: bool) {
        let word = index >> 6;
        let pos_in_word = index & 63;
        self.data[word] &= !(1_u64 << pos_in_word);
        self.data[word] |= (bit as u64) << pos_in_word;
    }

    /// Accesses `len` bits, with 1 <= `len` <= 64,
    /// starting at position `index`.
    ///
    /// Returns [`None`] if `index`+`len` is out of bounds or
    /// `len` == 0 or `len` > 64.
    #[inline]
    pub fn get_bits(&self, index: usize, len: usize) -> Option<u64> {
        if (len == 0) | (len > 64) | (index + len >= self.position) {
            return None;
        }
        let block = index >> 6;
        let shift = index & 63;

        let mask = if len == 64 {
            std::u64::MAX
        } else {
            (1_u64 << len) - 1
        };

        if shift + len <= 64 {
            return Some(self.data[block] >> shift & mask);
        }
        Some((self.data[block] >> shift) | (self.data[block + 1] << (64 - shift) & mask))
    }

    /// Sets `len` bits, with 1 <= `len` <= 64,
    /// starting at position `index` to the `len` least
    /// significant bits in `bits`.
    ///
    /// # Panics
    /// Panics if `index`+`len` is out of bounds or
    /// `len` > 64 or if most significant bit in `bits`
    /// ia at a position larger than of equal to `len`.
    #[inline]
    pub fn set_bits(&mut self, index: usize, len: usize, bits: u64) {
        assert!(index + len <= self.position);
        // check there are no spurious bits
        assert!(len == 64 || (bits >> len) == 0);
        assert!(len <= 64);

        if len == 0 {
            return;
        }
        let mask = if len == 64 {
            std::u64::MAX
        } else {
            (1_u64 << len) - 1
        };
        let word = index >> 6;
        let pos_in_word = index & 63;

        self.data[word] &= !(mask << pos_in_word);
        self.data[word] |= bits << pos_in_word;

        let stored = 64 - pos_in_word;
        if stored < len {
            self.data[word + 1] &= !(mask >> stored);
            self.data[word + 1] |= bits >> stored;
        }
    }

    /// Returns an iterator over positions of bit set to 1 in the bit vector.
    ///
    /// # Examples
    /// ```
    /// use qwt::BitVector;
    /// let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
    /// let bv: BitVector = vv.iter().copied().collect();
    ///
    /// let v: Vec<usize> = bv.ones().collect();
    /// assert_eq!(v, vv);
    /// ```
    pub fn ones(&self) -> BitVectorBitPositionsIter<true> {
        BitVectorBitPositionsIter {
            bv: self,
            curr_position: 0,
            curr_word_pos: 0,
            curr_word: 0,
        }
    }

    /// Gives an iterator over positions of bit set to 0 in the bit vector.
    pub fn zeros(&self) -> BitVectorBitPositionsIter<false> {
        BitVectorBitPositionsIter {
            bv: self,
            curr_position: 0,
            curr_word_pos: 0,
            curr_word: 0,
        }
    }

    /// Gets a whole 64bit word.
    #[inline(always)]
    pub fn get_word(&self, i: usize) -> u64 {
        self.data[i]
    }

    /// Shrinks the underlying vector of 64bit words to fit.
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }

    /// Checks if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.position == 0
    }

    /// Returns the number of bits in the bitvector.
    pub fn len(&self) -> usize {
        self.position
    }

    /// Returns a non-consuming iterator over bits of the bit vector.
    pub fn iter(&self) -> BitVectorIter {
        BitVectorIter { bv: self, i: 0 }
    }
}

impl AccessBin for BitVector {
    /// Returns the bit at the given position `i`,
    /// or [`None`] if `i` is out of bounds.
    ///
    /// # Examples
    /// ```
    /// use qwt::{BitVector, AccessBin};
    ///
    /// let mut bv = BitVector::new();
    /// bv.extend_with_zeros(10);
    /// assert_eq!(bv.get(8), Some(false));
    /// ```
    #[inline(always)]
    fn get(&self, i: usize) -> Option<bool> {
        if i >= self.len() {
            return None;
        }
        Some(unsafe { self.get_unchecked(i) })
    }

    /// Returns the symbol at position `i`.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> bool {
        let word = i >> 6;
        let pos_in_word = i & 63;

        self.data[word] >> pos_in_word & 1_u64 == 1
    }
}

impl SpaceUsage for BitVector {
    /// Returns the space usage in bytes.
    fn space_usage_byte(&self) -> usize {
        self.data.space_usage_byte() + 8
    }
}

pub struct BitVectorBitPositionsIter<'a, const BIT: bool> {
    bv: &'a BitVector,
    curr_position: usize,
    curr_word_pos: usize,
    curr_word: u64,
}

/// Iterator over the positions of bits set to BIT (false for zeros,
/// true for ones) in the bit vector.
impl<'a, const BIT: bool> Iterator for BitVectorBitPositionsIter<'a, BIT> {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr_position >= self.bv.position {
            return None;
        }

        while self.curr_word == 0 {
            if self.curr_word_pos < self.bv.data.len() {
                if BIT {
                    self.curr_word = self.bv.data[self.curr_word_pos];
                } else {
                    // for zeros, just negate the word and report the positions of bit set to one!
                    self.curr_word = !self.bv.data[self.curr_word_pos];
                }
                self.curr_position = self.curr_word_pos << 6;
            } else {
                return None;
            }
            self.curr_word_pos += 1;
        }
        let l = self.curr_word.trailing_zeros() as usize;
        self.curr_position += l;
        let pos = self.curr_position;

        self.curr_word = if l >= 63 {
            0
        } else {
            self.curr_word >> (l + 1)
        };

        self.curr_position += 1;
        if pos >= self.bv.position {
            None
        } else {
            Some(pos)
        }
    }
}

pub struct BitVectorIntoIter {
    bv: BitVector,
    i: usize,
}

pub struct BitVectorIter<'a> {
    bv: &'a BitVector,
    i: usize,
}

impl IntoIterator for BitVector {
    type IntoIter = BitVectorIntoIter;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter {
        BitVectorIntoIter { bv: self, i: 0 }
    }
}

impl Iterator for BitVectorIntoIter {
    type Item = bool;
    fn next(&mut self) -> Option<Self::Item> {
        // TODO: this may be faster without calling get.
        self.i += 1;
        self.bv.get(self.i - 1)
    }
}

impl<'a> IntoIterator for &'a BitVector {
    type IntoIter = BitVectorIter<'a>;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> Iterator for BitVectorIter<'a> {
    type Item = bool;
    fn next(&mut self) -> Option<Self::Item> {
        // TODO: this may be faster without calling get.
        self.i += 1;
        self.bv.get(self.i - 1)
    }
}

impl Extend<bool> for BitVector {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = bool>,
    {
        for bit in iter {
            self.push(bit);
        }
    }

    /* Nigthly
        fn extend_one(&mut self, item: bool) {
            self.push(item);
        }
        fn extend_reserve(&mut self, additional: usize) {
            self.data.reserve
        }
    */
}

impl FromIterator<bool> for BitVector {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = bool>,
    {
        let mut bv = BitVector::default();
        bv.extend(iter);
        bv.shrink_to_fit();

        bv
    }
}

impl Extend<usize> for BitVector {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = usize>,
    {
        for pos in iter {
            if pos >= self.position {
                self.extend_with_zeros(pos + 1 - self.position);
            }
            self.set(pos, true);
        }
    }
}

impl FromIterator<usize> for BitVector {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = usize>,
    {
        let mut bv = BitVector::default();
        bv.extend(iter);
        bv.shrink_to_fit();

        bv
    }
}

impl std::fmt::Debug for BitVector {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let data_str: Vec<String> = self.data.iter().map(|x| format!("{:b}", x)).collect();
        write!(
            fmt,
            "BitVector {{ position:{:?}, data:{:?}}}",
            self.position, data_str
        )
    }
}

#[cfg(test)]
mod tests;
