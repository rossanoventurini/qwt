//! Zero-copy byte I/O for quad wavelet trees.
//!
//! This module provides a canonical little-endian on-disk / in-memory byte
//! layout for [`crate::QWaveletTree`] (`QWTB`) and [`crate::HuffQWaveletTree`]
//! (`HQWB`), together with:
//!
//! - owned round-trips: `to_bytes` / `from_bytes`
//! - borrowed zero-copy views: [`QwtView`] / [`HqwtView`] over a caller-provided
//!   `&[u8]` (the caller owns any `mmap`; this crate never depends on one)
//!
//! No feature flags. No new dependencies. The existing serde path is unchanged.
//!
//! Prefetch-augmented trees (`*Pfs`) are rejected in v1
//! ([`LayoutError::PrefetchNotSupported`]).

mod error;
mod util;
mod qwtb;
mod hqwb;
mod level;
mod view;

pub use error::LayoutError;
pub use hqwb::{hqwt256_from_bytes, hqwt256_to_bytes, hqwt512_from_bytes, hqwt512_to_bytes};
pub use level::RSQVectorView;
pub use qwtb::{qwt256_from_bytes, qwt256_to_bytes, qwt512_from_bytes, qwt512_to_bytes};
pub use view::{AlignedBytes, HqwtView, QwtView};

#[allow(unused_imports)] // cast_slice_mut / write_slice kept for future writers
pub(crate) use util::{
    align_up, cast_slice, cast_slice_mut, copy_pod_slice, ensure_le, write_slice,
};







/// Magic for a plain quad wavelet tree byte blob.
pub const QWTB_MAGIC: &[u8; 4] = b"QWTB";
/// Magic for a Huffman quad wavelet tree byte blob.
pub const HQWB_MAGIC: &[u8; 4] = b"HQWB";
/// Container format version.
pub const FORMAT_VERSION: u16 = 1;

/// Header size shared by `QWTB` and `HQWB` (bytes before the level directory).
pub const HEADER_SIZE: usize = 32;
/// Bytes per plain-QWT level directory entry.
///
/// Layout: `off_data u64` + `n_datalines u32` + `pad0 u32` + `position_bits u64`
/// + `off_superblocks u64` + `n_superblocks u32` + `pad1 u32` + `off_sel[4] u64×4`
/// + `n_sel[4] u32×4` + `n_occs_smaller[5] u64×5`
/// = 8+4+4+8+8+4+4+32+16+40 = 128.
pub const LEVEL_DIR_SIZE: usize = 128;
/// Bytes per HQWT level directory entry (`LEVEL_DIR_SIZE` + `level_len` u64).
pub const HQWT_LEVEL_DIR_SIZE: usize = LEVEL_DIR_SIZE + 8;


/// Flag bit0: block size is 512 (else 256).
pub const FLAG_B512: u8 = 0b0000_0001;
/// Flag bit1: prefetch support present (rejected in v1).
pub const FLAG_PREFETCH: u8 = 0b0000_0010;

#[cfg(test)]
mod from_parts_tests;

