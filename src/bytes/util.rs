//! Low-level helpers for the byte I/O layer.

use super::LayoutError;
use std::mem::{align_of, size_of};

/// Reject big-endian hosts. The v1 format is little-endian only.
#[inline]
pub fn ensure_le() -> Result<(), LayoutError> {
    if cfg!(target_endian = "little") {
        Ok(())
    } else {
        Err(LayoutError::NotLittleEndian)
    }
}

/// Round `n` up to a multiple of `align` (power of two).
#[inline]
pub fn align_up(n: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (n + align - 1) & !(align - 1)
}

/// Reinterpret an aligned byte slice as a slice of `T`.
///
/// # Errors
/// - [`LayoutError::Misaligned`] if `bytes.as_ptr()` is not aligned for `T`
/// - [`LayoutError::Truncated`] if `bytes.len()` is not a multiple of `size_of::<T>()`
///
/// # Safety considerations
/// `T` must be a plain-old-data type with no padding that would make an
/// arbitrary bit pattern invalid (e.g. `DataLine`, `SuperblockPlain`, `u32`).
/// Callers must uphold that invariant; this helper only checks alignment and length.
#[inline]
pub fn cast_slice<T>(bytes: &[u8]) -> Result<&[T], LayoutError> {
    if bytes.is_empty() {
        return Ok(&[]);
    }
    let align = align_of::<T>();
    let addr = bytes.as_ptr() as usize;
    if addr % align != 0 {
        return Err(LayoutError::Misaligned);
    }
    if !bytes.len().is_multiple_of(size_of::<T>()) {
        return Err(LayoutError::Truncated);
    }
    let len = bytes.len() / size_of::<T>();
    // SAFETY: alignment and length checked; T is POD by caller contract.
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, len) })
}

/// Mutable variant of [`cast_slice`].
#[inline]
pub fn cast_slice_mut<T>(bytes: &mut [u8]) -> Result<&mut [T], LayoutError> {
    if bytes.is_empty() {
        return Ok(&mut []);
    }
    let align = align_of::<T>();
    let addr = bytes.as_ptr() as usize;
    if addr % align != 0 {
        return Err(LayoutError::Misaligned);
    }
    if !bytes.len().is_multiple_of(size_of::<T>()) {
        return Err(LayoutError::Truncated);
    }
    let len = bytes.len() / size_of::<T>();
    // SAFETY: alignment and length checked; T is POD by caller contract.
    Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len) })
}

/// Copy a POD array out of a (possibly unaligned) byte slice into a freshly
/// allocated, naturally aligned `Box<[T]>`.
///
/// Used by the owned `from_bytes` path, which must not require the source
/// buffer to be 64-byte aligned (heap `Vec<u8>` typically is not). Zero-copy
/// views should keep using [`cast_slice`] and reject misaligned inputs.
///
/// # Errors
/// - [`LayoutError::Truncated`] if `bytes.len() < n * size_of::<T>()`
#[inline]
pub fn copy_pod_slice<T: Copy + Default>(bytes: &[u8], n: usize) -> Result<Box<[T]>, LayoutError> {
    let need = n.checked_mul(size_of::<T>()).ok_or(LayoutError::Truncated)?;
    if bytes.len() < need {
        return Err(LayoutError::Truncated);
    }
    if n == 0 {
        return Ok(Box::from([]));
    }
    let mut out = vec![T::default(); n];
    // SAFETY: T is POD by caller contract; we copy exactly `need` bytes into
    // a freshly allocated, naturally aligned buffer of `n` T values.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, need);
    }
    Ok(out.into_boxed_slice())
}

/// Append the raw bytes of a POD slice to `out`.
///
/// Uses `std::slice::from_raw_parts` on the typed slice so the in-memory
/// `#[repr(C)]` layout is preserved byte-for-byte.
#[inline]
pub fn write_slice<T>(out: &mut Vec<u8>, data: &[T]) {

    if data.is_empty() {
        return;
    }
    let nbytes = data.len() * size_of::<T>();
    // SAFETY: T is POD; we only read its bytes.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, nbytes) };
    out.extend_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qvector::DataLine;

    #[test]
    fn align_up_basic() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
    }

    #[test]
    fn cast_empty() {
        let empty: &[DataLine] = cast_slice::<DataLine>(&[]).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn cast_aligned_dataline() {
        // Allocate a 64-aligned buffer holding one DataLine of zeros.
        let mut buf = vec![0u8; 128];
        let start = align_up(buf.as_ptr() as usize, 64) - buf.as_ptr() as usize;
        let slice = &buf[start..start + 64];
        let lines: &[DataLine] = cast_slice(slice).unwrap();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].words, [0; 4]);
        // silence unused mut
        let _ = &mut buf;
    }

    #[test]
    fn cast_rejects_short() {
        let buf = [0u8; 32]; // half a DataLine
        // May also fail alignment; either error is fine.
        assert!(cast_slice::<DataLine>(&buf).is_err());
    }
}
