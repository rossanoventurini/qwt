//! Errors for the zero-copy byte I/O layer.

use std::fmt;

/// Errors produced while encoding or decoding a `QWTB` / `HQWB` blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    /// Host endianness is not little-endian. The v1 format is LE-only.
    NotLittleEndian,
    /// Magic bytes do not match `QWTB` / `HQWB`.
    BadMagic,
    /// Unsupported or unknown format version.
    BadVersion,
    /// Byte slice is shorter than the header / directory / payload requires.
    Truncated,
    /// A required offset is not properly aligned (POD arrays need 64-byte alignment).
    Misaligned,
    /// Prefetch-augmented trees (`*Pfs`) are not supported in v1.
    PrefetchNotSupported,
    /// Structural inconsistency (e.g. `n_levels` disagrees with payload counts).
    Inconsistent {
        /// Human-readable detail.
        detail: &'static str,
    },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotLittleEndian => write!(f, "host is not little-endian (QWTB/HQWB v1 requires LE)"),
            Self::BadMagic => write!(f, "bad magic (expected QWTB or HQWB)"),
            Self::BadVersion => write!(f, "unsupported format version"),
            Self::Truncated => write!(f, "truncated byte blob"),
            Self::Misaligned => write!(f, "misaligned offset (need 64-byte alignment for POD arrays)"),
            Self::PrefetchNotSupported => {
                write!(f, "prefetch-augmented trees are not supported in v1")
            }
            Self::Inconsistent { detail } => write!(f, "inconsistent layout: {detail}"),
        }
    }
}

impl std::error::Error for LayoutError {}
