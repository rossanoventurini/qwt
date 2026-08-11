//! Host-safe selection for architecture-specific QWT primitives.
//!
//! Selection is internal and automatic: supported hosts use POPCNT, while
//! every other target uses the portable scalar implementation.

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "popcnt")
))]
use std::sync::OnceLock;

/// Selected implementation for the bounded popcount primitive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PopcountPath {
    #[cfg(not(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "popcnt"
    )))]
    Scalar,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Popcnt,
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "popcnt")
))]
static POPCOUNT_PATH: OnceLock<PopcountPath> = OnceLock::new();

/// Select the host-safe implementation once for the process.
#[inline(always)]
pub(crate) fn selected_popcount_path() -> PopcountPath {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "popcnt"
    ))]
    {
        PopcountPath::Popcnt
    }
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(target_feature = "popcnt")
    ))]
    {
        *POPCOUNT_PATH.get_or_init(detect_popcount_path)
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        PopcountPath::Scalar
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "popcnt")
))]
fn detect_popcount_path() -> PopcountPath {
    if std::arch::is_x86_feature_detected!("popcnt") {
        return PopcountPath::Popcnt;
    }
    PopcountPath::Scalar
}

/// Portable fallback; compiler lowering respects the build's target baseline.
#[cfg(any(
    test,
    not(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "popcnt"
    ))
))]
#[inline(always)]
pub(crate) fn scalar_popcount_u64(word: u64) -> usize {
    word.count_ones() as usize
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "popcnt")]
pub(crate) unsafe fn popcount_u64_popcnt(word: u64) -> usize {
    std::arch::x86_64::_popcnt64(word as i64) as usize
}

#[cfg(all(test, target_arch = "x86"))]
#[target_feature(enable = "popcnt")]
pub(crate) unsafe fn popcount_u64_popcnt(word: u64) -> usize {
    (std::arch::x86::_popcnt32(word as i32) + std::arch::x86::_popcnt32((word >> 32) as i32))
        as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_path_matches_rust_popcount() {
        let mut value = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..10_000 {
            value = value
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            assert_eq!(scalar_popcount_u64(value), value.count_ones() as usize);
        }
    }

    #[test]
    fn selected_path_is_host_safe_and_correct() {
        for value in [0, 1, u64::MAX, 0xf0f0_aaaa_5555_1234] {
            let count = match selected_popcount_path() {
                #[cfg(not(all(
                    any(target_arch = "x86", target_arch = "x86_64"),
                    target_feature = "popcnt"
                )))]
                PopcountPath::Scalar => scalar_popcount_u64(value),
                // SAFETY: `Popcnt` is selected only when the build or host supports it.
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                PopcountPath::Popcnt => unsafe { popcount_u64_popcnt(value) },
            };
            assert_eq!(count, value.count_ones() as usize);
        }
    }
}
