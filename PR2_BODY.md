# Add first-class byte layout and zero-copy I/O for QWT / HQWT

## Summary

Adds a stable little-endian on-disk / in-memory byte container for plain and Huffman quad wavelet trees, with both **owned** round-trips and **zero-copy borrowed views**. The crate never depends on an mmap library — the caller owns the buffer (heap `Vec`, `AlignedBytes`, or a page-aligned mmap).

This is complementary to the existing serde/`bincode` path, which is left unchanged.

Depends on / builds on #31 (`range-successor-and-fused-rank`).

## API

| Container | Tree | Owned | Zero-copy |
| --- | --- | --- | --- |
| `QWTB` | plain QWT 256/512 | `qwt256_to_bytes` / `qwt256_from_bytes` (+ `qwt512_*`) | `QwtView` |
| `HQWB` | Huffman QWT 256/512 | `hqwt256_to_bytes` / `hqwt256_from_bytes` (+ `hqwt512_*`) | `HqwtView` |

Supporting pieces:

- `LayoutError` — magic / version / truncation / misalignment / endianness / prefetch
- `RSQVectorView` — per-level borrowed get/rank/select + POD accessors for custom hot paths
- `AlignedBytes` — over-allocated buffer exposing a 64-aligned window (needed when opening a view over a heap `Vec`)
- Raw-parts constructors (`QVector::from_raw_parts`, `RSSupportPlain::from_parts`, `QWaveletTree::from_parts`, …) used by the owned decode path

Views implement the same `AccessUnsigned` / `RankUnsigned` / `SelectUnsigned` traits as the owned trees.

## Format (v1)

Little-endian only. Header is 32 bytes; plain level directories are 128 B; HQWT directories add `level_len` (136 B). POD payloads (`DataLine`, `SuperblockPlain`) are 64-aligned within the blob. Flags: bit0 = B_SIZE=512, bit1 = prefetch (rejected in v1 with `PrefetchNotSupported`).

Owned decode copies POD slices onto the heap, so the source buffer need not be 64-aligned. Zero-copy views cast in place and therefore require a 64-aligned absolute base (satisfied by page-aligned mmaps or `AlignedBytes`).

## Design choices

- **No new dependencies, no feature flags.** Serde path untouched.
- **Caller owns mmap.** Keeps this crate free of `memmap2`/etc. and lets embedders control mapping lifetime.
- **Prefetch trees out of scope for v1.** Can be added later behind a new flag/version without breaking the plain layout.
- **Layout fields stay crate-private.** Computational APIs and the new bytes/view surface are the public path; we do not expose internal POD fields for external mmap parsing.

## Testing

- Round-trip get/rank/select for QWTB and HQWB (including empty trees and large/skewed alphabets)
- View parity vs owned trees; unaligned-base rejection; bad-magic errors
- Full crate: `cargo test --lib` (110) and `cargo clippy --lib --all-targets -- -D warnings` clean

## Docs

- New README section **Byte layout and zero-copy I/O** with a short example
- Module and public-fn docs on the `bytes` API

## Checklist

- [x] Owned `to_bytes` / `from_bytes` for QWTB + HQWB
- [x] Zero-copy `QwtView` / `HqwtView`
- [x] Overflow-safe decode of untrusted headers (`checked_region`)
- [x] Clippy clean under `-D warnings`
- [x] README + rustdoc
- [ ] (optional follow-up) example that mmaps a `.qwtb` file end-to-end
