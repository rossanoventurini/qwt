//! Tests for raw-parts constructors (disassemble → reassemble).

use crate::qvector::rs_qvector::{RSQVector256, RSSupportPlain};
use crate::qvector::{DataLine, QVector};
use crate::{
    AccessQuad, AccessUnsigned, PrefixCode, RankUnsigned, SelectUnsigned, SuperblockPlain, HQWT256,
    QWT256,
};

#[test]
fn qvector_from_raw_parts_eq() {
    let original: QVector = (0..1000u32).map(|x| (x % 4) as u8).collect();

    let data: Box<[DataLine]> = original.data_lines().to_vec().into_boxed_slice();
    let position = original.position_bits();
    let rebuilt = QVector::from_raw_parts(data, position);
    assert_eq!(original, rebuilt);
    for i in 0..original.len() {
        assert_eq!(original.get(i), rebuilt.get(i));
    }
}

#[test]
fn rsqvector_from_parts_eq() {
    let original: RSQVector256 = (0..2000u64).map(|x| x % 4).collect();
    let qv = QVector::from_raw_parts(
        original.qvector().data_lines().to_vec().into_boxed_slice(),
        original.qvector().position_bits(),
    );
    let rs = original.rs_support();
    let superblocks: Box<[SuperblockPlain]> = rs.superblocks().to_vec().into_boxed_slice();
    let select_samples: [Box<[u32]>; 4] =
        std::array::from_fn(|s| rs.select_samples(s).to_vec().into_boxed_slice());
    // SAFETY: parts are copied verbatim from `rs`, which was built from `qv`.
    let rs_rebuilt = unsafe { RSSupportPlain::<256>::from_parts(superblocks, select_samples) };
    let rebuilt = unsafe { RSQVector256::from_parts(qv, rs_rebuilt, original.n_occs_smaller()) };
    assert_eq!(original, rebuilt);
}

fn clone_hqwt_levels(original: &HQWT256<u32>) -> Vec<RSQVector256> {
    original
        .levels()
        .iter()
        .map(|rs| {
            let qv = QVector::from_raw_parts(
                rs.qvector().data_lines().to_vec().into_boxed_slice(),
                rs.qvector().position_bits(),
            );
            let support = rs.rs_support();
            let superblocks = support.superblocks().to_vec().into_boxed_slice();
            let select_samples =
                std::array::from_fn(|s| support.select_samples(s).to_vec().into_boxed_slice());
            // SAFETY: parts are copied verbatim from an existing level built from `qv`.
            let rs_support =
                unsafe { RSSupportPlain::<256>::from_parts(superblocks, select_samples) };
            unsafe { RSQVector256::from_parts(qv, rs_support, rs.n_occs_smaller()) }
        })
        .collect()
}

#[test]
fn qwt_from_parts_eq_queries() {
    let data: Vec<u32> = (0..500).map(|x| (x * 7) % 64).collect();
    let original = QWT256::from(data.clone());

    let levels: Vec<_> = original
        .levels()
        .iter()
        .map(|rs| {
            let qv = QVector::from_raw_parts(
                rs.qvector().data_lines().to_vec().into_boxed_slice(),
                rs.qvector().position_bits(),
            );
            let support = rs.rs_support();
            let superblocks = support.superblocks().to_vec().into_boxed_slice();
            let select_samples =
                std::array::from_fn(|s| support.select_samples(s).to_vec().into_boxed_slice());
            // SAFETY: parts are copied verbatim from an existing level built from `qv`.
            let rs_support =
                unsafe { RSSupportPlain::<256>::from_parts(superblocks, select_samples) };
            unsafe { RSQVector256::from_parts(qv, rs_support, rs.n_occs_smaller()) }
        })
        .collect();

    let rebuilt = QWT256::from_parts(
        original.len(),
        original.n_levels(),
        original.sigma_raw(),
        levels,
    )
    .unwrap();

    assert_eq!(original.len(), rebuilt.len());
    assert_eq!(original.n_levels(), rebuilt.n_levels());
    for i in 0..data.len() {
        assert_eq!(original.get(i), rebuilt.get(i), "get mismatch at {i}");
    }
    for &sym in &[0u32, 1, 7, 31, 63] {
        for i in (0..=data.len()).step_by(17) {
            assert_eq!(
                original.rank(sym, i),
                rebuilt.rank(sym, i),
                "rank({sym},{i})"
            );
        }
        if let Some(cnt) = original.rank(sym, data.len()) {
            for k in 0..cnt.min(5) {
                assert_eq!(
                    original.select(sym, k),
                    rebuilt.select(sym, k),
                    "select({sym},{k})"
                );
            }
        }
    }
}

#[test]
fn hqwt_from_parts_rejects_terminal_prefix_with_populated_descendants() {
    let data: Vec<u32> = (0..800)
        .map(|x| if x % 10 == 0 { 0 } else { (x % 40) + 1 })
        .collect();
    let original = HQWT256::from(data);
    let mut codes_encode = original.codes_encode().to_vec();
    let mut codes_decode = original.codes_decode().to_vec();
    let long_code = codes_encode
        .iter()
        .find(|code| code.len >= 4)
        .expect("control tree must contain a multi-level code");
    let prefix = PrefixCode {
        content: long_code.content >> 2,
        len: long_code.len - 2,
    };
    let synthetic_symbol = codes_encode.len() as u32;
    codes_encode.push(prefix.clone());
    let bucket = &mut codes_decode[prefix.len as usize];
    assert!(
        bucket
            .binary_search_by_key(&prefix.content, |(content, _)| *content)
            .is_err(),
        "a valid Huffman table must not already terminate at this prefix"
    );
    bucket.push((prefix.content, synthetic_symbol));
    bucket.sort_unstable();

    let error = HQWT256::from_parts(
        original.len(),
        original.n_levels(),
        codes_encode,
        codes_decode,
        clone_hqwt_levels(&original),
        original.level_lens().to_vec(),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        crate::bytes::LayoutError::Inconsistent {
            detail: "Huffman decode entry has a populated descendant"
        }
    ));
}

#[test]
fn hqwt_from_parts_eq_queries() {
    let data: Vec<u32> = (0..800)
        .map(|x| {
            // Uneven freqs so Huffman codes matter.
            if x % 10 == 0 {
                0
            } else {
                (x % 40) + 1
            }
        })
        .collect();
    let original = HQWT256::from(data.clone());

    let levels = clone_hqwt_levels(&original);

    let rebuilt = HQWT256::from_parts(
        original.len(),
        original.n_levels(),
        original.codes_encode().to_vec(),
        original.codes_decode().to_vec(),
        levels,
        original.level_lens().to_vec(),
    )
    .unwrap();

    assert_eq!(original.len(), rebuilt.len());
    for i in 0..data.len() {
        assert_eq!(original.get(i), rebuilt.get(i), "hqwt get mismatch at {i}");
    }
    for &sym in &[0u32, 1, 5, 20, 40] {
        for i in (0..=data.len()).step_by(23) {
            assert_eq!(
                original.rank(sym, i),
                rebuilt.rank(sym, i),
                "hqwt rank({sym},{i})"
            );
        }
    }
}

#[test]
fn qwt_pfs_from_parts_rejected() {
    use crate::QWT256Pfs;
    let err = QWT256Pfs::<u32>::from_parts(0, 0, 0, vec![]).unwrap_err();

    assert_eq!(err, crate::bytes::LayoutError::PrefetchNotSupported);
}
