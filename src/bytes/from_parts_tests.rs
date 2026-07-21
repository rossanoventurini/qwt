//! Tests for raw-parts constructors (disassemble → reassemble).

use crate::qvector::rs_qvector::{RSQVector256, RSSupportPlain};
use crate::qvector::{DataLine, QVector};
use crate::{
    AccessQuad, AccessUnsigned, RankUnsigned, SelectUnsigned, SuperblockPlain, HQWT256, QWT256,
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
    let rs_rebuilt = RSSupportPlain::<256>::from_parts(superblocks, select_samples);
    let rebuilt = RSQVector256::from_parts(qv, rs_rebuilt, original.n_occs_smaller());
    assert_eq!(original, rebuilt);
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
            let rs_support = RSSupportPlain::<256>::from_parts(superblocks, select_samples);
            RSQVector256::from_parts(qv, rs_support, rs.n_occs_smaller())
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
            let rs_support = RSSupportPlain::<256>::from_parts(superblocks, select_samples);
            RSQVector256::from_parts(qv, rs_support, rs.n_occs_smaller())
        })
        .collect();

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
