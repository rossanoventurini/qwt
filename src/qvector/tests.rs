use super::*;

#[test]
fn test_empty() {
    let qv = QVector::default();
    assert!(qv.is_empty());
    assert_eq!(qv.len(), 0);

    let qv: QVector = [0, 1, 2, 3].into_iter().cycle().take(10).collect();
    assert!(!qv.is_empty());
    assert_eq!(qv.len(), 10);
}

#[test]
fn test_iterators() {
    let qv: QVector = [0, 1, 2, 3].into_iter().cycle().take(10).collect();
    for (i, v) in qv.iter().enumerate() {
        assert_eq!(v, (i % 4) as u8);
    }

    for (i, v) in (&qv).into_iter().enumerate() {
        assert_eq!(v, (i % 4) as u8);
    }

    for (i, v) in qv.into_iter().enumerate() {
        assert_eq!(v, (i % 4) as u8);
    }
}

// Test construction FromIterator of a quad vector starting from vectors of
// different integers types.
macro_rules! test_collect_and_get {
    ($($t:ty),*) => {
        $(::paste::paste! {
            #[test]
            fn [<test_collect_and_iterator_ $t>]() {
                let n = 1000;
                let v: [$t; 4] = [0, 1, 2, 3];
                let v: Vec<$t> = v.into_iter().cycle().take(n).collect();
                let qv: QVector = v.iter().copied().collect();

                for (get, ok) in qv.iter().zip(v) {
                    assert_eq!(get, ok as u8);
                }

                assert_eq!(qv.len(), n);
            }
        })*
    }
}

test_collect_and_get![i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize];

#[test]
fn test_data_line() {
    // new data_line full of `symbol``
    for symbol in 0..4 {
        let mut data_line = DataLine::default();

        for i in 0..=255 {
            data_line.set_symbol(symbol, i);
        }

        for i in 0..256 {
            assert_eq!(data_line.get(i), Some(symbol));
        }

        for i in 0..=256 {
            assert_eq!(data_line.rank((symbol + 1) % 4, i), Some(0));
        }

        for i in 0..=256 {
            dbg!(i, symbol);
            assert_eq!(data_line.rank(symbol, i), Some(i));
        }
    }
}

#[test]
fn superblock_public_accessors_reject_invalid_symbols_and_blocks() {
    let superblock = crate::SuperblockPlain::default();
    assert!(std::panic::catch_unwind(|| superblock.get_rank(4, 0)).is_err());
    assert!(std::panic::catch_unwind(|| superblock.get_rank(0, 8)).is_err());
    assert!(std::panic::catch_unwind(|| superblock.get_rank_all(8)).is_err());
    assert!(std::panic::catch_unwind(|| superblock.get_superblock_counter(4)).is_err());
    assert!(std::panic::catch_unwind(|| superblock.block_predecessor(4, 0)).is_err());
}

#[test]
fn data_line_rank_all_matches_scalar_rank() {
    let mut data_line = DataLine::default();
    for position in 0..=255u8 {
        data_line.set_symbol(position % 4, position);
    }
    for end in 0..=256 {
        // SAFETY: every tested boundary is within the 256-symbol line.
        let all = unsafe { data_line.rank_all_unchecked(end) };
        for symbol in 0..4u8 {
            assert_eq!(all[symbol as usize], data_line.rank(symbol, end).unwrap());
        }
    }
}
