use super::*;

#[test]
fn test_empty() {
    let qv = QVector::new();
    assert!(qv.is_empty());
    assert_eq!(qv.len(), 0);

    let qv = QVector::with_capacity(100);
    assert!(qv.is_empty());
    assert_eq!(qv.len(), 0);

    let qv = QVector::with_capacity_align64(100);
    assert!(qv.is_empty());
    assert_eq!(qv.len(), 0);
}

// Test construction FromIterator of a quaternary vector starting from vectors of
// integers types.
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
