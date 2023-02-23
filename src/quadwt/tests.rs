use super::*;
use crate::perf_and_test_utils::gen_sequence;
use crate::RSQVectorP512;

#[test]
fn test_small() {
    let data: [u8; 9] = [1, 0, 1, 0, 3, 4, 5, 3, 7];
    let wt = QWaveletTree::<_, RSQVectorP512>::new(&mut data.clone());

    assert_eq!(wt.rank(1, 4), Some(2));

    for (i, &v) in data.iter().enumerate() {
        let rank = wt.rank(v, i + 1).unwrap();
        let s = wt.select(v, rank).unwrap();
        assert_eq!(s, i);
    }
}

#[test]
fn test() {
    const N: usize = 1025;
    for sigma in [4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 255] {
        let mut sequence: [u8; N] = [0; N];
        sequence[N - 1] = sigma - 1;
        let wt = QWaveletTree::<_, RSQVectorP512>::new(&mut sequence.clone());

        for i in 0..N - 1 {
            assert_eq!(wt.rank(0, i).unwrap(), i);
        }

        for i in 0..N {
            assert_eq!(wt.rank(sigma - 2, i).unwrap(), 0);
        }

        for (i, &symbol) in sequence.iter().enumerate() {
            let rank = wt.rank(symbol, i + 1).unwrap();
            let s = wt.select(symbol, rank).unwrap();
            assert_eq!(s, i);
        }
    }

    for sigma in [4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 255, 256, 16000] {
        let mut sequence: [u16; N] = [1; N];
        sequence[N - 1] = sigma - 1;
        let wt = QWaveletTree::<_, RSQVectorP512>::new(&mut sequence.clone());

        dbg!(sigma, wt.n_levels());
        for i in 1..N - 1 {
            assert_eq!(wt.rank(1, i).unwrap(), i);
        }

        for i in 1..N {
            assert_eq!(wt.rank(sigma - 2, i).unwrap(), 0);
        }

        for (i, &symbol) in sequence.iter().enumerate() {
            let rank = wt.rank(symbol, i + 1).unwrap();
            let s = wt.select(symbol, rank).unwrap();
            assert_eq!(s, i);
        }
    }
}

#[test]
fn test_get() {
    let n = 1025;
    for sigma in [4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 255, 256] {
        let sequence = gen_sequence(n, sigma);
        let wt = QWaveletTree::<_, RSQVectorP512>::new(&mut sequence.clone());
        for (i, &symbol) in sequence.iter().enumerate() {
            unsafe {
                assert_eq!(wt.get_unchecked(i), symbol);
            }
        }
    }
}

use bincode;

#[test]
fn test_serialize() {
    let wt = QWaveletTree::<_, RSQVectorP512>::new(&mut [0_u8; 10]);
    let s = bincode::serialize(&wt).unwrap();

    let des_wt = bincode::deserialize::<QWaveletTree<u8, RSQVectorP512>>(&s).unwrap();

    assert_eq!(des_wt, wt);
}
