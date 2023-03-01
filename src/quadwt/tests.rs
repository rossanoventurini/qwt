use super::*;
use crate::perf_and_test_utils::gen_sequence;
use crate::RSQVectorP512;

#[test]
fn test_small() {
    let data: [u8; 9] = [1, 0, 1, 0, 3, 4, 5, 3, 7];
    let wt = QWaveletTree::<_, RSQVectorP512>::new(&mut data.clone());

    assert_eq!(wt.rank(1, 4), Some(2));
    assert_eq!(wt.rank(1, 0), Some(0));   
    assert_eq!(wt.rank(8, 1), None);     // too large symbol
    assert_eq!(wt.rank(1, 9), Some(2));
    assert_eq!(wt.rank(7, 9), Some(1));  
    assert_eq!(wt.rank(1, 10), None);      // too large position

    assert_eq!(wt.select(8, 1), None);     // too large symbol
    assert_eq!(wt.select(5, 0), None);     // no 0th occurrence

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

        // Select out of bound
        assert_eq!(wt.select(0, N), None);
        assert_eq!(wt.select(1, 1), None);
        assert_eq!(wt.select(sigma-1, 2), None);
    }

    for sigma in [4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 255, 256, 16000] {
        let mut sequence: [u16; N] = [0; N];
        sequence[N - 1] = sigma - 1;
        let wt = QWaveletTree::<_, RSQVectorP512>::new(&mut sequence.clone());

        for i in 1..N - 1 {
            assert_eq!(wt.rank(0, i).unwrap(), i);
        }

        for i in 1..N {
            assert_eq!(wt.rank(sigma - 2, i).unwrap(), 0);
        }

        for (i, &symbol) in sequence.iter().enumerate() {
            let rank = wt.rank(symbol, i + 1).unwrap();
            let s = wt.select(symbol, rank).unwrap();
            assert_eq!(s, i);
        }

        // Select out of bound
        assert_eq!(wt.select(0, N), None);
        assert_eq!(wt.select(1, 1), None);
        assert_eq!(wt.select(sigma-1, 2), None);
    }
}

#[test]
fn test_get() {
    let n = 1025;
    for sigma in [4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 255, 256] {
        let sequence = gen_sequence(n, sigma);
        let wt = QWaveletTree::<_, RSQVectorP512>::new(&mut sequence.clone());
        for (i, &symbol) in sequence.iter().enumerate() {
            assert_eq!(wt.get(i), Some(symbol));
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
