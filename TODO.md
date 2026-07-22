## TODO
- Use a binary vector at the first level when log sigma is odd.
- Implement an efficient iterator over a Wavelet Tree.
  - Partially addressed by `RangeDistinctIter` (fixed-stack distinct-symbol
    enumerator over a row range). A full position / value iterator is still open.
- Implement a binary wavelet tree. (`binwt` remains a stub.)
- Replace From with TryFrom
- Implement DoubleEndedIterator for all the collections
