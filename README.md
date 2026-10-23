# Wick's theorem

Author: Shuhang Li ([shuhangli98@gmail.com](mailto:shuhangli98@gmail.com))

Python-based coupled-cluster equation derivation using Wick's theorem. The latest version, `03_graph_opt2/`, avoids equivalent contractions during generation and combines terms through signed graph canonicalization.

## Derivation time

Single-thread timings for the energy and all amplitude equations:

| Method |     Time |
| ------ | -------: |
| CCSD   | 0.0881 s |
| CCSDT  | 0.4074 s |
| CCSDTQ | 2.5728 s |

Measured on macOS with Python 3.10.14 and igraph 1.0.0 (BLISS).

## Validation

CCSDT/Q validation is symbolic; independent numerical solver comparisons are not included.
