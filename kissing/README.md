# Kissing numbers

Attempt to improve the **dimension-13 kissing-number lower bound**. Success is an explicit set of unit vectors in R^n with all pairwise inner products ≤ 1/2, **strictly more numerous** than the live record, verified in exact arithmetic (sympy / integer quadratic fields). Floating-point searches are allowed only as a route to identifiable exact structure.

Live records, fetched 2026-08-21 from [Henry Cohn's table](https://cohn.mit.edu/kissing-numbers):

| dim | lower | upper | lower citation |
| --- | --- | --- | --- |
| 12 | 841 | 1355 | Takhanov, Assylbekov, Yun 2026, [arXiv:2606.18984](https://arxiv.org/abs/2606.18984) |
| 13 | **1154** | 2064 | Zinoviev–Ericson 1999, [mathnet ppi457](https://www.mathnet.ru/eng/ppi457); English: Problems Inform. Transmission 35 (1999) 287–294 |
| 14 | 1932 | 3174 | Ganzhinov, Linear Algebra Appl. 722 (2025) 12–37, [doi:10.1016/j.laa.2025.05.002](https://doi.org/10.1016/j.laa.2025.05.002), [arXiv:2207.08266](https://arxiv.org/abs/2207.08266) |

Upper bounds in the table are from de Laat–Leijenhorst 2024, [doi:10.1007/s12532-024-00264-w](https://doi.org/10.1007/s12532-024-00264-w).

Reproducing Zinoviev–Ericson 1154 is a **baseline**, not progress.

## Dimension 13 construction (ZE99 baseline)

All vectors have squared norm 16; kissing is `<vi,vj> ≤ 8`.

- **816 tetrads.** A maximum constant-weight code A(12,4,4)=51 of 4-subsets of the first 12 coordinates; each support appears with all 16 sign patterns of `(±2)^4`.
- **288 diamonds.** Steiner system S(5,6,12) (132 hexads as minus-sets of `(±1)^12`) plus the 1-factor `{(0,3),(1,2),(4,5),(6,10),(7,8),(9,11)}` and its complements; last coordinate `±2`.
- **2 axials.** `±4 e_12`.
- **48 irrationals.** For each of the first 12 axes, the four vectors with `±2√3` there and `±2` on coordinate 12.

This matches the Cohn archive format (inner-product coefficients all 1; some coordinates `±2*sqrt(3)`). The 48 irrationals are the layer that lifts the 1106-vector integer configuration to 1154.

## Methods tried (see `dim13/progress.log`)

Documented in that file as they run. Families:

1. ZE99 exact reproduction (baseline).
2. Leech lattice Λ24 minimal-vector 13-dimensional sections (lattice and non-lattice).
3. Barnes–Wall BW16 13-dimensional sections.
4. Equatorial sections of Ganzhinov's 1932-point code in R^14.
5. Binary/ternary spherical codes (Golay, Nordstrom–Robinson related, Hamming, constant-weight).
6. PSp(4,5) 13-dimensional Weil representation orbits.
7. Lifts of the dim-12 record / sections of the dim-14 record.
8. Shells in Q(√5) and Q(ζ10) added to the 1106-vector integer skeleton (after removing the √3 layer).
9. Greedy add / remove-and-reinsert / energy search, then exact identification.

## Files

- `dim13/verifier.py` — exact verifier (sympy / quadratic-integer).
- `dim13/constructions/` — generators.
- `dim13/configs/` — every configuration that verifies.
- `dim13/best.json` — best verified so far.
- `dim13/progress.log` — append-only attempt log.
