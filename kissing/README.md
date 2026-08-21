# Kissing numbers in dimensions 12, 13, 14

**Goal.** Beat a known kissing-number lower bound in dimension 13 (priority), 12 or 14:
an explicit set of unit vectors in R^n with all pairwise inner products <= 1/2,
strictly larger than the record, verified in exact arithmetic.

**Outcome: no record was beaten.** What follows is the verified state, the
constructions that were built and checked, and the families that were ruled out
(with the reason in each case).

## Live records

Fetched 2026-08-21 from [Henry Cohn's table](https://cohn.mit.edu/kissing-numbers)
and cross-checked against the MIT DSpace copy
([hdl.handle.net/1721.1/153312](https://hdl.handle.net/1721.1/153312), last
compiled June 2026).

| dim | lower | upper | lower-bound source |
| --- | --- | --- | --- |
| 12 | 841 | 1355 | Takhanov–Assylbekov–Yun 2026, [arXiv:2606.18984](https://arxiv.org/abs/2606.18984) — numerical (max inner product 0.499999937751) |
| 13 | **1154** | 2064 | Zinoviev–Ericson 1999, Problems Inform. Transmission 35, 287–294, [mathnet ppi457](https://www.mathnet.ru/eng/ppi457) |
| 14 | 1932 | 3174 | Ganzhinov, *Highly symmetric lines*, Linear Algebra Appl. 722 (2025) 12–37, [arXiv:2207.08266](https://arxiv.org/abs/2207.08266) |

Upper bounds: de Laat–Leijenhorst 2024, [doi:10.1007/s12532-024-00264-w](https://doi.org/10.1007/s12532-024-00264-w).

## Best verified counts produced here

All three are **reproductions of known constructions**, not new results.

| dim | count | field | max off-diagonal | verifier output |
| --- | --- | --- | --- | --- |
| 12 | 840 | Q(sqrt 2) | exactly 1/2 | `dim12/verifier_output.txt`, `logs/verify_d12_840.txt` |
| 13 | 1154 | Q(sqrt 3) | exactly 1/2, 59640 tight pairs | `dim13/verifier_output.txt` |
| 14 | 1932 | Q (integer, norm^2 = 8) | exactly 4/8 = 1/2, 159264 tight pairs | `logs/verify_d14_1932.txt` |

Verify with

```bash
python3 kissing/lib/verify_exact.py kissing/dim13/configs/ze99_1154_exact.json
python3 -c "import sys;sys.path.insert(0,'kissing/lib');from verify_exact import verify_integer;\
verify_integer('kissing/dim14/configs/ganzhinov_1932_exact.json')"
```

`verify_exact.py` uses `fractions.Fraction` in Q(sqrt D) with an exact
comparison for `a + b*sqrt(D) <= c`. The integer fast path uses numpy int64
products — exact, not floating point — after *proving* the no-overflow bound
`n * max|c|^2 < 2^62`, and independently re-computes 20000 random entries with
Python big integers.

## Structural results established

### 1. ZE99's 1154 is maximal, and every layer of it is individually optimal

In unit coordinates the 1154 configuration has exactly five heights
`x_13 in {0, +-1/2, +-1}`, giving the layered decomposition

> `N(13) >= 2 + |A| + |B_+| + |B_-|`, with `A` a **1/2-code** in R^12 (height 0),
> `B_+-` **1/3-codes** (height +-1/2), cross condition `cos(A,B) <= 1/sqrt3`, and
> **no** constraint between `B_+` and `B_-`.

ZE99 realises `2 + 816 + 168 + 168`. All three thresholds are attained exactly
(`dim13/v2/layers.py`). Each piece is at its own combinatorial optimum:

* `A` = 816 = 51 x 16 tetrads `(+-1/2)^4`; 51 = `A(12,4,4)`, the maximum number of
  4-subsets of a 12-set with pairwise intersections <= 2.
* `B` = 24 axials `+-e_i` + 144 diamonds `(+-1)^12/sqrt12`; 144 = `A(12,4)` = `A(11,3)`.
* The 48 irrational vectors `(+-2 sqrt3 e_i, +-2)` beat the 24 axials that would
  otherwise sit at height 0: the all-rational version is 816+24+288+2 = **1130**.

Three independent negative checks:

* **Hole search** (`lib/holes.py`). `P = {u : <v_i,u> <= 1/2}` is convex and
  contains 0, so a 1155th unit vector exists iff some point of `P` has norm >= 1.
  Maximising `||u||` over `P` by the LP fixed-point iteration `u <- argmax_P <u,.>`
  from 250 random starts gives **0.9014 < 1**: nothing can be added.
* **Local jamming** (`dim13/v2/jamming.py`). For every point, `0` is not strictly
  outside the convex hull of the tangent projections of its tight neighbours:
  **no single point can move**. (Same verdict for the dim-12 840 and the dim-14
  1932.)
* **Enlarging B** (`dim13/v2/completeB.py`). One extra point in `B` would be worth
  **two** (it goes into both `B_+` and `B_-`) and give 1156. With `A` = the 816
  tetrads the same LP gives max norm **0.7232**; with `A` = the 840
  (tetrads + axials) and `B` = the 144 diamonds it gives **0.8327**. Five randomised
  greedy completions of the 144 diamonds all terminate at exactly **168**.
* **K(12,1/3) = 168 numerically.** The penalty optimiser seeded with the known 168
  plus one point, over many restarts, converges to max inner product
  **0.33335 > 1/3** for 169 points.

So 1154 can only be beaten in this framework by *moving several points at once*,
and the framework itself is capped: `2 + K(12) + 2*K(12,1/3) <= 2 + 841 + 336 = 1179`.

**A calibration worth recording.** The same two tests applied to configurations
whose record *was* subsequently broken:

| configuration | hole search `max ||u||` | locally jammed | beaten later? |
| --- | --- | --- | --- |
| dim 12, 840 (Leech–Sloane / Clebsch) | 0.8164 | yes | **yes** — 841 by Takhanov et al. 2026, by deforming the O(4) family inside the 48-system |
| dim 13, 1154 (ZE99) | 0.9014 | yes | not here |
| dim 14, 1932 (Ganzhinov) | 0.9354 | yes | not here |

So "no hole and locally jammed" does **not** imply optimality — the dim-12 840 has
both properties and is still beatable by a collective deformation. This is why the
remove-k/add-(k+1) continuation search (`lib/shakeloop.sh`) is the right residual
attack, and it is what is left running.

### 2. Ganzhinov's dim-14 1932, decoded

Re-derived from his published coordinates (`lib/analyze_g14.py`): in the norm-8
shell of Z^14 it is

* **364** vectors `(+-2,+-2,0^12)` — every one of these is compatible with every
  other vector in the shell, so any configuration here should take all of them;
* **1568** vectors `(+-1^8, 0^6)` on **49 supports x 32 sign patterns**.

The 14 coordinates split 7+7; every support has 4 coordinates in each half, the
7 half-blocks are the complements of the **Fano plane** lines, and the 49 supports
are all products of them, indexed by the edges of `K_{7,7}`. Support pairs meet in
4 (882 pairs) or 6 (294 pairs); the t=6 pairs are exactly "same row or same column".

The 49 sign codes are all **cosets of one global linear code `V <= F_2^14` of
dimension 5** (`lib/g14_global.py`), weight distribution {0:1, 6:7, 7:16, 8:7, 14:1}.

That yields a complete framework (`lib/gcode.py`), **validated by rebuilding a
valid 1932 from scratch** out of `V` + supports + solved coset representatives:

* a support `S` is usable iff every nonzero codeword of `V` meets `S` in >= 2
  coordinates (then `|C_S| = 2^dim V` with min distance >= 2);
* for a pair with `t = |S & T|`: `t <= 4` is free; `t in {5,6}` needs
  `dim V|_(S&T) < t`; `t >= 7` is impossible;
* the coset representatives must satisfy `<a, c_S + c_T> = 1` for some
  `a in V^perp` supported inside `S & T` — an **F_2 linear system**, solved
  exactly by elimination plus a search over its solution space.

**Ganzhinov's 1568 is maximal**: `lib/addable.c` checks all 768768 weight-8
candidates against it and finds **0** addable (0.7 s). And no 50th support can join
his 49 — the breakdown over all 3003 8-subsets (`dim14/why_no_50th.py`) is
2392 blocked because some `t=5` pair has `V|_X` equal to all of `F_2^X`,
455 not good for `V`, 107 meeting a support in >= 7, 0 addable.
Since the max support family with intersections in {2,3,4,6} is exactly 49
(`lib/clique.c` finds 49, matching his design), beating 1568 requires either
`t=5` pairs — hence a different `V` — or `dim V = 6` with >= 25 supports.

### 3. The norm-8 shell of Z^13 cannot reach 1154 — ruled out

The natural dim-13 analogue of Ganzhinov's model is 312 D-roots plus weight-8
vectors; beating 1154 needs **843** weight-8 vectors. It cannot happen:

* In `[13]` two 8-subsets always meet in >= 3, and **at most 3** supports can
  pairwise meet in <= 4. (Complements are 5-sets pairwise meeting in <= 1; a point
  lies in at most 3 such blocks, and the degree count forces `2a + b >= 22`
  together with `3a + b <= 21`, a contradiction for 4 blocks.)
* So essentially every support pair is constrained, and for both `t=5` and `t=6`
  the annihilator is 1-dimensional — two kernel codewords would sum to one
  disjoint from a support, contradicting goodness. Every pair is therefore a
  **forced** F_2 equation.
* With `k = 13 - dim V` unknowns per support, consistency needs
  `C(m,2) <~ k*m`, i.e. `m <~ 2k+1`:

| dim V | signs/support | max m | max weight-8 | max total |
| --- | --- | --- | --- | --- |
| 5 | 32 | 17 | 544 | 856 |
| 6 | 64 | 15 | 960 | 1272 (search reaches m = 7, i.e. 760) |
| 7 | 128 | 3 (dim V = 7 forces every pair to `t <= 4`) | 384 | 696 |

Best actually built and Gram-checked in this model: **760**. A coordinate section
of Ganzhinov's 1932 gives 312 + 672 = 984. The maximum support family with
intersections in {4,6} is 21 (vs 49 in dim 14), giving 21 x 32 + 312 = 984 again.

An explicit `dim V = 6` example: `V = S_7 (+) S_7` (two copies of the [7,3,4]
simplex code) makes *all* 49 Fano-product supports good in dim 14 with 64 signs
each, but three supports in one row share the same functional `(0, 1_Q)`, so their
three equations sum to `0 = 1`. At most 2 per row and per column survive — 14
supports, 896 + 364 = 1260 < 1932. Bigger sign codes cost supports faster than
they gain.

## Families tried, and why they were dropped

| # | family | verdict |
| --- | --- | --- |
| 1 | ZE99 exact reproduction (dim 13) | baseline 1154, re-verified in Q(sqrt 3) |
| 2 | Adding a point to ZE99 | impossible: hole-search max norm 0.9014 |
| 3 | Enlarging the 1/3-code layer B | impossible with A fixed (0.7232 / 0.8327); 169-point 1/3-codes in R^12 not found numerically |
| 4 | Enlarging A with B fixed | tetrads are `A(12,4,4)`-optimal; no vector of the region admits a 52nd 4-subset, and triples/5-sets/6-sets all violate `sum_{i in S}|b_i| <= 2/sqrt3` |
| 5 | Norm-8 shell of Z^13 (D-roots + weight-8) | ruled out, see section 3; ceiling ~984, best built 760 |
| 6 | Generic max-independent-set on all 329472 weight-8 vectors of Z^13 | local search reaches 404 (+312 = 716); far below the structured 984 |
| 7 | Global-linear-code framework in dim 13, dim V in {4,5,6,7} | best Gram-verified 760 |
| 8 | Coordinate and hyperplane sections of Ganzhinov's 1932 | 984 |
| 9 | Extending Ganzhinov's 1932 (dim 14) | 1568 weight-8 is maximal (0 of 768768 addable); 0 of 3003 supports addable |
| 10 | dim-14 global-code search over random V | ongoing, best so far well below 1932 |
| 11 | Numerical continuation from ZE99 + 1 point, and remove-k/add-(k+1) shaking | ongoing; no feasible 1155 found |
| 12 | Highly symmetric (group-orbit) configurations in R^13 from PSL(2,13) / PGL(2,13) on the 13-dimensional standard representation | implemented (`dim13/v2/orbits.py`); orbit optimisation did not reach max cos <= 1/2 |

## Code

| file | what it does |
| --- | --- |
| `lib/verify_exact.py` | exact Gram verifier: Q(sqrt D) via `Fraction`, plus a proven-safe exact integer path |
| `lib/holes.py` | LP hole search — can a point be added at all? |
| `lib/opt.c` | penalty-method spherical-code optimiser, adaptive step + basin hopping (`KISS_T` sets the threshold) |
| `lib/shake.py`, `lib/shakeloop.sh` | remove k points / add k+1 / re-optimise (the dim-12 840 -> 841 method) |
| `lib/mis8.c` | max independent set over all `2^8 * C(n,8)` weight-8 vectors, implicit adjacency |
| `lib/addable.c` | is a weight-8 configuration maximal? |
| `lib/signmis.c` | dense-bitset MIS over (support, sign) pairs for a fixed support family |
| `lib/clique.c` | max clique on k-subsets with prescribed pairwise intersections |
| `lib/gclique.c` | max clique on an arbitrary graph |
| `lib/gcode.py`, `lib/pipeline.py` | the global-linear-code framework and the F_2 coset solver |
| `lib/analyze_g14.py`, `lib/g14_global.py` | decoding Ganzhinov's dim-14 configuration |
| `dim13/v2/`, `dim14/` | the dimension-specific searches |
| `logs/progress.log` | append-only attempt log |

Benchmarks that pin the machinery down: `mis8` loads Ganzhinov's 1568 with zero
violations; `gcode` rebuilds a valid 1932 from `V` + supports + solved cosets;
`clique` returns exactly 49 for the dim-14 {4,6} support design and 21 for dim 13.
