# Dimension-12 N=841 optimizer calibration

## Verdict

**Pass criterion not met.** The best independently recomputed maximum inner
product obtained in this pass was

```
0.5004772386288089
```

for the fixed 841-point hypercube-seeded calibration. This is a large improvement
over the previous 0.50519 result, but it is still not below 0.5 and must not be
reported as feasible.

## Fixed benchmark

```bash
python3 kissing/lib/seed841.py /tmp/kissing-scratch/cl840_841.txt --mode hypercube
./kissing/lib/riesz2 12 841 120000 51 /tmp/kissing-scratch/cl840_841.txt
```

The final file was checked independently rather than trusting `feasible`:

```python
X = np.loadtxt(path)
X /= np.linalg.norm(X, axis=1, keepdims=True)
G = X @ X.T
np.fill_diagonal(G, -9)
print(G.max())
```

All 841 rows were finite; the maximum norm error after normalization was
`2.22e-16`. The recomputed maximum was
`0.5004772386288089`.

## Measurements

| solver/configuration | independently recomputed max inner product |
| --- | ---: |
| reviewed BLAS + L-BFGS branch | 0.51123 |
| BLAS engine + projected GD, fixed 120000/51 benchmark | 0.5096199586372459 |
| BLAS engine + Adam, published exponent/LR schedule scaled to 120000 steps | **0.5004772386288089** |
| required threshold | **< 0.5** |

The Adam trajectory was basin-sensitive: repeated runs could land materially
worse because OpenMP reductions perturb the path at floating-point scale. The
0.500477 result is therefore a measured best, not a claim that every run
reproduces that number.

## What changed

* Imported the reviewed BLAS/OpenMP engine and retained `KISS_SOLVER=gd` and
  `KISS_SOLVER=lbfgs`.
* Added Adam with the exponent and learning-rate schedule published by
  Takhanov–Assylbekov–Yun; Adam is the default calibration solver.
* Added an automatic `scipy-openblas32` Makefile fallback without
  `-ffast-math` or `-Ofast`.
* Removed tracked `riesz` / `riesz2` binaries so stale executables cannot
  shadow source changes.
* Made the seed reader reject truncated inputs and accept the optimizer's own
  commented `.out` files.
* Preserved the independent BLAS-vs-naive self-test; measured gradient error was
  `1.55e-14`.

## Source fidelity

The published search used Adam in 64 batches and the exponent/LR schedule now
encoded in `riesz.c`; its separate ultra-high-exponent polishing stage starts
from candidates already below 0.501. See the
[paper](https://arxiv.org/abs/2606.18984) and
[authors' code](https://github.com/k-nic/841_in_12D).

No binary or float configuration is committed, and no numerical result here is
claimed as an exact kissing configuration.
