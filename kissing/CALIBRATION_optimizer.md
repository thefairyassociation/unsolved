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

## Follow-up bounded tests (2026-08-22)

The calibration still does **not** pass.  A fresh four-thread run of the fixed
seed, with the existing threshold-penalty phase disabled so the Adam basin could
be measured on its own, ended at `0.5007033607244914`:

```bash
KISS_POLISH=0 KISS_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  ./kissing/lib/riesz2 12 841 120000 51 \
  /tmp/kissing-scratch/cl840_841.txt
```

The written 841 by 12 output was finite and, after independent row
normalisation and a fresh NumPy Gram product, reproduced
`0.5007033607244914`.  The largest row-norm error before that normalisation was
`3.33e-16`.

The cheap follow-up hypotheses were bounded and negative:

| experiment | independently recomputed max inner product / cutoff |
| --- | ---: |
| five ultra-high Riesz exponents, 20,000 Adam steps each, LR scale 100 | `0.5006379823191830` |
| local `1e-4` kick, resume published tail at `s=1024` | `0.5006912110923703` |
| local `1e-3` kick, resume published tail at `s=1024` | `0.5006912214674577` |
| local `1e-3` kick, resume earlier at `s=256` | `0.5007043564409575` |
| exact structured seed with no jitter, stopped at `s=64` | `0.552076735372` |
| jitter `0.01`, stopped at `s=64` | `0.553212520746` |
| jitter `0.05`, stopped after dominated `s=32` | `0.574265236111` |
| three-thread reduction order, stopped at `s=512` | `0.506491838310` |
| two-thread reduction order, stopped at `s=512` | `0.506449770014` |
| four-thread retry, stopped at `s=512` | `0.506348517437` |

The five-stage high-exponent run used the exponent sequence in the authors'
`polish_841.py`, but deliberately short stages and a larger learning-rate scale
to test whether this checkpoint had room to move.  It improved almost entirely
in the first stage (`0.500638863699`) and plateaued at
`0.500637982319`; it did not approach `0.5`.  All completed local-tail outputs
in the table were independently recomputed from their coordinate files.  No
internal `feasible` flag was used as evidence.

These measurements rule out polishing this particular basin harder and the
tested small local restarts.  They do not prove that the fixed calibration
cannot pass: the published search used a large batch of independently perturbed
basins, while the command here follows one extremely basin-sensitive path.
