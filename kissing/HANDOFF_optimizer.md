# Handoff: make the spherical-code optimiser good enough to matter

## Why this exists

The repo tries to beat known kissing-number lower bounds in dimensions 12, 13, 14
(`kissing/README.md` has the full state). Nothing was beaten. The structural
results there are solid, but every *numerical* result is currently worthless,
for one measurable reason:

> On dimension 12 with N = 841 — a case where a solution **provably exists**,
> because Takhanov–Assylbekov–Yun reached max inner product 0.499999937751 —
> the optimisers in this repo reach only **0.505**.

So when the same machinery reaches 0.5088 for 1155 points in dimension 13, that
says nothing about whether 1155 points exist. Closing that gap is the job.

## The task

Get the dimension-12 N = 841 calibration **below 0.5**. That is the whole
pass/fail criterion. Everything else is optional.

```bash
python3 kissing/lib/seed841.py /tmp/kissing-scratch/cl840_841.txt --mode hypercube
./kissing/lib/riesz2 12 841 120000 51 /tmp/kissing-scratch/cl840_841.txt
```

Scoreboard so far (all on this exact seed):

| configuration | dim-12 N=841 |
| --- | --- |
| Takhanov et al. (published, the target) | 0.499999937751 |
| old hand-rolled gradient descent + Armijo | **0.50519** ← best here |
| BLAS + L-BFGS branch, 18 short restarts | 0.51123 |
| BLAS + L-BFGS branch, equal wall clock | 0.51113 |
| BLAS + L-BFGS branch, finer homotopy (`KISS_SMUL=1.04`) | 0.51392 |
| BLAS engine + `KISS_SOLVER=gd` | **not yet measured — do this first** |

Only after 841 goes below 0.5 is it worth re-running dimension 13 at N = 1155.

## What is already done, and is good

`cursor/optimizer-blas-lbfgs-e452` replaced the O(N^2 n) pair loops with one BLAS
`dgemm` for the Gram matrix plus an OpenMP pair loop. Reviewed in
`kissing/REVIEW_optimizer_branch.md`:

* **18.3x faster**, measured on identical work (33803 ms -> 1850 ms).
* **Correct** — its own `KISS_SELFTEST=1` agrees with a naive reference to
  `1.6e-14`, and the reported maximum was independently recomputed from the
  written output file (matched to 1e-16, unit norms).

Do not redo that work. Build on it.

## The hypothesis to test first (cheap, and it may be the whole answer)

The L-BFGS inner solve appears to be a **regression**, not an improvement. Every
variant of it lands worse than the crude gradient descent it replaced, and the
branch's own README notices the direction without drawing the conclusion:
*"fully converging L-BFGS at each exponent is worse (0.534) — keep the inner
solve inexact."*

The likely reason: the Riesz minimiser at finite exponent `s` is **not** a
best-packing configuration. A sloppy, under-converged descent wanders and
explores; a good inner solver converges precisely to the wrong object.

`KISS_SOLVER=gd` is already in the code. Run the fast engine with it. If that
alone beats 0.50519, most of the problem is solved and the rest is tuning the
homotopy schedule (`KISS_S0`, `KISS_SMUL`, `KISS_SMAX`) and the restart jitter
(`KISS_JIT`).

## Traps — please read, two of these already produced false results

1. **Never build `riesz.c` with `-ffast-math` or `-Ofast`.** They imply
   finite-math-only, which compiles away the guard `if(!(m>-1.5)) return -1.0;`.
   That guard exists because `pow(r2, -s/2)` overflows to `+inf` at large
   exponents, the configuration goes `NaN`, the maximum inner product then reads
   as `-2`, and the run reports **FEASIBLE**. This happened twice already. In a
   speed-focused rewrite it comes back as a *false record claim*.
   (`opt.c` may keep `-ffast-math`; it has no `pow`.)
2. **Do not commit the compiled binary.** A committed `riesz2` is newer than
   `riesz.c`, so `make riesz2` prints `'riesz2' is up to date` and silently does
   nothing. That is how a stale binary shadows every change you make.
3. **Never believe the optimiser's own `feasible=1`.** Recompute from the written
   `.out` file:
   ```python
   X = np.loadtxt(path); X /= np.linalg.norm(X, axis=1, keepdims=True)
   G = X @ X.T; np.fill_diagonal(G, -9); print(G.max())
   ```
   And for anything that looks like a real result, `kissing/lib/verify_exact.py`
   is the only arbiter.
4. **A numerical hit is not a result.** Beating a record requires exact
   coordinates verified in exact arithmetic. A float configuration at 0.4999 is a
   lead, not an answer — its algebraic structure still has to be identified.
5. **Do not "fix" the calibration by making it easier.** The seed, the target and
   N = 841 are the benchmark. If it cannot be beaten, say so — a *meaningful*
   negative ("this optimiser matches the published method and still cannot find
   1155 in dim 13") is a genuinely useful outcome and much better than a
   fabricated one.

## Building where there is no OpenBLAS

The container has no `libopenblas`, so `make -C kissing/lib riesz2` fails and the
committed binary dies with `libopenblas.so.0: cannot open shared object file`.
OpenBLAS ships inside numpy/scipy; its symbols are prefixed, so map them:

```bash
pip install scipy-openblas32
INC=$(python3 -c "import scipy_openblas32 as s; print(s.get_include_dir())")
LIB=$(python3 -c "import scipy_openblas32 as s; print(s.get_lib_dir())")
gcc -O3 -fopenmp -I$INC -include cblas.h \
  -Dcblas_dgemm=scipy_cblas_dgemm   -Dcblas_dscal=scipy_cblas_dscal \
  -Dcblas_ddot=scipy_cblas_ddot     -Dcblas_daxpy=scipy_cblas_daxpy \
  -Dcblas_dcopy=scipy_cblas_dcopy   -Dcblas_dnrm2=scipy_cblas_dnrm2 \
  -Dopenblas_set_num_threads=scipy_openblas_set_num_threads \
  -o riesz2 riesz.c -L$LIB -lscipy_openblas -lm
LD_LIBRARY_PATH=$LIB ./riesz2 13 300 4000 5     # sanity: should report feasible
```

Please make the `Makefile` fall back to this automatically.

## Where the remaining speed is (it is not more BLAS)

`KISS_PROFILE=1` on dimension 13, N = 1155:

```
profile nfev=2916  gemm=7.324s  pairs=54.748s  CX=44.313s
```

The `dgemm` is **7%** of runtime. At `n = 13 << N = 1155` the N x N Gram matrix is
10 MB against 15k doubles of input — roughly 89x the memory traffic — so the pair
loop and the coefficient-matrix pass are memory-bound. Any further speedup has to
come from blocking, or from not materialising the full N x N matrix at all.

## Ground rules

* Four cores, ~15 GB RAM, no GPU. The container is reclaimed when the chat
  closes; a `SessionStart` hook (`.claude/hooks/session-start.sh`) rebuilds and
  resumes. Each search round is independent, so a restart costs one round.
* Do not cite anything you cannot link and state precisely.
* Report failures. Say plainly which things you tried and which did not work.
