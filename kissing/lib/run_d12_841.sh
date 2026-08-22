#!/bin/bash
# Dim-12 calibration: classical 840 + 1, Riesz continuation with BLAS + L-BFGS.
# Run from the repo root.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

if [ ! -x kissing/lib/riesz2 ]; then
  make -C kissing/lib riesz2
fi

SEED=${SEED:-kissing/logs/cl840_841.txt}
MODE=${MODE:-hole}
STEPS=${STEPS:-80000}
S=${S:-1}

python3 kissing/lib/seed841.py "$SEED" --mode "$MODE"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export KISS_THREADS=${KISS_THREADS:-$OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=1
export KISS_INNER=${KISS_INNER:-16}
export KISS_JIT=${KISS_JIT:-0.03}
export KISS_LOSS=${KISS_LOSS:-riesz}
export KISS_SOLVER=${KISS_SOLVER:-lbfgs}
export KISS_PROFILE=${KISS_PROFILE:-1}

echo "running: riesz2 12 841 $STEPS $S $SEED" >&2
echo "  threads=$KISS_THREADS jit=$KISS_JIT solver=$KISS_SOLVER loss=$KISS_LOSS inner=$KISS_INNER" >&2
exec ./kissing/lib/riesz2 12 841 "$STEPS" "$S" "$SEED"
