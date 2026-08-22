#!/bin/bash
# Dim-12 calibration: classical 840 + 1, BLAS Riesz continuation with Adam.
# Run from the repo root.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

if [ ! -x kissing/lib/riesz2 ]; then
  make -C kissing/lib riesz2
fi

SEED=${SEED:-kissing/logs/cl840_841.txt}
MODE=${MODE:-hypercube}
S=${S:-51}

# Keep the historical CPU calibration defaults unless the source-fidelity
# mode is explicitly requested.  Faithful mode requires the authors' exact
# 35,000-step budget; KISS_JIT is intentionally ignored by riesz.c there.
if [ "${KISS_FAITHFUL:-0}" = "1" ]; then
  STEPS=${STEPS:-35000}
else
  STEPS=${STEPS:-120000}
fi

python3 kissing/lib/seed841.py "$SEED" --mode "$MODE"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export KISS_THREADS=${KISS_THREADS:-$OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=1
export KISS_INNER=${KISS_INNER:-16}
export KISS_LOSS=${KISS_LOSS:-riesz}
export KISS_SOLVER=${KISS_SOLVER:-adam}
export KISS_PROFILE=${KISS_PROFILE:-1}

if [ "${KISS_FAITHFUL:-0}" != "1" ]; then
  export KISS_JIT=${KISS_JIT:-0.03}
else
  unset KISS_JIT
fi

echo "running: riesz2 12 841 $STEPS $S $SEED" >&2
echo "  threads=$KISS_THREADS faithful=${KISS_FAITHFUL:-0} jit=${KISS_JIT:-0} solver=$KISS_SOLVER loss=$KISS_LOSS inner=$KISS_INNER" >&2
exec ./kissing/lib/riesz2 12 841 "$STEPS" "$S" "$SEED"
