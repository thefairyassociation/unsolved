#!/usr/bin/env python3
"""Small deterministic checks for the source-fidelity N=841 path.

This is deliberately a regression test, not a search driver.  It independently
normalises the published 841-point coordinates, recomputes their Gram maximum,
round-trips the optimiser's text serialization, and (when a built binary is
available) runs two bounded polish steps on the prepared successful witness.  The
optimizer's ``feasible`` field is never used as evidence.

The coordinate fixture is the authors' published successful numerical result.
It is intentionally kept as coordinates rather than a Gram matrix so the test
checks the complete coordinate -> normalise -> Gram path independently.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from prepare_841_polish import (  # noqa: E402
    decimal_gram_reconstruct,
    normalized_coordinates,
)


EXPECTED_MAX = 0.4999999377514321


def verified_max(path: Path) -> tuple[np.ndarray, float, float]:
    """Load, validate, normalise independently, and return max-IP metadata."""
    x = np.loadtxt(path)
    if x.shape != (841, 12):
        raise AssertionError(f"expected (841, 12), got {x.shape}")
    if not np.isfinite(x).all():
        raise AssertionError("coordinates contain non-finite values")
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms <= 1e-15):
        raise AssertionError("coordinates contain a zero row")
    pre_norm_error = float(np.max(np.abs(norms - 1.0)))
    z = x / norms[:, None]
    gram = z @ z.T
    upper = gram[np.triu_indices(841, 1)]
    return z, float(upper.max()), pre_norm_error


def check_fixture(path: Path) -> np.ndarray:
    z, max_ip, norm_error = verified_max(path)
    if abs(max_ip - EXPECTED_MAX) > 5e-13:
        raise AssertionError(
            f"published fixture max-IP changed: {max_ip:.17g} "
            f"(expected {EXPECTED_MAX:.17g})"
        )
    if norm_error > 5e-13:
        raise AssertionError(f"published fixture is not unit-normalised: {norm_error}")
    print(
        f"fixture: verified max-IP={max_ip:.17g}; "
        f"max pre-normalisation norm error={norm_error:.3g}"
    )
    return z


def check_serialization(z: np.ndarray, directory: Path) -> None:
    path = directory / "roundtrip.txt"
    np.savetxt(path, z, fmt="%.17g")
    loaded = np.loadtxt(path)
    if loaded.shape != (841, 12) or not np.isfinite(loaded).all():
        raise AssertionError("coordinate serialization did not round-trip")
    _, max_ip, _ = verified_max(path)
    if abs(max_ip - EXPECTED_MAX) > 5e-13:
        raise AssertionError(f"serialization changed max-IP: {max_ip:.17g}")
    print(f"serialization: 17-digit round-trip max-IP={max_ip:.17g}")


def check_source_contract(source: Path) -> None:
    text = source.read_text()
    required = {
        "explicit faithful mode": "KISS_FAITHFUL",
        "exact 35,000-step guard": "requires exactly 35000 search steps",
        "raw Adam forced in faithful mode": "int adam_raw=faithful_mode ||",
        "uniform-random hypercube extra": "exact seed core + uniform-random hypercube extra",
        "no faithful penalty fallback": "if(!faithful_mode &&",
        "authors polish learning-rate factor": "if(faithful_mode) stage_lr/=10.0",
        "raw Adam nonfinite guard": 'faithful_guard("raw step update"',
        "polish nonfinite guard": 'faithful_guard("polish step update"',
        "no nonfinite serialization": "no candidate was serialized",
    }
    for label, needle in required.items():
        if needle not in text:
            raise AssertionError(f"missing source contract: {label}")
    print("source contract: faithful guards and /10 polish factor present")


def check_bounded_polish(binary: Path, fixture: Path, directory: Path) -> None:
    """Exercise decimal-Gram handoff and explicit polish without a search run."""
    fixture_z = normalized_coordinates(fixture)
    gram_path = directory / "authors_gram_10dp.txt"
    prepared_path = directory / "authors_prepared_coords.txt"
    gram_max, reconstructed_max = decimal_gram_reconstruct(
        fixture_z, gram_path, prepared_path
    )
    if abs(gram_max - 0.4999999378) > 5e-11:
        raise AssertionError(f"decimal Gram max-IP changed: {gram_max:.17g}")
    if abs(reconstructed_max - 0.4999999377601447) > 5e-11:
        raise AssertionError(
            f"rank-12 Gram reconstruction max-IP changed: {reconstructed_max:.17g}"
        )
    gram_lines = gram_path.read_text().splitlines()
    if not gram_lines or "." not in gram_lines[0].split()[0] or len(gram_lines[0].split()[0].split(".")[1]) != 10:
        raise AssertionError("authors Gram handoff is not serialized at %.10f")
    print(
        f"decimal Gram handoff: Gram max-IP={gram_max:.17g}; "
        f"rank-12 reconstructed max-IP={reconstructed_max:.17g}"
    )

    candidate = prepared_path

    env = os.environ.copy()
    env.update(
        {
            "KISS_FAITHFUL": "1",
            "KISS_ADAM_POLISH": "1",
            "KISS_ADAM_POLISH_ONLY": "1",
            "KISS_ADAM_POLISH_STEPS": "2",
            "KISS_ADAM_POLISH_STAGES": "1",
            "KISS_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "KISS_POLISH": "1",
        }
    )
    proc = subprocess.run(
        [str(binary), "12", "841", "35000", "991", str(candidate)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"bounded faithful polish failed ({proc.returncode}):\n{proc.stderr}"
        )
    if "penalty polish" in proc.stderr:
        raise AssertionError("faithful polish unexpectedly fell through to penalty polish")
    if "adam polish from max=" not in proc.stderr or "updates=2" not in proc.stderr:
        raise AssertionError(
            "already-feasible prepared witness did not complete two faithful polish updates"
        )
    output = Path(f"{candidate}.riesz.s991.out")
    if not output.exists():
        raise AssertionError(f"optimizer did not write {output}")
    z, max_ip, norm_error = verified_max(output)
    if max_ip > 1.0 or norm_error > 5e-12:
        raise AssertionError(
            f"optimizer serialization invalid: max-IP={max_ip:.17g}, "
            f"norm error={norm_error:.3g}"
        )
    header_lines = output.read_text().splitlines()[:3]
    if not any(line.startswith("# faithful=1") for line in header_lines):
        raise AssertionError("faithful output marker missing")
    if not any("extra_mode=preserved-input extra_index=none" in line for line in header_lines):
        raise AssertionError("polish-only output falsely claimed a randomized extra")
    print(
        f"bounded two-step polish: output independently verified max-IP={max_ip:.17g}; "
        f"norm error={norm_error:.3g}"
    )

    later_start = directory / "later_start.txt"
    np.savetxt(later_start, np.loadtxt(prepared_path), fmt="%.17g")
    later_env = os.environ.copy()
    later_env.update(
        {
            "KISS_FAITHFUL": "1",
            "KISS_ADAM_BASE_START": "1",
            "KISS_POLISH": "0",
            "KISS_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    rejected = subprocess.run(
        [str(binary), "12", "841", "35000", "992", str(later_start)],
        text=True,
        capture_output=True,
        env=later_env,
        check=False,
    )
    rejected_output = Path(f"{later_start}.riesz.s992.out")
    if (
        rejected.returncode == 0
        or "rejects KISS_ADAM_BASE_START=1" not in rejected.stderr
        or rejected_output.exists()
    ):
        raise AssertionError("faithful later-stage start was not rejected cleanly")
    print("later-stage fresh-moment start: rejected as non-faithful")


def check_nonfinite_abort(binary: Path, fixture: Path, directory: Path) -> None:
    """Ensure malformed and overflowed faithful polish states cannot serialize."""
    env = os.environ.copy()
    env.update(
        {
            "KISS_FAITHFUL": "1",
            "KISS_ADAM_POLISH": "1",
            "KISS_ADAM_POLISH_ONLY": "1",
            "KISS_ADAM_POLISH_STEPS": "1",
            "KISS_ADAM_POLISH_STAGES": "1",
            "KISS_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "KISS_POLISH": "1",
        }
    )

    malformed = directory / "nonfinite.txt"
    x = np.loadtxt(fixture)
    x[0, 0] = np.nan
    np.savetxt(malformed, x, fmt="%.17g")
    proc = subprocess.run(
        [str(binary), "12", "841", "35000", "992", str(malformed)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    output = Path(f"{malformed}.riesz.s992.out")
    if proc.returncode == 0 or output.exists():
        raise AssertionError("nonfinite input was accepted or serialized")
    if "faithful" not in proc.stderr or "no candidate was serialized" not in proc.stderr:
        raise AssertionError(f"nonfinite input lacked an abort diagnostic:\n{proc.stderr}")

    raw_env = dict(env)
    raw_env.pop("KISS_ADAM_POLISH", None)
    raw_env.pop("KISS_ADAM_POLISH_ONLY", None)
    raw_env["KISS_ADAM_BASE_END"] = "1"
    raw_env["KISS_POLISH"] = "0"
    proc = subprocess.run(
        [str(binary), "12", "841", "35000", "994", str(malformed)],
        text=True,
        capture_output=True,
        env=raw_env,
        check=False,
    )
    output = Path(f"{malformed}.riesz.s994.out")
    if proc.returncode == 0 or output.exists():
        raise AssertionError("nonfinite raw-Adam input was accepted or serialized")
    if "faithful" not in proc.stderr or "no candidate was serialized" not in proc.stderr:
        raise AssertionError(f"raw nonfinite input lacked an abort diagnostic:\n{proc.stderr}")

    overflow = directory / "overflow.txt"
    prepared = directory / "overflow_prepared.txt"
    prepared_z = normalized_coordinates(fixture)
    decimal_gram_reconstruct(
        prepared_z, directory / "overflow_gram.txt", prepared
    )
    y = np.loadtxt(prepared)
    y[0, 0] += 1e-3
    np.savetxt(overflow, y, fmt="%.17g")
    overflow_env = dict(env)
    # The one-step update becomes non-finite; the guard must abort before the
    # output file is opened.  This is a bounded failure-path test, not a run.
    overflow_env["KISS_ADAM_POLISH_LR_SCALE"] = "1e309"
    proc = subprocess.run(
        [str(binary), "12", "841", "35000", "993", str(overflow)],
        text=True,
        capture_output=True,
        env=overflow_env,
        check=False,
    )
    output = Path(f"{overflow}.riesz.s993.out")
    if proc.returncode == 0 or output.exists():
        raise AssertionError("nonfinite polish update was accepted or serialized")
    if "faithful nonfinite guard" not in proc.stderr:
        raise AssertionError(f"nonfinite update lacked guard diagnostic:\n{proc.stderr}")
    print("nonfinite guards: raw input, polish input, and overflowed polish aborted safely")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--coordinates",
        type=Path,
        default=root / "kissing" / "lib" / "testdata" / "authors_841_coordinates.txt",
    )
    parser.add_argument(
        "--binary", type=Path, default=root / "kissing" / "lib" / "riesz2"
    )
    parser.add_argument(
        "--source", type=Path, default=root / "kissing" / "lib" / "riesz.c"
    )
    parser.add_argument(
        "--skip-binary", action="store_true", help="only run fixture/source checks"
    )
    args = parser.parse_args()
    if not args.coordinates.exists():
        parser.error(f"coordinate fixture not found: {args.coordinates}")
    if not args.source.exists():
        parser.error(f"source not found: {args.source}")

    with tempfile.TemporaryDirectory(prefix="faithful-841-") as tmp:
        directory = Path(tmp)
        z = check_fixture(args.coordinates)
        check_serialization(z, directory)
        check_source_contract(args.source)
        if args.skip_binary:
            print("binary smoke: skipped by request")
        elif not args.binary.exists():
            print(f"binary smoke: skipped; build {args.binary} first")
        else:
            check_bounded_polish(args.binary, args.coordinates, directory)
            check_nonfinite_abort(args.binary, args.coordinates, directory)
    print("PASS: faithful 841 regression checks")


if __name__ == "__main__":
    main()
