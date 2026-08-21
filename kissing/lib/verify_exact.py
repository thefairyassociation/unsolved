"""Exact-arithmetic verifier for kissing configurations.

A configuration is stored as JSON:
  {"dimension": n, "count": N, "scale2": s,        # squared norm of the raw vectors
   "field": "Q" | "Q(sqrt(D))",
   "vectors": [[c0, c1, ...], ...]}
Each coordinate is either an integer/rational string (field Q) or a pair
[a, b] meaning a + b*sqrt(D)  (field Q(sqrt(D))), with a, b rationals.

Check: Gram diagonal == scale2 exactly, off-diagonal <= scale2/2 exactly,
vectors distinct, count > record.
"""
import json, re, sys
from fractions import Fraction

RECORDS = {12: 841, 13: 1154, 14: 1932}

def _f(x, D=0):
    """Coordinate -> (a, b) with value a + b*sqrt(D).

    Accepts [a, b] pairs, plain rationals, and sympy-style strings such as
    "2*sqrt(3)", "-1/2", "3 - 2*sqrt(3)"."""
    if isinstance(x, list):
        return (Fraction(str(x[0])), Fraction(str(x[1])))
    s = str(x).replace(" ", "")
    try:
        return (Fraction(s), Fraction(0))
    except ValueError:
        pass
    import sympy
    e = sympy.nsimplify(sympy.sympify(s))
    r = sympy.sqrt(D)
    a = sympy.simplify(e.subs(r, 0))
    b = sympy.simplify(sympy.expand((e - a) / r))
    assert a.is_rational and b.is_rational, f"cannot parse coordinate {x!r} in Q(sqrt({D}))"
    return (Fraction(str(a)), Fraction(str(b)))

def load(path):
    d = json.load(open(path))
    fld = d.get("field")
    if fld is None:                                  # sniff sqrt(D) out of the coordinates
        ds = {int(m) for v in d["vectors"] for c in v
              for m in re.findall(r"sqrt\((\d+)\)", str(c))}
        assert len(ds) <= 1, f"multiple radicals {ds}: not a single quadratic field"
        D = ds.pop() if ds else 0
        fld = "Q" if D == 0 else f"Q(sqrt({D}))"
    else:
        D = 0 if fld == "Q" else int(re.search(r"sqrt\((\d+)\)", fld).group(1))
    d["field"] = fld
    V = [[_f(c, D) for c in v] for v in d["vectors"]]
    return d, V, D

def dot(u, v, D):
    """Exact inner product in Q(sqrt(D)); returns (a,b) meaning a + b*sqrt(D)."""
    a = Fraction(0); b = Fraction(0)
    for (p, q), (r, s) in zip(u, v):
        a += p * r + q * s * D
        b += p * s + q * r
    return (a, b)

def le(x, c):
    """Exact test  a + b*sqrt(D) <= c  with c rational, D >= 0."""
    a, b, D = x[0], x[1], x[2]
    d = c - a                      # need b*sqrt(D) <= d
    if b == 0:
        return d >= 0
    if b > 0:
        return d >= 0 and b * b * D <= d * d
    return d >= 0 or b * b * D >= d * d

def verify(path, record=None, verbose=True):
    d, V, D = load(path)
    n, N = d["dimension"], len(V)
    s2 = Fraction(str(d["scale2"])) if "scale2" in d else dot(V[0], V[0], D)[0]
    assert all(len(v) == n for v in V), "dimension mismatch"
    if d.get("count") is not None:
        assert d["count"] == N, f'count field {d["count"]} != {N} vectors'
    seen = set(); 
    for v in V:
        k = tuple((str(a), str(b)) for a, b in v)
        assert k not in seen, "duplicate vector"
        seen.add(k)
    half = s2 / 2
    worst = None; ties = 0
    for i in range(N):
        g = dot(V[i], V[i], D)
        assert g == (s2, Fraction(0)), f"vector {i} has norm^2 {g}, expected {s2}"
        for j in range(i + 1, N):
            g = dot(V[i], V[j], D)
            assert le((g[0], g[1], D), half), f"pair ({i},{j}) inner product {g} > {half}"
            if g == (half, Fraction(0)):
                ties += 1
            if worst is None or (g[0], g[1]) > worst:
                worst = (g[0], g[1])
    rec = RECORDS.get(n) if record is None else record
    res = {"ok": True, "dimension": n, "count": N, "scale2": str(s2),
           "field": d.get("field", "Q"), "distinct": True,
           "max_offdiag_raw": f"{worst[0]}+{worst[1]}*sqrt({D})" if worst[1] else str(worst[0]),
           "max_offdiag_unit": str(Fraction(worst[0], 1) / s2) if worst[1] == 0 else "irrational",
           "tight_pairs": ties, "record": rec, "beats_record": N > rec,
           "method": d.get("method", "")}
    if verbose:
        print(json.dumps(res, indent=2))
    return res

if __name__ == "__main__":
    verify(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
