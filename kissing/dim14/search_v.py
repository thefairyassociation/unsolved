"""dim-14: search for a global linear code V <= F_2^14 admitting MORE than
Ganzhinov's 49 supports.  m supports x 2^dim(V) weight-8 vectors + 364 D-roots;
beating 1932 needs m*2^d >= 1569 (d=5 -> m >= 50, d=6 -> m >= 25)."""
import sys, os, json, random, time
sys.path.insert(0, 'kissing/lib')
import numpy as np
from gcode import Ctx, Elim, pair_rows, solve_signs, build, check, code_words
from pipeline import structural_graph, max_clique

n = 14; DROOTS = 364; RECORD = 1932
PCT = np.array([bin(x).count('1') for x in range(1 << n)], dtype=np.int8)
SUPS = np.array([m for m in range(1 << n) if bin(m).count('1') == 8], dtype=np.int32)
OUT = 'kissing/dim14/configs'; os.makedirs(OUT, exist_ok=True)

def consistent(ctx, F, rng, tries=25):
    best = []
    for _ in range(tries):
        order = F[:]; rng.shuffle(order)
        keep = []; el = Elim()
        for S in order:
            rows = pair_rows(ctx, keep, S)
            if rows is None: continue
            tr = el.copy()
            if all(tr.add(r, 1) for r in rows): el = tr; keep.append(S)
        if len(keep) > len(best): best, bel = keep, el
    return best, bel

def run(seed, budget):
    rng = random.Random(seed); best = 0; t0 = time.time(); tried = 0
    while time.time() - t0 < budget:
        d = rng.choice([5, 5, 5, 6]); tried += 1
        V = code_words([rng.randrange(1, 1 << n) for _ in range(d)])
        if len(set(V)) != 1 << d: continue
        ctx = Ctx(V, n, PCT, SUPS)
        if len(ctx.good) < 60: continue
        G, edges = structural_graph(ctx, cap=1500)
        if not edges: continue
        cl = max_clique(len(G), edges, 1.5, rng.randrange(1 << 30))
        if len(cl) * (1 << d) + DROOTS < 900: continue
        F = [G[i] for i in cl]
        keep, el = consistent(ctx, F, rng)
        if len(keep) * (1 << d) + DROOTS <= best: continue
        c = solve_signs(ctx, keep, el, rng, tries=150)
        if c is None: continue
        W = build(keep, V, c, n)
        ok, mx = check(W, n)
        if not ok: continue
        tot = len(W) + DROOTS
        if tot > best:
            best = tot
            msg = (f'seed={seed} dimV={d} supports={len(keep)} weight8={len(W)} '
                   f'total={tot} ({"BEATS" if tot > RECORD else "below"} 1932)')
            print(msg, flush=True)
            with open('kissing/logs/progress.log', 'a') as f:
                f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' | 14 | gcode-search | ' + msg + '\n')
            if tot > RECORD:
                json.dump({'V': V, 'supports': keep, 'cosets': c, 'dim_V': d,
                           'weight8': len(W), 'total': tot, 'n': n},
                          open(f'{OUT}/gc_s{seed}_{tot}.json', 'w'))
    print(f'seed {seed}: {tried} codes, best {best}', flush=True)

if __name__ == '__main__':
    run(int(sys.argv[1]), float(sys.argv[2]))
