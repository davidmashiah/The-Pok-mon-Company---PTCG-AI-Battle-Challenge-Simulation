"""Which LOCAL metric actually predicts the LADDER? Measure it, don't assume.

This project just spent a night accepting and rejecting ideas on the strength of
head-to-head win rate against v14. Then the ladder returned:

    v14 twins   670.3 / 734.2      (identical bundles -- noise floor ~64)
    v23_dz      389.3              (local gauntlet said 0.5297 over 438 games)

So the proxy we trusted said "slightly better" about an agent that is ~300
points worse in reality. Every conclusion drawn from that proxy is suspect.

We do have ground truth for a handful of agents that were actually submitted.
That makes it possible to CHECK a proxy instead of believing it: compute each
candidate local metric over those agents and correlate it with their real
ladder scores. A proxy that cannot reproduce a known ordering must not be used
to choose what ships next.

Metrics compared:
  vs-v14        head-to-head win rate against v14 -- what we have been using
  round-robin   Bradley-Terry strength from a diverse pool of our own agents

Usage: python work/tools/proxy_check.py
"""
import itertools
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
STORE = os.path.join(WORK, "out", "gauntlet.json")

# Real ladder scores. v14 has three draws of the SAME bundle; average them,
# and remember the spread between them is the noise floor any proxy must beat.
LADDER = {
    "v1_greedy": 538.7,
    "v2_lucario": 606.3,
    "w3_alakazam_guard": 637.5,
    "v3_lucario_search": 701.3,
    "champion": 707.0,
    "v14_search_noloop2": (703.4 + 670.3 + 734.2) / 3.0,
    "v23_dz": 389.3,
}
V14 = "v14_search_noloop2"


def load_pairs():
    """-> {(a,b): (wins_a, wins_b)} for the agents AS THEY EXIST NOW.

    Cells are keyed by content hash, and several of these agents have had
    different model weights under the same name during the night. Pooling those
    would compare an average of incompatible versions against a single ladder
    score, so only the current bundle hash counts.
    """
    import sys as _s
    _s.path.insert(0, HERE)
    from gauntlet import bundle_hash
    cur = {}
    for a in LADDER:
        try:
            cur[a] = bundle_hash(a)
        except Exception:
            cur[a] = None
    raw = json.load(open(STORE, encoding="utf-8"))
    agg = {}
    skipped = 0
    for key, v in raw.items():
        ka, kb = key.split("|")
        a, ha = ka.split("@")
        b, hb = kb.split("@")
        if a not in LADDER or b not in LADDER:
            continue
        if cur.get(a) != ha or cur.get(b) != hb:
            skipped += 1
            continue
        wa, wb = v.get("wa", 0), v.get("wb", 0)
        x, y = agg.get((a, b), (0, 0))
        agg[(a, b)] = (x + wa, y + wb)
    if skipped:
        print(f"(ignored {skipped} stale cells from superseded bundle versions)")
    return agg


def bradley_terry(agents, pairs, iters=6000):
    """Iterative MM fit of latent strengths from pairwise wins."""
    p = {a: 1.0 for a in agents}
    W = {a: 0.0 for a in agents}
    N = {}
    for (a, b), (wa, wb) in pairs.items():
        W[a] += wa
        W[b] += wb
        N[(a, b)] = N.get((a, b), 0) + wa + wb
    for _ in range(iters):
        new = {}
        for a in agents:
            denom = 0.0
            for b in agents:
                if a == b:
                    continue
                n = N.get((a, b), 0) + N.get((b, a), 0)
                if n:
                    denom += n / (p[a] + p[b])
            new[a] = (W[a] / denom) if denom > 0 else p[a]
        s = sum(new.values()) / len(new)
        p = {a: max(v / s, 1e-9) for a, v in new.items()}
    return {a: 400.0 * math.log10(v) for a, v in p.items()}   # Elo-ish


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    return num / (dx * dy) if dx and dy else 0.0


def main():
    pairs = load_pairs()
    agents = sorted(LADDER)
    print("pairwise games available among ladder-known agents:")
    have = 0
    for a, b in itertools.combinations(agents, 2):
        n = sum(pairs.get(k, (0, 0))[0] + pairs.get(k, (0, 0))[1]
                for k in ((a, b), (b, a)))
        if n:
            have += 1
        print(f"  {a:<20} vs {b:<20} {n:>5}")
    print(f"  -> {have}/{len(list(itertools.combinations(agents,2)))} pairs covered\n")

    # metric 1: win rate vs v14 (what we have been steering on)
    vs14 = {}
    for a in agents:
        if a == V14:
            continue
        w = l = 0
        for k, (wa, wb) in pairs.items():
            if k == (a, V14):
                w += wa
                l += wb
            elif k == (V14, a):
                w += wb
                l += wa
        if w + l >= 30:
            vs14[a] = w / (w + l)

    # metric 2: Bradley-Terry over the whole pool
    bt = bradley_terry(agents, pairs)

    print(f"{'agent':<20}{'LADDER':>9}{'vs-v14':>9}{'BT-elo':>9}")
    print("-" * 47)
    for a in sorted(agents, key=lambda x: -LADDER[x]):
        v = f"{vs14[a]:.3f}" if a in vs14 else "  --  "
        print(f"{a:<20}{LADDER[a]:>9.1f}{v:>9}{bt[a]:>9.1f}")

    common = [a for a in agents if a in vs14]
    if len(common) >= 3:
        r1 = spearman([vs14[a] for a in common], [LADDER[a] for a in common])
        print(f"\nSpearman(vs-v14, ladder)      = {r1:+.3f}   n={len(common)}")
    r2 = spearman([bt[a] for a in agents], [LADDER[a] for a in agents])
    print(f"Spearman(round-robin, ladder) = {r2:+.3f}   n={len(agents)}")
    print("\n+1.0 = proxy reproduces the real ordering; 0 = it is noise;")
    print("negative = it actively points the wrong way. Any proxy at or below 0")
    print("must not be used to decide what ships.")


if __name__ == "__main__":
    main()
