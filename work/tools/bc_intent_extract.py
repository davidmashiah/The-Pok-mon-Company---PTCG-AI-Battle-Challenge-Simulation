"""Clone top pilots through the AUTHOR'S feature pipeline, not a generic one.

Two measurements set this up:

  * Ablating the GBM from the live agent cost **-0.0765** win rate on the mirror
    (0.4915 -> 0.4150, n~294 each). The learned imitation layer is the single
    most valuable component we have, and it decides 46% of our actions.
  * My own quick clone of the same top-50 pilots, on generic features (hashed
    card identity plus board summaries), has **no skill**: on the 465 held-out
    frames where the pilot did NOT take option 0, it scores 0.2430 against a
    random baseline of 0.2154 -- 1.4 sigma, not significant. It learned the
    option ordering and then deviates from it at chance.

So the idea is sound and my representation was the problem. The author's model
does not see hashed cards; it sees `pf.option_row` -- a schema-mapped row plus a
**180-entry intent vocabulary keyed to our exact card ids** -- and the eight most
recent semantic actions as history. That representation is already in the bundle
and is exactly what `_score` consumes.

This reproduces that input space **bit for bit**, including the parts that are
easy to get subtly wrong and impossible to notice afterwards:

  * duplicate_rank and the per-key `counts`, so identical-looking options are
    distinguished the same way at train and inference time
  * the six appended features (intent id, the previous action's type/source/
    target/attack, and history length)
  * `_HISTORY` advanced with the PILOT'S chosen action after every frame, capped
    at 8. Advancing it with our own choice, or not at all, would train the model
    on a history it will never see when it plays.

A model trained on this can be dropped into the existing scoring path instead of
bolted on as yet another override layer -- which matters, because this bundle
already carries five override layers that never fire.

  python work/tools/bc_intent_extract.py --out work/out/bc_intent.npz --min-score 1000
"""
import argparse
import csv
import glob
import gzip
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)

GRIMMSNARL_EX = 648


def load_bundle(name):
    """Import a bundle's policy_features + schema, then clean up sys.modules."""
    full = os.path.join(WORK, "agents", name)
    sys.path.insert(0, os.path.join(WORK, "lib"))
    sys.path.insert(0, full)
    cwd = os.getcwd()
    try:
        os.chdir(full)
        import policy_features as pf
        with gzip.open(os.path.join(full, "models",
                                    "feature_schema.pkl.gz"), "rb") as f:
            schema = pickle.load(f)
        with open(os.path.join(full, "final_schema.json"),
                  encoding="utf-8") as f:
            intent_to_id = {str(k): int(v) for k, v in
                            json.load(f)["intent_to_id"].items()}
    finally:
        os.chdir(cwd)
    return pf, schema, intent_to_id


def make_transform(schema):
    maps = schema["category_maps"]
    feats = schema["features"]

    def transform(row):
        values = []
        for name in feats:
            value = row.get(name, 0)
            if name in maps:
                try:
                    value = value.item()
                except Exception:
                    pass
                value = maps[name].get(value, -1)
            try:
                values.append(float(value))
            except Exception:
                values.append(float("nan"))
        return values
    return transform


def leaderboard():
    pats = [os.path.join(WORK, "out", "lb_now", "*.csv"),
            os.path.join(ROOT, "data", "*publicleaderboard*.csv")]
    files = sorted(sum((glob.glob(p) for p in pats), []))
    if not files:
        return {}
    best = {}
    with open(files[-1], encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("TeamName") or "").strip()
            try:
                score = float(row.get("Score") or 0)
            except ValueError:
                continue
            if name and score > best.get(name, -1e9):
                best[name] = score
    return best


def deck_of(steps, k):
    for st in steps:
        if k < len(st):
            a = st[k].get("action") or []
            if isinstance(a, list) and len(a) == 60:
                return sorted(int(x) for x in a)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="w34_koroll")
    ap.add_argument("--replays", default=os.path.join(WORK, "out",
                                                      "top_replays"))
    ap.add_argument("--out", default=os.path.join(WORK, "out",
                                                  "bc_intent.npz"))
    ap.add_argument("--min-score", type=float, default=0.0)
    a = ap.parse_args()

    pf, schema, intent_to_id = load_bundle(a.agent)
    transform = make_transform(schema)
    n_base = len(schema["features"])
    names = list(schema["features"]) + [
        "intent_id", "prev_type", "prev_source", "prev_target",
        "prev_attack", "history_len"]
    lb = leaderboard()

    X, Y, G = [], [], []
    gid = 0
    kept = 0
    scores_used = []
    import collections

    for path in sorted(glob.glob(os.path.join(a.replays, "*.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        rewards = d.get("rewards") or []
        if 1 not in rewards:
            continue
        w = rewards.index(1)
        steps = d.get("steps") or []
        deck = deck_of(steps, w)
        if deck is None or GRIMMSNARL_EX not in deck:
            continue
        teams = ((d.get("info") or {}).get("TeamNames") or [])
        team = teams[w] if w < len(teams) else None
        score = lb.get(team)
        if a.min_score > 0 and (score is None or score < a.min_score):
            continue
        if score is not None:
            scores_used.append(score)
        kept += 1

        history = []
        for st in steps:
            if w >= len(st):
                continue
            ag = st[w]
            obs = ag.get("observation") or {}
            act = ag.get("action")
            sel = obs.get("select")
            if not act or not isinstance(act, list) or len(act) == 60:
                continue
            if not sel:
                continue
            options = sel.get("option") or []
            if len(options) < 2:
                continue
            try:
                state = pf.base_state(obs, history)
                semantics = [pf.semantic(obs, o) for o in options]
                keys = [(s["type"], s["source_id"], s["target_id"],
                         s["attack_id"], s["area"], s["inplay_area"])
                        for s in semantics]
                counts = collections.Counter(keys)
                seen = collections.Counter()
                previous = history[-1] if history else {}
                rows = []
                for pos, (opt, item, key) in enumerate(
                        zip(options, semantics, keys)):
                    dup = seen[key]
                    seen[key] += 1
                    row, _ = pf.option_row(obs, state, opt, pos, item,
                                           counts[key], dup)
                    values = transform(row)
                    values.extend([
                        float(intent_to_id.get(pf.intent_text(item), -1)),
                        float(previous.get("type", -1)),
                        float(previous.get("source_id", 0)),
                        float(previous.get("target_id", 0)),
                        float(previous.get("attack_id", 0)),
                        float(len(history)),
                    ])
                    rows.append(values)
            except Exception:
                continue

            chosen = set(act)
            if not any(i in chosen for i in range(len(options))):
                continue
            for i, values in enumerate(rows):
                X.append(values)
                Y.append(1 if i in chosen else 0)
                G.append(gid)
            gid += 1
            # advance history with the PILOT's action, exactly as main.py does
            history = (history + [semantics[i] for i in chosen
                                  if 0 <= i < len(options)])[-8:]

    if not X:
        raise SystemExit("no rows extracted")
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int8)
    G = np.asarray(G, dtype=np.int32)
    np.savez_compressed(a.out, X=X, Y=Y, G=G,
                        names=np.array(names, dtype=object))
    print(f"episodes kept {kept}")
    if scores_used:
        scores_used.sort()
        print(f"cloned pilots: median ladder score "
              f"{scores_used[len(scores_used)//2]:.1f}, "
              f"range {scores_used[0]:.1f}-{scores_used[-1]:.1f}")
    print(f"rows {len(X)}  decisions {gid}  features {X.shape[1]} "
          f"({n_base} schema + 6 appended)  positive rate {Y.mean():.4f}")
    print(f"-> {a.out}")
    print("\nJudge it with bc_hard_frames.py, NOT top-1 accuracy: on this data "
          "the\ntrivial always-option-0 policy scores 0.4637 and only the HARD "
          "split is\ninformation. Generic features scored 0.2430 there against "
          "0.2154 random.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
