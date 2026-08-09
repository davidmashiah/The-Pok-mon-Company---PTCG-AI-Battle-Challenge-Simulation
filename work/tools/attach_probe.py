"""Where does our Energy actually go, and is that a decision or a reflex?

Two measurements point at the same place and neither was followed up:

  * `default_audit.py` found we answer ATTACH_TO with option index 0 in 97 of 97
    decisions -- a prompt we never actually decide.
  * `advisor_guard.py` exists specifically to fix that ("use accelerated Energy
    to build the next evolved attacker instead of repeatedly overloading an
    already functional/basic target") and it changes the action **0 times in
    2391 decisions**. It runs; every path returns `baseline`.

Energy attachment decides whether Grimmsnarl ex ever attacks, so a reflex here
is expensive in a way a preference-list ordering is not.

This plays real games and records, for every attachment decision, WHICH card the
energy went to and whether we simply took option 0. It reports the split by card
so "we always attach to the Active" or "we never fund the evolved attacker"
would be visible as a distribution rather than inferred.

Reported per agent so two policies can be compared on the same question.

  python work/tools/attach_probe.py --agent _sub_v28 --opponent w5_grimmsnarl -n 40
"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
sys.path.insert(0, os.path.join(WORK, "lib"))

from cg.api import all_card_data, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

CARDS = {c.cardId: c for c in all_card_data()}
ATTACH_CTXS = {21, 22}          # ATTACH_TO / attachment targeting


def load(name):
    full = os.path.join(AGENTS, name)
    if full not in sys.path:
        sys.path.insert(0, full)
    cwd = os.getcwd()
    try:
        os.chdir(full)
        env = {}
        exec(compile(open(os.path.join(full, "main.py"),
                          encoding="utf-8-sig").read(), "main.py", "exec"), env)
        fn = env.get("agent") or [v for v in env.values() if callable(v)][-1]
        try:
            d = fn({"current": None, "select": None, "logs": []})
        except Exception:
            d = None
    finally:
        os.chdir(cwd)
        for nm, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None) or ""
            if f.startswith(full + os.sep) or f.startswith(full + "/"):
                del sys.modules[nm]
        while full in sys.path:
            sys.path.remove(full)
    if not (isinstance(d, (list, tuple)) and len(d) == 60):
        d = [int(x) for x in open(os.path.join(full, "deck.csv"),
                                  encoding="utf-8").read().split() if x.strip()]
    return fn, [int(x) for x in d]


def card_name(obs_dict, opt):
    """Resolve the option's target card id from the raw observation dict."""
    try:
        cur = obs_dict.get("current") or {}
        pls = cur.get("players") or []
        pi = opt.get("playerIndex")
        if pi is None:
            pi = int(cur.get("yourIndex", 0) or 0)
        area = int(opt.get("area", 0) or 0)
        idx = opt.get("index")
        p = pls[pi]
        zone = {4: "active", 5: "bench"}.get(area)
        if zone is None or not isinstance(idx, int):
            return None
        seq = p.get(zone) or []
        if area == 4:
            c = seq[0] if seq else None
        else:
            c = seq[idx] if 0 <= idx < len(seq) else None
        if not isinstance(c, dict):
            return None
        cid = int(c.get("id", 0) or 0)
        return cid, len(c.get("energyCards") or [])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="_sub_v28")
    ap.add_argument("--opponent", default="w5_grimmsnarl")
    ap.add_argument("-n", "--games", type=int, default=40)
    a = ap.parse_args()

    fa, da = load(a.agent)
    fb, db = load(a.opponent)

    took_zero = 0
    total = 0
    by_card = collections.Counter()
    by_energy = collections.Counter()
    noptions = collections.Counter()

    for g in range(a.games):
        a_first = (g % 2 == 0)
        p0, p1 = (fa, fb) if a_first else (fb, fa)
        d0, d1 = (da, db) if a_first else (db, da)
        for f in (fa, fb):
            try:
                f({"current": None, "select": None, "logs": []})
            except Exception:
                pass
        obs, _ = battle_start(list(d0), list(d1))
        if obs is None:
            continue
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                st = o.current
                if st is None or st.result != -1:
                    break
                who = st.yourIndex
                mine = (who == 0) == a_first
                fn = p0 if who == 0 else p1
                act = list(fn(obs))
                sel = obs.get("select") or {}
                ctx = int(sel.get("context", -1)
                          if sel.get("context") is not None else -1)
                opts = sel.get("option") or []
                if mine and ctx in ATTACH_CTXS and len(opts) >= 2 and act:
                    total += 1
                    noptions[len(opts)] += 1
                    if act[0] == 0:
                        took_zero += 1
                    got = card_name(obs, opts[act[0]] if
                                    0 <= act[0] < len(opts) else {})
                    if got:
                        cid, en = got
                        by_card[getattr(CARDS.get(cid), "name", cid)] += 1
                        by_energy[en] += 1
                obs = battle_select(act)
        except Exception:
            pass
        finally:
            battle_finish()

    print(f"\n{a.agent} vs {a.opponent}: {total} attachment decisions "
          f"over {a.games} games")
    if not total:
        print("  none seen -- ATTACH context ids may differ; check SelectContext")
        return 0
    print(f"  took option 0: {took_zero}/{total} = {took_zero/total:.3f}"
          + ("   <- REFLEX, not a decision" if took_zero / total >= 0.9 else ""))
    print(f"  mean options offered: "
          f"{sum(k*v for k,v in noptions.items())/max(1,sum(noptions.values())):.2f}")
    print("\n  energy went to:")
    for name, n in by_card.most_common(8):
        print(f"    {str(name)[:28]:28s} {n:5d}  ({n/total:.3f})")
    print("\n  target already had this many Energy:")
    for k in sorted(by_energy):
        print(f"    {k}: {by_energy[k]:5d}  ({by_energy[k]/total:.3f})")
    print("\nAttaching repeatedly to an already-fuelled target is exactly what "
          "advisor_guard\nwas written to prevent, and it never fires.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
