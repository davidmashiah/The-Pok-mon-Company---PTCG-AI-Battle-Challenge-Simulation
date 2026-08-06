"""Put the coat on the 320 HP body, and never instead of attacking.

Hero's Cape is offered as OptionType.ATTACH (8) from the hand, with
inPlayArea/inPlayIndex naming the target -- one option per legal target, so a
single choice both plays the card and picks who wears it. Confirmed against a
real game rather than assumed (work/tools/cape_check.py).
"""
ATTACH = 8
ATTACK = 13
CAPE = 1159
GRIMMSNARL = 648
ACTIVE, BENCH = 4, 5


def _opt_type(o):
    try:
        return int(o.get("type", -1) if o.get("type") is not None else -1)
    except Exception:
        return -1


def choose(obs, base):
    """Return a replacement selection, or None to keep the policy's own."""
    sel = obs.get("select") or {}
    try:
        if int(sel.get("context", -1) if sel.get("context") is not None
               else -1) != 0:
            return None                      # MAIN phase only
    except Exception:
        return None
    opts = sel.get("option") or []
    if not opts:
        return None
    try:
        lo = int(sel.get("minCount", 0) or 0)
        hi = int(sel.get("maxCount", 0) or 0)
    except Exception:
        return None
    if lo > 1 or hi < 1:
        return None                          # this guard only makes 1-picks

    # Never pre-empt an attack. Attacking ends the turn, so the policy putting
    # a Cape on ahead of a knockout would cost the prize it was buying time for.
    for i in (base or []):
        if 0 <= i < len(opts) and _opt_type(opts[i]) == ATTACK:
            return None

    cur = obs.get("current") or {}
    try:
        me = (cur.get("players") or [])[int(cur.get("yourIndex", 0) or 0)]
    except Exception:
        return None
    hand = me.get("hand") or []

    best, best_rank = None, 0
    for i, o in enumerate(opts):
        if _opt_type(o) != ATTACH:
            continue
        try:
            ix = int(o.get("index", -1))
            if not (0 <= ix < len(hand)):
                continue
            if int((hand[ix] or {}).get("id", 0) or 0) != CAPE:
                continue
            area = int(o.get("inPlayArea", 0) or 0)
            idx = int(o.get("inPlayIndex", -1))
            zone = me.get("active") if area == ACTIVE else me.get("bench")
            if not zone or not (0 <= idx < len(zone)):
                continue
            mon = zone[idx]
            if not mon:
                continue
            if mon.get("tools"):
                continue                     # one coat per body
            if int(mon.get("id", 0) or 0) != GRIMMSNARL:
                continue                     # only the 320 -> 420 breakpoint
        except Exception:
            continue
        rank = 2 if area == ACTIVE else 1
        if rank > best_rank:
            best, best_rank = i, rank
    return [best] if best is not None else None
