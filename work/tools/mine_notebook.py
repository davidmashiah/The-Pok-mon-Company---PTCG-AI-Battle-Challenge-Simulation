"""Extract a runnable agent bundle out of a downloaded public notebook.

Adopting a published agent from a higher-scoring author is the only thing that
has ever moved our ladder score (697 -> 726 -> 849, twice). This turns a .ipynb
into work/agents/<name>/ so field_test.py can measure it on the same code path
as everything else.

Handles the two shapes seen in this competition:
  - `%%writefile main.py` cell  -> main.py verbatim
  - a DECK = [...] literal cell -> deck.csv
  - base64 asset blobs          -> unpacked next to main.py

  python work/tools/mine_notebook.py <notebook.ipynb> --name w20_lucario1084
"""
import argparse
import ast
import base64
import io
import json
import os
import re
import sys
import tarfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")

# makthanithin's 1084.5 notebook does not compile as published -- a stray "hi"
# sits inside an if-condition's closing paren. The surrounding block is a plain
# `if (A and B and C): continue`, so the intended token is unambiguous.
REPAIRS = [
    (") hi:\n", "):\n"),
]


def cells(nb):
    for c in nb["cells"]:
        yield c["cell_type"], "".join(c["source"])


def find_main(nb):
    """The %%writefile main.py cell, with the magic line stripped."""
    best = None
    for kind, src in cells(nb):
        if kind != "code":
            continue
        m = re.match(r"\s*%%writefile\s+(\S+)\s*\n", src)
        if m and os.path.basename(m.group(1)) == "main.py":
            body = src[m.end():]
            if best is None or len(body) > len(best):
                best = body
    return best


def find_deck(nb):
    """A 60-int DECK literal, taken from the cell that actually writes deck.csv."""
    # shape 1: a `%%writefile deck.csv` cell holding one card id per line
    for kind, src in cells(nb):
        if kind != "code":
            continue
        m = re.match(r"\s*%%writefile\s+(\S*deck\.csv)\s*\n", src)
        if m:
            try:
                v = [int(x) for x in src[m.end():].split() if x.strip()]
            except ValueError:
                continue
            if len(v) == 60:
                return v
    # shape 2: a DECK = [...] literal
    for kind, src in cells(nb):
        if kind != "code":
            continue
        for m in re.finditer(r"^\s*(DECK|deck|DECK_LIST)\s*=\s*(\[[^\]]*\])",
                             src, re.M):
            try:
                v = ast.literal_eval(m.group(2))
            except Exception:
                continue
            if isinstance(v, list) and len(v) == 60 and all(
                    isinstance(x, int) for x in v):
                return v
    return None


def find_assets(nb):
    """base64 blobs -> (name, bytes). Some bundles ship models this way."""
    out = []
    for kind, src in cells(nb):
        if kind != "code":
            continue
        for m in re.finditer(r'(["\'])([A-Za-z0-9+/=\s]{2000,})\1', src):
            blob = re.sub(r"\s+", "", m.group(2))
            try:
                raw = base64.b64decode(blob, validate=True)
            except Exception:
                continue
            if len(raw) > 1000:
                out.append(raw)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    nb = json.load(open(a.notebook, encoding="utf-8"))
    body = find_main(nb)
    deck = find_deck(nb)
    if body is None:
        raise SystemExit("no `%%writefile main.py` cell found")

    # Published-source typo repairs. Each one must be found or we stop: a silent
    # no-op patch is how a "fixed" bundle ships still broken. Only unambiguous
    # syntax damage is repaired here -- never semantics.
    for bad, good in REPAIRS:
        if bad in body:
            body = body.replace(bad, good)
            print(f"repaired published typo: {bad!r} -> {good!r}")

    out = os.path.join(AGENTS, a.name)
    os.makedirs(out, exist_ok=True)
    compile(body, "main.py", "exec")          # fail loudly, not at game time
    with open(os.path.join(out, "main.py"), "w", encoding="utf-8") as f:
        f.write(body)

    if deck is not None:
        with open(os.path.join(out, "deck.csv"), "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, deck)) + "\n")
        print(f"deck.csv: 60 cards, {len(set(deck))} unique")
    else:
        print("WARNING: no 60-card DECK literal found; agent must self-supply")

    n = 0
    for raw in find_assets(nb):
        # a tarball asset is the usual shape; unpack it in place
        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as t:
                t.extractall(out)
                n += len(t.getnames())
                continue
        except Exception:
            pass
        # ...and a zip is the other one (map1e114514 ships the whole bundle so)
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                z.extractall(out)
                n += len(z.namelist())
        except Exception:
            pass
    if n:
        print(f"unpacked {n} asset files")

    print(f"built work/agents/{a.name}  (main.py {len(body)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
