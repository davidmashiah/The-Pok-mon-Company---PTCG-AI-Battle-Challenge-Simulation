"""Download our own ladder episodes, so the real field can be measured.

Why this is needed. `our_field.py` reads our downloaded replays, and of the 155
we hold, **126 were played with decks we no longer run** -- 91 Mega Lucario, 35
Alakazam. Only 29 are the Grimmsnarl 60 we ship, with per-archetype samples of 1
to 12. Every "real ladder win rate" computed from that archive without a deck
filter mixed three eras and was wrong; see the retraction in memory.

The MCP kaggle server exposes list_submission_episodes / get_episode_replay, but
it caches credentials at startup and is not always connected. The REST endpoints
take the same KGAT bearer token `kaggle_submit.py` uses, so this goes straight
there and works headless.

Saves each episode as JSON in the same shape our tooling already parses
(`steps`, `rewards`, `info.TeamNames`), skipping any episode already on disk so
it can be re-run to top up.

  python work/tools/fetch_episodes.py --submission 55387402 --limit 60
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
TOKEN_PATH = os.path.join(os.path.expanduser("~"), ".kaggle", "access_token")
BASE = "https://www.kaggle.com/api/i/competitions.EpisodeService"


def token():
    t = os.environ.get("KAGGLE_ACCESS_TOKEN")
    if not t and os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH) as fh:
            t = fh.read().strip()
    if not t:
        raise SystemExit("no access token at ~/.kaggle/access_token")
    return t


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}/{path}", data=data, method="POST",
        headers={"Authorization": f"Bearer {token()}",
                 "Content-Type": "application/json",
                 "User-Agent": "ptcg-analysis"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read()[:300].decode(errors="replace")
        print(f"  HTTP {e.code} on {path}: {body}")
        return None
    except Exception as exc:
        print(f"  {type(exc).__name__} on {path}: {exc}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=int, action="append", required=True,
                    help="submission id; repeatable")
    ap.add_argument("--out", default=os.path.join(WORK, "out", "cur_replays"))
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    got = 0
    for sub in a.submission:
        print(f"submission {sub}")
        listing = post("ListEpisodes", {"submissionId": sub})
        if not listing:
            continue
        eps = listing.get("episodes") or []
        print(f"  {len(eps)} episodes listed")
        for ep in eps:
            if got >= a.limit:
                break
            eid = ep.get("id") or ep.get("episodeId")
            if eid is None:
                continue
            dst = os.path.join(a.out, f"{eid}.json")
            if os.path.exists(dst):
                continue
            rep = post("GetEpisodeReplay", {"episodeId": eid})
            if not rep:
                continue
            raw = rep.get("replay")
            try:
                obj = json.loads(raw) if isinstance(raw, str) else (raw or rep)
            except Exception:
                obj = rep
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(obj, f)
            got += 1
            if got % 10 == 0:
                print(f"    {got} saved", flush=True)
            time.sleep(0.4)          # be polite to the endpoint
    print(f"\nsaved {got} episodes -> {a.out}")
    print("Then: python work/tools/our_field.py  (it filters to our current 60)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
