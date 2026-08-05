"""Submit a tarball to the competition using a KGAT access token.

The MCP server caches its credentials at startup, so a token supplied mid-session
is never picked up and every call returns Unauthenticated. The REST API accepts
the token directly as a bearer credential, so this goes straight there.

Flow (same one the kaggle CLI uses):
  1. POST /api/v1/blobs/upload            -> {token, createUrl}
  2. POST createUrl  (X-Goog-Resumable)   -> Location: resumable session URI
  3. PUT  bytes to that session URI
  4. POST /api/v1/competitions/submissions/submit/<comp> with the blob token

Usage: python work/tools/kaggle_submit.py <file.tar.gz> "<description>"
"""
import json
import os
import sys
import urllib.error
import urllib.request

COMP = "pokemon-tcg-ai-battle"
BASE = "https://www.kaggle.com/api/v1"
TOKEN_PATH = os.path.join(os.path.expanduser("~"), ".kaggle", "access_token")


def token():
    t = os.environ.get("KAGGLE_ACCESS_TOKEN")
    if not t and os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH) as fh:
            t = fh.read().strip()
    if not t:
        raise SystemExit("no access token found")
    return t


def req(url, data=None, headers=None, method=None, auth=True, label=""):
    h = {}
    if auth:
        h["Authorization"] = f"Bearer {token()}"
    h.update(headers or {})
    r = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:600]
        print(f"  !! {label or url[:60]} -> HTTP {e.code}: {body}")
        raise


def main():
    path = sys.argv[1]
    desc = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(path)
    size = os.path.getsize(path)
    name = os.path.basename(path)
    print(f"submitting {name} ({size/1e6:.2f} MB)")
    print(f"  description: {desc}")

    body = json.dumps({
        "type": "COMPETITION_SOLUTION",
        "name": name,
        "contentLength": size,
        "lastModifiedEpochSeconds": int(os.path.getmtime(path)),
    }).encode()
    st, out, _ = req(f"{BASE}/blobs/upload", data=body,
                     headers={"Content-Type": "application/json"},
                     label="blobs/upload")
    info = json.loads(out)
    create_url, blob_token = info["createUrl"], info["token"]
    print(f"  blob token: {blob_token[:24]}...")

    # createUrl already carries upload_id, i.e. the resumable session is open.
    # POSTing to it again just returns 308 Resume Incomplete; PUT the bytes.
    session = create_url

    with open(path, "rb") as fh:
        payload = fh.read()
    st, _o, _h = req(session, data=payload, method="PUT", auth=False,
                     headers={"Content-Type": "application/octet-stream"},
                     label="gcs PUT")
    print(f"  upload status: {st}")

    # submit
    form = urllib.parse.urlencode({
        "blobFileTokens": blob_token,
        "submissionDescription": desc,
    }).encode()
    st, out, _ = req(f"{BASE}/competitions/submissions/submit/{COMP}", data=form,
                     headers={"Content-Type": "application/x-www-form-urlencoded"},
                     label="submissions/submit")
    print(f"  submit status: {st}")
    print(f"  response: {out.decode()[:400]}")


if __name__ == "__main__":
    import urllib.parse  # noqa: E402
    main()
