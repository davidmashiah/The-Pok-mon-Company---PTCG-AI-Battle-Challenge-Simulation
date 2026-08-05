"""Gate probe: does the learned re-ranker actually load and score FROM THE
STAGED BUNDLE?

Run as: python dz_probe.py <stage_dir>   -- prints 'DZOK <state_nf> <logit>'
and exits 0 only if the weights loaded and produce finite scores.

This exists because a component imported inside try/except does not announce
its own absence. Without this check, a missing dz_weights.npz would silently
turn the agent back into plain v14 and every log would look normal.
"""
import sys

import numpy as np

stage = sys.argv[1]
sys.path.insert(0, stage)

import dzfeat  # noqa: E402
import dznp  # noqa: E402

assert dznp.load(), "weights did not load from the staged bundle"
nf = dznp.state_nf()
assert nf == dzfeat.NF, f"state_nf mismatch: weights {nf} vs features {dzfeat.NF}"

s = np.zeros(nf, np.float32)
a = np.zeros((dzfeat.MAX_CAND, dzfeat.ACT_NF), np.float32)
c = np.zeros(dzfeat.MAX_CAND, np.int32)
m = np.zeros(dzfeat.MAX_CAND, np.float32)
m[:3] = 1.0
ht = np.zeros(dzfeat.HIST, np.int32)
hc = np.zeros(dzfeat.HIST, np.int32)

lg = dznp.logits(s, a, c, m, ht, hc)
assert np.isfinite(lg[:3]).all(), "non-finite logits"
assert lg[3] < -1e8, "masked options are not masked"

print("DZOK", nf, float(lg[0]))
