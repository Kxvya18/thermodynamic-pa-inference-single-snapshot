from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ba_inference import generate_ba_beta, reconstruct_with_mle_beta, degree_collision_bound
import numpy as np

taus, Cs = [], []
for seed in range(3):
    G, hist = generate_ba_beta(1000, 2, 1.0, seed=seed)
    dcb = degree_collision_bound(G)
    rec = reconstruct_with_mle_beta(G, hist, m=2, beam_width=6)
    taus.append(rec['accuracy']['kendall_tau'])
    Cs.append(dcb['collision_prob'])
    print(f"seed={seed}: tau={taus[-1]:.3f}, C={Cs[-1]:.3f}")

print(f"Mean: tau={np.mean(taus):.3f}, C={np.mean(Cs):.3f}")
