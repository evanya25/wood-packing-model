"""
Group model  vs  current rule of thumb
======================================
Model (from the group's formulation), cylinders aligned along the x-axis in a
1 x 1 x 1 m container:

    max  Z = sum( pi * r_i^2 * l_i * s_i ),       s_i in {0, 1}
    s.t. 0 <= x_i,  x_i + l_i <= 1
         r_i <= y_i <= 1 - r_i,   r_i <= z_i <= 1 - r_i
         for every pair i, j that are both chosen:
             x_i + l_i <= x_j   OR   x_j + l_j <= x_i
             OR   (y_i-y_j)^2 + (z_i-z_j)^2 >= (r_i + r_j)^2

Solver: a greedy "drop" heuristic with random restarts (a FEASIBLE solution, so
Z is a lower bound of the true optimum). Every result is checked against the
constraints above by check_feasible().

Input : wood_dataset.csv   (same folder; made by make_wood_dataset.py)
Output: model-vs-current table in the console and wood_model_vs_current.png

Needs only numpy + matplotlib:   pip install numpy matplotlib
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

SEED = 7
N_RESTARTS = 15          # random restarts per scenario (more = better, slower)
N_CAND = 400             # candidate positions tried for every piece
rng = np.random.default_rng(SEED)

CURRENT = {"stere -> scm": 0.70, "lcm -> scm": 0.49, "lcm -> stere": 0.70}

# ----------------------------------------------------------------------
# read the dataset
# ----------------------------------------------------------------------
if not os.path.exists("wood_dataset.csv"):
    raise SystemExit("wood_dataset.csv not found - put it in the same folder "
                     "(or run make_wood_dataset.py first).")
data = {}
with open("wood_dataset.csv") as f:
    for row in csv.DictReader(f):
        data.setdefault(row["scenario"], []).append((float(row["r_m"]), float(row["l_m"])))
data = {k: np.array(v) for k, v in data.items()}
print("scenarios found:", {k: len(v) for k, v in data.items()})


# ----------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------
def pack_once(pieces, order):
    """Place pieces one by one. For each piece try N_CAND (x, y) positions,
    let it fall to its lowest z, keep the lowest feasible one. Pieces that do
    not fit get s_i = 0. Returns lists x, y, z, r, l of the chosen pieces."""
    X, Y, Z, R, L = [np.zeros(len(pieces)) for _ in range(5)]
    n = 0
    for k in order:
        r, l = pieces[k]
        # x candidates: start of the container, ends of placed pieces, random
        pool = np.r_[0.0, X[:n] + L[:n], rng.uniform(0, 1 - l, 60)]
        pool = pool[pool <= 1 - l + 1e-12]
        x = rng.choice(pool, N_CAND)
        y = rng.uniform(r, 1 - r, N_CAND)
        z = np.full(N_CAND, r)
        if n:
            ox = (x[:, None] < X[None, :n] + L[None, :n]) & (X[None, :n] < (x + l)[:, None])
            dy = y[:, None] - Y[None, :n]
            rr = (r + R[None, :n])
            touch = ox & (np.abs(dy) < rr)
            zz = np.where(touch, Z[None, :n] + np.sqrt(np.maximum(rr**2 - dy**2, 0)), 0)
            z = np.maximum(r, zz.max(axis=1))
        ok = z + r <= 1.0
        if not ok.any():
            continue                                   # s_k = 0
        j = np.argmin(np.where(ok, z, np.inf))
        X[n], Y[n], Z[n], R[n], L[n] = x[j], y[j], z[j], r, l
        n += 1                                          # s_k = 1
    return X[:n], Y[:n], Z[:n], R[:n], L[:n]


def check_feasible(X, Y, Z, R, L, tol=1e-9):
    """Verify every constraint of the model."""
    if (X < -tol).any() or (X + L > 1 + tol).any():
        return False
    if (Y < R - tol).any() or (Y > 1 - R + tol).any():
        return False
    if (Z < R - tol).any() or (Z > 1 - R + tol).any():
        return False
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            sep_x = (X[i] + L[i] <= X[j] + tol) or (X[j] + L[j] <= X[i] + tol)
            sep_c = (Y[i] - Y[j])**2 + (Z[i] - Z[j])**2 >= (R[i] + R[j])**2 - tol
            if not (sep_x or sep_c):
                return False
    return True


def solve(pieces):
    """Random restarts; first restart uses pure 'largest radius first'."""
    best, history = None, []
    for t in range(N_RESTARTS):
        key = pieces[:, 0] * (1 + (0 if t == 0 else rng.uniform(-0.2, 0.2, len(pieces))))
        order = np.argsort(-key)
        sol = pack_once(pieces, order)
        Zval = float(np.sum(np.pi * sol[3]**2 * sol[4]))     # objective Z
        history.append(Zval)
        if best is None or Zval > best[0]:
            best = (Zval, sol)
    return best, np.array(history)


# ----------------------------------------------------------------------
# run both scenarios
# ----------------------------------------------------------------------
results, hist, sols = {}, {}, {}
for name in ["S1_logs_1m", "S3_pieces_30_40cm"]:
    (Zbest, sol), h = solve(data[name])
    assert check_feasible(*sol), f"{name}: constraint violated!"
    results[name], hist[name], sols[name] = Zbest, h, sol
    print(f"{name:<20} pieces chosen: {len(sol[0]):>3} of {len(data[name])}   "
          f"best Z = {Zbest:.3f}   (mean of restarts {h.mean():.3f}, "
          f"feasibility check: OK)")

model = {"stere -> scm": results["S1_logs_1m"],
         "lcm -> scm": results["S3_pieces_30_40cm"]}
model["lcm -> stere"] = model["lcm -> scm"] / model["stere -> scm"]

print(f"\n{'factor':<14}{'current':>9}{'model':>9}{'difference':>12}")
for k in CURRENT:
    print(f"{k:<14}{CURRENT[k]:>9.2f}{model[k]:>9.3f}{model[k] - CURRENT[k]:>+12.3f}")

# ----------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# A: model vs current
a = ax[0, 0]
names = list(CURRENT)
xp = np.arange(3)
a.bar(xp - .2, [CURRENT[k] for k in names], .4, color="lightgray", label="current rule of thumb")
a.bar(xp + .2, [model[k] for k in names], .4, color="tab:blue", label="group model (dataset)")
for i, k in enumerate(names):
    a.text(i - .2, CURRENT[k] + .01, f"{CURRENT[k]:.2f}", ha="center", fontsize=9)
    a.text(i + .2, model[k] + .01, f"{model[k]:.2f}", ha="center", fontsize=9)
a.set_xticks(xp)
a.set_xticklabels(names)
a.set(ylabel="conversion factor", ylim=(0, 1.05), title="A  Model vs current values")
a.legend(fontsize=8)

# B: spread over restarts
a = ax[0, 1]
a.boxplot([hist["S1_logs_1m"], hist["S3_pieces_30_40cm"]],
          tick_labels=["S1 logs (stere)", "S3 pieces (lcm)"])
a.axhline(0.70, color="gray", ls="--", label="current stere 0.70")
a.axhline(0.49, color="k", ls="--", label="current lcm 0.49")
a.set(ylabel="Z = solid wood in 1 m^3", title=f"B  Result over {N_RESTARTS} random restarts")
a.legend(fontsize=8)


# C, D: end views of the best packings (looking along the x-axis)
def end_view(a, sol, title, alpha):
    X, Y, Z, R, L = sol
    a.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=2))
    for y, z, r in zip(Y, Z, R):
        a.add_patch(Circle((y, z), r, alpha=alpha, ec="saddlebrown", fc="peru"))
    a.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), aspect="equal", title=title,
          xlabel="y (m)", ylabel="z (m)")


end_view(ax[1, 0], sols["S1_logs_1m"],
         f"C  S1 end view: {len(sols['S1_logs_1m'][0])} logs, Z = {model['stere -> scm']:.2f}", 0.9)
end_view(ax[1, 1], sols["S3_pieces_30_40cm"],
         f"D  S3 end view (pieces overlap in view): Z = {model['lcm -> scm']:.2f}", 0.15)

plt.tight_layout()
plt.savefig("wood_model_vs_current.png", dpi=150)
print("figure saved: wood_model_vs_current.png")
plt.show()
