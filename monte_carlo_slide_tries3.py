"""
Monte Carlo slide - random packing of 1 m logs in a 1 m^3 container (TRIES = 3)
Group 1 : Estimating the Amount of Woods

Makes the slide graph (monte_carlo_slide_tries3.png), a backup convergence graph
(monte_carlo_convergence_tries3.png) and prints the numbers to quote on the slide.

Run:  python monte_carlo_slide_tries3.py     (needs numpy, pandas, matplotlib, scipy)
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------- SETTINGS --------------------------------------
CSV_PATH      = "wood_dataset.csv"
SCENARIO      = "S1_logs_1m"
N_TRIALS      = 1000
TRIES         = 3                 # each piece tries 3 random spots, settles at the lowest
RULE_OF_THUMB = 0.49
GROUP_MODEL   = 0.53
SEED          = 42
L = W = H = 1.0
# -----------------------------------------------------------------------------

def run_trial(r, l, rng, tries=1):
    """One random packing. Returns packed volume and the placed pieces."""
    placed = []                                   # (x0, y, z, r, l)
    px = np.empty(0); py = np.empty(0); pz = np.empty(0)
    pr = np.empty(0); pl = np.empty(0)
    vol = 0.0
    for i in rng.permutation(len(r)):
        ri, li = r[i], l[i]
        if li > L or 2 * ri > W:
            continue
        best = None
        for _ in range(tries):
            x0 = rng.uniform(0, L - li)           # start of the piece along x
            y = rng.uniform(ri, W - ri)           # centre of circle across width
            z = ri                                # start on the floor
            if len(px):
                m = (px < x0 + li) & (px + pl > x0)        # overlap along length
                if m.any():
                    dy = np.abs(py[m] - y)
                    s = pr[m] + ri
                    k = dy < s                              # circles overlap in y
                    if k.any():
                        z = max(z, (pz[m][k] + np.sqrt(s[k] ** 2 - dy[k] ** 2)).max())
            if z + ri <= H + 1e-12 and (best is None or z < best[2]):
                best = (x0, y, z)
        if best is not None:
            x0, y, z = best
            px = np.append(px, x0); py = np.append(py, y); pz = np.append(pz, z)
            pr = np.append(pr, ri); pl = np.append(pl, li)
            placed.append((x0, y, z, ri, li))
            vol += np.pi * ri ** 2 * li
    return vol, placed


def has_overlap(placed, tol=1e-9):
    """Validation: True if any two cylinders overlap or one leaves the container."""
    a = np.array(placed)
    x0, y, z, r, l = a.T
    if (x0 < -tol).any() or (x0 + l > L + tol).any(): return True
    if (y - r < -tol).any() or (y + r > W + tol).any(): return True
    if (z - r < -tol).any() or (z + r > H + tol).any(): return True
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if x0[i] < x0[j] + l[j] and x0[j] < x0[i] + l[i]:
                if np.hypot(y[i] - y[j], z[i] - z[j]) < r[i] + r[j] - tol:
                    return True
    return False




def main():
    d = pd.read_csv(CSV_PATH).query("scenario == @SCENARIO")
    r, l = d.r_m.values, d.l_m.values
    rng = np.random.default_rng(SEED)

    x, best_v, best_layout = [], -1, None
    for _ in range(N_TRIALS):
        v, placed = run_trial(r, l, rng, TRIES)
        x.append(v)
        if v > best_v: best_v, best_layout = v, placed
    x = np.array(x)
    assert not has_overlap(best_layout), "overlap found!"

    n = len(x); mean = x.mean(); sd = x.std(ddof=1); se = sd / np.sqrt(n)
    lo, hi = mean - 1.96*se, mean + 1.96*se
    p49 = stats.ttest_1samp(x, RULE_OF_THUMB).pvalue
    print(f"Trials                : {n}")
    print(f"Mean packing ratio    : {mean:.3f}   (SD {sd:.3f})")
    print(f"95% CI of the mean    : [{lo:.3f}, {hi:.3f}]   (+-{100*1.96*se/mean:.2f}% relative)")
    print(f"Rule of thumb 0.49    : diff {mean-RULE_OF_THUMB:+.3f}, p = {p49:.3f}")
    print(f"Group model 0.53      : diff {mean-GROUP_MODEL:+.3f}")
    print(f"Trials >= 0.49        : {100*(x>=RULE_OF_THUMB).mean():.1f} %")
    print(f"Trials >= 0.53        : {100*(x>=GROUP_MODEL).mean():.1f} %")
    print(f"Min / Max trial       : {x.min():.3f} / {x.max():.3f}")
    print(f"Middle 90% of trials  : {np.percentile(x,5):.3f} - {np.percentile(x,95):.3f}")

    # ------------------ SLIDE GRAPH (16:9) -------------------------------------
    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(13.33, 7.0))
    ax.hist(x, bins=30, color="#4C9F70", edgecolor="white")
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.22)
    ax.axvline(mean, color="navy", lw=3, label=f"Monte Carlo mean = {mean:.3f}")
    ax.axvline(RULE_OF_THUMB, color="#7F8791", lw=3, ls="--", label=f"Current rule of thumb = {RULE_OF_THUMB:.2f}")
    ax.axvline(GROUP_MODEL, color="#0072B2", lw=3, ls="--", label=f"Group model = {GROUP_MODEL:.2f}")
    ax.set(xlabel="Packing ratio (solid wood per 1 m³ of loose pile)",
           ylabel=f"Number of trials (out of {n})",
           title="Monte Carlo simulation: random packing of 1 m logs")
    ax.title.set_fontsize(22); ax.title.set_fontweight("bold")
    ax.legend(loc="upper left", frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.text(0.01, 0.01,
             f"{n} random packings, each piece tries {TRIES} random positions and settles at the lowest. "
             f"95% CI of mean: {lo:.3f}-{hi:.3f}.", fontsize=11, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("monte_carlo_slide_tries3.png", dpi=200); plt.close(fig)

    # ------------------ BACKUP: convergence ------------------------------------
    k = np.arange(1, n + 1); rm = np.cumsum(x) / k
    rse = np.array([x[:i].std(ddof=1) / np.sqrt(i) if i > 1 else np.nan for i in k])
    fig, ax = plt.subplots(figsize=(13.33, 7.0))
    ax.plot(k, rm, color="navy", lw=2.5, label="Running mean of Monte Carlo")
    ax.fill_between(k, rm - 1.96*rse, rm + 1.96*rse, color="navy", alpha=0.2, label="95% CI")
    ax.axhline(RULE_OF_THUMB, color="#7F8791", lw=3, ls="--", label="Current rule of thumb = 0.49")
    ax.axhline(GROUP_MODEL, color="#0072B2", lw=3, ls="--", label="Group model = 0.53")
    ax.set(xlabel="Number of trials", ylabel="Average packing ratio",
           title="Monte Carlo estimate stabilises as trials increase")
    ax.title.set_fontsize(22); ax.title.set_fontweight("bold")
    ax.legend(frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig("monte_carlo_convergence_tries3.png", dpi=200); plt.close(fig)
    print("\nSaved: monte_carlo_slide_tries3.png, monte_carlo_convergence_tries3.png")


if __name__ == "__main__":
    main()
