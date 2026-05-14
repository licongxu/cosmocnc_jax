"""Generate posterior plots comparing hmfast vs classy_sz_jax cobaya chains."""
import os
import sys

import numpy as np

# Load chain data
chain_dir = os.path.join(os.path.dirname(__file__), "chains")


def load_chain(tag):
    """Load a cobaya chain by tag prefix."""
    # Find chain files
    paths = [
        os.path.join(chain_dir, f"cnc_planck_scatter_{tag}.1.txt"),
        os.path.join(chain_dir, f"cnc_planck_scatter_{tag}.txt"),
    ]
    fpath = None
    for p in paths:
        if os.path.exists(p):
            fpath = p
            break
    if fpath is None:
        raise FileNotFoundError(f"No chain file found for {tag}")

    data = np.loadtxt(fpath, dtype=float)
    # First column is weight, second is minuslogpost, then params
    weights = data[:, 0]
    minuslogpost = data[:, 1]

    # Read header to get param names
    with open(fpath) as f:
        header = f.readline().strip("# ").strip().split()
    param_names = header[2:]  # skip weight, minuslogpost

    # Extract params (excluding derived like omega_cdm, B)
    params = {}
    for i, name in enumerate(param_names):
        if i + 2 < data.shape[1]:
            params[name] = data[:, i + 2]

    return weights, minuslogpost, params


def plot_comparison():
    print("Loading chains...")
    try:
        w_hmfast, mlogp_hmfast, p_hmfast = load_chain("hmfast")
        print(f"  hmfast:         {len(w_hmfast)} samples")
    except FileNotFoundError as e:
        print(f"  hmfast not found: {e}")
        p_hmfast = None

    try:
        w_classy, mlogp_classy, p_classy = load_chain("jax")
        print(f"  classy_sz_jax:  {len(w_classy)} samples")
    except FileNotFoundError as e:
        print(f"  classy_sz_jax not found: {e}")
        p_classy = None

    # Plot params of interest
    plot_params = ["H0", "ln10_10A_s", "n_s", "omega_b", "Omega_m",
                   "A_SZ", "alpha_SZ", "sigma_lnY"]

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.ravel()

    for ax_idx, pname in enumerate(plot_params):
        ax = axes[ax_idx]
        bins = 40

        if p_hmfast is not None and pname in p_hmfast:
            vals_h = p_hmfast[pname]
            vals_h = vals_h[np.isfinite(vals_h)]
            # Use burn-in of 0.5
            burn = len(vals_h) // 2
            ax.hist(vals_h[burn:], bins=bins, density=True, alpha=0.7,
                    label="hmfast", color="tab:blue", histtype="step", lw=1.5)
            ax.axvline(np.median(vals_h[burn:]), color="tab:blue", ls="--", lw=1)

        if p_classy is not None and pname in p_classy:
            vals_c = p_classy[pname]
            vals_c = vals_c[np.isfinite(vals_c)]
            burn = len(vals_c) // 2
            ax.hist(vals_c[burn:], bins=bins, density=True, alpha=0.7,
                    label="classy_sz", color="tab:orange", histtype="step", lw=1.5)
            ax.axvline(np.median(vals_c[burn:]), color="tab:orange", ls="--", lw=1)

        ax.set_xlabel(pname, fontsize=10)
        ax.set_ylabel("Density", fontsize=9)
        if ax_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Posterior comparison: hmfast vs classy_sz_jax (Planck scatter)", fontsize=13)
    plt.tight_layout()
    outpath = os.path.join(chain_dir, "posterior_comparison.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved posterior plot to {outpath}")
    plt.close(fig)

    # Print summary stats
    print()
    print("=" * 70)
    print("SUMMARY STATISTICS (burn-in = last 50%)")
    print("=" * 70)
    print(f"{'Param':<16s} {'hmfast_mean':>12s} {'hmfast_std':>12s} "
          f"{'classy_mean':>12s} {'classy_std':>12s}")
    print("-" * 70)

    for pname in plot_params:
        vals_h = p_hmfast[pname] if (p_hmfast and pname in p_hmfast) else None
        vals_c = p_classy[pname] if (p_classy and pname in p_classy) else None

        def fmt(v):
            if v is None:
                return ("       N/A", "       N/A")
            v = v[np.isfinite(v)]
            burn = len(v) // 2
            return f"{np.mean(v[burn:]):12.4f}", f"{np.std(v[burn:]):12.4f}"

        m_h, s_h = fmt(vals_h)
        m_c, s_c = fmt(vals_c)
        print(f"{pname:<16s} {m_h:>12s} {s_h:>12s} {m_c:>12s} {s_c:>12s}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_comparison()
