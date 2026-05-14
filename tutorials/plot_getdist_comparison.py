"""GetDist-based posterior comparison: hmfast vs classy_sz_jax.

Matches the plotting style of plot_cnc_only_posterior_normalized.ipynb.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from getdist.mcsamples import loadMCSamples
from getdist import plots
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 22,
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    "text.latex.preamble": (
        r"\usepackage[T1]{fontenc}"
        r"\usepackage{type1cm}"
        r"\usepackage{amsmath}"
        r"\usepackage{amssymb}"
        r"\usepackage{siunitx}"
    ),
})

TUTORIAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TUTORIAL_DIR)
CHAINS_DIR = os.path.join(TUTORIAL_DIR, "chains")

# Shared constants
one_minus_b_const = 0.709
fid = {
    "Omega_m": 0.309576,
    "sigma8": 0.78,
    "S8": 0.78 * np.sqrt(0.309576 / 0.3),
    "F": 0.78 * (0.309576 * one_minus_b_const)**0.40 * (67.66/100.)**(-0.21),
    "A_SZ": -4.31,
    "alpha_SZ": 1.12,
    "sigma_lnY": 0.173,
}


def load_and_add_derived(chain_root, label_name):
    """Load a cobaya chain with getdist, derive sigma8/S8/F."""
    from classy_sz import Class as Class_sz

    classy_sz_init = {
        "omega_b": 0.02242,
        "omega_cdm": 0.11933,
        "H0": 67.66,
        "tau_reio": 0.0544,
        "ln10^{10}A_s": 2.9718,
        "n_s": 0.9665,
    }

    # Init classy_sz for sigma8 computation (non-JAX path)
    classy = Class_sz()
    classy.set(classy_sz_init)
    classy.compute_class_szfast()

    s = loadMCSamples(chain_root, settings={"ignore_rows": 0.3})
    p = s.getParams()

    sigma8_values = np.empty(len(p.H0), dtype=float)
    for i in range(len(p.H0)):
        params_dict = {
            "omega_b": float(p.omega_b[i]),
            "omega_cdm": float(p.omega_cdm[i]),
            "H0": float(p.H0[i]),
            "n_s": float(p.n_s[i]),
            "ln10^{10}A_s": float(p.ln10_10A_s[i]),
        }
        sigma8_values[i] = classy.get_sigma8_and_der(params_values_dict=params_dict)[1]

    s.addDerived(sigma8_values, name="sigma8", label=r"\sigma_8")
    p = s.getParams()
    S8 = p.sigma8 * np.sqrt(p.Omega_m / 0.3)
    s.addDerived(S8, name="S8", label=r"S_8")
    p = s.getParams()
    F = p.sigma8 * (p.Omega_m * one_minus_b_const)**0.40 * (p.H0 / 100.)**(-0.21)
    s.addDerived(F, name="F", label=r"F")

    # Add nice labels
    s.updateBaseStatistics()
    # Massage param labels for display
    for name, lab in [
        ("Omega_m", r"\Omega_m"),
        ("sigma8", r"\sigma_8"),
        ("S8", r"S_8"),
        ("H0", r"H_0"),
        ("ln10_10A_s", r"\ln(10^{10}A_s)"),
        ("n_s", r"n_s"),
        ("omega_b", r"\Omega_b h^2"),
        ("omega_cdm", r"\Omega_c h^2"),
        ("A_SZ", r"A_\mathrm{SZ}"),
        ("alpha_SZ", r"\alpha_\mathrm{SZ}"),
        ("sigma_lnY", r"\sigma_{\ln Y}"),
    ]:
        par = s.getParamNames().parWithName(name)
        if par is not None:
            par.label = lab

    print(f"Loaded {label_name}: {s.numrows} samples")
    return s


if __name__ == "__main__":
    chain_jax = os.path.join(CHAINS_DIR, "cnc_planck_scatter_jax")
    chain_hmfast = os.path.join(CHAINS_DIR, "cnc_planck_scatter_hmfast")

    print("Loading classy_sz_jax chain...")
    s_jax = load_and_add_derived(chain_jax, "classy_sz_jax")
    print("Loading hmfast chain...")
    s_hmfast = load_and_add_derived(chain_hmfast, "hmfast")

    params_to_plot = [
        "Omega_m", "sigma8", "S8", "F",
        "H0", "ln10_10A_s", "n_s", "omega_b",
        "A_SZ", "alpha_SZ", "sigma_lnY",
    ]

    # Print summary stats
    print()
    for label, s in [("classy_sz_jax", s_jax), ("hmfast", s_hmfast)]:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        stats = s.getMargeStats()
        for pname in params_to_plot:
            lim = stats.parWithName(pname)
            if lim is not None:
                print(f"  {pname:14s}: {lim.mean:.6g} +/- {lim.err:.6g}")

    # Triangle plot
    g = plots.get_subplot_plotter(width_inch=20)
    g.settings.lab_fontsize = 16
    g.settings.axes_fontsize = 12
    g.settings.legend_fontsize = 18

    g.triangle_plot(
        [s_jax, s_hmfast],
        params_to_plot,
        filled=True,
        contour_colors=["tab:orange", "tab:blue"],
        legend_labels=["classy\\_sz\\_jax", "hmfast"],
        markers={k: fid[k] for k in params_to_plot if k in fid},
        marker_args={"ms": 8, "mfc": "red", "mec": "darkred"},
    )

    # Fiducial lines
    vline_style = {"color": "red", "ls": "--", "lw": 1.2, "alpha": 0.7}
    hline_style = {"color": "red", "ls": "--", "lw": 1.2, "alpha": 0.7}
    for i, p1 in enumerate(params_to_plot):
        for j, p2 in enumerate(params_to_plot):
            ax = g.subplots[i, j]
            if ax is None or p1 not in fid or p2 not in fid:
                continue
            if i == j:
                ax.axvline(fid[p1], **vline_style)
            else:
                ax.axvline(fid[p2], **vline_style)
                ax.axhline(fid[p1], **hline_style)

    outpath = os.path.join(CHAINS_DIR, "posterior_comparison_getdist.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"\nSaved to {outpath}")
    plt.close()
