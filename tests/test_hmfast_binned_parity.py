"""Stress test: binned N(z,q) histogram parity between hmfast and classy_sz_jax.

Verifies that the cluster number count binned histogram matches between
the hmfast cosmology backend and the reference classy_sz_jax backend,
using the same Planck scatter setup as the cobaya likelihood.
"""
import os
import sys
import time

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.30"

# Prevent TF from claiming CUDA context before JAX initialization.
_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf  # noqa: E402

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_visible_devices

import numpy as np
import jax
import jax.numpy as jaxnp


def _build_cnc_params(survey_sr, survey_cat, tszsbi_noise_dir, tszsbi_filter_name,
                      bin_edges_z, bin_edges_q, z_min, z_max, q_min, q_max,
                      n_points, n_z, M_min, M_max, M_pivot):
    from cosmocnc_jax.params import cnc_params_default

    cnc_params = dict(cnc_params_default)
    cnc_params["survey_sr"] = survey_sr
    cnc_params["survey_cat"] = survey_cat
    cnc_params["tszsbi_noise_dir"] = tszsbi_noise_dir
    cnc_params["tszsbi_filter_name"] = tszsbi_filter_name

    cnc_params["load_catalogue"] = False
    cnc_params["likelihood_type"] = "binned"
    cnc_params["binned_lik_type"] = "z_and_obs_select"
    cnc_params["data_lik_from_abundance"] = False

    cnc_params["obs_select"] = "q_planck_sim"
    cnc_params["observables"] = [["q_planck_sim"]]
    cnc_params["obs_select_min"] = q_min
    cnc_params["obs_select_max"] = q_max
    cnc_params["z_min"] = z_min
    cnc_params["z_max"] = z_max
    cnc_params["bins_edges_z"] = bin_edges_z
    cnc_params["bins_edges_obs_select"] = bin_edges_q

    cnc_params["n_points"] = int(n_points)
    cnc_params["n_z"] = int(n_z)
    cnc_params["M_min"] = float(M_min)
    cnc_params["M_max"] = float(M_max)
    cnc_params["planck_sim_M_pivot"] = float(M_pivot)

    cnc_params["hmf_calc"] = "cnc"
    cnc_params["cosmo_param_density"] = "physical"
    cnc_params["cosmo_amplitude_parameter"] = "A_s"
    cnc_params["cosmocnc_verbose"] = "none"

    return cnc_params


def _run_binned_test(cosmology_tool, cosmo_params, scalrel_params, cnc_params_base):
    """Run one binned N(z,q) computation and return the histogram + timing."""
    from cosmocnc_jax import cluster_number_counts

    cnc_params = dict(cnc_params_base)
    cnc_params["cosmology_tool"] = cosmology_tool

    cnc = cluster_number_counts(cnc_params=cnc_params)
    cnc.cosmo_params = dict(cosmo_params)
    cnc.scal_rel_params = dict(scalrel_params)

    t0 = time.perf_counter()
    cnc.initialise()
    init_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    _ = cnc.get_log_lik_binned()
    theory_time = time.perf_counter() - t1

    n_binned = np.asarray(cnc.n_binned, dtype=float)
    return n_binned, init_time, theory_time


def main():
    # --- Setup: same paths and params as cobaya_planck_scatter_jax.yaml ---
    survey_sr = "/scratch/scratch-lxu/compute_packages/cosmocnc_jax/cosmocnc_jax/surveys/survey_sr_planck_sim.py"
    survey_cat = "/scratch/scratch-lxu/compute_packages/cosmocnc_jax/cosmocnc_jax/surveys/survey_cat_planck_sim.py"
    tszsbi_noise_dir = "/scratch/scratch-lxu/tszsbi/noise_files"
    tszsbi_filter_name = "immf6"

    z_min, z_max = 0.005, 1.0
    n_z_bins = 10
    q_min, q_max = 5.0, 40.0
    n_q_bins = 5
    bin_edges_z = np.linspace(z_min, z_max, n_z_bins + 1)
    bin_edges_q = np.exp(np.linspace(np.log(q_min), np.log(q_max), n_q_bins + 1))

    cnc_params_base = _build_cnc_params(
        survey_sr, survey_cat, tszsbi_noise_dir, tszsbi_filter_name,
        bin_edges_z, bin_edges_q, z_min, z_max, q_min, q_max,
        n_points=2048, n_z=50, M_min=1e14, M_max=1e16, M_pivot=2.1e14,
    )

    # --- Cosmological and scaling relation parameters ---
    from cosmocnc_jax.params import cosmo_params_default, scaling_relation_params_default

    cosmo = dict(cosmo_params_default)
    # Use physical density params as in the cobaya YAML
    cosmo["Ob0h2"] = 0.022
    cosmo["Oc0h2"] = 0.3096 * 0.674**2 - 0.022
    cosmo["h"] = 0.674
    cosmo["A_s"] = 1.0e-10 * np.exp(2.9718)
    cosmo["n_s"] = 0.962
    cosmo["tau_reio"] = 0.0544
    cosmo["m_nu"] = 0.06

    scal = dict(scaling_relation_params_default)
    scal["A_szifi"] = -4.31
    scal["alpha_szifi"] = 1.12
    scal["sigma_lnq_szifi"] = 0.173
    scal["bias_sz"] = 0.709

    # --- Run both backends ---
    print(f"JAX devices: {jax.devices()}")
    print()

    # Reference: classy_sz_jax
    print("=" * 60)
    print("Running CLASSY_SZ_JAX (reference)...")
    print("=" * 60)
    n_binned_classy, init_classy, theory_classy = _run_binned_test(
        "classy_sz_jax", cosmo, scal, cnc_params_base)
    print(f"  Init time:      {init_classy:.3f}s")
    print(f"  Theory time:    {theory_classy:.4f}s")
    print(f"  Binned shape:   {n_binned_classy.shape}")
    print(f"  Total counts:   {n_binned_classy.sum():.4f}")
    print(f"  Per-bin counts:\n{n_binned_classy}")
    print()

    # New: hmfast
    print("=" * 60)
    print("Running HMFAST (new)...")
    print("=" * 60)
    n_binned_hmfast, init_hmfast, theory_hmfast = _run_binned_test(
        "hmfast", cosmo, scal, cnc_params_base)
    print(f"  Init time:      {init_hmfast:.3f}s")
    print(f"  Theory time:    {theory_hmfast:.4f}s")
    print(f"  Binned shape:   {n_binned_hmfast.shape}")
    print(f"  Total counts:   {n_binned_hmfast.sum():.4f}")
    print(f"  Per-bin counts:\n{n_binned_hmfast}")
    print()

    # --- Comparison ---
    print("=" * 60)
    print("PARITY CHECK: hmfast vs classy_sz_jax")
    print("=" * 60)

    abs_diff = np.abs(n_binned_hmfast - n_binned_classy)
    rel_diff = abs_diff / np.maximum(np.abs(n_binned_classy), 1e-10)

    print(f"  Max absolute difference:     {abs_diff.max():.6e}")
    print(f"  Mean absolute difference:    {abs_diff.mean():.6e}")
    print(f"  Max relative difference:     {rel_diff.max():.6e}")
    print(f"  Mean relative difference:    {rel_diff.mean():.6e}")
    print()

    # Check each bin
    print("  Per-bin comparison (z_bin, q_bin): classy_sz  hmfast  rel_diff")
    for iz in range(n_z_bins):
        for iq in range(n_q_bins):
            cv = n_binned_classy[iz, iq]
            hv = n_binned_hmfast[iz, iq]
            rd = abs(cv - hv) / max(abs(cv), 1e-10)
            flag = " <<<" if rd > 0.01 else ""
            print(f"    ({iz:2d}, {iq:2d}): {cv:12.4f}  {hv:12.4f}  {rd:.6e}{flag}")

    # Timing comparison
    print()
    print(f"  Timing: classy_sz_jax theory={theory_classy:.4f}s, hmfast theory={theory_hmfast:.4f}s")
    speedup = theory_classy / theory_hmfast if theory_hmfast > 0 else float("inf")
    print(f"  Speedup factor: {speedup:.2f}x")

    # Verdict
    max_rel = rel_diff.max()
    if max_rel < 0.01:
        print(f"\n  PASS: max relative difference {max_rel:.6e} < 1%")
    elif max_rel < 0.05:
        print(f"\n  WARN: max relative difference {max_rel:.6e} between 1-5%")
    else:
        print(f"\n  FAIL: max relative difference {max_rel:.6e} > 5%")

    return max_rel


if __name__ == "__main__":
    max_rel = main()
    sys.exit(0 if max_rel < 0.05 else 1)
