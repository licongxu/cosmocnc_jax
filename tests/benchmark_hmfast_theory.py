"""Benchmark: measure per-evaluation theory computation time for hmfast.

After JIT warmup, the theory computation time should be < 0.04s (faster
than the previous classy_sz_jax benchmark of 0.04s per evaluation).
"""
import os
import sys
import time

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.30"

# Prevent TF from claiming CUDA context before JAX
_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_visible_devices

import numpy as np
import jax

from cosmocnc_jax import cluster_number_counts
from cosmocnc_jax.params import (
    cnc_params_default,
    cosmo_params_default,
    scaling_relation_params_default,
)


def build_cnc(cosmology_tool):
    survey_sr = "/scratch/scratch-lxu/compute_packages/cosmocnc_jax/cosmocnc_jax/surveys/survey_sr_planck_sim.py"
    survey_cat = "/scratch/scratch-lxu/compute_packages/cosmocnc_jax/cosmocnc_jax/surveys/survey_cat_planck_sim.py"
    z_min, z_max = 0.005, 1.0
    n_z_bins, n_q_bins = 10, 5
    q_min, q_max = 5.0, 40.0

    cnc_params = dict(cnc_params_default)
    cnc_params["survey_sr"] = survey_sr
    cnc_params["survey_cat"] = survey_cat
    cnc_params["tszsbi_noise_dir"] = "/scratch/scratch-lxu/tszsbi/noise_files"
    cnc_params["tszsbi_filter_name"] = "immf6"
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
    cnc_params["bins_edges_z"] = np.linspace(z_min, z_max, n_z_bins + 1)
    cnc_params["bins_edges_obs_select"] = np.exp(
        np.linspace(np.log(q_min), np.log(q_max), n_q_bins + 1))
    cnc_params["n_points"] = 2048
    cnc_params["n_z"] = 50
    cnc_params["M_min"] = 1e14
    cnc_params["M_max"] = 1e16
    cnc_params["planck_sim_M_pivot"] = 2.1e14
    cnc_params["cosmology_tool"] = cosmology_tool
    cnc_params["hmf_calc"] = "cnc"
    cnc_params["cosmo_param_density"] = "physical"
    cnc_params["cosmo_amplitude_parameter"] = "A_s"
    cnc_params["cosmocnc_verbose"] = "none"

    cosmo = dict(cosmo_params_default)
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

    cnc = cluster_number_counts(cnc_params=cnc_params)
    cnc.cosmo_params = dict(cosmo)
    cnc.scal_rel_params = dict(scal)
    cnc.initialise()

    return cnc, cosmo, scal


def benchmark_evaluations(cnc, cosmo_base, scal_base, n_eval=20, label=""):
    """Run n_eval evaluations with small parameter perturbations and report timing."""
    rng = np.random.RandomState(42)
    times = []

    for i in range(n_eval):
        cosmo = dict(cosmo_base)
        scal = dict(scal_base)

        # Small perturbation to mimic MCMC proposals
        cosmo["h"] = cosmo_base["h"] + rng.normal(0, 0.002)
        cosmo["A_s"] = cosmo_base["A_s"] * (1 + rng.normal(0, 0.005))
        cosmo["n_s"] = cosmo_base["n_s"] + rng.normal(0, 0.002)
        cosmo["Ob0h2"] = cosmo_base["Ob0h2"] + rng.normal(0, 0.0002)
        cosmo["Oc0h2"] = 0.3096 * (cosmo["h"] * 100)**2 * 0.01 - cosmo["Ob0h2"]
        scal["A_szifi"] = scal_base["A_szifi"] + rng.normal(0, 0.01)
        scal["alpha_szifi"] = scal_base["alpha_szifi"] + rng.normal(0, 0.01)
        scal["sigma_lnq_szifi"] = scal_base["sigma_lnq_szifi"] + rng.normal(0, 0.005)

        t0 = time.perf_counter()
        cnc.update_params(cosmo, scal)
        _ = cnc.get_log_lik_binned()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    # Skip first 3 (JIT warmup)
    warm_times = times[:3]
    steady_times = times[3:]

    print(f"\n{label}")
    print(f"  Total evaluations: {n_eval}")
    print(f"  Warmup (first 3): mean={warm_times.mean():.4f}s, max={warm_times.max():.4f}s")
    print(f"  Steady state:     mean={steady_times.mean():.4f}s, min={steady_times.min():.4f}s, "
          f"max={steady_times.max():.4f}s, median={np.median(steady_times):.4f}s")
    print(f"  Per-eval breakdown (steady):")
    for i, t in enumerate(steady_times):
        print(f"    eval {i+4}: {t:.4f}s")

    return steady_times


def main():
    print(f"JAX devices: {jax.devices()}")
    print()

    # --- HMFAST benchmark ---
    print("=" * 60)
    print("HMFAST benchmark")
    print("=" * 60)
    t0 = time.perf_counter()
    cnc_hmfast, cosmo, scal = build_cnc("hmfast")
    init_time = time.perf_counter() - t0
    print(f"  Init time: {init_time:.3f}s")

    hmfast_times = benchmark_evaluations(cnc_hmfast, cosmo, scal, n_eval=20,
                                          label="HMFAST evaluations")

    # --- CLASSY_SZ_JAX benchmark ---
    print()
    print("=" * 60)
    print("CLASSY_SZ_JAX benchmark")
    print("=" * 60)
    t0 = time.perf_counter()
    cnc_classy, cosmo, scal = build_cnc("classy_sz_jax")
    init_time = time.perf_counter() - t0
    print(f"  Init time: {init_time:.3f}s")

    classy_times = benchmark_evaluations(cnc_classy, cosmo, scal, n_eval=20,
                                          label="CLASSY_SZ_JAX evaluations")

    # --- Comparison ---
    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  hmfast steady-state mean:    {hmfast_times.mean():.4f}s")
    print(f"  classy_sz_jax steady-state mean: {classy_times.mean():.4f}s")
    speedup = classy_times.mean() / hmfast_times.mean()
    print(f"  Speedup: {speedup:.2f}x")

    if hmfast_times.mean() < 0.04:
        print(f"  PASS: hmfast mean time {hmfast_times.mean():.4f}s < 0.04s target")
    else:
        print(f"  WARN: hmfast mean time {hmfast_times.mean():.4f}s >= 0.04s target")

    return hmfast_times.mean()


if __name__ == "__main__":
    mean_time = main()
    sys.exit(0 if mean_time < 0.1 else 1)
