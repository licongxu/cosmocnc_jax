"""Run cobaya MCMC with both classy_sz_jax and hmfast backends on GPU 0.

Runs classy_sz_jax first (reference), then hmfast (new), then plots posteriors.
"""
import os
import sys
import time

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.30"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from cobaya.run import run

TUTORIAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TUTORIAL_DIR)
CHAINS_DIR = os.path.join(TUTORIAL_DIR, "chains")

sys.path.insert(0, TUTORIAL_DIR)
sys.path.insert(0, PROJECT_DIR)


def build_info(output_dir, cosmology_tool):
    """Build cobaya info dict for a given cosmology backend."""
    return {
        "output": output_dir,
        "logging_level": 30,  # WARNING
        "likelihood": {
            "cobaya_planck_scatter_jax.CNCBinnedPlanckScatterLikelihood": {
                "input_params": [
                    "H0", "ln10_10A_s", "n_s", "omega_b", "Omega_m",
                    "omega_cdm", "tau_reio", "m_nu", "one_minus_b", "B",
                    "A_SZ", "alpha_SZ", "sigma_lnY",
                ],
                "data_file": "/scratch/scratch-lxu/tsz_cnc_scatter/synthetic_data/N2d_z_q_bin_scatter.txt",
                "survey_sr": os.path.join(PROJECT_DIR, "cosmocnc_jax", "surveys", "survey_sr_planck_sim.py"),
                "survey_cat": os.path.join(PROJECT_DIR, "cosmocnc_jax", "surveys", "survey_cat_planck_sim.py"),
                "tszsbi_noise_dir": "/scratch/scratch-lxu/tszsbi/noise_files",
                "tszsbi_filter_name": "immf6",
                "lambda_floor": 1.0e-12,
                "z_min": 0.005, "z_max": 1.0, "n_z_bins": 10,
                "q_min": 5.0, "q_max": 40.0, "n_q_bins": 5,
                "n_points": 2048, "n_z": 50,
                "M_min": 1.0e14, "M_max": 1.0e16,
                "f_sky": 1.0, "M_pivot": 2.1e14,
                "cosmology_tool": cosmology_tool,
            }
        },
        "params": {
            "H0": {"prior": {"dist": "norm", "loc": 67.4, "scale": 1.0},
                   "ref": {"dist": "norm", "loc": 67.4, "scale": 1.0}, "proposal": 0.6},
            "ln10_10A_s": {"prior": {"min": 2.5, "max": 3.5},
                           "ref": {"dist": "norm", "loc": 2.9718, "scale": 0.13}, "proposal": 0.13},
            "n_s": {"prior": {"dist": "norm", "loc": 0.962, "scale": 0.014},
                    "ref": {"dist": "norm", "loc": 0.962, "scale": 0.014}, "proposal": 0.014},
            "omega_b": {"prior": {"dist": "norm", "loc": 0.022, "scale": 0.002},
                        "ref": {"dist": "norm", "loc": 0.022, "scale": 0.002}, "proposal": 0.002},
            "Omega_m": {"prior": {"min": 0.2, "max": 0.5},
                        "ref": {"dist": "norm", "loc": 0.3096, "scale": 0.02}, "proposal": 0.02},
            "omega_cdm": {"value": "lambda Omega_m, H0, omega_b: Omega_m * (H0 / 100.0)**2 - omega_b"},
            "tau_reio": {"value": 0.0544},
            "m_nu": {"value": 0.06},
            "one_minus_b": {"value": 0.709},
            "B": {"value": "lambda one_minus_b: 1.0 / one_minus_b"},
            "A_SZ": {"prior": {"min": -4.41, "max": -4.21},
                     "ref": {"dist": "norm", "loc": -4.31, "scale": 0.02}, "proposal": 0.02},
            "alpha_SZ": {"prior": {"min": 1.0, "max": 1.24},
                         "ref": {"dist": "norm", "loc": 1.12, "scale": 0.02}, "proposal": 0.02},
            "sigma_lnY": {"prior": {"dist": "norm", "loc": 0.173, "scale": 0.023},
                          "ref": {"dist": "norm", "loc": 0.173, "scale": 0.023}, "proposal": 0.01},
        },
        "sampler": {"mcmc": {"Rminus1_stop": 0.05, "burn_in": 50, "max_tries": 100000,
                             "learn_proposal": True, "learn_every": 40,
                             "proposal_scale": 1.2, "drag": False}},
        "timing": True,
    }


def run_backend(name, cosmology_tool):
    output_dir = os.path.join(CHAINS_DIR, f"cnc_planck_scatter_{name}")
    os.makedirs(output_dir, exist_ok=True)
    info = build_info(output_dir, cosmology_tool)

    print(f"\n{'='*60}")
    print(f"Running {name} (cosmology_tool={cosmology_tool})")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    result = run(info, resume=False)
    if len(result) == 3:
        _, _, _ = result
    else:
        _, _ = result
    t_total = time.perf_counter() - t0
    print(f"\n{name} total time: {t_total:.0f}s ({t_total/60:.1f} min)")
    return t_total


if __name__ == "__main__":
    # 1) classy_sz_jax (reference)
    t_classy = run_backend("jax", "classy_sz_jax")

    # 2) hmfast (new)
    t_hmfast = run_backend("hmfast", "hmfast")

    print(f"\n{'='*60}")
    print(f"DONE. classy_sz_jax={t_classy:.0f}s, hmfast={t_hmfast:.0f}s")
