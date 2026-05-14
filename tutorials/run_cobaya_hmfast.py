"""Run cobaya MCMC with hmfast cosmology backend on GPU.

Uses the same configuration as cobaya_planck_scatter_jax.yaml but with
cosmology_tool="hmfast" and output to a separate chain directory.
"""
import os
import sys
import time

# --- GPU setup ---
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.30"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add the tutorials directory to sys.path so cobaya can find the likelihood module
tutorial_dir = os.path.dirname(os.path.abspath(__file__))
if tutorial_dir not in sys.path:
    sys.path.insert(0, tutorial_dir)

# Also add parent dir for cosmocnc_jax
parent_dir = os.path.dirname(tutorial_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from cobaya.run import run

# Output directory for hmfast chains
output_dir = os.path.join(tutorial_dir, "chains", "cnc_planck_scatter_hmfast")
os.makedirs(output_dir, exist_ok=True)

info = {
    "output": output_dir,
    "logging_level": 20,  # INFO

    "likelihood": {
        "cobaya_planck_scatter_jax.CNCBinnedPlanckScatterLikelihood": {
            "input_params": [
                "H0", "ln10_10A_s", "n_s", "omega_b", "Omega_m",
                "omega_cdm", "tau_reio", "m_nu", "one_minus_b", "B",
                "A_SZ", "alpha_SZ", "sigma_lnY",
            ],
            "data_file": "/scratch/scratch-lxu/tsz_cnc_scatter/synthetic_data/N2d_z_q_bin_scatter.txt",
            "survey_sr": os.path.join(parent_dir, "cosmocnc_jax", "surveys", "survey_sr_planck_sim.py"),
            "survey_cat": os.path.join(parent_dir, "cosmocnc_jax", "surveys", "survey_cat_planck_sim.py"),
            "tszsbi_noise_dir": "/scratch/scratch-lxu/tszsbi/noise_files",
            "tszsbi_filter_name": "immf6",
            "lambda_floor": 1.0e-12,
            "z_min": 0.005,
            "z_max": 1.0,
            "n_z_bins": 10,
            "q_min": 5.0,
            "q_max": 40.0,
            "n_q_bins": 5,
            "n_points": 2048,
            "n_z": 50,
            "M_min": 1.0e14,
            "M_max": 1.0e16,
            "f_sky": 1.0,
            "M_pivot": 2.1e14,
        }
    },

    "params": {
        "H0": {
            "prior": {"dist": "norm", "loc": 67.4, "scale": 1.0},
            "ref": {"dist": "norm", "loc": 67.4, "scale": 1.0},
            "proposal": 0.6,
            "latex": "H_0",
        },
        "ln10_10A_s": {
            "prior": {"min": 2.5, "max": 3.5},
            "ref": {"dist": "norm", "loc": 2.9718, "scale": 0.13},
            "proposal": 0.13,
            "latex": "\\ln(10^{10}A_s)",
        },
        "n_s": {
            "prior": {"dist": "norm", "loc": 0.962, "scale": 0.014},
            "ref": {"dist": "norm", "loc": 0.962, "scale": 0.014},
            "proposal": 0.014,
            "latex": "n_\\mathrm{s}",
        },
        "omega_b": {
            "prior": {"dist": "norm", "loc": 0.022, "scale": 0.002},
            "ref": {"dist": "norm", "loc": 0.022, "scale": 0.002},
            "proposal": 0.002,
            "latex": "\\Omega_\\mathrm{b} h^2",
        },
        "Omega_m": {
            "prior": {"min": 0.2, "max": 0.5},
            "ref": {"dist": "norm", "loc": 0.3096, "scale": 0.02},
            "proposal": 0.02,
            "latex": "\\Omega_m",
        },
        "omega_cdm": {
            "value": "lambda Omega_m, H0, omega_b: Omega_m * (H0 / 100.0)**2 - omega_b",
            "latex": "\\Omega_\\mathrm{c} h^2",
        },
        "tau_reio": {"value": 0.0544},
        "m_nu": {"value": 0.06},
        "one_minus_b": {"value": 0.709},
        "B": {"value": "lambda one_minus_b: 1.0 / one_minus_b"},
        "A_SZ": {
            "prior": {"min": -4.41, "max": -4.21},
            "ref": {"dist": "norm", "loc": -4.31, "scale": 0.02},
            "proposal": 0.02,
            "latex": "A_\\mathrm{SZ}",
        },
        "alpha_SZ": {
            "prior": {"min": 1.0, "max": 1.24},
            "ref": {"dist": "norm", "loc": 1.12, "scale": 0.02},
            "proposal": 0.02,
            "latex": "\\alpha_\\mathrm{SZ}",
        },
        "sigma_lnY": {
            "prior": {"dist": "norm", "loc": 0.173, "scale": 0.023},
            "ref": {"dist": "norm", "loc": 0.173, "scale": 0.023},
            "proposal": 0.01,
            "latex": "\\sigma_{\\ln Y}",
        },
    },

    "sampler": {
        "mcmc": {
            "Rminus1_stop": 0.05,
            "burn_in": 50,
            "max_tries": 100000,
            "learn_proposal": True,
            "learn_every": 40,
            "proposal_scale": 1.2,
            "drag": False,
        }
    },

    "timing": True,
}

print(f"Running cobaya with hmfast backend")
print(f"Output directory: {output_dir}")
print(f"JAX will use CUDA device: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
t_start = time.perf_counter()

result = run(info)
if len(result) == 3:
    upd_info, sampler, _ = result
else:
    upd_info, sampler = result

t_elapsed = time.perf_counter() - t_start
print(f"\nTotal cobaya run time: {t_elapsed:.1f}s")
