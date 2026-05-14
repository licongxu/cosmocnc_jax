import os
import time

import numpy as np
from cobaya.likelihood import Likelihood

# MPI-aware GPU pinning (OpenMPI/torchrun style envs).
# If CUDA_VISIBLE_DEVICES is "0,1" and local rank is 1, this rank is pinned to GPU "1".
_local_rank_env = (
    os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
    or os.environ.get("LOCAL_RANK")
    or os.environ.get("MPI_LOCALRANKID")
)
_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
_gpu_list = [d.strip() for d in _visible.split(",") if d.strip() != ""]
if _local_rank_env is not None and len(_gpu_list) > 1:
    _lr = int(_local_rank_env)
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_list[_lr % len(_gpu_list)]
elif _local_rank_env is not None and len(_gpu_list) == 0:
    # Plain `mpirun -np N` fallback when CUDA_VISIBLE_DEVICES is not set:
    # pin local rank i to GPU i.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(_local_rank_env))

# Enforce runtime behavior for this Cobaya likelihood.
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.30"

# Prevent TF from claiming CUDA context before JAX/classy_sz_jax initialization.
_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf  # noqa: E402

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_visible_devices

from cosmocnc_jax import cluster_number_counts  # noqa: E402
from cosmocnc_jax.params import (  # noqa: E402
    cnc_params_default,
    cosmo_params_default,
    scaling_relation_params_default,
)


class CNCBinnedPlanckScatterLikelihood(Likelihood):
    # Paths / data
    data_file: str = "/scratch/scratch-lxu/tsz_cnc_scatter/synthetic_data/N2d_z_q_bin_scatter.txt"
    survey_sr: str = "/scratch/scratch-lxu/compute_packages/cosmocnc_jax/cosmocnc_jax/surveys/survey_sr_planck_sim.py"
    survey_cat: str = "/scratch/scratch-lxu/compute_packages/cosmocnc_jax/cosmocnc_jax/surveys/survey_cat_planck_sim.py"
    tszsbi_noise_dir: str = "/scratch/scratch-lxu/tszsbi/noise_files"
    tszsbi_filter_name: str = "immf6"
    lambda_floor: float = 1.0e-12

    # Binning (must match the data file)
    z_min: float = 0.005
    z_max: float = 1.0
    n_z_bins: int = 10
    q_min: float = 5.0
    q_max: float = 40.0
    n_q_bins: int = 5

    # Theory config
    n_points: int = 2048
    n_z: int = 50
    M_min: float = 1.0e14
    M_max: float = 1.0e16
    f_sky: float = 1.0
    M_pivot: float = 3.0 * 0.7e14

    def initialize(self):
        self.n_obs = np.loadtxt(self.data_file, dtype=float)
        if self.n_obs.ndim != 2:
            raise ValueError(f"Expected 2D count matrix in {self.data_file}, got shape {self.n_obs.shape}")
        if self.n_obs.shape != (self.n_z_bins, self.n_q_bins):
            raise ValueError(
                f"Observed count shape {self.n_obs.shape} does not match "
                f"(n_z_bins, n_q_bins)=({self.n_z_bins}, {self.n_q_bins})."
            )

        self.bin_edges_z = np.linspace(self.z_min, self.z_max, self.n_z_bins + 1)
        self.bin_edges_q = np.exp(np.linspace(np.log(self.q_min), np.log(self.q_max), self.n_q_bins + 1))

        cnc_params = dict(cnc_params_default)
        cnc_params["survey_sr"] = self.survey_sr
        cnc_params["survey_cat"] = self.survey_cat
        cnc_params["tszsbi_noise_dir"] = self.tszsbi_noise_dir
        cnc_params["tszsbi_filter_name"] = self.tszsbi_filter_name

        cnc_params["load_catalogue"] = False
        cnc_params["likelihood_type"] = "binned"
        cnc_params["binned_lik_type"] = "z_and_obs_select"
        cnc_params["data_lik_from_abundance"] = False

        cnc_params["obs_select"] = "q_planck_sim"
        cnc_params["observables"] = [["q_planck_sim"]]
        cnc_params["obs_select_min"] = self.q_min
        cnc_params["obs_select_max"] = self.q_max
        cnc_params["z_min"] = self.z_min
        cnc_params["z_max"] = self.z_max
        cnc_params["bins_edges_z"] = self.bin_edges_z
        cnc_params["bins_edges_obs_select"] = self.bin_edges_q

        cnc_params["n_points"] = int(self.n_points)
        cnc_params["n_z"] = int(self.n_z)
        cnc_params["M_min"] = float(self.M_min)
        cnc_params["M_max"] = float(self.M_max)
        cnc_params["planck_sim_M_pivot"] = float(self.M_pivot)

        cnc_params["cosmology_tool"] = "hmfast"
        cnc_params["hmf_calc"] = "cnc"
        cnc_params["cosmo_param_density"] = "physical"
        cnc_params["cosmo_amplitude_parameter"] = "A_s"
        cnc_params["cosmocnc_verbose"] = "none"

        self.cnc = cluster_number_counts(cnc_params=cnc_params)
        self.cosmo_base = dict(cosmo_params_default)
        self.scal_base = dict(scaling_relation_params_default)

        self.cnc.cosmo_params = dict(self.cosmo_base)
        self.cnc.scal_rel_params = dict(self.scal_base)
        self.cnc.initialise()
        self._eval_counter = 0

        self.log.info(
            "Loaded observed counts from %s with shape %s; total observed=%d",
            self.data_file,
            self.n_obs.shape,
            int(self.n_obs.sum()),
        )

    def get_requirements(self):
        # All sampled/derived params are retrieved through params_values in logp.
        return {}

    def logp(self, **params_values):
        t0 = time.perf_counter()
        cosmo = dict(self.cosmo_base)
        scal = dict(self.scal_base)

        # Cosmology: accept both native cosmocnc_jax names and tsz_cnc_scatter YAML names.
        for key in ("h", "Ob0h2", "Oc0h2", "n_s", "m_nu", "tau_reio", "A_s"):
            if key in params_values:
                cosmo[key] = float(params_values[key])
        if "H0" in params_values:
            cosmo["h"] = float(params_values["H0"]) / 100.0
        if "omega_b" in params_values:
            cosmo["Ob0h2"] = float(params_values["omega_b"])
        # Cobaya configs may sample Omega_m and define omega_cdm as a derived param.
        # In that case omega_cdm will NOT be present in params_values, so derive it here
        # to ensure Omega_m actually affects the likelihood.
        if "Omega_m" in params_values:
            cosmo["Om0"] = float(params_values["Omega_m"])
        if "omega_cdm" in params_values:
            cosmo["Oc0h2"] = float(params_values["omega_cdm"])
        elif "Omega_m" in params_values and "h" in cosmo and "Ob0h2" in cosmo:
            cosmo["Oc0h2"] = float(params_values["Omega_m"]) * float(cosmo["h"]) ** 2 - float(cosmo["Ob0h2"])
        if "ln10_10A_s" in params_values:
            cosmo["A_s"] = 1.0e-10 * float(np.exp(float(params_values["ln10_10A_s"])))

        # Scaling relation: map tszpower-like names to cosmocnc_jax names.
        for key in ("A_szifi", "alpha_szifi", "sigma_lnq_szifi", "bias_sz"):
            if key in params_values:
                scal[key] = float(params_values[key])
        if "A_SZ" in params_values:
            scal["A_szifi"] = float(params_values["A_SZ"])
        if "alpha_SZ" in params_values:
            scal["alpha_szifi"] = float(params_values["alpha_SZ"])
        if "sigma_lnY" in params_values:
            scal["sigma_lnq_szifi"] = float(params_values["sigma_lnY"])
        if "one_minus_b" in params_values:
            scal["bias_sz"] = float(params_values["one_minus_b"])
        if "B" in params_values:
            scal["bias_sz"] = 1.0 / float(params_values["B"])

        self.cnc.update_params(cosmo, scal)

        # Populate n_binned theory matrix
        _ = self.cnc.get_log_lik_binned()
        lam = np.asarray(self.cnc.n_binned, dtype=float)
        if lam.shape != self.n_obs.shape:
            raise ValueError(f"Theory bin shape {lam.shape} != observed shape {self.n_obs.shape}")

        lam = np.clip(lam, self.lambda_floor, None)
        # Poisson binned log-likelihood (up to additive constant)
        log_like = np.sum(-lam + self.n_obs * np.log(lam))
        elapsed = time.perf_counter() - t0
        self._eval_counter += 1
        log_like_float = float(log_like)
        n_theory = float(np.sum(lam))
        self.log.info(
            "CNC eval %d: log_like=%.8f, N_theory=%.6f, N_obs=%d, elapsed=%.4fs",
            self._eval_counter,
            log_like_float,
            n_theory,
            int(self.n_obs.sum()),
            elapsed,
        )
        return log_like_float
