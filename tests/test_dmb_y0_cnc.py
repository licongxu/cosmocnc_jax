"""Tests for DMB Pe → central y0 CNC path (separate from GNFW power-law SR)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# TF must not see GPU before classy_szfast init (same as other CNC tests).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = _cuda
tf.config.set_visible_devices([], "GPU")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from cosmocnc_jax import cluster_number_counts  # noqa: E402
from cosmocnc_jax.config import path_to_cosmocnc  # noqa: E402
from cosmocnc_jax.dmb_pressure import (  # noqa: E402
    DMB_DEFAULTS,
    dmb_halo_quantities,
    central_y0_m200,
)
from cosmocnc_jax.params import (  # noqa: E402
    cnc_params_default,
    cosmo_params_default,
    scaling_relation_params_default,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GODMAX_SRC = os.path.join(REPO_ROOT, "ref_packages", "GODMAX", "src")
NOISE_DIR = os.environ.get(
    "TSZSBI_NOISE_DIR", "/scratch/scratch-lxu/tszsbi/noise_files"
)
_HAS_NOISE = os.path.isfile(os.path.join(NOISE_DIR, "sigma_dict_szifi.npy"))


@pytest.mark.skipif(
    not os.path.isdir(GODMAX_SRC), reason="GODMAX reference package not present"
)
def test_dmb_pe_parity_vs_godmax():
    """Pe on 0.05–3 R200c matches GODMAX BCM_18_wP (hmfast gate)."""
    if GODMAX_SRC not in sys.path:
        sys.path.insert(0, GODMAX_SRC)
    from get_BCMP_profile_jit import BCM_18_wP

    Ob0, Om0, h = 0.04897, 0.315, 0.674
    H0 = 100.0 * h
    M_h = 1e14
    z0 = 0.2
    c0 = 5.0

    sim = dict(
        cosmo=dict(H0=H0, Om0=Om0, Ob0=Ob0, sigma8=0.81, ns=0.96, w0=-1.0),
        theta_ej_0=4.0,
        theta_co_0=0.1,
        log10_Mc0=14.83,
        mu_beta=0.21,
        eta_star=0.3,
        eta_cga=0.6,
        A_starcga=0.09,
        log10_M1_starcga=11.4,
        alpha_nt=0.18,
        beta_nt=0.5,
        n_nt=0.3,
        gamma_rhogas=2.0,
        delta_rhogas=7.0,
        nfw_trunc=True,
        epsilon_rt=4.0,
    )
    halo = dict(
        rmin=5e-3,
        rmax=3.0,
        nr=32,
        z_array=[z0],
        lg10_Mmin=np.log10(M_h),
        lg10_Mmax=np.log10(M_h),
        nM=1,
        cmin=c0,
        cmax=c0,
        nc=1,
    )
    bcmp = BCM_18_wP(sim, halo, num_points_trapz_int=48)

    m_phys = M_h / h
    q = dmb_halo_quantities(
        m_phys,
        z0,
        c0,
        Ob0,
        Om0,
        h,
        {**DMB_DEFAULTS, "num_points_trapz_int": 48},
    )

    r_god = np.asarray(bcmp.r_array) / h
    r200 = float(q["r200c_comoving"])
    mask = (r_god > 0.05 * r200) & (r_god < 3.0 * r200)
    mask_core = (r_god > 0.05 * r200) & (r_god < 1.5 * r200)

    pe_h = np.exp(
        np.interp(
            np.log(r_god),
            np.log(np.asarray(q["r_comoving"])),
            np.log(np.asarray(q["Pe"]) + 1e-30),
        )
    )
    pe_g = np.asarray(bcmp.Pe_mat_physical[:, 0, 0, 0]) * 1000.0
    rel = np.abs(pe_h[mask] - pe_g[mask]) / np.maximum(pe_g[mask], 1e-30)
    rel_core = np.abs(pe_h[mask_core] - pe_g[mask_core]) / np.maximum(
        pe_g[mask_core], 1e-30
    )
    assert float(np.median(rel)) < 0.02
    assert float(np.max(rel_core)) < 0.05


def test_central_y0_finite_and_theta_ej_response():
    """Larger theta_ej → lower central y0 at fixed M,z."""
    Ob0, Om0, h = 0.04897, 0.315, 0.674
    m200 = 3e14
    z = 0.3
    c = 5.0
    base = {**DMB_DEFAULTS, "num_points_trapz_int": 48}
    y_lo = float(
        central_y0_m200(m200, z, Ob0, Om0, h, {**base, "theta_ej_0": 3.0}, c200c=c)
    )
    y_hi = float(
        central_y0_m200(m200, z, Ob0, Om0, h, {**base, "theta_ej_0": 6.0}, c200c=c)
    )
    assert np.isfinite(y_lo) and y_lo > 0
    assert np.isfinite(y_hi) and y_hi > 0
    assert y_hi < y_lo


@pytest.mark.skipif(not _HAS_NOISE, reason="Planck σ(θ) noise files not available")
def test_dmb_cnc_abundance_smoke_theta_ej():
    """DMB survey path: abundance finite and decreases when theta_ej increases."""
    survey = os.path.join(path_to_cosmocnc, "surveys", "survey_sr_planck_sim_dmb.py")
    cnc = dict(cnc_params_default)
    cnc["survey_sr"] = survey
    cnc["obs_select"] = "q_planck_sim_dmb"
    cnc["observables"] = [["q_planck_sim_dmb"]]
    cnc["cosmology_tool"] = "classy_sz_jax"
    cnc["hmf_calc"] = "cnc"
    cnc["hmf_type"] = "Tinker08"
    cnc["mass_definition"] = "500c"
    cnc["M_min"] = 1e14
    cnc["M_max"] = 1e16
    cnc["z_min"] = 0.01
    cnc["z_max"] = 0.51
    cnc["n_z"] = 8
    cnc["n_points"] = 64
    cnc["load_catalogue"] = False
    cnc["likelihood_type"] = "binned"
    cnc["binned_lik_type"] = "z_and_obs_select"
    cnc["data_lik_from_abundance"] = False
    cnc["bins_edges_z"] = np.linspace(0.01, 0.51, 5)
    cnc["bins_edges_obs_select"] = np.exp(np.linspace(np.log(6.0), np.log(40.0), 5))
    cnc["obs_select_min"] = 6.0
    cnc["obs_select_max"] = 40.0
    cnc["tszsbi_noise_dir"] = NOISE_DIR
    cnc["cosmocnc_verbose"] = "none"

    def _run(theta_ej):
        nc = cluster_number_counts(cnc_params=cnc)
        nc.cosmo_params = dict(cosmo_params_default)
        sr = dict(scaling_relation_params_default)
        sr["theta_ej_0"] = theta_ej
        sr["dof"] = 0.0
        nc.scal_rel_params = sr
        nc.initialise()
        nc.update_params(nc.cosmo_params, nc.scal_rel_params)
        nc.get_hmf()
        nc.get_cluster_abundance()
        return float(np.sum(np.asarray(nc.n_tot_vec)))

    n_lo = _run(3.0)
    n_hi = _run(6.0)
    assert np.isfinite(n_lo) and n_lo > 0
    assert np.isfinite(n_hi) and n_hi > 0
    assert n_hi < n_lo


def test_gnfw_survey_module_untouched_import():
    """Existing Planck GNFW SR module still imports with power-law layer0."""
    import importlib.util

    path = os.path.join(path_to_cosmocnc, "surveys", "survey_sr_planck_sim.py")
    spec = importlib.util.spec_from_file_location("sr_planck_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "sr_q_planck_sim_layer0")
    assert hasattr(mod, "precompute_q_prefactors")
    # Spot-check power-law form still present in source
    with open(path) as f:
        src = f.read()
    assert "A_szifi" in src and "alpha_szifi" in src
    assert "central_y0" not in src
