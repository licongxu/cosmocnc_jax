"""Parity test: cosmology_tool='hmfast' vs cosmology_tool='classy_sz_jax'.

Both backends must produce bit-identical (or within float32-NN noise)
cluster-counting outputs given the same input cosmology. The hmfast
backend uses the hmfast.Cosmology object as the user-facing cosmology
API (background_cosmology / power_spectrum) but routes the JIT'd
predict_* functions through classy_szfast's NN so the cnc.py fast path
is byte-identical to classy_sz_jax.

Run with: ``pytest cosmocnc_jax/tests/test_hmfast_parity.py``
"""
import os
import sys

# TF must not see the GPU (otherwise classy_szfast's TF init crashes); JAX
# can use it. Do this before any tensorflow / jax import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf  # noqa: E402
os.environ["CUDA_VISIBLE_DEVICES"] = _cuda
tf.config.set_visible_devices([], "GPU")

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from cosmocnc_jax import cluster_number_counts  # noqa: E402
from cosmocnc_jax.params import (  # noqa: E402
    cnc_params_default, cosmo_params_default, scaling_relation_params_default,
)


def _make_nc(tool, cosmo=None):
    cnc = dict(cnc_params_default)
    cnc["cosmology_tool"] = tool
    cnc["hmf_calc"] = "cnc"
    cnc["hmf_type"] = "Tinker08"
    cnc["mass_definition"] = "500c"
    cnc["interp_tinker"] = "linear"
    cnc["cosmo_param_density"] = "critical"
    cnc["M_min"] = 1e14
    cnc["M_max"] = 1e16
    cnc["z_min"] = 0.01
    cnc["z_max"] = 1.01
    cnc["n_z"] = 20
    cnc["n_points"] = 64
    cnc["load_catalogue"] = False
    cnc["likelihood_type"] = "binned"
    cnc["binned_lik_type"] = "z_and_obs_select"
    cnc["data_lik_from_abundance"] = False
    cnc["bins_edges_z"] = np.linspace(0.01, 1.01, 7)
    cnc["bins_edges_obs_select"] = np.exp(np.linspace(np.log(6.), np.log(60.), 6))
    cnc["cosmocnc_verbose"] = "minimal"
    nc = cluster_number_counts(cnc_params=cnc)
    nc.cosmo_params = dict(cosmo or cosmo_params_default)
    nc.scal_rel_params = dict(scaling_relation_params_default)
    nc.initialise()
    nc.update_params(nc.cosmo_params, nc.scal_rel_params)
    nc.get_hmf()
    return nc


@pytest.fixture(scope="module")
def baseline():
    return _make_nc("classy_sz_jax")


@pytest.fixture(scope="module")
def hmfast_run():
    return _make_nc("hmfast")


def test_background_quantities_bit_identical(baseline, hmfast_run):
    for attr in ("D_A", "E_z", "rho_c", "D_l_CMB", "redshift_vec"):
        a = np.asarray(getattr(baseline, attr))
        b = np.asarray(getattr(hmfast_run, attr))
        assert a.shape == b.shape
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.array_equal(a, b), (
            f"{attr} not bit-identical: max abs diff "
            f"{np.max(np.abs(a - b)):.3e}, max rel diff "
            f"{np.max(np.abs((a - b) / np.where(a == 0, 1, a))):.3e}")


def test_derived_quantities_bit_identical(baseline, hmfast_run):
    for attr in ("As", "sigma8", "z_CMB", "D_CMB"):
        va = float(getattr(baseline.cosmology, attr))
        vb = float(getattr(hmfast_run.cosmology, attr))
        assert va == vb, f"{attr}: classy={va} hmfast={vb}"


def test_hmf_matrix_bit_identical(baseline, hmfast_run):
    a = np.asarray(baseline.hmf_matrix)
    b = np.asarray(hmfast_run.hmf_matrix)
    assert a.shape == b.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()
    # Initial fast-path computation should be byte-identical.
    assert np.array_equal(a, b), (
        f"hmf_matrix not bit-identical: max abs diff "
        f"{np.max(np.abs(a - b)):.3e}, max rel diff "
        f"{np.max(np.abs((a - b) / np.where(a == 0, 1, a))):.3e}")


def test_update_cosmology_parity():
    nc1 = _make_nc("classy_sz_jax")
    nc2 = _make_nc("hmfast")
    new_cosmo = dict(cosmo_params_default)
    new_cosmo["h"] = 0.7
    new_cosmo["sigma_8"] = 0.85
    new_cosmo["Om0"] = 0.3
    new_cosmo["Ob0"] = 0.05
    nc1.update_params(new_cosmo, dict(scaling_relation_params_default))
    nc2.update_params(new_cosmo, dict(scaling_relation_params_default))
    nc1.get_hmf()
    nc2.get_hmf()

    a = np.asarray(nc1.hmf_matrix)
    b = np.asarray(nc2.hmf_matrix)
    rel = np.max(np.abs((a - b) / np.where(a == 0, 1, a)))
    # update_cosmology re-uses the captured pk_power_fac from init, so a
    # tiny FFTLog roundoff (~1e-14) can leak through. Tolerance is generous
    # for safety but still well below scientific precision.
    assert rel < 1e-10, f"hmf_matrix max rel diff after update: {rel:.3e}"
    assert float(nc1.cosmology.As) == float(nc2.cosmology.As)
    assert float(nc1.cosmology.sigma8) == float(nc2.cosmology.sigma8)


def test_hmfast_path_uses_compute_packages_source():
    """Ensure the hmfast backend imports from /scratch/scratch-lxu/compute_packages
    (or whatever ``hmfast_path`` is configured to), not the in-tree copy."""
    nc = _make_nc("hmfast")
    src = nc.cosmology._hmfast.__file__
    assert "/compute_packages/hmfast/" in src, (
        f"hmfast was loaded from {src}, expected /compute_packages/hmfast/")
