"""Run with cmbagent_env and PYTHONPATH=.; optionally set HMFAST_PATH.

Checks the mass marginal, padded JIT reuse, and the hmfast H0 density anchor.
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from cosmocnc_jax.cosmo import cosmology_model
from cosmocnc_jax.params import cnc_params_default, cosmo_params_default
from cosmocnc_jax.sim import (
    get_samples_pdf_2d_jax, sample_lonlat_jax, _padded_size,
    _sample_pdf_2d_logit, _sample_lonlat_jit,
)

x = jnp.linspace(.005, 3., 32)
y = jnp.linspace(0., np.log(100.), 64)
pdf = jnp.ones((len(x), len(y)))
key = jr.PRNGKey(17)
for n in (0, 1, 1000, 1024, 1025):
    z, mass = get_samples_pdf_2d_jax(key, n, x, y, pdf)
    assert z.shape == mass.shape == (n,)
    u = np.asarray(jr.uniform(jr.split(key)[0], (_padded_size(n),)))[:n]
    expected = np.maximum(0., (u - 1. / 65536) / (1. - 1. / 65536)) * float(y[-1])
    np.testing.assert_allclose(mass, expected, rtol=0., atol=2e-12)
    assert np.all(np.isfinite(z)) and np.all((z >= x[0]) & (z <= x[-1]))
    lon, lat = sample_lonlat_jax(key, n)
    assert lon.shape == lat.shape == (n,)
    assert np.all((lon >= 0.) & (lon <= 2 * np.pi))
    assert np.all((lat >= 0.) & (lat <= np.pi))

cache_sizes = (_sample_pdf_2d_logit._cache_size(), _sample_lonlat_jit._cache_size())
get_samples_pdf_2d_jax(key, 999, x, y, pdf)[0].block_until_ready()
sample_lonlat_jax(key, 999)[0].block_until_ready()
assert cache_sizes == (_sample_pdf_2d_logit._cache_size(), _sample_lonlat_jit._cache_size())
for draw in (get_samples_pdf_2d_jax, sample_lonlat_jax):
    try:
        args = (x, y, pdf) if draw is get_samples_pdf_2d_jax else ()
        draw(key, -1, *args)
    except ValueError:
        pass
    else:
        raise AssertionError("negative draw size accepted")

params = dict(cnc_params_default, cosmo_param_density="physical",
              hmfast_path=os.environ.get("HMFAST_PATH"))
cp = dict(cosmo_params_default, h=.6766, Ob0h2=.02242, Oc0h2=.1193,
          sigma_8=.78, n_s=.9665, m_nu=.06, tau_reio=.0544)
cosmo = cosmology_model(cp, cosmology_tool="hmfast", cnc_params=params)
for h in (.6766, .62, .72):
    cosmo.update_cosmology(dict(cosmo.cosmo_params, h=h), cosmology_tool="hmfast")
    bg = cosmo.background_cosmology
    convert = 1000 * bg.const.mpc**3 / bg.const.solar
    expected = cosmo._hmfast_cosmo._cosmo_params()["Rho_crit_0"]
    scalar = bg.critical_density(0.).value * convert
    vector = bg.critical_density(np.array([0., .5])).value * convert
    np.testing.assert_allclose(scalar, expected, rtol=2e-14)
    np.testing.assert_allclose(vector[0], expected, rtol=2e-14)
    H = cosmo._predict_H(bg._cosmo_vec("h"), np.array([.5]))[0]
    expected_z = (3 / (8 * np.pi * 6.67428e-11 * 1.98855e30)) * 3.085677581282e22 * 2.99792458e8**2 * H**2
    np.testing.assert_allclose(vector[1], expected_z, rtol=2e-14)
assert "classy_sz" not in sys.modules
print("PASS: mass marginal, padding, JIT reuse, sky bounds, hmfast density updates;", jax.devices())
