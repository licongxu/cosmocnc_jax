"""
Planck-sim DMB survey scaling relations (`survey_sr_planck_sim_dmb`).

Separate path from ``survey_sr_planck_sim``: central Compton-y0 is evaluated
from the DMB (BCM_18_wP) electron pressure profile

    y0 = 2 (σ_T / m_e c²) ∫ Pe(r) dr

then q = y0 / σ(θ_500) with the same σ(θ) noise tables as the GNFW power-law
Planck-sim module. Existing GNFW SR files are not modified.

Observable: ``q_planck_sim_dmb``.

Key cnc_params keys (same noise footprint as survey_sr_planck_sim):
- ``tszsbi_noise_dir``, ``tszsbi_filter_name``, ``tszsbi_theta_*``
- ``planck_sim_abundance_fsky``
"""
import os
from pathlib import Path
import logging

import numpy as np
import jax
import jax.numpy as jnp

import cosmocnc_jax
from cosmocnc_jax.utils import simpson
from cosmocnc_jax.dmb_pressure import (
    DMB_PARAM_KEYS,
    pack_dmb_params,
    central_y0_m500,
    dmb_params_from_sr,
)


# =====================================================================
# Pure JAX scaling relation functions
# =====================================================================

def precompute_dmb_prefactors(z, E_z, H0, D_A, omega_b, omega_m, bias_sz):
    """z-dependent + cosmology pieces needed by DMB layer 0 / θ_500."""
    h = H0 / 100.
    prefactor_M_500_to_theta = (
        6.997 * (H0 / 70.) ** (-2. / 3.)
        * (bias_sz / 3.) ** (1. / 3.)
        * E_z ** (-2. / 3.) * (500. / D_A)
    )
    return z, omega_b, omega_m, h, prefactor_M_500_to_theta


# =====================================================================
# Class: scaling_relations
# =====================================================================

class scaling_relations:

    def __init__(self, observable="q_planck_sim_dmb", cnc_params=None, catalogue=None):
        self.logger = logging.getLogger(__name__)
        self.observable = observable
        self.cnc_params = cnc_params or {}
        self.preprecompute = False
        self.catalogue = catalogue
        self.root_path = cosmocnc_jax.root_path
        self._nfw_trunc = True
        self._n_trapz = 64

        if observable != "q_planck_sim_dmb":
            raise ValueError(
                f"survey_sr_planck_sim_dmb only supports "
                f"observable='q_planck_sim_dmb'; got '{observable}'."
            )

    def get_n_layers(self):
        return 2

    def get_n_layers_stacked(self):
        return self.get_n_layers()

    def initialise_scaling_relation(self, cosmology=None):
        self.const = cosmocnc_jax.constants()

        nf = Path(
            self.cnc_params.get(
                "tszsbi_noise_dir",
                os.environ.get("TSZSBI_NOISE_DIR", "/scratch/scratch-lxu/tszsbi/noise_files"),
            )
        ).resolve()
        sigma_obj_file = nf / "sigma_dict_szifi.npy"
        skyfr_file = nf / "skyfracs_szifi_cosmology.npy"
        filter_name = self.cnc_params.get(
            "tszsbi_filter_name", os.environ.get("TSZSBI_FILTER_NAME", "immf6"),
        )
        theta_min_arcmin = float(self.cnc_params.get("tszsbi_theta_min_arcmin", 0.5))
        theta_max_arcmin = float(self.cnc_params.get("tszsbi_theta_max_arcmin", 32.0))

        sigma_obj = np.load(sigma_obj_file, allow_pickle=True).item()
        skyfr = np.load(skyfr_file).ravel()
        data = sigma_obj[filter_name]
        first = next(iter(data.values()))
        ntheta = len(first)
        theta_500_vec = np.exp(
            np.linspace(np.log(theta_min_arcmin), np.log(theta_max_arcmin), ntheta)
        )
        num = np.zeros(ntheta, dtype=float)
        den = 0.0
        for tile, arr in data.items():
            w = skyfr[int(tile)]
            num += w * np.asarray(arr, dtype=float)
            den += w
        sigma_sz_vec = num / den

        self.theta_500_vec = theta_500_vec
        x = np.log(theta_500_vec)
        y = np.log(sigma_sz_vec)
        sigma_sz_poly_np = np.polyfit(x, y, deg=3)
        self.sigma_sz_poly = jnp.asarray(sigma_sz_poly_np)
        self.sigma_sz_polyder = jnp.asarray(np.polyder(sigma_sz_poly_np))

        _fsky = self.cnc_params.get("planck_sim_abundance_fsky", "full_sky")
        if _fsky == "full_sky" or _fsky is True:
            self.skyfracs = [1.0]
        elif _fsky == "from_noise_files":
            self.skyfracs = [float(np.sum(skyfr))]
        else:
            self.skyfracs = [float(_fsky)]

        q_vec = np.linspace(5.0, 10.0, self.cnc_params["n_points"])
        pdf_fd = np.exp(-((q_vec - 3.0) ** 2) / 1.5 ** 2)
        pdf_fd = pdf_fd / simpson(pdf_fd, x=q_vec)
        self.pdf_false_detection = [q_vec, pdf_fd]

        # Cache cosmo for abundance (also refreshed via prefactor_args)
        if cosmology is not None:
            self._omega_b = float(cosmology.cosmo_params.get(
                "Ob0", cosmology.cosmo_params.get("Omega_b", 0.04897)))
            self._omega_m = float(cosmology.cosmo_params.get(
                "Om0", cosmology.cosmo_params.get("Omega_m", 0.315)))
            self._h = float(cosmology.cosmo_params["h"])

    def precompute_scaling_relation(self, params=None, other_params=None, patch_index=0):
        self.params = params
        self.other_params = other_params
        E_z = other_params["E_z"]
        H0 = other_params["H0"]
        D_A = other_params["D_A"]
        bias_sz = self.params["bias_sz"]
        self.prefactor_M_500_to_theta = (
            6.997 * (H0 / 70.) ** (-2. / 3.) * (bias_sz / 3.) ** (1. / 3.)
            * E_z ** (-2. / 3.) * (500. / D_A)
        )
        self._z = other_params.get("z", 0.0)

    def eval_scaling_relation(self, x0, layer=0, patch_index=0, other_params=None):
        if layer == 0:
            op = other_params or self.other_params
            z = op.get("z", self._z)
            H0 = op["H0"]
            h = H0 / 100.
            omega_b = op.get("omega_b", getattr(self, "_omega_b", 0.04897))
            omega_m = op.get("omega_m", getattr(self, "_omega_m", 0.315))
            params = dmb_params_from_sr(self.params)
            m500 = self.params["bias_sz"] * 1.0e14 * jnp.exp(x0)
            y0 = jax.vmap(
                lambda m: central_y0_m500(m, z, omega_b, omega_m, h, params)
            )(m500)
            log_y0 = jnp.log(jnp.maximum(y0, 1e-30))
            log_theta_500 = jnp.log(self.prefactor_M_500_to_theta) + x0 / 3.
            self.log_theta_500 = log_theta_500
            log_sigma_sz = jnp.polyval(self.sigma_sz_poly, log_theta_500)
            x1 = log_y0 - log_sigma_sz
        elif layer == 1:
            x1 = jnp.sqrt(jnp.exp(x0) ** 2 + self.params["dof"])
        else:
            raise ValueError(f"Unsupported layer={layer}")
        self.x1 = x1
        return x1

    def eval_derivative_scaling_relation(self, x0, layer=0, patch_index=0,
                                         scalrel_type_deriv="numerical"):
        # DMB has no closed-form d log y0 / d ln M — always numerical.
        return jnp.gradient(self.x1, x0)

    def eval_scaling_relation_no_precompute(self, x0, layer=0, patch_index=0,
                                            other_params=None, params=None):
        self.params = params
        self.other_params = other_params
        if layer == 0:
            E_z = other_params["E_z"]
            H0 = other_params["H0"]
            D_A = other_params["D_A"]
            bias_sz = self.params["bias_sz"]
            pref_theta = (
                6.997 * (H0 / 70.) ** (-2. / 3.) * (bias_sz / 3.) ** (1. / 3.)
                * E_z ** (-2. / 3.) * (500. / D_A)
            )
            self.prefactor_M_500_to_theta = pref_theta
            return self.eval_scaling_relation(x0, layer=0, other_params=other_params)
        elif layer == 1:
            return jnp.sqrt(jnp.exp(x0) ** 2 + self.params["dof"])
        raise ValueError(f"Unsupported layer={layer}")

    def get_mean(self, x0, patch_index=0, scatter=None, compute_var=False, other_params=None):
        raise NotImplementedError("survey_sr_planck_sim_dmb has no stacked mean.")

    def get_cutoff(self, layer=0):
        if layer == 0:
            return -jnp.inf
        elif layer == 1:
            return self.params["q_cutoff"]
        return -jnp.inf

    # =================================================================
    # Factory methods for cnc.py JIT pipeline
    # =================================================================

    def get_layer_fn(self, layer):
        if layer == 0:
            nfw_trunc = bool(self.cnc_params.get("dmb_nfw_trunc", True))
            n_trapz = int(self.cnc_params.get("dmb_num_points_trapz_int", 64))

            def layer0(x0, z, omega_b, omega_m, h, pref_theta,
                       sigma_sz_poly, bias_sz, *dmb_packed):
                params = {k: dmb_packed[i] for i, k in enumerate(DMB_PARAM_KEYS)}
                params["nfw_trunc"] = nfw_trunc
                params["num_points_trapz_int"] = n_trapz
                params["dmb_c200c"] = jnp.nan
                m500 = bias_sz * 1.0e14 * jnp.exp(x0)

                def y0_one(m):
                    return central_y0_m500(m, z, omega_b, omega_m, h, params)

                y0 = jax.vmap(y0_one)(m500)
                log_y0 = jnp.log(jnp.maximum(y0, 1e-30))
                log_theta_500 = jnp.log(pref_theta) + x0 / 3.
                log_sigma_sz = jnp.polyval(sigma_sz_poly, log_theta_500)
                return log_y0 - log_sigma_sz, log_theta_500

            return layer0
        elif layer == 1:
            def layer1(x0, z, omega_b, omega_m, h, pref_theta, dof):
                return jnp.sqrt(jnp.exp(x0) ** 2 + dof)
            return layer1
        raise ValueError(f"No layer fn for layer={layer}")

    def get_layer_returns_aux(self, layer):
        return layer == 0

    def get_layer_deriv_fn(self, layer):
        # Force numerical d(log y0)/d lnM via abundance kernel gradient path.
        return None

    def get_layer_deriv_uses_aux(self, layer):
        return False

    def get_prefactor_fn(self):
        def fn(z, E_z, H0, D_A, omega_b, omega_m, bias_sz):
            return precompute_dmb_prefactors(
                z, E_z, H0, D_A, omega_b, omega_m, bias_sz,
            )
        return fn

    def get_prefactor_vmap_axes(self):
        # z, E_z, H0, D_A, omega_b, omega_m, bias_sz
        return (0, 0, None, 0, None, None, None)

    def get_prefactor_args(self, cosmo_quantities, sr_params):
        H0 = cosmo_quantities["H0"]
        # Prefer explicit Om/Ob from cosmo_q if present; else from module cache.
        omega_b = cosmo_quantities.get(
            "omega_b", jnp.float64(getattr(self, "_omega_b", 0.04897)))
        omega_m = cosmo_quantities.get(
            "omega_m", jnp.float64(getattr(self, "_omega_m", 0.315)))
        return (
            cosmo_quantities["z"],
            cosmo_quantities["E_z"],
            H0,
            cosmo_quantities["D_A"],
            jnp.float64(omega_b),
            jnp.float64(omega_m),
            jnp.float64(sr_params["bias_sz"]),
        )

    def get_n_prefactors(self):
        return 5  # z, omega_b, omega_m, h, pref_theta

    def get_prefactor_fn_unified(self):
        def fn(E_z, D_A, D_l_CMB, rho_c, H0, D_CMB, gamma, z_val,
               omega_b, omega_m, bias_sz):
            return precompute_dmb_prefactors(
                z_val, E_z, H0, D_A, omega_b, omega_m, bias_sz,
            )
        return fn

    def get_prefactor_sr_params(self, sr_params):
        # Used by unified mass-range path; include Om/Ob from cached cosmology.
        return (
            jnp.float64(getattr(self, "_omega_b", 0.04897)),
            jnp.float64(getattr(self, "_omega_m", 0.315)),
            jnp.float64(sr_params["bias_sz"]),
        )

    def get_n_prefactor_sr_params(self):
        return 3

    def get_layer_sr_params(self, layer, sr_params):
        if layer == 0:
            packed = pack_dmb_params(sr_params)
            return (self.sigma_sz_poly, jnp.float64(sr_params["bias_sz"])) + packed
        elif layer == 1:
            return (jnp.float64(sr_params["dof"]),)
        raise ValueError(f"No layer SR params for layer={layer}")

    def get_layer_deriv_sr_params(self, layer, sr_params):
        return ()

    def get_scatter_sigma(self, sr_params):
        return float(sr_params.get("sigma_lnq_szifi", 0.))

    def get_mean_fn(self):
        return None

    def get_mean_fn_sr_params(self, sr_params):
        return ()


class scatter:

    def __init__(self, params=None, catalogue=None):
        self.params = params
        self.catalogue = catalogue

    def get_cov(self, observable1=None, observable2=None,
                patch1=0, patch2=0, layer=0, other_params=None):
        if layer == 0:
            if observable1 == "q_planck_sim_dmb" and observable2 == "q_planck_sim_dmb":
                return self.params["sigma_lnq_szifi"] ** 2
            return 0.
        elif layer == 1:
            if observable1 == "q_planck_sim_dmb" and observable2 == "q_planck_sim_dmb":
                return 1.
            return 0.
        return 0.
