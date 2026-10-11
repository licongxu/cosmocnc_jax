import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
from functools import partial
from .cnc import *
from .sr import *
from .utils import simpson, RegularGridInterpolator
import cosmocnc_jax


# =====================================================================
# Pure JAX functions for catalogue generation (JIT-compatible)
# =====================================================================

@jax.jit
def _build_cdfs(redshift_vec, ln_M, hmf_matrix):
    """Build CDF arrays for inverse-CDF 2D sampling from HMF.

    Args:
        redshift_vec: (n_z,) redshift grid
        ln_M: (n_M,) log-mass grid
        hmf_matrix: (n_z, n_M) HMF matrix (already scaled by 4*pi*sky_frac)

    Returns:
        cpdf_lnM: (n_M,) marginal CDF of lnM, normalized to [0, 1]
        inv_cdf_matrix: (n_z, n_M) inverse CDF matrix: z = f(u, lnM)
        u_grid: (n_z,) uniform grid on [0, 1]
    """
    dx = redshift_vec[1] - redshift_vec[0]
    dy = ln_M[1] - ln_M[0]

    # Conditional CDF of z given lnM: cumsum along z axis (axis 0)
    cpdf_zglnM = jnp.cumsum(hmf_matrix, axis=0) * dx
    # Normalize each column to [0, 1]
    col_max = jnp.max(cpdf_zglnM, axis=0, keepdims=True)
    cpdf_zglnM = cpdf_zglnM / jnp.where(col_max > 0., col_max, 1.)

    # Marginal CDF of lnM
    cpdf_lnM = jnp.cumsum(jnp.sum(hmf_matrix, axis=0)) * dy * dx
    cpdf_lnM = cpdf_lnM / jnp.where(cpdf_lnM[-1] > 0., cpdf_lnM[-1], 1.)

    # Build inverse CDF matrix: for each lnM column, invert z -> u to u -> z
    u_grid = jnp.linspace(0., 1., len(redshift_vec))

    def _invert_column(cpdf_col):
        return jnp.interp(u_grid, cpdf_col, redshift_vec)

    # vmap over columns (transpose so each row is a column of cpdf_zglnM)
    inv_cdf_matrix = jax.vmap(_invert_column)(cpdf_zglnM.T)  # (n_M, n_z)
    inv_cdf_matrix = inv_cdf_matrix.T  # (n_z, n_M)

    return cpdf_lnM, inv_cdf_matrix, u_grid


@partial(jax.jit, static_argnums=(1,))
def _sample_2d_jax(key, max_samples, cpdf_lnM, inv_cdf_matrix, u_grid, ln_M):
    """Sample (z, lnM) pairs from precomputed CDFs via inverse-CDF method.

    Always draws max_samples; caller slices to actual n_clusters.

    Args:
        key: JAX PRNG key
        max_samples: buffer size (static — compile once per unique value)
        cpdf_lnM: (n_M,) marginal CDF of lnM
        inv_cdf_matrix: (n_z, n_M) inverse CDF matrix
        u_grid: (n_z,) uniform grid
        ln_M: (n_M,) log-mass grid

    Returns:
        z_samples: (max_samples,) sampled redshifts
        lnM_samples: (max_samples,) sampled log-masses
    """
    key1, key2 = jrandom.split(key)

    # Sample lnM from marginal CDF
    u_lnM = jrandom.uniform(key1, (max_samples,))
    lnM_samples = jnp.interp(u_lnM, cpdf_lnM, ln_M)

    # Sample z from conditional CDF(z|lnM) via 2D interpolation
    u_z = jrandom.uniform(key2, (max_samples,))
    # Bilinear interpolation of inv_cdf_matrix at (u_z, lnM)
    interp = RegularGridInterpolator((u_grid, ln_M), inv_cdf_matrix,
                                      method='linear', bounds_error=False)
    points = jnp.stack([u_z, lnM_samples], axis=-1)
    z_samples = interp(points)

    return z_samples, lnM_samples


# =====================================================================
# Free functions (pure JAX replacements)
# =====================================================================

def get_samples_pdf_jax(key, n_samples, x, cpdf):
    """1D inverse CDF sampling using JAX."""
    cpdf_norm = cpdf / jnp.max(cpdf)
    u = jrandom.uniform(key, (n_samples,))
    x_samples = jnp.interp(u, cpdf_norm, x)
    return x_samples


def _padded_size(n):
    """Round n up to a power of two (>= 1024) so the jitted samplers compile O(log n) times, not once per n."""
    n = int(n)
    if n < 0:
        raise ValueError("n_samples must be non-negative")
    return max(1024, 1 << (n - 1).bit_length())


def get_samples_pdf_2d_jax(key, n_samples, x, y, pdf):
    """2D inverse CDF sampling using JAX with logit-space interpolation."""
    x_samples, y_samples = _sample_pdf_2d_logit(key, _padded_size(n_samples), x, y, pdf)
    return (x_samples[:n_samples], y_samples[:n_samples])


@partial(jax.jit, static_argnums=(1,))
def _sample_pdf_2d_logit(key, n_samples, x, y, pdf):
    key1, key2 = jrandom.split(key)

    eps = 1e-12

    cpdf_xgy = jnp.cumsum(pdf, axis=0) * (x[1] - x[0])
    col_max = jnp.max(cpdf_xgy, axis=0, keepdims=True)
    cpdf_xgy = cpdf_xgy / jnp.where(col_max > 0., col_max, 1.)

    # Match the benchmark's 65536-point marginal CDF without enlarging the 2D HMF grid.
    y_cdf = jnp.linspace(y[0], y[-1], max(65536, len(y)))
    cpdf_y = jnp.cumsum(jnp.interp(y_cdf, y, jnp.sum(pdf, axis=0)))
    y_samples = get_samples_pdf_jax(key1, n_samples, y_cdf, cpdf_y)

    # Build logit-space grid for interpolation
    wmin = jnp.log(eps) - jnp.log1p(-eps)
    wmax = jnp.log(1.0 - eps) - jnp.log1p(-(1.0 - eps))
    w = jnp.linspace(wmin, wmax, len(x))
    w_norm = (w - wmin) / (wmax - wmin)

    # Map x to logit space
    xmin, xmax = jnp.min(x), jnp.max(x)
    x_unit = jnp.clip((x - xmin) / (xmax - xmin), eps, 1.0 - eps)
    x_logit = jnp.log(x_unit) - jnp.log1p(-x_unit)

    # Invert each conditional CDF column in logit-logit space
    def _invert_column(cpdf_col):
        u_col = jnp.clip(cpdf_col, eps, 1.0 - eps)
        w_col = jnp.log(u_col) - jnp.log1p(-u_col)
        return jnp.interp(w, w_col, x_logit)

    x_matrix = jax.vmap(_invert_column)(cpdf_xgy.T).T

    # Sample and transform to logit space
    u_x = jrandom.uniform(key2, (n_samples,))
    u_x = jnp.clip(u_x, eps, 1.0 - eps)
    w_x = jnp.log(u_x) - jnp.log1p(-u_x)
    w_x_norm = (w_x - wmin) / (wmax - wmin)

    interp = RegularGridInterpolator((w_norm, y), x_matrix,
                                      method='linear', bounds_error=False,
                                      fill_value=x_logit[0])
    points = jnp.stack([w_x_norm, y_samples], axis=-1)
    x_logit_samp = interp(points)

    # Map back from logit space
    lo = jnp.log(eps) - jnp.log1p(-eps)
    hi = jnp.log(1.0 - eps) - jnp.log1p(-(1.0 - eps))
    x_logit_samp = jnp.clip(x_logit_samp, lo, hi)
    x_unit_samp = 1.0 / (1.0 + jnp.exp(-x_logit_samp))
    x_samples = xmin + x_unit_samp * (xmax - xmin)

    return (x_samples, y_samples)


def sample_lonlat_jax(key, n_clusters):
    """Sample uniform longitude and latitude on the sphere using JAX."""
    lon, lat = _sample_lonlat_jit(key, _padded_size(n_clusters))
    return lon[:n_clusters], lat[:n_clusters]


@partial(jax.jit, static_argnums=(1,))
def _sample_lonlat_jit(key, n_clusters):
    key1, key2 = jrandom.split(key)
    lon = 2. * jnp.pi * jrandom.uniform(key1, (n_clusters,))
    lat = jnp.arccos(2. * jrandom.uniform(key2, (n_clusters,)) - 1.)
    return lon, lat


# Backward-compatible aliases (use np.random for legacy callers)
def get_samples_pdf(n_samples, x, cpdf):
    cpdf = cpdf / np.max(cpdf)
    cpdf_samples = np.random.rand(n_samples)
    x_samples = np.interp(cpdf_samples, cpdf, x)
    return x_samples


def get_samples_pdf_2d(n_samples, x, y, pdf):
    key = jrandom.PRNGKey(np.random.randint(0, 2**31))
    result = get_samples_pdf_2d_jax(key, n_samples, x, y, pdf)
    return result


def sample_lonlat(n_clusters):
    lon = 2. * np.pi * np.random.rand(n_clusters)
    lat = np.arccos(2. * np.random.rand(n_clusters) - 1.)
    return lon, lat


# =====================================================================
# Catalogue generator class
# =====================================================================

class catalogue_generator:

    def __init__(self, number_counts=None, n_catalogues=1, seed=None,
                 get_sky_coords=False, sky_frac=None, get_theta=False,
                 std_vec_dict=None, patches_from_coord=False):

        self.n_catalogues = n_catalogues
        self.get_sky_coords = get_sky_coords
        self.sky_frac = sky_frac
        self.number_counts = number_counts
        self.params_cnc = self.number_counts.cnc_params
        self.get_theta = get_theta
        self.std_vec_dict = std_vec_dict
        self.patches_from_coord = patches_from_coord

        # Initialize JAX PRNG key
        if seed is not None:
            self.key = jrandom.PRNGKey(seed)
        else:
            self.key = jrandom.PRNGKey(0)

        self.number_counts.get_hmf()

        self.scaling_relations = self.number_counts.scaling_relations
        self.scatter = self.number_counts.scatter
        self.skyfracs = self.scaling_relations[self.params_cnc["obs_select"]].skyfracs

        if self.sky_frac is None:
            self.sky_frac = np.sum(self.skyfracs)

        print("Sky frac", self.sky_frac)

        self.hmf_matrix = self.number_counts.hmf_matrix * 4. * jnp.pi * self.sky_frac
        self.ln_M = jnp.asarray(self.number_counts.ln_M)
        self.redshift_vec = jnp.asarray(self.number_counts.redshift_vec)

        # Cache constant cosmological arrays
        self._cache_cosmo_arrays()

        # Precompute n_tot
        self._compute_n_tot()

    def _compute_n_tot(self):
        """Compute n_tot from HMF matrix. Also sets dndz and dndln_M."""
        self.dndz = simpson(self.hmf_matrix, x=self.ln_M, axis=1)
        self.dndln_M = simpson(self.hmf_matrix, x=self.redshift_vec, axis=0)
        self.n_tot = simpson(self.dndz, x=self.redshift_vec)

    def update_hmf(self):
        """Update HMF matrix and derived quantities after cosmology change."""
        self.number_counts.get_hmf()
        self.hmf_matrix = self.number_counts.hmf_matrix * 4. * jnp.pi * self.sky_frac
        self._cache_cosmo_arrays()
        self._compute_n_tot()

    def _next_key(self):
        """Split and advance the PRNG key."""
        self.key, subkey = jrandom.split(self.key)
        return subkey

    def _cache_cosmo_arrays(self):
        """Cache cosmological arrays from number_counts."""
        nc = self.number_counts
        self._D_A_vec = jnp.asarray(nc.D_A)
        self._E_z_vec = jnp.asarray(nc.E_z)
        self._D_l_CMB_vec = jnp.asarray(nc.D_l_CMB)
        self._rho_c_vec = jnp.asarray(nc.rho_c)
        self._H0 = nc.cosmology.cosmo_params["h"] * 100.
        self._D_CMB = nc.cosmology.D_CMB
        self._gamma = cosmocnc_jax.constants().gamma

    def get_total_number_clusters(self):
        if not hasattr(self, 'n_tot'):
            self._compute_n_tot()
        print("Total mean number of clusters", self.n_tot)

    def sample_total_number_clusters(self):

        key = self._next_key()
        self.n_tot_obs = np.array(jrandom.poisson(key, lam=jnp.float64(self.n_tot),
                                                    shape=(self.n_catalogues,)))

    def get_sky_patches_multinomial(self):
        """Assign sky patches using categorical sampling."""

        self.sky_patches = {}
        skyfracs_jax = jnp.asarray(self.skyfracs, dtype=jnp.float64)
        log_probs = jnp.log(skyfracs_jax / jnp.sum(skyfracs_jax))

        for i in range(0, self.n_catalogues):

            key = self._next_key()
            n = int(self.n_tot_obs[i])
            self.sky_patches[i] = jrandom.categorical(key, log_probs, shape=(n,))

    def get_sky_patches_from_coord(self, observables, n_clusters):

        self.sky_patches = {}

        key = self._next_key()
        lon, lat = sample_lonlat_jax(key, int(np.round(n_clusters / np.sum(self.skyfracs) * 1.2)))
        lon, lat = np.asarray(lon), np.asarray(lat)

        patches = self.scaling_relations[self.params_cnc["obs_select"]].get_patch(lon, lat).astype(int)
        indices_select = np.where(patches > -0.5)[0][0:n_clusters]

        self.sky_patches[self.params_cnc["obs_select"]] = patches[indices_select]
        lon = lon[indices_select]
        lat = lat[indices_select]

        for observable in observables[1:]:

            self.sky_patches[observable] = self.scaling_relations[observable].get_patch(lon, lat).astype(int)

        return lon, lat

    def generate_catalogues_hmf(self):

        self.get_total_number_clusters()
        self.sample_total_number_clusters()

        self.catalogue_list = []

        for i in range(0, self.n_catalogues):

            n_clusters = int(self.n_tot_obs[i])

            key = self._next_key()
            z_samples, ln_M_samples = get_samples_pdf_2d_jax(
                key, n_clusters, self.redshift_vec, self.ln_M, self.hmf_matrix)

            catalogue = {}
            catalogue["z"] = z_samples
            catalogue["M"] = jnp.exp(ln_M_samples)

            if self.get_sky_coords == True:

                key = self._next_key()
                lon, lat = sample_lonlat_jax(key, n_clusters)
                catalogue["lon"] = lon
                catalogue["lat"] = lat

            if self.get_theta == True:

                catalogue["theta_so"] = self.get_theta_so(catalogue["M"], catalogue["z"])

            self.catalogue_list.append(catalogue)

    def get_theta_so(self, M, z):

        bias = self.number_counts.scal_rel_params["bias_sz"]
        H0 = self.number_counts.cosmo_params["h"] * 100.
        D_A = self.number_counts.D_A
        E_z = self.number_counts.E_z

        D_A = jnp.interp(z, self.number_counts.redshift_vec, D_A)
        E_z = jnp.interp(z, self.number_counts.redshift_vec, E_z)

        prefactor_M_500_to_theta = 6.997 * (H0 / 70.)**(-2. / 3.) * (bias / 3.)**(1. / 3.) * E_z**(-2. / 3.) * (500. / D_A)
        theta_so = prefactor_M_500_to_theta * M**(1. / 3.)  # in arcmin

        return theta_so

    def generate_catalogues(self):

        self.get_total_number_clusters()
        self.sample_total_number_clusters()

        if self.patches_from_coord == False:
            self.get_sky_patches_multinomial()

        self.catalogue_list = []

        for ii in range(0, self.n_catalogues):

            n_clusters = int(self.n_tot_obs[ii])
            catalogue = self._generate_catalogue(ii, n_clusters)
            self.catalogue_list.append(catalogue)

    def _generate_catalogue(self, ii, n_clusters):
        """Generate a single catalogue with generic observables."""

        catalogue = {}
        key_sample = self._next_key()

        if self.get_sky_coords == True and self.patches_from_coord == False:

            key_lonlat = self._next_key()
            lon, lat = sample_lonlat_jax(key_lonlat, n_clusters)

        if self.patches_from_coord == True:

            lon, lat = self.get_sky_patches_from_coord(
                self.params_cnc["observables"][0], n_clusters)

        z_samples, ln_M_samples = get_samples_pdf_2d_jax(
            key_sample, n_clusters, self.redshift_vec, self.ln_M, self.hmf_matrix)

        n_observables = len(self.params_cnc["observables"][0])

        D_A = jnp.interp(z_samples, self.redshift_vec, self._D_A_vec)
        E_z = jnp.interp(z_samples, self.redshift_vec, self._E_z_vec)
        D_l_CMB = jnp.interp(z_samples, self.redshift_vec, self._D_l_CMB_vec)
        rho_c = jnp.interp(z_samples, self.redshift_vec, self._rho_c_vec)

        other_params = {"D_A": D_A,
                        "E_z": E_z,
                        "H0": float(self._H0),
                        "D_l_CMB": D_l_CMB,
                        "rho_c": rho_c,
                        "D_CMB": float(self._D_CMB),
                        "zc": z_samples,
                        "cosmology": self.number_counts.cosmology}

        n_layers = self.scaling_relations[self.params_cnc["obs_select"]].get_n_layers()

        observable_patches = {}
        x0 = {}

        for observable in self.params_cnc["observables"][0]:

            x0[observable] = ln_M_samples

            if self.patches_from_coord == False:

                observable_patches[observable] = jnp.zeros(n_clusters, dtype=jnp.int32)
                observable_patches[self.params_cnc["obs_select"]] = self.sky_patches[ii]

            elif self.patches_from_coord == True:

                observable_patches[observable] = jnp.asarray(self.sky_patches[observable])

        for i in range(0, n_layers):

            x1 = {}

            for j in range(0, n_observables):

                observable = self.params_cnc["observables"][0][j]
                scal_rel = self.scaling_relations[observable]

                vec = self.params_cnc["observable_vectorised"]

                if (isinstance(vec, dict) and vec.get(observable, True)) or (isinstance(vec, bool) and vec):

                    x1[observable] = scal_rel.eval_scaling_relation_no_precompute(x0[observable],
                        layer=i, patch_index=observable_patches[observable],
                        params=self.number_counts.scal_rel_params,
                        other_params=other_params)

                elif (isinstance(vec, dict) and not vec.get(observable, True)) or (isinstance(vec, bool) and not vec):

                    x1[observable] = []

                    for k in range(0, n_clusters):

                        other_params_cluster = {}

                        for key_name in other_params.keys():

                            if isinstance(other_params[key_name], float) or key_name == "cosmology":

                                other_params_cluster[key_name] = other_params[key_name]

                            else:

                                other_params_cluster[key_name] = other_params[key_name][k]

                        a = scal_rel.eval_scaling_relation_no_precompute(jnp.array([x0[observable][k]]),
                            layer=i, patch_index=observable_patches[observable][k],
                            params=self.number_counts.scal_rel_params,
                            other_params=other_params_cluster)[0]
                        x1[observable].append(a)

            # Same covariance for all clusters

            if self.params_cnc["cov_constant"][str(i)] is True:

                covariance = covariance_matrix(self.scatter, self.params_cnc["observables"][0],
                    observable_patches=observable_patches, layer=np.arange(n_layers), other_params=other_params)
                cov = covariance.cov[i]

                key_noise = self._next_key()
                noise = jnp.transpose(jrandom.multivariate_normal(
                    key_noise, jnp.zeros(n_observables), cov, shape=(n_clusters,)))

                for ll in range(0, len(self.params_cnc["observables"][0])):

                    x1[self.params_cnc["observables"][0][ll]] = x1[self.params_cnc["observables"][0][ll]] + noise[ll, :]

            # Different covariance

            elif self.params_cnc["cov_constant"][str(i)] is False:

                # Convert JAX arrays to numpy for in-place element assignment
                for obs in self.params_cnc["observables"][0]:
                    if isinstance(x1[obs], jnp.ndarray):
                        x1[obs] = np.array(x1[obs])

                for k in range(0, n_clusters):

                    observable_patches_cluster = {}
                    other_params_cluster = {}

                    for key_name in observable_patches.keys():

                        observable_patches_cluster[key_name] = observable_patches[key_name][k]

                    for key_name in other_params.keys():

                        if isinstance(other_params[key_name], float) or key_name == "cosmology":

                            other_params_cluster[key_name] = other_params[key_name]

                        else:

                            other_params_cluster[key_name] = other_params[key_name][k]

                    if i < n_layers - 1:

                        covariance = covariance_matrix(self.scatter, self.params_cnc["observables"][0],
                            observable_patches=observable_patches_cluster, layer=np.arange(n_layers), other_params=other_params_cluster)
                        cov = covariance.cov[i]

                        key_noise = self._next_key()
                        noise = np.array(jrandom.multivariate_normal(
                            key_noise, jnp.zeros(n_observables), cov, shape=(1,)).T)

                        kk = 0

                        for observable in self.params_cnc["observables"][0]:

                            x1[observable][k] = x1[observable][k] + noise[kk, 0]
                            kk = kk + 1

                    elif i == n_layers - 1:

                        obs_vec = self.params_cnc["observable_vector"]
                        has_vector_obs = isinstance(obs_vec, dict) and any(obs_vec.values())

                        if has_vector_obs:

                            for observable in self.params_cnc["observables"][0]:

                                if self.params_cnc["observable_vector"][observable] is False:

                                    covariance = covariance_matrix(self.scatter, [observable],
                                        observable_patches=observable_patches_cluster, layer=np.arange(n_layers), other_params=other_params_cluster)
                                    cov = covariance.cov[i]

                                    key_noise = self._next_key()
                                    noise = np.array(jrandom.multivariate_normal(
                                        key_noise, jnp.array([0.]), cov, shape=(1,)).T)
                                    x1[observable][k] = x1[observable][k] + noise[0][0]

                                elif self.params_cnc["observable_vector"][observable] is True:

                                    key_noise = self._next_key()
                                    std = self.std_vec_dict[observable]
                                    noise = np.array(jrandom.normal(key_noise, shape=(len(std),)) * std)
                                    x1[observable][k] = x1[observable][k] + noise

            x0 = x1

        obs_select = self.params_cnc["obs_select"]
        obs_select_arr = jnp.array(x1[obs_select]) if not isinstance(x1[obs_select], jnp.ndarray) else x1[obs_select]
        indices_select = jnp.where(obs_select_arr > self.params_cnc["obs_select_min"])[0]

        z_samples_select = z_samples[indices_select]
        ln_M_samples_select = ln_M_samples[indices_select]

        catalogue["z"] = z_samples_select
        catalogue["M"] = jnp.exp(ln_M_samples_select)

        if self.get_sky_coords == True:

            catalogue["lon"] = lon[indices_select]
            catalogue["lat"] = lat[indices_select]

        vec = self.params_cnc["observable_vectorised"]

        for k in range(0, len(self.params_cnc["observables"][0])):

            observable = self.params_cnc["observables"][0][k]

            if isinstance(vec, bool) and vec:

                catalogue[self.params_cnc["observables"][0][k]] = x1[observable][indices_select]

            else:

                catalogue[self.params_cnc["observables"][0][k]] = []

                for kk in range(0, len(catalogue["M"])):

                    catalogue[self.params_cnc["observables"][0][k]].append(x1[observable][indices_select[kk]])

            obs_patches = observable_patches[observable]
            if isinstance(obs_patches, jnp.ndarray):
                catalogue[observable + "_patch"] = obs_patches[indices_select]
            else:
                catalogue[observable + "_patch"] = jnp.asarray(obs_patches)[indices_select]

        return catalogue
