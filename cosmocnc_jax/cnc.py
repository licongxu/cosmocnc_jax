import jax
import jax.numpy as jnp
import functools
import numpy as np
# scipy imports removed -- all replaced with JAX equivalents
from .cosmo import *
from .hmf import *
from .sr import *
from .cat import *
from .params import *
from .utils import *
import time
import importlib.util
import sys
from itertools import combinations


# =====================================================================
# Generic factory functions for building JIT-compiled kernels
# =====================================================================

# Number of points for coarse mass range estimation grid
_N_COARSE_MASS = 128


def _tile_to_patches(params_tuple, n_patches):
    """Add leading n_patches dim to each param via broadcast_to (zero-copy).

    Scalar () -> (n_patches,). Array (d,) -> (n_patches, d). Etc.
    After p[patch_idx] inside JIT, the original shape is recovered.

    If a param already has n_patches as its first dimension, it is kept as-is
    (supports per-patch data like sigma matrices).  The shape check is
    Python-level (trace-time only) so the JIT graph is unchanged.
    """
    def _tile_one(p):
        p = jnp.asarray(p)
        if p.ndim >= 1 and p.shape[0] == n_patches:
            return p  # already per-patch
        return jnp.broadcast_to(p[None], (n_patches,) + p.shape)
    return tuple(_tile_one(p) for p in params_tuple)


def _bilinear_interp_3d(xi0, xi1, patch_idx, tensor, x0_start, dx0, n0, x1_start, dx1, n1):
    """Vectorized bilinear interpolation on a 3D tensor indexed by patch.

    Gathers only the 4 corner values per query point (no full-slice expansion).

    Args:
        xi0: (N,) query points along axis 0 (e.g. redshift)
        xi1: (N,) query points along axis 1 (e.g. observable)
        patch_idx: (N,) int32 patch indices into tensor axis 0
        tensor: (n_patches, n0, n1) full abundance tensor
        x0_start, dx0, n0: regular grid spec for axis 0
        x1_start, dx1, n1: regular grid spec for axis 1

    Returns:
        (N,) interpolated values
    """
    fi0 = (xi0 - x0_start) / dx0
    fi1 = (xi1 - x1_start) / dx1
    fi0 = jnp.clip(fi0, 0., n0 - 1 - 1e-7)
    fi1 = jnp.clip(fi1, 0., n1 - 1 - 1e-7)
    i0 = jnp.floor(fi0).astype(jnp.int32)
    i1 = jnp.floor(fi1).astype(jnp.int32)
    i0 = jnp.clip(i0, 0, n0 - 2)
    i1 = jnp.clip(i1, 0, n1 - 2)
    t0 = fi0 - i0
    t1 = fi1 - i1
    # Gather only 4 corners per cluster from the 3D tensor
    v00 = tensor[patch_idx, i0, i1]
    v01 = tensor[patch_idx, i0, i1 + 1]
    v10 = tensor[patch_idx, i0 + 1, i1]
    v11 = tensor[patch_idx, i0 + 1, i1 + 1]
    return v00 * (1 - t0) * (1 - t1) + v01 * (1 - t0) * t1 + v10 * t0 * (1 - t1) + v11 * t0 * t1


def _bilinear_interp_2d(xi0, xi1, grid, x0_start, dx0, n0, x1_start, dx1, n1):
    """Vectorized bilinear interpolation on a single 2D regular grid.

    Args:
        xi0: (M,) query points along axis 0
        xi1: (M,) query points along axis 1
        grid: (n0, n1) 2D values grid
        x0_start, dx0, n0: regular grid spec for axis 0
        x1_start, dx1, n1: regular grid spec for axis 1

    Returns:
        (M,) interpolated values
    """
    fi0 = (xi0 - x0_start) / dx0
    fi1 = (xi1 - x1_start) / dx1
    fi0 = jnp.clip(fi0, 0., n0 - 1 - 1e-7)
    fi1 = jnp.clip(fi1, 0., n1 - 1 - 1e-7)
    i0 = jnp.floor(fi0).astype(jnp.int32)
    i1 = jnp.floor(fi1).astype(jnp.int32)
    i0 = jnp.clip(i0, 0, n0 - 2)
    i1 = jnp.clip(i1, 0, n1 - 2)
    t0 = fi0 - i0
    t1 = fi1 - i1
    v00 = grid[i0, i1]
    v01 = grid[i0, i1 + 1]
    v10 = grid[i0 + 1, i1]
    v11 = grid[i0 + 1, i1 + 1]
    return v00 * (1 - t0) * (1 - t1) + v01 * (1 - t0) * t1 + v10 * t0 * (1 - t1) + v11 * t0 * t1



def build_backward_conv_nd(layer0_fns, layer1_fns, layer0_returns_aux_list, n_obs,
                           nd_circular=True):
    """Factory: build an N-dimensional backward conv for a correlation set.

    Handles N correlated observables with full covariance structure across layers.
    For n_obs=1, this reduces to the same algorithm as build_backward_conv_1d.

    Args:
        layer0_fns: list of N pure JAX layer-0 functions
        layer1_fns: list of N pure JAX layer-1 functions
        layer0_returns_aux_list: list of N booleans
        n_obs: number of observables in the correlation set

    Returns:
        backward_conv_nd(lnM, obs_vals, all_layer0_args, all_layer1_args,
                          cov_layer0, cov_layer1, n_points,
                          apply_cutoff, cutoff_val)
        → (n_points,) cpdf on the mass grid

        obs_vals: (n_obs,) observed values
        all_layer0_args: tuple of n_obs tuples (prefactors + sr_params per obs)
        all_layer1_args: tuple of n_obs tuples (sr_params per obs)
        cov_layer0: (n_obs, n_obs) covariance matrix for layer 0
        cov_layer1: (n_obs, n_obs) covariance matrix for layer 1
    """

    def backward_conv_nd(lnM, obs_vals, all_layer0_args, all_layer1_args,
                          cov_layer0, cov_layer1, n_points,
                          apply_cutoff, cutoff_val):

        # ── Forward pass: layer 0 on lnM grid ──
        x_l0_list = []
        x_l0_linear_list = []
        for j in range(n_obs):
            if layer0_returns_aux_list[j]:
                x_l0_j, _ = layer0_fns[j](lnM, *all_layer0_args[j])
            else:
                x_l0_j = layer0_fns[j](lnM, *all_layer0_args[j])
            x_l0_list.append(x_l0_j)
            x_l0_linear_list.append(jnp.linspace(x_l0_j[0], x_l0_j[-1], n_points))

        # ── Backward pass: start from layer 1 (measurement layer) ──

        # Evaluate layer 1 on linear grids and compute residuals
        x_l1_residuals = []
        x_l1_raw = []
        for j in range(n_obs):
            if len(all_layer1_args[j]) > 0:
                x_l1_j = layer1_fns[j](x_l0_linear_list[j], *all_layer1_args[j])
            else:
                x_l1_j = layer1_fns[j](x_l0_linear_list[j])
            x_l1_raw.append(x_l1_j)
            x_l1_residuals.append(x_l1_j - obs_vals[j])

        if n_obs == 1:
            # ── 1D path (same as build_backward_conv_1d) ──
            cpdf = eval_gaussian_nd(
                jnp.array(x_l1_residuals),  # (1, n_points)
                cov=cov_layer1)

            # Observable cutoff (on first/only observable)
            cpdf = jnp.where(apply_cutoff & (x_l1_raw[0] < cutoff_val), 0., cpdf)

            # Layer-0 convolution
            x_lin = x_l0_linear_list[0]
            dx = x_lin[1] - x_lin[0]
            x_kernel = x_lin - jnp.mean(x_lin) + 0.5 * dx

            cov_l0_scalar = cov_layer0[0, 0]
            kernel = gaussian_1d(x_kernel, jnp.maximum(jnp.sqrt(cov_l0_scalar), 1e-30))
            cpdf_conv = convolve_nd(cpdf, kernel, circular=nd_circular)
            cpdf = jnp.where(cov_l0_scalar > 1e-20, cpdf_conv, cpdf)

            cpdf = jnp.maximum(cpdf, 0.)

            # Interpolate back to original (SR-distorted) grid
            cpdf = interp_uniform(x_l0_list[0], x_l0_list[0][0], x_l0_list[0][-1],
                                   n_points, cpdf)
        elif n_obs == 2:
            # ── Optimized 2D path ──
            # Use meshgrid + eval_gaussian_nd (XLA-friendly kernel fusion)
            x1_stack = jnp.stack(x_l1_residuals)  # (2, n_points)
            x_mesh = get_mesh(x1_stack)  # (2, n_pts, n_pts)
            cpdf = eval_gaussian_nd(x_mesh, cov=cov_layer1)

            # Observable cutoff (selection obs = index 0)
            mask = x_mesh[0] + obs_vals[0] < cutoff_val
            cpdf = jnp.where(apply_cutoff & mask, 0., cpdf)

            # ── Layer-0 convolution kernel ──
            x_p_m_list = []
            for j in range(2):
                x_lin = x_l0_linear_list[j]
                dx = x_lin[1] - x_lin[0]
                x_p_m_list.append(x_lin - jnp.mean(x_lin) + 0.5 * dx)

            x_p_m_stack = jnp.stack(x_p_m_list)
            x_p_mesh = get_mesh(x_p_m_stack)
            has_scatter = ~jnp.all(cov_layer0 == 0.)
            kernel = eval_gaussian_nd(x_p_mesh, cov=cov_layer0)
            cpdf_conv = convolve_nd(cpdf, kernel, circular=nd_circular)
            cpdf = jnp.where(has_scatter, cpdf_conv, cpdf)

            cpdf = jnp.maximum(cpdf, 0.)

            # Bilinear interpolation at diagonal points only
            xi0 = x_l0_list[0]
            xi1 = x_l0_list[1]
            x0s = x_l0_linear_list[0][0]
            x1s = x_l0_linear_list[1][0]
            dx0 = x_l0_linear_list[0][1] - x0s
            dx1 = x_l0_linear_list[1][1] - x1s
            n_p = n_points

            fi0 = jnp.clip((xi0 - x0s) / dx0, 0., n_p - 1 - 1e-7)
            fi1 = jnp.clip((xi1 - x1s) / dx1, 0., n_p - 1 - 1e-7)
            i0 = jnp.clip(jnp.floor(fi0).astype(jnp.int32), 0, n_p - 2)
            i1 = jnp.clip(jnp.floor(fi1).astype(jnp.int32), 0, n_p - 2)
            t0 = fi0 - i0
            t1 = fi1 - i1
            cpdf = (cpdf[i0, i1] * (1 - t0) * (1 - t1) +
                    cpdf[i0, i1 + 1] * (1 - t0) * t1 +
                    cpdf[i0 + 1, i1] * t0 * (1 - t1) +
                    cpdf[i0 + 1, i1 + 1] * t0 * t1)
        else:
            # ── General N-D path (3+ observables) ──
            x1_stack = jnp.stack(x_l1_residuals)
            x_mesh = get_mesh(x1_stack)
            cpdf = eval_gaussian_nd(x_mesh, cov=cov_layer1)

            mask = x_mesh[0] + obs_vals[0] < cutoff_val
            cpdf = jnp.where(apply_cutoff & mask, 0., cpdf)

            x_p_m_list = []
            for j in range(n_obs):
                x_lin = x_l0_linear_list[j]
                dx = x_lin[1] - x_lin[0]
                x_p_m_list.append(x_lin - jnp.mean(x_lin) + 0.5 * dx)

            x_p_m_stack = jnp.stack(x_p_m_list)
            x_p_mesh = get_mesh(x_p_m_stack)
            has_scatter = ~jnp.all(cov_layer0 == 0.)
            kernel = eval_gaussian_nd(x_p_mesh, cov=cov_layer0)
            cpdf_conv = convolve_nd(cpdf, kernel, circular=nd_circular)
            cpdf = jnp.where(has_scatter, cpdf_conv, cpdf)
            cpdf = jnp.maximum(cpdf, 0.)

            diag_points = jnp.stack(x_l0_list, axis=-1)
            linear_grids = tuple(x_l0_linear_list[j] for j in range(n_obs))
            interp = RegularGridInterpolator(linear_grids, cpdf,
                                              fill_value=0., bounds_error=False)
            cpdf = interp(diag_points)

        return cpdf

    return backward_conv_nd


def build_2d_forward_fn(layer0_fns, layer1_fns, layer0_returns_aux_list,
                        pref_fns, n_psr_list, n_pts,
                        n_pc_l0_list=(0, 0), n_pc_l1_list=(0, 0)):
    """Factory: forward pass for 2D backward conv (separate JIT stage A).

    Computes per-cluster: lnM grid → prefactors → layer-0 → layer-1 → residuals.
    Returns arrays consumed by the 2D conv core (stage B).
    """

    def per_cluster(mn, mx, obs_vals,
                    E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
                    H0, D_CMB, gamma, z_c,
                    all_pref_sr, all_layer0_sr, all_layer1_sr,
                    all_layer0_sr_pc, all_layer1_sr_pc,
                    patch_idx):
        lnM = jnp.linspace(mn, mx, n_pts)

        x_l0_list = []
        x_l0_lin_list = []
        x_l1_residuals = []

        for j in range(2):
            psr_j = tuple(p[patch_idx] for p in all_pref_sr[j])
            l0_j = tuple(p[patch_idx] for p in all_layer0_sr[j])
            l1_j = tuple(p[patch_idx] for p in all_layer1_sr[j])
            l0_pc_j = all_layer0_sr_pc[j]
            l1_pc_j = all_layer1_sr_pc[j]
            prefactors = pref_fns[j](E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
                                     H0, D_CMB, gamma, z_c, *psr_j)
            layer0_args = prefactors + l0_j + l0_pc_j

            if layer0_returns_aux_list[j]:
                x_l0_j, _ = layer0_fns[j](lnM, *layer0_args)
            else:
                x_l0_j = layer0_fns[j](lnM, *layer0_args)

            x_l0_list.append(x_l0_j)
            x_l0_lin = jnp.linspace(x_l0_j[0], x_l0_j[-1], n_pts)
            x_l0_lin_list.append(x_l0_lin)

            layer1_args = prefactors + l1_j + l1_pc_j
            if len(layer1_args) > 0:
                x_l1_j = layer1_fns[j](x_l0_lin, *layer1_args)
            else:
                x_l1_j = layer1_fns[j](x_l0_lin)

            x_l1_residuals.append(x_l1_j - obs_vals[j])

        # Pack outputs: residuals, kernel coords, interp coords, grid info
        r0 = x_l1_residuals[0]
        r1 = x_l1_residuals[1]
        kc0 = x_l0_lin_list[0] - jnp.mean(x_l0_lin_list[0]) + 0.5 * (x_l0_lin_list[0][1] - x_l0_lin_list[0][0])
        kc1 = x_l0_lin_list[1] - jnp.mean(x_l0_lin_list[1]) + 0.5 * (x_l0_lin_list[1][1] - x_l0_lin_list[1][0])
        x_l0_0 = x_l0_list[0]
        x_l0_1 = x_l0_list[1]
        x_lin_0_start = x_l0_lin_list[0][0]
        x_lin_1_start = x_l0_lin_list[1][0]
        dx0 = x_l0_lin_list[0][1] - x_l0_lin_list[0][0]
        dx1 = x_l0_lin_list[1][1] - x_l0_lin_list[1][0]

        return r0, r1, kc0, kc1, x_l0_0, x_l0_1, x_lin_0_start, x_lin_1_start, dx0, dx1

    vmap_in = (
        0, 0,               # mn, mx
        0,                  # obs_vals (batch, 2)
        0, 0, 0, 0,        # cosmo per-cluster
        None, None, None, 0,  # cosmo scalars + z_c (per-cluster)
        tuple([tuple([None]*n) for n in n_psr_list]),  # all_pref_sr (n_patches, ...)
        tuple([None]*2),    # all_layer0_sr (n_patches, ...)
        tuple([None]*2),    # all_layer1_sr (n_patches, ...)
        tuple([tuple([0]*n) if n > 0 else () for n in n_pc_l0_list]),  # per-cluster L0
        tuple([tuple([0]*n) if n > 0 else () for n in n_pc_l1_list]),  # per-cluster L1
        0,                  # patch_idx (per-cluster)
    )

    return jax.jit(jax.vmap(per_cluster, in_axes=vmap_in))


def build_2d_conv_fn(n_pts, nd_circular=True):
    """Factory: pure-math 2D conv core (separate JIT stage B).

    Takes pre-computed arrays from the forward pass. No SR functions inside —
    prevents XLA from fusing with the forward pass, avoiding shared memory
    overflow on Blackwell GPUs (128×128×8 = 128KB > 101KB limit).
    """

    def per_cluster(r0, r1, kc0, kc1, x_l0_0, x_l0_1,
                    x_lin_0_start, x_lin_1_start, dx0, dx1,
                    obs_val_0, inv_cov1, norm1, inv_cov0, norm0,
                    has_scatter, apply_cutoff, cutoff_val):
        # Layer-1 Gaussian (outer product — no meshgrid)
        maha1 = (inv_cov1[0, 0] * r0[:, None]**2 +
                 inv_cov1[1, 1] * r1[None, :]**2 +
                 2. * inv_cov1[0, 1] * r0[:, None] * r1[None, :])
        cpdf = norm1 * jnp.exp(-0.5 * maha1)

        # Observable cutoff
        raw1 = r0 + obs_val_0
        cpdf = jnp.where(apply_cutoff & (raw1[:, None] < cutoff_val), 0., cpdf)

        # Layer-0 kernel (outer product)
        maha0 = (inv_cov0[0, 0] * kc0[:, None]**2 +
                 inv_cov0[1, 1] * kc1[None, :]**2 +
                 2. * inv_cov0[0, 1] * kc0[:, None] * kc1[None, :])
        kernel = norm0 * jnp.exp(-0.5 * maha0)

        cpdf_conv = convolve_nd(cpdf, kernel, circular=nd_circular)
        cpdf = jnp.where(has_scatter, cpdf_conv, cpdf)
        cpdf = jnp.maximum(cpdf, 0.)

        # Bilinear interpolation at diagonal points
        n_p = n_pts
        fi0 = jnp.clip((x_l0_0 - x_lin_0_start) / dx0, 0., n_p - 1 - 1e-7)
        fi1 = jnp.clip((x_l0_1 - x_lin_1_start) / dx1, 0., n_p - 1 - 1e-7)
        i0 = jnp.clip(jnp.floor(fi0).astype(jnp.int32), 0, n_p - 2)
        i1 = jnp.clip(jnp.floor(fi1).astype(jnp.int32), 0, n_p - 2)
        t0 = fi0 - i0; t1 = fi1 - i1
        cpdf = (cpdf[i0, i1] * (1 - t0) * (1 - t1) +
                cpdf[i0, i1 + 1] * (1 - t0) * t1 +
                cpdf[i0 + 1, i1] * t0 * (1 - t1) +
                cpdf[i0 + 1, i1 + 1] * t0 * t1)

        return cpdf

    vmap_in = (
        0, 0, 0, 0, 0, 0,  # r0, r1, kc0, kc1, x_l0_0, x_l0_1
        0, 0, 0, 0,        # x_lin_0_start, x_lin_1_start, dx0, dx1
        0,                  # obs_val_0 (per-cluster for cutoff)
        None, None,         # inv_cov1, norm1
        None, None,         # inv_cov0, norm0
        None, None, None,   # has_scatter, apply_cutoff, cutoff_val
    )

    return jax.jit(jax.vmap(per_cluster, in_axes=vmap_in))


def build_sub_bc_jit(layer0_fns, layer1_fns, layer0_returns_aux_list,
                     pref_fns, n_psr_list, n_sub, n_pts, nd_circular=True):
    """Factory: build vmapped JIT for backward conv of a sub-pattern.

    For a subset of observables within a correlation set, wraps
    build_backward_conv_nd with prefactor computation and vmaps over clusters.

    Use for n_sub=1 or n_sub>=3.  For n_sub=2, use the split-JIT
    (build_2d_forward_fn + build_2d_conv_fn) to avoid XLA shared-memory
    overflow on Blackwell GPUs.

    Args:
        layer0_fns: list of n_sub pure JAX layer-0 functions
        layer1_fns: list of n_sub pure JAX layer-1 functions
        layer0_returns_aux_list: list of n_sub booleans
        pref_fns: list of n_sub prefactor_fn_unified functions
        n_psr_list: list of n_sub ints (n_prefactor_sr_params per obs)
        n_sub: number of observables in the sub-pattern
        n_pts: n_points_data_lik (mass grid resolution)
        nd_circular: whether to use circular convolution

    Returns:
        jit(vmap(per_cluster, ...))
        per_cluster(mn, mx, obs_vals,
                    E_z, D_A, D_l_CMB, rho_c,
                    H0, D_CMB, gamma,
                    all_pref_sr, all_layer0_sr, all_layer1_sr,
                    cov_l0, cov_l1, apply_cut, cut_val) → cpdf (n_pts,)
    """
    bc_fn = build_backward_conv_nd(
        layer0_fns, layer1_fns, layer0_returns_aux_list, n_sub, nd_circular)

    def per_cluster(mn, mx, obs_vals, E_z, D_A, D_l_CMB, rho_c,
                    H0, D_CMB, gamma, z_c,
                    all_pref_sr, all_layer0_sr, all_layer1_sr,
                    cov_l0, cov_l1, apply_cut, cut_val,
                    patch_idx):
        lnM = jnp.linspace(mn, mx, n_pts)
        layer0_args = []
        layer1_args = []
        for k in range(n_sub):
            psr_k = tuple(p[patch_idx] for p in all_pref_sr[k])
            l0_k = tuple(p[patch_idx] for p in all_layer0_sr[k])
            l1_k = tuple(p[patch_idx] for p in all_layer1_sr[k])
            prefs = pref_fns[k](E_z, D_A, D_l_CMB, rho_c,
                                H0, D_CMB, gamma, z_c, *psr_k)
            layer0_args.append(prefs + l0_k)
            layer1_args.append(prefs + l1_k)
        return bc_fn(lnM, obs_vals, tuple(layer0_args), tuple(layer1_args),
                     cov_l0, cov_l1, n_pts, apply_cut, cut_val)

    vmap_in = (
        0, 0, 0,                                         # mn, mx, obs_vals
        0, 0, 0, 0,                                      # cosmo per-cluster
        None, None, None, 0,                              # H0, D_CMB, gamma, z_c
        tuple([tuple([None]*n) for n in n_psr_list]),     # all_pref_sr (n_patches, ...)
        tuple([None]*n_sub),                              # all_layer0_sr (n_patches, ...)
        tuple([None]*n_sub),                              # all_layer1_sr (n_patches, ...)
        None, None,                                       # cov_l0, cov_l1
        None, None,                                       # apply_cut, cut_val
        0,                                                # patch_idx (per-cluster)
    )
    return jax.jit(jax.vmap(per_cluster, in_axes=vmap_in))


def build_backward_conv_1d(layer0_fn, layer1_fn, layer0_returns_aux=False):
    """Factory: build a generic 1D 2-layer backward conv function.

    Args:
        layer0_fn: pure JAX layer-0 function. If layer0_returns_aux,
                   returns (x1, aux); otherwise returns x1.
        layer1_fn: pure JAX layer-1 function. Returns x1.
        layer0_returns_aux: whether layer0 returns (x1, aux).

    Returns:
        backward_conv_1d(lnM, obs_val, layer0_args, layer1_args,
                          sigma_scatter_0, n_points, apply_cutoff, cutoff_val)
        where layer0_args = prefactors + layer_sr_params (tuple)
              layer1_args = layer_sr_params (tuple, may be empty)
    """
    def backward_conv_1d(lnM, obs_val, layer0_args, layer1_args,
                          sigma_scatter_0, n_points, apply_cutoff, cutoff_val):
        # Forward through layer 0
        if layer0_returns_aux:
            x_l0, _aux = layer0_fn(lnM, *layer0_args)
        else:
            x_l0 = layer0_fn(lnM, *layer0_args)

        x_l0_min = x_l0[0]
        x_l0_max = x_l0[-1]
        x_l0_linear = jnp.linspace(x_l0_min, x_l0_max, n_points)

        # Evaluate layer 1 on linear grid
        if len(layer1_args) > 0:
            x_l1 = layer1_fn(x_l0_linear, *layer1_args)
        else:
            x_l1 = layer1_fn(x_l0_linear)
        residual = x_l1 - obs_val

        # Gaussian PDF (layer-1 scatter = 1)
        cpdf = gaussian_1d(residual, 1.0)

        # Observable cutoff
        cpdf = jnp.where(apply_cutoff & (x_l1 < cutoff_val), 0., cpdf)

        # Layer-0 convolution kernel
        dx = x_l0_linear[1] - x_l0_linear[0]
        x_kernel = x_l0_linear - jnp.mean(x_l0_linear) + 0.5 * dx
        kernel = gaussian_1d(x_kernel, jnp.maximum(sigma_scatter_0, 1e-30))
        cpdf_conv = convolve_nd(cpdf, kernel)
        cpdf = jnp.where(sigma_scatter_0 > 1e-10, cpdf_conv, cpdf)

        cpdf = jnp.maximum(cpdf, 0.)

        # Interpolate back to original (SR-distorted) grid
        cpdf = interp_uniform(x_l0, x_l0_min, x_l0_max, n_points, cpdf)
        return cpdf

    return backward_conv_1d


def build_mass_range_fn(layer0_fn, layer0_deriv_fn, layer1_fn, layer1_deriv_fn,
                         layer0_returns_aux=False, layer0_deriv_uses_aux=False):
    """Factory: build a mass range estimation function for the selection observable.

    Returns:
        mass_range_fn(lnM_coarse, obs_val, layer0_args, layer1_args,
                       layer0_deriv_args, layer1_deriv_args,
                       sigma_scatter_0, sigma_mass_prior, lnM0_min, lnM0_max)
        -> (lnM_min, lnM_max)
    """
    def mass_range_fn(lnM_coarse, obs_val,
                       layer0_args, layer1_args,
                       layer0_deriv_args, layer1_deriv_args,
                       sigma_scatter_0, sigma_mass_prior, lnM0_min, lnM0_max):
        # Forward through layers on coarse grid
        if layer0_returns_aux:
            x1, aux = layer0_fn(lnM_coarse, *layer0_args)
        else:
            x1 = layer0_fn(lnM_coarse, *layer0_args)
            aux = None

        if len(layer1_args) > 0:
            x2 = layer1_fn(x1, *layer1_args)
        else:
            x2 = layer1_fn(x1)

        i_min = jnp.argmin(jnp.abs(obs_val - x2))
        lnM_c = lnM_coarse[i_min]

        # Derivatives at centre
        if layer0_deriv_fn is not None:
            if layer0_deriv_uses_aux and aux is not None:
                d0 = layer0_deriv_fn(aux[i_min], *layer0_deriv_args)
            else:
                d0 = layer0_deriv_fn(lnM_coarse[i_min], *layer0_deriv_args)
        else:
            d0 = jnp.float64(1.0)

        if layer1_deriv_fn is not None:
            d1 = layer1_deriv_fn(x1[i_min], *layer1_deriv_args)
        else:
            d1 = jnp.float64(1.0)

        sd0 = jnp.maximum(jnp.abs(d0), 1e-30)
        sd1 = jnp.maximum(jnp.abs(d1), 1e-30)
        DlnM = jnp.sqrt((1.0 / sd1)**2 + (sigma_scatter_0 / sd0)**2)

        lnM_min = jnp.maximum(lnM_c - sigma_mass_prior * DlnM, lnM0_min)
        lnM_max = jnp.minimum(lnM_c + sigma_mass_prior * DlnM, lnM0_max)
        return lnM_min, lnM_max

    return mass_range_fn


def build_stacked_kernel(layer0_fn, mean_fn, layer0_returns_aux=False):
    """Factory: build a stacked observable kernel for one cluster.

    Returns:
        stacked_kernel(cpdf_with_hmf, lnM, layer0_args,
                        mean_fn_pref_args, mean_fn_sr_args,
                        sigma_scatter_0, sigma_scatter_min,
                        n_points_stacked, compute_stacked_cov,
                        n_layers_stacked)
        -> (obs_mean, obs_var)
    """
    def stacked_kernel(cpdf_with_hmf, lnM,
                        layer0_args, mean_fn_pref_args, mean_fn_sr_args,
                        sigma_scatter_0, sigma_scatter_min,
                        n_points_stacked, compute_stacked_cov,
                        n_layers_stacked):
        # Normalise cpdf
        norm = simpson(cpdf_with_hmf, x=lnM)
        cpdf = cpdf_with_hmf / jnp.maximum(norm, 1e-300)
        x0 = lnM

        # Multi-layer propagation
        if n_layers_stacked > 1:
            if layer0_returns_aux:
                x1, _aux = layer0_fn(x0, *layer0_args)
            else:
                x1 = layer0_fn(x0, *layer0_args)

            dx1_dx0 = jnp.gradient(x1, x0)
            safe_deriv = jnp.where(jnp.abs(dx1_dx0) < 1e-30, 1e-30, jnp.abs(dx1_dx0))
            cpdf = cpdf / safe_deriv

            x1_interp = jnp.linspace(jnp.min(x1), jnp.max(x1), n_points_stacked)
            cpdf = jnp.interp(x1_interp, x1, cpdf, left=0., right=0.)

            cpdf = convolve_1d(x1_interp, cpdf, sigma=sigma_scatter_0,
                                sigma_min=sigma_scatter_min)
            x0 = x1_interp

            norm2 = simpson(cpdf, x=x0)
            cpdf = cpdf / jnp.maximum(norm2, 1e-300)
            sigma_intrinsic = jnp.float64(0.)
        else:
            sigma_intrinsic = sigma_scatter_0

        # mean_fn signature: (x0, *pref_args, *sr_args, sigma_intrinsic, compute_var=...)
        mean_vec, var_vec = mean_fn(
            x0, *mean_fn_pref_args, *mean_fn_sr_args,
            sigma_intrinsic, compute_var=compute_stacked_cov)

        obs_mean = simpson(mean_vec * cpdf, x=x0)
        obs_second_moment = simpson((var_vec + mean_vec**2) * cpdf, x=x0)
        obs_var = obs_second_moment - obs_mean**2

        return obs_mean, obs_var

    return stacked_kernel


def build_abundance_kernel(layer_fns, layer_deriv_fns,
                            layer_returns_aux_list, layer_deriv_uses_aux_list,
                            n_layers, n_points):
    """Factory: build an N-layer abundance kernel for one redshift slice.

    Layer functions are captured in the closure (constant across MCMC).
    All dynamic parameters are explicit arguments.
    n_layers and n_points are captured in closure (constant, control trace structure).

    Args:
        layer_fns: list of N pure JAX layer functions
        layer_deriv_fns: list of N derivative functions (or None per layer)
        layer_returns_aux_list: list of N booleans
        layer_deriv_uses_aux_list: list of N booleans
        n_layers: int, number of layers (for trace-time loop unrolling)
        n_points: int, grid size (controls array shapes)

    Returns:
        abundance_one_z(hmf_row, ln_M, obs_select_vec,
                        all_layer_args, all_deriv_args,
                        all_scatters, all_cutoff_vals, all_apply_cutoffs,
                        sigma_scatter_min, skyfrac, pad_abundance)
        where:
            all_layer_args: tuple of N tuples (layer k args = prefactors + sr_params)
            all_deriv_args: tuple of N tuples
            all_scatters: tuple of N scalars
            all_cutoff_vals: tuple of N scalars
            all_apply_cutoffs: tuple of N booleans
    """
    def abundance_one_z(hmf_row, ln_M, obs_select_vec,
                        all_layer_args, all_deriv_args,
                        all_scatters, all_cutoff_vals, all_apply_cutoffs,
                        sigma_scatter_min, skyfrac, pad_abundance):
        x0 = ln_M
        dn_dx0 = hmf_row

        for k in range(n_layers):
            layer_fn = layer_fns[k]
            layer_deriv_fn = layer_deriv_fns[k]
            returns_aux = layer_returns_aux_list[k]
            deriv_uses_aux = layer_deriv_uses_aux_list[k]

            layer_args = all_layer_args[k]
            deriv_args = all_deriv_args[k]
            scatter_k = all_scatters[k]
            cutoff_k = all_cutoff_vals[k]
            apply_cut_k = all_apply_cutoffs[k]

            # Forward through layer k
            if returns_aux:
                x1, aux = layer_fn(x0, *layer_args)
            else:
                x1 = layer_fn(x0, *layer_args)
                aux = x0

            # Derivative
            if layer_deriv_fn is not None:
                if deriv_uses_aux:
                    dx1_dx0 = layer_deriv_fn(aux, *deriv_args)
                else:
                    dx1_dx0 = layer_deriv_fn(x0, *deriv_args)
            else:
                dx1_dx0 = jnp.gradient(x1, x0)

            safe_d = jnp.where(dx1_dx0 == 0, 1.0, dx1_dx0)
            dn_dx1 = jnp.where((dx1_dx0 == 0) | jnp.isnan(dx1_dx0), 0.0, dn_dx0 / safe_d)

            # Apply cutoff before interpolation
            dn_dx1 = jnp.where(apply_cut_k & (x1 < cutoff_k), 0., dn_dx1)

            # Pad + interpolate to fixed-size grid
            pad = jnp.where(pad_abundance & (scatter_k > sigma_scatter_min),
                            8. * scatter_k, 0.)
            x_min = jnp.min(x1) - pad
            x_max = jnp.max(x1) + pad
            x1_interp = jnp.linspace(x_min, x_max, n_points)
            dn_dx1 = jnp.interp(x1_interp, x1, dn_dx1, left=0., right=0.)

            # Convolve with scatter
            dn_dx1 = convolve_1d(x1_interp, dn_dx1, sigma=scatter_k,
                                  sigma_min=sigma_scatter_min)

            x0 = x1_interp
            dn_dx0 = dn_dx1

        # Final: interpolate to obs_select_vec
        abundance = jnp.interp(obs_select_vec, x0, dn_dx0, left=0.) * 4. * jnp.pi * skyfrac
        return abundance

    return abundance_one_z


class cluster_number_counts:

    def __init__(self,cnc_params=None):

        self.cnc_params = dict(cnc_params_default) if cnc_params is None else dict(cnc_params)
        self.cosmo_params = dict(cosmo_params_default)
        self.scal_rel_params = dict(scaling_relation_params_default)

        self.abundance_matrix = None
        self.n_obs_matrix = None
        self.hmf_matrix = None
        self.n_tot = None
        self.n_binned = None
        self.abundance_tensor = None
        self.n_obs_false = 0.

        self.hmf_extra_params = {}

        self.cnc_params["M_min_cutoff"] = None

        if self.cnc_params["M_min_extended"] is not None:

                self.cnc_params["M_min_cutoff"] = self.cnc_params["M_min"]
                self.cnc_params["M_min"] =  self.cnc_params["M_min_extended"]

    #Loads data (catalogue and scaling relation data)

    def initialise(self):

        #Verbosity

        set_verbosity(self.cnc_params["cosmocnc_verbose"])
        self.logger = logging.getLogger(__name__)

        # Load the survey data

        path_to_survey = self.cnc_params["survey_sr"]
        spec = importlib.util.spec_from_file_location("scaling_relations_module",path_to_survey)
        self.survey_module = importlib.util.module_from_spec(spec)

        try:

            spec.loader.exec_module(self.survey_module)

        except Exception as e:

            print(f"Error loading survey module: {e}")
            print("Path to survey: ", path_to_survey)
            print("check file exists if you need that.")

        self.scaling_relations_survey = self.survey_module.scaling_relations
        self.scatter_survey = self.survey_module.scatter

        #Set cosmology

        self.cosmology = cosmology_model(cosmo_params=self.cosmo_params,
                                         cosmology_tool = self.cnc_params["cosmology_tool"],
                                         amplitude_parameter=self.cnc_params["cosmo_amplitude_parameter"],
                                         cnc_params = self.cnc_params,
                                         logger = self.logger
                                         )

        if self.cnc_params["load_catalogue"] == True:

            self.logger.debug("Loading catalogue")
            self.logger.debug(self.cnc_params["cluster_catalogue"])

            self.catalogue = cluster_catalogue(catalogue_name=self.cnc_params["cluster_catalogue"],
                                               precompute_cnc_quantities=self.cnc_params["precompute_cnc_quantities_catalogue"],
                                               bins_obs_select_edges=self.cnc_params["bins_edges_obs_select"],
                                               bins_z_edges=self.cnc_params["bins_edges_z"],
                                               observables=self.cnc_params["observables"],
                                               obs_select=self.cnc_params["obs_select"],
                                               cnc_params = self.cnc_params,
                                               scal_rel_params=self.scal_rel_params)

        elif self.cnc_params["load_catalogue"] == False:

            self.catalogue = None

        self.scaling_relations = {}

        for observable_set in self.cnc_params["observables"]:

            for observable in observable_set:

                self.scaling_relations[observable] = self.scaling_relations_survey(observable=observable,cnc_params=self.cnc_params,catalogue=self.catalogue)
                self.scaling_relations[observable].initialise_scaling_relation(cosmology=self.cosmology)

        if self.cnc_params["stacked_likelihood"] == True:

            self.stacked_data_labels = self.cnc_params["stacked_data"]

            for key in self.stacked_data_labels:

                observable = self.catalogue.stacked_data[key]["observable"]

                if observable not in self.cnc_params["observables"]:

                    self.scaling_relations[observable] = self.scaling_relations_survey(observable=observable,cnc_params=self.cnc_params,catalogue=self.catalogue)
                    self.scaling_relations[observable].initialise_scaling_relation(cosmology=self.cosmology)

        self.scatter = self.scatter_survey(params=self.scal_rel_params,catalogue=self.catalogue)
        self.scatter_ref = self.scatter_survey(params=self.scal_rel_params,catalogue=self.catalogue)

        if self.cnc_params["hmf_calc"] == "MiraTitan":

            import MiraTitanHMFemulator

            self.MT_emulator = MiraTitanHMFemulator.Emulator()
            self.hmf_extra_params["emulator"] = self.MT_emulator

        # Build generic JIT functions from scaling relation interface
        self._build_jit_functions()

    def _build_jit_functions(self):
        """Build all generic JIT functions at init time.

        Uses the scaling_relations factory interface to capture pure JAX functions
        in closures. Called once at init — the resulting JIT functions are reused
        across all MCMC iterations with SR params as explicit args.
        """
        obs_select_name = self.cnc_params["obs_select"]
        sr_sel = self.scaling_relations[obs_select_name]
        use_analytical = (self.cnc_params.get("scalrel_type_deriv", "analytical") == "analytical")
        self._use_analytical_deriv = use_analytical

        # ── 1. Backward conv functions per correlation set ──
        # _bc_set_fns: list of (bc_fn, obs_names_in_set) per correlation set
        # _bc_obs_list: flat list of all backward-conv observable names (for data gathering)
        self._bc_fns = {}          # per-observable 1D backward conv (kept for compatibility)
        self._bc_obs_list = []     # flat list of all bc observable names
        self._bc_set_fns = []      # list of (bc_fn, obs_names) per correlation set
        self._bc_set_obs = []      # list of obs_name lists per correlation set
        self._all_sets_are_1d = True  # fast-path flag
        self._1layer_obs_list = []   # 1-layer observables (direct PDF, no backward conv)

        for observable_set in self.cnc_params["observables"]:
            # Filter to observables with >= 2 layers (backward conv applicable)
            bc_obs_in_set = []
            for obs_name in observable_set:
                sr = self.scaling_relations[obs_name]
                if sr.get_n_layers() >= 2:
                    bc_obs_in_set.append(obs_name)
                    if obs_name not in self._bc_obs_list:
                        self._bc_obs_list.append(obs_name)
                elif sr.get_n_layers() == 1:
                    if obs_name not in self._1layer_obs_list:
                        self._1layer_obs_list.append(obs_name)

            if len(bc_obs_in_set) == 0 and len(self._1layer_obs_list) == 0:
                continue
            if len(bc_obs_in_set) == 0:
                continue

            n_obs_set = len(bc_obs_in_set)
            if n_obs_set > 1:
                self._all_sets_are_1d = False

            # Build layer functions for this set
            layer0_fns = []
            layer1_fns = []
            layer0_returns_aux_list = []
            for obs_name in bc_obs_in_set:
                sr = self.scaling_relations[obs_name]
                layer0_fns.append(sr.get_layer_fn(0))
                layer1_fns.append(sr.get_layer_fn(1))
                layer0_returns_aux_list.append(sr.get_layer_returns_aux(0))

            # Build N-D backward conv for this correlation set
            nd_circular = self.cnc_params.get("nd_convolution_mode", "linear") == "circular"
            bc_set_fn = build_backward_conv_nd(
                layer0_fns, layer1_fns, layer0_returns_aux_list, n_obs_set,
                nd_circular=nd_circular)
            self._bc_set_fns.append(bc_set_fn)
            self._bc_set_obs.append(bc_obs_in_set)

            # Also build per-observable 1D fns (for legacy/compatibility)
            for obs_name in bc_obs_in_set:
                if obs_name not in self._bc_fns:
                    sr = self.scaling_relations[obs_name]
                    self._bc_fns[obs_name] = build_backward_conv_1d(
                        sr.get_layer_fn(0), sr.get_layer_fn(1),
                        layer0_returns_aux=sr.get_layer_returns_aux(0))

        # ── 2. Mass range function for selection observable ──
        layer0_fn_sel = sr_sel.get_layer_fn(0)
        layer1_fn_sel = sr_sel.get_layer_fn(1)
        layer0_deriv_sel = sr_sel.get_layer_deriv_fn(0)
        layer1_deriv_sel = sr_sel.get_layer_deriv_fn(1)
        layer0_returns_aux_sel = sr_sel.get_layer_returns_aux(0)
        layer0_deriv_uses_aux_sel = sr_sel.get_layer_deriv_uses_aux(0)
        self._mass_range_fn = build_mass_range_fn(
            layer0_fn_sel, layer0_deriv_sel, layer1_fn_sel, layer1_deriv_sel,
            layer0_returns_aux=layer0_returns_aux_sel,
            layer0_deriv_uses_aux=layer0_deriv_uses_aux_sel)

        # ── 3. Vmapped prefactor functions for abundance computation ──
        self._pref_vmaps = {}
        for obs_name in self._bc_obs_list:
            sr = self.scaling_relations[obs_name]
            pref_fn = sr.get_prefactor_fn()
            vmap_axes = sr.get_prefactor_vmap_axes()
            self._pref_vmaps[obs_name] = jax.vmap(pref_fn, in_axes=vmap_axes)
        if obs_select_name not in self._pref_vmaps:
            pref_fn_sel = sr_sel.get_prefactor_fn()
            vmap_axes_sel = sr_sel.get_prefactor_vmap_axes()
            self._pref_vmaps[obs_select_name] = jax.vmap(pref_fn_sel, in_axes=vmap_axes_sel)

        # ── 4. All-in-one backward conv JIT: correlation sets + combine + integrate ──
        # Builds a single JIT function that processes all correlation sets per cluster,
        # computes prefactors, calls N-D backward conv per set, multiplies cpdf products
        # across sets, multiplies by HMF × 4π × skyfrac, and integrates.
        n_points_dl = int(self.cnc_params["n_points_data_lik"])
        self._n_pref_sr = {}
        for obs_name in self._bc_obs_list:
            sr = self.scaling_relations[obs_name]
            self._n_pref_sr[obs_name] = sr.get_n_prefactor_sr_params()

        # Build per-set metadata for the all-in-one JIT
        # Each set needs: bc_set_fn, list of pref_fns, list of n_psr, n_obs_in_set
        set_bc_fns = []          # N-D backward conv function per set
        set_pref_fns = []        # list of lists of pref_fn_unified per set
        set_n_psr = []           # list of lists of n_psr per set
        set_sizes = []           # n_obs per set
        set_obs_indices = []     # indices into flat _bc_obs_list per set

        for s_idx, (bc_set_fn, obs_names) in enumerate(
                zip(self._bc_set_fns, self._bc_set_obs)):
            set_bc_fns.append(bc_set_fn)
            pref_fns_set = []
            n_psr_set = []
            obs_idx_set = []
            for obs_name in obs_names:
                sr = self.scaling_relations[obs_name]
                pref_fns_set.append(sr.get_prefactor_fn_unified())
                n_psr_set.append(sr.get_n_prefactor_sr_params())
                obs_idx_set.append(self._bc_obs_list.index(obs_name))
            set_pref_fns.append(pref_fns_set)
            set_n_psr.append(n_psr_set)
            set_sizes.append(len(obs_names))
            set_obs_indices.append(obs_idx_set)

        # Also keep flat lists for the vmap interface (data is still stacked flat)
        flat_pref_fn_list = []
        flat_n_psr_list = []
        for obs_name in self._bc_obs_list:
            sr = self.scaling_relations[obs_name]
            flat_pref_fn_list.append(sr.get_prefactor_fn_unified())
            flat_n_psr_list.append(sr.get_n_prefactor_sr_params())

        # ── 4b. Split-JIT: build separate 2D core JITs for 2D+ correlation sets ──
        # This prevents XLA from over-fusing the 2D FFT conv with surrounding ops
        # (which causes shared memory overflow on Blackwell GPUs).
        padding_frac = self.cnc_params.get("padding_fraction", 0.)
        n_drop_int = int(padding_frac * n_points_dl) if padding_frac > 1e-5 else 0
        n_obs_bc = len(self._bc_obs_list)
        n_sets = len(set_bc_fns)

        # Determine which sets are N-D (2+) and build separate JITs for them.
        # Two-stage split: forward pass JIT (stage A) + conv core JIT (stage B).
        # Prevents XLA from fusing forward pass with 2D FFT conv, which causes
        # shared memory overflow (128×128×8 = 128KB > 101KB on Blackwell GPUs).
        s_is_nd = []
        self._2d_forward_jits = {}  # s_idx → jit(vmap(forward_pass))
        self._2d_conv_jits = {}    # s_idx → jit(vmap(conv_core))
        self._2d_core_obs_indices = {}  # s_idx → indices into _bc_obs_list

        for s_idx, obs_names in enumerate(self._bc_set_obs):
            if len(obs_names) >= 2:
                s_is_nd.append(True)
                l0_fns = [self.scaling_relations[o].get_layer_fn(0) for o in obs_names]
                l1_fns = [self.scaling_relations[o].get_layer_fn(1) for o in obs_names]
                l0_aux = [self.scaling_relations[o].get_layer_returns_aux(0) for o in obs_names]
                p_fns = [self.scaling_relations[o].get_prefactor_fn_unified() for o in obs_names]
                p_nsr = [self.scaling_relations[o].get_n_prefactor_sr_params() for o in obs_names]
                _pc_l0_counts = [len(self.scaling_relations[o].get_layer_sr_params_per_cluster(0, self.scal_rel_params))
                                 if hasattr(self.scaling_relations[o], 'get_layer_sr_params_per_cluster') else 0
                                 for o in obs_names]
                _pc_l1_counts = [len(self.scaling_relations[o].get_layer_sr_params_per_cluster(1, self.scal_rel_params))
                                 if hasattr(self.scaling_relations[o], 'get_layer_sr_params_per_cluster') else 0
                                 for o in obs_names]
                self._2d_forward_jits[s_idx] = build_2d_forward_fn(
                    l0_fns, l1_fns, l0_aux, p_fns, p_nsr, n_points_dl,
                    _pc_l0_counts, _pc_l1_counts)
                nd_circ = self.cnc_params.get("nd_convolution_mode", "linear") == "circular"
                self._2d_conv_jits[s_idx] = build_2d_conv_fn(n_points_dl, nd_circular=nd_circ)
                self._2d_core_obs_indices[s_idx] = [
                    self._bc_obs_list.index(o) for o in obs_names]
            else:
                s_is_nd.append(False)

        # ── 4b-bis. Sub-pattern JITs for partial observable availability ──
        # For each 2D+ correlation set, pre-build backward conv JITs for all
        # proper non-empty subsets.  When a cluster is missing some observables
        # in a set, we dispatch the appropriate sub-dimensional JIT.
        self._sub_bc_jits = {}          # (s_idx, sub_indices_tuple) → dict
        for s_idx, obs_names in enumerate(self._bc_set_obs):
            n_obs_s = len(obs_names)
            if n_obs_s < 2:
                continue
            full_pat = tuple(range(n_obs_s))
            # Full pattern → reference to existing 2D split-JIT
            self._sub_bc_jits[(s_idx, full_pat)] = {
                'type': '2d_split',
                'fwd_jit': self._2d_forward_jits[s_idx],
                'conv_jit': self._2d_conv_jits[s_idx],
            }
            # Enumerate all non-empty proper subsets
            for r in range(1, n_obs_s):
                for sub_indices in combinations(range(n_obs_s), r):
                    sub_obs = [obs_names[k] for k in sub_indices]
                    sub_l0 = [self.scaling_relations[o].get_layer_fn(0) for o in sub_obs]
                    sub_l1 = [self.scaling_relations[o].get_layer_fn(1) for o in sub_obs]
                    sub_aux = [self.scaling_relations[o].get_layer_returns_aux(0)
                               for o in sub_obs]
                    sub_pref = [self.scaling_relations[o].get_prefactor_fn_unified()
                                for o in sub_obs]
                    sub_npsr = [self.scaling_relations[o].get_n_prefactor_sr_params()
                                for o in sub_obs]
                    if r == 2:
                        sub_pc_l0 = [len(self.scaling_relations[o].get_layer_sr_params_per_cluster(0, self.scal_rel_params))
                                     if hasattr(self.scaling_relations[o], 'get_layer_sr_params_per_cluster') else 0
                                     for o in sub_obs]
                        sub_pc_l1 = [len(self.scaling_relations[o].get_layer_sr_params_per_cluster(1, self.scal_rel_params))
                                     if hasattr(self.scaling_relations[o], 'get_layer_sr_params_per_cluster') else 0
                                     for o in sub_obs]
                        fwd = build_2d_forward_fn(
                            sub_l0, sub_l1, sub_aux, sub_pref, sub_npsr,
                            n_points_dl, sub_pc_l0, sub_pc_l1)
                        conv = build_2d_conv_fn(n_points_dl, nd_circular=nd_circ)
                        self._sub_bc_jits[(s_idx, sub_indices)] = {
                            'type': '2d_split', 'fwd_jit': fwd, 'conv_jit': conv,
                        }
                    else:
                        # 1D or 3+D: generic wrapper
                        jit_fn = build_sub_bc_jit(
                            sub_l0, sub_l1, sub_aux, sub_pref, sub_npsr,
                            r, n_points_dl, nd_circ)
                        self._sub_bc_jits[(s_idx, sub_indices)] = {
                            'type': 'generic', 'jit_fn': jit_fn,
                        }

        # ── 4c. All-in-one backward conv + combine + integrate JIT ──
        # For 2D+ sets, uses pre-computed cpdf from split JIT (passed as input).
        # For 1D sets, computes backward conv inline.
        def _make_allinone_bc(s_bc_fns, s_pref_fns, s_n_psr, s_sizes, s_obs_idx,
                               f_pref_fns, f_n_psr, n_pts, n_o, n_s, n_drop,
                               s_is_nd_flags, n_pc_l0, n_pc_l1,
                               f_layer0_fns_1layer, f_pref_fns_1layer, n_1layer,
                               n_psr_1layer, n_pc_l0_1layer):
            def per_cluster(mn, mx, obs_vals, has_obs_vals, hz, skyfrac,
                            E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
                            H0, D_CMB, gamma, z_c,
                            all_pref_sr, all_layer0_sr, all_layer1_sr,
                            all_layer0_sr_pc, all_layer1_sr_pc,
                            all_cov_layer0, all_cov_layer1,
                            all_apply_cut, all_cut_val,
                            lnM0_min, lnM0_max, n_lnM0,
                            patch_idx,
                            obs_vals_1layer, has_obs_1layer,
                            all_pref_sr_1layer, all_layer0_sr_1layer,
                            all_layer0_sr_pc_1layer, all_cov_1layer,
                            pre_nd_cpdfs):
                # Shared: lnM grid and HMF interp (computed once)
                lnM = jnp.linspace(mn, mx, n_pts)
                hmf = interp_uniform(lnM, lnM0_min, lnM0_max, n_lnM0, hz,
                                     left=0., right=0.)
                cpdf_product = jnp.ones(n_pts)

                for s in range(n_s):
                    n_obs_s = s_sizes[s]
                    idx = s_obs_idx[s]  # indices into flat obs arrays

                    # Check if ANY observable in this set is present
                    any_has = False
                    for k in range(n_obs_s):
                        any_has = any_has | has_obs_vals[idx[k]]

                    if s_is_nd_flags[s]:
                        # 2D+ set: use pre-computed cpdf from split JIT
                        cpdf = pre_nd_cpdfs[s]
                    else:
                        # 1D set: compute prefactors + backward conv inline
                        set_layer0_args = []
                        set_layer1_args = []
                        set_obs = []
                        for k in range(n_obs_s):
                            i = idx[k]
                            psr_i = tuple(p[patch_idx] for p in all_pref_sr[i])
                            l0_i = tuple(p[patch_idx] for p in all_layer0_sr[i])
                            l1_i = tuple(p[patch_idx] for p in all_layer1_sr[i])
                            # Per-cluster params (already sliced by vmap)
                            l0_pc_i = all_layer0_sr_pc[i]
                            l1_pc_i = all_layer1_sr_pc[i]
                            prefactors = f_pref_fns[i](E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
                                                        H0, D_CMB, gamma, z_c, *psr_i)
                            set_layer0_args.append(prefactors + l0_i + l0_pc_i)
                            set_layer1_args.append(prefactors + l1_i + l1_pc_i)
                            set_obs.append(obs_vals[idx[k]])

                        set_obs_arr = jnp.array(set_obs)

                        cpdf = s_bc_fns[s](
                            lnM, set_obs_arr,
                            tuple(set_layer0_args), tuple(set_layer1_args),
                            all_cov_layer0[s], all_cov_layer1[s],
                            n_pts, all_apply_cut[s], all_cut_val[s])

                    cpdf_product = cpdf_product * jnp.where(any_has, cpdf, 1.)

                # 1-layer observables: direct Gaussian PDF (no backward conv)
                # Matches cosmocnc lines 771-781
                for ol_idx in range(n_1layer):
                    ol_obs_val = obs_vals_1layer[ol_idx]
                    ol_has = has_obs_1layer[ol_idx]
                    psr_ol = tuple(p[patch_idx] for p in all_pref_sr_1layer[ol_idx])
                    l0_ol = tuple(p[patch_idx] for p in all_layer0_sr_1layer[ol_idx])
                    l0_pc_ol = all_layer0_sr_pc_1layer[ol_idx]
                    prefactors_ol = f_pref_fns_1layer[ol_idx](
                        E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
                        H0, D_CMB, gamma, z_c, *psr_ol)
                    layer0_args_ol = prefactors_ol + l0_ol + l0_pc_ol
                    predicted = f_layer0_fns_1layer[ol_idx](lnM, *layer0_args_ol)
                    residual = predicted - ol_obs_val
                    cov_ol = all_cov_1layer[ol_idx]
                    # Gaussian PDF: N(residual; 0, cov) = exp(-0.5*r^2/cov) / sqrt(2*pi*cov)
                    cpdf_1l = jnp.exp(-0.5 * residual**2 / cov_ol) / jnp.sqrt(2. * jnp.pi * cov_ol)
                    cpdf_product = cpdf_product * jnp.where(ol_has, cpdf_1l, 1.)

                cwh = cpdf_product * hmf * 4. * jnp.pi * skyfrac
                if n_drop > 0:
                    log_lik = jnp.log(jnp.maximum(
                        simpson(cwh[n_drop:-n_drop], x=lnM[n_drop:-n_drop]), 1e-300))
                else:
                    log_lik = jnp.log(jnp.maximum(simpson(cwh, x=lnM), 1e-300))
                return log_lik, cwh, lnM

            # Build vmap axes
            vmap_in = (
                0, 0,       # mn, mx
                1, 1,       # obs_vals (n_obs, n_bc), has_obs_vals (n_obs, n_bc)
                0, 0,       # hz, skyfrac
                0, 0, 0, 0, # cosmo per-cluster
                None, None, None, 0,  # H0, D_CMB, gamma, z_c
                tuple([tuple([None]*n) for n in f_n_psr]),  # all_pref_sr (n_patches, ...)
                tuple([None]*n_o),  # all_layer0_sr (n_patches, ...)
                tuple([None]*n_o),  # all_layer1_sr (n_patches, ...)
                tuple([tuple([0]*n) if n > 0 else () for n in n_pc_l0]),  # per-cluster L0
                tuple([tuple([0]*n) if n > 0 else () for n in n_pc_l1]),  # per-cluster L1
                tuple([None]*n_s),  # all_cov_layer0 (per-set)
                tuple([None]*n_s),  # all_cov_layer1 (per-set)
                tuple([None]*n_s),  # all_apply_cut (per-set)
                tuple([None]*n_s),  # all_cut_val (per-set)
                None, None, None,   # lnM0_min, lnM0_max, n_lnM0
                0,                  # patch_idx (per-cluster)
                # 1-layer observables (obs_vals/has always (max(n_1layer,1), n_bc), axis 1)
                1, 1,  # obs_vals_1layer, has_obs_1layer
                tuple([tuple([None]*n) for n in n_psr_1layer]) if n_1layer > 0 else (),
                tuple([None]*n_1layer) if n_1layer > 0 else (),
                tuple([tuple([0]*n) if n > 0 else () for n in n_pc_l0_1layer]) if n_1layer > 0 else (),
                tuple([None]*n_1layer) if n_1layer > 0 else (),
                # end 1-layer
                tuple([0]*n_s),     # pre_nd_cpdfs: all vmapped on axis 0
            )
            return jax.jit(jax.vmap(per_cluster, in_axes=vmap_in)), per_cluster

        # Count per-cluster params per observable
        _n_pc_l0 = []
        _n_pc_l1 = []
        for o in self._bc_obs_list:
            sr = self.scaling_relations[o]
            if hasattr(sr, 'get_layer_sr_params_per_cluster'):
                _n_pc_l0.append(len(sr.get_layer_sr_params_per_cluster(0, self.scal_rel_params)))
                _n_pc_l1.append(len(sr.get_layer_sr_params_per_cluster(1, self.scal_rel_params)))
            else:
                _n_pc_l0.append(0)
                _n_pc_l1.append(0)

        # Build 1-layer observable data
        _1layer_l0_fns = []
        _1layer_pref_fns = []
        _1layer_n_psr = []
        _1layer_n_pc_l0 = []
        for o in self._1layer_obs_list:
            sr = self.scaling_relations[o]
            _1layer_l0_fns.append(sr.get_layer_fn(0))
            _1layer_pref_fns.append(sr.get_prefactor_fn_unified())
            _1layer_n_psr.append(sr.get_n_prefactor_sr_params())
            if hasattr(sr, 'get_layer_sr_params_per_cluster'):
                _1layer_n_pc_l0.append(len(sr.get_layer_sr_params_per_cluster(0, self.scal_rel_params)))
            else:
                _1layer_n_pc_l0.append(0)
        n_1layer = len(self._1layer_obs_list)

        self._allinone_bc_jit, self._allinone_bc_per_cluster = _make_allinone_bc(
            set_bc_fns, set_pref_fns, set_n_psr, set_sizes, set_obs_indices,
            flat_pref_fn_list, flat_n_psr_list, n_points_dl,
            n_obs_bc, n_sets, n_drop_int, s_is_nd,
            _n_pc_l0, _n_pc_l1,
            _1layer_l0_fns, _1layer_pref_fns, n_1layer,
            _1layer_n_psr, _1layer_n_pc_l0)

        # ── 4c. Merged mass_range with prefactors JIT ──
        mass_range_fn_inner = self._mass_range_fn
        pref_fn_sel_u = sr_sel.get_prefactor_fn_unified()
        n_psr_sel = sr_sel.get_n_prefactor_sr_params()

        def _make_mass_range_with_pref(mr_fn, pref_fn_i, n_psr_i):
            def per_cluster(obs_val,
                            E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
                            H0, D_CMB, gamma, z_c,
                            ref_pref_sr,
                            ref_layer0_sr, ref_layer1_sr,
                            ref_layer0_deriv_sr, ref_layer1_deriv_sr,
                            ref_scatter, sigma_mass_prior,
                            lnM0_min, lnM0_max, lnM_coarse,
                            patch_idx):
                psr = tuple(p[patch_idx] for p in ref_pref_sr)
                l0 = tuple(p[patch_idx] for p in ref_layer0_sr)
                l1 = tuple(p[patch_idx] for p in ref_layer1_sr)
                l0d = tuple(p[patch_idx] for p in ref_layer0_deriv_sr)
                l1d = tuple(p[patch_idx] for p in ref_layer1_deriv_sr)
                scat = ref_scatter[patch_idx]
                ref_prefactors = pref_fn_i(E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
                                           H0, D_CMB, gamma, z_c, *psr)
                layer0_args = ref_prefactors + l0
                layer1_args = ref_prefactors + l1
                return mr_fn(lnM_coarse, obs_val,
                             layer0_args, layer1_args,
                             l0d, l1d,
                             scat, sigma_mass_prior,
                             lnM0_min, lnM0_max)
            vmap_in = (0,                         # obs_val
                       0, 0, 0, 0,                # cosmo per-cluster
                       None, None, None, 0,        # cosmo scalars + z_c (per-cluster)
                       tuple([None]*n_psr_i),      # ref_pref_sr (n_patches, ...)
                       None, None, None, None,     # layer sr + deriv (n_patches, ...)
                       None, None,                 # scatter (n_patches,), sigma_mp
                       None, None, None,           # lnM0 bounds, lnM_coarse
                       0)                          # patch_idx (per-cluster)
            return jax.jit(jax.vmap(per_cluster, in_axes=vmap_in))

        self._mass_range_with_pref_jit = _make_mass_range_with_pref(
            mass_range_fn_inner, pref_fn_sel_u, n_psr_sel)

        # ── 5. N-layer abundance kernel for selection observable ──
        n_layers_sel = sr_sel.get_n_layers()
        self._n_layers_sel = n_layers_sel
        self._n_pref_sel = sr_sel.get_n_prefactors()

        layer_fns_sel = []
        layer_deriv_fns_sel = []
        layer_returns_aux_sel = []
        layer_deriv_uses_aux_sel = []
        for k in range(n_layers_sel):
            layer_fns_sel.append(sr_sel.get_layer_fn(k))
            layer_deriv_fns_sel.append(sr_sel.get_layer_deriv_fn(k) if use_analytical else None)
            layer_returns_aux_sel.append(sr_sel.get_layer_returns_aux(k))
            layer_deriv_uses_aux_sel.append(sr_sel.get_layer_deriv_uses_aux(k))

        n_points_abund = int(self.cnc_params["n_points"])
        abundance_kernel_fn = build_abundance_kernel(
            layer_fns_sel, layer_deriv_fns_sel,
            layer_returns_aux_sel, layer_deriv_uses_aux_sel,
            n_layers_sel, n_points_abund)

        # Build vmap axes for all_layer_args (nested tuple)
        # All layers: prefactors (per-z axis 0) + sr_params (shared None)
        dummy_sr = self.scal_rel_params
        z_layer_args_axes = []
        for k in range(n_layers_sel):
            sr_k = sr_sel.get_layer_sr_params(k, dummy_sr)
            n_sr_k = len(sr_k)
            z_layer_args_axes.append(tuple([0] * self._n_pref_sel + [None] * n_sr_k))
        z_layer_args_axes = tuple(z_layer_args_axes)

        z_deriv_axes = []
        for k in range(n_layers_sel):
            dfn = sr_sel.get_layer_deriv_fn(k) if use_analytical else None
            if dfn is not None:
                dp = sr_sel.get_layer_deriv_sr_params(k, dummy_sr)
                z_deriv_axes.append(tuple([None] * len(dp)) if len(dp) > 0 else ())
            else:
                z_deriv_axes.append(())
        z_deriv_axes = tuple(z_deriv_axes)

        z_vmap_in_axes = (
            0, None, None,             # hmf_row, ln_M, obs_select_vec
            z_layer_args_axes,          # all_layer_args
            z_deriv_axes,               # all_deriv_args
            tuple([None]*n_layers_sel), # all_scatters
            tuple([None]*n_layers_sel), # all_cutoff_vals
            tuple([None]*n_layers_sel), # all_apply_cutoffs
            None, None, None,           # sigma_scatter_min, skyfrac, pad_abundance
        )
        abundance_vmap_z = jax.vmap(abundance_kernel_fn, in_axes=z_vmap_in_axes)

        # Vmap over patches on top of z-vmap
        # All layer args (prefactors + SR params) are now per-patch (axis 0)
        p_layer_args_axes = []
        for k in range(n_layers_sel):
            sr_k = sr_sel.get_layer_sr_params(k, dummy_sr)
            n_sr_k = len(sr_k)
            p_layer_args_axes.append(tuple([0] * (self._n_pref_sel + n_sr_k)))
        p_layer_args_axes = tuple(p_layer_args_axes)

        # Deriv args: also per-patch (axis 0) in patch vmap
        p_deriv_axes = []
        for k in range(n_layers_sel):
            dfn = sr_sel.get_layer_deriv_fn(k) if use_analytical else None
            if dfn is not None:
                dp = sr_sel.get_layer_deriv_sr_params(k, dummy_sr)
                p_deriv_axes.append(tuple([0] * len(dp)) if len(dp) > 0 else ())
            else:
                p_deriv_axes.append(())
        p_deriv_axes = tuple(p_deriv_axes)

        p_vmap_in_axes = (
            None, None, None,           # hmf_matrix, ln_M, obs_select_vec
            p_layer_args_axes,           # all_layer_args (per-patch)
            p_deriv_axes,                # all_deriv_args (per-patch)
            tuple([0]*n_layers_sel),     # all_scatters (per-patch)
            tuple([None]*n_layers_sel),  # all_cutoff_vals
            tuple([None]*n_layers_sel),  # all_apply_cutoffs
            None, 0, None,              # sigma_scatter_min, skyfrac, pad_abundance
        )
        abundance_vmap_pz = jax.vmap(abundance_vmap_z, in_axes=p_vmap_in_axes)

        # JIT wrapper for full abundance computation
        def _make_jit_abundance(vmap_pz):
            @jax.jit
            def compute(hmf_matrix, ln_M, obs_select_vec, redshift_vec,
                        all_layer_args, all_deriv_args,
                        all_scatters, all_cutoff_vals, all_apply_cutoffs,
                        sigma_scatter_min, skyfracs, pad_abundance, total_skyfrac):
                abundance_tensor = vmap_pz(
                    hmf_matrix, ln_M, obs_select_vec,
                    all_layer_args, all_deriv_args,
                    all_scatters, all_cutoff_vals, all_apply_cutoffs,
                    sigma_scatter_min, skyfracs, pad_abundance)
                n_obs_matrix = simpson(abundance_tensor, x=redshift_vec, axis=1)
                n_tot_vec = simpson(n_obs_matrix, x=obs_select_vec, axis=-1)
                abundance_matrix = jnp.sum(abundance_tensor, axis=0)
                n_z_vec = simpson(abundance_matrix, x=obs_select_vec)
                n_tot = jnp.sum(n_tot_vec)
                dndz_hmf = simpson(hmf_matrix * 4. * jnp.pi * total_skyfrac, x=ln_M, axis=1)
                n_tot_hmf = simpson(dndz_hmf, x=redshift_vec)
                return (abundance_tensor, n_obs_matrix, n_tot_vec,
                        abundance_matrix, n_z_vec, n_tot, dndz_hmf, n_tot_hmf)
            return compute
        self._jit_compute_abundance = _make_jit_abundance(abundance_vmap_pz)

        # ── 6. Cosmo interpolation JIT (all dynamic data as args) ──
        @jax.jit
        def _interp_cosmo_jit(z_obs, D_A, E_z, D_l_CMB, rho_c, hmf_ds,
                               z_min, z_max, n_z):
            D_A_c = jax.vmap(lambda z: interp_uniform(z, z_min, z_max, n_z, D_A))(z_obs)
            E_z_c = jax.vmap(lambda z: interp_uniform(z, z_min, z_max, n_z, E_z))(z_obs)
            D_l_CMB_c = jax.vmap(lambda z: interp_uniform(z, z_min, z_max, n_z, D_l_CMB))(z_obs)
            rho_c_c = jax.vmap(lambda z: interp_uniform(z, z_min, z_max, n_z, rho_c))(z_obs)
            hmf_z_c = jax.vmap(lambda z: interp_along_axis0_uniform(z, z_min, z_max, n_z, hmf_ds))(z_obs)
            return D_A_c, E_z_c, D_l_CMB_c, rho_c_c, hmf_z_c
        self._interp_cosmo_jit = _interp_cosmo_jit

        # ── 7. Stacked kernel functions ──
        self._stacked_kernels = {}
        if self.cnc_params.get("stacked_likelihood", False):
            for key in self.cnc_params.get("stacked_data", []):
                stacked_obs = self.catalogue.stacked_data[key]["observable"]
                sr_st = self.scaling_relations.get(stacked_obs)
                if sr_st is not None:
                    mean_fn = sr_st.get_mean_fn()
                    if mean_fn is not None:
                        layer0_fn_st = sr_st.get_layer_fn(0)
                        layer0_aux_st = sr_st.get_layer_returns_aux(0)
                        self._stacked_kernels[stacked_obs] = build_stacked_kernel(
                            layer0_fn_st, mean_fn, layer0_returns_aux=layer0_aux_st)

    #Updates parameter values (cosmological and scaling relation)

    def reinitialise(self):

        self.abundance_matrix = None
        self.n_obs_matrix = None
        self.hmf_matrix = None
        self.n_tot = None
        self.abundance_tensor = None

    def update_params(self,cosmo_params,scal_rel_params):

        self.cosmo_params = cosmo_params
        self.cosmology.update_cosmology(cosmo_params,cosmology_tool=self.cnc_params["cosmology_tool"])
        self.scal_rel_params = {}

        for key in scal_rel_params.keys():

            self.scal_rel_params[key] = scal_rel_params[key]

        self.scatter = self.scatter_survey(params=self.scal_rel_params,catalogue=self.catalogue)

        self.abundance_matrix = None
        self.n_obs_matrix = None
        self.hmf_matrix = None
        self.n_tot = None
        self.abundance_tensor = None
        self.n_obs_matrix_fd = None

    #Computes the hmf as a function of redshift

    def get_hmf(self,volume_element=True):

        self.const = constants()

        #Define redshift and observable ranges

        self.redshift_vec = jnp.linspace(self.cnc_params["z_min"],self.cnc_params["z_max"],self.cnc_params["n_z"])
        self.obs_select_vec = jnp.linspace(self.cnc_params["obs_select_min"],self.cnc_params["obs_select_max"],self.cnc_params["n_points"])

        #Evaluate some useful quantities (to be potentially passed to scaling relations)

        if self.cnc_params["cosmology_tool"] in ("classy_sz_jax", "hmfast"):
            # === Fast path: direct emulator calls (no Cython, no monkey-patching) ===
            from cosmocnc_jax.emulators import build_cosmo_vec
            _pvd = self.cosmology._pvd
            _h = self.cosmology.cosmo_params["h"]
            cosmo = self.cosmology

            # H/c at z=0 and at redshift_vec (direct H emulator, JIT'd)
            cosmo_vec_h = build_cosmo_vec(_pvd, cosmo._emu_param_orders['h'])
            z_with_0 = jnp.concatenate([jnp.array([0.]), self.redshift_vec])
            H_over_c_all = cosmo._predict_H(cosmo_vec_h, z_with_0)
            H_over_c_0 = H_over_c_all[0]
            H_over_c_z = H_over_c_all[1:]
            self.E_z = H_over_c_z / H_over_c_0

            # D_A at redshift_vec (direct DA emulator, JIT'd)
            cosmo_vec_da = build_cosmo_vec(_pvd, cosmo._emu_param_orders['da'])
            self.D_A = cosmo._predict_DA(cosmo_vec_da, self.redshift_vec)

            # D_l_CMB: emulator can't handle z_CMB~1089, use chi-based formula
            _chi_z = self.D_A * (1. + self.redshift_vec)
            _chi_CMB = cosmo.D_CMB * (1. + cosmo.z_CMB)
            self.D_l_CMB = (_chi_CMB - _chi_z) / (1. + cosmo.z_CMB)

            # rho_c in M_sun/Mpc³ (matches astropy critical_density conversion)
            # ρ_c = 3H²/(8πG) in kg/m³, then * Mpc_m³/M_sun to get M_sun/Mpc³
            # With H_over_c in 1/Mpc: H_SI = H_over_c * c / Mpc_m
            # => ρ_c = (3/(8πG·M_sun)) · Mpc_m · c² · H_over_c²
            _G = 6.67428e-11            # m³/(kg·s²)
            _M_sun = 1.98855e30         # kg
            _Mpc_m = 3.085677581282e22  # m
            _c_ms = 2.99792458e8        # m/s
            _rho_prefactor = 3. / (8. * jnp.pi * _G * _M_sun) * _Mpc_m * _c_ms**2
            self.rho_c = _rho_prefactor * H_over_c_z**2

        else:
            self.D_A = jnp.asarray(self.cosmology.background_cosmology.angular_diameter_distance(np.asarray(self.redshift_vec)).value)
            self.E_z = jnp.asarray(self.cosmology.background_cosmology.H(np.asarray(self.redshift_vec)).value/(self.cosmology.cosmo_params["h"]*100.))
            self.D_l_CMB = jnp.asarray(self.cosmology.background_cosmology.angular_diameter_distance_z1z2(np.asarray(self.redshift_vec),self.cosmology.z_CMB).value)
            self.rho_c = jnp.asarray(self.cosmology.background_cosmology.critical_density(np.asarray(self.redshift_vec)).value*1000.*self.const.mpc**3/self.const.solar)

        # Update cosmology-dependent per-cluster quantities (e.g., beta_avg for WL)
        for sr in self.scaling_relations.values():
            if hasattr(sr, 'update_beta_avg'):
                sr.update_beta_avg(D_A=self.D_A, redshift_vec=self.redshift_vec)

        #Evaluate the halo mass function

        self.halo_mass_function = halo_mass_function(cosmology=self.cosmology,hmf_type=self.cnc_params["hmf_type"],
        mass_definition=self.cnc_params["mass_definition"],M_min=self.cnc_params["M_min"],M_min_cutoff=self.cnc_params["M_min_cutoff"],
        M_max=self.cnc_params["M_max"],n_points=self.cnc_params["n_points"],type_deriv=self.cnc_params["hmf_type_deriv"],
        hmf_calc=self.cnc_params["hmf_calc"],extra_params=self.hmf_extra_params,logger = self.logger,interp_tinker=self.cnc_params["interp_tinker"])

        t0 = time.time()

        if self.cnc_params["hmf_calc"] == "cnc" or self.cnc_params["hmf_calc"] == "hmf":

            if self.cnc_params["hmf_calc"] == "cnc" and self.cnc_params["hmf_type"] == "Tinker08":

                # Precompute sigma arrays for all redshifts using mcfit JAX backend
                M_vec = jnp.exp(jnp.linspace(jnp.log(self.cnc_params["M_min"]),jnp.log(self.cnc_params["M_max"]),self.cnc_params["n_points"]))
                rho_m = self.halo_mass_function.rho_c_0 * self.cosmology.cosmo_params["Om0"]

                delta_num = float(self.cnc_params["mass_definition"][0:-1])

                # === Fast JAX path: direct emulator calls ===
                if self.cnc_params["cosmology_tool"] in ("classy_sz_jax", "hmfast"):

                    # Batch P(k) from direct PKL emulator (JIT'd, vmapped)
                    pkl_keys = [k for k in cosmo._emu_param_orders['pkl']
                                if k != 'z_pk_save_nonclass']
                    cosmo_vec_pkl = build_cosmo_vec(_pvd, pkl_keys)
                    pk_batch = cosmo._predict_pk_batch(cosmo_vec_pkl,
                                                       self.redshift_vec)
                    k_arr = cosmo._k_arr

                    # Delta: pure JAX (same formula as classy.pyx lines 3266-3275)
                    if self.cnc_params["mass_definition"][-1] == "c":
                        Delta_vec = delta_num / cosmo._Omega_m_z_nonu(self.redshift_vec)
                    else:
                        Delta_vec = jnp.full(self.cnc_params["n_z"], delta_num)

                    # Volume: dV/(dz·dΩ) = (c/H₀) · chi²/E_z = chi²/H_over_c  [Mpc³]
                    if volume_element:
                        volume_element_vec = _chi_z**2 / H_over_c_z
                    else:
                        volume_element_vec = jnp.ones(self.cnc_params["n_z"])

                # === Original serial path for other cosmology tools ===
                else:
                    z_vals = np.asarray(self.redshift_vec)
                    pk_list = []
                    k_arr = None
                    Delta_list = []
                    vol_list = []

                    for i in range(self.cnc_params["n_z"]):
                        z_i = float(z_vals[i])

                        k, ps = self.cosmology.power_spectrum.get_linear_power_spectrum(z_i)
                        if k_arr is None:
                            k_arr = jnp.asarray(k)
                        pk_list.append(jnp.asarray(ps))

                        # Compute Delta (overdensity w.r.t. mean) for this redshift
                        if self.cnc_params["mass_definition"][-1] == "c":
                            if self.cosmology.cnc_params["cosmology_tool"] == "cobaya_cosmo":
                                rescale = self.cosmology.Om(z_i)/(self.cosmology.H(z_i)/100.)**2
                            else:
                                rescale = self.cosmology.cosmo_params["Om0"]*(1.+z_i)**3/(self.cosmology.background_cosmology.H(z_i).value/(self.cosmology.cosmo_params["h"]*100.))**2
                        else:
                            rescale = 1.0
                        Delta_list.append(delta_num/rescale)

                        # Volume element
                        if volume_element:
                            vol_list.append(float(self.cosmology.background_cosmology.differential_comoving_volume(z_i).value))
                        else:
                            vol_list.append(1.0)

                    pk_batch = jnp.stack(pk_list, axis=0)  # (n_z, n_k)
                    Delta_vec = jnp.array(Delta_list)
                    volume_element_vec = jnp.array(vol_list)

                # Create TophatVar objects and cached vmap functions once (reuse across MCMC iterations)
                from mcfit import TophatVar
                if not hasattr(self, '_tv0'):
                    self._tv0 = TophatVar(np.asarray(k_arr), lowring=True, deriv=0, backend='jax')
                    self._tv1 = TophatVar(np.asarray(k_arr), lowring=True, deriv=1, backend='jax')
                    self._batch_sigma_fns = build_batch_sigma_fns(
                        self._tv0, self._tv1, k_arr,
                        type_deriv=self.cnc_params["hmf_type_deriv"])

                # Batch compute sigma(M) and dsigma/dR(M) via cached vmapped FFTLog
                sigma_matrix, dsigma_matrix, R_matrix = batch_sigma_R_from_tophat(
                    self._tv0, self._tv1, pk_batch, k_arr, M_vec, rho_m,
                    type_deriv=self.cnc_params["hmf_type_deriv"],
                    _cached_fns=self._batch_sigma_fns)

                # Use interp_tinker setting to choose parameter grid
                interp_log = (self.cnc_params["interp_tinker"] == "log")
                if interp_log:
                    tinker_Delta = TINKER08_DELTA_LOG
                else:
                    tinker_Delta = TINKER08_DELTA_LIN

                M_min_cutoff = self.cnc_params["M_min_cutoff"] if self.cnc_params["M_min_cutoff"] is not None else -1.0

                # JIT-compiled HMF computation
                self.hmf_matrix = compute_hmf_matrix_jit(
                    sigma_matrix, dsigma_matrix, R_matrix,
                    M_vec, rho_m, self.redshift_vec, Delta_vec, volume_element_vec,
                    tinker_Delta, TINKER08_A, TINKER08_a, TINKER08_b, TINKER08_c,
                    M_min_cutoff, interp_log)

                self.ln_M = jnp.log(M_vec/1e14)

            else:
                # Fallback for hmf_calc="hmf" — keep original loop
                hmf_list = []
                z_vals_fb = np.asarray(self.redshift_vec)
                for i in range(self.cnc_params["n_z"]):
                    ln_M, hmf_eval = self.halo_mass_function.eval_hmf(float(z_vals_fb[i]),log=True,volume_element=volume_element)
                    hmf_list.append(hmf_eval)
                self.ln_M = ln_M
                self.hmf_matrix = jnp.stack(hmf_list)

        elif self.cnc_params["hmf_calc"] == "MiraTitan":

            self.ln_M,self.hmf_matrix = self.halo_mass_function.eval_hmf(np.asarray(self.redshift_vec),log=True,volume_element=volume_element)

        t1 = time.time()

        self.t_hmf = t1-t0

        self.time_back = 0.
        self.time_hmf2 = 0.
        self.time_select = 0.
        self.time_mass_range = 0.
        self.t_00 = 0.
        self.t_11 = 0.
        self.t_22 = 0.
        self.t_33 = 0.
        self.t_44 = 0.
        self.t_55 = 0.
        self.t_66 = 0.
        self.t_77 = 0.
        self.t_88 = 0.
        self.t_99 = 0.

    #Computes the cluster abundance across selection observable and redshift

    def get_cluster_abundance(self):

        if self.hmf_matrix is None:

            self.get_hmf()

        self.scal_rel_selection = self.scaling_relations[self.cnc_params["obs_select"]]

        skyfracs = self.scal_rel_selection.skyfracs
        self.n_patches = len(skyfracs)

        obs_select = self.cnc_params["obs_select"]
        sr_sel = self.scal_rel_selection
        sr_sel.params = self.scal_rel_params  # ensure get_cutoff can access params
        H0 = self.cosmology.cosmo_params["h"] * 100.
        gamma = constants().gamma

        # Build cosmo quantities dict for prefactor computation
        cosmo_q = {
            "E_z": self.E_z, "H0": jnp.float64(H0),
            "D_A": self.D_A, "D_CMB": jnp.float64(self.cosmology.D_CMB),
            "D_l_CMB": self.D_l_CMB, "rho_c": self.rho_c,
            "gamma": jnp.float64(gamma),
            "z": self.redshift_vec,
        }

        # Compute prefactors for all redshifts (vmapped)
        pref_args = sr_sel.get_prefactor_args(cosmo_q, self.scal_rel_params)
        prefactors = self._pref_vmaps[obs_select](*pref_args)  # tuple of (n_z,) arrays

        # Build all_layer_args: tuple of n_layers tuples
        # All layers receive: prefactors + sr_params (matching cosmocnc's other_params)
        all_layer_args_list = []
        for k in range(self._n_layers_sel):
            sr_params_k = sr_sel.get_layer_sr_params(k, self.scal_rel_params)
            all_layer_args_list.append(prefactors + sr_params_k)
        all_layer_args = tuple(all_layer_args_list)

        # Build all_deriv_args
        all_deriv_args_list = []
        for k in range(self._n_layers_sel):
            if self._use_analytical_deriv and sr_sel.get_layer_deriv_fn(k) is not None:
                all_deriv_args_list.append(sr_sel.get_layer_deriv_sr_params(k, self.scal_rel_params))
            else:
                all_deriv_args_list.append(())
        all_deriv_args = tuple(all_deriv_args_list)

        # Scatter per layer per patch: tuple of n_layers arrays each (n_patches,)
        all_scatters_list = []
        for k in range(self._n_layers_sel):
            scatter_per_patch = jnp.array([
                jnp.sqrt(self.scatter.get_cov(
                    observable1=obs_select, observable2=obs_select,
                    layer=k, patch1=i, patch2=i))
                for i in range(self.n_patches)])
            all_scatters_list.append(scatter_per_patch)
        all_scatters = tuple(all_scatters_list)

        # Cutoff per layer
        apply_cutoff_cfg = self.cnc_params["apply_obs_cutoff"]
        do_cutoff = (apply_cutoff_cfg != False and
                     apply_cutoff_cfg.get(str([obs_select]), False) == True)
        all_cutoff_vals_list = []
        all_apply_cutoffs_list = []
        for k in range(self._n_layers_sel):
            if do_cutoff and hasattr(sr_sel, 'get_cutoff'):
                cv = sr_sel.get_cutoff(layer=k)
                all_cutoff_vals_list.append(jnp.float64(cv))
                all_apply_cutoffs_list.append(jnp.bool_(cv > -1e30))
            else:
                all_cutoff_vals_list.append(jnp.float64(-jnp.inf))
                all_apply_cutoffs_list.append(jnp.bool_(False))
        all_cutoff_vals = tuple(all_cutoff_vals_list)
        all_apply_cutoffs = tuple(all_apply_cutoffs_list)

        # For patches vmap: all layer args need (n_patches, ...) shape
        # Default: tile (zero-copy broadcast). Multi-patch surveys can
        # override get_layer_sr_params to return genuinely per-patch data.
        all_layer_args_patched_list = []
        patched_pref = _tile_to_patches(prefactors, self.n_patches)
        for k in range(self._n_layers_sel):
            sr_params_k_patched = _tile_to_patches(
                sr_sel.get_layer_sr_params(k, self.scal_rel_params), self.n_patches)
            all_layer_args_patched_list.append(patched_pref + sr_params_k_patched)
        all_layer_args_patched = tuple(all_layer_args_patched_list)

        # Deriv args: also per-patch
        all_deriv_args_patched_list = []
        for k in range(self._n_layers_sel):
            if self._use_analytical_deriv and sr_sel.get_layer_deriv_fn(k) is not None:
                all_deriv_args_patched_list.append(_tile_to_patches(
                    sr_sel.get_layer_deriv_sr_params(k, self.scal_rel_params), self.n_patches))
            else:
                all_deriv_args_patched_list.append(())
        all_deriv_args_patched = tuple(all_deriv_args_patched_list)

        sigma_scatter_min = jnp.float64(self.cnc_params["sigma_scatter_min"])
        skyfracs_arr = jnp.array(skyfracs)
        pad_abundance = jnp.bool_(self.cnc_params.get("pad_abundance", False))
        total_skyfrac = jnp.sum(skyfracs_arr)

        (self.abundance_tensor, self.n_obs_matrix, self.n_tot_vec,
         abundance_matrix_out, n_z_vec_out, n_tot_out,
         self.dndz_hmf, self.n_tot_hmf) = self._jit_compute_abundance(
            self.hmf_matrix, self.ln_M, self.obs_select_vec, self.redshift_vec,
            all_layer_args_patched, all_deriv_args_patched,
            all_scatters, all_cutoff_vals, all_apply_cutoffs,
            sigma_scatter_min, skyfracs_arr, pad_abundance, total_skyfrac)

        if self.cnc_params["compute_abundance_matrix"] == True:
            self.abundance_matrix = abundance_matrix_out
            self.n_z = n_z_vec_out
            if self.cnc_params["convolve_nz"] == True:
                self.n_z = convolve_1d(self.redshift_vec, self.n_z,
                                        sigma=self.cnc_params["sigma_nz"], type="fft")

    #Computes the data part of the unbinned likelihood

    def get_log_lik_data(self):

        indices_no_z = self.catalogue.indices_no_z #indices of clusters with no redshift
        indices_obs_select = self.catalogue.indices_obs_select #indices of clusters with redshift and only the selection observable
        indices_other_obs = self.catalogue.indices_other_obs #indices of clusters with redshift, the selection observable, and other observables

        #Computes log lik of data for clusters with no redshift measurement

        log_lik_data = 0.
        obs_key = self.cnc_params["obs_select"]

        if len(indices_no_z) > 0:

            ci_no_z = indices_no_z.astype(int)
            patches_no_z = jnp.asarray(self.catalogue.catalogue_patch[obs_key])[ci_no_z].astype(jnp.int32)
            obs_no_z = jnp.asarray(self.catalogue.catalogue[obs_key])[ci_no_z]

            if self.cnc_params["z_bounds"] == False:

                # Gather n_obs for each cluster's patch: (n_noz, n_points)
                n_obs_per_cluster = self.n_obs_matrix[patches_no_z]

                if self.cnc_params["non_validated_clusters"] == True:
                    validated = jnp.asarray(self.catalogue.catalogue["validated"])[ci_no_z]
                    is_not_validated = validated < 0.5
                    n_obs_fd_per_cluster = self.n_obs_matrix_fd[patches_no_z]
                    f_tv = self.cnc_params["f_true_validated"]
                    n_obs_per_cluster = jnp.where(
                        is_not_validated[:, None],
                        n_obs_per_cluster * (1. - f_tv) + n_obs_fd_per_cluster,
                        n_obs_per_cluster)

                # Vectorized interp: evaluate each cluster's n_obs at its obs_select value
                log_liks_no_z = jax.vmap(
                    lambda obs_val, n_obs_row: jnp.log(jnp.interp(obs_val, self.obs_select_vec, n_obs_row))
                )(obs_no_z, n_obs_per_cluster)
                log_lik_data = log_lik_data + jnp.sum(log_liks_no_z)

            elif self.cnc_params["z_bounds"] == True:

                # Split into bounded and unbounded clusters
                has_z_bounds = jnp.asarray(self.catalogue.catalogue["z_bounds"])[ci_no_z]
                idx_no_bounds = jnp.where(~has_z_bounds, size=len(ci_no_z))[0]
                idx_bounds = jnp.where(has_z_bounds, size=len(ci_no_z))[0]

                # Unbounded: same as z_bounds=False path (vectorized)
                if jnp.sum(~has_z_bounds) > 0:
                    n_obs_nb = self.n_obs_matrix[patches_no_z[idx_no_bounds]]
                    obs_nb = obs_no_z[idx_no_bounds]
                    if self.cnc_params["non_validated_clusters"] == True:
                        validated_nb = jnp.asarray(self.catalogue.catalogue["validated"])[ci_no_z[idx_no_bounds]]
                        is_not_val = validated_nb < 0.5
                        n_obs_fd_nb = self.n_obs_matrix_fd[patches_no_z[idx_no_bounds]]
                        f_tv = self.cnc_params["f_true_validated"]
                        n_obs_nb = jnp.where(is_not_val[:, None],
                                              n_obs_nb * (1. - f_tv) + n_obs_fd_nb, n_obs_nb)
                    log_liks_nb = jax.vmap(
                        lambda obs_val, n_obs_row: jnp.log(jnp.interp(obs_val, self.obs_select_vec, n_obs_row))
                    )(obs_nb, n_obs_nb)
                    log_lik_data = log_lik_data + jnp.sum(log_liks_nb)

                # Bounded: z-bounded integration (loop — different z ranges per cluster)
                for idx in idx_bounds:
                    ci = int(ci_no_z[idx])
                    cp = int(patches_no_z[idx])
                    lower_z = self.catalogue.catalogue["low_z"][ci]
                    upper_z = self.catalogue.catalogue["up_z"][ci]
                    z_bounded = jnp.linspace(lower_z, upper_z, self.cnc_params["n_z"])
                    abund_interp = RegularGridInterpolator(
                        (self.redshift_vec, self.obs_select_vec),
                        self.abundance_tensor[cp, :, :], fill_value=0., bounds_error=False)
                    Z_grid, O_grid = jnp.meshgrid(z_bounded, self.obs_select_vec, indexing='ij')
                    pts = jnp.column_stack([Z_grid.ravel(), O_grid.ravel()])
                    n_obs = simpson(abund_interp(pts).reshape(Z_grid.shape), x=z_bounded, axis=0)
                    if self.cnc_params["non_validated_clusters"] == True:
                        if self.catalogue.catalogue["validated"][ci] < 0.5:
                            n_obs = n_obs + self.n_obs_matrix_fd[cp]
                    obs_val = self.catalogue.catalogue[obs_key][ci]
                    log_lik_data = log_lik_data + jnp.log(jnp.interp(obs_val, self.obs_select_vec, n_obs))

        #Computes log lik of data for clusters with z if there is only the selection observable

        if self.cnc_params["data_lik_from_abundance"] == True:

            if len(indices_obs_select) > 0:

                # Gather all clusters in indices_obs_select (all patches at once)
                z_all = jnp.asarray(self.catalogue.catalogue["z"][indices_obs_select])
                obs_all = jnp.asarray(self.catalogue.catalogue[self.cnc_params["obs_select"]][indices_obs_select])
                z_std_all = jnp.asarray(self.catalogue.catalogue["z_std"][indices_obs_select])
                patch_all = jnp.asarray(
                    self.catalogue.catalogue_patch[self.cnc_params["obs_select"]][indices_obs_select]
                ).astype(jnp.int32)

                # Grid metadata for bilinear interpolation (regular grids)
                z0 = self.redshift_vec[0]
                dz = self.redshift_vec[1] - self.redshift_vec[0]
                nz = self.redshift_vec.shape[0]
                o0 = self.obs_select_vec[0]
                do = self.obs_select_vec[1] - self.obs_select_vec[0]
                no = self.obs_select_vec.shape[0]

                if self.cnc_params["z_errors"] == False:

                    # Vectorized bilinear interp: gathers 4 corners per cluster from 3D tensor
                    log_lik_obs_sel = jnp.sum(jnp.log(jnp.maximum(
                        _bilinear_interp_3d(z_all, obs_all, patch_all,
                                            self.abundance_tensor,
                                            z0, dz, nz, o0, do, no),
                        1e-300)))
                    log_lik_data = log_lik_data + log_lik_obs_sel

                elif self.cnc_params["z_errors"] == True:

                    z_error_min = self.cnc_params["z_error_min"]
                    has_z_err = z_std_all >= z_error_min

                    # Clusters without significant z-errors: simple bilinear interp
                    lik_simple = _bilinear_interp_3d(
                        z_all, obs_all, patch_all, self.abundance_tensor,
                        z0, dz, nz, o0, do, no)

                    # Clusters with z-errors: integrate over z
                    z_min_bound = jnp.float64(self.cnc_params["z_min"])
                    z_max_bound = jnp.float64(self.cnc_params["z_max"])
                    sigma_range = jnp.float64(self.cnc_params["z_error_sigma_integral_range"])
                    n_z_err_int = int(self.cnc_params["n_z_error_integral"])
                    abund_tensor_local = self.abundance_tensor

                    def _z_err_cluster_lik(z_c, obs_c, z_std_c, p_idx):
                        z_lo = jnp.maximum(z_c - sigma_range * z_std_c, z_min_bound)
                        z_hi = jnp.minimum(z_c + sigma_range * z_std_c, z_max_bound)
                        z_eval = jnp.linspace(z_lo, z_hi, n_z_err_int)
                        z_err_lik = gaussian_1d(z_eval - z_c, z_std_c)
                        # Interpolate on this cluster's patch abundance at multiple z points
                        abund_mat = abund_tensor_local[p_idx]  # (n_z, n_points) — single slice
                        abund_z = _bilinear_interp_2d(
                            z_eval, jnp.full(n_z_err_int, obs_c), abund_mat,
                            z0, dz, nz, o0, do, no)
                        return simpson(abund_z * z_err_lik, x=z_eval)

                    lik_z_err = jax.vmap(_z_err_cluster_lik)(
                        z_all, obs_all, z_std_all, patch_all)

                    # Use z-error result where z_std >= threshold, simple otherwise
                    lik_final = jnp.where(has_z_err, lik_z_err, lik_simple)
                    log_lik_data = log_lik_data + jnp.sum(jnp.log(jnp.maximum(lik_final, 1e-300)))

        #Computes log lik of data if there are more observables than the selection observable

        elif self.cnc_params["data_lik_from_abundance"] == False:

            indices_other_obs = np.concatenate((indices_other_obs,indices_obs_select))
            self.indices_bc = indices_other_obs

        if len(indices_other_obs) > 0:

            # ── Cache per-cluster catalogue data (constant across MCMC iterations) ──
            if not hasattr(self, '_bc_cached'):
                idx_bc = np.asarray(indices_other_obs, dtype=int)
                n_bc = len(idx_bc)
                z_clusters = jnp.asarray(self.catalogue.catalogue["z"])[idx_bc]
                obs_data = {}
                for obs_name in self._bc_obs_list:
                    if obs_name in self.catalogue.catalogue:
                        vals = jnp.asarray(self.catalogue.catalogue[obs_name])[idx_bc]
                        has_mask = ~jnp.isnan(vals)
                        obs_data[obs_name] = (jnp.where(has_mask, vals, 0.), has_mask)
                    else:
                        n_bc = len(idx_bc)
                        obs_data[obs_name] = (jnp.zeros(n_bc), jnp.full(n_bc, False))
                obs_select_key = self.cnc_params["obs_select"]
                patch_clusters = jnp.asarray(
                    self.catalogue.catalogue_patch[obs_select_key])[idx_bc].astype(jnp.int32)
                skyfracs_arr = jnp.array(self.scal_rel_selection.skyfracs)
                skyfracs_clusters = skyfracs_arr[patch_clusters]
                # ── Pattern groups: cluster indices per availability pattern per 2D+ set ──
                pattern_groups = {}
                for s_idx, obs_names_s in enumerate(self._bc_set_obs):
                    n_obs_s = len(obs_names_s)
                    if n_obs_s < 2:
                        continue
                    groups = {}
                    for c in range(n_bc):
                        present = tuple(k for k in range(n_obs_s)
                                        if bool(np.asarray(obs_data[obs_names_s[k]][1][c])))
                        groups.setdefault(present, []).append(c)
                    pattern_groups[s_idx] = {
                        pat: np.array(idxs, dtype=int) for pat, idxs in groups.items()
                    }

                # Pre-stack obs arrays (immutable across MCMC steps)
                _all_obs_vals = jnp.stack([obs_data[o][0] for o in self._bc_obs_list])
                _all_has_obs = jnp.stack([obs_data[o][1] for o in self._bc_obs_list])

                # Pre-build 1-layer observable arrays (immutable)
                n_1l = len(self._1layer_obs_list)
                if n_1l > 0:
                    _obs_vals_1l = jnp.stack([
                        jnp.asarray(self.catalogue.catalogue[o])[idx_bc]
                        for o in self._1layer_obs_list])
                    _has_obs_1l = jnp.stack([
                        ~jnp.isnan(jnp.asarray(self.catalogue.catalogue[o])[idx_bc])
                        for o in self._1layer_obs_list])
                else:
                    _obs_vals_1l = jnp.zeros((1, n_bc))
                    _has_obs_1l = jnp.zeros((1, n_bc), dtype=jnp.bool_)

                # Pre-build pattern-loop index mappings:
                # for each (s_idx, pattern), store the obs indices into _bc_obs_list
                _pattern_obs_indices = {}
                for s_idx, obs_names_s in enumerate(self._bc_set_obs):
                    for pattern in pattern_groups.get(s_idx, {}):
                        sub_obs = [obs_names_s[k] for k in pattern]
                        _pattern_obs_indices[(s_idx, pattern)] = [
                            self._bc_obs_list.index(o) for o in sub_obs]

                self._bc_cached = {
                    'idx_bc': idx_bc,
                    'z_clusters': z_clusters,
                    'obs_data': obs_data,
                    'skyfracs_clusters': skyfracs_clusters,
                    'patch_clusters': patch_clusters,
                    'pattern_groups': pattern_groups,
                    'all_obs_vals': _all_obs_vals,
                    'all_has_obs': _all_has_obs,
                    'obs_vals_1l': _obs_vals_1l,
                    'has_obs_1l': _has_obs_1l,
                    'pattern_obs_indices': _pattern_obs_indices,
                }
            else:
                idx_bc = self._bc_cached['idx_bc']
                z_clusters = self._bc_cached['z_clusters']
                obs_data = self._bc_cached['obs_data']
                skyfracs_clusters = self._bc_cached['skyfracs_clusters']
                patch_clusters = self._bc_cached['patch_clusters']
                obs_select_key = self.cnc_params["obs_select"]
                pattern_groups = self._bc_cached['pattern_groups']

            n_bc = len(idx_bc)
            sr_sel = self.scaling_relations[obs_select_key]
            H0 = self.cosmology.background_cosmology.H0.value
            gamma = constants().gamma
            H0_jnp = jnp.float64(H0)
            D_CMB_jnp = jnp.float64(self.cosmology.D_CMB)
            gamma_jnp = jnp.float64(gamma)
            sigma_mass_prior = jnp.float64(self.cnc_params["sigma_mass_prior"])

            # Cutoff (for selection observable)
            apply_cutoff_cfg = self.cnc_params["apply_obs_cutoff"]
            if apply_cutoff_cfg != False and apply_cutoff_cfg.get(str([obs_select_key]), False) == True:
                apply_cutoff = True
                cutoff_val = jnp.float64(self.scal_rel_params.get("q_cutoff",
                    sr_sel.get_cutoff(layer=sr_sel.get_n_layers()-1) if hasattr(sr_sel, 'get_cutoff') else 0.0))
            else:
                apply_cutoff = False
                cutoff_val = jnp.float64(-jnp.inf)

            # Downsampled HMF grid
            hmf_matrix_ds = self.hmf_matrix[:,::self.cnc_params["downsample_hmf_bc"]]
            lnM0 = self.ln_M[::self.cnc_params["downsample_hmf_bc"]]
            lnM0_min = lnM0[0]
            lnM0_max = lnM0[-1]
            n_lnM0 = lnM0.shape[0]

            # ── Stage 1: Cosmo interpolation at cluster redshifts ──
            z_min_grid = self.redshift_vec[0]
            z_max_grid = self.redshift_vec[-1]
            n_z_grid = self.redshift_vec.shape[0]
            D_A_c, E_z_c, D_l_CMB_c, rho_c_c, hmf_z_c = self._interp_cosmo_jit(
                z_clusters, self.D_A, self.E_z, self.D_l_CMB, self.rho_c,
                hmf_matrix_ds, z_min_grid, z_max_grid, n_z_grid)

            # ── Stage 2: Mass range (with prefactors computed inside JIT) ──
            ref_sr_params = self.scal_rel_params

            n_p = self.n_patches
            ref_pref_sr = _tile_to_patches(sr_sel.get_prefactor_sr_params(ref_sr_params), n_p)
            ref_layer0_sr = _tile_to_patches(sr_sel.get_layer_sr_params(0, ref_sr_params), n_p)
            ref_layer1_sr = _tile_to_patches(sr_sel.get_layer_sr_params(1, ref_sr_params), n_p)
            ref_layer0_deriv_sr = _tile_to_patches(sr_sel.get_layer_deriv_sr_params(0, ref_sr_params), n_p)
            ref_layer1_deriv_sr = _tile_to_patches(sr_sel.get_layer_deriv_sr_params(1, ref_sr_params), n_p)
            ref_scatter_sigma = jnp.broadcast_to(
                jnp.float64(sr_sel.get_scatter_sigma(ref_sr_params)),
                (n_p,))
            n_points_dl = int(self.cnc_params["n_points_data_lik"])
            lnM_coarse = jnp.linspace(lnM0_min, lnM0_max, n_points_dl)
            obs_sel_vals = obs_data[obs_select_key][0]

            lnM_min, lnM_max = self._mass_range_with_pref_jit(
                obs_sel_vals,
                E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
                H0_jnp, D_CMB_jnp, gamma_jnp, z_clusters,
                ref_pref_sr,
                ref_layer0_sr, ref_layer1_sr,
                ref_layer0_deriv_sr, ref_layer1_deriv_sr,
                ref_scatter_sigma, sigma_mass_prior,
                lnM0_min, lnM0_max, lnM_coarse,
                patch_clusters)

            # ── Stage 3: All-in-one backward conv + combine + integrate (single JIT) ──
            # Use pre-stacked obs arrays (immutable, cached)
            all_obs_vals = self._bc_cached['all_obs_vals']
            all_has_obs = self._bc_cached['all_has_obs']

            # Per-observable SR params tiled to (n_patches, ...) for patch indexing
            # Reuse mass-range results for the selection observable to avoid redundant SR calls
            _sel_idx = self._bc_obs_list.index(obs_select_key) if obs_select_key in self._bc_obs_list else -1
            all_pref_sr = tuple(
                ref_pref_sr if i == _sel_idx else
                _tile_to_patches(self.scaling_relations[o].get_prefactor_sr_params(self.scal_rel_params), n_p)
                for i, o in enumerate(self._bc_obs_list))
            all_layer0_sr = tuple(
                ref_layer0_sr if i == _sel_idx else
                _tile_to_patches(self.scaling_relations[o].get_layer_sr_params(0, self.scal_rel_params), n_p)
                for i, o in enumerate(self._bc_obs_list))
            all_layer1_sr = tuple(
                ref_layer1_sr if i == _sel_idx else
                _tile_to_patches(self.scaling_relations[o].get_layer_sr_params(1, self.scal_rel_params), n_p)
                for i, o in enumerate(self._bc_obs_list))

            # Per-cluster SR params (vmapped over cluster axis, NOT tiled to patches)
            # These are (n_cat, ...) arrays sliced to bc clusters via idx_bc.
            # Uses identity-based caching: only re-slices arrays whose object identity
            # changed (e.g. beta_avg after update_beta_avg). Static arrays (shear radii,
            # tomo weights, etc.) are sliced once and reused.
            def _get_pc(o, k):
                sr = self.scaling_relations[o]
                if hasattr(sr, 'get_layer_sr_params_per_cluster'):
                    return sr.get_layer_sr_params_per_cluster(k, self.scal_rel_params)
                return ()

            if not hasattr(self, '_pc_slice_cache'):
                self._pc_slice_cache = {}

            def _get_pc_sliced(o, k):
                raw = _get_pc(o, k)
                if len(raw) == 0:
                    return ()
                cache_key = (o, k)
                cached = self._pc_slice_cache.get(cache_key)
                if cached is not None and len(cached[0]) == len(raw) and all(
                        p is cp for p, cp in zip(raw, cached[0])):
                    return cached[1]
                sliced = tuple(jnp.asarray(p)[idx_bc] for p in raw)
                self._pc_slice_cache[cache_key] = (raw, sliced)
                return sliced

            all_layer0_sr_pc = tuple(_get_pc_sliced(o, 0) for o in self._bc_obs_list)
            all_layer1_sr_pc = tuple(_get_pc_sliced(o, 1) for o in self._bc_obs_list)

            # Per-correlation-set: covariance matrices, cutoff config
            # Build cov in numpy then convert once (avoids N² jnp.at[].set() dispatches)
            all_cov_layer0 = []
            all_cov_layer1 = []
            all_apply_cut_sets = []
            all_cut_val_sets = []

            for obs_names in self._bc_set_obs:
                n_obs_s = len(obs_names)
                cov_l0_np = np.array([[
                    self.scatter.get_cov(observable1=obs_names[i], observable2=obs_names[j],
                                         layer=0, patch1=0, patch2=0)
                    for j in range(n_obs_s)] for i in range(n_obs_s)])
                cov_l1_np = np.array([[
                    self.scatter.get_cov(observable1=obs_names[i], observable2=obs_names[j],
                                         layer=1, patch1=0, patch2=0)
                    for j in range(n_obs_s)] for i in range(n_obs_s)])
                all_cov_layer0.append(jnp.asarray(cov_l0_np))
                all_cov_layer1.append(jnp.asarray(cov_l1_np))

                # Cutoff: applied if selection observable is in this set
                set_has_cutoff = obs_select_key in obs_names
                all_apply_cut_sets.append(apply_cutoff and set_has_cutoff)
                all_cut_val_sets.append(
                    cutoff_val if set_has_cutoff else jnp.float64(-jnp.inf))

            all_cov_layer0 = tuple(all_cov_layer0)
            all_cov_layer1 = tuple(all_cov_layer1)
            all_apply_cut_sets = tuple(all_apply_cut_sets)
            all_cut_val_sets = tuple(all_cut_val_sets)

            bc_chunk = int(self.cnc_params.get("bc_chunk_size", 0))
            n_points_dl = int(self.cnc_params["n_points_data_lik"])

            # Shared (non-per-cluster) args for all-in-one
            shared_args = (all_pref_sr, all_layer0_sr, all_layer1_sr,
                           all_cov_layer0, all_cov_layer1,
                           all_apply_cut_sets, all_cut_val_sets,
                           lnM0_min, lnM0_max, n_lnM0)
            # Per-cluster args (vmapped on axis 0)
            pc_args = (all_layer0_sr_pc, all_layer1_sr_pc)

            # 1-layer observable data (obs/has arrays are cached, SR params rebuilt)
            n_1l = len(self._1layer_obs_list)
            obs_vals_1l = self._bc_cached['obs_vals_1l']
            has_obs_1l = self._bc_cached['has_obs_1l']
            if n_1l > 0:
                pref_sr_1l = tuple(
                    _tile_to_patches(self.scaling_relations[o].get_prefactor_sr_params(
                        self.scal_rel_params), n_p)
                    for o in self._1layer_obs_list)
                l0_sr_1l = tuple(
                    _tile_to_patches(self.scaling_relations[o].get_layer_sr_params(
                        0, self.scal_rel_params), n_p)
                    for o in self._1layer_obs_list)
                l0_pc_1l = tuple(_get_pc_sliced(o, 0) for o in self._1layer_obs_list)
                cov_1l = tuple(
                    jnp.float64(self.scatter.get_cov(
                        observable1=o, observable2=o, layer=0, patch1=0, patch2=0))
                    for o in self._1layer_obs_list)
            else:
                pref_sr_1l = ()
                l0_sr_1l = ()
                l0_pc_1l = ()
                cov_1l = ()

            # ── Pattern-aware split-JIT: compute cpdfs for 2D+ sets ──
            # For each 2D+ correlation set, groups clusters by which observables
            # are present and dispatches the appropriate sub-dimensional JIT.
            # 1D sets get dummy arrays (computed inline by the all-in-one JIT).

            def _dispatch_2d_split(fwd_jit, conv_jit, mn, mx, set_obs,
                                   ez, da, dl, rc, zc,
                                   sub_pref_sr, sub_layer0_sr, sub_layer1_sr,
                                   sub_l0_pc, sub_l1_pc,
                                   sub_cov_l0, sub_cov_l1,
                                   sub_apply_cut, sub_cut_val,
                                   patch_sub):
                """Run 2D split-JIT (forward + conv) for a cluster group."""
                (r0, r1, kc0, kc1, x_l0_0, x_l0_1,
                 x_lin_0_start, x_lin_1_start,
                 dx0_arr, dx1_arr) = fwd_jit(
                    mn, mx, set_obs, ez, da, dl, rc,
                    H0_jnp, D_CMB_jnp, gamma_jnp, zc,
                    sub_pref_sr, sub_layer0_sr, sub_layer1_sr,
                    sub_l0_pc, sub_l1_pc,
                    patch_sub)

                det1 = sub_cov_l1[0, 0] * sub_cov_l1[1, 1] - sub_cov_l1[0, 1]**2
                inv_cov1 = jnp.array([[sub_cov_l1[1, 1], -sub_cov_l1[0, 1]],
                                       [-sub_cov_l1[0, 1], sub_cov_l1[0, 0]]]) / det1
                norm1 = 1.0 / jnp.sqrt((2. * jnp.pi)**2 * det1)

                det0 = sub_cov_l0[0, 0] * sub_cov_l0[1, 1] - sub_cov_l0[0, 1]**2
                inv_cov0 = jnp.array([[sub_cov_l0[1, 1], -sub_cov_l0[0, 1]],
                                       [-sub_cov_l0[0, 1], sub_cov_l0[0, 0]]]) / det0
                norm0 = 1.0 / jnp.sqrt((2. * jnp.pi)**2 * det0)
                has_scatter = ~jnp.all(sub_cov_l0 == 0.)

                obs_val_0 = set_obs[:, 0]
                return conv_jit(
                    r0, r1, kc0, kc1, x_l0_0, x_l0_1,
                    x_lin_0_start, x_lin_1_start, dx0_arr, dx1_arr,
                    obs_val_0, inv_cov1, norm1, inv_cov0, norm0,
                    has_scatter, sub_apply_cut, sub_cut_val)

            def _compute_all_set_cpdfs(sl=None):
                """Compute pre_nd_cpdfs for all sets with pattern-aware dispatch.

                For each 2D+ correlation set, groups clusters by observable
                availability pattern and dispatches the appropriate sub-JIT
                (full 2D, 1D marginal, etc.).  1D sets get dummy arrays.
                """
                pre_nd_list = []
                n_cl = n_bc if sl is None else sl.stop - sl.start

                for s_idx in range(len(self._bc_set_obs)):
                    obs_names_s = self._bc_set_obs[s_idx]
                    n_obs_s = len(obs_names_s)

                    if n_obs_s < 2:
                        # 1D set: dummy (computed inline by all-in-one JIT)
                        pre_nd_list.append(jnp.zeros((n_cl, n_points_dl)))
                        continue

                    # 2D+ set: pattern-aware dispatch
                    cpdf_set = jnp.zeros((n_cl, n_points_dl))
                    idx_flat = self._2d_core_obs_indices[s_idx]

                    for pattern, ci_global in pattern_groups.get(s_idx, {}).items():
                        n_sub = len(pattern)
                        if n_sub == 0:
                            continue

                        # Intersect with chunk slice
                        if sl is not None:
                            mask = (ci_global >= sl.start) & (ci_global < sl.stop)
                            ci_g = ci_global[mask]
                            if len(ci_g) == 0:
                                continue
                            ci_l = ci_g - sl.start
                        else:
                            ci_g = ci_global
                            ci_l = ci_global
                        ci_g_jnp = jnp.array(ci_g)
                        ci_l_jnp = jnp.array(ci_l)

                        # Gather per-cluster quantities
                        mn = lnM_min[ci_g_jnp]
                        mx = lnM_max[ci_g_jnp]
                        ez = E_z_c[ci_g_jnp]
                        da = D_A_c[ci_g_jnp]
                        dl = D_l_CMB_c[ci_g_jnp]
                        rc = rho_c_c[ci_g_jnp]

                        # Use cached obs-index mapping instead of rebuilding SR params
                        obs_idxs = self._bc_cached['pattern_obs_indices'][(s_idx, pattern)]
                        sub_obs = [self._bc_obs_list[i] for i in obs_idxs]

                        # Index into already-computed SR params (no redundant SR calls)
                        sub_pref_sr = tuple(all_pref_sr[i] for i in obs_idxs)
                        sub_layer0_sr = tuple(all_layer0_sr[i] for i in obs_idxs)
                        sub_layer1_sr = tuple(all_layer1_sr[i] for i in obs_idxs)

                        # Per-cluster patch indices for this group
                        patch_sub = patch_clusters[ci_g_jnp]

                        # Covariance submatrix
                        si = jnp.array(list(pattern))
                        sub_cov_l0 = all_cov_layer0[s_idx][si][:, si]
                        sub_cov_l1 = all_cov_layer1[s_idx][si][:, si]

                        # Cutoff: apply only if selection obs is first in sub-pattern
                        sel_at_0 = (len(sub_obs) > 0
                                    and sub_obs[0] == obs_select_key)
                        sub_apply_cut = (apply_cutoff and sel_at_0)
                        sub_cut_val = (cutoff_val if sel_at_0
                                       else jnp.float64(-jnp.inf))

                        jit_info = self._sub_bc_jits.get((s_idx, pattern))
                        if jit_info is None:
                            continue

                        if jit_info['type'] == '2d_split':
                            set_obs = jnp.stack(
                                [obs_data[sub_obs[j]][0][ci_g_jnp]
                                 for j in range(2)], axis=1)
                            zc = z_clusters[ci_g_jnp]
                            # Per-cluster data: index into already-computed arrays
                            sub_l0_pc = tuple(
                                tuple(p[ci_g_jnp] for p in all_layer0_sr_pc[i])
                                for i in obs_idxs)
                            sub_l1_pc = tuple(
                                tuple(p[ci_g_jnp] for p in all_layer1_sr_pc[i])
                                for i in obs_idxs)
                            cpdf_sub = _dispatch_2d_split(
                                jit_info['fwd_jit'], jit_info['conv_jit'],
                                mn, mx, set_obs, ez, da, dl, rc, zc,
                                sub_pref_sr, sub_layer0_sr, sub_layer1_sr,
                                sub_l0_pc, sub_l1_pc,
                                sub_cov_l0, sub_cov_l1,
                                sub_apply_cut, sub_cut_val,
                                patch_sub)
                        elif jit_info['type'] == 'generic':
                            set_obs = jnp.stack(
                                [obs_data[sub_obs[j]][0][ci_g_jnp]
                                 for j in range(n_sub)], axis=1)
                            zc = z_clusters[ci_g_jnp]
                            cpdf_sub = jit_info['jit_fn'](
                                mn, mx, set_obs, ez, da, dl, rc,
                                H0_jnp, D_CMB_jnp, gamma_jnp, zc,
                                sub_pref_sr, sub_layer0_sr, sub_layer1_sr,
                                sub_cov_l0, sub_cov_l1,
                                sub_apply_cut, sub_cut_val,
                                patch_sub)
                        else:
                            continue

                        cpdf_set = cpdf_set.at[ci_l_jnp].set(cpdf_sub)

                    pre_nd_list.append(cpdf_set)

                return tuple(pre_nd_list)

            if bc_chunk <= 0 or n_bc <= bc_chunk:
                # Full vmap: all clusters at once
                pre_nd_cpdfs = _compute_all_set_cpdfs()
                log_liks, cpdf_with_hmf, lnM_grid = self._allinone_bc_jit(
                    lnM_min, lnM_max, all_obs_vals, all_has_obs,
                    hmf_z_c, skyfracs_clusters,
                    E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
                    H0_jnp, D_CMB_jnp, gamma_jnp, z_clusters,
                    shared_args[0], shared_args[1], shared_args[2],  # pref_sr, l0_sr, l1_sr
                    *pc_args,  # per-cluster L0, L1
                    *shared_args[3:],  # cov, cutoff, lnM bounds
                    patch_clusters,
                    obs_vals_1l, has_obs_1l,
                    pref_sr_1l, l0_sr_1l, l0_pc_1l, cov_1l,
                    pre_nd_cpdfs)
            else:
                # Chunked: process bc_chunk clusters at a time
                log_liks_list = []
                cpdf_list = []
                lnM_list = []
                for c_start in range(0, n_bc, bc_chunk):
                    c_end = min(c_start + bc_chunk, n_bc)
                    sl = slice(c_start, c_end)
                    pre_nd_chunk = _compute_all_set_cpdfs(sl)
                    # Slice per-cluster args
                    pc_args_sl = tuple(
                        tuple(p[sl] for p in pc_tuple) for pc_tuple in pc_args)
                    # Slice 1-layer per-cluster data (always slice, even dummy arrays)
                    obs_1l_sl = obs_vals_1l[:, sl]
                    has_1l_sl = has_obs_1l[:, sl]
                    l0_pc_1l_sl = tuple(
                        tuple(p[sl] for p in pc) for pc in l0_pc_1l)
                    ll, cw, lm = self._allinone_bc_jit(
                        lnM_min[sl], lnM_max[sl],
                        all_obs_vals[:, sl], all_has_obs[:, sl],
                        hmf_z_c[sl], skyfracs_clusters[sl],
                        E_z_c[sl], D_A_c[sl], D_l_CMB_c[sl], rho_c_c[sl],
                        H0_jnp, D_CMB_jnp, gamma_jnp, z_clusters[sl],
                        shared_args[0], shared_args[1], shared_args[2],
                        *pc_args_sl,
                        *shared_args[3:],
                        patch_clusters[sl],
                        obs_1l_sl, has_1l_sl,
                        pref_sr_1l, l0_sr_1l, l0_pc_1l_sl, cov_1l,
                        pre_nd_chunk)
                    log_liks_list.append(ll)
                    cpdf_list.append(cw)
                    lnM_list.append(lm)
                log_liks = jnp.concatenate(log_liks_list)
                cpdf_with_hmf = jnp.concatenate(cpdf_list)
                lnM_grid = jnp.concatenate(lnM_list)

            log_lik_data_rank = jnp.sum(log_liks)

            # Store cpdf and lnM arrays for stacked likelihood
            self.bc_cpdf_array = cpdf_with_hmf
            self.bc_lnM_array = lnM_grid
            self.bc_cluster_indices = idx_bc
            self.bc_z_clusters = z_clusters

            # Mass estimates if requested
            if self.cnc_params["get_masses"] == True:
                norms = jax.vmap(lambda c, m: simpson(c, x=m))(cpdf_with_hmf, lnM_grid)
                lnM_means = jax.vmap(lambda c, m, n: simpson(m * c, x=m) / n)(cpdf_with_hmf, lnM_grid, norms)
                lnM_stds = jax.vmap(lambda c, m, n, mu: jnp.sqrt(simpson(m**2 * c, x=m) / n - mu**2))(
                    cpdf_with_hmf, lnM_grid, norms, lnM_means)
                self.bc_lnM_means = lnM_means
                self.bc_lnM_stds = lnM_stds

            log_lik_data = log_lik_data + log_lik_data_rank

        return log_lik_data

    #Computes the stacked likelihood. Must be called after the unbinned likelihood has been computed.

    def get_log_lik_stacked(self):

        log_lik = 0.

        H0 = self.cosmology.background_cosmology.H0.value
        gamma_const = constants().gamma
        compute_stacked_cov = bool(self.cnc_params["compute_stacked_cov"])

        self.stacked_model = {}
        self.stacked_variance = {}

        sigma_scatter_min = jnp.float64(self.cnc_params["sigma_scatter_min"])
        n_points_stacked = int(self.cnc_params["n_points"])

        # Build index map once (Python, outside JIT) using numpy to avoid GPU syncs
        bc_indices_np = np.asarray(self.bc_cluster_indices)
        bc_idx_map = {int(bc_indices_np[pos]): pos for pos in range(len(bc_indices_np))}

        for stacked_data_label in self.stacked_data_labels:

            stacked_observable = self.catalogue.stacked_data[stacked_data_label]["observable"]
            stacked_cluster_indices = self.catalogue.stacked_data[stacked_data_label]["cluster_index"]
            # Cache stacked data as JAX arrays (constant across iterations)
            _cache_key = "_stacked_jax_" + stacked_data_label
            if not hasattr(self, _cache_key):
                setattr(self, _cache_key, {
                    "data_vec": jnp.asarray(self.catalogue.stacked_data[stacked_data_label]["data_vec"]),
                    "inv_cov": jnp.asarray(self.catalogue.stacked_data[stacked_data_label]["inv_cov"]),
                })
            stacked_obs_vec = getattr(self, _cache_key)["data_vec"]
            stacked_inv_cov = getattr(self, _cache_key)["inv_cov"]
            n_clusters = len(stacked_cluster_indices)

            # Determine n_layers_stacked from scaling relation (static per observable)
            stacked_sr = self.scaling_relations.get(stacked_observable)
            if stacked_sr is not None and hasattr(stacked_sr, 'get_n_layers_stacked'):
                n_layers_stacked = int(stacked_sr.get_n_layers_stacked())
            else:
                n_layers_stacked = 1

            # Map stacked cluster indices to bc array positions (Python)
            st_indices_np = np.asarray(stacked_cluster_indices)
            st_positions = jnp.array([bc_idx_map[int(st_indices_np[k])] for k in range(len(st_indices_np))])

            # Gather cpdf/lnM/z for stacked clusters
            cpdf_st = self.bc_cpdf_array[st_positions]
            lnM_st = self.bc_lnM_array[st_positions]
            z_st = self.bc_z_clusters[st_positions]

            # ── Generic stacked likelihood using factory-built kernel ──
            stacked_kernel = self._stacked_kernels.get(stacked_observable)
            if stacked_kernel is None:
                self.logger.warning(f"No stacked kernel for {stacked_observable}, skipping")
                continue

            # Interpolate cosmology at stacked cluster redshifts
            cosmo_at_st = {
                "E_z": jax.vmap(lambda z: jnp.interp(z, self.redshift_vec, self.E_z))(z_st),
                "H0": jnp.float64(H0),
                "D_A": jax.vmap(lambda z: jnp.interp(z, self.redshift_vec, self.D_A))(z_st),
                "D_CMB": jnp.float64(self.cosmology.D_CMB),
                "D_l_CMB": jax.vmap(lambda z: jnp.interp(z, self.redshift_vec, self.D_l_CMB))(z_st),
                "rho_c": jax.vmap(lambda z: jnp.interp(z, self.redshift_vec, self.rho_c))(z_st),
                "gamma": jnp.float64(gamma_const),
            }

            # Compute prefactors for stacked observable
            pref_fn = stacked_sr.get_prefactor_fn()
            pref_vmap_axes = stacked_sr.get_prefactor_vmap_axes()
            pref_args = stacked_sr.get_prefactor_args(cosmo_at_st, self.scal_rel_params)
            pref_vmap = jax.vmap(pref_fn, in_axes=pref_vmap_axes)
            pref_st = pref_vmap(*pref_args)

            # Layer 0 SR params and scatter — tiled to (n_patches, ...) for patch indexing
            n_p = self.n_patches
            layer0_sr_patched = _tile_to_patches(
                stacked_sr.get_layer_sr_params(0, self.scal_rel_params), n_p)
            scatter_sigma_patched = jnp.broadcast_to(
                jnp.float64(stacked_sr.get_scatter_sigma(self.scal_rel_params)),
                (n_p,))
            mean_fn_sr_patched = _tile_to_patches(
                stacked_sr.get_mean_fn_sr_params(self.scal_rel_params), n_p)

            # Per-cluster patch indices for stacked observable
            if stacked_observable in self.catalogue.catalogue_patch:
                patch_st = jnp.asarray(
                    self.catalogue.catalogue_patch[stacked_observable]
                )[jnp.asarray(stacked_cluster_indices)].astype(jnp.int32)
            else:
                patch_st = jnp.zeros(n_clusters, dtype=jnp.int32)

            # Per-cluster stacked kernel (vmapped), with patch indexing
            def _stacked_one(cpdf_wh, lnM, patch_idx_c, *pref_vals):
                l0_c = tuple(p[patch_idx_c] for p in layer0_sr_patched)
                mfn_c = tuple(p[patch_idx_c] for p in mean_fn_sr_patched)
                scat_c = scatter_sigma_patched[patch_idx_c]
                layer0_args = pref_vals + l0_c
                mean_pref_args = pref_vals
                return stacked_kernel(cpdf_wh, lnM,
                                       layer0_args, mean_pref_args, mfn_c,
                                       scat_c, sigma_scatter_min,
                                       n_points_stacked, compute_stacked_cov,
                                       n_layers_stacked)

            if isinstance(pref_st, tuple):
                obs_means, obs_vars = jax.jit(jax.vmap(_stacked_one))(
                    cpdf_st, lnM_st, patch_st, *pref_st)
            else:
                obs_means, obs_vars = jax.jit(jax.vmap(_stacked_one))(
                    cpdf_st, lnM_st, patch_st, pref_st)

            # Aggregate
            stacked_model_vec = jnp.sum(obs_means, axis=0) / n_clusters
            stacked_var_vec = jnp.sum(obs_vars, axis=0) / n_clusters**2

            if compute_stacked_cov:
                if stacked_var_vec.ndim == 0 or stacked_var_vec.size == 1:
                    stacked_inv_cov = jnp.array([1. / stacked_var_vec.ravel()[0]])
                else:
                    stacked_inv_cov = jnp.linalg.inv(jnp.diag(stacked_var_vec))

            res = stacked_obs_vec - stacked_model_vec

            self.stacked_model[stacked_data_label] = stacked_model_vec
            self.stacked_variance[stacked_data_label] = stacked_var_vec

            log_lik = log_lik - 0.5 * jnp.dot(res, jnp.dot(stacked_inv_cov, res))

        return log_lik

    #Retrieve cluster mean log masses

    def get_masses(self):

        # Use vectorized mass estimates from backward conv (no loop needed)
        if hasattr(self, 'bc_lnM_means'):
            self.cluster_lnM = self.bc_lnM_means
            self.cluster_lnM_std = self.bc_lnM_stds
        else:
            # Compute from cpdf arrays if not already done
            norms = jax.vmap(lambda c, m: simpson(c, x=m))(self.bc_cpdf_array, self.bc_lnM_array)
            self.cluster_lnM = jax.vmap(lambda c, m, n: simpson(m * c, x=m) / n)(
                self.bc_cpdf_array, self.bc_lnM_array, norms)
            self.cluster_lnM_std = jax.vmap(
                lambda c, m, n, mu: jnp.sqrt(simpson(m**2 * c, x=m) / n - mu**2)
            )(self.bc_cpdf_array, self.bc_lnM_array, norms, self.cluster_lnM)

    def get_number_counts_false_detections(self):

        f_false_detection = self.scal_rel_params["f_false_detection"]

        [obs_select_fd,pdf_fd] = self.scaling_relations[self.cnc_params["obs_select"]].pdf_false_detection
        fd_interp = jnp.interp(self.obs_select_vec, jnp.asarray(obs_select_fd), jnp.asarray(pdf_fd))

        # Vectorized over patches: (n_patches,1) * (n_points,) -> (n_patches, n_points)
        self.n_obs_matrix_fd = fd_interp[None, :] * self.n_tot_vec[:, None] * f_false_detection / (1. - f_false_detection)

        self.n_tot_vec_fd = self.n_tot_vec * f_false_detection / (1. - f_false_detection)

    def get_number_counts(self):

        if self.abundance_tensor is None:

            self.get_cluster_abundance()

        self.n_obs = jnp.sum(self.n_obs_matrix,axis=0)
        self.n_tot = jnp.sum(self.n_tot_vec)

        self.logger.info("Total clusters: %.5f",self.n_tot)

        if self.cnc_params["non_validated_clusters"] == True:

            self.get_number_counts_false_detections()

            self.n_obs_fd = jnp.sum(self.n_obs_matrix_fd,axis=0)
            self.n_tot_fd = jnp.sum(self.n_tot_vec_fd,axis=0)

    def get_log_lik_extreme_value(self,obs_max=None):

        if self.n_tot is None:

            self.get_number_counts()

        n_obs = self.n_obs

        if obs_max is None:

            obs_max = self.catalogue.obs_select_max

        if self.cnc_params["non_validated_clusters"] == True:

            n_obs = n_obs + self.n_obs_fd

        obs_select_vec_interp = jnp.linspace(obs_max,self.cnc_params["obs_select_max"],100)
        n_interp = jnp.interp(obs_select_vec_interp,self.obs_select_vec,n_obs)
        n_theory = simpson(n_interp,x=obs_select_vec_interp)

        log_lik = -n_theory

        return log_lik

    def eval_extreme_value_quantities(self):

        n_obs = self.n_obs
        if self.cnc_params["non_validated_clusters"] == True:
            n_obs = n_obs + self.n_obs_fd

        obs_select_max_limit = self.cnc_params["obs_select_max"]
        obs_select_vec = self.obs_select_vec

        def _ev_at_obs(obs_max):
            obs_interp = jnp.linspace(obs_max, obs_select_max_limit, 100)
            n_interp = jnp.interp(obs_interp, obs_select_vec, n_obs)
            return -simpson(n_interp, x=obs_interp)

        self.log_lik_ev_eval = jax.vmap(_ev_at_obs)(self.obs_select_vec)

        self.lik_ev_eval = jnp.exp(self.log_lik_ev_eval)

        self.obs_select_max_pdf = jnp.gradient(self.lik_ev_eval,self.obs_select_vec)
        self.obs_select_max_mean = simpson(self.obs_select_max_pdf*self.obs_select_vec,x=self.obs_select_vec)
        self.obs_select_max_std = jnp.sqrt(simpson(self.obs_select_max_pdf*(self.obs_select_vec-self.obs_select_max_mean)**2,x=self.obs_select_vec))

    def get_log_lik(self):

        t0 = time.time()

        log_lik = 0.

        if self.cnc_params["priors"] == True:

            log_lik = log_lik + self.priors.eval_priors(self.cosmo_params,self.scal_rel_params)

        if self.cnc_params["likelihood_type"] == "unbinned":

            log_lik = log_lik + self.get_log_lik_unbinned()

        elif self.cnc_params["likelihood_type"] == "binned":

            log_lik = log_lik + self.get_log_lik_binned()

        elif self.cnc_params["likelihood_type"] == "extreme_value":

            log_lik = log_lik + self.get_log_lik_extreme_value()

        self.t_total = time.time()-t0

        self.logger.info("Time: %.5f",self.t_total)

        self.logger.info("log_lik: %.5f",log_lik)

        log_lik = jnp.where(jnp.isnan(log_lik), -jnp.inf, log_lik)

        self.log_lik = log_lik

        return log_lik

    #Computes the unbinned log likelihood

    def get_log_lik_unbinned(self):

        t0 = time.time()

        if self.hmf_matrix is None:

            self.get_hmf()

        t1 = time.time()

        self.time_hmf = t1-t0

        if self.n_tot is None:

            self.get_number_counts()

        #Abundance term

        n_tot = self.n_tot

        if self.cnc_params["non_validated_clusters"] == True:

            n_tot = n_tot + self.n_tot_fd

        log_lik = -n_tot

        if self.cnc_params["non_validated_clusters"] == True:

            if self.catalogue.n_val > 0.5:

                log_lik = log_lik + self.catalogue.n_val*jnp.log(self.cnc_params["f_true_validated"])

        t2 = time.time()

        self.t_abundance = t2-t1

        #Cluster data term

        log_lik = log_lik + self.get_log_lik_data()

        self.t_data = time.time()-t2

        #Stacked_term

        if self.cnc_params["stacked_likelihood"] == True:
            log_lik = log_lik + self.get_log_lik_stacked()

        return log_lik

    def get_abundance_matrix(self):

        self.abundance_matrix = jnp.sum(self.abundance_tensor,axis=0)
        self.n_z = simpson(self.abundance_matrix,x=self.obs_select_vec)

        if self.cnc_params["convolve_nz"] == True:

            self.n_z = convolve_1d(self.redshift_vec,self.n_z,sigma=self.cnc_params["sigma_nz"],type="fft")

    #Computes the binned log likelihood

    def get_log_lik_binned(self):

        if self.n_tot is None:

            self.get_number_counts()

        log_lik = 0.

        if self.cnc_params["binned_lik_type"] == "z_and_obs_select":

            if self.abundance_matrix is None:

                self.get_abundance_matrix()

            bins_edges_z = jnp.asarray(self.cnc_params["bins_edges_z"])
            bins_edges_obs = jnp.asarray(self.cnc_params["bins_edges_obs_select"])
            n_z_bins = len(bins_edges_z) - 1
            n_obs_bins = len(bins_edges_obs) - 1

            self.bins_centres_z = (bins_edges_z[1:] + bins_edges_z[:-1]) * 0.5
            self.bins_centres_obs = (bins_edges_obs[1:] + bins_edges_obs[:-1]) * 0.5

            n_bins_redshift = int(len(self.redshift_vec) / max(n_z_bins - 1, 1))
            n_bins_obs_select = int(len(self.obs_select_vec) / max(n_obs_bins - 1, 1))

            # Build JAX RegularGridInterpolator once
            abundance_interp = RegularGridInterpolator(
                (self.redshift_vec, self.obs_select_vec),
                self.abundance_matrix, fill_value=0., bounds_error=False)

            # Prepare observed counts
            if self.cnc_params["load_catalogue"] is True:
                number_counts_obs = jnp.asarray(self.catalogue.number_counts)
            else:
                number_counts_obs = jnp.zeros((n_z_bins, n_obs_bins))

            # Vectorized bin integral: compute all bins at once
            def _bin_integral_2d(z_lo, z_hi, obs_lo, obs_hi):
                z_interp = jnp.linspace(z_lo, z_hi, n_bins_redshift)
                obs_interp = jnp.linspace(obs_lo, obs_hi, n_bins_obs_select)
                X, Y = jnp.meshgrid(z_interp, obs_interp)
                pts = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
                grid_vals = abundance_interp(pts).reshape(X.shape)
                return simpson(simpson(grid_vals, x=z_interp, axis=1), x=obs_interp)

            # Build all bin edge pairs
            Z_lo, O_lo = jnp.meshgrid(bins_edges_z[:-1], bins_edges_obs[:-1], indexing='ij')
            Z_hi, O_hi = jnp.meshgrid(bins_edges_z[1:], bins_edges_obs[1:], indexing='ij')

            self.n_binned = jax.vmap(_bin_integral_2d)(
                Z_lo.ravel(), Z_hi.ravel(), O_lo.ravel(), O_hi.ravel()
            ).reshape(n_z_bins, n_obs_bins)
            self.n_binned_obs = number_counts_obs

            log_lik = jnp.sum(-self.n_binned + self.n_binned_obs * jnp.log(self.n_binned))

        elif self.cnc_params["binned_lik_type"] == "obs_select":

            bins_edges_obs = jnp.asarray(self.cnc_params["bins_edges_obs_select"])
            n_obs_bins = len(bins_edges_obs) - 1
            self.bins_centres = (bins_edges_obs[1:] + bins_edges_obs[:-1]) * 0.5
            n_bins_obs_select = int(len(self.obs_select_vec) / max(n_obs_bins - 1, 1))

            n_obs_local = self.n_obs
            if self.cnc_params["non_validated_clusters"] == True:
                n_obs_local = n_obs_local + self.n_obs_fd

            # Vectorized bin integral
            def _bin_integral_obs(obs_lo, obs_hi):
                obs_interp = jnp.linspace(obs_lo, obs_hi, n_bins_obs_select)
                n_interp = jnp.interp(obs_interp, self.obs_select_vec, n_obs_local)
                return simpson(n_interp, x=obs_interp)

            self.n_binned = jax.vmap(_bin_integral_obs)(bins_edges_obs[:-1], bins_edges_obs[1:])

            if self.cnc_params["load_catalogue"] is True:
                self.n_binned_obs = jnp.asarray(self.catalogue.number_counts)
            else:
                self.n_binned_obs = jnp.zeros(n_obs_bins)

            log_lik = jnp.sum(-self.n_binned + self.n_binned_obs * jnp.log(self.n_binned))

        elif self.cnc_params["binned_lik_type"] == "z":

            if self.abundance_matrix is None:

                self.get_abundance_matrix()

            bins_edges_z = jnp.asarray(self.cnc_params["bins_edges_z"])
            n_z_bins = len(bins_edges_z) - 1
            self.bins_centres = (bins_edges_z[1:] + bins_edges_z[:-1]) * 0.5
            n_bins_redshift = int(len(self.redshift_vec) / max(n_z_bins - 1, 1))

            # Vectorized bin integral
            def _bin_integral_z(z_lo, z_hi):
                z_interp = jnp.linspace(z_lo, z_hi, n_bins_redshift)
                n_interp = jnp.interp(z_interp, self.redshift_vec, self.n_z)
                return simpson(n_interp, x=z_interp)

            self.n_binned = jax.vmap(_bin_integral_z)(bins_edges_z[:-1], bins_edges_z[1:])

            if self.cnc_params["load_catalogue"] is True:
                self.n_binned_obs = jnp.asarray(self.catalogue.number_counts)
            else:
                self.n_binned_obs = jnp.zeros(n_z_bins)

            log_lik = jnp.sum(-self.n_binned + self.n_binned_obs * jnp.log(self.n_binned))

        return log_lik

    def get_c_statistic(self):

        if self.n_binned is None:

            self.get_log_lik_binned()

        n_binned_mean = self.n_binned.flatten()
        n_binned_obs = self.n_binned_obs.flatten()

        self.C,self.C_mean,self.C_std = get_cash_statistic(n_binned_obs,n_binned_mean)

        return (self.C,self.C_mean,self.C_std)

    #Calculates derivative of log likelihood with respect to parameter param on param_vec

    def get_log_lik_derivative(self,param,param_vec=None,param_type=None):

        log_lik_vec = jnp.zeros(len(param_vec))

        if param_type == "cosmo":

            param_0 = self.cosmo_params[param]

        elif param_type == "scal_rel":

            param_0 = self.scal_rel_params[param]

        for i in range(0,len(param_vec)):

            if param_type == "cosmo":

                self.cosmo_params[param] = param_vec[i]

            elif param_type == "scal_rel":

                self.scal_rel_params[param] = param_vec[i]

            self.update_params(self.cosmo_params,self.scal_rel_params)
            log_lik_vec = log_lik_vec.at[i].set(self.get_log_lik())

        log_lik_derivative_vec = jnp.gradient(log_lik_vec,param_vec)
        log_lik_derivative = log_lik_derivative_vec[(len(param_vec)-1)//2]

        if param_type == "cosmo":

            self.cosmo_params[param] = param_0

        elif param_type == "scal_rel":

            self.scal_rel_params[param] = param_0

        self.update_params(self.cosmo_params,self.scal_rel_params)

        return log_lik_derivative
