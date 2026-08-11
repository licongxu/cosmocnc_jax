"""DMB (BCM_18_wP) electron pressure and central Compton-y0 for CNC.

Ports the GODMAX / hmfast ``dmb_halo_quantities`` HSE stack into cosmocnc_jax.
Central Compton-y is the LOS integral through the cluster centre:

    y0 = 2 (σ_T / m_e c²) ∫_0^∞ Pe(r) dr

This is the same observable the GNFW power-law SR approximates under
self-similarity; DMB evaluates it from Pe directly.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

# G in (Msun/h, Mpc/h) → keV/cm^3 (GODMAX / astropy).
_G_KEV = 1.8167909997572036e-30
_PE_KEV_TO_EV = 1000.0
_PTH_TO_PE = 1.932
_RHO_CRIT_0_H = 2.77536627245708e11  # Msun/h / (Mpc/h)^3

_SIGMA_T_CM2 = 6.6524587321e-25
_ME_EV = 510_998.95
_MPC_TO_CM = 3.0856775814913673e24

# Flat param keys passed through CNC layer factories (order fixed for JIT).
DMB_PARAM_KEYS = (
    "theta_ej_0",
    "log10_Mstar0_theta_ej",
    "nu_theta_ej_M",
    "nu_theta_ej_z",
    "nu_theta_ej_c",
    "theta_co_0",
    "log10_Mstar0_theta_co",
    "nu_theta_co_M",
    "nu_theta_co_z",
    "nu_theta_co_c",
    "mu_beta",
    "eta_star",
    "eta_cga",
    "A_starcga",
    "log10_M1_starcga",
    "epsilon_rt",
    "log10_Mc0",
    "nu_z",
    "nu_M",
    "log10_Mstar0",
    "a_zeta",
    "n_zeta",
    "alpha_nt",
    "beta_nt",
    "n_nt",
    "gamma_rhogas",
    "delta_rhogas",
)

DMB_DEFAULTS = dict(
    theta_ej_0=4.0,
    log10_Mstar0_theta_ej=14.0,
    nu_theta_ej_M=0.0,
    nu_theta_ej_z=0.0,
    nu_theta_ej_c=0.0,
    theta_co_0=0.1,
    log10_Mstar0_theta_co=14.0,
    nu_theta_co_M=0.0,
    nu_theta_co_z=0.0,
    nu_theta_co_c=0.0,
    mu_beta=0.21,
    eta_star=0.3,
    eta_cga=0.6,
    A_starcga=0.09,
    log10_M1_starcga=11.4,
    epsilon_rt=4.0,
    log10_Mc0=14.83,
    nu_z=0.0,
    nu_M=0.0,
    log10_Mstar0=13.0,
    a_zeta=0.3,
    n_zeta=2.0,
    alpha_nt=0.18,
    beta_nt=0.5,
    n_nt=0.3,
    gamma_rhogas=2.0,
    delta_rhogas=7.0,
    nfw_trunc=True,
    num_points_trapz_int=64,
    # If None / NaN: Duffy 2008 c200c(M,z). Else fixed concentration.
    dmb_c200c=jnp.nan,
)


def dmb_params_from_sr(sr_params):
    """Build a DMB params dict from scaling_relation_params (with defaults)."""
    out = dict(DMB_DEFAULTS)
    for k in DMB_PARAM_KEYS:
        if k in sr_params:
            out[k] = sr_params[k]
    for k in ("nfw_trunc", "num_points_trapz_int", "dmb_c200c"):
        if k in sr_params:
            out[k] = sr_params[k]
    return out


def pack_dmb_params(sr_params):
    """Tuple of float DMB params in ``DMB_PARAM_KEYS`` order for JIT layers."""
    p = dmb_params_from_sr(sr_params)
    return tuple(jnp.float64(p[k]) for k in DMB_PARAM_KEYS)


def unpack_dmb_params(packed, nfw_trunc=True, num_points_trapz_int=64, dmb_c200c=jnp.nan):
    p = {k: packed[i] for i, k in enumerate(DMB_PARAM_KEYS)}
    p["nfw_trunc"] = nfw_trunc
    p["num_points_trapz_int"] = int(num_points_trapz_int)
    p["dmb_c200c"] = dmb_c200c
    return p


def _trapz_mass(f_r, log_r):
    r = jnp.exp(log_r)
    return jnp.trapezoid(f_r * 4.0 * jnp.pi * r**2 * r, x=log_r)


def _cum_mass(rho, r):
    log_r = jnp.log(r)
    integ = 4.0 * jnp.pi * rho * r**3
    dln = jnp.diff(log_r)
    dM = 0.5 * (integ[:-1] + integ[1:]) * dln
    m_inner = (4.0 / 3.0) * jnp.pi * r[0] ** 3 * rho[0]
    return jnp.concatenate(
        [jnp.array([m_inner], dtype=rho.dtype), m_inner + jnp.cumsum(dM)]
    )


def _reverse_cumtrapz(f, r):
    dr = jnp.diff(r)
    dI = 0.5 * (f[:-1] + f[1:]) * dr
    from_out = jnp.cumsum(dI[::-1])[::-1]
    return jnp.concatenate([from_out, jnp.zeros((1,), dtype=f.dtype)])


def duffy_c200c(m200_phys, z, h):
    """Duffy et al. (2008) full-sample c200c; M in physical Msun."""
    m_h = m200_phys * h
    return 5.71 * (m_h / 2.0e12) ** (-0.084) * (1.0 + z) ** (-0.47)


def _nfw_f(c):
    return jnp.log(1.0 + c) - c / (1.0 + c)


def m500c_to_m200c(m500_phys, z, omega_m, h, c200_override=jnp.nan):
    """Convert M500c → M200c (physical Msun) via NFW + Duffy (or fixed c)."""
    ez2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m)
    # Physical critical density [Msun / Mpc^3]
    rho_c = (_RHO_CRIT_0_H * h**2) * ez2
    r500 = (3.0 * m500_phys / (4.0 * jnp.pi * 500.0 * rho_c)) ** (1.0 / 3.0)

    def body(_, m200):
        c_duffy = duffy_c200c(m200, z, h)
        c = jnp.where(jnp.isnan(c200_override), c_duffy, c200_override)
        r200 = (3.0 * m200 / (4.0 * jnp.pi * 200.0 * rho_c)) ** (1.0 / 3.0)
        m500_pred = m200 * _nfw_f(c * r500 / r200) / _nfw_f(c)
        return m200 * (m500_phys / jnp.maximum(m500_pred, 1e-30))

    m2000 = 1.5 * m500_phys
    return jax.lax.fori_loop(0, 8, body, m2000)


def dmb_halo_quantities(
    m_phys,
    z,
    c200c,
    omega_b,
    omega_m,
    h,
    params,
    r_work_over_r200=None,
):
    """DMB densities and Pe on a work radial grid (GODMAX BCM_18_wP).

    Parameters
    ----------
    m_phys : float
        M_200c in physical Msun.
    """
    n_int = int(params["num_points_trapz_int"])
    if r_work_over_r200 is None:
        r_work_over_r200 = jnp.logspace(-3.0, jnp.log10(16.0), 96)

    m_h = m_phys * h
    ez2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m)
    rho_c_z_h = _RHO_CRIT_0_H * ez2
    r200c_h = (m_h * 3.0 / (4.0 * jnp.pi * 200.0 * rho_c_z_h)) ** (1.0 / 3.0)
    r200c_h = r200c_h * (1.0 + z)  # comoving Mpc/h
    r200c_comoving = r200c_h / h

    r_h = r_work_over_r200 * r200c_h
    rt = params["epsilon_rt"] * r200c_h
    log_r_h = jnp.log(r_h)

    Mc0 = 10.0 ** params["log10_Mc0"]
    Mstar0 = 10.0 ** params["log10_Mstar0"]
    Mc = Mc0 * (m_h / Mstar0) ** params["nu_M"] * (1.0 + z) ** params["nu_z"]
    beta = (
        3.0
        * (m_h / Mc) ** params["mu_beta"]
        / (1.0 + (m_h / Mc) ** params["mu_beta"])
    )

    theta_ej = (
        params["theta_ej_0"]
        * (m_h / 10.0 ** params["log10_Mstar0_theta_ej"]) ** params["nu_theta_ej_M"]
        * (1.0 + z) ** params["nu_theta_ej_z"]
        * (1.0 / c200c) ** params["nu_theta_ej_c"]
    )
    theta_co = (
        params["theta_co_0"]
        * (m_h / 10.0 ** params["log10_Mstar0_theta_co"]) ** params["nu_theta_co_M"]
        * (1.0 + z) ** params["nu_theta_co_z"]
        * (1.0 / c200c) ** params["nu_theta_co_c"]
    )
    r_co = theta_co * r200c_h
    r_ej = theta_ej * r200c_h

    M1 = 10.0 ** params["log10_M1_starcga"]
    fstar = params["A_starcga"] * (M1 / m_h) ** params["eta_star"]
    fcga = params["A_starcga"] * (M1 / m_h) ** params["eta_cga"]
    fgas = (omega_b / omega_m) - fstar
    fclm = (1.0 - omega_b / omega_m) + fstar - fcga
    Rh = 0.015 * r200c_h

    nfw_trunc = bool(params["nfw_trunc"])
    gamma_g = params["gamma_rhogas"]
    delta_g = params["delta_rhogas"]

    def nfw_unnorm(r):
        rs = r200c_h / c200c
        x = r / rs
        y = r / rt
        rho = 1.0 / (x * (1.0 + x) ** 2)
        if nfw_trunc:
            rho = rho / (1.0 + y**2) ** 2
        return rho

    def gas_unnorm(r):
        u = r / r_co
        v = r / r_ej
        return 1.0 / (
            (1.0 + u) ** beta
            * (1.0 + v**gamma_g) ** ((delta_g - beta) / gamma_g)
        )

    log_norm = jnp.linspace(jnp.log(0.01 * r200c_h), jnp.log(r200c_h), n_int)
    rho_nfw_0 = m_h / _trapz_mass(nfw_unnorm(jnp.exp(log_norm)), log_norm)

    def rho_nfw(r):
        return rho_nfw_0 * nfw_unnorm(r)

    log_tot = jnp.linspace(jnp.log(0.01 * r200c_h), jnp.log(16.0 * r200c_h), n_int)
    Mtot = _trapz_mass(rho_nfw(jnp.exp(log_tot)), log_tot)
    rho_gas_0 = fgas * Mtot / _trapz_mass(gas_unnorm(jnp.exp(log_tot)), log_tot)

    def rho_gas(r):
        return rho_gas_0 * gas_unnorm(r)

    def rho_cga(r):
        return (
            (fcga * Mtot)
            / (4.0 * (jnp.pi**1.5) * Rh * r**2)
            * jnp.exp(-((0.5 * r / Rh) ** 2))
        )

    rho_nfw_arr = rho_nfw(r_h)
    rho_gas_arr = rho_gas(r_h)
    rho_cga_arr = rho_cga(r_h)

    Mnfw = _cum_mass(rho_nfw_arr, r_h)
    Mgas = _cum_mass(rho_gas_arr, r_h)
    Mcga = _cum_mass(rho_cga_arr, r_h)

    n_zeta_grid = 32
    zeta_grid = jnp.linspace(0.5, 1.5, n_zeta_grid)
    a_zeta = params["a_zeta"]
    n_zeta = params["n_zeta"]
    rf = zeta_grid[:, None] * r_h[None, :]
    log_rf = jnp.log(rf)

    def _interp_rows(log_q):
        return jnp.interp(log_q, log_r_h, Mcga), jnp.interp(log_q, log_r_h, Mgas)

    Mcga_rf, Mgas_rf = jax.vmap(_interp_rows)(log_rf)
    Mi = Mnfw[None, :]
    Mf = fclm * Mi + Mcga_rf + Mgas_rf
    Mf = jnp.maximum(Mf, 1e-30 * m_h)
    eq = (zeta_grid[:, None] - 1.0) - a_zeta * ((Mi / Mf) ** n_zeta - 1.0)
    zeta_arr = jax.vmap(
        lambda eq_col: jnp.interp(0.0, eq_col, zeta_grid), in_axes=1
    )(eq)

    rho_clm = (fclm / zeta_arr**3) * rho_nfw(r_h / zeta_arr)
    rho_dmb_arr = rho_gas_arr + rho_cga_arr + rho_clm
    mdmb_arr = _cum_mass(rho_dmb_arr, r_h)

    r_out = 6.0 * r200c_h
    w_out = jnp.where(r_h <= r_out, 1.0, 0.0)
    f_hse = w_out * (_G_KEV * rho_gas_arr * mdmb_arr / (r_h**2))
    ptot_comoving = jnp.clip(_reverse_cumtrapz(f_hse, r_h), 1e-30) * h**2

    a = 1.0 / (1.0 + z)
    ptot_phys = ptot_comoving / a**4
    fmax = 6.0 ** (-params["n_nt"]) / params["alpha_nt"]
    fz = jnp.minimum(
        (1.0 + z) ** params["beta_nt"],
        (fmax - 1.0) * jnp.tanh(params["beta_nt"] * z) + 1.0,
    )
    pnt_fac = params["alpha_nt"] * fz * (r_work_over_r200**params["n_nt"])
    pe_ev = (ptot_phys * jnp.maximum(0.0, 1.0 - pnt_fac) / _PTH_TO_PE) * _PE_KEV_TO_EV

    rho_to_phys = h**2
    return dict(
        r_comoving=r_h / h,
        r_over_r200=r_work_over_r200,
        r200c_comoving=r200c_comoving,
        rho_gas=rho_gas_arr * rho_to_phys,
        rho_cga=rho_cga_arr * rho_to_phys,
        rho_clm=rho_clm * rho_to_phys,
        rho_dmb=rho_dmb_arr * rho_to_phys,
        Pe=pe_ev,
        zeta=zeta_arr,
    )


def central_y0_m200(
    m200_phys,
    z,
    omega_b,
    omega_m,
    h,
    params,
    c200c=None,
):
    """Central Compton-y0 from DMB Pe at M_200c (physical Msun)."""
    if c200c is None:
        c_ov = params.get("dmb_c200c", jnp.nan)
        c200c = jnp.where(
            jnp.isnan(jnp.asarray(c_ov, dtype=jnp.float64)),
            duffy_c200c(m200_phys, z, h),
            jnp.asarray(c_ov, dtype=jnp.float64),
        )
    q = dmb_halo_quantities(m200_phys, z, c200c, omega_b, omega_m, h, params)
    r_phys_mpc = q["r_comoving"] / (1.0 + z)
    r_cm = r_phys_mpc * _MPC_TO_CM
    pe = q["Pe"]
    return 2.0 * (_SIGMA_T_CM2 / _ME_EV) * jnp.trapezoid(pe, r_cm)


def central_y0_m500(
    m500_phys,
    z,
    omega_b,
    omega_m,
    h,
    params,
):
    """Central y0 for a halo specified by M_500c (physical Msun)."""
    c_ov = params.get("dmb_c200c", jnp.nan)
    m200 = m500c_to_m200c(m500_phys, z, omega_m, h, c200_override=c_ov)
    return central_y0_m200(m200, z, omega_b, omega_m, h, params, c200c=None)


@partial(jax.jit, static_argnames=("nfw_trunc", "num_points_trapz_int"))
def central_y0_m500_packed(
    m500_phys,
    z,
    omega_b,
    omega_m,
    h,
    packed_dmb,
    nfw_trunc=True,
    num_points_trapz_int=64,
    dmb_c200c=jnp.nan,
):
    """JIT-friendly central y0; ``packed_dmb`` is ``pack_dmb_params`` tuple."""
    params = unpack_dmb_params(
        packed_dmb,
        nfw_trunc=nfw_trunc,
        num_points_trapz_int=num_points_trapz_int,
        dmb_c200c=dmb_c200c,
    )
    return central_y0_m500(m500_phys, z, omega_b, omega_m, h, params)
