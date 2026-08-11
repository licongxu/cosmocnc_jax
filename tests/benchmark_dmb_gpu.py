"""GPU vs CPU vs GODMAX timing / memory for DMB central y0 and CNC abundance.

Run with a 10% GPU memory fraction (recommended):

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 CUDA_VISIBLE_DEVICES=0 \\
      JAX_ENABLE_X64=1 python tests/benchmark_dmb_gpu.py

Reports wall times and JAX device memory peak vs the allocator limit.
"""
from __future__ import annotations

import os
import sys
import time

# Keep TF off the GPU before any TF import via cosmocnc stack.
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
from jax import block_until_ready  # noqa: E402

from cosmocnc_jax.dmb_pressure import (  # noqa: E402
    DMB_DEFAULTS,
    pack_dmb_params,
    central_y0_m200,
    central_y0_m500_vmap,
    dmb_halo_quantities,
)
from cosmocnc_jax.config import path_to_cosmocnc  # noqa: E402
from cosmocnc_jax.params import scaling_relation_params_default  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GODMAX_SRC = os.path.join(REPO, "ref_packages", "GODMAX", "src")


def _mem():
    s = jax.devices()[0].memory_stats() or {}
    peak = s.get("peak_bytes_in_use", 0) / 1024**3
    used = s.get("bytes_in_use", 0) / 1024**3
    lim = s.get("bytes_limit", 0) / 1024**3
    return peak, used, lim


def _bench(fn, n=5, warmup=1):
    for _ in range(warmup):
        out = fn()
        block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
        block_until_ready(out)
    return (time.perf_counter() - t0) / n, out


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"backend: {jax.default_backend()}")
    Ob0, Om0, h = 0.04897, 0.315, 0.674
    params = {**DMB_DEFAULTS, "num_points_trapz_int": 64}
    packed = pack_dmb_params(dict(scaling_relation_params_default))
    c0, z0 = 5.0, 0.3
    m = 3e14

    # --- single Pe ---
    dt, _ = _bench(
        lambda: dmb_halo_quantities(m, z0, c0, Ob0, Om0, h, params)["Pe"], n=10
    )
    print(f"\nPe single halo:           {dt*1e3:8.2f} ms")

    # --- batched y0 (256 / 512) ---
    for nM in (256, 512, 1024):
        Ms = jnp.logspace(14.0, 15.5, nM)

        def run(Ms=Ms):
            return central_y0_m500_vmap(
                Ms, z0, Ob0, Om0, h, packed,
                nfw_trunc=True, num_points_trapz_int=64, dmb_c200c=c0,
            )

        dt, y = _bench(run, n=8)
        peak, used, lim = _mem()
        print(
            f"y0 vmap nM={nM:4d}:         {dt*1e3:8.2f} ms  "
            f"({nM/dt:.0f} halos/s)  peak={peak:.3f} GiB "
            f"({100*peak/lim if lim else 0:.1f}% of limit)"
        )

    # --- GODMAX ---
    if os.path.isdir(GODMAX_SRC):
        if GODMAX_SRC not in sys.path:
            sys.path.insert(0, GODMAX_SRC)
        from get_BCMP_profile_jit import BCM_18_wP

        M_h = m * h
        sim = dict(
            cosmo=dict(H0=100 * h, Om0=Om0, Ob0=Ob0, sigma8=0.81, ns=0.96, w0=-1.0),
            theta_ej_0=4.0, theta_co_0=0.1, log10_Mc0=14.83, mu_beta=0.21,
            eta_star=0.3, eta_cga=0.6, A_starcga=0.09, log10_M1_starcga=11.4,
            alpha_nt=0.18, beta_nt=0.5, n_nt=0.3, gamma_rhogas=2.0, delta_rhogas=7.0,
            nfw_trunc=True, epsilon_rt=4.0,
        )
        halo1 = dict(
            rmin=5e-3, rmax=3.0, nr=32, z_array=[z0],
            lg10_Mmin=np.log10(M_h), lg10_Mmax=np.log10(M_h), nM=1,
            cmin=c0, cmax=c0, nc=1,
        )
        halo32 = dict(
            rmin=5e-3, rmax=3.0, nr=32, z_array=[z0],
            lg10_Mmin=14.0, lg10_Mmax=15.5, nM=32,
            cmin=c0, cmax=c0, nc=1,
        )

        def god(hdict):
            b = BCM_18_wP(sim, hdict, num_points_trapz_int=64)
            return b.Pe_mat_physical

        dt1, _ = _bench(lambda: god(halo1), n=3, warmup=1)
        dt32, _ = _bench(lambda: god(halo32), n=2, warmup=1)
        print(f"GODMAX Pe 1 halo:         {dt1*1e3:8.2f} ms")
        print(f"GODMAX Pe 32 M grid:      {dt32*1e3:8.2f} ms  ({32/dt32:.1f} profiles/s)")

    # --- CNC abundance (optional; needs noise files) ---
    noise = os.environ.get("TSZSBI_NOISE_DIR", "/scratch/scratch-lxu/tszsbi/noise_files")
    if os.path.isfile(os.path.join(noise, "sigma_dict_szifi.npy")):
        from cosmocnc_jax import cluster_number_counts
        from cosmocnc_jax.params import cnc_params_default, cosmo_params_default

        survey = os.path.join(path_to_cosmocnc, "surveys", "survey_sr_planck_sim_dmb.py")
        cnc = dict(cnc_params_default)
        cnc.update(
            dict(
                survey_sr=survey,
                obs_select="q_planck_sim_dmb",
                observables=[["q_planck_sim_dmb"]],
                cosmology_tool="classy_sz_jax",
                hmf_calc="cnc",
                hmf_type="Tinker08",
                mass_definition="500c",
                M_min=1e14,
                M_max=1e16,
                z_min=0.01,
                z_max=1.01,
                n_z=20,
                n_points=256,
                load_catalogue=False,
                likelihood_type="binned",
                binned_lik_type="z_and_obs_select",
                data_lik_from_abundance=False,
                bins_edges_z=np.linspace(0.01, 1.01, 6),
                bins_edges_obs_select=np.exp(np.linspace(np.log(6), np.log(40), 5)),
                obs_select_min=6.0,
                obs_select_max=40.0,
                tszsbi_noise_dir=noise,
                cosmocnc_verbose="none",
            )
        )
        nc = cluster_number_counts(cnc_params=cnc)
        nc.cosmo_params = dict(cosmo_params_default)
        nc.scal_rel_params = dict(scaling_relation_params_default)
        nc.scal_rel_params["dof"] = 0.0
        nc.initialise()
        nc.update_params(nc.cosmo_params, nc.scal_rel_params)
        nc.get_hmf()

        def abund():
            nc.get_cluster_abundance()
            return nc.n_tot_vec

        dt, nvec = _bench(abund, n=5, warmup=1)
        peak, used, lim = _mem()
        print(
            f"\nCNC abundance (n_z=20, n_points=256): {dt*1e3:8.2f} ms  "
            f"n_tot={float(np.sum(np.asarray(nvec))):.3f}"
        )
        print(
            f"GPU memory peak={peak:.3f} GiB  used={used:.3f} GiB  "
            f"limit={lim:.3f} GiB  ({100*peak/lim if lim else 0:.1f}% of allocator limit)"
        )
    else:
        print("\n[skip CNC abundance — noise files not found]")


if __name__ == "__main__":
    main()
