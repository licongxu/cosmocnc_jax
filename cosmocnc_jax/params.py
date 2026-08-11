import numpy as np
from .config import *

cnc_params_default = {

    "survey_sr": f"{path_to_cosmocnc}/surveys/survey_sr_so_sim.py", #File where the survey scaling relations are defined
    "survey_cat": f"{path_to_cosmocnc}/surveys/survey_cat_so_sim.py", #File where the survey catalogue(s) are defined

    #Number of cores

    "number_cores_hmf": 1,
    "number_cores_abundance": 1,
    "number_cores_data": 8,
    "number_cores_stacked":8,

    "parallelise_type": "patch", #"patch" or "redshift"

    #Precision parameters

    "n_points": 4096, ##number of points in which the mass function at each redshift (and all the convolutions) is evaluated
    "n_z": 50,
    "n_points_data_lik": 128, #number of points for the computation of the cluster data part of the likelihood
    "sigma_mass_prior": 5.,
    "downsample_hmf_bc": 1,
    "padding_fraction": 0.,
    "pad_abundance": False,
    "bc_chunk_size": 2000,  # backward conv chunk size (GPU-optimal for no-zerr path)

    # nd_convolution_mode: Convolution mode for N-D (2D+) backward convolution.
    #   "circular": Circular FFT convolution — ~4x faster (90ms vs 420ms at npts=128
    #     for 2D). Safe when both signal and kernel decay to ~0 at the grid boundaries,
    #     which is typically the case. Should be verified for each new observable setup.
    #   "linear": Zero-padded linear convolution (equivalent to scipy convolve 'same').
    #     Stable and correct regardless of boundary conditions, but slower due to
    #     padding to next power-of-2.
    #   Does not affect 1D convolutions (always linear).
    "nd_convolution_mode": "linear",

    #Observables and catalogue

    "load_catalogue": True,
    "precompute_cnc_quantities_catalogue": True,
    "likelihood_type": "unbinned", #"unbinned", "binned", or "extreme_value"
    "obs_select": "q_so_sim", #"q_mmf3_mean",
    "observables": [["q_so_sim"],["p_so_sim"]],
    "cluster_catalogue":"SO_sim_0",#"Planck_MMF3_cosmo",
    "data_lik_from_abundance":True, #if True, and if the only observable is the selection observable,
    "data_lik_type":"backward_convolutional", #"backward_convolutional" or "direct_integral". Note that "direct_integral" only works with one correlation set
    "abundance_integral_type":"fft", #"fft" or "direct"
    "compute_abundance_matrix":False, #only true if the abundance matrix is needed
    "catalogue_params":{"downsample":True},
    "apply_obs_cutoff":False,
    "get_masses":False,
    "delta_m_with_ref":False,

    #Range of abundance observables

    "obs_select_min": 6.,
    "obs_select_max": 100.,
    "z_min": 0.01,
    "z_max": 1.01,

    # Cosmology and HMF parameters
    #
    # cosmology_tool: "classy_sz_jax" uses direct CosmoPowerJAX emulators,
    #   keeping the entire hot path (update_params -> get_log_lik) in JAX.
    #   update_params is ~5ms (pure JAX, no Cython in hot path).
    #
    # hmf_calc: "cnc" computes the HMF internally using mcfit + Tinker08,
    #   fully in JAX. Required for classy_sz_jax.

    "cosmology_tool": "classy_sz_jax",
    "M_min": 5e13,
    "M_max": 5e15,
    "M_min_extended": None,
    "M_min_cutoff": None,
    "hmf_calc": "cnc", # "cnc" (JAX, required for classy_sz_jax), "hmf", "MiraTitan"
    "hmf_type": "Tinker08",
    "mass_definition": "500c",
    "hmf_type_deriv": "numerical", #"analytical" or "numerical"
    "power_spectrum_type": "cosmopower",
    "cosmo_amplitude_parameter": "sigma_8", #"sigma_8" or "A_s"
    "cosmo_param_density": "critical", #"physical" or "critical"
    "scalrel_type_deriv": "analytical", #"analytical" or "numerical"
    "sigma_scatter_min": 1e-5,
    "interp_tinker": "linear", #"linear" or "log", only if "hmf_calc"=="cnc"

    "cosmo_model": "lcdm",

    "class_sz_ndim_masses" : 100,  # when using emulators this is automatically fixed.
    "class_sz_concentration_parameter" : "B13",
    "class_sz_output": 'mPk,m500c_to_m200c,m200c_to_m500c',
    "class_sz_hmf": "T08M500c", # M500 or T08M500c for Tinker et al 208 HMF defined at m500 critical.
    "class_sz_use_m500c_in_ym_relation": 1,
    "class_sz_use_m200c_in_ym_relation": 0,

    # hmfast backend (used only when cosmology_tool == "hmfast"):
    #   "hmfast_emulator_set" picks the hmfast Cosmology emulator set,
    #   "hmfast_path" forces import of the hmfast package from a specific
    #   source path (prepended to sys.path) so an environment-installed
    #   hmfast doesn't shadow the intended one. Set to None to use whatever
    #   "import hmfast" resolves to.
    "hmfast_emulator_set": "lcdm:v1",
    "hmfast_path": "/scratch/scratch-lxu/compute_packages/hmfast/src",

    #Redshift errors parameters

    "z_errors": False,
    "n_z_error_integral": 100,
    "z_error_sigma_integral_range": 4.,
    "z_error_min": 1e-5, #minimum z std for which an integral over redshift in the cluster data term is performed (if "z_errors" = True)
    "z_bounds": False, #redshift bounds if there's no redshift measurement by the redshift is bounded (as in, e.g., SPT)

    "convolve_nz": False,
    "sigma_nz": 0.,

    #False detections

    "non_validated_clusters": False, #True if there are clusters which aren't validated. If so, a distribution for their selection obsevable pdf must be provided

    #Binned likelihood params

    "binned_lik_type":"z_and_obs_select", #can be "obs_select", "z", or "z_and_obs_select"
    "bins_edges_z": np.linspace(0.01,1.01,11),
    "bins_edges_obs_select": np.exp(np.linspace(np.log(6.),np.log(60),6)),

    #Stacked likelihood params

    "stacked_likelihood": False,
    "stacked_data": ["p_zc19_stacked"], #list of stacked data
    "compute_stacked_cov": True,

    #Only for simulator

    "cov_constant": {"0": True, "1": True},
    "observable_vectorised": True,
    "observable_vector": False,


    #Priors

    "priors": False,
    "theta_mc_prior": False,


    # Verbose:
    "cosmocnc_verbose": "none" # none, minimal or extensive

    }

scal_rel_params_ref = {
#Planck
"alpha":1.79,
"beta":0.66,
"log10_Y_star":-0.19,
"sigma_lnq":0.173,
"bias_sz":0.8, #a.k.a. 1-b
"sigma_lnmlens":0.2,
"sigma_mlens":0.5,
"bias_lens":1.,
"dof":0.,
"bias_cmblens":0.92,
"sigma_lnp":0.22,
"corr_lnq_lnp":0.,
"a_lens":1.,
"f_false_detection":0.0, #N_F / (N_F + N_T) fraction of false detections to total detections
"f_true_validated":1.,#fraction of true clusters which have been validated
"q_cutoff":0.,

#SZiFi Planck

"alpha_szifi":1.1233,#1233, #1.1233 ?
"A_szifi": -4.3054, #Arnaud values, respectively
"sigma_lnq_szifi": 0.173,

#SPT
# spt style lkl:
"A_sz": 5.1,
"B_sz": 1.75,
"C_sz": 0.5,

"A_x": 6.5,
"B_x": 0.69,
"C_x": -0.25,

"sigma_lnYx":0.255, # 'Dx' in Bocquet's code
"dlnMg_dlnr" : 0.,

'WLbias' : 0.,
'WLscatter': 0.,

'HSTbias': 0.,
'HSTscatterLSS':0.,

'MegacamBias': 0.,
'MegacamScatterLSS': 0.,

'corr_xi_Yx': 0.1, # 'rhoSZX' in Bocquet's code
'corr_xi_WL': 0.1, # 'rhoSZWL' in Bocquet's code
'corr_Yx_WL': 0.1,  # 'rhoWLX' in Bocquet's code

'SZmPivot': 3e14
}

scaling_relation_params_default = {

#Planck

"alpha":1.79,
"beta":0.66,
"log10_Y_star":-0.19,
"sigma_lnq":0.173,
"bias_sz":0.62, #a.k.a. 1-b
"sigma_lnmlens":0.2,
"sigma_mlens":0.5,
"bias_lens":1.,
"dof":0.,
"bias_cmblens":0.92,
"sigma_lnp":0.22,
"corr_lnq_lnp":0.77,
"a_lens":1.,
"f_false_detection":0.0, #N_F / (N_F + N_T) fraction of false detections to total detections
"f_true_validated":1.,#fraction of true clusters which have been validated
"q_cutoff":0.,

#SZiFi Planck

"alpha_szifi":1.1233, #1.1233 ? True value in synthetic catalogues is 1.1233, for some reason
"A_szifi": -4.3054, #Arnaud values, respectively
"sigma_lnq_szifi": 0.173,

#SPT
# spt style lkl:
"A_sz": 5.1,
"B_sz": 1.75,
"C_sz": 0.5,

"A_x": 6.5,
"B_x": 0.69,
"C_x": -0.25,

"sigma_lnYx":0.255, # 'Dx' in Bocquet's code
"dlnMg_dlnr" : 0.,

'WLbias' : 0.,
'WLscatter': 0.,

'HSTbias': 0.,
'HSTscatterLSS':0.,

'MegacamBias': 0.,
'MegacamScatterLSS': 0.,

'corr_xi_Yx': 0.1, # 'rhoSZX' in Bocquet's code
'corr_xi_WL': 0.1, # 'rhoSZWL' in Bocquet's code
'corr_Yx_WL': 0.1,  # 'rhoWLX' in Bocquet's code

'SZmPivot': 3e14,

#ACT

"A0": np.log10(1.9e-5),
"B0": 0.08,
"C0": 0.,
"sigma_lnq_act": 0.2,

#Planck DES Y3:

"lnb_wl_sigma": 0., #prior: unit standard deviation, mean 0
"b_wl_m": 1.029,
"s_wl_m": -0.226,

# DMB / BCM_18_wP pressure → central y0 (survey_sr_*_dmb only; unused by GNFW power-law path)
"theta_ej_0": 4.0,
"log10_Mstar0_theta_ej": 14.0,
"nu_theta_ej_M": 0.0,
"nu_theta_ej_z": 0.0,
"nu_theta_ej_c": 0.0,
"theta_co_0": 0.1,
"log10_Mstar0_theta_co": 14.0,
"nu_theta_co_M": 0.0,
"nu_theta_co_z": 0.0,
"nu_theta_co_c": 0.0,
"mu_beta": 0.21,
"eta_star": 0.3,
"eta_cga": 0.6,
"A_starcga": 0.09,
"log10_M1_starcga": 11.4,
"epsilon_rt": 4.0,
"log10_Mc0": 14.83,
"nu_z": 0.0,
"nu_M": 0.0,
"log10_Mstar0": 13.0,
"a_zeta": 0.3,
"n_zeta": 2.0,
"alpha_nt": 0.18,
"beta_nt": 0.5,
"n_nt": 0.3,
"gamma_rhogas": 2.0,
"delta_rhogas": 7.0,
"nfw_trunc": True,
"num_points_trapz_int": 64,
# NaN → Duffy 2008 c200c(M,z); set a float to fix concentration
"dmb_c200c": float("nan"),
}

cosmo_params_default = {

"Om0":0.315,
"Ob0":0.04897,
"Ob0h2":0.04897*0.674**2,
"Oc0h2":(0.315-0.04897)*0.674**2,
"h":0.674,
"A_s":2.08467e-09, #if amplitude_parameter == "sigma_8", this is overriden by the value given to "sigma_8" in this dictionary
"n_s":0.96,
"m_nu":0.06, #m_nu is sum of the three neutrino masses
"sigma_8":0.811, #if amplitude_paramter == "A_s", this is overriden; the amplitude is taken by the value given to "A_s" in this dictionary
"tau_reio": 0.0544,
"w0": -1.,
"Onu0": 0.00141808,
"N_eff": 3.046,

"k_cutoff": 1e8,
"ps_cutoff": 1,

}
