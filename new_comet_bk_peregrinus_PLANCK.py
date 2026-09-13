"""
Script for c000 hod mocks  --  P(k) + B(k)  --  ported to the NEW desilike API
(branch `refactor-jax`, /pscratch/sd/p/prakharb/desilike).

Kept from the previous variant:
  1) `use_custom_priors` flag -> False keeps desilike DEFAULT priors for the
     chosen prior_basis (priors are printed either way).
  2) Analytic marginalization matched to the new pipeline: alpha*/sn2* are
     marginalized, sn0p is NOT.
  3) emcee with the GitHub `propose_fiducial_sampler_options` settings.

BGS support + nbar (2026-06-18):
  - simple_tracer_of() now resolves BGS correctly (the data tracer string passed
    to hod_data_tools2 is the *simple* tracer: BGS / LRG / ELG / QSO).
  - BGS HF snapshot is z=0.300; the redshift table uses 0.300 for BGS so the
    template z matches the data. BGS EZmock covariance is NOT volume-rescaled
    (handled inside hod_data_tools2).
  - Per-(hod_case, tracer) number density nbar from the Box MC table is passed
    to both the P(k) and B(k) theory classes (constructor kwarg `nbar`).
"""
import os

LOCAL_SAFE_THREADS = False
LOCAL_SAFE_THREAD_ENV = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
if LOCAL_SAFE_THREADS or os.environ.get('LOCAL_SAFE_THREADS', '').lower() in ('1', 'true'):
    for _name, _val in LOCAL_SAFE_THREAD_ENV.items():
        os.environ.setdefault(_name, _val)

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
os.environ["FOLPS_BACKEND"] = "jax"

import sys
# sys.path.insert(0, "/pscratch/sd/p/prakharb/desilike")  # NEW desilike (refactor-jax)

#sys.path.insert(0, "/global/homes/e/elbers/desilike2/desilike")
sys.path.insert(0, "/global/homes/e/elbers/decnu_main/pyclass_updated/pyclass")
sys.path.insert(0, "/global/homes/e/elbers/decnu_main/cosmoprimo_updated_again/cosmoprimo")

import desilike, inspect
print('desilike:', inspect.getfile(desilike))

# GPU / rank orchestration. MUST run before ANY JAX initialization.
from desilike import distributed
distributed.initialize()
import jax
from jax import config
config.update('jax_enable_x64', True)

import folps
print('folps:', inspect.getfile(folps))

import numpy as np
from mpi4py import MPI

from desilike.theories import CosmoprimoCosmology
from desilike.theories.galaxy_clustering import (
    DirectSpectrum2Template, ShapeFitSpectrum2Template,
    REPTVelocileptorsTracerSpectrum2Poles,
    FOLPSPTSpectrum2Poles, FOLPSTracerSpectrum2Poles, FOLPSTracerSpectrum3Poles)
from desilike.theories.galaxy_clustering.full_shape import get_physical_stochastic_settings
from desilike.observables.galaxy_clustering import Spectrum2PolesObservable, Spectrum3PolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.base import SumLikelihood, Posterior, compile as dk_compile, replace, params as get_params
from desilike.parameter import Parameter
from desilike.samplers import Sampler, Emcee
from desilike.profilers import Profiler, Minuit
from desilike import setup_logging
from desilike import Parameter
from cosmoprimo.fiducial import DESI

from cubic_peregrinus_data_tools import *
# from hod_data_tools2 import *

import argparse

setup_logging("info")


# ---------------------------------------------------------------------------
# Helper: full tracer (LRG1/LRG2/LRG3/ELG/QSO/BGS) -> simple tracer used by the
# data tools, stochastic-settings and nbar tables (BGS/LRG/ELG/QSO).
# ---------------------------------------------------------------------------
def simple_tracer_of(tracer):
    for s in ('BGS', 'LRG', 'ELG', 'QSO'):
        if tracer.startswith(s):
            return s
    return tracer


######### Settings #########

# CLI: select tracer + PT model up front so the whole build is parametrised.
parser = argparse.ArgumentParser()
parser.add_argument("--tracer",   default="BGS", help="BGS / LRG2 / ELG / LRG1 / LRG3 / QSO")
parser.add_argument("--pt_model", default="comet", choices=["EFT", "comet"])
# Optional k-cut / damping overrides (default: per-model KCUT_CONFIG below).
parser.add_argument("--kr_max",   type=float, default=None, help="P0 k_max [h/Mpc]")
parser.add_argument("--kr2_max",  type=float, default=None, help="P2 k_max [h/Mpc]")
parser.add_argument("--kb0_max",  type=float, default=None, help="B0 k_max [h/Mpc]")
parser.add_argument("--kb2_max",  type=float, default=None, help="B2 k_max [h/Mpc]")
parser.add_argument("--damping",  default=None, choices=["lor", "exp", "vdg"], help="RSD damping")
parser.add_argument("--c1p_width", type=float, default=None,
                    help="override the Gaussian c1p prior to N(0, width) (default: desilike default)")
parser.add_argument("--A_full", action="store_true", help="use A_full=True PT (own emulator)")
parser.add_argument("--no-bispectrum", dest="bispectrum", action="store_false",
                    help="fit P(k) only; by default the script fits P(k)+B(k)")
parser.set_defaults(bispectrum=True)
parser.add_argument("--run_chains",   action="store_true")
parser.add_argument("--run_profiler", action="store_true")
parser.add_argument("--test",         action="store_true")
args = parser.parse_args()

model = 'mnu_neg_fixnsob'
base_dir = '/global/cfs/cdirs/desi/science/cpe/elbers/v2/mock_challenge_peregrinus_comet/chains'
restart_chain = False
prior_basis = 'physical_aap'        # 'physical', 'physical_aap' or 'standard'

# Choose the Peregrinus simulation
simulation = 'PLANCK_M240_L4400_N6000_NU3000'
short_name = simulation.split('_')[0]
covariance_factor = 27.0 # convert from 6 Gpc/h to 2 Gpc/h (EZmock to Abacus)
covariance_factor *= ((2000/0.6736) / 4400.)**3 # convert from 2 Gpc/h to 4.4 Gpc (Abacus to Peregrinus)
covariance_factor *= 25.0 # equivalent to taking the average of 25 mocks
covariance_factor *= 0.2 # Volume x5

if (simulation == 'PLANCK_M240_L4400_N6000_NU3000'):
    simulation_ns = 0.968468027272727
    simulation_omega_b = 0.05101 * 0.6623**2
elif (simulation == 'DESIY1_M060_L4400_N6000_NU3000'):
    simulation_ns = 0.968454824867455
    simulation_omega_b = 0.04848101350823 * 0.681130127150482**2

# Per-model k-cuts [h/Mpc] (P0, P2, B0, B2) and RSD damping.
# EFT fixes X_FoG=0 so damping is inert; comet frees X_FoG (damping lor).
KCUT_CONFIG = {
    'EFT':    dict(kr_max=0.20, kr2_max=0.20, kr_b0_max=0.20, kr_b2_max=0.03, damping='lor'),
    'comet': dict(kr_max=0.30, kr2_max=0.30, kr_b0_max=0.20, kr_b2_max=0.03, damping='lor'),
}

pt_model  = args.pt_model           # 'comet' or 'EFT'/'folpsEFT' (comet => X_FoG free)
_cfg      = KCUT_CONFIG[pt_model]
# CLI overrides win; otherwise fall back to the per-model defaults.
kr_max    = args.kr_max  if args.kr_max  is not None else _cfg['kr_max']     # P0 k_max
kr2_max   = args.kr2_max if args.kr2_max is not None else _cfg['kr2_max']    # P2 k_max
kr_b0_max = args.kb0_max if args.kb0_max is not None else _cfg['kr_b0_max']  # B0 k_max
kr_b2_max = args.kb2_max if args.kb2_max is not None else _cfg['kr_b2_max']  # B2 k_max
damping   = args.damping if args.damping is not None else _cfg['damping']    # 'lor','exp','vdg'

hexa = False
bispectrum = args.bispectrum
bispectrum = False
if bispectrum is False:
    kr_b0_max = None
    kr_b2_max = None

# Only used when use_custom_priors=True
width_EFT = 12.5
width_SN0 = 2.0
width_SN2 = 5.0

set_emulator = False
A_full_status = args.A_full          # CLI toggle (default False); True trains its own emulator
b3_coev = False

# --- CHANGE 1 ---------------------------------------------------------------
use_custom_priors = False
# ----------------------------------------------------------------------------

# tracers = [args.tracer]            # LRG1/LRG2/LRG3/ELG/QSO/BGS
tracers = ['LRG1', 'LRG2']            # LRG1/LRG2/LRG3/ELG/QSO/BGS
# tracers = ['LRG1']            # LRG1/LRG2/LRG3/ELG/QSO/BGS
all_tracers = {'BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO'}
tracers_str = "all" if set(tracers) == all_tracers else "_".join(tracers)

tracer_str = simple_tracer_of(tracers[0])          # BGS/LRG/ELG/QSO (was LRG-only before)

# Auto-constructed output path (no override): chains land in their proper dir.
bk_tag = f"_kb0{kr_b0_max:.3f}_kb2{kr_b2_max:.3f}" if bispectrum else ""
damping_tag = f"_damping_{damping}" if damping != 'lor' else ""
c1p_tag = f"_c1pw{args.c1p_width:g}" if args.c1p_width is not None else ""
chain_name = (
    f"{base_dir}/{tracers_str}"
    f"_{'std' if prior_basis == 'standard' else 'phys'}"
    f"_kr{kr_max:.3f}_kr2{kr2_max:.3f}"
    f"{bk_tag}"
    f"{'_hexa' if hexa else ''}"
    f"_{pt_model}"
    f"_{model}"
    f"_V5"
    f"_{short_name}"
    f"_{'Afull' if A_full_status else 'Ano'}"
    f"_{'b3_coev' if b3_coev else 'b3_samp'}"
    f"{damping_tag}"
    f"{c1p_tag}"
)
# desilike Sampler rejects an output_dir whose final folder has a '.' (treated as
# a file suffix), so make the k-cut tokens dot-free: kr0.200 -> kr0p200.
chain_name = chain_name.replace('.', 'p')
output_dir = chain_name            # NEW API: sampler writes checkpoints into a directory
print('output_dir:', output_dir)


# --- Custom priors (only applied when use_custom_priors=True) ----------------
#   PS: b1p,b2p,bsp,b3p,alpha0p,alpha2p,alpha4p,sn0p,sn2p,X_FoGp,ctp
#   BS (shared b1p,b2p,bsp): c1p,c2p,snb0p,X_FoGp
def make_params(prior_basis, width_EFT, width_SN0, width_SN2, sigma8_fid, pt_model='comet', b3_coev=False):
    params = {}
    if prior_basis in ('physical', 'physical_aap'):
        params['b1p'] = {'fixed': False, 'prior': {'dist': 'uniform', 'limits': [0.1, 4]}}
        params['b2p'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['bsp'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': -2 / 7 * sigma8_fid**2, 'scale': 5}}
        if b3_coev:
            params['b3p'] = {'fixed': True}
        else:
            params['b3p'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': 23 / 42 * sigma8_fid**4, 'scale': sigma8_fid**4}}
        params['alpha0p'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha2p'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha4p'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['sn0p'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2p'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}
        if pt_model in ('EFT', 'folpsEFT'):
            params['X_FoGp'] = {'fixed': True, 'value': 0.0}
        else:
            params['X_FoGp'] = {'fixed': False, 'prior': {'dist': 'uniform', 'limits': [0, 10]}}
        params['c1p'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['c2p'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['snb0p'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
    else:
        params['b1'] = {'fixed': False, 'prior': {'dist': 'uniform', 'limits': [1e-5, 10]}}
        params['b2'] = {'fixed': False, 'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        params['bs'] = {'fixed': False, 'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        if b3_coev:
            params['b3'] = {'fixed': True}
        else:
            params['b3'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': 0, 'scale': 25}}
        params['alpha0'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha2'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha4'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['sn0'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2'] = {'fixed': False, 'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}
    return params


# --- Cosmology (NEW: CosmoprimoCosmology) -----------------------------------
fiducial = DESI()
# cosmo = CosmoprimoCosmology(engine='class', fiducial='DESI')
cosmo = CosmoprimoCosmology(engine='decnuclass', fiducial=('DESI', dict(kmax_pk=1)))

cosmo.params.set(Parameter('deg_ncdm', value=3, fixed=True))
cosmo.params.set(Parameter('m_ncdm', fixed=False, value=0.00333, ref=dict(dist='norm', loc=0.02, scale=0.12, limits=[-0.33333, 0.33333])))
cosmo.params.set(Parameter('N_eff', derived=True))
cosmo.params.set(Parameter('compensate_ur_for_negative_mnu', fixed=True, value=1))

# cosmo.params.set(Parameter('N_ur', fixed=False, value=0.00441, derived='0.00441 if m_ncdm >= 0 else 6.08627', depends={'m_ncdm': cosmo.params['m_ncdm']}))
# cosmo.params.set(Parameter('ncdm_fluid_approximation', fixed=True, value=3))
# cosmo.params.set(Parameter('k_per_decade_for_bao', fixed=True, value=30))
# cosmo.params.set(Parameter('tol_perturbations_integration', fixed=True, value=1e-3))



# cosmo.params.set(Parameter('N_ur', derived='0.00441 if {m_ncdm} >= 0 else 6.08627'))
# cosmo.params['deg_ncdm'].update(derived=False, fixed=True, value=3)
# cosmo.params['deg_ncdm'].update(derived='0.00441 if {m_ncdm} >= 0 else 6.08627')
# cosmo.params['N_eff'].update(derived=True)

cosmo_vc = get_params(cosmo, level=1)
for p in cosmo_vc:
    name = p.basename
    if name == 'tau_reio':
        p.update(fixed=True)
    elif name == 'n_s':
        # p.update(fixed=False, prior={'dist': 'norm', 'loc': simulation_ns, 'scale': 0.042})   # Planck n_s prior
        p.update(fixed=True, value=simulation_ns)   # Simulation value
    elif name == 'omega_b':
        # p.update(fixed=False, prior={'dist': 'norm', 'loc': simulation_omega_b, 'scale': 0.00055})
        p.update(fixed=True, value=simulation_omega_b)   # Simulation value
    elif name == 'h':
        p.update(fixed=False, prior={'dist': 'uniform', 'limits': [0.5, 0.9]})
    elif name == 'omega_cdm':
        p.update(fixed=False, prior={'dist': 'uniform', 'limits': [0.05, 0.2]})
    elif name == 'logA':
        p.update(fixed=False, prior={'dist': 'uniform', 'limits': [2.0, 4.0]})
    elif name == 'm_ncdm':
        p.update(fixed=False, prior={'dist': 'uniform', 'limits': [-0.33333, 0.33333]}, fd_eps = 0.16)
        # p.update(fixed=False, prior={'dist': 'uniform', 'limits': [-0.33333, 0.33333]}, fd_eps = [0.00333, 0.16, 0.16])
        # p.update(fixed=False, prior={'dist': 'uniform', 'limits': [0, 0.33333]}, fd_eps = 0.07)

# Derived cosmology params surfaced in the chain (updated desilike computes these
# in-pipeline, so sigma8 no longer needs emulator post-processing). See
# /global/homes/p/prakharb/desi-clustering/full_shape/tools.py L117-121.
add_derived_cosmo = True
if add_derived_cosmo:
    for dname, latex in [('H0', 'H_0'), ('Omega_m', r'\Omega_\mathrm{m}'),
                         ('sigma8_m', r'\sigma_{8,\mathrm{m}}'),
                         ('sigma8_cb', r'\sigma_{8,\mathrm{cb}}'),
                         ('rs_drag', 'r_s')]:
        cosmo_vc.set(Parameter(dname, derived=True, latex=latex))
cosmo.update(params=cosmo_vc)


# --- Redshifts (BGS = z0.300 HF snapshot) -----------------------------------
all_tracer_redshifts = {'LRG1': 0.5, 'LRG2': 0.8, 'QSO': 1.4}
tracer_redshifts = {tracer: all_tracer_redshifts[tracer] for tracer in tracers}
                        
                        
# --- Number density nbar [(Mpc/h)^-3] per (simulation, simple tracer) ----------
# From the Box MC table (BGS/LRG2/ELG1/QSO). Passed to the theory; stochastic
# terms enter as sn0/nbar, sn2/nbar*fsat*sigv^2, ...  (theory default is 1e-4).
NBAR_TABLE = {
    'DESIY1_M060_L4400_N6000_NU3000': {'LRG1': 0.00056495, 'LRG2': 0.00067901},
    'PLANCK_M240_L4400_N6000_NU3000': {'LRG1': 0.00061236, 'LRG2': 0.00056586},
}


# --- Build theories (NEW classes) -------------------------------------------
theories = {}
for tracer in tracers:
    z = tracer_redshifts[tracer]
    simple = simple_tracer_of(tracer)
    nbar = NBAR_TABLE[simulation][tracer]
    template = DirectSpectrum2Template(fiducial=fiducial, cosmo=cosmo, z=z)
    sigma8_fid = fiducial.sigma8_z(z)

    if pt_model == 'rept_velocileptors':
        ps_theory = REPTVelocileptorsTracerSpectrum2Poles(template=template, tracers=tracer,
                                                          prior_basis=prior_basis, nbar=nbar)
    elif pt_model == 'comet':
        pt = COMETPTSpectrum2Poles(A_full=A_full_status)
        ps_theory = COMETTracerSpectrum2Poles(template=template, pt=pt, tracers=tracer,
                                              prior_basis=prior_basis, damping=damping, nbar=nbar)
    else:
        pt = FOLPSPTSpectrum2Poles(A_full=A_full_status)
        ps_theory = FOLPSTracerSpectrum2Poles(template=template, pt=pt, tracers=tracer,
                                              prior_basis=prior_basis, damping=damping, nbar=nbar,
                                              damping_method='tree+loop')
    bs_theory = None
    if bispectrum:
        bs_theory = FOLPSTracerSpectrum3Poles(template=template, tracers=tracer,
                                              prior_basis=prior_basis, damping=damping,
                                              A_full=A_full_status, nbar=nbar,
                                              damping_method='tree+loop')

    # per-tracer physical stochastic settings (fsat, sigv)
    kw_stoch = get_physical_stochastic_settings(tracer=simple)
    ps_theory.update(**kw_stoch)
    if bs_theory is not None:
        bs_theory.update(**kw_stoch)
    print(f"[{tracer}] simple={simple}  z={z}  nbar={nbar}  fsat={kw_stoch['fsat']}  sigv={kw_stoch['sigv']:.4f}")

    theories[tracer] = {'ps': ps_theory, 'bs': bs_theory, 'sigma8_fid': sigma8_fid}


# --- Apply (optional custom) priors, marginalization, b3 coevolution; print --
marg_basenames = ('alpha0p', 'alpha2p', 'alpha4p', 'sn2p',           # physical_aap
                  'alpha0', 'alpha2', 'alpha4', 'sn2')               # standard  (sn0 NOT marg)
for tracer in tracers:
    sigma8_fid = theories[tracer]['sigma8_fid']
    custom = make_params(prior_basis, width_EFT, width_SN0, width_SN2, sigma8_fid,
                         pt_model=pt_model, b3_coev=b3_coev) if use_custom_priors else {}
    for comp in (['ps', 'bs'] if bispectrum else ['ps']):
        th = theories[tracer][comp]
        vc = get_params(th, level=1)
        for p in vc:
            name = p.basename
            if use_custom_priors and name in custom:
                p.update(**custom[name])
            if comp == 'ps' and name in marg_basenames:
                p.update(derived='marg')
            if pt_model not in ('EFT', 'folpsEFT') and name in ('X_FoGp', 'X_FoG'):
                lim = [0, 15] if comp == 'bs' else [0, 10]
                p.update(fixed=False, prior={'dist': 'uniform', 'limits': lim})
            if b3_coev and name in ('b3p', 'b3'):
                p.update(fixed=True)
            if args.c1p_width is not None and name in ('c1p', 'c1'):
                p.update(fixed=False, prior={'dist': 'norm', 'loc': 0.0, 'scale': args.c1p_width})
                print(f"[{tracer}] {comp} {name}: c1p prior widened to N(0, {args.c1p_width})")
        th.update(params=vc)

    if not use_custom_priors:
        print(f"[{tracer}] use_custom_priors=False -> desilike DEFAULT priors for prior_basis='{prior_basis}'")
    for comp in (['ps', 'bs'] if bispectrum else ['ps']):
        print(f"\n===== {tracer} {comp.upper()} priors =====")
        for p in get_params(theories[tracer][comp], level=1):
            print(f"  {p.basename:12s} fixed={str(p.fixed):5s} solved={str(getattr(p,'solved',False)):5s} prior={p.prior}")


# --- Observables (NEW classes; still take data/covariance/k/ells) ------------
def create_observable(comp, tracer, tracer_str = 'LRG', z_ev = 0.8,
                      k_max=0.301, k_max_b0=None, k_max_b2=None, P4=False, k_max_p2=None):
    if k_max_b0 is None:
        k_max_b0 = 0.12
    if k_max_b2 is None:
        k_max_b2 = 0.08
        
    z_string = f"z{z_ev:.3f}"
    print("z_string:", z_string)

    # dataset = build_pk_bk_data_hod(
    #     tracer=tracer_str, cosmo='c000', param=param_challenge_case,
    #     k_min_p=0.02, k_max_p=k_max, k_max_p2=k_max_p2,
    #     k_min_b=0.02, k_max_b0=k_max_b0, k_max_b2=k_max_b2,
    # )
    dataset = build_pk_data_peregrinus_cubic(
        tracer=tracer_str, z_string = z_string, sim=simulation,
        k_min_p=0.02, k_max_p0=k_max, k_max_p2=k_max_p2,
        k_max_b0=k_max_b0, k_max_b2=k_max_b2,
        Vol = covariance_factor
    )
    print(dataset.keys())

    ps_obs = bs_obs = None
    ps_ells = (0, 2, 4) if P4 else (0, 2)
    if "ps" in comp:
        ps_obs = Spectrum2PolesObservable(
            data=dataset['pk_lsstypes'], covariance=dataset['cov_pk'],
            theory=theories[tracer]['ps'], k=dataset['k_data'], ells=ps_ells)
    if "bs" in comp:
        bs_obs = Spectrum3PolesObservable(
            data=dataset['bk_lsstypes'], covariance=dataset['cov_bk'],
            theory=theories[tracer]['bs'], k=(dataset['kr_b0'], dataset['kr_b2']),
            ells=[(0, 0, 0), (2, 0, 2)])
        return ps_obs, bs_obs, dataset['cov_array']
    return ps_obs, None, None


observables = {}
for tracer in tracers:
    ps_obs, bs_obs, cov_array = create_observable(
        ["ps", "bs"] if bispectrum else ["ps"], tracer, tracer_str, tracer_redshifts[tracer],
        kr_max, kr_b0_max, kr_b2_max, hexa, kr2_max)
    observables[tracer] = {"ps": ps_obs, "bs": bs_obs, "cov": cov_array}


# --- Emulator (NEW: TaylorEmulator + replace) --------------------------------
# The PT (Taylor) emulator covers k up to the run's k-cut(s). A cached emulator
# whose every k-cut is >= the requested one therefore COVERS this run, so we
# reuse it instead of refitting; we only train when no cached emulator covers the
# request (i.e. the requested kmax exceeds all trained ones). Among the covering
# emulators we pick the smallest (least over-coverage / closest to requested).
import glob as _glob

def find_reusable_emulator(pattern, kidxs, req):
    """Return the smallest cached emulator that covers `req` (all parsed k-cuts
    >= req), or None. `kidxs` are the field indices of the k-cuts in the
    underscore-split basename (sans the '_derived.h5' suffix)."""
    best, best_key = None, None
    for fn in _glob.glob(pattern):
        parts = os.path.basename(fn)[:-len('_derived.h5')].split('_')
        try:
            kv = [float(parts[i]) for i in kidxs]
        except (IndexError, ValueError):
            continue
        if all(a >= b - 1e-9 for a, b in zip(kv, req)):
            key = tuple(kv)
            if best_key is None or key < best_key:
                best, best_key = fn, key
    return best

if set_emulator:
    for tracer in tracers:
        z = all_tracer_redshifts[tracer]
        for comp in (["ps", "bs"] if bispectrum else ["ps"]):
            obs = observables[tracer][comp]
            theory = theories[tracer][comp]
            edir = f'./Emulators/Emulator_{comp}'
            pattern = f'{edir}/{comp}_emu_{tracer}_z{z}_*_Afull_{A_full_status}_{pt_model}_{model}_{short_name}_derived.h5'
            if comp == 'ps':
                emu_fn = f'{edir}/{comp}_emu_{tracer}_z{z}_{kr_max}_Afull_{A_full_status}_{pt_model}_{model}_{short_name}_derived.h5'
                kidxs, req = [4], [kr_max]                       # basename field: ...z{z}_{kmax}_Afull...
            else:
                # Use the ACTUAL B2 cut in the name/req (was clamped to 0.08 before,
                # which mislabeled B2=0.03 emulators as 0.08 and caused false reuse
                # across different B2 cuts). Existing files renamed 0.08 -> 0.03.
                emu_fn = f'{edir}/{comp}_emu_{tracer}_z{z}_{kr_max}_{kr_b0_max}_{kr_b2_max}_Afull_{A_full_status}_{pt_model}_{model}_{short_name}_derived.h5'
                kidxs, req = [4, 5, 6], [kr_max, kr_b0_max, kr_b2_max]   # {kmax}_{kb0}_{kb2}
            os.makedirs(edir, exist_ok=True)

            reuse = find_reusable_emulator(pattern, kidxs, req)
            if reuse is not None:
                print(f"{comp.upper()} emulator for {tracer}: reusing covering cache "
                      f"{os.path.basename(reuse)} (requested k-cuts={req})")
                emulator = TaylorEmulator.read(reuse)
            else:
                print(f"Fitting {comp.upper()} emulator for {tracer} -> {emu_fn} "
                      f"(no cache covers k-cuts={req})")
                emulator = TaylorEmulator(dk_compile(theory.pt), order=4)
                emulator.fit()
                if MPI.COMM_WORLD.rank == 0:
                    emulator.write(emu_fn)
            replace(obs, theory.pt, emulator.to_calculator())

print('All theories emulated' if set_emulator else 'EMULATOR NOT ACTIVATED')


# --- Likelihood --------------------------------------------------------------
Likelihoods = []
for tracer in tracers:
    tracer_observables = [observables[tracer]['ps']]
    if bispectrum:
        tracer_observables.append(observables[tracer]['bs'])
    Likelihoods.append(ObservablesGaussianLikelihood(
        tracer_observables,
        covariance=observables[tracer]['cov']))
likelihood = SumLikelihood(Likelihoods)


# --- Run (args parsed at top) ------------------------------------------------
if args.test:
    pipe = dk_compile(likelihood)
    logp = pipe()
    print('logposterior:', logp)
    center = {p.name: p.value for p in pipe.params}
    _, deriveds = pipe(center, return_derived=True)
    print('derived params:', list(deriveds))
    if 'Omega_m' in deriveds:
        print('Omega_m (derived, in-pipeline) =', float(np.ravel(deriveds['Omega_m'])[0]))


if args.run_profiler:
    posterior = dk_compile(Posterior(likelihood))
    profiler = Profiler(posterior, kernel=Minuit(), output_fn=f'{output_dir}_profiles')
    profiler.maximize(niterations=4)
    if MPI.COMM_WORLD.rank == 0:
        print(profiler.profiles.to_stats(tablefmt='pretty'))


if args.run_chains:
    # emcee with the GitHub propose_fiducial_sampler_options settings:
    #   init : rng=42, nparallel=4, batch_size=16
    #   run  : min_steps=50, gelman_rubin=1.05, ess=800, thinning=5
    os.makedirs(output_dir, exist_ok=True)
    posterior = dk_compile(Posterior(likelihood))
    sampler = Sampler(posterior, kernel=Emcee(), output_dir=output_dir,
                      nparallel=4, rng=187, batch_size=16)
    sampler.run(min_steps=50, gelman_rubin=1.1, ess=800)
