import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math
import glob,os
import sys
import scipy
from importlib import  reload
from time import process_time 
#from libraries.lib_gather_data import get_hhid_FIES

from main_function_library import run_model, run_joyplots
from survey_libraries import *
# from shock_libraries import *
# from mc_storage_libraries import *
# from plotting_libraries import plot_shock, plot_income_distributions, plot_poverty_time_series
# from response_libraries import get_response_sp
# from demographic_libraries import summarize_demographics
# from predictive_libraries import  df_to_linear_fit
# from shock_comparison_libraries import relaxation_plots
# from maps_libraries import make_choropleths


#from income_shock_libraries_ps import rand_weighted_shock_1
#
from libraries.lib_country_dir import set_directories, load_survey_data, get_places_dict
from libraries.lib_get_hh_savings import get_hh_savings
from libraries.pandas_helper import broadcast_simple

# formatting & aesthetics
font = {'family':'sans serif', 'size':10}
plt.rc('font', **font)
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.facecolor'] = 'white'
sns.set_style("white")
plt.grid(False)

myC = 'PH'
base_code = 'base'


##################################################################
# JOYPLOTS
# run_joyplots()
# assert(False)


##################################################################
# run model

# number of simulations for command line
if len(sys.argv) > 1: Nsims = int(sys.argv[1])
else: Nsims = 100

# Load hh survey
df = load_hh_survey(myC,troubleshoot_merge=False)

# loop / generate MC for sectoral relaxation
if True:
	full_run = [base_code]+['relax_{}50'.format(sc) for sc in df['LFS_sector'].unique() if sc != 'unclassified']
	print('\nRunning {} shocks with {} sims each'.format(len(full_run),Nsims))

	for scode in full_run:
	    print('\n',scode)
	    run_model(df,scode,Nsims,write_out=True)
	    if scode == 'base': run_joyplots()

# sensitivity to savings
if True:
	print('\nRunning sensitivity to savings')
	run_model(df,'base_sav2x',200,savings_flow_to_stock_factor=2.0)
	run_model(df,'base_sav4x',200,savings_flow_to_stock_factor=3.0)
	run_model(df,'base_sav4x',200,savings_flow_to_stock_factor=4.0)

# Run impact channels separately to study non-linear effects
if True:
	print('\nRunning shock channels independently')
	run_model(df,'W1E0R0',100,cancel_entrep_shock=True,remits_intl_shock_mean=0.)
	run_model(df,'W1E1R0',100,remits_intl_shock_mean=0.)
	run_model(df,'W1E0R1',100,cancel_entrep_shock=True)
	run_model(df,'W0E1R0',100,cancel_wage_shock=True,remits_intl_shock_mean=0.0)
	run_model(df,'W0E0R1',100,cancel_wage_shock=True,cancel_entrep_shock=True)
	run_model(df,'W0E1R1',100,cancel_wage_shock=True)