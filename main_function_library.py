import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math
import glob,os
import sys
import scipy
from time import process_time 
#from libraries.lib_gather_data import get_hhid_FIES

from survey_libraries import *
from shock_libraries import *
from mc_storage_libraries import *
from plotting_libraries import plot_shock, plot_income_distributions, plot_poverty_time_series
from response_libraries import get_response_sp,load_social_amelioration_program
from demographic_libraries import summarize_demographics
from predictive_libraries import  df_to_linear_fit
from shock_comparison_libraries import relaxation_plots
from maps_libraries import make_choropleths

try: from joyplot_libraries import *
except: pass

# Unbreakable libraries
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


def run_model(df,scode,Nsims=100,
              write_out=True,
              verbose=False,
              wage_disruption_smear=0.1,
              wage_sector_value_smear=0.1,
              entrep_disruption_smear=0.1,
              sectoral_smear=0.1,
              remits_intl_shock_mean=0.13,
              remits_dom_shock_mean=0.0,
              savings_flow_to_stock_factor=1.0,
              cancel_wage_shock=False,
              cancel_entrep_shock=False):
    # df =  merged FIES/LFS dataframe (DataFrame)
    # scode = shock code (string)
    # Nsims = model runs per shock (int > 0)
    # write_out = record results (overwrite existing) (bool)
    
    # results vectorae
    mc = monte_carlo(Nsims,scode,
                             wage_disruption_smear=wage_disruption_smear,
                             wage_sector_value_smear=wage_sector_value_smear,
                             entrep_disruption_smear=entrep_disruption_smear,
                             sectoral_smear=sectoral_smear,
                             remits_intl_shock_mean=remits_intl_shock_mean,
                             remits_dom_shock_mean=remits_dom_shock_mean,
                             savings_flow_to_stock_factor=savings_flow_to_stock_factor,
                             cancel_wage_shock=cancel_wage_shock,
                             cancel_entrep_shock=cancel_entrep_shock)
    # apply savings adjustment if necessary
    df['savings'] *= mc.savings_params['savings_flow_to_stock_factor']

    # load shock
    df,mc = load_shock_template(df,mc)
    hh_df = None

    # prime sp (potential payout loaded once here)
    df = load_social_amelioration_program(df)
    
    # sort
    df = df.sort_index()

    # Start the stopwatch / counter  
    t_start = process_time() 

    # this is the loop that simulates income shock Nsims times:
    while mc.nsim < mc.mc_params['Nsims']:
    
        ####################################################
        # INCOME
        # step 1 - simulate income shock at individual level
        df = adjust_income_and_weight(df,mc)
        # ^ all flows are annual here
        
        # step 2 - sum impacts to household (propagate to non-workers, dependents)
        # IMPORTANT: all flows converted to MONTHLY (PPP$2011) here
        sum_to_sectoral_impact(df,mc)
        hh_df = sum_to_households(df,hh_df)
    
        # step 2.5 - record simulation info
        mc.store_income_impacts_by_class(hh_df)
        mc.collect_regional_results(hh_df)
    
        ####################################################
        # CONSUMPTION

        # step 3 - load social protection response
        hh_df = get_response_sp(hh_df,mc)
    
        # step 4 - poverty headcount time series, net of savings
        sim_stats = project_consumption_series(hh_df,mc)
    
        # step 5 - record poverty time series
        mc.store_consumption_time_series(sim_stats)
            
        ####################################################
        # ITERATE & REPORTING
        # To do: optimize for time...this is too slow!
        mc.nsim+=1
        if mc.nsim%50==0:
            print('-- {} - runtime = {} per 50 sims'.format(mc.nsim,round(process_time()-t_start,1)))
            t_start = process_time()

            
    ####################################################       
    # STORE & REPORT RESULTS
    if write_out:
        mc.write_out_results()
    if verbose: express_herself(hh_df)

    return hh_df

def run_joyplots():

    plot_losses_total_value('base')
    plot_sectoral_impacts('base')


def express_herself(hh_df):
    
    # total population:
    tpop = hh_df['popwgt'].sum()
    print('Total pop = ',tpop)
    
    # test income losses
    pct_aff = round(1E2*hh_df.loc[hh_df.income_loss > 0,'popwgt'].sum()/hh_df['popwgt'].sum(),2)
    pct_loss = round(1E2*hh_df.eval('popwgt*(income_loss/hhsize)').sum()/hh_df.eval('popwgt*pcinc_initial').sum(),2)
    pct_aff_loss = round(1E2*hh_df.loc[hh_df.income_loss > 0].eval('popwgt*(income_loss/hhsize)').sum()/hh_df.loc[hh_df.income_loss > 0].eval('popwgt*pcinc_initial').sum(),2)
    print('\n{}% affected'.format(pct_aff))
    print('{}% of all income lost for duration of shock.'.format(pct_loss))
    print('{}% of aff hh income lost for duration of shock.'.format(pct_aff_loss))
    #
    pct_aff_q1 = round(1E2*hh_df.loc[(hh_df.income_loss > 0)&(hh_df.quintile==1),'popwgt'].sum()/hh_df.loc[hh_df.quintile==1,'popwgt'].sum(),2)
    pct_loss_q1 = round(1E2*hh_df.loc[hh_df.quintile==1].eval('popwgt*(income_loss/hhsize)').sum()/hh_df.loc[hh_df.quintile==1].eval('popwgt*pcinc_initial').sum(),2)
    pct_aff_loss_q1 = round(1E2*hh_df.loc[(hh_df.quintile==1)&(hh_df.income_loss > 0)].eval('popwgt*(income_loss/hhsize)').sum()/hh_df.loc[(hh_df.quintile==1)&(hh_df.income_loss > 0)].eval('popwgt*pcinc_initial').sum(),2)
    print('\n{}% of Q1 affected'.format(pct_aff_q1))
    print('{}% of Q1 income lost for duration of shock.'.format(pct_loss_q1))
    print('{}% of aff Q1 income lost for duration of shock.'.format(pct_aff_loss_q1))
    #
    pct_aff_bottom50 = round(1E2*hh_df.loc[(hh_df.income_loss > 0)&(hh_df.decile<=5),'popwgt'].sum()/hh_df.loc[hh_df.decile<=5,'popwgt'].sum(),2)
    pct_loss_bottom50 = round(1E2*hh_df.loc[hh_df.decile<=5].eval('popwgt*(income_loss/hhsize)').sum()/hh_df.loc[hh_df.decile<=5].eval('popwgt*pcinc_initial').sum(),2)
    pct_aff_loss_bottom50 = round(1E2*hh_df.loc[(hh_df.decile<=5)&(hh_df.income_loss > 0)].eval('popwgt*(income_loss/hhsize)').sum()/hh_df.loc[(hh_df.decile<=5)&(hh_df.income_loss > 0)].eval('popwgt*pcinc_initial').sum(),2)
    print('\n{}% of poorest half affected'.format(pct_aff_bottom50))
    print('{}% of poorest half income lost for duration of shock.'.format(pct_loss_bottom50))
    print('{}% of aff poorest half income lost for duration of shock.'.format(pct_aff_loss_bottom50))
    #
        
    pct_cct_aff = round(1E2*hh_df.loc[(hh_df.income_loss > 0)&(hh_df.cct4P!=0)&(hh_df.decile<=5),'popwgt'].sum()/hh_df.loc[(hh_df.income_loss>0)&(hh_df.decile<=5),'popwgt'].sum(),2)
    pct_totpub_aff = round(1E2*hh_df.loc[(hh_df.income_loss > 0)&(hh_df.total_public!=0)&(hh_df.decile<=5),'popwgt'].sum()/hh_df.loc[(hh_df.income_loss>0)&(hh_df.decile<=5),'popwgt'].sum(),2)
    print(pct_cct_aff,pct_totpub_aff)


