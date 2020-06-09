import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math
import glob,os
import sys
import scipy
from importlib import reload

from income_shock_libraries_ps import *
from mc_storage_libraries import monte_carlo
from libraries.lib_country_dir import set_directories, load_survey_data, get_places_dict
from libraries.pandas_helper import broadcast_simple
from predictive_libraries import df_to_linear_fit

# formatting & aesthetics
font = {'family':'sans serif', 'size':10}
plt.rc('font', **font)
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.facecolor'] = 'white'
sns.set_style("white")

sns_pal = sns.color_palette('Set1', n_colors=8)
greys_pal = sns.color_palette('Greys', n_colors=9)


def get_OG_shock(ix_name='LFS_sector'):
    all_sectors = ['ag','mining','utilities','construction','manufacturing','wholesale','retail','transportation',
                   'information','finance','professional_services','eduhealth','food_entertainment','government','other']
    shutdown = {sec:[1,1] for sec in all_sectors}
    
    shock = { 'ag':           [ 0.,  0], #(1) agriculture, forestry, fishing, and hunting – 0% affected
             'mining':        [ 0.,  0], #(2) mining – 0% affected
             'utilities':     [ 0.,  0], #(3) utilities – 0% affected 
             'construction':  [0.5,1.0], #(4) construction – 50% affected, income dropped by 100%
             'manufacturing': [0.1,1.0], #(5) manufacturing – 10% affected, income dropped by 100%
             'wholesale':     [0.1,1.0], #(6) wholesale trade - 10% affected, income dropped 100%
             'retail':        [0.5,1.0], #(7) retail trade – 50% affected, income dropped by 100% 
             'transportation':[0.5,1.0], #(8) transportation and warehousing – 50% affected, income dropped by 100%
             'information':   [0.1,1.0], #(9) information – 10% affected, income dropped 100%  
             'finance':       [0.1,1.0], #(10) finance, insurance, real estate, rental and leasing – 10% affected, income dropped 100%
             'professional_services':[0.1,1.0], #(11) professional and business services – 10% affected, income dropped 100%
             'eduhealth':     [0.1,1.0], #(12) educational services, health care and social assistance – 10% affected, income dropped 100%
             'food_entertainment':[0.8,1.0], #(13) arts, entertainment, recreation, accommodation and food services – 80% affected, income dropped 100%
             'government':    [  0,  0], 
             'other':         [0.8,1.0]} #(14) other services, except government – 80% affected, income dropped 100%
    df_shock = pd.DataFrame(data=shock).T
    df_shock.columns = ['fa','di']
    df_shock.index.name = ix_name
    return df_shock
    #maybe we can compare the number of people "affected" (who are here assumed to lose their job and have no 
    # substitute income), and see if we are in the 20-30% ballpark (estimated unemployment in a month)

    #what's trickier is the people who are losing income but not fully (like owner of small shop who has   
    #20% of usual number of customer)
    
def bifurcate_households(df):
    ix_aff = pd.Index({'a','na'},name='affected_cat')
    return broadcast_simple(df,ix_aff)

def wgt_sanity_check(df,wgt='pwgt'):
    a = int(df[wgt].sum())
    b = int(2*df[wgt+'_shock'].sum())
    try: assert(a == b)
    except:
        print(a,'\n',b,'\n',round(1E2*b/a,2))
        assert(False)

def load_shock_template(df,mc):

    #########################################################
    # Load wage vulnerability to shock, by sector-> [[fa,di]]
    # reloads just once per shock type
    if mc.shock_code == 'base_v0': wage_shock_df = get_OG_shock()
    else: wage_shock_df = get_phi_shock(flavor=4)

    df = pd.merge(df.copy(),wage_shock_df,left_on='LFS_sector',right_index=True,how='left')
    df['fa'].fillna(0,inplace=True)
    df['di'].fillna(0,inplace=True)

    # relax sectoral shutdowns:
    if mc.shock_code[:6] == 'relax_':
        relax_sec = mc.shock_code[6:-2]
        relax_frac = 1E-2*float(mc.shock_code[-2:])
        #
        df.loc[df.LFS_sector==relax_sec,'fa'] *= (1.-relax_frac)

    # put sectoral info into sectoral results vectors
    # not sure this belongs here...
    for sec in df.LFS_sector:
        mc.total_value_wages[sec] = None
        mc.total_loss_wages[sec] = None
        mc.frac_loss_wages[sec] = None

    #########################################################
    # load entrepreneurial income shock
    mc.entrepreneurial_shock = get_phi_shock(flavor=3)

    # distribute entrepreneurial income among entrepreneurs
    for sec in mc.entrepreneurial_shock_dict:
        sec_code = mc.entrepreneurial_shock_dict[sec]

        df.loc[df.is_entrepreneur==0,sec_code] = 0
        df[sec_code] = df[sec_code]/df.groupby(level='hhid_lfs')['is_entrepreneur'].transform('sum')

    return df.sort_index(),mc

def adjust_income_and_weight(df,mc):

    if mc.nsim==0:
        print('population: {} million'.format(round(df['pwgt'].sum()*1E-6,2)))
        print('non-ag wages: {} bil. $PPP/month'.format(round(df.eval('(nonagri_sal*pwgt*ppp_factor)/hhsize_lfs').sum()*1E-9/12,2)))
        print('ag wages: {} bil. $PPP/month'.format(round(df.eval('(agri_sal*pwgt*ppp_factor)/hhsize_lfs').sum()*1E-9/12,2)))


    #################################################   
    # reset results cols
    df['affected'] = 0
    df['nonag_wage_loss'] = 0
    df['ag_wage_loss'] = 0
    df['entrep_loss'] = 0
    df['remits_loss_intl'] = 0
    df['remits_loss_dom'] = 0
    df['fa_sim'] = df['fa'].copy()


    #################################################
    # UNCERTAINTY: smear shutdown p/m (sectoral_smear)% (rel)
    # smear affected fraction at sectoral level
    if mc.mc_params['sectoral_smear'] != 0:
        for sec in df['LFS_sector'].unique():
            df.loc[df.LFS_sector==sec,'fa_sim'] *= (1+np.random.uniform(-mc.mc_params['sectoral_smear'],mc.mc_params['sectoral_smear']))

    # UNCERTAINTY: smear shutdown p/m (wage_smear) [0-1] (abs)
    # smear affected fraction at for entire formal economy 
    if mc.mc_params['wage_disruption_smear'] != 0:
        df['fa_sim'] += np.random.uniform(-mc.mc_params['wage_disruption_smear'],mc.mc_params['wage_disruption_smear'])


    #################################################
    # Check if affected
    #
    # eg: fa = 0.8 --> 80% chance rand < fa --> affected = True
    is_affected = df['fa_sim']>np.random.uniform(0,1,df.shape[0])
    if not mc.cancel_wage_shock: df.loc[is_affected,'affected'] = 1
    elif mc.nsim == 0: print('zeroing out wage shock for all sectors (mc.cancel_wage_shock)')
    # ^ includes switch to cancel wage shock (study channel contributions)

    # NOT EMPLOYED = NOT AFFECTED
    # not_employed = "(cempst1!='Employed')"
    # df.loc[df.eval(not_employed),'affected'] = 0
    # ^ affected includes employment (redundant)
    # ^ frac_hours_worked & di = 0 for non-workers
    # ^ nonagri_sal is constant for entire household


    #################################################
    # ANNUAL income loss, formal sector:
    # -> apply normal distribution to smear sectoral contributions & renormalize at hh level, then calculate losses
    
    # non-agricultural wages
    df['sim_frac_nonag_hours'] = df['frac_nonag_hours'].copy()
    if mc.mc_params['wage_sector_value_smear'] != 0:
        df['sim_frac_nonag_hours'] *= np.random.normal(1,mc.mc_params['wage_sector_value_smear'],df.shape[0])
    df['sim_frac_nonag_hours'] = df['sim_frac_nonag_hours']/(df.groupby(level='hhid_lfs')['sim_frac_nonag_hours']).transform('sum')
    
    # agricultural wages
    df['sim_frac_ag_hours'] = df['frac_ag_hours'].copy()
    if mc.mc_params['wage_sector_value_smear'] != 0:
        df['sim_frac_ag_hours'] *= np.random.normal(1,mc.mc_params['wage_sector_value_smear'],df.shape[0])
    df['sim_frac_ag_hours'] = df['sim_frac_ag_hours']/(df.groupby(level='hhid_lfs')['sim_frac_ag_hours']).transform('sum')

    # calculate wage loss
    # could merge ag & non-ag wages here
    df['nonag_wage_loss'] = df[['affected','di','sim_frac_nonag_hours','nonagri_sal']].prod(axis=1)
    df['ag_wage_loss'] = df[['affected','di','sim_frac_ag_hours','agri_sal']].prod(axis=1)


    # #######################
    # grab one value for entrepreneurial income smear 
    # (meant to be same for all sectors)
    fa_entrep_covariate_smear = np.random.uniform(-mc.mc_params['entrep_disruption_smear'],mc.mc_params['entrep_disruption_smear']) 

    # entrepreneurial income loss
    for sec in mc.entrepreneurial_shock_dict:

        # 0) initialize
        sec_code = mc.entrepreneurial_shock_dict[sec]        
        df['entrep_affected'] = 0

        # 1) smear fa_entrep
        fa_entrep_mean = float(mc.entrepreneurial_shock.loc[sec,'fa'])
        fa_entrep_sectoral_smear = np.random.uniform(-mc.mc_params['sectoral_smear'],mc.mc_params['sectoral_smear'])
        fa_entrep = fa_entrep_mean*(1+fa_entrep_sectoral_smear)+fa_entrep_covariate_smear

        # 2) determine if affected
        entrep_isaff = (fa_entrep>np.random.uniform(0,1,df.shape[0]))
        if not mc.cancel_entrep_shock: df.loc[entrep_isaff,'entrep_affected'] = 1
        elif mc.nsim == 0: print('zeroing out entrepreneurial shock for all sectors (mc.cancel_entrep_shock)')

        # 3) calculate (sum) loss
        loss_vector = float(mc.entrepreneurial_shock.loc[sec,'di'])*df[['entrep_affected',sec_code]].prod(axis=1)

        # 4) add to total entrepreneurial losses
        df['entrep_loss'] += loss_vector

        # 5) store in MC
        mc.total_loss_ent.loc[mc.nsim,sec] = (loss_vector*df[['pwgt','ppp_factor']].prod(axis=1)).sum()*1E-6/12 # to mil. PPP$ and per month 
        mc.total_value_ent.loc[mc.nsim,sec] = df[[sec_code,'pwgt','ppp_factor']].prod(axis=1).sum()*1E-6/12 # to mil. PPP$ and per month
        mc.frac_loss_ent.loc[mc.nsim,sec] = float(1E2*mc.total_loss_ent.loc[mc.nsim,sec]/mc.total_value_ent.loc[mc.nsim,sec])

    # drop in international remittances
    if mc.mc_params['remits_intl_shock_mean'] > 0: 
        # has to be as individual level...therefore divide cash_abroad by hh count
        df['remits_loss_intl'] = df['cash_abroad']/df['hhsize_lfs']
        # apply smearing factor
        df['remits_loss_intl'] *= np.random.normal(mc.mc_params['remits_intl_shock_mean'],mc.mc_params['remits_intl_shock_stdev'],df.shape[0])

    # drop in domestic remittances
    if mc.mc_params['remits_dom_shock_mean'] > 0: 
        print('trying to shock domestic remittances, but this needs to be balanced in hh spending on outgoing transfers!')
        assert(False)
        # has to be as individual level...therefore divide cash_abroad by hh count
        df['remits_loss_dom'] = df['cash_domestic']/df['hhsize_lfs']
        # apply smearing factor
        df['remits_loss_dom'] *= np.random.normal(mc.mc_params['remits_shock_mean'],mc.mc_params['remits_shock_stdev'],df.shape[0])

    # total income loss
    df['income_loss'] = df[['nonag_wage_loss','ag_wage_loss','remits_loss_intl','remits_loss_dom','entrep_loss']].sum(axis=1)
    #
    return df

def sum_to_sectoral_impact(df,mc):
    # values in df are PHP/year --> switch to million PPP/month

    # WAGES
    # initialize
    sectoral_df = df[['decile','LFS_sector']].copy()
    # losses
    sectoral_df['nonag_wage_loss'] = df[['nonag_wage_loss','ppp_factor','pwgt']].prod(axis=1)*1E-6/12 # to mil PPP and per month
    sectoral_df['ag_wage_loss'] = df[['ag_wage_loss','ppp_factor','pwgt']].prod(axis=1)*1E-6/12 # to mil PPP and per month
    # total value
    sectoral_df['agri_sal'] = df[['agri_sal','frac_ag_hours','ppp_factor','pwgt']].prod(axis=1)*1E-6/12 # to mil PPP and per month
    sectoral_df['nonagri_sal'] = df[['nonagri_sal','frac_nonag_hours','ppp_factor','pwgt']].prod(axis=1)*1E-6/12 # to mil PPP and per month
    # sum to sectors
    if mc.nsim == 0: 
        sectoral_df.reset_index(drop=True).set_index(['decile','LFS_sector']).sum(level=['decile','LFS_sector']).to_csv('monte_carlo/sectoral_income_by_decile.csv')
    sectoral_df = sectoral_df.reset_index(drop=True).set_index('LFS_sector').sum(level='LFS_sector')

    # RESULTS
    # wage loss vector
    sectoral_loss = sectoral_df.drop('ag',axis=0)['nonag_wage_loss']
    sectoral_loss['ag'] = sectoral_df.loc['ag','ag_wage_loss']
    sectoral_loss['unclassified'] = sectoral_df.loc['unclassified','ag_wage_loss']
    # total sectoral value
    sectoral_value = sectoral_df.drop('ag',axis=0)['nonagri_sal']
    sectoral_value['ag'] = sectoral_df.loc['ag','agri_sal']
    sectoral_value['unclassified'] += sectoral_df.loc['unclassified','agri_sal']
    # loss fraction
    sectoral_floss = sectoral_df.drop('ag',axis=0).eval('1E2*nonag_wage_loss/nonagri_sal')
    sectoral_floss['ag'] = float(1E2*sectoral_df.loc['ag'].to_frame().T.eval('ag_wage_loss/agri_sal'))
    sectoral_floss['unclassified'] = float(1E2*sectoral_df.loc['unclassified'].to_frame().T.eval('(nonag_wage_loss+ag_wage_loss)/(nonagri_sal+agri_sal)'))

    # record in mc
    mc.total_loss_wages.iloc[mc.nsim] = sectoral_loss.T
    mc.total_value_wages.iloc[mc.nsim] = sectoral_value.T
    mc.frac_loss_wages.iloc[mc.nsim] = sectoral_floss.T

    # REMITTANCES
    mc.total_loss_remits.loc[mc.nsim,'intl'] = 1E-6/12*df[['pwgt','remits_loss_intl','ppp_factor']].prod(axis=1).sum()
    mc.total_value_remits.loc[mc.nsim,'intl'] = 1E-6/12*(df[['pwgt','cash_abroad','ppp_factor']].prod(axis=1)/df['hhsize_lfs']).sum()
    mc.total_loss_remits.loc[mc.nsim,'dom'] = 1E-6/12*df[['pwgt','remits_loss_dom','ppp_factor']].prod(axis=1).sum()
    mc.total_value_remits.loc[mc.nsim,'dom'] = 1E-6/12*(df[['pwgt','cash_domestic','ppp_factor']].prod(axis=1)/df['hhsize_lfs']).sum()
    mc.frac_loss_remits.loc[mc.nsim] = 1E2*mc.total_loss_remits.loc[mc.nsim]/mc.total_value_remits.loc[mc.nsim]

    # TOTAL LOSSES
    mc.total_value_economy.loc[mc.nsim,'loss'] = 1E-6/12*df[['pwgt','income_loss','ppp_factor']].prod(axis=1).sum()
    mc.total_value_economy.loc[mc.nsim,'income'] = 1E-6/12*(df[['pwgt','hhinc','ppp_factor']].prod(axis=1)/df['hhsize_lfs']).sum()   
    mc.total_value_economy.loc[mc.nsim,'consumption'] = 1E-6/12*(df[['pwgt','hhexp','ppp_factor']].prod(axis=1)/df['hhsize_lfs']).sum()   


def sum_to_households(df,hh_df=None):

    success=False
    while not success:

        try:
            # this code runs for every MC simulation

            ####################################################
            # All flows (income) should be PPP/monthly!
            for chan in ['nonag_wage_loss','ag_wage_loss','remits_loss_intl','remits_loss_dom','entrep_loss','income_loss']:
                hh_df[chan] = df[chan].sum(level=0)
                hh_df[chan] = (hh_df[[chan,'ppp_factor']].prod(axis=1)/12)# <-- annual to monthly PPP$
            hh_df['income_loss'] = hh_df['income_loss'].clip(upper=hh_df['hhinc'])

            hh_df['hhwgt'] = hh_df.eval('popwgt/hhsize')
            #
            hh_df['pcinc_final'] = hh_df.eval('(hhinc-income_loss)/hhsize')
            hh_df['pcexp_final'] = hh_df.eval('(hhexp-income_loss)/hhsize')
            #
            success=True

        except:
            # Code runs once per MC ensemble run
            df = df.reset_index('cc101_lno')

            ####################################################
            # these are all the income flows we want to keep
            # all will be converted to PPP per hh & month below
            hh_descriptors = ['region','prov_code','ppp_factor','hhsize','savings','quintile','decile','SAP_value']
            hh_flows = ['hhid_fies','hhinc','hhexp',
                        'nonagri_sal','agri_sal','total_entrepreneurial',
                        'cash_abroad','cash_domestic',
                        'total_public','cct4P']
            hh_df = df.loc[~(df.index.duplicated(keep='first')),hh_descriptors+hh_flows]  
            
            # These are not included in flows b/c they need to be summed at hh level, (can't just grab first instance)
            #[['net_cfg','net_lpr','net_fish','net_for','net_ret','net_mfg','net_com','net_trans','net_min','net_cons','net_nec']]

            ####################################################
            # recover weighting
            hh_df['popwgt'] = df['pwgt'].sum(level=0)*1E-6

            ####################################################
            # convert to PPP (still at household level)            
            # stock
            hh_df['savings'] = hh_df[['savings','ppp_factor']].prod(axis=1) # <-- stock, in PPP$

            # annual flows
            for istream in hh_flows: 
                hh_df[istream] = hh_df[[istream,'ppp_factor']].prod(axis=1)/12 # <-- annual to monthly PPP$

            # monthly flows
            hh_df['SAP_value'] = hh_df[['SAP_value','ppp_factor']].prod(axis=1)

            ####################################################
            # constructed metrics (PPP per cap, month)
            hh_df['pcinc_initial'] = hh_df.eval('hhinc/hhsize')
            hh_df['pcexp_initial'] = hh_df.eval('hhexp/hhsize')
            #
            #assign income cats (initial)
            m2d = 12/365
            hh_df['initial_class'] = '-'
            hh_df.loc[hh_df.eval("(initial_class=='-')&(pcinc_initial*@m2d<=1.9)"),'initial_class'] = 'sub'
            hh_df.loc[hh_df.eval("(initial_class=='-')&(pcinc_initial*@m2d<=3.2)"),'initial_class'] = 'pov'
            hh_df.loc[hh_df.eval("(initial_class=='-')&(pcinc_initial*@m2d<=5.5)"),'initial_class'] = 'vul'
            hh_df.loc[hh_df.eval("(initial_class=='-')&(pcinc_initial*@m2d<=15.0)"),'initial_class'] = 'sec'
            hh_df.loc[hh_df.eval("(initial_class=='-')&(pcinc_initial*@m2d>15.0)"),'initial_class'] = 'mc'

            # initial data: shock-independent, could be written out only once
            hh_df.reset_index().set_index('initial_class')['popwgt'].sum(level=0).to_csv('monte_carlo/inital_pop_by_class.csv')
            hh_df.loc[hh_df['initial_class']=='sub'].reset_index().set_index('region')['popwgt'].sum(level=0).to_csv('monte_carlo/regional_subsistence.csv')
            hh_df.loc[m2d*hh_df['pcinc_initial']<=3.2].reset_index().set_index('region')['popwgt'].sum(level=0).to_csv('monte_carlo/regional_poverty.csv')
            
    return hh_df

def project_consumption_series(df,mc):
    #t_sav can be 'random' or 'known'

    # crisis duration in months, so go to monthly here
    d2m = 365/12
    segments = ['zero','sub','pov','vul','sec','mc']    
    seg_def = {'zero':-1E9,'sub':1.90*d2m,'pov':3.20*d2m,'vul':5.50*d2m,'sec':15.00*d2m,'mc':1E9}

    # set savings expenditure based on function input
    if mc.savings_params['t_sav'] == 'random':
        df['t_sav'] = np.random.randint(1,4,df.shape[0])
        df['savings_expenditure'] = 0
        df.loc[df.eval('income_loss>0'),'savings_expenditure'] = df.loc[df.eval('income_loss>0')].eval('savings/t_sav')
    
        # check that affected hh don't spend more than they lost
        _slc = "(income_loss>0)&(savings_expenditure>income_loss)"
        df.loc[df.eval(_slc),'t_sav'] = df.loc[df.eval(_slc)].eval('savings/income_loss').apply(np.ceil)
        df.loc[df.eval(_slc),'savings_expenditure'] = df.loc[df.eval(_slc)].eval('savings/t_sav')

    if mc.savings_params['t_sav'] == 'known': pass

    # common denominator
    tot_pop = df['popwgt'].sum()
            
    # initialize
    consumption_time_series = pd.DataFrame({'sub':-1,'pov':-1,'vul':-1,'sec':-1,'mc':-1},index=list(range(mc.savings_params['nMonths']+1)))

    for _tc in consumption_time_series.index:

        for _nseg, _seg in enumerate(segments):

            if _nseg == 0: continue
            if mc.savings_params['t_sav'] == 'random': df.loc[df.eval('(savings_expenditure!=0)&(@_tc>t_sav)'),'savings_expenditure'] = 0
            elif mc.savings_params['t_sav'] == 'known' and _tc > 0: df['savings_expenditure'] = df.eval('savings/@_tc').clip(upper=df['income_loss'])

            _cons_t='((hhinc{})/hhsize)'.format('' if _tc == 0 else '-income_loss+savings_expenditure')
            _cons_cut = '({}>{})&({}<={})'.format(_cons_t,seg_def[segments[_nseg-1]],_cons_t,seg_def[_seg])
            consumption_time_series.loc[_tc,_seg] = 1E2*df.loc[df.eval(_cons_cut),'popwgt'].sum()/tot_pop
            
    return consumption_time_series.T
# 