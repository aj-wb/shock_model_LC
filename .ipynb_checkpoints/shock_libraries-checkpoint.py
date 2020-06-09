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
#from libraries.lib_gather_data import get_hhid_FIES
#%load_ext autoreload

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

sns_pal = sns.color_palette('Set1', n_colors=8, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)

def get_income_shock():
    all_sectors = ['ag','mining','utilities','construction','manufacturing','wholesale','retail','transportation',
                   'information','finance','professional_services','eduhealth','food_entertainment','government','other']
    shutdown = {sec:[1,1] for sec in all_sectors}
    
    shock = { 'ag':           [  0,  0],
             'mining':        [  0,  0],
             'utilities':     [  0,  0],
             'construction':  [0.5,1.0],
             'manufacturing': [0.1,1.0],
             'wholesale':     [0.1,1.0],
             'retail':        [0.5,1.0],
             'transportation':[0.5,1.0],
             'information':   [0.1,1.0],
             'finance':       [0.1,1.0],
             'professional_services':[0.1,1.0],
             'eduhealth':     [0.1,1.0],
             'food_entertainment':[0.8,1.0],
             'government':    [  0,  0],
             'other':         [0.8,1.0]}
    df_shock = pd.DataFrame(data=shock).T
    df_shock.columns = ['fa','di']
    df_shock.index.name = 'LFS_sector'
    return df_shock

#(1) agriculture, forestry, fishing, and hunting – 0% affected
#(2) mining – 0% affected
#(3) utilities – 0% affected 
#(4) construction – 50% affected, income dropped by 100%
#(5) manufacturing – 10% affected, income dropped by 100%
#(6) wholesale trade - 10% affected, income dropped 100%
#(7) retail trade – 50% affected, income dropped by 100% 
#(8) transportation and warehousing – 50% affected, income dropped by 100%
#(9) information – 10% affected, income dropped 100%  
#(10) finance, insurance, real estate, rental and leasing – 10% affected, income dropped 100%
#(11) professional and business services – 10% affected, income dropped 100%
#(12) educational services, health care and social assistance – 10% affected, income dropped 100%
#(13) arts, entertainment, recreation, accommodation and food services – 80% affected, income dropped 100%
#(14) other services, except government – 80% affected, income dropped 100%

#maybe we can compare the number of people "affected" (who are here assumed to lose their job and have no 
# substitute income), and see if we are in the 20-30% ballpark (estimated unemployment in a month)

#what's trickier is the people who are losing income but not fully (like owner of small shop who has 
#20% of usual number of customer)

#pd.DataFrame({'a09_pqkb':df['a09_pqkb'].unique()}).to_csv('csv/_lfs_a09_pqkb.csv')





def load_fies_sectoral_employment(df):
    sector_dict = pd.read_csv('csv/occupations.csv').set_index('desc')['sector'].to_dict()
    df['sector'] = df['occup_fin'].replace(sector_dict)
    df = df.drop('occup_fin',axis=1)
    return df

def load_lfs_sectoral_employment(df):
    sector_dict = pd.read_csv('csv/lfs_a09_pqkb.csv').set_index('a09_pqkb')['sector'].to_dict()
    df['LFS_sector'] = df['a09_pqkb'].replace(sector_dict)
    df = df.drop('a09_pqkb',axis=1)
    
    # load sectoral vulnerability to shock -> [[fa,di]]
    df = pd.merge(get_income_shock().reset_index(),df.reset_index(),on='LFS_sector',how='right')
    df[['fa','di']].fillna(0,inplace=True)
    
    return df

def set_employment_flags(df):
    
    df['priminc'] = df['majsr'].replace({'Wage/Salaries':0,
                                         'Enterpreneurial Activities':1,
                                         'Other sources of Income':2})
    
    #pd.DataFrame(df['minsr'].unique()).to_csv('csv/_primary_income_descriptors.csv')
    pinc_dict = pd.read_csv('csv/primary_income_descriptors.csv').set_index('minsr')['desc'].to_dict()
    df['priminc_desc'] = df['minsr'].replace(pinc_dict)
    
    # 'employed_pay' = number in hh employed for pay
    # 'employed_prof' = number employed for profit
    
    df = df.drop(['majsr','minsr','job','cw','walls','roof'],axis=1)

    return df


def get_hhid_lfs(df):
    try: 
        #df['hhid'] =  (df['w_regn'].map(str)+
        df['hhid_lfs'] = (df['province'].map(str)
                          + df['w_mun'].map(str)
                          + df['w_bgy'].map(str)
                          + df['w_ea'].map(str)
                          + df['w_shsn'].map(str)
                          + df['w_hcn'].map(str)).astype(str)
    except:
        
        prov_code,region_code = get_places_dict('PH',reverse=True)
        prov_code = prov_code['province_code']
        df['prov_code'] = df['prov'].replace(prov_code,inplace=False)     
        #df['creg'].replace(region_code,inplace=True)
    
        df['hhid_lfs'] = (df['prov_code'].map(str)
                          +df['mun'].map(str)
                          +df['bgy'].map(str)
                          +df['ea'].map(str)
                          +df['shsn'].map(str)
                          +df['hcn'].map(str)).astype(str) 
        #df = df.set_index(['hhid','cc101_lno']).sort_index()
        
        if not df.index.is_unique:
            df.loc[df.index.duplicated(keep=False)].to_csv('csv/test.csv')
            assert(False)

#%autoreload
def load_hh_survey(myC):
    set_directories(myC)
    
    #try: df = pd.read_csv('./csv/FIEScut.csv')#.set_index('hhid') # ---- commented out to enforce local dropbox path [below] <ps>20200415
    try: df = pd.read_csv('./csv/FIEScut.csv')
    except:
        df = load_survey_data(myC)#.set_index('hhid')
        print('getting FIES/LFS id for FIES')
        get_hhid_lfs(df)
        #df.to_csv('csv/_FIEScut.csv') # ---- commented out to enforce local dropbox path [below] <ps>20200415
        df.to_csv('./csv/_FIEScut.csv') 
        
        
    # load, format FIES
    df = load_fies_sectoral_employment(df)
    df = set_employment_flags(df)
    df['hhid'] = df['hhid'].astype(int)    
        
    # load savings --> (consumption - income) in FIES, averaged at regional deciles
    hh_sav = get_hh_savings(myC,'region',pol='',return_regional_avg=False) 
    df = pd.merge(df.reset_index(),hh_sav.reset_index(),on='hhid').rename(columns={'precautionary_savings':'savings'})
    #df['savings'] *= 2
    # ^ SENTISIVITY will need to test for savings
        
    # load Labor Force Survey, 
    lfs = load_lfs()
    _ = int(lfs.pwgt.sum())
    lfs = load_lfs_sectoral_employment(lfs)
    assert(int(_) == int(lfs.pwgt.sum()))

    # inner merge LSF & FIES, with reporting on how many hh fail to match
    _ = lfs.pwgt.sum()

    df = pd.merge(df.reset_index(drop=True),lfs.reset_index(),on='hhid_lfs',how='inner')
    df['pcwgt'] /= (df.groupby(['hhid_lfs'])['cc101_lno'].transform('count'))#.clip(lower=1)    
    print('NB: have only {}% of LFS population'.format(round(1E2*df.pwgt.sum()/_,1)),'\n')
    
    # cleanup
    df = df.reset_index(drop=True).set_index(['hhid_lfs','cc101_lno'])
    df = df.drop([_c for _c in ['index','level_0','index_x','hhid','hhnum','aew',
                                'w_mu','w_bgy','w_ea','w_shsn','w_hcn','w_mun'] if _c in df.columns],axis=1)
    return df.sort_index()


def load_lfs():
     
    #lfs = pd.read_stata('2015FIES/LFSJul2015_merge.dta',preserve_dtypes=False)    
    lfs = pd.read_csv('2015FIES/LFSJul2015_merge.csv')   
    print('getting FIES/LFS id for LFS')
    get_hhid_lfs(lfs)
    
    # SENSITIVITY: play with skill premium here
    lfs['frac_hours_worked'] = (lfs['a04_thours']/(lfs.groupby(['hhid_lfs'])['a04_thours']).transform('sum')).fillna(0)
    
    lfs = lfs.reset_index(drop=True).set_index(['hhid_lfs','cc101_lno'])
    
    lfs = lfs.drop(['creg','prov','mun','bgy','ea',
                    'shsn','hcn','stratum','psu','psu_no','crpm','svymo',
                    'c23_pwmore','c25_pfwrk','c08_mstat','j01_usocc','c24_pladdw',
                    'j03_okb','j04_oclass','j05_ohours','j06_obasis','j07_obasic','c38_lookw','c42_wynot',
                    'c37_avail','a07_willing','c43_lbef','svyyr','j12intvw','a06_ltlookw',
                    'c39_jobsm','c45_pocc','c40_weeks','c41_flwrk','c05_rel','c06_sex',
                    'j12c11_gradtech','j12c11course'],axis=1)
    
    #lfs.to_csv('csv/lfs_slimmed.csv')
    return lfs
    
def bifurcate_households(df):
    ix_aff = pd.Index({'a','na'},name='affected_cat')
    return broadcast_simple(df,ix_aff)

def get_weight_handle(useLFS=True):
    if useLFS: return 'pwgt'
    else: return 'pcwgt'

def wgt_sanity_check(df,wgt='pwgt'):
    a = int(df[wgt].sum())
    b = int(2*df[wgt+'_shock'].sum())
    try: assert(a == b)
    except:
        print(a,'\n',b,'\n',round(1E2*b/a,2))
        assert(False)
            
def adjust_income_and_weight(df,useLFS=True):
    wgt = get_weight_handle(useLFS)
    df['wage_loss'] = 0
    
           
    # this is for bifurcated (a/na) setup
    #df.loc[df.eval("affected_cat=='na'"),wgt+'_shock'] = df.loc[df.eval("affected_cat=='na'"),wgt].copy()
    #wgt_sanity_check(df,wgt=wgt)
    #_a = "(affected_cat=='a')&(cempst1=='Employed')"
        
        
    df['affected'] = 0
        
    # ex: fa = 0.8 --> 80% chance rand < fa --> affected = True
    is_affected = df['fa']>np.random.uniform(0,1,df.shape[0])
    df.loc[is_affected,'affected'] = 1
        
    # not employed = not affected
    not_employed = "(cempst1!='Employed')"
    df.loc[df.eval(not_employed),'affected'] = 0
    
    # ^ affected includes employment (redundant)
    # frac_hours_worked & di = 0 for non-workers
    # nonagri_sal is constant for entire household
    
    df['wage_loss'] = df[['affected','frac_hours_worked','di','nonagri_sal']].prod(axis=1)
    # income loss ANNUAL here
        
    return df


def sum_to_households(df,hh_df=None):
    success=False

    while not success:
        try:
            # All flows (income) should be annual!
            #df = df.reset_index().set_index(['hhid_lfs','cc101_lno'])
            hh_df[['wage_loss','popwgt']] = df[['wage_loss','pwgt']].sum(level=0)
            hh_df['wage_loss']  = hh_df[['wage_loss','ppp_factor']].prod(axis=1)/12 # <-- annual to monthly PPP$
            hh_df['popwgt'] *= 1E-6
            #
            hh_df['pcinc_final'] = hh_df.eval('(hhinc-wage_loss)/hhsize')
            hh_df['pcexp_final'] = hh_df.eval('(hhexp-wage_loss)/hhsize')
            success=True

        except:
            df = df.reset_index('cc101_lno')
            # this is slow, so do it once
            hh_df = df.loc[~(df.index.duplicated(keep='first')),['hhinc','hhexp','ppp_factor','hhsize','savings',
                                                                 'total_public','cct4P','quintile','decile']]  
            
            # scale --> Should put everything in PPP/cap?
            hh_df['savings']      = hh_df[['savings','ppp_factor']].prod(axis=1) # <-- stock, in PPP$
            #
            hh_df['hhinc']        = hh_df[['hhinc','ppp_factor']].prod(axis=1)/12 # <-- annual to monthly PPP$
            hh_df['hhexp']        = hh_df[['hhexp','ppp_factor']].prod(axis=1)/12 # <-- annual to monthly PPP$
            #
            hh_df['total_public'] = hh_df[['total_public','ppp_factor']].prod(axis=1)/12 # <-- annual to monthly PPP$
            hh_df['cct4P']        = hh_df[['cct4P','ppp_factor']].prod(axis=1)/12 # <-- annual to monthly PPP$
            # 

            # results
            # PPP per cap, day
            hh_df['pcinc_initial'] = hh_df.eval('hhinc/hhsize')
            hh_df['pcexp_initial'] = hh_df.eval('hhinc/hhsize')

    return hh_df


def calculate_summary_stats(df,t_sav):

    # t_sav in months, so go to monthly here
    d2m = 365/12

    segments = ['zero','sub','pov','vul','sec','mc']
    seg_def = {'zero':-1E9,'sub':1.90*d2m,'pov':3.20*d2m,'vul':5.50*d2m,'sec':15.00*d2m,'mc':1E99}

    # set savings expenditure based on function input
    df['t_sav'] = t_sav
    df['savings_expenditure'] = df.eval('savings/t_sav')
    
    # check that affected hh don't spend more than they lost
    _slc = "(wage_loss!=0)&(savings_expenditure>wage_loss)"
    df.loc[df.eval(_slc),'t_sav'] = df.loc[df.eval(_slc)].eval('savings/wage_loss').apply(np.ceil)
    df.loc[df.eval(_slc),'savings_expenditure'] = df.loc[df.eval(_slc)].eval('savings/t_sav')

    # check that unaffected hh don't spend savings
    _slc = "(wage_loss==0)"
    df.loc[df.eval(_slc),['t_sav','savings_expenditure']] = 0,0

    # common denominator
    tot_pop = df['popwgt'].sum()
            
    # initialize
    income_time_series = pd.DataFrame({'sub':-1,'pov':-1,'vul':-1,'sec':-1,'mc':-1},index=list(range(0,13)))

    for _nseg, _seg in enumerate(segments):
        if _nseg == 0: continue
            
        for _tc in income_time_series.index:
            
            df.loc[df.eval('@_tc>t_sav'),'savings_expenditure'] = 0
            _='((hhinc{})/hhsize)'.format('' if _tc == 0 else '-wage_loss+savings_expenditure')
            _cut = '({}>{})&({}<={})'.format(_,seg_def[segments[_nseg-1]],_,seg_def[_seg])

            income_time_series.loc[_tc,_seg] = 1E2*df.loc[df.eval(_cut),'popwgt'].sum()/tot_pop
            
    #df.to_csv('csv/test.csv')  
    #income_time_series.to_csv('csv/its{}.csv'.format(t_sav))
    return income_time_series.T


def append_simulation(sim,series,nsim,tot_sim):
    try: sub,pov,vul,sec,mc = series
    except:
        _ix = list(range(0,tot_sim))
        _cols = {_:None for _ in list(range(0,13))}
        sub=pd.DataFrame(index=_ix,columns=_cols)
        pov=pd.DataFrame(index=_ix,columns=_cols)
        vul=pd.DataFrame(index=_ix,columns=_cols)
        sec=pd.DataFrame(index=_ix,columns=_cols)
        mc=pd.DataFrame(index=_ix,columns=_cols)
    
    sub.loc[nsim] = sim.loc['sub']
    pov.loc[nsim] = sim.loc['pov']
    vul.loc[nsim] = sim.loc['vul']
    sec.loc[nsim] = sim.loc['sec']
    mc.loc[nsim] = sim.loc['mc']
    return (sub,pov,vul,sec,mc)

def write_results(series):
    sub,pov,vul,sec,mc = series
    
    sub.to_csv('monte_carlo/sub.csv')
    pov.to_csv('monte_carlo/pov.csv')
    vul.to_csv('monte_carlo/vul.csv')
    sec.to_csv('monte_carlo/sec.csv')
    mc.to_csv('monte_carlo/mc.csv')

def load_series():
    
    sub = pd.read_csv('monte_carlo/sub.csv',index_col=[0],dtype=np.float64).T
    sub.index = sub.index.astype('int')
    #
    pov = pd.read_csv('monte_carlo/pov.csv',index_col=[0]).T
    pov.index = pov.index.astype('int')
    #
    vul = pd.read_csv('monte_carlo/vul.csv',index_col=[0]).T
    vul.index = vul.index.astype('int')
    #
    sec = pd.read_csv('monte_carlo/sec.csv',index_col=[0]).T
    sec.index = sec.index.astype('int')
    #
    mc = pd.read_csv('monte_carlo/mc.csv',index_col=[0]).T
    mc.index = mc.index.astype('int')
    #
    return (sub,pov,vul,sec,mc)
