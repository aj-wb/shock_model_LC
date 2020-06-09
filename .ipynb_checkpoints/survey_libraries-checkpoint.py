import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from libraries.lib_country_dir import set_directories, load_survey_data, get_places_dict
from libraries.lib_get_hh_savings import get_hh_savings
from libraries.pandas_helper import broadcast_simple
from predictive_libraries import df_to_linear_fit

from shock_libraries import get_income_shock


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


def load_hh_survey(myC,use_FIES_weights=True,troubleshoot_merge=False,):
    set_directories(myC)
    
    # load, format FIES
    #try: df = pd.read_csv('./csv/FIEScut.csv')#.set_index('hhid') # ---- commented out to enforce local dropbox path [below] <ps>20200415
    try: df = pd.read_csv('./csv/FIEScut.csv')
    except:
        df = load_survey_data(myC)#.set_index('hhid')
        print('getting FIES/LFS id for FIES')
        get_hhid_lfs(df)
        #df.to_csv('csv/_FIEScut.csv') # ---- commented out to enforce local dropbox path [below] <ps>20200415
        df.to_csv('./csv/_FIEScut.csv') 

    df['hhid'] = df['hhid'].astype(int)    
    
    # NOT using hh-level employment data from FIES
    #df = load_fies_sectoral_employment(df)
    #df = set_employment_flags(df)
        
    # load savings --> (consumption - income) in FIES, averaged at regional deciles
    hh_sav = get_hh_savings(myC,'region',pol='',return_regional_avg=False)
    df = pd.merge(df.reset_index(),hh_sav.reset_index(),on='hhid').rename(columns={'precautionary_savings':'savings'})
        
    # load Labor Force Survey
    # --> preloaded wih sectoral info, fractional contributions to wages (FIES)
    lfs = load_lfs()

    # inner merge LSF & FIES, with reporting on how many hh fail to match
    if troubleshoot_merge:
        df['dummy_FIES'] = -1
        lfs['dummy_LFS'] = -1
            
        df_merged = pd.merge(df.reset_index(drop=True),lfs.reset_index(),on='hhid_lfs',how='outer')
        df_merged['pcwgt'] /= (df_merged.groupby(['hhid_lfs'])['cc101_lno'].transform('count'))
        
        id_cols = ['region','province','w_mun','w_bgy','w_ea','w_shsn','w_hcn','prov','prov_code','mun','bgy','ea','shsn','hcn','pcwgt','pwgt']
        df_merged.loc[(df_merged.dummy_FIES!=-1)|(df_merged.dummy_LFS!=-1),id_cols].to_csv('csv/unmerged_lfs_fies.csv')

        print('NB: have only {}% of LFS population'.format(round(1E2*df_merged.pwgt.sum()/_,1)),'\n')
        assert(False)

    # merge
    lfspop = lfs.pwgt.sum()
    fiespop = df.pcwgt.sum()
    df = pd.merge(df.reset_index(drop=True),lfs.reset_index(),on='hhid_lfs',how='inner')

    # adjust weights
    if use_FIES_weights: df['pwgt'] = df['pcwgt']/(df.groupby(['hhid_lfs'])['cc101_lno'].transform('count'))
    else:
        assert(False)
        # use individual weights from LFS ('pwgt'), but scale these to match pcwgt from FIES
        df['fies_to_lfs_sf'] = df['pcwgt']/df.groupby(['hhid_lfs'])['pwgt'].transform('sum')
        df['pwgt'] = df[['fies_to_lfs_sf','pwgt']].prod(axis=1)

    print('NB: merged df has {}% of LFS pop, {}% of FIES pop\n'.format(round(1E2*df.pwgt.sum()/lfspop,1),round(1E2*df.pwgt.sum()/fiespop,1)))

    # plot hhsize vs. LFS individual count ('cc101_lno')
    if True:
        df = df.reset_index(drop=True).set_index('hhid_lfs')
        tmp_df = df.loc[~(df.index.duplicated(keep='first')),['hhsize','fsize']]
        tmp_df['lfs_personcount'] = df.reset_index().set_index(['hhid_lfs','cc101_lno'])['hhsize'].count(level=0)
        tmp_df = tmp_df.sample(int(tmp_df.shape[0]/10))
        ax = tmp_df.plot.scatter('hhsize','lfs_personcount')

        fit,fit_coef = df_to_linear_fit(tmp_df,'hhsize','lfs_personcount')
        plt.plot(tmp_df['hhsize'],fit)
        plt.annotate('fit slope = {}'.format(round(fit_coef,2)),xy=(0.05,0.9),xycoords='axes fraction',ha='left')

        ax.set_aspect('equal')
        sns.despine()
        plt.xlim(0,15)
        plt.ylim(0,15)
        plt.savefig('figs/lfs_fies_merge/hhsize.pdf',format='pdf',bbox_inches='tight')
        plt.close('all')

        df = df.reset_index()

    # cleanup
    df = df.reset_index(drop=True).set_index(['hhid_lfs','cc101_lno'])
    df = df.drop([_c for _c in ['index','level_0','index_x','hhid','hhnum','aew',
                                'w_mu','w_bgy','w_ea','w_shsn','w_hcn','w_mun'] if _c in df.columns],axis=1)

    return df.sort_index()


def load_lfs():

    lfs = pd.read_csv('2015FIES/LFSJul2015_merge.csv')   

    # get hhid for merging with FIES
    print('getting FIES/LFS id for LFS')
    get_hhid_lfs(lfs)
    
    # standardize sectoral info
    _ = int(lfs.pwgt.sum())
    lfs = load_lfs_sectoral_employment(lfs)
    assert(int(_) == int(lfs.pwgt.sum()))

    #############################################
    # fraction of nonag hours worked
    lfs['nonag_hours'] = lfs.loc[((lfs.LFS_sector!='ag')
                                  &(lfs.c19_pclass!='Employer')
                                  &(lfs.c19_pclass!='Self Employed')
                                  &(lfs.c19_pclass!='Without Pay (Family owned Business)')),'a04_thours']
    lfs['nonag_hours'].fillna(0,inplace=True)
    lfs['frac_nonag_hours'] = (lfs['nonag_hours']/((lfs.groupby(['hhid_lfs'])['nonag_hours']).transform('sum')).clip(lower=1)).fillna(0)
    # SENSITIVITY: play with skill premium here

    #############################################
    # fraction of ag hours worked
    lfs['ag_hours'] = lfs.loc[((lfs.LFS_sector=='ag')
                               &(lfs.c19_pclass!='Employer')
                               &(lfs.c19_pclass!='Self Employed')
                               &(lfs.c19_pclass!='Without Pay (Family owned Business)')),'a04_thours']
    lfs['ag_hours'].fillna(0,inplace=True)
    lfs['frac_ag_hours'] = (lfs['ag_hours']/((lfs.groupby(['hhid_lfs'])['ag_hours']).transform('sum')).clip(lower=1)).fillna(0)
    # SENSITIVITY: play with skill premium here
    
    
    lfs = lfs.reset_index(drop=True).set_index(['hhid_lfs','cc101_lno'])
    lfs = lfs.drop(['creg','prov','mun','bgy','ea',
                    'shsn','hcn','stratum','psu','psu_no','crpm','svymo',
                    'c23_pwmore','c25_pfwrk','c08_mstat','j01_usocc','c24_pladdw','c19_pclass','a04_thours',
                    'j03_okb','j04_oclass','j05_ohours','j06_obasis','j07_obasic','c38_lookw','c42_wynot',
                    'c37_avail','a07_willing','c43_lbef','svyyr','j12intvw','a06_ltlookw',
                    'c39_jobsm','c45_pocc','c40_weeks','c41_flwrk','c05_rel','c06_sex',
                    'j12c11_gradtech','j12c11course'],axis=1)
    
    return lfs

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

def load_fies_sectoral_employment(df):
    sector_dict = pd.read_csv('csv/occupations.csv').set_index('desc')['sector'].to_dict()
    df['sector'] = df['occup_fin'].replace(sector_dict)
    df = df.drop('occup_fin',axis=1)
    return df

def load_lfs_sectoral_employment(df):
    sector_dict = pd.read_csv('csv/lfs_a09_pqkb.csv').set_index('a09_pqkb')['sector'].to_dict()
    df['LFS_sector'] = df['a09_pqkb'].replace(sector_dict).fillna('none')
    df = df.drop('a09_pqkb',axis=1)
    
    # load sectoral vulnerability to shock -> [[fa,di]]
    df = pd.merge(get_income_shock().reset_index(),df.reset_index(),on='LFS_sector',how='right')
    df[['fa','di']].fillna(0,inplace=True)
    
    return df
