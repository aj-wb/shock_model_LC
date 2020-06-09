import pandas as pd

def get_hh_savings(myC, econ_unit, pol, fstr=None,return_regional_avg=False):
    
    # original: 
    # hh_df = pd.read_csv('~/Desktop/BANK/hh_resilience_model/intermediate/{}/cat_info.csv'.format('PH')).set_index('hhid')
    # modified for covid_phl local: <ps>20200415
    hh_df = pd.read_csv('./csv/cat_info.csv'.format('PH')).set_index('hhid')

    
    print('NB: loading cat_info from resilience model, instead of FIES in covid directory')

    # First check the policy string, in case we're doing something experimental
    #if pol == '_nosavings': return hh_df.eval('0').to_frame(name='precautionary_savings')
    #elif pol == '_nosavingsdata': return hh_df.eval('c/12').to_frame(name='precautionary_savings')
    #elif pol == '_infsavings': return hh_df.eval('1.E9').to_frame(name='precautionary_savings')


    # Now run country-dependent options:
    if myC == 'PH':

        # LOAD DECILE INFO
        df_decile = pd.read_csv('in/hh_rankings.csv')[['hhid','decile']].astype('int')
        hh_df = pd.merge(hh_df.reset_index(),df_decile.reset_index(),on='hhid')

        # LOAD SAVINGS INFO
        df_sav = pd.read_csv('csv/hh_savings_by_decile_and_region.csv').rename(columns={'w_regn':'region',
                                                                                                         'decile_reg':'decile'})
        r_code = pd.read_excel('in/FIES_regions.xlsx')[['region_code','region_name']].set_index('region_code').squeeze()
        df_sav['region'].replace(r_code,inplace=True)
        df_sav['precautionary_savings'] = df_sav['precautionary_savings'].clip(lower=0)

        hh_df = pd.merge(hh_df.reset_index(),df_sav.reset_index(),on=['region','decile'])

        if not return_regional_avg: 
            hh_df = hh_df.set_index('hhid')
            return hh_df[['precautionary_savings']]
            
        else: 
            hh_df = hh_df.set_index(['region','hhid'])
            regional_avg = hh_df[['precautionary_savings','pcwgt']].prod(axis=1).sum(level='region')/hh_df['pcwgt'].sum(level='region')
            return regional_avg

    assert(False)
