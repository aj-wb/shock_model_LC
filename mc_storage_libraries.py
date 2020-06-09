import pandas as pd
import numpy as np
import os,glob
import seaborn as sns

class monte_carlo():
    
    def __init__(self,Nsims=0,shock_code='base',
                 wage_disruption_smear=0.1,
                 wage_sector_value_smear=0.1,
                 entrep_disruption_smear=0.1,
                 sectoral_smear=0.1,
                 remits_intl_shock_mean=0.13,remits_dom_shock_mean=0.0,
                 savings_flow_to_stock_factor=1.0,
                 cancel_wage_shock=False,cancel_entrep_shock=False):
        #
        # BASICS
        self.classes = ['sub','pov','vul','sec','mc']
        self.class_labels = ['extreme\npoverty','poverty','vulnerable','secure','middle class']

        self.m2d = 12/365
        self.nsim = 0
        # aesthethic
        self.sns_pal = sns.color_palette('Set1', n_colors=9,desat=0.6)
        self.ichan_cols = {'ag_wage':self.sns_pal[2],
                           'nonag_wage':self.sns_pal[1],
                           'pub_trans':self.sns_pal[0],
                           'remits':self.sns_pal[5],
                           'dom_remits':self.sns_pal[6],
                           'entrep':self.sns_pal[3]
                           }

        # mc_params dictionary
        self.shock_code = shock_code
        self.cancel_wage_shock = cancel_wage_shock
        self.cancel_entrep_shock = cancel_entrep_shock 
        self.mc_params = {'Nsims':Nsims,
                          'wage_disruption_smear':wage_disruption_smear, # absolute (0-1)
                          'wage_sector_value_smear':wage_sector_value_smear, # stdev on normal distribution with mean 1 (0-1)
                          'entrep_disruption_smear':entrep_disruption_smear, # absolute (0-1)
                          'sectoral_smear':sectoral_smear, # relative (0-1)
                          'remits_intl_shock_mean':remits_intl_shock_mean,
                          'remits_intl_shock_stdev':2*remits_intl_shock_mean,
                          'remits_dom_shock_mean':remits_dom_shock_mean,
                          'remits_dom_shock_stdev':2*remits_dom_shock_mean,
                          'n_sigma_CI':3
                          }

        self.savings_params = {'savings_flow_to_stock_factor':savings_flow_to_stock_factor,
                               't_sav':'known',
                               'nMonths':12
                               }

        self.ESP_params = {'sp_name':'social_amelioration_program', # PhP
                           'nEligible':18., # millions
                           'eligibility_error_array':[0.0,0.25,0.5,0.75,1.0],
                           'eligerr_to_record':0.5}

        # store representation of entrepreneurial shock
        self.entrepreneurial_shock = None
        self.entrepreneurial_shock_dict = {'CSRP services':'net_com', # Community,Social,Rec'l,Personal Services net receipts (gross = 'eacpsgrs')
                                            'Construction':'net_cons', # Construction net receipts (gross = 'eacongrs')
                                            'Crop Farming and Gardening':'net_cfg', # Crop Farming and Gardening net receipts (gross = 'eacfggrs')
                                            'Entrep. Activities NEC':'net_nec', # Entrepreneurial Activities NEC net receipts (gross = 'eanecgrs')
                                            'Fishing':'net_fish', # Fishing gross receipts (gross = 'eafisgrs')
                                            'Forestry and Hunting':'net_for', # Forestry and Hunting net receipts (gross = 'eaforgrs')
                                            'Livestock and Poultry Raising':'net_lpr', # Livestock and Poultry Raising net receipts (gross = 'ealprgrs')
                                            'Manufacturing':'net_mfg', # Manufacturing net receipts (gross = 'eamfggrs')
                                            'Mining and Quarrying':'net_min', # Mining and Quarrying gross receipts (gross = eamnggrs)
                                            'Transportation, Storage and Comm. Services':'net_trans',# Trans,Storage and Comms Srvcs net receipts (gross = 'eatcsgrs')
                                            'Wholesale and Retail':'net_ret' # Wholesale and Retail Trade net receipts (gross = 'eatrdgrs')
                                            }
        self.sector_labels = {'CSRP services':'community, social, recreational\n& personal services',
                              'Mining and Quarrying':'mining & quarrying',
                              'Transportation, Storage and Comm. Services':'transportation, storage\n& communications',
                              'Livestock and Poultry Raising':'livestock & poultry',
                              'Crop Farming and Gardening':'crop farming & gardening',
                              'Wholesale and Retail':'wholesale & retail',
                              'Forestry and Hunting':'forestry & hunting',
                              'Entrep. Activities NEC':'other entrepreneurial\nactivities',
                              'ag':'agriculture',
                              'professional_services':'professional services',
                              'food_entertainment':'food & entertainment',
                              'intl':'international remittances',
                              'eduhealth':'education & healthcare',
                              'other':'other service activities'}


        # Initialize results vectors
        self.ix = list(range(0,self.mc_params['Nsims']))

        # SECTORAL IMPACTS
        self.total_value_wages = pd.DataFrame(index=self.ix) # empty columns, set in load_shock_template 
        self.total_loss_wages = pd.DataFrame(index=self.ix) # empty columns, set in load_shock_template
        self.frac_loss_wages = pd.DataFrame(index=self.ix) # empty columns, set in load_shock_template
        self.total_value_ent = pd.DataFrame(index=self.ix,columns={_:None for _ in self.entrepreneurial_shock_dict})
        self.total_loss_ent = pd.DataFrame(index=self.ix,columns={_:None for _ in self.entrepreneurial_shock_dict})
        self.frac_loss_ent = pd.DataFrame(index=self.ix,columns={_:None for _ in self.entrepreneurial_shock_dict})
        self.total_value_remits = pd.DataFrame(index=self.ix,columns={'intl':None,'dom':None})
        self.total_loss_remits = pd.DataFrame(index=self.ix,columns={'intl':None,'dom':None})
        self.frac_loss_remits = pd.DataFrame(index=self.ix,columns={'intl':None,'dom':None})
        self.total_value_economy = pd.DataFrame(index=self.ix,columns={'income':None,'consumption':None,'loss':None})

        # RESULTS by REGION
        self.regional_sub_covid = pd.DataFrame(index=self.ix)
        self.regional_sub_esp = pd.DataFrame(index=self.ix)
        self.regional_pov_covid = pd.DataFrame(index=self.ix)
        self.regional_pov_esp = pd.DataFrame(index=self.ix)

        # RESULTS by INCOME CLASS
        self.pop_aff = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.frac_aff = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})

        self.tot_loss = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.tot_inc_initial = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.tot_inc_final = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.tot_inc_initial_affected = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.tot_inc_final_affected = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        #
        self.ag_wage_loss = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.ag_wage_lossfrac = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.nonag_wage_loss = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.nonag_wage_lossfrac = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.entrep_loss = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.entrep_lossfrac = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.remits_loss = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.remits_lossfrac = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        #
        self.di_tot = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.di_aff = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        #
        self.affpop_sub = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.affpop_pov = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.affpop_vul = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})  

        self.totpop_sub = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.totpop_pov = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.totpop_vul = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})

        # SP RESULTS BY INCOME CLASS
        self.sp_adequacy = None
        self.sp_net_win = None
        self.sp_poverty = None
        self.sp_subsistence = None

        # CONSUMPTION TIME SERIES
        self.cts_sub = pd.DataFrame(index=self.ix,columns={_:None for _ in list(range(0,13))})
        self.cts_pov = pd.DataFrame(index=self.ix,columns={_:None for _ in list(range(0,13))})
        self.cts_vul = pd.DataFrame(index=self.ix,columns={_:None for _ in list(range(0,13))})
        self.cts_sec = pd.DataFrame(index=self.ix,columns={_:None for _ in list(range(0,13))})
        self.cts_mc = pd.DataFrame(index=self.ix,columns={_:None for _ in list(range(0,13))})

    def load_regions(self,index='code'):
        # index takes 'code' or 'name'
        return pd.DataFrame(index=pd.read_excel('in/FIES_regions.xlsx',index_col='region_{}'.format(index)).index)

    def store_income_impacts_by_class(self,hh_df):
        #
        # record simulation results by income class
        for _ in self.classes:

            # subset df
            df_cls = hh_df.loc[(hh_df['initial_class']==_)]
            df_aff = hh_df.loc[(hh_df['initial_class']==_)&(hh_df['income_loss']!=0)]

            # floats
            tot_pop = hh_df.loc[(hh_df['initial_class']==_),'popwgt'].sum()
            aff_pop = df_aff['popwgt'].sum()
            loss = df_aff.eval('popwgt*(pcinc_initial-pcinc_final)').sum()
        
            #########################################################
            # RESULTS, by income class

            # population affected [millions]
            self.pop_aff.loc[self.nsim,_] = float(df_aff['popwgt'].sum())
            # fraction affected [%]
            self.frac_aff.loc[self.nsim,_] = float(1E2*aff_pop/tot_pop)
            # total loss [PPP$/cap,month]
            self.tot_loss.loc[self.nsim,_] = float(loss)
            # total initial income [million PPP$/month]
            self.tot_inc_initial.loc[self.nsim,_] = float(df_cls.eval('popwgt*pcinc_initial').sum())
            # total final income [million PPP$/month]
            self.tot_inc_final.loc[self.nsim,_] = float(df_cls.eval('popwgt*pcinc_final').sum())
            # total initial income, affected only [million PPP$/month]
            self.tot_inc_initial_affected.loc[self.nsim,_] = float(df_aff.eval('popwgt*pcinc_initial').sum())
            # total final income, affected only [million PPP$/month]
            self.tot_inc_final_affected.loc[self.nsim,_] = float(df_aff.eval('popwgt*pcinc_final').sum())
            #
            # Income channels:
            # agricultural wages [million PPP$/month]
            self.ag_wage_loss.loc[self.nsim,_] = float(df_aff.eval('popwgt*(ag_wage_loss/hhsize)').sum())
            self.ag_wage_lossfrac.loc[self.nsim,_] = 1E2*self.ag_wage_loss.loc[self.nsim,_]/float(df_cls.eval('popwgt*(agri_sal/hhsize)').sum())
            # non-ag wages [million PPP$/month]
            self.nonag_wage_loss.loc[self.nsim,_] = float(df_aff.eval('popwgt*(nonag_wage_loss/hhsize)').sum())
            self.nonag_wage_lossfrac.loc[self.nsim,_] = 1E2*self.nonag_wage_loss.loc[self.nsim,_]/float(df_cls.eval('popwgt*(nonagri_sal/hhsize)').sum())
            # remittances (international ONLY) [million PPP$/month]
            self.remits_loss.loc[self.nsim,_] = float(df_aff.eval('popwgt*(remits_loss_intl/hhsize)').sum())
            self.remits_lossfrac.loc[self.nsim,_] = 1E2*self.remits_loss.loc[self.nsim,_]/float(df_cls.eval('popwgt*(cash_abroad/hhsize)').sum())
            # entrepreneurial losses [million PPP$/month]
            self.entrep_loss.loc[self.nsim,_] = float(df_aff.eval('popwgt*(entrep_loss/hhsize)').sum())
            self.entrep_lossfrac.loc[self.nsim,_] = 1E2*self.entrep_loss.loc[self.nsim,_]/float(df_cls.eval('popwgt*(total_entrepreneurial/hhsize)').sum())

            # income loss [PPP$/cap,month]
            self.di_tot.loc[self.nsim,_] = float(loss/tot_pop)
            # income loss [PPP$/(affeted cap),month]
            self.di_aff.loc[self.nsim,_] = float(loss/aff_pop)


            # AFFECTED population in subsistence during shock (incl those initially in subsistence) [millions]
            self.affpop_sub.loc[self.nsim,_] = float(df_aff.loc[self.m2d*df_aff['pcinc_final']<=1.90,'popwgt'].sum())
            # AFFECTED population in poverty (incl subsistence) during shock (incl those initially in poverty) [millions]
            self.affpop_pov.loc[self.nsim,_] = float(df_aff.loc[self.m2d*df_aff['pcinc_final']<=3.20,'popwgt'].sum())
            # AFFECTED population in vulnerability (incl pov & sub) during shock (incl those initially vulnerable) [millions]
            self.affpop_vul.loc[self.nsim,_] = float(df_aff.loc[self.m2d*df_aff['pcinc_final']<=5.50,'popwgt'].sum())
            #    

            # TOTAL population in subsistence during shock (incl those initially in subsistence) [millions]
            self.totpop_sub.loc[self.nsim,_] = float(df_cls.loc[self.m2d*df_cls['pcinc_final']<=1.90,'popwgt'].sum())
            # TOTAL population in poverty (incl subsistence) during shock (incl those initially in poverty) [millions]
            self.totpop_pov.loc[self.nsim,_] = float(df_cls.loc[self.m2d*df_cls['pcinc_final']<=3.20,'popwgt'].sum())
            # TOTAL population in vulnerability (incl pov & sub) during shock (incl those initially vulnerable) [millions]
            self.totpop_vul.loc[self.nsim,_] = float(df_cls.loc[self.m2d*df_cls['pcinc_final']<=5.50,'popwgt'].sum())
            #    

    def store_consumption_time_series(self,cts):
        #
        self.cts_sub.loc[self.nsim] = cts.loc['sub']
        self.cts_pov.loc[self.nsim] = cts.loc['pov']
        self.cts_vul.loc[self.nsim] = cts.loc['vul']
        self.cts_sec.loc[self.nsim] = cts.loc['sec']
        self.cts_mc.loc[self.nsim] = cts.loc['mc']


    def collect_regional_results(self,hh_df,ESPconfig=None):

        # Initialize storage, if necessary
        if len(self.regional_sub_covid.columns)==0:
            for _r in hh_df['region'].unique(): 
                self.regional_sub_covid[_r] = None
                self.regional_sub_esp[_r] = None
                self.regional_pov_covid[_r] = None
                self.regional_pov_esp[_r] = None


        if ESPconfig is not None: 
            self.regional_sub_esp.loc[self.nsim] = hh_df.loc[hh_df.eval('@self.m2d*(pcinc_final+'+ESPconfig+'/hhsize)<1.90')].reset_index().set_index('region')['popwgt'].sum(level='region').T
            self.regional_pov_esp.loc[self.nsim] = hh_df.loc[hh_df.eval('@self.m2d*(pcinc_final+'+ESPconfig+'/hhsize)<3.20')].reset_index().set_index('region')['popwgt'].sum(level='region').T 
        else:
            self.regional_sub_covid.loc[self.nsim] = hh_df.loc[self.m2d*hh_df['pcinc_final']<1.90].reset_index().set_index('region')['popwgt'].sum(level='region').T
            self.regional_pov_covid.loc[self.nsim] = hh_df.loc[self.m2d*hh_df['pcinc_final']<3.20].reset_index().set_index('region')['popwgt'].sum(level='region').T

        return True

    def write_out_results(self):
        #
        if not os.path.isdir('monte_carlo/{}'.format(self.shock_code)):
            os.makedirs('monte_carlo/{}'.format(self.shock_code))
        #
        # RESULTS by income class
        self.total_value_wages.to_csv('monte_carlo/{}/total_value_wages.csv'.format(self.shock_code))
        self.total_loss_wages.to_csv('monte_carlo/{}/total_loss_wages.csv'.format(self.shock_code))
        self.frac_loss_wages.to_csv('monte_carlo/{}/frac_loss_wages.csv'.format(self.shock_code))
        #
        self.total_value_ent.to_csv('monte_carlo/{}/total_value_ent.csv'.format(self.shock_code))
        self.total_loss_ent.to_csv('monte_carlo/{}/total_loss_ent.csv'.format(self.shock_code))
        self.frac_loss_ent.to_csv('monte_carlo/{}/frac_loss_ent.csv'.format(self.shock_code))        
        #
        self.total_value_remits.to_csv('monte_carlo/{}/total_value_remits.csv'.format(self.shock_code))
        self.total_loss_remits.to_csv('monte_carlo/{}/total_loss_remits.csv'.format(self.shock_code))
        self.frac_loss_remits.to_csv('monte_carlo/{}/frac_loss_remits.csv'.format(self.shock_code))
        #
        self.total_value_economy.to_csv('monte_carlo/{}/total_value_economy.csv'.format(self.shock_code))
        #
        self.pop_aff.to_csv('monte_carlo/{}/pop_aff.csv'.format(self.shock_code))
        self.frac_aff.to_csv('monte_carlo/{}/frac_aff.csv'.format(self.shock_code))
        self.tot_loss.to_csv('monte_carlo/{}/tot_loss.csv'.format(self.shock_code))
        self.tot_inc_initial.to_csv('monte_carlo/{}/tot_inc_initial.csv'.format(self.shock_code))
        self.tot_inc_final.to_csv('monte_carlo/{}/tot_inc_final.csv'.format(self.shock_code))
        self.tot_inc_initial_affected.to_csv('monte_carlo/{}/tot_inc_initial_affected.csv'.format(self.shock_code))
        self.tot_inc_final_affected.to_csv('monte_carlo/{}/tot_inc_final_affected.csv'.format(self.shock_code))

        self.ag_wage_loss.to_csv('monte_carlo/{}/ag_wage_loss.csv'.format(self.shock_code))
        self.ag_wage_lossfrac.to_csv('monte_carlo/{}/ag_wage_lossfrac.csv'.format(self.shock_code))
        self.nonag_wage_loss.to_csv('monte_carlo/{}/nonag_wage_loss.csv'.format(self.shock_code))
        self.nonag_wage_lossfrac.to_csv('monte_carlo/{}/nonag_wage_lossfrac.csv'.format(self.shock_code))
        self.remits_loss.to_csv('monte_carlo/{}/remits_loss.csv'.format(self.shock_code))
        self.remits_lossfrac.to_csv('monte_carlo/{}/remits_lossfrac.csv'.format(self.shock_code))
        self.entrep_loss.to_csv('monte_carlo/{}/entrep_loss.csv'.format(self.shock_code))
        self.entrep_lossfrac.to_csv('monte_carlo/{}/entrep_lossfrac.csv'.format(self.shock_code))

        self.di_tot.to_csv('monte_carlo/{}/di_tot.csv'.format(self.shock_code))
        self.di_aff.to_csv('monte_carlo/{}/di_aff.csv'.format(self.shock_code))
        self.affpop_sub.to_csv('monte_carlo/{}/affpop_sub.csv'.format(self.shock_code))
        self.affpop_pov.to_csv('monte_carlo/{}/affpop_pov.csv'.format(self.shock_code))
        self.affpop_vul.to_csv('monte_carlo/{}/affpop_vul.csv'.format(self.shock_code))
        self.totpop_sub.to_csv('monte_carlo/{}/totpop_sub.csv'.format(self.shock_code))
        self.totpop_pov.to_csv('monte_carlo/{}/totpop_pov.csv'.format(self.shock_code))
        self.totpop_vul.to_csv('monte_carlo/{}/totpop_vul.csv'.format(self.shock_code))

        # SP RESULTS BY INCOME CLASS
        self.sp_adequacy.to_csv('monte_carlo/{}/ESP_adequacy.csv'.format(self.shock_code))
        self.sp_net_win.to_csv('monte_carlo/{}/ESP_net_win.csv'.format(self.shock_code))
        self.sp_poverty.to_csv('monte_carlo/{}/ESP_poverty.csv'.format(self.shock_code))
        self.sp_subsistence.to_csv('monte_carlo/{}/ESP_subsistence.csv'.format(self.shock_code))

        # RESULTS BY REGION
        self.regional_sub_covid.to_csv('monte_carlo/{}/regional_sub_covid.csv'.format(self.shock_code))
        self.regional_sub_esp.to_csv('monte_carlo/{}/regional_sub_ESP_eligerr{}.csv'.format(self.shock_code,int(1E2*self.ESP_params['eligerr_to_record'])))
        self.regional_pov_covid.to_csv('monte_carlo/{}/regional_pov_covid.csv'.format(self.shock_code))
        self.regional_pov_esp.to_csv('monte_carlo/{}/regional_pov_ESP_eligerr{}.csv'.format(self.shock_code,int(1E2*self.ESP_params['eligerr_to_record'])))

        # consumption_time_series
        self.cts_sub.to_csv('monte_carlo/{}/time_series_subsistence.csv'.format(self.shock_code))
        self.cts_pov.to_csv('monte_carlo/{}/time_series_poverty.csv'.format(self.shock_code))
        self.cts_vul.to_csv('monte_carlo/{}/time_series_vulnerable.csv'.format(self.shock_code))
        self.cts_sec.to_csv('monte_carlo/{}/time_series_secure.csv'.format(self.shock_code))
        self.cts_mc.to_csv('monte_carlo/{}/time_series_middleclass.csv'.format(self.shock_code))


#######################
# Load results

def load_income_impacts(shock_code):
    print('returning (pop_aff,frac_aff,tot_loss,di_tot,di_aff,affpop_sub,affpop_pov,affpop_vul)')
    pop_aff = pd.read_csv('monte_carlo/{}/pop_aff.csv'.format(shock_code),index_col=[0])
    frac_aff = pd.read_csv('monte_carlo/{}/frac_aff.csv'.format(shock_code),index_col=[0])
    tot_loss = pd.read_csv('monte_carlo/{}/tot_loss.csv'.format(shock_code),index_col=[0])
    di_tot = pd.read_csv('monte_carlo/{}/di_tot.csv'.format(shock_code),index_col=[0])
    di_aff = pd.read_csv('monte_carlo/{}/di_aff.csv'.format(shock_code),index_col=[0])
    affpop_sub = pd.read_csv('monte_carlo/{}/affpop_sub.csv'.format(shock_code),index_col=[0])
    affpop_pov = pd.read_csv('monte_carlo/{}/affpop_pov.csv'.format(shock_code),index_col=[0])
    affpop_vul = pd.read_csv('monte_carlo/{}/affpop_vul.csv'.format(shock_code),index_col=[0])
    #
    return (pop_aff,frac_aff,tot_loss,di_tot,di_aff,affpop_sub,affpop_pov,affpop_vul)
  
def load_consumption_time_series(shock_code):
    #
    ts_sub = pd.read_csv('monte_carlo/{}/time_series_subsistence.csv'.format(shock_code),index_col=[0],dtype=np.float64).T
    ts_sub.index = ts_sub.index.astype('int')
    #
    ts_pov = pd.read_csv('monte_carlo/{}/time_series_poverty.csv'.format(shock_code),index_col=[0]).T
    ts_pov.index = ts_pov.index.astype('int')
    #
    ts_vul = pd.read_csv('monte_carlo/{}/time_series_vulnerable.csv'.format(shock_code),index_col=[0]).T
    ts_vul.index = ts_vul.index.astype('int')
    #
    ts_sec = pd.read_csv('monte_carlo/{}/time_series_secure.csv'.format(shock_code),index_col=[0]).T
    ts_sec.index = ts_sec.index.astype('int')
    #
    ts_mc = pd.read_csv('monte_carlo/{}/time_series_middleclass.csv'.format(shock_code),index_col=[0]).T
    ts_mc.index = ts_mc.index.astype('int')
    #
    print('returning time series: (sub,pov,vul,sec,mc)')
    return (ts_sub,ts_pov,ts_vul,ts_sec,ts_mc)
