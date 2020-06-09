import pandas as pd
import numpy as np
import os,glob

class monte_carlo():
    
    def __init__(self,shock_code,Nsims):

        self.classes = ['sub','pov','vul','sec','mc']
        self.shock_code = shock_code
        self.m2d = 12/365
        #
        self.ix = list(range(0,Nsims))
        # INCOME
        self.pop_aff = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.frac_aff = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.tot_loss = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.di_tot = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.di_aff = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.income_sub = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.income_pov = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        self.income_vul = pd.DataFrame(index=self.ix,columns={_:None for _ in self.classes})
        
        # CONSUMPTION TIME SERIES
        self.cts_sub = pd.DataFrame(index=self.ix,columns={_:None for _ in list(range(0,13))})
        self.cts_pov = pd.DataFrame(index=self.ix,columns={_:None for _ in list(range(0,13))})
        self.cts_vul = pd.DataFrame(index=self.ix,columns={_:None for _ in list(range(0,13))})
        self.cts_sec = pd.DataFrame(index=self.ix,columns={_:None for _ in list(range(0,13))})
        self.cts_mc = pd.DataFrame(index=self.ix,columns={_:None for _ in list(range(0,13))})

    
    def store_income_impacts_by_class(self,hh_df,nsim):
        #
        # record simulation results by income class
        for _ in self.classes:

            # subset df
            df_aff = hh_df.loc[(hh_df['initial_class']==_)&(hh_df['income_loss']>0)]

            # floats
            tot_pop = hh_df.loc[(hh_df['initial_class']==_),'popwgt'].sum()
            aff_pop = df_aff['popwgt'].sum()
            loss = df_aff.eval('popwgt*(pcinc_initial-pcinc_final)').sum()
        
            # results
            self.pop_aff.loc[nsim,_] = float(df_aff['popwgt'].sum())
            self.frac_aff.loc[nsim,_] = float(1E2*aff_pop/tot_pop)
            self.tot_loss.loc[nsim,_] = float(loss)
            self.di_tot.loc[nsim,_] = float(loss/tot_pop)
            self.di_aff.loc[nsim,_] = float(loss/aff_pop)
            #
            self.income_sub.loc[nsim,_] = float(df_aff.loc[self.m2d*df_aff['pcinc_final']<=1.90,'popwgt'].sum())
            self.income_pov.loc[nsim,_] = float(df_aff.loc[self.m2d*df_aff['pcinc_final']<=3.20,'popwgt'].sum())
            self.income_vul.loc[nsim,_] = float(df_aff.loc[self.m2d*df_aff['pcinc_final']<=5.50,'popwgt'].sum())        

    def store_consumption_time_series(self,cts,nsim):
        #
        self.cts_sub.loc[nsim] = cts.loc['sub']
        self.cts_pov.loc[nsim] = cts.loc['pov']
        self.cts_vul.loc[nsim] = cts.loc['vul']
        self.cts_sec.loc[nsim] = cts.loc['sec']
        self.cts_mc.loc[nsim] = cts.loc['mc']

    def write_income_results(self):
        #
        if not os.path.isdir('monte_carlo/{}'.format(self.shock_code)):
            os.makedirs('monte_carlo/{}'.format(self.shock_code))
        #
        self.pop_aff.to_csv('monte_carlo/{}/pop_aff.csv'.format(self.shock_code))
        self.frac_aff.to_csv('monte_carlo/{}/frac_aff.csv'.format(self.shock_code))
        self.tot_loss.to_csv('monte_carlo/{}/tot_loss.csv'.format(self.shock_code))
        self.di_tot.to_csv('monte_carlo/{}/di_tot.csv'.format(self.shock_code))
        self.di_aff.to_csv('monte_carlo/{}/di_aff.csv'.format(self.shock_code))
        self.income_sub.to_csv('monte_carlo/{}/income_sub.csv'.format(self.shock_code))
        self.income_pov.to_csv('monte_carlo/{}/income_pov.csv'.format(self.shock_code))
        self.income_vul.to_csv('monte_carlo/{}/income_vul.csv'.format(self.shock_code))

    def write_consumption_time_series(self):
        #
        if not os.path.isdir('monte_carlo/{}'.format(self.shock_code)):
            os.makedirs('monte_carlo/{}'.format(self.shock_code))
        #
        self.cts_sub.to_csv('monte_carlo/{}/time_series_subsistence.csv'.format(self.shock_code))
        self.cts_pov.to_csv('monte_carlo/{}/time_series_poverty.csv'.format(self.shock_code))
        self.cts_vul.to_csv('monte_carlo/{}/time_series_vulnerable.csv'.format(self.shock_code))
        self.cts_sec.to_csv('monte_carlo/{}/time_series_secure.csv'.format(self.shock_code))
        self.cts_mc.to_csv('monte_carlo/{}/time_series_middleclass.csv'.format(self.shock_code))


#######################
# Load results

def load_income_impacts(shock_code):
    print('returning (pop_aff,frac_aff,tot_loss,di_tot,di_aff,income_sub,income_pov,income_vul)')
    pop_aff = pd.read_csv('monte_carlo/{}/pop_aff.csv'.format(shock_code),index_col=[0])
    frac_aff = pd.read_csv('monte_carlo/{}/frac_aff.csv'.format(shock_code),index_col=[0])
    tot_loss = pd.read_csv('monte_carlo/{}/tot_loss.csv'.format(shock_code),index_col=[0])
    di_tot = pd.read_csv('monte_carlo/{}/di_tot.csv'.format(shock_code),index_col=[0])
    di_aff = pd.read_csv('monte_carlo/{}/di_aff.csv'.format(shock_code),index_col=[0])
    income_sub = pd.read_csv('monte_carlo/{}/income_sub.csv'.format(shock_code),index_col=[0])
    income_pov = pd.read_csv('monte_carlo/{}/income_pov.csv'.format(shock_code),index_col=[0])
    income_vul = pd.read_csv('monte_carlo/{}/income_vul.csv'.format(shock_code),index_col=[0])
    #
    return (pop_aff,frac_aff,tot_loss,di_tot,di_aff,income_sub,income_pov,income_vul)
  
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
