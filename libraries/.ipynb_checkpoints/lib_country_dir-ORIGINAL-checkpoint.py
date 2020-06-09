import json
import matplotlib.pyplot as plt
import os, glob
import pandas as pd
import seaborn as sns

from libraries.lib_drought import *
from libraries.lib_gather_data import *
from libraries.plot_hist import plot_simple_hist
from libraries.pandas_helper import categorize_strings
#
# country-specific
from libraries.lib_RO_housing_plots import housing_plots

pd.set_option('display.width', 220)
sns.set_style('whitegrid')
sns_pal = sns.color_palette('Set1', n_colors=8, desat=.5)

global model
model = os.getcwd()

# People/hh will be affected or not_affected, and helped or not_helped
affected_cats = pd.Index(['a', 'na'], name='affected_cat') # categories for social protection
helped_cats   = pd.Index(['helped','not_helped'], name='helped_cat')

# These parameters could vary by country
reconstruction_time = 3.00 # time needed for reconstruction
reduction_vul       = 0.20 # how much early warning reduces vulnerability
inc_elast           = 1.50 # income elasticity (of capital?)
max_support         = 0.05 # max expenditure on PDS as fraction of GDP (set to 5% based on PHL)
nominal_asset_loss_covered_by_PDS = 0.80 # also, elsewhere, called 'shareable'

# Dictionary for standardizing HIES column names that is SPECIFIC TO ROMANIA
hbs_dict = {'regiune':'Region','judet':'County','codla':'hhcode','coefj':'hhwgt','cpers':'hhsize','R44':'hhinc',
            'REGIUNE':'Region','JUDET':'County','CODLA':'hhcode','COEFJ':'hhwgt','CPERS':'hhsize',
            'NRGL':'nrgl','MEDIU':'mediu','CENTRA':'centra'}

# Define directories
def set_directories(myCountry):  # get current directory
    """Sets global directories for all functions in the library, so set_directories
    needs to be run before other functions here

    Parameters
    ----------
    myCountry : ISO code for country

    Returns
    -------
    str
        intermediates relative folder path
    """
    global inputs, intermediate
    inputs        = '~/Desktop/BANK/hh_resilience_model/inputs/'+myCountry+'/'       # get inputs data directory
    intermediate  = '~/Desktop/BANK/hh_resilience_model/intermediate/'+myCountry+'/' # get outputs data directory

    return True

def get_economic_unit(myC):
    eca_countries = ['AM','BG','TR','HR','GR','GE','RO','AL']
    if myC in eca_countries: return 'Region'
    if myC == 'PH': return 'region'#'province'
    if myC == 'FJ': return 'Division'#'tikina'
    if myC == 'SL': return 'district'
    if myC == 'MW': return 'district'
    if myC == 'BO': return 'departamento'
    assert(False)

def get_currency(myC):
    """Dictionary lookup of currency, multiplier, and exchange rate from ISO key"""
    d = {'PH': ['b. PhP',1.E9,1./50.],
         'FJ': ['k. F\$',1.E3,1./2.],
         'SL': ['LKR',1.E9,1./150.],
         'MW': ['MWK',1.E9,1./724.64],
         'BO': ['BoB',1.E9,1./6.9],
         'AM': ['PPP',1E0,1.],
         'AL': ['PPP',1E0,1.],
         'HR': ['PPP',1E0,1.],
         'BG': ['PPP',1E0,1.],
         'RO': ['PPP',1.E9,1/4.166667],
         'TR': ['PPP',1E0,1.],
         'GE': ['PPP',1E0,1.],
         'GR': ['PPP',1E0,1.]
         }
    try:
        return d[myC]
    except KeyError:
        return ['XXX',1E0,1]

def get_hhid_elements(myC):
    if myC == 'RO': return ['Region','County','centra','hhcode','nrgl','mediu']
    return None

def get_places(myC):
    """Returns a df with economic unit as key and the population per economic unit.

    Country Notes
    -------------
    For SL, in addition to calculating population by district which is
    sum of psus in household * household weights, this function also saves
    the household weights, sizes, and # children, religion, ethnicity to a
    csv for future use.

    Parameters
    ----------
    myC : str
        ISO Country reference

    Returns
    -------
    df : DataFrame
        economic unit is the index
        population is the first and only column.

    """
    economy = get_economic_unit(myC)

    if myC == 'PH':
        df_prov = pd.read_excel('~/Desktop/BANK/hh_resilience_model/inputs/PH/population_2015.xlsx',sheet_name='population').set_index('province').rename(columns={'population':'psa_pop'})
        df_reg = pd.read_csv(inputs+'prov_to_reg_dict.csv').set_index('region')
        df_reg = pd.merge(df_prov,df_reg,left_index=True,right_on='province').sum(level='region')[['psa_pop']]

        return df_reg



    else: return None

def get_places_dict(myC,reverse=False):
    """Get economy-level names of provinces or districts (p-code)
    and larger regions if available (r-code)
    Parameters
    ----------
    myC : str
        ISO

    Returns
    -------
    p_code : Series
        Series, province/district code as index, province/district names as values.
    r_code : Series
        region code as index, region names as values.
    """

    p_code,r_code = None,None

    if myC == 'PH':
        p_code = pd.read_excel('in/FIES_provinces.xlsx')[['province_code','province_AIR']].set_index('province_code').squeeze()
        #p_code[97] = 'Zamboanga del Norte'
        #p_code[98] = 'Zamboanga Sibugay'
        if reverse: p_code = p_code.reset_index().set_index('province_AIR')

        r_code = pd.read_excel('in/FIES_regions.xlsx')[['region_code','region_name']].set_index('region_code').squeeze()
        if reverse: r_code = r_code.reset_index().set_index('region_name')

    try: p_code = p_code.to_dict()
    except: pass
    try: r_code = r_code.to_dict()
    except: pass

    return p_code,r_code

def load_survey_data(myC):
    df = None
    #Each survey/country should have the following:
    # -> hhid household id
    # -> hhinc household income? but seems to be expenditure (SL)
    # -> pcinc household income per person
    # -> hhwgt number of households this line is 'representative' of
    # -> pcwgt total population this line is representative of
    # -> hhsize household size
    # -> hhsize_ae household size2
    # -> hhsoc social payments (government and remittances)
    # -> pcsoc per person social payments
    # -> ispoor
    # -> has_ew

    if myC == 'PH':
        #path = '2015FIES/fies2015_complete.csv'
        path = '~/Desktop/BANK/hh_resilience_model/inputs/PH/FIES2015.csv'
        df = pd.read_csv(path)[['w_regn','w_prov','w_mun','w_bgy','w_ea','w_shsn','w_hcn',
                                'walls','roof',
                                'totex','cash_abroad','regft',
                                'hhwgt','poorhh','totdis','tothrec','pcinc_s','pcinc_ppp11','pcwgt',
                                'agri_sal','nonagri_sal','cash_abroad','cash_domestic','othin',
                                't930220',# total public receipts
                                't930221',# cct incl 4Ps transfers
                                #'eacfggrs',# Crop Farming and Gardening gross receipts        
                                #'ealprgrs',# Livestock and Poultry Raising gross receipts        
                                #'eafisgrs',# Fishing gross receipts        
                                #'eaforgrs',# Forestry and Hunting gross receipts        
                                'eatrdgrs',# Wholesale and Retail Trade gross receipts        
                                'eamfggrs',# Manufacturing gross receipts        
                                'eacpsgrs',# Community,Social,Rec'l,Personal Services gross receipts        
                                'eatcsgrs',# Transportation,Storage and Comcn Services gross receipts        
                                'eamnggrs',# Mining and Quarrying gross receipts   
                                'net_cons',
                                'eacongrs',# Construction gross receipts       
                                'eanecgrs',# Entrepreneurial Activities NEC gross receipts  
                                'eainc',   # Total Income from Entrepreneurial Activites
                                'job',# Household Head Job or Business Indicator  (2nd visit only)
                                'occup_fin',# Household Head Occupation  (2nd visit only)
                                'employed_pay',# Total number of family members employed for pay   (2nd visit only)
                                'employed_prof',# Total number of family members employed for profit   (2nd visit only)
                                'job',# Household Head Job or Business Indicator  (2nd visit only)
                                'cw', # Household Head Class of Worker  (2nd visit only)
                                'spouse_emp', #Spouse has job/business   (2nd visit only)
                                'majsr',# Major Grouping of Main Source of Income        
                                'minsr',# Detailed Grouping of Main Source of income 
                                'radio_qty','tv_qty','cellphone_qty','pc_qty',
                                'savings','invest',
                                ]]

        df = df.rename(columns={'tothrec':'hhsoc','poorhh':'ispoor','totex':'hhexp',
                                't930220':'total_public','t930221':'cct4P'})

        df['hhsize']     = df['pcwgt']/df['hhwgt']
        #df['hhsize_ae']  = df['pcwgt']/df['hhwgt']
        #df['aewgt']   = df['pcwgt'].copy()

        # Per capita expenditures
        df['pcexp'] = df['hhexp']/df['hhsize']

        # These lines use income as income
        df = df.rename(columns={'pcinc_s':'pcinc'})
        df['hhinc'] = df[['pcinc','hhsize']].prod(axis=1)

        # These lines use disbursements as proxy for income
        #df = df.rename(columns={'totdis':'hhinc'})
        #df['pcinc'] = df['hhinc']/df['hhsize']
        #df['aeinc']   = df['pcinc'].copy()

        df['ppp_factor'] = df.eval('(365*pcinc_ppp11*hhsize)/hhinc')# <-- annual PPP/LCU

        df['pcsoc']  = df['hhsoc']/df['hhsize']

        #df['tot_savings'] = df[['savings','invest']].sum(axis=1,skipna=False)
        df['savings'] = df['savings'].fillna(-1)
        df['invest'] = df['invest'].fillna(-1)

        df['axfin']  = 0
        df.loc[(df.savings>0)|(df.invest>0),'axfin'] = 1

        df['est_sav'] = df[['axfin','pcinc']].prod(axis=1)/2.

        #df['has_ew'] = df[['radio_qty','tv_qty','cellphone_qty','pc_qty']].sum(axis=1).clip(upper=1)
        #df = df.drop(['radio_qty','tv_qty','cellphone_qty','pc_qty'],axis=1)

        _mc_lo,_mc_hi = get_middleclass_range('PH')
        df['ismiddleclass'] = (df.pcinc>=_mc_lo)#&(df.pcinc<=_mc_hi)

        _lo,_hi = get_secure_range('PH')
        df['issecure'] = (df.pcinc>=_lo)&(df.pcinc<=_hi)

        _lo,_hi = get_vulnerable_range('PH')
        df['isvulnerable'] = (df.pcinc>=_lo)&(df.pcinc<=_hi)

        # Run savings script
        df['country'] = 'PH'
        listofquintiles=np.arange(0.10, 1.01, 0.10)
        df = df.reset_index().groupby('country',sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.pcinc),reshape_data(x.pcwgt),listofquintiles),
                                                                                            'decile_nat',sort_val='pcinc')).drop(['index'],axis=1)
        df = df.reset_index().groupby('w_regn',sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.pcinc),reshape_data(x.pcwgt),listofquintiles),
                                                                                           'decile_reg',sort_val='pcinc')).drop(['index'],axis=1)
        df = df.reset_index().set_index(['w_regn','decile_nat','decile_reg']).drop('index',axis=1)

        df['precautionary_savings'] = df['pcinc']-df['pcexp']

        # Savings rate by national decile
        _ = pd.DataFrame(index=df.sum(level='decile_nat').index)
        _['income'] = df[['pcinc','pcwgt']].prod(axis=1).sum(level='decile_nat')/df['pcwgt'].sum(level='decile_nat')
        _['expenditures'] = df[['pcexp','pcwgt']].prod(axis=1).sum(level='decile_nat')/df['pcwgt'].sum(level='decile_nat')
        _['precautionary_savings'] = _['income']-_['expenditures']
        _.sort_index().to_csv('csv/hh_savings_by_decile.csv')

        # Savings rate by decile (regionally-defined) & region
        _ = pd.DataFrame(index=df.sum(level=['w_regn','decile_reg']).index)
        _['income'] = df[['pcinc','pcwgt']].prod(axis=1).sum(level=['w_regn','decile_reg'])/df['pcwgt'].sum(level=['w_regn','decile_reg'])
        _['expenditures'] = df[['pcexp','pcwgt']].prod(axis=1).sum(level=['w_regn','decile_reg'])/df['pcwgt'].sum(level=['w_regn','decile_reg'])
        _['precautionary_savings'] = _['income']-_['expenditures']
        _.sort_index().to_csv('csv/hh_savings_by_decile_and_region.csv')

        # Savings rate for hh in subsistence (natl average)
        listofquartiles=np.arange(0.25, 1.01, 0.25)
        df = df.reset_index().groupby('country',sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.precautionary_savings),reshape_data(x.pcwgt),listofquartiles),
                                                                                            'nat_sav_quartile',sort_val='precautionary_savings'))
        df = df.reset_index().groupby('w_regn',sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.precautionary_savings),reshape_data(x.pcwgt),listofquartiles),
                                                                                           'reg_sav_quartile',sort_val='precautionary_savings')).drop(['index'],axis=1)
        df = df.reset_index().set_index(['w_regn','decile_nat','decile_reg']).drop('index',axis=1).sort_index()

        _ = pd.DataFrame()
        _.loc['subsistence_savings_rate','hh_avg'] = (df.loc[df.pcinc<get_subsistence_line(myC)].eval('pcwgt*(pcinc-pcexp)').sum()
                                                      /df.loc[df.pcinc<get_subsistence_line(myC),'pcwgt'].sum())
        _.loc['subsistence_savings_rate','hh_q1'] = df.loc[df.nat_sav_quartile==1,'precautionary_savings'].max()
        _.loc['subsistence_savings_rate','hh_q2'] = df.loc[df.nat_sav_quartile==2,'precautionary_savings'].max()
        _.loc['subsistence_savings_rate','hh_q3'] = df.loc[df.nat_sav_quartile==3,'precautionary_savings'].max()


        _.sort_index().to_csv('csv/hh_savings_in_subsistence_natl.csv')

        # Savings rate for hh in subsistence (by region)
        _ = pd.DataFrame()
        _['hh_avg'] = (df.loc[df.pcinc<get_subsistence_line(myC)].eval('pcwgt*(pcinc-pcexp)').sum(level='w_regn')
                       /df.loc[df.pcinc<get_subsistence_line(myC),'pcwgt'].sum(level='w_regn'))
        _['hh_q1'] = df.loc[df.reg_sav_quartile==1,'precautionary_savings'].max(level='w_regn')
        _['hh_q2'] = df.loc[df.reg_sav_quartile==2,'precautionary_savings'].max(level='w_regn')
        _['hh_q3'] = df.loc[df.reg_sav_quartile==3,'precautionary_savings'].max(level='w_regn')
        _.sort_index().to_csv('csv/hh_savings_in_subsistence_reg.csv')

        if False:
            _.plot.scatter('income','expenditures')
            plt.gcf().savefig('figs/income_vs_exp_by_decile_PH.pdf',format='pdf')
            plt.cla()

            _.plot.scatter('income','precautionary_savings')
            plt.gcf().savefig('figs/net_income_vs_exp_by_decile_PH.pdf',format='pdf')
            plt.cla()

            df.boxplot(column='aprecautionary_savings',by='decile')
            plt.ylim(-1E5,1E5)
            plt.gcf().savefig('figs/net_income_by_exp_decile_boxplot_PH.pdf',format='pdf')
            plt.cla()

        # Rename one column to be 'c' --> consumption
        print('Using per cap income as c')
        df['c'] = df['pcinc'].copy()

        # Drop unused columns
        df = df.reset_index().set_index(['w_regn','w_prov','w_mun','w_bgy','w_ea','w_shsn','w_hcn'])
        df = df.drop([_c for _c in ['country','decile_nat','decile_reg','est_sav','tot_savings','savings','invest',
                                    'precautionary_savings','index','level_0','cash_domestic'] if _c in df.columns],axis=1)

        # Standardize province info
        prov_code,region_code = get_places_dict(myC)

        df = df.reset_index()
        get_hhid_FIES(df)
        df = df.rename(columns={'w_prov':'province','w_regn':'region'}).reset_index()
        df['province'].replace(prov_code,inplace=True)     
        df['region'].replace(region_code,inplace=True)
        df = df.reset_index().set_index(get_economic_unit(myC)).drop(['index','level_0'],axis=1)
        #
        #print(df.head())
        #assert(False)
        #
        #get_hhid_FIES(cat_info)
        #cat_info = cat_info.rename(columns={'w_prov':'province','w_regn':'region'}).reset_index()
        #cat_info['province'].replace(prov_code,inplace=True)     
        #cat_info['region'].replace(region_code,inplace=True)
        #cat_info = cat_info.reset_index().set_index(economy).drop(['index','level_0'],axis=1)
        
        # There's no region info in df--put that in...
        #df = df.reset_index().set_index('province')
        #cat_info = cat_info.reset_index().set_index('province')
        #df['region'] = cat_info[~cat_info.index.duplicated(keep='first')].region
        
        #try: df.reset_index()[['province','region']].to_csv('../inputs/PH/prov_to_reg_dict.csv',header=True)
        #except: print('Could not update regional-provincial dict')
        
        # Manipulate PSA (non-FIES) dataframe
        #df = df.reset_index().set_index(economy)
        #df['psa_pop'] = df.sum(level=economy)
        #df = df.mean(level=economy)


    # Assing weighted household consumption to quintiles within each province
    print('Finding quintiles')
    economy = df.index.names[0]
    listofquintiles=np.arange(0.20, 1.01, 0.20)
    # groupby apply takes each economy and then applies the function separately to each economy.
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.core.groupby.GroupBy.apply.html
    # Finds quintiles by district
    df = df.reset_index().groupby(economy,sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.pcwgt),listofquintiles),'quintile'))

    print('Finding deciles')
    # finds deciles by district
    listofdeciles = np.arange(0.10, 1.01, 0.10)
    df = df.reset_index().groupby(economy,sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.pcwgt),listofdeciles),'decile'))
    # drop extraneous columns
    df.drop([icol for icol in ['level_0','index','pctle_05','pctle_05_nat'] if icol in df.columns],axis=1,inplace=True)

    # Last thing: however 'c' was set (income or consumption), pcsoc can't be higher than 0.99*that!
    df['pcsoc'] = df['pcsoc'].clip(upper=0.99*df['c'])
    
    return df

def get_df2(myC):
    if myC == 'PH':
        df2 = pd.read_excel('~/Desktop/BANK/hh_resilience_model/inputs/PH/PSA_compiled.xlsx',skiprows=1)[['province','gdp_pc_pp','pop','shewp','shewr']].set_index('province')
        df2['gdp_pp'] = df2['gdp_pc_pp']*df2['pop']
        return df2
    else: return None

def get_vul_curve(myC,struct):
    """Get vulnerability of materials used in construction.

    Parameters
    ----------
    myC : str
        ISO2 of country
    struct : str
        which building structure? Sheet names in vulnerrability curves in xlsx.

    Returns
    -------
    df
        df with two columns -
            desc:   key used in census for housing material

    """

    df = None

    if myC == 'PH':
        df = pd.read_excel('~/Desktop/BANK/hh_resilience_model/inputs/PH/vulnerability_curves_FIES.xlsx',sheet_name=struct)[['desc','v']]

    return df

def get_infra_stocks_data(myC):
    if myC == 'FJ':
        infra_stocks = pd.read_csv(inputs+'infra_stocks.csv',index_col='sector')
        return infra_stocks
    else:return None

def get_wb_or_penn_data(myC):
    #iso2 to iso3 table
    names_to_iso2 = pd.read_csv(inputs+'names_to_iso.csv')[['iso2','country']].drop_duplicates().set_index('country').squeeze()
    K = pd.read_csv(inputs+'avg_prod_k_with_gar_for_sids.csv',index_col='Unnamed: 0')
    wb = pd.read_csv(inputs+'wb_data.csv',index_col='country')
    wb['Ktot'] = wb.gdp_pc_pp*wb['pop']/K.avg_prod_k
    wb['GDP'] = wb.gdp_pc_pp*wb['pop']
    wb['avg_prod_k'] = K.avg_prod_k

    wb['iso2'] = names_to_iso2
    return wb.set_index('iso2').loc[myC,['Ktot','GDP','avg_prod_k']]

def get_rp_dict(myC):
    return pd.read_csv(inputs+"rp_dict.csv").set_index("old_rp").new_rp

def get_infra_destroyed(myC,df_haz):

    #print(get_infra_stocks_data(myC))

    infra_stocks = get_infra_stocks_data(myC).loc[['transport','energy','water'],:]
    infra_stocks['infra_share'] = infra_stocks.value_k/infra_stocks.value_k.sum()

    print(infra_stocks)

    hazard_ratios_infra = broadcast_simple(df_haz[['frac_inf','frac_destroyed_inf']],infra_stocks.index)
    hazard_ratios_infra = pd.merge(hazard_ratios_infra.reset_index(),infra_stocks.infra_share.reset_index(),on='sector',how='outer').set_index(['Division','hazard','rp','sector'])
    hazard_ratios_infra['share'] = hazard_ratios_infra['infra_share']*hazard_ratios_infra['frac_inf']

    transport_losses = pd.read_csv(inputs+"frac_destroyed_transport.csv").rename(columns={"ti_name":"Tikina"})
    transport_losses['Division'] = (transport_losses['tid']/100).astype('int')
    prov_code,_ = get_places_dict(myC)
    rp_dict   = get_rp_dict(myC)
    transport_losses['Division'] = transport_losses.Division.replace(prov_code)
    #sums at Division level to be like df_haz
    transport_losses = transport_losses.set_index(['Division','hazard','rp']).sum(level=['Division','hazard','rp'])
    transport_losses["frac_destroyed"] = transport_losses.damaged_value/transport_losses.value
    #if there is no result in transport_losses, use the PCRAFI data (from df_haz):
    transport_losses = pd.merge(transport_losses.reset_index(),hazard_ratios_infra.frac_destroyed_inf.unstack('sector')['transport'].to_frame(name="frac_destroyed_inf").reset_index(),on=['Division','hazard','rp'],how='outer')
    transport_losses['frac_destroyed'] = transport_losses.frac_destroyed.fillna(transport_losses.frac_destroyed_inf)
    transport_losses = transport_losses.set_index(['Division','hazard','rp'])

    hazard_ratios_infra = hazard_ratios_infra.reset_index('sector')
    hazard_ratios_infra.ix[hazard_ratios_infra.sector=='transport','frac_destroyed_inf'] = transport_losses["frac_destroyed"]
    hazard_ratios_infra = hazard_ratios_infra.reset_index().set_index(['Division','hazard','rp','sector'])

    return hazard_ratios_infra.rename(columns={'frac_destroyed_inf':'frac_destroyed'})

def get_service_loss(myC):
    if myC == 'FJ':
        service_loss = pd.read_csv(inputs+'service_loss.csv').set_index(['hazard','rp'])[['transport','energy','water']]
        service_loss.columns.name='sector'
        a = service_loss.stack()
        a.name = 'cost_increase'
        infra_stocks = get_infra_stocks_data(myC).loc[['transport','energy','water'],:]
        service_loss = pd.merge(pd.DataFrame(a).reset_index(),infra_stocks.e.reset_index(),on=['sector'],how='outer').set_index(['sector','hazard','rp'])
        return service_loss
    else:return None


def get_poverty_line(myC,by_district=True,sec=None):
    """Get poverty line either as a Series (if by_district is True)
    or as a float (if by_district is False).

    Parameters
    ----------
    myC : str
        ISO of country
    by_district : bool
        use a district poverty line, else, use a national level poverty line
    sec : str
        data may have urban or rural poverty lines, instead of by district.
        for the countries that need this, it should return a float.

    Returns
    -------
    Series/float
        poverty lines either by district (if series), or float (if not)

    """

    pov_line = 0.0

    if myC == 'PH': pov_line = 22302.6775#21240.2924
 
    return pov_line

def get_middleclass_range(myC):
    if myC in ['AL','AM','BG','HR','GE','GR','RO','TR']:
        _pl = get_poverty_line(myC)
        _lower = _pl*(15/5.5)
        _upper = _pl*(45/5.5)
        #_upper = 

    elif myC == 'PH':
        _pl = get_poverty_line(myC)
        _lower = _pl*(15/3.2)
        _upper = 0#_pl*(50/1.90)   

    else: assert(False)
    return(_lower,_upper)

def get_secure_range(myC):
    if myC == 'PH':
        _pl = get_poverty_line(myC)
        _lower = _pl*(5.5/3.2)
        _upper = _pl*(15/3.2)   

    else: assert(False)
    return(_lower,_upper)

def get_vulnerable_range(myC):
    if myC == 'PH':
        _pl = get_poverty_line(myC)
        _lower = _pl*(3.2/3.2)
        _upper = _pl*(5.5/3.2)   

    else: assert(False)
    return(_lower,_upper)



def get_subsistence_line(myC):

    if myC == 'PH': return 14832.0962*(22302.6775/21240.2924)
    else:
        print('No subsistence info. Returning False')
        return False

def get_to_USD(myC):
    eca_countries = ['AM','BG','TR','HR','GR','GE','RO','AL']
    if myC in eca_countries: return  1.0

    if myC == 'PH': return 50.70
    if myC == 'FJ': return 2.01
    if myC == 'SL': return 153.76
    if myC == 'MW': return 720.0
    if myC == 'RO': return 4.0
    if myC == 'BO': return 6.93
    if myC == 'AM': return 477
    assert(False)

def get_pop_scale_fac(myC):
    #if myC == 'PH' or myC == 'FJ' or myC == 'MW' or myC == 'SL':
    return [1.E3,' (,000)']

def get_avg_prod(myC):
    """Returns values from the global resilience model for the average productivity of capital"""
    if myC == 'PH': return 0.273657188280276
    elif myC == 'FJ': return 0.336139019412
    elif myC == 'SL': return 0.337960802589002
    elif myC == 'MW': return 0.253076569219416
    elif myC == 'RO': return (277174.8438/1035207.75)
    elif myC == 'BO': return 0.4218342
    assert(False)

def get_demonym(myC):

    if myC == 'PH': return 'Filipinos'
    if myC == 'FJ': return 'Fijians'
    if myC == 'SL': return 'Sri Lankans'
    if myC == 'MW': return 'Malawians'
    if myC == 'RO': return 'Romanians'
    if myC == 'BO': return 'Bolivians'
    return 'individuals'



def get_all_hazards(myC,df):
    temp = (df.reset_index().set_index(['hazard'])).copy()
    temp = temp[~temp.index.duplicated(keep='first')]
    return [i for i in temp.index.values if i != 'x']

def get_all_rps(myC,df):
    temp = (df.reset_index().set_index(['rp'])).copy()
    temp = temp[~temp.index.duplicated(keep='first')]
    return [int(i) for i in temp.index.values]

def int_w_commas(in_int):
    in_str = str(in_int)
    in_list = list(in_str)
    out_str = ''

    if in_int < 1E3:  return in_str
    if in_int < 1E6:  return in_str[:-3]+','+in_str[-3:]
    if in_int < 1E9:  return in_str[:-6]+','+in_str[-6:-3]+','+in_str[-3:]
    if in_int < 1E12: return in_str[:-9]+','+in_str[-9:-6]+','+in_str[-6:-3]+','+in_str[-3:]
