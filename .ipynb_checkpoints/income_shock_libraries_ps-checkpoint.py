# DATE: 20200409
# Auth: P. Saylor & B. Walsh
# purpose: supplemental library for <'income_shock_plots.ipynb'>
#####

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
from numpy import random
#from libraries.lib_gather_data import get_hhid_FIES
#%load_ext autoreload

from libraries.lib_country_dir import set_directories, load_survey_data, get_places_dict
from libraries.lib_get_hh_savings import get_hh_savings
from libraries.pandas_helper import broadcast_simple

#import shock_libraries as sl

# formatting & aesthetics
font = {'family':'sans serif', 'size':10}
plt.rc('font', **font)
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.facecolor'] = 'white'
sns.set_style("white")

sns_pal = sns.color_palette('Set1', n_colors=8, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)
#######
### ----------> added 20200504: direct input into Brians model-- condensed version of the formula

## create the shock table
def get_phi_shock_sectors(flavor='nonag_wage'):
    """
    20200427: loads shock tables
    20200501: Condensed version of --> get_phi_shock():
        - isolates the desired options of sectoral non-ag wage and entrepreneural tables, each with their desired sub-categories.
    input: flavor
        options:
         
            'nonag_wage' -- sectoral nonag wage shock table
            'entre' -- sector entrepreneurial shock table
           

    
    
    """
    
    
    if flavor == 'nonag_wage':
        # sectoral nonag wage shock table
        df = pd.read_csv('./temp/phi_shocks/sect_nonag_wage_0427.csv').set_index('sector')
        df = df.rename(columns={'mean': 'fa', 'std_dev':'di'}) 
        df['di'] = 1
        shock_table = df
        
    elif flavor == 'entre':
        # sector entrepreneurial shock table
        df = pd.read_csv('./temp/phi_shocks/phi_entre.csv').set_index('sector')
        df = df.rename(columns={'mean': 'fa', 'std_dev':'di'}) 
        df['di'] = 1
        shock_table = df
    
        
        #print(flavor)
    else:
        print('error: input did not trigger either option, please enter <nonag_wage> or <entre>')
    
    return(shock_table)




#######


### ----------> added 20200426: direct input into other model
## create the shock table
def get_phi_shock(flavor=0):
    """
    20200427: loads shock tables
    
    input: flavor
        options:
            0 -- default shock table, in original model
            1 -- sectoral shock table, based on scoring
            2 -- nonag_shock table, sector names v2
            3 -- entrepreneurial shock table, sector names v2
            4 -- sectoral nonag wage shock table
            5 -- sector entrepreneurial shock table
            6 -- sectoral full shock table with weights

    
    
    """
    if flavor == 0:
        # default shock table:
        shock_default = { 'ag':           [  0,  0],
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
        df_shock = pd.DataFrame(data=shock_default).T
        df_shock.columns = ['fa','di']
        df_shock.index.name = 'sector'
        shock_table = df_shock
    
    if flavor == 1:
        # sectoral shock table
        df = pd.read_csv('./temp/phi_shocks/phi_shock_3dimv2.csv').set_index('LFS_sector')
        df = df.rename(columns={'mean': 'fa', 'std_dev':'di'}) 
        df['di'] = 1
        
        #print(flavor)
        #print(df)
        shock_table = df
        
    if flavor == 2:
        #nonag shock table
        df = pd.read_csv('./temp/phi_shocks/phi_nonag.csv').set_index('sector')
        df = df.rename(columns={'mean': 'fa', 'std_dev':'di'}) 
        df['di'] = 1
       # print(flavor)
        shock_table = df
        
    if flavor == 3:
        #entrepreneurial shock table
        df = pd.read_csv('./temp/phi_shocks/phi_entre.csv').set_index('sector')
        df = df.rename(columns={'mean': 'fa', 'std_dev':'di'}) 
        df['di'] = 1
        shock_table = df
    
    if flavor == 4:
        # sectoral nonag wage shock table
        df = pd.read_csv('./temp/phi_shocks/sect_nonag_wage_0427.csv').set_index('sector')
        df = df.rename(columns={'mean': 'fa', 'std_dev':'di'}) 
        df['di'] = 1
        shock_table = df
        
    if flavor == 5:
        # sector entrepreneurial shock table
        df = pd.read_csv('./temp/phi_shocks/sect_entre_0427.csv').set_index('sector')
        df = df.rename(columns={'mean': 'fa', 'std_dev':'di'}) 
        df['di'] = 1
        shock_table = df
    
    if flavor == 6:
        # sectoral full shock table with weights
        df = pd.read_csv('./temp/phi_shocks/fullsector_0427.csv').set_index('sector')
        df = df.rename(columns={'mean': 'fa', 'std_dev':'di'}) 
        df['di'] = 1
        shock_table = df
        
        #print(flavor)
   
    
    return(shock_table)




#######

# create function for load & merge kayenat data table
def merge_rank(rank_file='./temp/lfs_a09_pqkb_ranked_V2_entrpreneurial_20200423.csv', labor_file='./csv/ph_labor_force.csv',outfile='./csv/_labor_rank_merge.csv',merge_col='a09_pqkb'):

#def merge_rank(rank_file='./2015FIES/LFSJul2015_merge.csv', labor_file='./csv/ph_labor_force.csv',outfile='./csv/_labor_rank_merge.csv',merge_col='a09_pqkb'):
    """
    Description:
        - combines merged inputs from labor rankings with full household/individual dataset
    
    Assumes:
        - consistent file structure based on descriptive industry names
        - directory structure consistent with DIR: <covid_phl> when using default settings
        -left merge
    inputs:
        - rank_file = ranking file of job essential/work from home scoring // type: <STR>
        - labor_file = primary database to be merged to // type: <STR>
        - merge_col = column name for merging, must match in both files// type: <STR>
    outputs:
        - outfile = filename to print output, // type: <STR>
    returns:
        - labor_rank_merge = output dataframe with merged files // type: <pandas.df>
    
    """
    rank = pd.read_csv(rank_file) # load Kayenat file
    labor = pd.read_csv(labor_file) # load full survey

    # set index column to match types
    rank[merge_col] = [str(q) for q in rank[merge_col]] # enforce type = string
    labor[merge_col] = [str(q) for q in labor[merge_col]] # enforce type = string

    labor_rank_merge = pd.merge(labor,rank, on=merge_col, how='left') # merge on a09_pqkb
    labor_rank_merge.to_csv(outfile,index=False)
    print('The_rank_file ='+ rank_file)
    return(labor_rank_merge) 


# create function incorporating kayenat data table into shock factor:

# goal: generate function: weight_sectors

# generate counts of each of the industry descriptions, 
def rand_weighted_shock_1():
    """
    Update 20200411:

    module added to covid_phl: <income_shock_libraries_ps.py>

    primary development: <FCT> rand_weighted_shock_1() <FCT>
    - function to replace: get_income_shock() in <shock_libraries.py>
    - description:
        * matches existing df_shock dataframe (compatibiility)
        * uses Kayenat table of job descriptions demand value for 'a09_pqkb' by sector to create weighted probability of job disruption by sector, as input to 'fa' column of df_shock             dataframe -- representative FIES and LFS data 
        * for values 0.0, 0.5,1 : assigns each job description a random: 0-50%, 50-99%, 100% chance of disruption, weighting each by the prevalence of that role in each sector, to               generate cumulative probability of disruption. 

    - notes: at present slow, bc originally designed for individual-level iteration
        * each time run will produce different result, due to weighted random uniform sampling in each sector
        * 
    - future work:
        * 1] can be built as option into: get_income_shock()
        * 2] can be iterated to to produce a mean static value
        * 3] can be dramatically sped up (depending on use case)
    
    
    """
    mr = merge_rank() # calls merge_rank() in this module

    # get subset:
    mr_subset = mr[['hhid_lfs','cc101_lno','LFS_sector','a09_pqkb','demand_scale']]

    # Delete 'nan' row indexes from dataFrame
    indexNames = mr_subset[mr_subset['a09_pqkb'] == 'nan' ].index
    mr_subset.drop(indexNames , inplace=True)
    mr_subset = mr_subset.reset_index(drop=True)

    mr_subset2 = mr_subset
    #mr_subset.to_csv('./scratch_csv/after2.csv')
    #mr_subset2.to_csv('./scratch_csv/after22.csv')
    # enforce string:
    mr_subset['a09_pqkb'] = [str(q) for q in mr_subset['a09_pqkb']] # enforce type = string
    mr_subset['LFS_sector'] = [str(q) for q in mr_subset['LFS_sector']] # enforce type = string


    # generate fraction by sector
    mr_subset['desc_count'] = mr_subset.groupby('a09_pqkb')['a09_pqkb'].transform('count')# count unique jobs and append to mr_subset
    mr_subset['sector_count'] = mr_subset.groupby('LFS_sector')['LFS_sector'].transform('count') #count total unique sectors and append to mr_subset
    mr_subset['sector_frac'] = mr_subset['desc_count'] / mr_subset['sector_count'] # get fraction of sector as weighting



    # generate probability and combine with relative weighting
    mr_subset['partial_prob'] = np.nan
    mr_subset['dummy'] = np.nan


    # this section runs fine without the dropnan section
    i=0
    while i < len(mr_subset):
        #   if i % 10000 == 0:
        #   print('test'+ str(i))
        if mr_subset.demand_scale[i] == 0:
            mr_subset.partial_prob[i] = mr_subset.sector_frac[i] * (random.randint(0,50)/100)

        elif mr_subset.demand_scale[i] == 0.5: 
            mr_subset.partial_prob[i] = mr_subset.sector_frac[i] * (random.randint(50,100)/100)

        elif mr_subset.demand_scale[i] == 1.0:
            mr_subset.partial_prob[i] = mr_subset.sector_frac[i]
        else:
            mr_subset.dummy[i] = -99
        i = i + 1

    # remove dummy storage [testing artifact]
    del mr_subset['dummy']


    shock_null = { 'ag':           [  0,  0],
                 'mining':        [  0,  0],
                 'utilities':     [  0,  0],
                 'construction':  [0.0,1.0],
                 'manufacturing': [0.0,1.0],
                 'wholesale':     [0.0,1.0],
                 'retail':        [0.0,1.0],
                 'transportation':[0.0,1.0],
                 'information':   [0.0,1.0],
                 'finance':       [0.0,1.0],
                 'professional_services':[0.0,1.0],
                 'eduhealth':     [0.0,1.0],
                 'food_entertainment':[0.0,1.0],
                 'government':    [  0,  0],
                 'other':         [0.0,1.0]}
    df_shock_null = pd.DataFrame(data=shock_null).T
    df_shock_null.columns = ['fa','di']
    df_shock_null.index.name = 'LFS_sector'


    df_shock_cum = df_shock_null
   
   
    for seclist in df_shock_cum.index: # hard-coded to existing shock table

        pp = mr_subset[mr_subset.LFS_sector == seclist]
        ppp = pp.drop_duplicates(subset='a09_pqkb')
        p4 = 1 - sum(ppp.partial_prob)

        # build shock table:
        df_shock_cum['fa'][seclist] = df_shock_cum['fa'][seclist] + p4
       

    rand_weighted_shock = df_shock_cum
    
    return(rand_weighted_shock)

##### --------------- Added: 20200413: <PS>

def rand_weighted_shock_distance():
   
    """
    Updated 20200413:

    module added to covid_phl: <income_shock_libraries_ps.py>

    primary development: <FCT> rand_weigthed_shock_distance() <FCT>
    
    - function to replace: rand_weighted_shock_1() --> get_income_shock(): in <shock_libraries.py>
    
    - description:
        * matches existing df_shock dataframe (compatibiility)
        * uses Kayenat table of job descriptions demand value for 'a09_pqkb' by sector to create weighted probability of job disruption by sector, as input to 'fa' column of df_shock             
        dataframe -- representative FIES and LFS data 
       
       * for values 0.0, 0.5,1 : assigns each job description a random: 0-50%, 50-99%, 100% chance of disruption, weighting each by the prevalence of that role in each sector, to generate cumulative probability of disruption. 

        * now incorporates enforcement of social distancing measures, by enforcing social distance in non-essential jobs based on K.Kabirs' 0-4 'work-from-home' scoring.
    """

    mr = merge_rank()

    # get subset:
    mr_subset = mr[['hhid_lfs','cc101_lno','LFS_sector','a09_pqkb','demand_scale', 'w_home']]

    indexNames = mr_subset[mr_subset['a09_pqkb'] == 'nan' ].index

    # Delete these row indexes from dataFrame
    mr_subset.drop(indexNames , inplace=True)
    mr_subset = mr_subset.reset_index(drop=True)

    # enforce string:
    mr_subset['a09_pqkb'] = [str(q) for q in mr_subset['a09_pqkb']] # enforce type = string
    mr_subset['LFS_sector'] = [str(q) for q in mr_subset['LFS_sector']] # enforce type = string


    # generate fraction by sector
    mr_subset['desc_count'] = mr_subset.groupby('a09_pqkb')['a09_pqkb'].transform('count')# count unique jobs and append to mr_subset
    mr_subset['sector_count'] = mr_subset.groupby('LFS_sector')['LFS_sector'].transform('count') #count total unique sectors and append to mr_subset
    mr_subset['sector_frac'] = mr_subset['desc_count'] / mr_subset['sector_count'] # get fraction of sector as weighting
    
    
    # drop duplicates (now that overall weighting established)
    mr_subset = mr_subset.drop_duplicates(subset='a09_pqkb')
    mr_subset = mr_subset.reset_index(drop=True)
    

    
    # generate probability and combine with relative weighting
    mr_subset['partial_prob'] = np.nan
    mr_subset['dummy'] = np.nan

    # incorporate Kayenat tables into 'di' &&
    # nested logic to incorporate 0-4 scale for social distancing measures
    ## where scores of 0 & 1 result in complete job lost, due to unable to distance
    i=0
    while i < len(mr_subset):
        
        if mr_subset.demand_scale[i] == 0:
            
            # incorporate 0-4 scale logic:
            
            if mr_subset.w_home[i] == 0:
                mr_subset.partial_prob[i] = 0
            
            elif mr_subset.w_home[i] == 1:
                mr_subset.partial_prob[i] = 0
            
            else:
                mr_subset.partial_prob[i] = mr_subset.sector_frac[i] * (random.randint(0,50)/100)

                
        elif mr_subset.demand_scale[i] == 0.5: 
            
            # incorporate 0-4 scale logic:
            if mr_subset.w_home[i] == 0:
                mr_subset.partial_prob[i] = 0
            
            elif mr_subset.w_home[i] == 1:
                mr_subset.partial_prob[i] = 0
            
            else: 
                mr_subset.partial_prob[i] = mr_subset.sector_frac[i] * (random.randint(50,100)/100)

        elif mr_subset.demand_scale[i] == 1.0:
            mr_subset.partial_prob[i] = mr_subset.sector_frac[i]
        else:
            mr_subset.dummy[i] = -99
        i = i + 1

    # remove nans in summing fields, and dummy storage
    del mr_subset['dummy']

    #define shock table:
    shock_null = { 'ag':           [  0,  0],
                 'mining':        [  0,  0],
                 'utilities':     [  0,  0],
                 'construction':  [0.0,1.0],
                 'manufacturing': [0.0,1.0],
                 'wholesale':     [0.0,1.0],
                 'retail':        [0.0,1.0],
                 'transportation':[0.0,1.0],
                 'information':   [0.0,1.0],
                 'finance':       [0.0,1.0],
                 'professional_services':[0.0,1.0],
                 'eduhealth':     [0.0,1.0],
                 'food_entertainment':[0.0,1.0],
                 'government':    [  0,  0],
                 'other':         [0.0,1.0]}
    df_shock_null = pd.DataFrame(data=shock_null).T
    df_shock_null.columns = ['fa','di']
    df_shock_null.index.name = 'LFS_sector'


    df_shock_cum = df_shock_null

    # get mean probability by sector:
    
    for seclist in df_shock_cum.index: # hard-coded to existing shock table

        pp = mr_subset[mr_subset.LFS_sector == seclist]
        p4 = 1 - sum(pp.partial_prob)

        # build shock table:
        df_shock_cum['fa'][seclist] = df_shock_cum['fa'][seclist] + p4
    
    # save to separate var for testing    
    rand_weighted_shock = df_shock_cum

    return(rand_weighted_shock)



#---------------------------------- Added: 20200415: <PS>
def generate_shock_100():  # initialize shock sector storage dataframe
   
    '''
   current hard coding for sensitivity analysis, 20200413: requires cleaning for further implementation
   - addition of modularity
   - 
   - current functionality:
       - outputs csv to location: './temp/sect_iter_100.csv
       - containing data frame with 101 simulations of <rand_weighted_shock_distance():
    - runtime: ~10minutes
   '''

    stor = rand_weighted_shock_3dim_v2()
    del stor['di']

    # set number of iterations
    p = 0
    n_iter = 99

    # model and store stochastic sector response
    while p < n_iter:
        new_val = rand_weighted_shock_3dim_v2()
        del new_val['di']
        new_val = new_val.rename(columns={'fa': ('iter'+str(p))})

        # pd.merge(labor,rank, on=merge_col, how='left')
        stor = pd.merge(stor,new_val,on='LFS_sector', how='left')
        p = p+ 1
        print(p)
    stor.to_csv('./temp/sect_iter_100_3dv2_20200422.csv')

#---------------------------------- Added: 20200415: <PS>
def get_shock_stats():
    # generate shock table statistics
    #df['mean'] = df.mean(axis=1)

    # load csv to dataframe:
    #dfs = pd.read_csv('./temp/sect_iter_100.csv') # original
    dfs = pd.read_csv('./temp/sect_iter_100_3dv2_20200422.csv') # modified 20200420
    # set index to LFS_sector
    dfs.set_index('LFS_sector')

    # compute statistics:
    dfs['mean'] = dfs.mean(axis=1)
    #print(dfs['mean'])
    dfs['std_dev'] = dfs.std(axis=1)
    #print(dfs['std_dev'])

    #round to 3 dec:
    dfs['mean'] = [(round(q, 3)) for q in dfs['mean']]
    dfs['std_dev'] = [(round(q, 3)) for q in dfs['std_dev']]

    # new datafame storing just info:
    df_stat = dfs[['LFS_sector','mean','std_dev']].set_index('LFS_sector')
    df_stat
    # df_stat.to_csv('./temp/phi_get_shock_input.csv') # original
    df_stat.to_csv('./temp/phi_shock_3dimv2_stats_20200422.csv') # modified 20200420
    return(df_stat)


#---------------------------------- Added: 20200415: <PS>
# create shock table
def create_country_shock():
    #define null shock table:
    shock_null = { 'ag':           [  0,  0],
                     'mining':        [  0,  0],
                     'utilities':     [  0,  0],
                     'construction':  [0.0,1.0],
                     'manufacturing': [0.0,1.0],
                     'wholesale':     [0.0,1.0],
                     'retail':        [0.0,1.0],
                     'transportation':[0.0,1.0],
                     'information':   [0.0,1.0],
                     'finance':       [0.0,1.0],
                     'professional_services':[0.0,1.0],
                     'eduhealth':     [0.0,1.0],
                     'food_entertainment':[0.0,1.0],
                     'government':    [  0,  0],
                     'other':         [0.0,1.0]}
    df_shock_null = pd.DataFrame(data=shock_null).T
    df_shock_null.columns = ['fa','di']
    df_shock_null.index.name = 'LFS_sector'

    # define default shock table:
    shock_default = { 'ag':           [  0,  0],
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
    df_shock = pd.DataFrame(data=shock_default).T
    df_shock.columns = ['fa','di']
    df_shock.index.name = 'LFS_sector'

    # create phillipine shock
    shock_country = df_shock # set up output

    temp = pd.read_csv('./temp/phi_get_shock_input.csv')
    temp = pd.read_csv('./temp/phi_get_shock_input.csv').set_index('LFS_sector')
    temp = temp.rename(columns={"mean":"fa"})

    # set 'ag' to zero (hardcode default)
    temp.at['ag', 'fa'] = 0.000

    # list comprehension replace default shocks with country shock 'fa'
    shock_country.fa = temp.fa

    shock_country.to_csv('./csv/phi_fa_shock_input.csv')

    
#---------------------------------- Added: 20200415: <PS>
def shock_comparison_table():
    # define default shock table:
    shock_default = { 'ag':           [  0,  0],
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
    df_shock = pd.DataFrame(data=shock_default).T
    df_shock.columns = ['fa','di']
    df_shock.index.name = 'LFS_sector'

    ## load phillipine country stats
    phi_stats = pd.read_csv('./temp/phi_get_shock_input.csv').set_index('LFS_sector')
    phi_stats.at['ag', 'mean'] = 0.000

    phi_stats = phi_stats.rename(columns={"mean":"phi_sector_model", "std_dev":"phi_std_dev"})
    phi_stats['model_default'] = df_shock['fa']

    phi_stats['difference'] = phi_stats.model_default - phi_stats.phi_sector_model

    final_table = phi_stats[['model_default','phi_sector_model','difference','phi_std_dev']]

    return(final_table)

#---------------------------------- Added: 20200419: <PS>


def rand_weighted_shock_3dim():

    """    Updated 20200419 
        - incorporate 3rd dimension for sector (public/private/gov) and impact on essentiality 
    Updated 20200413:
        - incorporate 2nd dimension for social distancing potential
    module added to covid_phl: <income_shock_libraries_ps.py>

    primary development: <FCT> rand_weigthed_shock_distance() <FCT>

    - function to replace: rand_weighted_shock_distance() --> rand_weighted_shock_1() --> get_income_shock(): in <shock_libraries.py>

    - description:
        * matches existing df_shock dataframe (compatibiility)
        * uses Kayenat table of job descriptions demand value for 'a09_pqkb' by sector to create weighted probability of job disruption by sector, as input to 'fa' column of df_shock             
        dataframe -- representative FIES and LFS data 

       * for values 0.0, 0.5,1 : assigns each job description a random: 0-50%, 50-99%, 100% chance of disruption, weighting each by the prevalence of that role in each sector, to generate cumulative probability of disruption. 


        * now incorporates enforcement of social distancing measures, by enforcing social distance in non-essential jobs based on K.Kabirs' 0-4 'work-from-home' scoring.



    """
    # develop 3 factor code here:

    # make each factor modular

    mr = merge_rank()

        # get subset: a09_pqkb
    mr_subset = mr[['hhid_lfs','cc101_lno','LFS_sector','a09_pqkb','c19_pclass','demand_scale', 'w_home']]

    indexNames = mr_subset[mr_subset['a09_pqkb'] == 'nan' ].index

        # Delete these row indexes from dataFrame
    mr_subset.drop(indexNames , inplace=True)
    mr_subset = mr_subset.reset_index(drop=True)

     # get subset: c19_pclass

    indexNames2 = mr_subset[mr_subset['c19_pclass'] == 'nan' ].index

        # Delete these row indexes from dataFrame
    mr_subset.drop(indexNames2 , inplace=True)
    mr_subset = mr_subset.reset_index(drop=True)

    # make new column of combined string a09 && c19:
    mr_subset['a09c19'] = mr_subset['a09_pqkb'] +'-'+mr_subset['c19_pclass']

        # enforce string:
    mr_subset['a09_pqkb'] = [str(q) for q in mr_subset['a09_pqkb']] # enforce type = string
    mr_subset['LFS_sector'] = [str(q) for q in mr_subset['LFS_sector']] # enforce type = string
    mr_subset['c19_pclass'] = [str(q) for q in mr_subset['c19_pclass']] # enforce type = string
    mr_subset['a09c19'] = [str(q) for q in mr_subset['a09c19']] # enforce type = string


        # generate fraction by sector
    mr_subset['desc_count'] = mr_subset.groupby('a09c19')['a09c19'].transform('count')# count unique jobs and append to mr_subset
    mr_subset['sector_count'] = mr_subset.groupby('LFS_sector')['LFS_sector'].transform('count') #count total unique sectors and append to mr_subset
    mr_subset['sector_frac'] = mr_subset['desc_count'] / mr_subset['sector_count'] # get fraction of sector as weighting



    #####
    # here, need to insert a new column that merges a09 and c19 -- done
    # then, drop duplicates off of this column, so that we can minimize computation

    # still need logic to build the logic for each job sector
    ## may need to restructure this whole section of code

    #####
        # drop duplicates (now that overall weighting established)
    mr_subset = mr_subset.drop_duplicates(subset='a09c19')
    mr_subset = mr_subset.reset_index(drop=True)



        # generate probability and combine with relative weighting
    mr_subset['partial_prob'] = np.nan
    mr_subset['third_col'] = np.nan
    mr_subset['dummy'] = np.nan

        # incorporate Kayenat tables into 'di' &&
        # nested logic to incorporate 0-4 scale for social distancing measures
        ## where scores of 0 & 1 result in complete job lost, due to unable to distance
    i=0
    while i < len(mr_subset):



        if mr_subset.demand_scale[i] == 0:

                # incorporate 0-4 scale logic:

            if mr_subset.w_home[i] == 0:
                mr_subset.partial_prob[i] = 0

            elif mr_subset.w_home[i] == 1:
                mr_subset.partial_prob[i] = 0

            else:
                mr_subset.partial_prob[i] = mr_subset.sector_frac[i] * (random.randint(0,50)/100)


        elif mr_subset.demand_scale[i] == 0.5: 

                # incorporate 0-4 scale logic:
            if mr_subset.w_home[i] == 0:
                mr_subset.partial_prob[i] = 0

            elif mr_subset.w_home[i] == 1:
                mr_subset.partial_prob[i] = 0

            else: 
                mr_subset.partial_prob[i] = mr_subset.sector_frac[i] * (random.randint(50,100)/100)

        elif mr_subset.demand_scale[i] == 1.0:
            mr_subset.partial_prob[i] = mr_subset.sector_frac[i]
        else:
            mr_subset.dummy[i] = -99

        # incorporate 3rd column modifiers here:
        if (mr_subset['c19_pclass'][i] == "Gov't/Gov't Corporation"):
            mr_subset.partial_prob[i] = mr_subset.partial_prob[i] * 1

        elif (mr_subset['c19_pclass'][i] == 'Private Household' or 'Self Employed' or 'Employer' or 'Without Pay (Family owned Business)' or 'With pay (Family owned Business)'):
            mr_subset.partial_prob[i] = mr_subset.partial_prob[i] * (random.randint(0,50)/100)

        elif (mr_subset['c19_pclass'][i] == 'Private Establishment'):
            mr_subset.partial_prob[i] = mr_subset.partial_prob[i] * (random.randint(50,100)/100)
        else:
            mr_subset.dummy[i] = -89


        i = i + 1

        # remove nans in summing fields, and dummy storage
    del mr_subset['dummy']

        #define shock table:
    shock_null = { 'ag':           [  0,  0],
                     'mining':        [  0,  0],
                     'utilities':     [  0,  0],
                     'construction':  [0.0,1.0],
                     'manufacturing': [0.0,1.0],
                     'wholesale':     [0.0,1.0],
                     'retail':        [0.0,1.0],
                     'transportation':[0.0,1.0],
                     'information':   [0.0,1.0],
                     'finance':       [0.0,1.0],
                     'professional_services':[0.0,1.0],
                     'eduhealth':     [0.0,1.0],
                     'food_entertainment':[0.0,1.0],
                     'government':    [  0,  0],
                     'other':         [0.0,1.0]}
    df_shock_null = pd.DataFrame(data=shock_null).T
    df_shock_null.columns = ['fa','di']
    df_shock_null.index.name = 'LFS_sector'


    df_shock_cum = df_shock_null

        # get mean probability by sector:

    for seclist in df_shock_cum.index: # hard-coded to existing shock table

        pp = mr_subset[mr_subset.LFS_sector == seclist]
        p4 = 1 - sum(pp.partial_prob)

            # build shock table:
        df_shock_cum['fa'][seclist] = df_shock_cum['fa'][seclist] + p4

        # save to separate var for testing    
    rand_weighted_shock = df_shock_cum
    
    return(rand_weighted_shock)

#---------------------------------- Added: 20200420: <PS>
def shock_comparison_table_3method():
    """
    modified from: shock_comparison_table()
    on: 20200420
    
    functionality: 
        - creates table to visualize difference between 'fraction affected' by sector
        - for default, social distancing, and 3rd dimension cases
    
    """
    # define default shock table:
    shock_default = { 'ag':           [  0,  0],
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
    df_shock = pd.DataFrame(data=shock_default).T
    df_shock.columns = ['fa','di']
    df_shock.index.name = 'LFS_sector'

    ## load phillipine country stats
    phi_stats = pd.read_csv('./temp/phi_get_shock_input.csv').set_index('LFS_sector')
    #phi_stats.at['ag', 'mean'] = 0.000

    phi_stats = phi_stats.rename(columns={"mean":"social_dist", "std_dev":"SD_std_dev"}) # rename columns
    phi_stats['default'] = df_shock['fa'] # get default values for table

    # phi_stats['difference'] = phi_stats.model_default - phi_stats.phi_sector_model # removed 20200420 <ps>

    
    # add 3dim stats here
    dim3_stats = pd.read_csv('./temp/phi_shock_3dimv2_stats_20200422.csv').set_index('LFS_sector')
    #dim3_stats.at['ag', 'mean'] = 0.000
    dim3_stats = dim3_stats.rename(columns={"mean":"Full_model", "std_dev":"Full_std_dev"}) # rename columns

    
    # create final table 
    phi_stats = phi_stats[['default','social_dist','SD_std_dev']]
    dim3_stats = dim3_stats[['Full_model','Full_std_dev']]
    
    phi_stats['Full_model'] = dim3_stats.Full_model
    phi_stats['Full_std_dev'] = dim3_stats.Full_std_dev
    phi_stats['diff(SD - Full)'] = phi_stats.social_dist - phi_stats.Full_model
    final_table = phi_stats
    
    return(final_table)

#---------------------------------- Added: 20200422: <PS>

def rand_weighted_shock_3dim_v2():

    """    
    Updated 20200422 
        - incorporate 3rd dimension for sector (public/private/gov) and impact on essentiality
            - now, only enforce that government jobs are maintained, across all sectors
    
    Updated 20200419 
        - incorporate 3rd dimension for sector (public/private/gov) and impact on essentiality 
    Updated 20200413:
        - incorporate 2nd dimension for social distancing potential
    module added to covid_phl: <income_shock_libraries_ps.py>

    primary development: <FCT> rand_weigthed_shock_distance() <FCT>

    - function to replace: 
        rand_weighted_shock_3dim()--> rand_weighted_shock_distance() --> rand_weighted_shock_1() --> get_income_shock(): in <shock_libraries.py>

    - description:
        * matches existing df_shock dataframe (compatibiility)
        * uses Kayenat table of job descriptions demand value for 'a09_pqkb' by sector to create weighted probability of job disruption by sector, as input to 'fa' column of df_shock             
        dataframe -- representative FIES and LFS data 

       * for values 0.0, 0.5,1 : assigns each job description a random: 0-50%, 50-99%, 100% chance of disruption, weighting each by the prevalence of that role in each sector, to generate cumulative probability of disruption. 


        * now incorporates enforcement of social distancing measures, by enforcing social distance in non-essential jobs based on K.Kabirs' 0-4 'work-from-home' scoring.



    """
    # develop 3 factor code here:

    # make each factor modular

    mr = merge_rank()

        # get subset: a09_pqkb
    mr_subset = mr[['hhid_lfs','cc101_lno','LFS_sector','a09_pqkb','c19_pclass','demand_scale', 'w_home']]

    indexNames = mr_subset[mr_subset['a09_pqkb'] == 'nan' ].index

        # Delete these row indexes from dataFrame
    mr_subset.drop(indexNames , inplace=True)
    mr_subset = mr_subset.reset_index(drop=True)

     # get subset: c19_pclass

    indexNames2 = mr_subset[mr_subset['c19_pclass'] == 'nan' ].index

        # Delete these row indexes from dataFrame
    mr_subset.drop(indexNames2 , inplace=True)
    mr_subset = mr_subset.reset_index(drop=True)

    # make new column of combined string a09 && c19:
    mr_subset['a09c19'] = mr_subset['a09_pqkb'] +'-'+mr_subset['c19_pclass']

        # enforce string:
    mr_subset['a09_pqkb'] = [str(q) for q in mr_subset['a09_pqkb']] # enforce type = string
    mr_subset['LFS_sector'] = [str(q) for q in mr_subset['LFS_sector']] # enforce type = string
    mr_subset['c19_pclass'] = [str(q) for q in mr_subset['c19_pclass']] # enforce type = string
    mr_subset['a09c19'] = [str(q) for q in mr_subset['a09c19']] # enforce type = string


        # generate fraction by sector
    mr_subset['desc_count'] = mr_subset.groupby('a09_pqkb')['a09_pqkb'].transform('count')# count unique jobs and append to mr_subset
    mr_subset['sector_count'] = mr_subset.groupby('LFS_sector')['LFS_sector'].transform('count') #count total unique sectors and append to mr_subset
    mr_subset['sector_frac'] = mr_subset['desc_count'] / mr_subset['sector_count'] # get fraction of sector as weighting



    #####
    # here, need to insert a new column that merges a09 and c19 -- done
    # then, drop duplicates off of this column, so that we can minimize computation

    # still need logic to build the logic for each job sector
    ## may need to restructure this whole section of code

    #####
        # drop duplicates (now that overall weighting established)
    mr_subset = mr_subset.drop_duplicates(subset='a09_pqkb')
    mr_subset = mr_subset.reset_index(drop=True)



        # generate probability and combine with relative weighting
    mr_subset['partial_prob'] = np.nan
    mr_subset['third_col'] = np.nan
    mr_subset['dummy'] = np.nan

        # incorporate Kayenat tables into 'di' &&
        # nested logic to incorporate 0-4 scale for social distancing measures
        ## where scores of 0 & 1 result in complete job lost, due to unable to distance
    i=0
    while i < len(mr_subset):



        if mr_subset.demand_scale[i] == 0:

                # incorporate 0-4 scale logic:

            if mr_subset.w_home[i] == 0:
                mr_subset.partial_prob[i] = 0

            elif mr_subset.w_home[i] == 1:
                mr_subset.partial_prob[i] = 0

            else:
                mr_subset.partial_prob[i] = mr_subset.sector_frac[i] * (random.randint(0,50)/100)


        elif mr_subset.demand_scale[i] == 0.5: 

                # incorporate 0-4 scale logic:
            if mr_subset.w_home[i] == 0:
                mr_subset.partial_prob[i] = 0

            elif mr_subset.w_home[i] == 1:
                mr_subset.partial_prob[i] = 0

            else: 
                mr_subset.partial_prob[i] = mr_subset.sector_frac[i] * (random.randint(50,100)/100)

        elif mr_subset.demand_scale[i] == 1.0:
            mr_subset.partial_prob[i] = mr_subset.sector_frac[i]
        else:
            mr_subset.dummy[i] = -99

            
        # incorporate 3rd column modifiers here:
        if (mr_subset['c19_pclass'][i] == "Gov't/Gov't Corporation"):
            mr_subset.partial_prob[i] = mr_subset.sector_frac[i]  # essentially reverts the random uniform logic implemented above


        i = i + 1

        # remove nans in summing fields, and dummy storage
    del mr_subset['dummy']

        #define shock table:
    shock_null = { 'ag':           [  0,  0],
                     'mining':        [  0,  0],
                     'utilities':     [  0,  0],
                     'construction':  [0.0,1.0],
                     'manufacturing': [0.0,1.0],
                     'wholesale':     [0.0,1.0],
                     'retail':        [0.0,1.0],
                     'transportation':[0.0,1.0],
                     'information':   [0.0,1.0],
                     'finance':       [0.0,1.0],
                     'professional_services':[0.0,1.0],
                     'eduhealth':     [0.0,1.0],
                     'food_entertainment':[0.0,1.0],
                     'government':    [  0,  0],
                     'other':         [0.0,1.0]}
    df_shock_null = pd.DataFrame(data=shock_null).T
    df_shock_null.columns = ['fa','di']
    df_shock_null.index.name = 'LFS_sector'


    df_shock_cum = df_shock_null

        # get mean probability by sector:

    for seclist in df_shock_cum.index: # hard-coded to existing shock table

        pp = mr_subset[mr_subset.LFS_sector == seclist]
        p4 = 1 - sum(pp.partial_prob)

            # build shock table:
        df_shock_cum['fa'][seclist] = df_shock_cum['fa'][seclist] + p4

        # save to separate var for testing    
    rand_weighted_shock = df_shock_cum
    
    return(rand_weighted_shock)
