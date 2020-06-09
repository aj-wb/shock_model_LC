import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns_pal = sns.color_palette('Set1', n_colors=20, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)

from mc_storage_libraries import monte_carlo


def plot_sectoral_income(plot_losses=False):

    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,7))

    df = pd.read_csv('monte_carlo/sectoral_income_by_decile.csv',index_col=['decile','LFS_sector']).sort_index()
    df[['nonagri_sal','agri_sal','nonag_wage_loss','ag_wage_loss']]*= 1E-3

    df_sum = df.sum(level='LFS_sector')
    df_sum.loc['total'] = df_sum.sum(axis=0)

    # split into 2 dfs: categories > (threshold)%, and all others
    threshold = 0.07
    df_sum_other = df_sum.drop('ag',axis=0).loc[df_sum['nonagri_sal']<threshold*float(df_sum.loc['total','nonagri_sal'])]

    # get categories > (threshold)%
    df_sum = df_sum.loc[df_sum['nonagri_sal']>=threshold*float(df_sum.loc['total','nonagri_sal'])].drop('total',axis=0)
    df_sum.loc['remaining categories'] = df_sum_other.sum(axis=0)

    # put this info back into decile-level df 
    df_other = df.reset_index().set_index('LFS_sector').loc[df_sum_other.index]
    df_other = df_other.reset_index().set_index(['decile','LFS_sector']).sum(level='decile')
    df_other['LFS_sector'] = 'remaining categories'

    # drop other sectors from df
    df = df.reset_index().set_index('LFS_sector').drop(df_sum_other.index,axis=0)

    # merge
    df = pd.concat([df.reset_index(),df_other.reset_index()],ignore_index=True).set_index(['decile','LFS_sector']).sort_index().reset_index()

    # merge ag & non-ag columns
    df['wage_loss'] = df['nonag_wage_loss'].copy()
    df.loc[df.LFS_sector=='ag','wage_loss'] = df.loc[df.LFS_sector=='ag','ag_wage_loss']
    df['total_wages'] = df['nonagri_sal'].copy()
    df.loc[df.LFS_sector=='ag','total_wages'] = df.loc[df.LFS_sector=='ag','agri_sal']
    df.drop(['nonag_wage_loss','ag_wage_loss','nonagri_sal','agri_sal'],axis=1,inplace=True)

    # calculate sectoral contribution to wage income, by decile
    df['sector_frac_total_wages'] = df['total_wages']/df.groupby('decile')['total_wages'].transform('sum')
    df['sector_losses_frac_total_wages'] = df['wage_loss']/df.groupby('decile')['total_wages'].transform('sum')

    # plotz
    # left subplot: total value of each sector
    plt.axes(ax[0])

    output_col = 'wage_loss' if plot_losses else 'total_wages'

    bot = [0 for _ in range(1,11)]
    sectors = df.LFS_sector.unique()
    for _n,_sec in enumerate(np.append(sectors[sectors!='remaining categories'],'remaining categories')):

        try: sec_lbl = monte_carlo().sector_labels[_sec]
        except: sec_lbl = _sec

        ddf = df.loc[df['LFS_sector']==_sec]

        plt.bar(ddf['decile'],ddf[output_col],bottom=bot,label=sec_lbl,facecolor=sns_pal[_n],alpha=0.7,lw=0)
        bot+=np.array(ddf[output_col].squeeze().T)

    plt.xlim(0.75,11)
    plt.xticks([_+0.4 for _ in range(1,11)],range(1,11),linespacing=0.80,fontsize=9,ha='center')
    plt.xlabel('Income decile',fontsize=9,labelpad=10)

    plt.yticks([0.5*_ for _ in range(1,8)],linespacing=0.80,fontsize=9,ha='center')
    plt.ylabel('Wage income {}[billion PPP\$]'.format('losses ' if plot_losses else ''),fontsize=9,labelpad=10)

    plt.grid(True,axis='y',alpha=0.3)
    plt.legend(loc='upper left',labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
    sns.despine(left=True)

    # right subplot: relative contribution of each sector to total wage inome
    plt.axes(ax[1])

    output_col = 'sector_losses_frac_total_wages' if plot_losses else 'sector_frac_total_wages'

    bot = [0 for _ in range(1,11)]
    sectors = df.LFS_sector.unique()
    for _n,_sec in enumerate(np.append(sectors[sectors!='remaining categories'],'remaining categories')):

        try: sec_lbl = monte_carlo().sector_labels[_sec]
        except: sec_lbl = _sec

        ddf = df.loc[df['LFS_sector']==_sec]

        plt.bar(ddf['decile'],ddf[output_col],bottom=bot,label=sec_lbl,facecolor=sns_pal[_n],alpha=0.7,lw=0)
        bot+=np.array(ddf[output_col].squeeze().T)

    plt.xlim(0.75,11)
    plt.xticks([_+0.4 for _ in range(1,11)],range(1,11),linespacing=0.80,fontsize=9,ha='center')
    plt.xlabel('Income decile',fontsize=9,labelpad=10)

    plt.ylim(0,1)
    plt.yticks([0.2*_ for _ in range(1,6)],linespacing=0.80,fontsize=9,ha='center')
    plt.ylabel('Fraction of total wage income {}[%]'.format('lost ' if plot_losses else ''),fontsize=9,labelpad=10)

    plt.grid(True,axis='y',alpha=0.3) 

    plt.savefig('figs/wage_income_sectoral_{}.pdf'.format('losses' if plot_losses else 'composition'),format='pdf',bbox_inches='tight')
    plt.close('all')


def plot_income_profile(hh_df,affected_only=False):
    pal = monte_carlo().ichan_cols

    _ul = 500
    nbins = int(40)

    # total income <-- use to structure plot
    tot_hgt, _bins = np.histogram(hh_df.eval('pcinc_initial').clip(upper=_ul),bins=nbins,weights=hh_df['popwgt'])
    wid = (_bins[1]-_bins[0])

    frac_nonag_hgt = []; frac_ag_hgt = []; frac_totent_hgt = []
    frac_public_hgt = []; frac_remit_hgt = [];frac_dom_remit_hgt = []
    for n,b in enumerate(_bins): 
        try:
            bin_slice = '(pcinc_initial>@b)&(pcinc_initial<=@_bins[@n+1]){}'.format('&(income_loss>0)' if affected_only else '')
            frac_nonag_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('1E2*popwgt*nonagri_sal').sum()
                                        /hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*hhinc').sum()))
            frac_ag_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('1E2*popwgt*agri_sal').sum()
                                     /hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*hhinc').sum()))
            frac_public_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('1E2*popwgt*total_public').sum()
                                         /hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*hhinc').sum()))
            frac_remit_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('1E2*popwgt*cash_abroad').sum()
                                        /hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*hhinc').sum()))
            frac_dom_remit_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('1E2*popwgt*cash_domestic').sum()
                                            /hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*hhinc').sum()))
            frac_totent_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('1E2*popwgt*total_entrepreneurial').sum()
                                        /hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*hhinc').sum()))
        except: pass

    _btm = [0 for _ in frac_remit_hgt]
    for istream in [[frac_totent_hgt,'entrepreneurial',pal['entrep']],
                    [frac_nonag_hgt,'wages (non-ag)',pal['nonag_wage']],
                    [frac_ag_hgt,'agricultural wages',pal['ag_wage']],
                    [frac_public_hgt,'public transfers',pal['pub_trans']],
                    [frac_remit_hgt,'intl. remittances',pal['remits']],
                    [frac_dom_remit_hgt,'domestic remittances',pal['dom_remits']]]:

        ax = plt.bar(_bins[:-1],istream[0],bottom=_btm,width=wid,align='edge',linewidth=0,alpha=0.6,facecolor=istream[2],label=istream[1])
        _btm = [i+j for i,j in zip(_btm,istream[0])]


    plt.legend(loc='center right',labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,
                           fancybox=True,frameon=True,framealpha=0.9)
    plt.xlabel('pre-COVID income [PPP$/cap/month]',labelpad=8)
    plt.ylabel('fraction of total income [%]',labelpad=8)
    plt.xlim(0)
    plt.ylim(0,100)

    plt.grid(True,axis='y',alpha=0.3)    
    sns.despine(left=True)
    #plt.plot([0,_ul],[0,0],color=greys_pal[4],lw=1)
    plt.savefig('figs/income_composition{}.pdf'.format('_affected_only' if affected_only else ''),format='pdf',bbox_inches='tight') 
    plt.close('all')

