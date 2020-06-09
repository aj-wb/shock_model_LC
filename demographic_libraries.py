import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns_pal = sns.color_palette('Set1', n_colors=8, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)
blues_pal = sns.color_palette('Blues_r', n_colors=6)
purples_pal = sns.color_palette('Purples_r', n_colors=6)

def summarize_demographics(hh_df):
    
    barplot_affected(hh_df)

#m2d = 12/365
def barplot_affected(hh_df):

    fig, ax = plt.subplots(nrows=2, ncols=1,figsize=(5,5))

    nbins = int(50)

    # total income <-- use to structure plot
    tot_hgt, _bins = np.histogram(hh_df.eval('pcinc_initial').clip(upper=500),bins=nbins,weights=hh_df['popwgt'])
    wid = (_bins[1]-_bins[0])

    bin_slice = '(pcinc_initial>@b)&(pcinc_initial<=@_bins[@n+1])'
    is_Aff = '&(income_loss!=0)'
    is_NAff = '&(income_loss==0)'

    affpop_hgt = []; naffpop_hgt = []; totpop_hgt = []
    for n,b in enumerate(_bins): 

        try:
            affpop_hgt.append(hh_df.loc[hh_df.eval(bin_slice+is_Aff),'popwgt'].sum())
            naffpop_hgt.append(hh_df.loc[hh_df.eval(bin_slice+is_NAff),'popwgt'].sum())
            totpop_hgt.append(hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum())
        except: pass
    

    # plot them
    plt.axes(ax[0])
    ax[0].bar(_bins[:-1],affpop_hgt,width=wid,align='edge',linewidth=0,alpha=0.6,facecolor=purples_pal[1],label='affected')
    ax[0].bar(_bins[:-1],naffpop_hgt,bottom=affpop_hgt,width=wid,align='edge',linewidth=0,alpha=0.5,facecolor=purples_pal[3],label='unaffected')
    ax[0].plot([0,500],[0,0],clip_on=False,color=greys_pal[5],lw=1)

    plt.legend(loc='upper right',labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)   
    #plt.xlabel('Income per cap, pre-shock [PPP$ per month]',labelpad=8)
    plt.ylabel('millions',labelpad=10)
    plt.xticks([100*_ for _ in range(1,6)],[])
    #plt.yticks(10*np.array(range(-8,2)))
    #plt.ylim(-70,10)
    plt.xlim(0)

    plt.grid(True,axis='x',alpha=0.5) 
    sns.despine(left=True,bottom=True)
    # plt.plot([0,500],[0,0],color=greys_pal[4],lw=1)
    # plt.savefig('figs/affected_pop_vs_income.pdf',format='pdf',bbox_inches='tight') 


    na_frac = [1E2*i/j for i,j in zip(naffpop_hgt,totpop_hgt)]
    a_frac = [1E2*i/j for i,j in zip(affpop_hgt,totpop_hgt)]
    plt.axes(ax[1])
    ax[1].bar(_bins[:-1],na_frac,width=wid,align='edge',linewidth=0,alpha=0.5,facecolor=purples_pal[3],label='unaffected')
    ax[1].bar(_bins[:-1],a_frac,bottom=na_frac,width=wid,align='edge',linewidth=0,alpha=0.6,facecolor=purples_pal[1],label='affected')
    ax[1].plot([0,500],[100,100],clip_on=False,color=greys_pal[5],lw=1)

    plt.xlim(0,500)

    plt.grid(True,axis='x',alpha=0.5)  
    plt.ylim(0,100)
    plt.xlabel('pre-COVID income [PPP$/cap/month]',labelpad=10)
    plt.xticks([100*_ for _ in range(1,6)])
    plt.yticks([20*_ for _ in range(0,5)],[20*_ for _ in range(1,6)[::-1]])
    plt.ylabel('as % of total',labelpad=10)

    fig.tight_layout()
    plt.subplots_adjust(wspace=None, hspace=0.015)
    plt.savefig('figs/affected_vs_income.pdf',format='pdf',bbox_inches='tight')     

    plt.close('all')
