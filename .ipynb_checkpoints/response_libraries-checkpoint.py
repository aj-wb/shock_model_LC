import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns_pal = sns.color_palette('Set1', n_colors=8, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)

# this is the function called by main script
# only this one loaded into main
# choose from options below & include here
def get_response_sp(hh_df,nsim):
    
    hh_df = apply_CCT_scaleup(hh_df)
    if nsim == 0: 
        plot_savings_vs_wage_loss(hh_df)
        plot_nominal_receipts_frac_income(hh_df)
        plot_nominal_receipts_vs_wage_loss(hh_df) 
        plot_nominal_receipts_vs_wage_loss_fractional(hh_df)
        scatter_plot_savings(hh_df)
        plot_savings_adequacy(hh_df)
        plot_pub_transfer_adequacy(hh_df,enrollees_only=True)

    return hh_df

# SP policies
def apply_CCT_scaleup(hh_df,scaleup=1.0,duration=3):
    # income in daily PPP/cap
    hh_df['CCT_scaleup'] = hh_df['cct4P']*scaleup
    hh_df['CCT_scaleup_duration'] = duration
    return hh_df




# Plot household savings vs. household income loss
def plot_savings_vs_wage_loss(hh_df):

    ######################################
    nbins = int(50)
    
    # total income <-- use to structure plot
    tot_hgt, _bins = np.histogram(hh_df.eval('pcinc_initial').clip(upper=500),bins=nbins,weights=hh_df['popwgt'])
    wid = (_bins[1]-_bins[0])

    sav_hgt = []; loss_hgt = []; transfer_hgt = []
    for n,b in enumerate(_bins): 
        try: 
            bin_slice = '(pcinc_initial>@b)&(pcinc_initial<=@_bins[@n+1])'
            sav_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*savings').sum()
                                 /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
            loss_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*wage_loss').sum()
                                  /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
            transfer_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*total_public').sum()
                                      /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))            
        except: pass

        
    # plot them
    ax = plt.bar(_bins[:-1],sav_hgt,width=wid,align='edge',linewidth=0,alpha=0.4,facecolor=sns_pal[2],label='Savings',zorder=91) 
    ax = plt.bar(_bins[:-1],transfer_hgt,bottom=sav_hgt,width=wid,align='edge',linewidth=0,alpha=0.5,facecolor=sns_pal[3],label='Public transfers',zorder=91) 
    #
    ax = plt.step(_bins[:-1]+wid,loss_hgt,linewidth=1,alpha=0.8,color=greys_pal[7],zorder=99,label='Wage loss (1 mo.)')
    ax = plt.step(_bins[:-1]+wid,[2*_h for _h in loss_hgt],linewidth=1,alpha=0.8,color=greys_pal[7],zorder=99,linestyle='--',label='Wage loss (2 mos.)')
    ax = plt.step(_bins[:-1]+wid,[3*_h for _h in loss_hgt],linewidth=1,alpha=0.8,color=greys_pal[7],zorder=99,linestyle=':',label='Wage loss (3 mos.)')

    plt.legend()
        
    plt.xlabel('Income per cap, pre-shock [PPP$ per month]',labelpad=8)
    plt.ylabel('Household value [PPP$]',labelpad=8)
    plt.xlim(0)
    
    plt.grid(False) 
    sns.despine(left=True,bottom=True)
    plt.plot([0,500],[0,0],color=greys_pal[4],lw=1)
    plt.savefig('figs/savings_vs_wage_loss.pdf',format='pdf',bbox_inches='tight') 

    plt.yscale('log')    
    plt.legend(loc='upper left')
        
    plt.draw()
    plt.savefig('figs/savings_vs_wage_loss_log.pdf',format='pdf',bbox_inches='tight') 

    plt.close('all')


######################################
# this block creates 'figs/public_transfer_hist.pdf'
# --> sp income as fraction of total income, for hh that receive public transfers
######################################
def plot_nominal_receipts_frac_income(hh_df):
    
    _ul = 20
    nbins = int(50)

    is_recipient = hh_df['total_public'] > 0

    # All public transfers (incl CCT)
    tot_hgt, _bins = np.histogram(hh_df.loc[is_recipient].eval('1E2*(total_public/hhinc)').clip(upper=25),
                                  bins=nbins,weights=hh_df.loc[is_recipient,'popwgt'])
    wid = (_bins[1]-_bins[0])

    # CCT only
    cct_hgt, _ = np.histogram(hh_df.loc[is_recipient].eval('1E2*(cct4P/hhinc)').clip(upper=25),
                              bins=_bins,weights=hh_df.loc[is_recipient,'popwgt'])

    # plot them
    ax = plt.bar(_bins[:-1],tot_hgt,width=wid,align='edge',linewidth=0,alpha=0.5,facecolor=sns_pal[2],label='All public transfers')
    ax = plt.bar(_bins[:-1],cct_hgt,width=wid,align='edge',linewidth=0,alpha=0.8,facecolor=sns_pal[2],label='CCT (incl. 4Ps)')
    #ax = plt.step(_bins[:-1]+wid,cct_hgt,linewidth=0.5,alpha=0.4)

    plt.legend()
        
    plt.xlabel('Fraction of household income [%]',labelpad=8)
    plt.ylabel('Beneficiaries [millions]',labelpad=8)
    plt.yticks(0.5*np.array(range(1,4)))
    plt.xlim(0)
    
    plt.grid(False) 
    sns.despine(left=True)
    plt.savefig('figs/public_transfer_hist.pdf',format='pdf',bbox_inches='tight') 
    plt.close('all')



######################################
# this block creates 'figs/income_hist_with_transfers.pdf'
# - x ax: pc income pre-shock
# - y ax: hist with hh transfers (CCT & other public) and wage loss in PPP$
######################################
def plot_nominal_receipts_vs_wage_loss(hh_df):
    nbins = int(50)

    # total income <-- use to structure plot
    tot_hgt, _bins = np.histogram(hh_df.eval('pcinc_initial').clip(upper=500),bins=nbins,weights=hh_df['popwgt'])
    wid = (_bins[1]-_bins[0])

    other_hgt = []; cct_hgt  = []; loss_hgt = []
    for n,b in enumerate(_bins): 
        try: 
            bin_slice = '(pcinc_initial>@b)&(pcinc_initial<=@_bins[@n+1])'
            other_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*(total_public-cct4P)/hhsize').sum()
                                   /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
            cct_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*(cct4P/hhsize)').sum()
                                 /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
            loss_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('-1*popwgt*(wage_loss/hhsize)').sum()
                                  /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
        except: pass

        
    # plot them
    ax = plt.bar(_bins[:-1],cct_hgt,width=wid,align='edge',linewidth=0,alpha=0.8,facecolor=sns_pal[2],label='CCT (incl. 4Ps)')
    ax = plt.bar(_bins[:-1],other_hgt,width=wid,align='edge',linewidth=0,alpha=0.4,bottom=cct_hgt,facecolor=sns_pal[2],label='Other public transfers')   
    ax = plt.bar(_bins[:-1],loss_hgt,width=wid,align='edge',linewidth=0,alpha=0.4,facecolor=sns_pal[0],label='COVID shock')

    plt.legend()
        
    plt.xlabel('Income per cap, pre-shock [PPP$ per month]',labelpad=8)
    plt.ylabel('Value [PPP$ per cap & month]',labelpad=8)
    plt.yticks(10*np.array(range(-8,2)))
    plt.ylim(-70,10)
    plt.xlim(0)

    plt.grid(False) 
    sns.despine(left=True,bottom=True)
    plt.plot([0,500],[0,0],color=greys_pal[4],lw=1)
    plt.savefig('figs/income_hist_with_transfers.pdf',format='pdf',bbox_inches='tight') 
    plt.close('all')


######################################
# this block creates 'figs/income_hist_with_transfers_fractional.pdf'
# - x ax: pc income pre-shock
# - y ax: hist with hh transfers (CCT & other public) and wage loss as % of total income
######################################
def plot_nominal_receipts_vs_wage_loss_fractional(hh_df):
    nbins = int(50)

    # total income <-- use to structure plot
    tot_hgt, _bins = np.histogram(hh_df.eval('pcinc_initial').clip(upper=500),bins=nbins,weights=hh_df['popwgt'])
    wid = (_bins[1]-_bins[0])

    other_hgt = []; cct_hgt  = []; loss_hgt = []; net_hgt = []
    for n,b in enumerate(_bins): 
        try: 
            bin_slice = '(pcinc_initial>@b)&(pcinc_initial<=@_bins[@n+1])'
            other_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('1E2*popwgt*(total_public-cct4P)/hhinc').sum()
                                   /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
            cct_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('1E2*popwgt*(cct4P)/hhinc').sum()
                                 /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
            loss_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('-1*1E2*popwgt*(wage_loss)/hhinc').sum()
                                  /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
            net_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('1E2*popwgt*(total_public-wage_loss)/hhinc').sum()
                                 /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
        except: pass

        
    # plot them

    #ax = plt.bar(_bins[:-1],tot_hgt,width=wid,align='edge',linewidth=0,alpha=0.5,facecolor=sns_pal[2],label='Total income')
    # ^ income hist

    ax = plt.bar(_bins[:-1],cct_hgt,width=wid,align='edge',linewidth=0,alpha=0.8,facecolor=sns_pal[2],label='CCT (incl. 4Ps)')
    ax = plt.bar(_bins[:-1],other_hgt,width=wid,align='edge',linewidth=0,alpha=0.4,bottom=cct_hgt,facecolor=sns_pal[2],label='Other public transfers')   
    ax = plt.bar(_bins[:-1],loss_hgt,width=wid,align='edge',linewidth=0,alpha=0.4,facecolor=sns_pal[0],label='COVID shock')
    ax = plt.step(_bins[:-1]+wid,net_hgt,linewidth=1,alpha=0.8,color=greys_pal[7],zorder=99,label='Net effect')
    
    plt.legend()#loc='upper right')
        
    plt.xlabel('Income per cap, pre-shock [PPP$ per month]',labelpad=8)
    plt.ylabel('Value [% of income]',labelpad=8)
    #plt.yticks(2*np.array(range(-3,4)))
    #plt.ylim(-6,6)
    #plt.gca().xaxis.set_ticks_position('top')
    #plt.gca().xaxis.tick_top()
    plt.xlim(0)

    plt.grid(False) 
    sns.despine(left=True,bottom=True)
    plt.plot([0,500],[0,0],color=greys_pal[4],lw=1)
    plt.savefig('figs/income_hist_with_transfers_fractional.pdf',format='pdf',bbox_inches='tight') 
    plt.close('all')


######################################
# this block creates 'figs/savings_frac_income_loss.pdf'
# plot savings & wages lost (1-3 mos) as fraction of hh income 
######################################
def get_rolling_average(_array,_bins,window=2):
    _avg = [];_avgbin=[]
    for n,v in enumerate(_array):
        num = 0; dnm = 0
        for i in range(int(-window),int(window+1)):
            try:
                if not np.isinf(_array[n+i]):
                    num += _array[n+i]
                    dnm += 1
            except: pass
        if dnm!=0: 
            _avg.append(num/dnm)
            _avgbin.append(_bins[n])
    return _avg,_avgbin


def plot_pub_transfer_adequacy(hh_df,enrollees_only=True):
    nbins = int(50)
    _ul = 500

    # total income <-- use to structure plot
    tot_hgt, _bins = np.histogram(hh_df.loc[hh_df.eval('total_public!=0'),'pcinc_initial'].clip(upper=_ul),bins=nbins,
                                  weights=hh_df.loc[hh_df.eval('total_public!=0'),'popwgt'])
    wid = (_bins[1]-_bins[0])
    
    for enrollees_only in [True]:#False]:
        cct_frac_hgt = []; oth_frac_hgt = []; tot_frac_hgt = [];
        for n,b in enumerate(_bins): 
            try:
                bin_slice = '(pcinc_initial>@b)&(pcinc_initial<=@_bins[@n+1]){}'.format('&(total_public!=0)' if enrollees_only else '')
                try: cct_frac_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*wage_loss').sum()
                                               /hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*cct4P').sum()))
                except ZeroDivisionError: cct_frac_hgt.append(0)
                #
                try: oth_frac_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*wage_loss').sum()
                                               /hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*(total_public-cct4P)').sum()))
                except ZeroDivisionError: cct_frac_hgt.append(0)
                #
                try: tot_frac_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*wage_loss').sum()
                                               /hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*total_public').sum()))
                except ZeroDivisionError: cct_frac_hgt.append(0)
                
            except: pass

        _ls = '-' if enrollees_only else ':'
        _ra, _rabins = get_rolling_average(cct_frac_hgt,_bins,window=4)
        plt.plot(_rabins[2:],_ra[2:],ls=_ls,color=sns_pal[2],
                 label='CCT{}'.format(' (enrollees only)' if enrollees_only else ''))
        plt.scatter(_bins[:-1],cct_frac_hgt,s=6,alpha=0.25,color=sns_pal[2])

        _ra, _rabins = get_rolling_average(oth_frac_hgt,_bins,window=4)
        plt.plot(_rabins[2:],_ra[2:],ls=_ls,color=sns_pal[3],
                 label='other transfers{}'.format(' (enrollees only)' if enrollees_only else ''))
        plt.scatter(_bins[:-1],oth_frac_hgt,s=6,alpha=0.25,color=sns_pal[3])

        _ra, _rabins = get_rolling_average(tot_frac_hgt,_bins,window=4)
        plt.plot(_rabins[2:],_ra[2:],ls=_ls,color=sns_pal[4],
                 label='total transfers{}'.format(' (enrollees only)' if enrollees_only else ''))
        plt.scatter(_bins[:-1],tot_frac_hgt,s=30,marker="1",alpha=0.25,color=sns_pal[4])

        
    
    plt.legend()
        
    plt.xlabel('Income per cap, pre-shock [PPP$ per month]',labelpad=8)
    plt.ylabel('Ratio (monthly wage loss/transfer)',labelpad=8)
    plt.xlim(0)

    plt.yscale('log')  
    plt.grid(False)#True,axis='y')
    sns.despine(left=True)
    plt.plot([0,_ul],[0,0],color=greys_pal[4],lw=1)
    plt.savefig('figs/public_transfer_adequacy{}.pdf'.format('_enrollees_only' if enrollees_only else ''),format='pdf',bbox_inches='tight') 
    plt.close('all')

######################################
# this block creates 'figs/savings_frac_income_loss.pdf'
# plot savings & wages lost (1-3 mos) as fraction of hh income 
######################################
def scatter_plot_savings(hh_df):
    nbins = int(50)

    # total income <-- use to structure plot
    tot_hgt, _bins = np.histogram(hh_df.eval('pcinc_initial').clip(upper=500),bins=nbins,weights=hh_df['popwgt'])
    wid = (_bins[1]-_bins[0])

    sav_frac_hgt = []; loss_frac_hgt = [];
    for n,b in enumerate(_bins): 
        try: 
            bin_slice = '(pcinc_initial>@b)&(pcinc_initial<=@_bins[@n+1])'
            sav_frac_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*1E2*savings/(hhinc*12)').sum()
                                      /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
            loss_frac_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*1E2*wage_loss/(hhinc*12)').sum()
                                       /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))

        except: pass

    plt.scatter(_bins[:-1],sav_frac_hgt,label='Savings',marker="1")
    plt.plot(_bins[:-1],loss_frac_hgt,label='Wage loss (1 mo.)')#,s=9)
    plt.plot(_bins[:-1],[2*b for b in loss_frac_hgt],label='Wage loss (2 mos.)')#,s=9)
    plt.plot(_bins[:-1],[3*b for b in loss_frac_hgt],label='Wage loss (3 mos.)')#,s=9)
    
    plt.legend()
        
    plt.xlabel('Income per cap, pre-shock [PPP$ per month]',labelpad=8)
    plt.ylabel('Value as % of annual income',labelpad=8)
    plt.xlim(0)

    plt.grid(False)    
    sns.despine(left=True,bottom=True)
    plt.plot([0,500],[0,0],color=greys_pal[4],lw=1)
    plt.savefig('figs/savings_frac_income_loss.pdf',format='pdf',bbox_inches='tight') 
    plt.close('all')


######################################
# this block creates 'figs/savings_adequacy.pdf'
######################################
def plot_savings_adequacy(hh_df):
    _ul = 500
    _uly = 7
    nbins = int(50)

    # total income <-- use to structure plot
    tot_hgt, _bins = np.histogram(hh_df.eval('pcinc_initial').clip(upper=_ul),bins=nbins,weights=hh_df['popwgt'])
    wid = (_bins[1]-_bins[0])

    sav_over_loss_hgt = [];
    for n,b in enumerate(_bins): 
        try: 
            bin_slice = '(pcinc_initial>@b)&(pcinc_initial<=@_bins[@n+1])&(wage_loss!=0)'
            sav_over_loss_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*(savings/wage_loss)').sum()
                                           /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
        except: pass
    ax = plt.bar(_bins[:-1],sav_over_loss_hgt,width=wid,align='edge',linewidth=0,alpha=0.8,facecolor=sns_pal[2])

    # annotate pop sizes
    for nmo in range(1,_uly):
        pop = round(hh_df.loc[hh_df.eval('@nmo*wage_loss>savings'),'popwgt'].sum(),1)
        plt.plot([0,_ul],[nmo,nmo],lw=0.7,linestyle=':',color=greys_pal[6])
        plt.annotate('{} mil. {}in {} month{}'.format(pop,('exhaust\nsavings ' if nmo == 1 else ''),nmo,('' if nmo == 1 else 's')),
                     xy=(510,nmo),fontsize=6,ha='left',va='center',annotation_clip=False)
              
    plt.xlabel('Income per cap, pre-shock [PPP$ per month]',labelpad=8)
    plt.ylabel('Savings adequacy [months]',labelpad=8)
    plt.xlim(0)
    plt.ylim(0,_uly)

    plt.grid(False)    
    sns.despine(left=True,bottom=True)
    plt.plot([0,_ul],[0,0],color=greys_pal[4],lw=1)
    plt.savefig('figs/savings_adequacy.pdf',format='pdf',bbox_inches='tight') 
    plt.close('all')
