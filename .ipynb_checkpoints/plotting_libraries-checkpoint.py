import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
sns_pal = sns.color_palette('Set1', n_colors=9, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)

m2d = 12/365

def plot_income_distribution(hh_df):

    explore_poverty_mysteries(hh_df,scaleup_CCT=0)
    explore_poverty_mysteries(hh_df,scaleup_CCT=0,mirror_subsistence=True)
    explore_poverty_mysteries(hh_df,scaleup_CCT=1)
    explore_poverty_mysteries(hh_df,scaleup_CCT=1,mirror_subsistence=True)

    # sub-routines
    plot_income_hist(hh_df)
    plot_income_hist(hh_df,use_expenditures=True)
    #
    plot_income_scatter(hh_df)
    plot_income_scatter(hh_df,with_migration=False)


def plot_income_hist(hh_df,use_expenditures=False):
    _fom = 'inc'; _label = 'Income'
    if use_expenditures:
        _fom = 'exp'
        _label = 'Expenditures'

    _ul = 20
    nbins = int(50)

    # Income dist before disaster
    ci_hgt, _bins = np.histogram((m2d*hh_df['pc'+_fom+'_initial']).clip(upper=_ul),bins=nbins,weights=hh_df['popwgt'])
    wid = (_bins[1]-_bins[0])/2

    # Income dist after disaster
    cf_hgt, _ = np.histogram((m2d*hh_df['pc'+_fom+'_final']).clip(upper=_ul),bins=_bins,weights=hh_df['popwgt'])
    
    # plot them
    ax = plt.bar(_bins[:-1],ci_hgt,width=wid,align='edge',linewidth=0,alpha=0.6,facecolor=sns_pal[1])
    ax = plt.bar(_bins[:-1]-wid,cf_hgt,width=wid,align='edge',linewidth=0,alpha=0.7,facecolor=sns_pal[4])

    # annotate shifts
    thresholds = [0,1.9,3.2,5.5,15,1E9]
    use_integers = True
    for n,l in enumerate(thresholds):
        if l == 0: continue

        if l < _ul: plt.plot([l,l],[0,9],color=greys_pal[7],lw=1.,ls=':')
        popi = round(hh_df.loc[hh_df.eval('(@m2d*pc'+_fom+'_initial<=@l)&(@m2d*pc'+_fom+'_initial>@thresholds[@n-1])'),'popwgt'].sum(),1)
        popf = round(hh_df.loc[hh_df.eval('(@m2d*pc'+_fom+'_final<=@l)&(@m2d*pc'+_fom+'_final>@thresholds[@n-1])'),'popwgt'].sum(),1)
        dp = round(popf-popi,1)

        if use_integers:
            popi = int(round(popi,0))
            popf = int(round(popf,0))
        
        _anno = 'i: {}\nf: {}\n'.format(popi,popf)+r'$\Delta$: {}'.format(dp)
        plt.annotate(_anno,xy=(min(17.5,thresholds[n-1]+(l-thresholds[n-1])/2),8.25),ha='center',va='bottom',fontsize=9,color=greys_pal[7],annotation_clip=False)

    plt.xlabel(_label+' [PPP$ per cap, day]',labelpad=8)
    plt.ylabel('Population [millions]',labelpad=8)
    plt.xlim(0)
    plt.ylim(0,8.5)
    plt.yticks([n for n in range(1,9)])

    sns.despine(left=True)
    plt.savefig('figs/{}_hist.pdf'.format(_label.lower()),format='pdf',bbox_inches='tight')
    plt.close('all')


def plot_income_scatter(hh_df,_ul=20,with_migration=True):

    nbins = 11
    dy = 11

    plt.scatter(m2d*hh_df['pcinc_initial'],m2d*hh_df['pcinc_final'],s=6,alpha=0.4)

    base_hgt, _bins = np.histogram((m2d*hh_df['pcinc_initial']).clip(upper=_ul),bins=[2*n for n in range(0,11)],weights=hh_df['popwgt'])
    shock_hgt, _ = np.histogram((m2d*hh_df['pcinc_final']).clip(upper=_ul),bins=_bins,weights=hh_df['popwgt'])  
    wid = (_bins[1]-_bins[0])
  
    for n,b in enumerate(_bins):
        
        plt.plot([b,b],[0,_ul],lw=0.6,color=greys_pal[5],ls=':',zorder=100)

        try: plt.annotate(r'$\Delta$P'+': {}%'.format(int(round(1E2*(shock_hgt[n]-base_hgt[n])/base_hgt[n],0))),
                          xy=(b+wid/2,np.e**(np.log(dy)/_ul*(b+wid/2))+_ul/2+1),color=greys_pal[7],fontsize=5.5,ha='center',va='bottom',annotation_clip=False)
        except: pass

        try:
            bin_slice = '(@m2d*pcinc_initial>@b)&(@m2d*pcinc_initial<=@_bins[@n+1])'
            i_i = m2d*hh_df.loc[hh_df.eval(bin_slice),['popwgt','pcinc_initial']].prod(axis=1).sum()/hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()

            if with_migration: bin_slice = '(@m2d*pcinc_final>@b)&(@m2d*pcinc_final<=@_bins[@n+1])'
            else: bin_slice = '(@m2d*pcinc_initial>@b)&(@m2d*pcinc_initial<=@_bins[@n+1])'
            i_f = m2d*hh_df.loc[hh_df.eval(bin_slice),['popwgt','pcinc_final']].prod(axis=1).sum()/hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()
        
            di = round(1E2*(i_f-i_i)/i_i,1)
            plt.annotate(r'$\Delta$i'+': {}%'.format(di),
                         xy=(b+wid/2,np.e**(np.log(dy)/_ul*(b+wid/2))+_ul/2+0.2),color=greys_pal[7],fontsize=5.5,ha='center',va='bottom',annotation_clip=False)

            if n < 5:
                if n == 0:
                    plt.annotate('poverty gap',style='italic',
                                 xy=(b+wid/2,np.e**(np.log(dy)/_ul*(b+wid/2))+_ul/2-1),color=greys_pal[7],fontsize=5.5,ha='center',va='bottom')

                pgap_i = round(1E2*(i_i-3.2)/3.2,1)
                plt.annotate(r'initial:'+'\n{}%'.format(pgap_i),
                             xy=(b+wid/20,np.e**(np.log(dy)/_ul*(b+wid/2))+_ul/2-2.2),color=greys_pal[7],fontsize=5.5,ha='left',va='bottom',annotation_clip=False)
                
                pgap_f = round(1E2*(i_f-3.2)/3.2,1)
                plt.annotate(r'final:'+'\n{}%'.format(pgap_f),
                             xy=(b+wid*19/20,np.e**(np.log(dy)/_ul*(b+wid/2))+_ul/2-3.4),color=greys_pal[7],fontsize=5.5,ha='right',va='bottom',annotation_clip=False)

                #pgap_i = None

        except: pass

    plt.xlabel('Income [PPP$ per cap, day]',labelpad=8)
    plt.ylabel('Income during shock [PPP$ per cap, day]',labelpad=8)

    plt.xlim(0,_ul)
    plt.ylim(0,_ul)

    plt.xticks([2*n for n in range(0,11)])
    sns.despine()

    plt.savefig('figs/income_scatter_{}.pdf'.format('with_bin_migrants' if with_migration else 'no_bin_migrants'),format='pdf',bbox_inches='tight') 
    plt.close('all')


def explore_poverty_mysteries(hh_df,scaleup_CCT=0,mirror_subsistence=False):    
    ul = 250
    nbins = int(50)

    # total income <-- use to structure plot
    tot_hgt, _bins = np.histogram(hh_df.eval('pcinc_initial').clip(upper=ul),bins=nbins,weights=hh_df['popwgt'])
    wid = (_bins[1]-_bins[0])

    m2d = 12/365
    bin_slice = '(pcinc_initial>@b)&(pcinc_initial<=@_bins[@n+1])'

    # households falling into poverty
    newpov = '&(@m2d*pcinc_initial>3.20)&(@m2d*pcinc_final<=3.20)'
    month1 = '&(@m2d*(hhinc-wage_loss+savings/1+cct4P*@scaleup_CCT)/hhsize<=3.20)'
    month2 = '&(@m2d*(hhinc-wage_loss+savings/1+cct4P*@scaleup_CCT)/hhsize>3.20)&(@m2d*(hhinc-wage_loss+savings/2+cct4P*@scaleup_CCT)/hhsize<=3.20)'
    month3 = '&(@m2d*(hhinc-wage_loss+savings/2+cct4P*@scaleup_CCT)/hhsize>3.20)&(@m2d*(hhinc-wage_loss+savings/3+cct4P*@scaleup_CCT)/hhsize<=3.20)'
    month4 = '&(@m2d*(hhinc-wage_loss+savings/3+cct4P*@scaleup_CCT)/hhsize>3.20)'
    # households falling into extreme poverty
    newsub = '&(@m2d*pcinc_initial>1.90)&(@m2d*pcinc_final<=1.90)'
    month1s = '&(@m2d*(hhinc-wage_loss+savings/1+cct4P*@scaleup_CCT)/hhsize<=1.90)'
    month2s = '&(@m2d*(hhinc-wage_loss+savings/1+cct4P*@scaleup_CCT)/hhsize>1.90)&(@m2d*(hhinc-wage_loss+savings/2+cct4P*@scaleup_CCT)/hhsize<=1.90)'
    month3s = '&(@m2d*(hhinc-wage_loss+savings/2+cct4P*@scaleup_CCT)/hhsize>1.90)&(@m2d*(hhinc-wage_loss+savings/3+cct4P*@scaleup_CCT)/hhsize<=1.90)'
    month4s = '&(@m2d*(hhinc-wage_loss+savings/3+cct4P*@scaleup_CCT)/hhsize>1.90)'    

    newpov_hgt = []; newpov_1m_hgt = []; newpov_2m_hgt = []; newpov_3m_hgt = []; newpov_4m_hgt = [];
    newsub_hgt = []; newsub_1m_hgt = []; newsub_2m_hgt = []; newsub_3m_hgt = []; newsub_4m_hgt = [];
    for n,b in enumerate(_bins[:-1]): 

        newpov_hgt.append(hh_df.loc[hh_df.eval(bin_slice+newpov),'popwgt'].sum())
        newpov_1m_hgt.append(hh_df.loc[hh_df.eval(bin_slice+newpov+month1),'popwgt'].sum())
        newpov_2m_hgt.append(hh_df.loc[hh_df.eval(bin_slice+newpov+month2),'popwgt'].sum())
        newpov_3m_hgt.append(hh_df.loc[hh_df.eval(bin_slice+newpov+month3),'popwgt'].sum())
        newpov_4m_hgt.append(hh_df.loc[hh_df.eval(bin_slice+newpov+month4),'popwgt'].sum())
        #
        newsub_1m_hgt.append(-1*hh_df.loc[hh_df.eval(bin_slice+newsub+month1s),'popwgt'].sum())
        newsub_2m_hgt.append(-1*hh_df.loc[hh_df.eval(bin_slice+newsub+month2s),'popwgt'].sum())
        newsub_3m_hgt.append(-1*hh_df.loc[hh_df.eval(bin_slice+newsub+month3s),'popwgt'].sum())
        newsub_4m_hgt.append(-1*hh_df.loc[hh_df.eval(bin_slice+newsub+month4s),'popwgt'].sum())
        #
    
    # plot them

    #ax = plt.step(_bins[:-1]+wid,newpov_hgt,linewidth=0.5,alpha=0.4)
    btm = 0
    lbl = '< 1 month ({} m.)'.format(round(np.sum(newpov_1m_hgt),1))
    ax = plt.bar(_bins[:-1],newpov_1m_hgt,bottom=btm,width=wid,align='edge',linewidth=0,alpha=0.5,facecolor=sns_pal[0],label=lbl)

    btm = newpov_1m_hgt
    lbl = '1-2 months ({} m.)'.format(round(np.sum(newpov_2m_hgt),1))
    ax = plt.bar(_bins[:-1],newpov_2m_hgt,width=wid,bottom=btm,align='edge',linewidth=0,alpha=0.5,facecolor=sns_pal[5],label=lbl)

    btm = [i+j for i,j in zip(newpov_1m_hgt,newpov_2m_hgt)]
    lbl = '2-3 months ({} m.)'.format(round(np.sum(newpov_3m_hgt),1))
    ax = plt.bar(_bins[:-1],newpov_3m_hgt,width=wid,bottom=btm,align='edge',linewidth=0,alpha=0.6,facecolor=sns_pal[2],label=lbl)

    btm = [i+j+k for i,j,k in zip(newpov_1m_hgt,newpov_2m_hgt,newpov_3m_hgt)]
    lbl = '> 3 months ({} m.)'.format(round(np.sum(newpov_4m_hgt),1))
    ax = plt.bar(_bins[:-1],newpov_4m_hgt,width=wid,bottom=btm,align='edge',linewidth=0,alpha=0.4,facecolor=sns_pal[1],label=lbl)

    # plot hh falling into subsistence
    if mirror_subsistence:
        btm = 0
        lbl = '< 1 month ({} m.)'.format(round(-1*np.sum(newsub_1m_hgt),1))
        ax = plt.bar(_bins[:-1],newsub_1m_hgt,bottom=btm,width=wid,align='edge',linewidth=0,alpha=0.25,facecolor=sns_pal[0],label=lbl)
        
        btm = newsub_1m_hgt
        lbl = '1-2 months ({} m.)'.format(-1*round(np.sum(newsub_2m_hgt),1))
        ax = plt.bar(_bins[:-1],newsub_2m_hgt,width=wid,bottom=btm,align='edge',linewidth=0,alpha=0.25,facecolor=sns_pal[5],label=lbl)

        btm = [i+j for i,j in zip(newsub_1m_hgt,newsub_2m_hgt)]
        lbl = '2-3 months ({} m.)'.format(-1*round(np.sum(newsub_3m_hgt),1))
        ax = plt.bar(_bins[:-1],newsub_3m_hgt,width=wid,bottom=btm,align='edge',linewidth=0,alpha=0.3,facecolor=sns_pal[2],label=lbl)
        
        btm = [i+j+k for i,j,k in zip(newsub_1m_hgt,newsub_2m_hgt,newsub_3m_hgt)]
        lbl = '> 3 months ({} m.)'.format(-1*round(np.sum(newsub_4m_hgt),1))
        ax = plt.bar(_bins[:-1],newsub_4m_hgt,width=wid,bottom=btm,align='edge',linewidth=0,alpha=0.2,facecolor=sns_pal[1],label=lbl)
        #

    plt.legend(title='Shock duration{}'.format('\n({}% CCT scaleup)'.format(int(1E2*scaleup_CCT)) if scaleup_CCT != 0 else ''))
    plt.xlabel('Income per cap, pre-shock [PPP$ per month]',labelpad=9)
    plt.ylabel('Impoverished population [millions]',labelpad=9)
    if not mirror_subsistence: plt.xlim(90,ul)
    else: plt.xlim(0,ul)

    plt.grid(False) 
    sns.despine(left=True,bottom=True)
    plt.plot([0,ul],[0,0],color=greys_pal[4],lw=1)
    plt.savefig('figs/new_poverty{}{}.pdf'.format('_net_sub' if mirror_subsistence else '',
                                                  '_CCT{}x'.format(scaleup_CCT) if scaleup_CCT != 0 else ''),format='pdf',bbox_inches='tight')
    plt.close('all') 



def plot_poverty_time_series(series,stacked=False,xlim=4):
    _dx= 0.08

    sub,pov,vul,sec,mc = series
    dy = 2 if stacked else 0.5
    
    stack=pd.DataFrame({'floor':0},index=sub.index).squeeze()

    # plot subsistence
    _col = sns_pal[0]
    if stacked: 
        plt.fill_between(sub.index,0,sub.max(axis=1),alpha=0.15,color=_col,zorder=90)
    plt.plot(sub.index,sub.mean(axis=1),color=_col,zorder=95)
    plt.fill_between(sub.index,sub.min(axis=1),sub.max(axis=1),alpha=0.4,color=_col,zorder=91)

    init = round(sub.iloc[0].mean(),1)
    fin = round(sub.iloc[-1].mean(),1)
    plt.annotate('{}%'.format(init),xy=(+_dx,sub.iloc[0].min()-dy),ha='left',va='top',color=_col,style='italic',annotation_clip=False)
    plt.annotate('extreme poverty\n{}%'.format(fin),xy=(xlim-_dx,sub.iloc[-1].min()-dy),ha='right',va='top',
                 color=_col,style='italic',annotation_clip=False)

    # plot moderate poverty
    _col = sns_pal[2]
    if stacked: 
        stack += sub.mean(axis=1)
        plt.fill_between(pov.index,sub.max(axis=1),stack+pov.min(axis=1),alpha=0.2,color=_col,zorder=91)
    plt.plot(pov.index,stack+pov.mean(axis=1),color=sns_pal[2],zorder=95)
    plt.fill_between(pov.index,stack+pov.min(axis=1),stack+pov.max(axis=1),alpha=0.4,color=_col,zorder=91)

    init = round(pov.iloc[0].mean(),1)
    fin = round(pov.iloc[-1].mean(),1)
    plt.annotate('{}%'.format(init),xy=(+_dx,stack.iloc[0]+pov.iloc[0].min()-dy),ha='left',va='top',color=_col,style='italic',annotation_clip=False)
    plt.annotate('poverty\n{}%'.format(fin),xy=(xlim-_dx,(stack.iloc[-1]+pov.iloc[-1].min()-dy)),ha='right',va='top',
                 color=_col,style='italic',annotation_clip=False)

    # plot vulnerable
    _col = sns_pal[4]
    if stacked: 
        stack += pov.mean(axis=1)
        plt.fill_between(vul.index,stack,stack+vul.min(axis=1),alpha=0.2,color=_col,zorder=91)
    plt.plot(vul.index,stack+vul.mean(axis=1),color=_col,zorder=95)
    plt.fill_between(vul.index,stack+vul.min(axis=1),stack+vul.max(axis=1),alpha=0.4,color=_col,zorder=91)

    init = round(vul.iloc[0].mean(),1)
    fin = round(vul.iloc[-1].mean(),1)
    plt.annotate('{}%'.format(init),xy=(+_dx,stack.iloc[0]+vul.iloc[0].min()-dy),ha='left',va='top',color=_col,style='italic',annotation_clip=False)
    plt.annotate('vulnerable\n{}%'.format(fin),xy=(xlim-_dx,(stack.iloc[-1]+vul.iloc[-1].min()-dy)),ha='right',va='top',
                 color=_col,style='italic',annotation_clip=False)

    # plot secure
    _col = sns_pal[1]
    if stacked: 
        stack += vul.mean(axis=1)
        plt.fill_between(sec.index,stack,stack+sec.min(axis=1),alpha=0.2,color=_col,zorder=91)
    plt.plot(sec.index,stack+sec.mean(axis=1),color=_col,zorder=95)
    plt.fill_between(sec.index,stack+sec.min(axis=1),stack+sec.max(axis=1),alpha=0.4,color=_col,zorder=91)

    init = round(sec.iloc[0].mean(),1)
    fin = round(sec.iloc[-1].mean(),1)
    plt.annotate('{}%'.format(init),xy=(+_dx,stack.iloc[0]+sec.iloc[0].min()-dy),ha='left',va='top',color=_col,style='italic',annotation_clip=False)
    plt.annotate('secure\n{}%'.format(fin),xy=(xlim-_dx,(stack.iloc[-1]+sec.iloc[-1].min()-dy)),ha='right',va='top',
                 color=_col,style='italic',annotation_clip=False)

    # plot mc
    _col = sns_pal[5]
    if stacked: 
        stack += sec.mean(axis=1)
        plt.fill_between(mc.index,stack,stack+mc.min(axis=1),alpha=0.2,color=_col,zorder=91)
    plt.plot(mc.index,stack+mc.mean(axis=1),color=_col,zorder=95)
    plt.fill_between(mc.index,stack+mc.min(axis=1),stack+mc.max(axis=1),alpha=0.4,color=_col,zorder=91)

    init = round(mc.iloc[0].mean(),1)
    fin = round(mc.iloc[-1].mean(),1)
    plt.annotate('{}%'.format(init),xy=(+_dx,stack.iloc[0]+mc.iloc[0].min()-dy),ha='left',va='top',color=_col,style='italic',annotation_clip=False)
    plt.annotate('middle class\n{}%'.format(fin),xy=(xlim-_dx,(stack.iloc[-1]+mc.iloc[-1].min()-dy)),ha='right',va='top',
                 color=_col,style='italic',annotation_clip=False)


    plt.xlim(0,xlim)
    if stacked: plt.ylim(0,100)
    else: plt.ylim(0)
    plt.xticks([n for n in range(1,5)])

    plt.xlabel('Crisis time $T_c$ [months]',labelpad=10)
    plt.ylabel('Percent population [%]',labelpad=10)

    plt.grid(False)
    #plt.grid(True,axis='y')
    sns.despine(left=True)
    plt.savefig('figs/poverty_incidence{}.pdf'.format('_stacked' if stacked else ''),format='pdf')
    plt.close('all')
