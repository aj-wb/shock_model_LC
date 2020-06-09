import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
from mc_storage_libraries import monte_carlo, load_income_impacts, load_consumption_time_series
from libraries.lib_country_dir import get_places_dict

sns_pal = sns.color_palette('Set1', n_colors=9, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)
blues_pal = sns.color_palette('Blues_r', n_colors=4)
reds_pal = sns.color_palette('Reds_r', n_colors=4)
cool_pal = sns.color_palette('RdGy', n_colors=6)
ryg_pal = sns.color_palette('RdYlGn', n_colors=15)

def plot_shock(scode):
    
    # barplots abs & rel value of losses, by income class
    plot_income_loss_by_channel(scode)

    # look at (nonlinear) poverty impacts by channel
    build_shock()

def plot_income_distributions(hh_df):

    ######################################################
    # compare income (expenditures, not implemented) before & during shock (HIST)
    # args:
    # 1) simulation results at hh level (df)
    # 2) MC results, by income class (array of dfs)
    # 3) use expenditures instead of income (bool)
    #    ^ not fully implemented, because we don't store consumption series, and consumption should be net of savings
    #
    plot_income_hist(hh_df)
    #plot_income_hist(hh_df,5*[None],use_expenditures=True)
    #
    
    ######################################################
    # income before (x), & during (y) shock, with poverty gap & other stats (SCATTER)
    # args:
    # 1) simulation results at hh level (df)
    # 2) calculate summary stats based on initial income, or final income (call these bin migrants) (bool) 
    #
    plot_income_scatter(hh_df)
    plot_income_scatter(hh_df,with_migration=False)
    #

    ######################################################
    # number of households falling into poverty (i < 3.2/, by income (x-ax) and sorted by time/savings adequacy (HIST)    
    # args:
    # 1) simulation results at hh level (df)
    # 2) CCT scaleup X times nominal CCT income (float)
    # 3) show households moving into extreme poverty, mirrored below x-axis
    #    ^ this to identify net flows into moderate, extreme poverty separately
    explore_poverty_mysteries(hh_df,scaleup_CCT=0)
    explore_poverty_mysteries(hh_df,scaleup_CCT=0,mirror_subsistence=True)
    explore_poverty_mysteries(hh_df,scaleup_CCT=1)
    #explore_poverty_mysteries(hh_df,scaleup_CCT=1,mirror_subsistence=True)
    #

def build_shock(fom='totpop_pov'):

    pal = monte_carlo().ichan_cols
    # create plot showing how impact channels add to affected, poverty

    initial = pd.read_csv('monte_carlo/inital_pop_by_class.csv'.format(fom))
    base = pd.read_csv('monte_carlo/base/{}.csv'.format(fom),index_col=0)[['vul','sec','mc']].sum(axis=1)

    W1E0R0 = pd.read_csv('monte_carlo/W1E0R0/{}.csv'.format(fom),index_col=0)[['vul','sec','mc']].sum(axis=1)
    W0E1R0 = pd.read_csv('monte_carlo/W0E1R0/{}.csv'.format(fom),index_col=0)[['vul','sec','mc']].sum(axis=1)
    W0E0R1 = pd.read_csv('monte_carlo/W0E0R1/{}.csv'.format(fom),index_col=0)[['vul','sec','mc']].sum(axis=1)

    W1E1R0 = pd.read_csv('monte_carlo/W1E1R0/{}.csv'.format(fom),index_col=0)[['vul','sec','mc']].sum(axis=1)
    W0E1R1 = pd.read_csv('monte_carlo/W0E1R1/{}.csv'.format(fom),index_col=0)[['vul','sec','mc']].sum(axis=1)
    W1E0R1 = pd.read_csv('monte_carlo/W1E0R1/{}.csv'.format(fom),index_col=0)[['vul','sec','mc']].sum(axis=1)


    wid = 0.8; alp = 0.6; lw = 0.0
    plt.barh(7,W0E0R1.mean(),facecolor=pal['remits'],lw=lw,height=wid,alpha=alp)
    plt.annotate(' {} mil.'.format(round(W0E0R1.mean(),1)),xy=(W0E0R1.mean(),7+wid/2),va='center',ha='left',color=greys_pal[6])
    plt.barh(6,W0E1R0.mean(),facecolor=pal['entrep'],lw=lw,height=wid,alpha=alp)
    plt.annotate(' {} mil.'.format(round(W0E1R0.mean(),1)),xy=(W0E1R0.mean(),6+wid/2),va='center',ha='left',color=greys_pal[6])
    plt.barh(5,W1E0R0.mean(),facecolor=pal['nonag_wage'],lw=lw,height=wid,alpha=alp)
    plt.annotate(' {} mil.'.format(round(W1E0R0.mean(),1)),xy=(W1E0R0.mean(),5+wid/2),va='center',ha='left',color=greys_pal[6])
    #

    plt.barh(3,W0E1R1.mean(),facecolor=pal['entrep'],lw=lw,height=wid*2/3,alpha=alp)
    plt.barh(3+wid/3,W0E1R1.mean(),facecolor=pal['remits'],lw=lw,height=wid*2/3,alpha=alp)
    plt.annotate(' {} mil.'.format(round(W0E1R1.mean(),1)),xy=(W0E1R1.mean(),3+wid/2),va='center',ha='left',color=greys_pal[6])

    plt.barh(2,W1E0R1.mean(),facecolor=pal['nonag_wage'],lw=lw,height=wid*2/3,alpha=alp)
    plt.barh(2+wid/3,W1E0R1.mean(),facecolor=pal['remits'],lw=lw,height=wid*2/3,alpha=alp)
    plt.annotate(' {} mil.'.format(round(W1E0R1.mean(),1)),xy=(W1E0R1.mean(),2+wid/2),va='center',ha='left',color=greys_pal[6])

    plt.barh(1,W1E1R0.mean(),facecolor=pal['nonag_wage'],lw=lw,height=wid*2/3,alpha=alp)
    plt.barh(1+wid/3,W1E1R0.mean(),facecolor=pal['entrep'],lw=lw,height=wid*2/3,alpha=alp)
    plt.annotate(' {} mil.'.format(round(W1E1R0.mean(),1)),xy=(W1E1R0.mean(),1+wid/2),va='center',ha='left',color=greys_pal[6])
    
    # plt.bar(9,base.mean(),facecolor=pal['remits'],lw=lw,width=wid/3,alpha=alp)
    # plt.bar(9+wid/3,base.mean(),facecolor=pal['entrep'],lw=lw,width=wid/3,alpha=alp)
    # plt.bar(9+wid*2/3,base.mean(),facecolor=pal['nonag_wage'],lw=lw,width=wid/3,alpha=alp)
    plt.barh([1,2,3,5,6,7],6*[base.mean()],facecolor="None",lw=1,height=wid,edgecolor=greys_pal[7],alpha=0.3)

    plt.grid(True,axis='x',alpha=0.3)

    # plt.legend(loc='upper left',labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)

    plt.yticks([_+wid/2 for _ in range(1,8)],['wages &\nentrepreneurial','wages &\nremittances','entrepreneurial &\n remittances',
                                            '','wage\neffect','entrepreneurial\neffect','remittance\neffect'],va='center',ha='right')
    plt.xlabel('Poverty increase [mil.]',labelpad=10)
    plt.ylim(0.75,8.05)

    sns.despine()
    plt.savefig('figs/build_a_crisis.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')    
    

def plot_regional_poverty(scode):
    mc = monte_carlo(0,'base')

    pov_init = pd.read_csv('monte_carlo/regional_poverty.csv',index_col=0,dtype='float',header=None).fillna(0)
    pov_init.columns = ['init']
    pov_init.index = pov_init.index.astype('int')
    #
    pov_covid = pd.read_csv('monte_carlo/{}/regional_pov_covid.csv'.format(scode),index_col=0,dtype='float').mean(axis=0).T.to_frame(name='covid')
    pov_covid.index = pov_covid.index.astype('int')
    #
    pov_ESP = pd.read_csv('monte_carlo/{}/regional_pov_ESP_eligerr50.csv'.format(scode),index_col=0,dtype='float').mean(axis=0).T.to_frame(name='ESP')
    pov_ESP.index = pov_ESP.index.astype('int')
    #
    pov = pd.concat([pov_init,pov_covid,pov_ESP],axis=1)
    pov.index.name = 'region'
    pov = pov.reset_index()
    prov_code,reg_code = get_places_dict('PH')
    pov['region'].replace(reg_code,inplace=True)
    pov = pov.reset_index(drop=True).set_index('region')

    pov = pov.sort_values(by='ESP',ascending=True)
    pov.to_csv('csv/regional_poverty.csv')

    for _n,_ in enumerate(pov.index):
        _lbl = 'pre-COVID (total = {} mil.)'.format(round(pov['init'].sum(),1)) if _n == 0 else ''
        plt.plot([3*_n,3*_n+2],[pov.loc[_,'init'],pov.loc[_,'init']],lw=1,color=greys_pal[6],zorder=90,label=_lbl)

    plt.bar([3*_ for _ in range(len(pov.index))],pov['covid'],facecolor=mc.sns_pal[2],lw=0,width=1,alpha=0.6,label='COVID shock ({} mil.)'.format(round(pov['covid'].sum(),1)),zorder=80)
    plt.bar([3*_+1 for _ in range(len(pov.index))],pov['ESP'],facecolor=mc.sns_pal[1],lw=0,width=1,alpha=0.6,label='with SAP benefits ({} mil.)'.format(round(pov['ESP'].sum(),1)),zorder=80)

    plt.grid(True,axis='y',alpha=0.3)

    plt.legend(loc='upper left',labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)

    plt.xticks([3*_+1 for _ in range(len(pov.index))],pov.index,rotation=90)
    plt.ylabel('Poverty incidence [mil.]',labelpad=10)

    sns.despine(left=True)
    plt.savefig('figs/regional_poverty.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')    


def plot_income_loss_by_channel(scode):
    pal = monte_carlo(0,'base').ichan_cols
    _w = 0.75
    _fs = 8

    #flag
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(8,5))
    plt.axes(ax[0])

    rel = 'loss' # 'lossfrac' returns fraction *of each channel* 
    agwage = pd.read_csv('monte_carlo/{}/ag_wage_{}.csv'.format(scode,rel),index_col=0)
    nagwage = pd.read_csv('monte_carlo/{}/nonag_wage_{}.csv'.format(scode,rel),index_col=0)
    remits = pd.read_csv('monte_carlo/{}/remits_{}.csv'.format(scode,rel),index_col=0)
    entrep = pd.read_csv('monte_carlo/{}/entrep_{}.csv'.format(scode,rel),index_col=0)
    #
    tot_loss= pd.read_csv('monte_carlo/{}/tot_loss.csv'.format(scode),index_col=0)
    # load initial income [mil. PPP/month]
    init_inc_aff = pd.read_csv('monte_carlo/{}/tot_inc_initial_affected.csv'.format(scode),index_col=0)
    # load affected population
    aff_pop = pd.read_csv('monte_carlo/{}/pop_aff.csv'.format(scode),index_col=0)
    # load total population

    # plot losses in PPP/cap
    btm = [0 for _ in agwage.columns]
    for ncl, cl in enumerate(['sub','pov','vul','sec','mc']):

        plt.bar(ncl,nagwage[cl].mean()/aff_pop[cl].mean(),bottom=btm[ncl],color=pal['nonag_wage'],width=_w,alpha=0.6,linewidth=0,label=('wages (non-ag)' if ncl == 0 else ''))
        btm[ncl] += nagwage[cl].mean()/aff_pop[cl].mean()

        plt.bar(ncl,entrep[cl].mean()/aff_pop[cl].mean(),bottom=btm[ncl],color=pal['entrep'],width=_w,alpha=0.6,linewidth=0,label=('entrepreneurial' if ncl == 0 else ''))
        btm[ncl] += entrep[cl].mean()/aff_pop[cl].mean()

        plt.bar(ncl,remits[cl].mean()/aff_pop[cl].mean(),bottom=btm[ncl],color=pal['remits'],width=_w,alpha=0.6,linewidth=0,label=('intl. remittances' if ncl == 0 else ''))
        btm[ncl] += remits[cl].mean()/aff_pop[cl].mean()

        plt.bar(ncl,agwage[cl].mean()/aff_pop[cl].mean(),bottom=btm[ncl],color=pal['ag_wage'],width=_w,alpha=0.6,linewidth=0,label=('agricultural wages' if ncl == 0 else ''))
        btm[ncl] += agwage[cl].mean()/aff_pop[cl].mean()

        # annotate with range
        _low = int(round(tot_loss[cl].quantile(0.25)/aff_pop[cl].mean()))
        _high = int(round(tot_loss[cl].quantile(0.75)/aff_pop[cl].mean()))

        plt.annotate('\${}$\endash${}'.format(_low,_high),xy=(_w/2+ncl,btm[ncl]),fontsize=_fs,color=greys_pal[7],ha='center',va='bottom')

    plt.xticks([_w/2+_ for _ in range(0,5)],['extreme\npoverty','poverty','vulnerable','secure','middle\nclass'],fontsize=_fs,rotation=0)
    plt.xlim(-0.1,4+_w+0.1)
    plt.yticks([_*50 for _ in range(0,6)],fontsize=_fs)
    plt.ylabel('Value [PPP$/cap/month]',labelpad=10,linespacing=1.75,fontsize=_fs)

    plt.legend(loc='upper left',labelspacing=0.75,ncol=1,fontsize=_fs,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
    plt.grid(False);plt.grid(True,axis='y',alpha=0.4)
    sns.despine(left=True)
    # plt.savefig('figs/income_{}_by_channel.pdf'.format(rel),format='pdf',bbox_inches='tight')
    # plt.close('all')


    # plot fractional losses
    plt.axes(ax[1])
    btm = [0 for _ in agwage.columns]
    for ncl, cl in enumerate(['sub','pov','vul','sec','mc']):

        plt.bar(ncl,1E2*nagwage[cl].mean()/init_inc_aff[cl].mean(),bottom=btm[ncl],color=pal['nonag_wage'],width=_w,alpha=0.6,linewidth=0,label=('non-ag wages' if ncl == 0 else ''))
        btm[ncl] += 1E2*nagwage[cl].mean()/init_inc_aff[cl].mean()

        plt.bar(ncl,1E2*entrep[cl].mean()/init_inc_aff[cl].mean(),bottom=btm[ncl],color=pal['entrep'],width=_w,alpha=0.6,linewidth=0,label=('entrepreneurial income' if ncl == 0 else ''))
        btm[ncl] += 1E2*entrep[cl].mean()/init_inc_aff[cl].mean()

        plt.bar(ncl,1E2*remits[cl].mean()/init_inc_aff[cl].mean(),bottom=btm[ncl],color=pal['remits'],width=_w,alpha=0.6,linewidth=0,label=('remittances' if ncl == 0 else ''))
        btm[ncl] += 1E2*remits[cl].mean()/init_inc_aff[cl].mean()

        plt.bar(ncl,1E2*agwage[cl].mean()/init_inc_aff[cl].mean(),bottom=btm[ncl],color=pal['ag_wage'],width=_w,alpha=0.6,linewidth=0,label=('ag wages' if ncl == 0 else ''))
        btm[ncl] += 1E2*agwage[cl].mean()/init_inc_aff[cl].mean()

        # annotate with range
        _low = int(round(1E2*tot_loss[cl].quantile(0.25)/init_inc_aff[cl].mean()))
        _high = int(round(1E2*tot_loss[cl].quantile(0.75)/init_inc_aff[cl].mean()))

        plt.annotate('{}$\endash${}%'.format(_low,_high),xy=(_w/2+ncl,btm[ncl]),fontsize=_fs,color=greys_pal[7],ha='center',va='bottom')

    plt.xticks([_w/2+_ for _ in range(0,5)],['extreme\npoverty','poverty','vulnerable','secure','middle\nclass'],fontsize=_fs,rotation=0)
    plt.xlim(-0.1,4+_w+0.1)
    plt.yticks([_*10 for _ in range(0,5)],fontsize=_fs)
    plt.ylabel('Percentage of total income [%]',labelpad=10,linespacing=1.5,fontsize=_fs)
    #plt.legend(loc='upper left',labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
    plt.grid(False);plt.grid(True,axis='y',alpha=0.4)
    #plt.grid(True,axis='y')
    sns.despine(left=True)
    plt.savefig('figs/income_loss_by_channel.pdf'.format(rel),format='pdf',bbox_inches='tight')
    plt.close('all')

######################################################
# Plotting functions (income distributions)
def plot_income_hist(hh_df,shock_code='base',use_expenditures=False):
    mc = monte_carlo(0,'base')
    
    ###############################################
    # function can plot income or expenditures
    # --> based on flag use_expenditures above
    _fom = 'inc'; _label = 'Income'
    if use_expenditures:
        _fom = 'exp'
        _label = 'Expenditures'

    # use central value of MC series
    ts = load_consumption_time_series (shock_code)
    classes = {'sub':[ts[0].T,-1E9,1.9],
               'pov':[ts[1].T,1.9,3.2],
               'vul':[ts[2].T,3.2,5.5],
               'sec':[ts[3].T,5.5,15.],
               'mc':[ts[4].T,15.,1E9]}

    ###############################################
    # upper limit, binning of histogram
    _ul = 20
    nbins = int(50) 

    # Income dist before disaster
    ci_hgt, _bins = np.histogram((mc.m2d*hh_df['pc'+_fom+'_initial']).clip(upper=_ul),bins=nbins,weights=hh_df['popwgt'])
    wid = (_bins[1]-_bins[0])/2

    # Income dist after disaster
    cf_hgt, _ = np.histogram((mc.m2d*hh_df['pc'+_fom+'_final']).clip(upper=_ul),bins=_bins,weights=hh_df['popwgt'])
    
    # plot them
    ax = plt.bar(_bins[:-1],ci_hgt,width=wid,align='edge',linewidth=0,alpha=0.4,facecolor=cool_pal[5],label='2015 FIES')
    ax = plt.bar(_bins[:-1]-wid,cf_hgt,width=wid,align='edge',linewidth=0,alpha=0.4,facecolor=cool_pal[0],label='COVID shock')

    # annotate shifts
    pop_aff,frac_aff,tot_loss,di_tot,di_aff,affpop_sub,affpop_pov,affpop_vul = load_income_impacts(shock_code)
    
    _y = 0.0
    _dy = {'sub':9.2,'pov':7.8,'vul':6.3,'sec':3.7,'mc':0.5}
    lbl = {'sub':'extreme poverty','pov':'poverty','vul':'vulnerable','sec':'secure','mc':'middle class'}
    for _c in ['sub','pov','vul','sec','mc']:
        _mc,_min,_max = classes[_c]

        # segment plot
        # if _c != 'sub': 
        plt.plot([max(0.05,_min),max(0.05,_min)],[0,_dy[_c]+1.0],color=greys_pal[6],lw=1.4,ls='-',clip_on=False)
        plt.plot([max(0.05,_min),max(0.05,_min)+0.1],[_dy[_c]+1.0,_dy[_c]+1.10],color=greys_pal[6],lw=1.4,ls='-',clip_on=False)


        # annotate population shifts
        # This loads population values from MC (consumption series)
        if not use_expenditures: print('\n\nusing consumption (net sav), month = 12 for income hist')

        tot_pop = hh_df['popwgt'].sum()
        popi = round(1E-2*tot_pop*_mc[0].mean(),1)
        popf = round(1E-2*tot_pop*_mc[12].mean(),1)
        popf_min = round(1E-2*tot_pop*_mc[12].quantile(0.25),1) # popf-pop_err
        popf_max = round(1E-2*tot_pop*_mc[12].quantile(0.75),1) # popf+pop_err

        # get delta
        dp = round(popf-popi,1)
        dp_max = round(popf_max-popi,1)
        dp_min = round(popf_min-popi,1)

        # pop frac affected
        frac_aff_min = int(round(frac_aff[_c].quantile(0.25)))
        frac_aff_max = int(round(frac_aff[_c].quantile(0.75)))

        if frac_aff_min != frac_aff_max: _anno = r'{}% $\endash$ {}%'.format(frac_aff_min,frac_aff_max)
        else: _anno = r'{}%'.format(int(round(frac_aff[_c].mean())))
        _anno += ' of {} m. affected'.format(popi)+'\n'

        _anno += 'final pop: {} $\endash$ {} m.\n'.format(popf_min,popf_max)
        _anno += 'net shift: {}{} $\endash$ {}{} m.'.format('+' if dp_min > 0 else '',dp_min,'+' if dp_max > 0 else '',dp_max)

        # income loss among affected
        # pcinc_init = mc.m2d*(hh_df.loc[hh_df['initial_class']==_c,['pcinc_initial','popwgt']].prod(axis=1).sum()
                          # /hh_df.loc[hh_df['initial_class']==_c,'popwgt'].sum())

        # di_aff_min = round(mc.m2d*di_aff[_c].quantile(0.25),1)
        # di_aff_max = round(mc.m2d*di_aff[_c].quantile(0.75),1)

        # if di_aff_min == di_aff_max :
            # _anno += r'\${}0{}'.format(round(mc.m2d*di_aff[_c].mean(),1),'/cap/day')
            # _anno += r' ({}%)'.format(int(1E2*mc.m2d*di_aff[_c].mean()/pcinc_init))+'lost \n'
        # else: 
            # _anno += r'\${}0 $\endash$ \${}0{}'.format(di_aff_min,di_aff_max,'/cap/day')
            # _anno += r' ({}% $\endash$ {}%)'.format(int(1E2*di_aff_min/pcinc_init),int(1E2*di_aff_max/pcinc_init))+' lost\n'



        # if use_expenditures:
        plt.annotate(lbl[_c],xy=(max(0,_min)+0.3,_y+_dy[_c]+1.05),ha='left',va='bottom',fontsize=8,color=greys_pal[7],annotation_clip=False,weight='bold')
        plt.annotate(_anno,xy=(max(0,_min)+.55,_y+_dy[_c]+0.94),ha='left',va='top',fontsize=8,color=greys_pal[7],annotation_clip=False,linespacing=1.5)
            

    plt.xlabel(_label+' [PPP$/cap/day]',labelpad=10)
    plt.ylabel('Population [millions]',labelpad=10)
    plt.xlim(0)
    plt.ylim(0,8.5)
    plt.yticks([n for n in range(1,9)])

    plt.grid(False)

    plt.legend(labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)

    sns.despine(left=True)
    plt.savefig('figs/{}_hist.pdf'.format(_label.lower()),format='pdf',bbox_inches='tight')
    plt.close('all')

def plot_income_scatter(hh_df,_ul=20,with_migration=True):
    mc = monte_carlo(0,'base')
    nbins = 11
    dy = 11

    plt.scatter(mc.m2d*hh_df.loc[hh_df.income_loss!=0,'pcinc_initial'],mc.m2d*hh_df.loc[hh_df.income_loss!=0,'pcinc_final'],s=6,alpha=0.4)

    base_hgt, _bins = np.histogram((mc.m2d*hh_df['pcinc_initial']).clip(upper=_ul),bins=[2*n for n in range(0,11)],weights=hh_df['popwgt'])
    shock_hgt, _ = np.histogram((mc.m2d*hh_df['pcinc_final']).clip(upper=_ul),bins=_bins,weights=hh_df['popwgt'])  
    wid = (_bins[1]-_bins[0])
  
    for n,b in enumerate(_bins):
        
        plt.plot([b,b],[0,_ul],lw=0.6,color=greys_pal[5],ls=':',zorder=100)

        try: plt.annotate(r'$\Delta$P'+': {}%'.format(int(round(1E2*(shock_hgt[n]-base_hgt[n])/base_hgt[n],0))),
                          xy=(b+wid/2,np.e**(np.log(dy)/_ul*(b+wid/2))+_ul/2+1),color=greys_pal[7],fontsize=5.5,ha='center',va='bottom',annotation_clip=False)
        except: pass

        try:
            bin_slice = '(@mc.m2d*pcinc_initial>@b)&(@mc.m2d*pcinc_initial<=@_bins[@n+1])'
            i_i = mc.m2d*hh_df.loc[hh_df.eval(bin_slice),['popwgt','pcinc_initial']].prod(axis=1).sum()/hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()

            if with_migration: bin_slice = '(@mc.m2d*pcinc_final>@b)&(@mc.m2d*pcinc_final<=@_bins[@n+1])'
            else: bin_slice = '(@mc.m2d*pcinc_initial>@b)&(@mc.m2d*pcinc_initial<=@_bins[@n+1])'
            i_f = mc.m2d*hh_df.loc[hh_df.eval(bin_slice),['popwgt','pcinc_final']].prod(axis=1).sum()/hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()
        
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

    plt.xlabel('Income [PPP$/cap/day]',labelpad=10)
    plt.ylabel('Income during shock [PPP$/cap/day]',labelpad=10)

    plt.xlim(0,_ul)
    plt.ylim(0,_ul)

    plt.xticks([2*n for n in range(0,11)])
    sns.despine()

    plt.savefig('figs/income_scatter_{}.pdf'.format('with_bin_migrants' if with_migration else 'no_bin_migrants'),format='pdf',bbox_inches='tight') 
    plt.close('all')

def explore_poverty_mysteries(hh_df,scaleup_CCT=0,mirror_subsistence=False): 
    mc = monte_carlo(0,'base')   
    ul = 250
    nbins = int(50)

    # total income <-- use to structure plot
    tot_hgt, _bins = np.histogram(hh_df.eval('pcinc_initial').clip(upper=ul),bins=nbins,weights=hh_df['popwgt'])
    wid = (_bins[1]-_bins[0])

    bin_slice = '(pcinc_initial>@b)&(pcinc_initial<=@_bins[@n+1])'

    # households falling into poverty
    newpov = '&(@mc.m2d*pcinc_initial>3.20)&(@mc.m2d*pcinc_final<=3.20)'
    month1 = '&(@mc.m2d*(hhinc-income_loss+savings/1+cct4P*@scaleup_CCT)/hhsize<=3.20)'
    month2 = '&(@mc.m2d*(hhinc-income_loss+savings/1+cct4P*@scaleup_CCT)/hhsize>3.20)&(@mc.m2d*(hhinc-income_loss+savings/2+cct4P*@scaleup_CCT)/hhsize<=3.20)'
    month3 = '&(@mc.m2d*(hhinc-income_loss+savings/2+cct4P*@scaleup_CCT)/hhsize>3.20)&(@mc.m2d*(hhinc-income_loss+savings/3+cct4P*@scaleup_CCT)/hhsize<=3.20)'
    month4 = '&(@mc.m2d*(hhinc-income_loss+savings/3+cct4P*@scaleup_CCT)/hhsize>3.20)'
    # households falling into extreme poverty
    newsub = '&(@mc.m2d*pcinc_initial>1.90)&(@mc.m2d*pcinc_final<=1.90)'
    month1s = '&(@mc.m2d*(hhinc-income_loss+savings/1+cct4P*@scaleup_CCT)/hhsize<=1.90)'
    month2s = '&(@mc.m2d*(hhinc-income_loss+savings/1+cct4P*@scaleup_CCT)/hhsize>1.90)&(@mc.m2d*(hhinc-income_loss+savings/2+cct4P*@scaleup_CCT)/hhsize<=1.90)'
    month3s = '&(@mc.m2d*(hhinc-income_loss+savings/2+cct4P*@scaleup_CCT)/hhsize>1.90)&(@mc.m2d*(hhinc-income_loss+savings/3+cct4P*@scaleup_CCT)/hhsize<=1.90)'
    month4s = '&(@mc.m2d*(hhinc-income_loss+savings/3+cct4P*@scaleup_CCT)/hhsize>1.90)'    

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
    ax = plt.bar(_bins[:-1],newpov_1m_hgt,bottom=btm,width=wid,align='edge',linewidth=0,alpha=0.5,facecolor=reds_pal[0],label=lbl)

    btm = newpov_1m_hgt
    lbl = '1-2 months ({} m.)'.format(round(np.sum(newpov_2m_hgt),1))
    ax = plt.bar(_bins[:-1],newpov_2m_hgt,width=wid,bottom=btm,align='edge',linewidth=0,alpha=0.5,facecolor=reds_pal[1],label=lbl)

    btm = [i+j for i,j in zip(newpov_1m_hgt,newpov_2m_hgt)]
    lbl = '2-3 months ({} m.)'.format(round(np.sum(newpov_3m_hgt),1))
    ax = plt.bar(_bins[:-1],newpov_3m_hgt,width=wid,bottom=btm,align='edge',linewidth=0,alpha=0.5,facecolor=reds_pal[2],label=lbl)

    btm = [i+j+k for i,j,k in zip(newpov_1m_hgt,newpov_2m_hgt,newpov_3m_hgt)]
    lbl = '> 3 months ({} m.)'.format(round(np.sum(newpov_4m_hgt),1))
    ax = plt.bar(_bins[:-1],newpov_4m_hgt,width=wid,bottom=btm,align='edge',linewidth=0,alpha=0.5,facecolor=reds_pal[3],label=lbl)

    # plot hh falling into subsistence
    if mirror_subsistence:
        btm = 0
        lbl = '< 1 month ({} m.)'.format(round(-1*np.sum(newsub_1m_hgt),1))
        ax = plt.bar(_bins[:-1],newsub_1m_hgt,bottom=btm,width=wid,align='edge',linewidth=0,alpha=0.35,facecolor=reds_pal[0],label=lbl)
        
        btm = newsub_1m_hgt
        lbl = '1-2 months ({} m.)'.format(-1*round(np.sum(newsub_2m_hgt),1))
        ax = plt.bar(_bins[:-1],newsub_2m_hgt,width=wid,bottom=btm,align='edge',linewidth=0,alpha=0.35,facecolor=reds_pal[1],label=lbl)

        btm = [i+j for i,j in zip(newsub_1m_hgt,newsub_2m_hgt)]
        lbl = '2-3 months ({} m.)'.format(-1*round(np.sum(newsub_3m_hgt),1))
        ax = plt.bar(_bins[:-1],newsub_3m_hgt,width=wid,bottom=btm,align='edge',linewidth=0,alpha=0.35,facecolor=reds_pal[2],label=lbl)
        
        btm = [i+j+k for i,j,k in zip(newsub_1m_hgt,newsub_2m_hgt,newsub_3m_hgt)]
        lbl = '> 3 months ({} m.)'.format(-1*round(np.sum(newsub_4m_hgt),1))
        ax = plt.bar(_bins[:-1],newsub_4m_hgt,width=wid,bottom=btm,align='edge',linewidth=0,alpha=0.35,facecolor=reds_pal[3],label=lbl)
        #

    plt.legend(title='Shock duration{}'.format('\n({}% CCT scaleup)'.format(int(1E2*scaleup_CCT)) if scaleup_CCT != 0 else ''),
            labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
    plt.xlabel('pre-COVID income [PPP$/cap/month]',labelpad=10)
    plt.ylabel('Impoverished population [millions]',labelpad=10)
    if not mirror_subsistence: plt.xlim(90,ul)
    else: plt.xlim(0,ul)

    plt.grid(False) 
    sns.despine(left=True,bottom=True)
    plt.plot([0,ul],[0,0],color=greys_pal[4],lw=1)
    plt.savefig('figs/new_poverty{}{}.pdf'.format('_net_sub' if mirror_subsistence else '',
                                                  '_CCT{}x'.format(scaleup_CCT) if scaleup_CCT != 0 else ''),format='pdf',bbox_inches='tight')
    plt.close('all') 

def plot_poverty_time_series(shock_code='base',stacked=False,xlim=4):
    sub,pov,vul,sec,mc = load_consumption_time_series(shock_code)

    _dx= 0.08
    dy = 2 if stacked else 0.5
    
    stack=pd.DataFrame({'floor':0},index=sub.index).squeeze()

    _res = {'sub':sub,'pov':pov,'vul':vul,'sec':sec,'mc':mc}
    _lbl = {'sub':'extreme poverty','pov':'poverty','vul':'vulnerable','sec':'secure','mc':'middle class'}
    _col = {'sub':sns_pal[0],'pov':sns_pal[2],'vul':sns_pal[4],'sec':sns_pal[1],'mc':sns_pal[5]}

    for _ic in ['sub','pov','vul','sec','mc']:

        # plot
        
        if stacked: 
            plt.fill_between(_res[_ic].index,stack,stack+_res[_ic].max(axis=1),alpha=0.15,color=_col[_ic],zorder=90)
        plt.plot(_res[_ic].index,stack+_res[_ic].mean(axis=1),color=_col[_ic],zorder=95,alpha=0.4)
        plt.fill_between(_res[_ic].index,stack+_res[_ic].min(axis=1),stack+_res[_ic].max(axis=1),alpha=0.4,color=_col[_ic],zorder=91)

        init = round(_res[_ic].iloc[0].mean(),1)
        fin = round(_res[_ic].iloc[-1].mean(),1)
        plt.annotate('{}%'.format(init),xy=(+_dx,stack.iloc[0]+_res[_ic].iloc[0].min()-dy),ha='left',va='top',color=_col[_ic],style='italic',annotation_clip=False)

        if stacked:
            __x = xlim-_dx
            __y = stack.iloc[-1]+_res[_ic].iloc[-1].min()-dy
        else:
            __x = xlim+_dx
            __y = _res[_ic].iloc[-1].mean()
        plt.annotate('{}\n{}%'.format(_lbl[_ic],fin),xy=(__x,__y),
                     ha=('right' if stacked else 'left'),
                     va=('top' if stacked else 'center'),
                     color=_col[_ic],style='italic',annotation_clip=False)

        if stacked: stack += _res[_ic].mean(axis=1)

    plt.xlim(0,xlim)
    if stacked: plt.ylim(0,100)
    else: plt.ylim(0)
    plt.xticks([n for n in range(1,5)])

    plt.xlabel('Crisis time $T_c$ [months]',labelpad=10)
    plt.ylabel('Percent population [%]',labelpad=10)

    plt.grid(False)
    #plt.grid(True,axis='y')
    sns.despine(left=True)
    plt.savefig('figs/poverty_incidence{}_{}.pdf'.format(('_stacked' if stacked else ''),shock_code),format='pdf',bbox_inches='tight') 
    plt.close('all')

