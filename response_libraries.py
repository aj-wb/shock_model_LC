import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib.legend_handler import HandlerTuple

sns_pal = sns.color_palette('Set1', n_colors=8, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)
blues_pal = sns.color_palette('PuBu_r', n_colors=6)
reds_pal = sns.color_palette('PuRd_r', n_colors=6)

# this is the function called by main script
# only this one loaded into main
# choose from options below & include here
def get_response_sp(hh_df,mc):
    #
    #hh_df = apply_4Ps_scaleup(hh_df)
    hh_df = apply_extraordinary_social_protection(hh_df,mc)
    #
    if ('base' in mc.shock_code or mc.shock_code == 'test') and mc.nsim == 0: 
        #
        plot_nominal_receipts_frac_income(hh_df)
        #
        for affected_only in [True,False]:
                plot_savings_adequacy(hh_df,mc.shock_code,affected_only)
                plot_savings_vs_income_loss(hh_df,mc.shock_code,affected_only)
                plot_nominal_receipts_vs_income_loss(hh_df,affected_only) 
                plot_nominal_receipts_vs_income_loss_fractional(hh_df,affected_only)
                # scatter_plot_savings(hh_df,affected_only)
                # plot_pub_transfer_adequacy(hh_df,affected_only,enrollees_only=True)
                #
    return hh_df

# SP policies
def apply_4Ps_scaleup(hh_df,scaleup=1.0,duration=3):

    # income in monthly PPP/cap
    hh_df['CCT_scaleup'] = hh_df['cct4P']*scaleup
    hh_df['CCT_scaleup_duration'] = duration
    return hh_df

def load_social_amelioration_program(df,index='code'):
    # load regional payouts
    values = pd.read_csv('in/social_amelioration_program_regional_value.csv',index_col='region_{}'.format(index),encoding='utf-8-sig')[['SAP_value_php_per_hh']]
    values['SAP_value_php_per_hh'] *= 1/(1.013*1.029*1.052*1.025)
    # ^ deflate 2020 pesos to 2015
    # SOURCE: http://www.psa.gov.ph/statistics/survey/price/summary-inflation-report-consumer-price-index-2012100-january-2020

    # merge & return
    return pd.merge(df,values,left_on='region',right_index=True).rename(columns={'SAP_value_php_per_hh':'SAP_value'})

def apply_extraordinary_social_protection(hh_df,mc,verbose=False):
    if mc.nsim==0: verbose=True

    hh_df.sort_values(by='pcinc_initial',ascending=True,inplace=True)
    hh_df['cum_hhwgt'] = hh_df['hhwgt'].cumsum()
    nIneligible = hh_df['hhwgt'].sum()-mc.ESP_params['nEligible']

    for ie in mc.ESP_params['eligibility_error_array']:

        # initialize vector:
        # ESP = extraordinary social protection (any)
        # SAP = Social Amelioration Prorgam (specific)


        hh_df['ESP_payout'] = 0

        nIncluded = mc.ESP_params['nEligible']+ie*nIneligible

        is_recipient = (hh_df['cum_hhwgt']<nIncluded)&(np.random.uniform(0,1,hh_df.shape[0])<mc.ESP_params['nEligible']/nIncluded)
        hh_df.loc[is_recipient,'ESP_payout'] = hh_df.loc[is_recipient,'SAP_value']
        # SAP_value already in in PPP (from sum_to_hh())

        success = False
        while not success:
            try:
                for _cl in mc.classes:

                    my_cl = hh_df.loc[hh_df['initial_class']==_cl]
                    my_cl_pop = my_cl['popwgt'].sum()

                    my_cl_aff = hh_df.loc[(hh_df['initial_class']==_cl)&(hh_df['income_loss']>0)]
                    my_cl_aff_pop = my_cl_aff['popwgt'].sum()

                    m2d = 12/365
                    _col = '{}_ie{}'.format(_cl,int(1E2*ie))
                    mc.sp_adequacy.loc[mc.nsim,_col] = float(my_cl_aff.eval('popwgt*ESP_payout').sum()/my_cl_aff.eval('popwgt*income_loss').sum())
                    mc.sp_net_win.loc[mc.nsim,_col] = float(my_cl.loc[my_cl.eval('ESP_payout>income_loss'),'popwgt'].sum())
                    mc.sp_poverty.loc[mc.nsim,_col] = float(my_cl.loc[my_cl.eval('@m2d*(hhinc-income_loss+ESP_payout)/hhsize<3.2'),'popwgt'].sum())
                    mc.sp_subsistence.loc[mc.nsim,_col] = float(my_cl.loc[my_cl.eval('@m2d*(hhinc-income_loss+ESP_payout)/hhsize<1.9'),'popwgt'].sum())

                success = True
            except: 
                try: initialize_sp_results(mc)
                except: 
                    print('infinite loop')
                    assert(False)

        if ie == mc.ESP_params['eligerr_to_record']: mc.collect_regional_results(hh_df,'ESP_payout')

    if verbose: print('Total monthly cost of ESP: {}'.format(round(hh_df[['ESP_payout','hhwgt']].prod(axis=1).sum(),1)))
    return hh_df.drop(['cum_hhwgt','ESP_payout'],axis=1)

def initialize_sp_results(mc):
    mc.sp_adequacy = pd.DataFrame(index=mc.ix)
    mc.sp_net_win = pd.DataFrame(index=mc.ix)
    mc.sp_poverty = pd.DataFrame(index=mc.ix)
    mc.sp_subsistence = pd.DataFrame(index=mc.ix)

    for _cl in mc.classes:
        for _ie in mc.ESP_params['eligibility_error_array']:
                _col = '{}_ie{}'.format(_cl,int(1E2*_ie))
                mc.sp_adequacy[_col] = -1
                mc.sp_net_win[_col] = -1
                mc.sp_poverty[_col] = -1
                mc.sp_subsistence[_col] = -1

def plot_ESP_impact(mc):
    plt.close('all')
    init_class_size = pd.read_csv('monte_carlo/inital_pop_by_class.csv',header=None,index_col=0).squeeze()

    ###################################
    ###################################
    # ADEQUACY OF BENEFITS
    #################
    adequacy = pd.read_csv('monte_carlo/{}/ESP_adequacy.csv'.format(mc.shock_code),index_col=0)

    ie2plot = [0,25,50,75,100]
    for nie,ie in enumerate(ie2plot):

        plt.bar([_*(len(ie2plot)+1)+nie+0.5 for _ in range(len(mc.classes))],[adequacy['{}_ie{}'.format(_,ie)].mean() for _ in mc.classes],
                linewidth=0,facecolor=blues_pal[nie],alpha=0.6,zorder=90)

        if nie == 0: 
            for _n,_ie in enumerate(ie2plot): 

                _x = float(nie+_n+0.8)
                _y = float(adequacy['{}_ie{}'.format('sub',_ie)].mean())

                annostr = 'perfect targeting' if _n == 0 else ('{}% eligibility error'.format(_ie))# if _n == 1 else '{}%'.format(_ie))
                plt.annotate(annostr,xy=(_x,_y),xytext=(_x+1,_y+0.15),fontsize=8,style='italic',
                             arrowprops=dict(arrowstyle="-", color="0.5",shrinkA=5, shrinkB=5,patchA=None, patchB=None,
                             connectionstyle="angle,angleA=180,angleB=-90,rad=5",relpos=(0,0.5)))


    plt.xticks([_*(len(ie2plot)+1)+(len(ie2plot)+1)/2 for _ in range(len(mc.classes))],[_ for _ in mc.class_labels],fontsize=8,style='italic')
    plt.xlabel('Pre-COVID income class',labelpad=10,linespacing=1.5)

    plt.yticks([1,2,3],['100%\nmonthly\nlosses','200%','300%'],fontsize=8,style='italic')
    plt.ylabel('Social Amelioration Program adequacy\n(affected households only)',labelpad=8,linespacing=1.5)
    sns.despine(left=True)
    plt.grid(True,which='major',axis='y',zorder=10)
    plt.savefig('figs/sp_ESP_adequacy.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')


    ###################################
    ###################################
    # EXTREME POVERTY
    #################
    epov_base = pd.read_csv('monte_carlo/{}/totpop_sub.csv'.format(mc.shock_code),index_col=0).mean(axis=0)
    epov = pd.read_csv('monte_carlo/{}/ESP_subsistence.csv'.format(mc.shock_code),index_col=0)

    ie2plot = [0,25,50,75,100]
    pie_1 = [[] for _ in ie2plot]; 
    pie_2 = [[] for _ in ie2plot];
    ntl_1 = [None for _ in ie2plot];
    ntl_2 = [None for _ in ie2plot];
    for nie,ie in enumerate(ie2plot):

        ntl_1[nie] = round(sum([epov_base[_]-epov['{}_ie{}'.format(_,ie)].mean() for _ in mc.classes]),1)
        ntl_2[nie] = round(sum([epov['{}_ie{}'.format(_,ie)].mean() for _ in mc.classes]),1)
        for _n,_ in enumerate(mc.classes):
            _a = 0.9 if _ == 'sub' else 0.20
            
            pie_1[nie].append(plt.bar([_n*(len(ie2plot)+1)+nie+0.5],[epov_base[_]-epov['{}_ie{}'.format(_,ie)].mean()],
                              linewidth=0,facecolor=blues_pal[nie],alpha=_a,zorder=90))

            _a = 0.20 if _ == 'sub' else 0.9
            pie_2[nie].append(plt.bar([_n*(len(ie2plot)+1)+nie+0.5],[-epov['{}_ie{}'.format(_,ie)].mean()],
                              linewidth=0,facecolor=reds_pal[nie],alpha=_a,zorder=90))

        # mark initial incidence of extreme poverty
        # if nie == 0: 
            # for _n,_ in enumerate(['sub']): 
                # plt.plot([_n*(len(ie2plot)+1)+nie+0,_n*(len(ie2plot)+1)+nie+5],[-init_class_size[_],-init_class_size[_]],color=greys_pal[6],ls=':',lw=1,alpha=0.8)
                # plt.annotate('extreme poverty\npre-COVID incidence',xy=(_n*(len(ie2plot)+1)+nie+2.5,-init_class_size[_]+0.1),fontsize=7,
                             # color=greys_pal[6],ha='center',va='bottom',style='italic')


        _ytext = -1
        if nie == 0: 
            for _n,_ie in enumerate(ie2plot): 

                _x = float(nie*(len(ie2plot)+1)+_n+0.8)

                epov_subset = epov[[_c for _c in epov.columns if '_ie{}'.format(_ie) in _c]].mean(axis=0).to_frame().T
                epov_subset = epov_subset.rename(columns={_c:_c.replace('_ie{}'.format(_ie),'') for _c in epov.columns})

                _y = epov_base['sub']-epov['{}_ie{}'.format('sub',_ie)].mean()
                _ytext = max(float((epov_base.to_frame().T-epov_subset).max(axis=1))+1.4,_ytext)

                annostr = 'perfect targeting' if _n == 0 else ('{}% eligibility error'.format(_ie))# if _n == 1 else '{}%'.format(_ie))
                plt.annotate(annostr,xy=(_x,_y),xytext=(_x+1,_ytext-0.50*_n),fontsize=8,weight=500,annotation_clip=False,
                             arrowprops=dict(arrowstyle="-", color="0.5",shrinkA=5, shrinkB=5,patchA=None, patchB=None,
                             connectionstyle="angle,angleA=180,angleB=-90,rad=5",relpos=(0,0.5)))

    lgd = plt.gca().legend([(pie_1[nie][0],pie_2[nie][4]) for nie in range(len(ie2plot))],
                           ['{} m. / {} m.'.format(ntl_1[nie],ntl_2[nie]) for nie in range(len(ie2plot))],
                           loc='upper right',labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,
                           fancybox=True,frameon=True,framealpha=0.9,title='Total avoided / remain',
                           handler_map={tuple:HandlerTuple(ndivide=None,pad=0)})
    lgd.get_title().set_fontsize('8') 

    # Arrow for y-axis
    x_lo = -1
    x_lim = [x_lo,plt.gca().get_xlim()[1]]
    # plt.ylabel('Extreme poverty reduction due to\nSocial Amelioration Program [millions]',labelpad=12,linespacing=2.)
    plt.arrow(x_lo/2,0,0,6,clip_on=False,ec=blues_pal[0], fc='white', alpha=0.4, width=0.35,head_width=0.35, head_length=0.21,linewidth=2.5)
    plt.arrow(x_lo/2,0,0,-4,clip_on=False,ec=reds_pal[0], fc='white', alpha=0.4, width=0.35,head_width=0.35, head_length=0.21,linewidth=2.5)
    plt.annotate('Kept out of extreme\npoverty by SAP [mil.]',xy=(3.5*x_lo,3),rotation=90,va='center',ha='center',annotation_clip=False,fontsize=9,linespacing=1.8)
    plt.annotate('Not kept out of extreme\npoverty by SAP [mil.]',xy=(3.5*x_lo,-2),rotation=90,va='center',ha='center',annotation_clip=False,fontsize=9,linespacing=1.8)
    plt.plot([x_lo/2,x_lim[1]],[0,0],zorder=99,linewidth=0.75,color=greys_pal[6],alpha=0.9)

    plt.xlim(x_lo)
    plt.xticks([_*(len(ie2plot)+1)+(len(ie2plot)+0.5)/2 for _ in range(len(mc.classes))],[_ for _ in mc.class_labels],fontsize=8,ha='center')
    plt.xlabel('pre-COVID income class',labelpad=9,linespacing=1.5,fontsize=9)
    plt.yticks([2*_ for _ in range(-2,4)],[abs(2*_) for _ in range(-2,4)],fontsize=8)

    sns.despine(left=True,bottom=True,trim=True)
    #plt.gca().xaxis.tick_top()
    plt.grid(True,which='major',axis='y',zorder=10,alpha=0.2)
    plt.savefig('figs/sp_ESP_subsistence.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')



    ###################################
    ###################################
    # POVERTY
    #########
    # drop subsustence from mc.classes:
    # mc.classes = mc.classes[1:] 
    # mc.class_labels = mc.class_labels[1:]

    pov_base = pd.read_csv('monte_carlo/{}/totpop_pov.csv'.format(mc.shock_code),index_col=0).mean(axis=0)
    pov = pd.read_csv('monte_carlo/{}/ESP_poverty.csv'.format(mc.shock_code),index_col=0)


    ie2plot = [0,25,50,75,100]
    pie_1 = [[] for _ in ie2plot]; 
    pie_2 = [[] for _ in ie2plot];
    ntl_1 = [None for _ in ie2plot];
    ntl_2 = [None for _ in ie2plot];
    for nie,ie in enumerate(ie2plot):

        ntl_1[nie] = round(sum([pov_base[_]-pov['{}_ie{}'.format(_,ie)].mean() for _ in mc.classes]),1)
        ntl_2[nie] = round(sum([pov['{}_ie{}'.format(_,ie)].mean() for _ in mc.classes]),1)

        for _n,_ in enumerate(mc.classes):

            _a = 0.9 if _ == 'sub' or _ == 'pov' else 0.20
            pie_1[nie].append(plt.bar([_n*(len(ie2plot)+1)+nie+0.5],[pov_base[_]-pov['{}_ie{}'.format(_,ie)].mean()],
                                 linewidth=0,facecolor=blues_pal[nie],alpha=_a,zorder=90))

            _a = 0.20 if _ == 'sub' or _ == 'pov' else 0.9
            pie_2[nie].append(plt.bar([_n*(len(ie2plot)+1)+nie+0.5],[-pov['{}_ie{}'.format(_,ie)].mean()],
                                 linewidth=0,facecolor=reds_pal[nie],alpha=_a,zorder=90))

        # mark initial incidence of extreme poverty
        # if nie == 0: 
            # for _n,_ in enumerate(['sub','pov']): 
                # plt.plot([_n*(len(ie2plot)+1)+nie+0.5,_n*(len(ie2plot)+1)+nie+5.25],[-init_class_size[_],-init_class_size[_]],color=greys_pal[6],ls=':',lw=1,alpha=0.8)
                # if _ == 'pov': plt.annotate('pre-COVID\npoverty incidence',xy=(_n*(len(ie2plot)+1)+nie+3.0,-init_class_size[_]+0.3),fontsize=7,
                                            # color=greys_pal[6],ha='center',va='bottom',style='italic')

        _ytext = -1
        if nie == 1: 
            for _n,_ie in enumerate(ie2plot): 

                _x = float(nie*(len(ie2plot)+1)+_n+0.8)

                pov_subset = pov[[_c for _c in pov.columns if '_ie{}'.format(_ie) in _c]].mean(axis=0).to_frame().T
                pov_subset = pov_subset.rename(columns={_c:_c.replace('_ie{}'.format(_ie),'') for _c in pov.columns})

                _y = pov_base['pov']-pov['{}_ie{}'.format('pov',_ie)].mean()
                _ytext = max(float((pov_base.to_frame().T-pov_subset).max(axis=1))+2.75,_ytext)

                annostr = 'perfect targeting' if _n == 0 else '{}% eligibility error'.format(_ie)
                plt.annotate(annostr,xy=(_x,_y),xytext=(_x+1,_ytext-1.2*_n),fontsize=8,weight=500,annotation_clip=False,
                             arrowprops=dict(arrowstyle="-", color="0.5",shrinkA=5, shrinkB=5,patchA=None, patchB=None,
                             connectionstyle="angle,angleA=180,angleB=-90,rad=5",relpos=(0,0.5)))


    lgd = plt.gca().legend([(pie_1[nie][0],pie_2[nie][4]) for nie in range(len(ie2plot))],
                           ['{} m. / {} m.'.format(ntl_1[nie],ntl_2[nie]) for nie in range(len(ie2plot))],
                           loc='upper right',labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,
                           fancybox=True,frameon=True,framealpha=0.9,title='Total avoided / remain',
                           handler_map={tuple:HandlerTuple(ndivide=None,pad=0)})
    lgd.get_title().set_fontsize('8') 

    # Arrow for y-axis
    x_lo = -1
    x_lim = [x_lo,plt.gca().get_xlim()[1]]
    plt.arrow(x_lo/2,0,0,12,clip_on=False,ec=blues_pal[0], fc='white', alpha=0.4, width=0.35,head_width=0.35, head_length=0.67,linewidth=2.5)
    plt.arrow(x_lo/2,0,0,-12,clip_on=False,ec=reds_pal[0], fc='white', alpha=0.4, width=0.35,head_width=0.35, head_length=0.67,linewidth=2.5)
    plt.plot([x_lo/2,x_lim[1]],[0,0],zorder=99,linewidth=0.75,color=greys_pal[6],alpha=0.9)

    x_lim[0] = x_lo
    plt.xticks([_*(len(ie2plot)+1)+(len(ie2plot)+1)/2 for _ in range(len(mc.classes))],[_ for _ in mc.class_labels],fontsize=8)
    plt.xlabel('pre-COVID income class',labelpad=10,linespacing=1.5,fontsize=9)
    plt.yticks([4*_ for _ in range(-3,4)],[abs(4*_) for _ in range(-3,4)],fontsize=8)
    plt.xlim(x_lo)
    # plt.ylim(-12,12)

    # plt.ylabel('Poverty reduction due to\nSocial Amelioration Program [millions]',labelpad=12,linespacing=2.)
    plt.annotate('Kept out of poverty\nby SAP [mil.]',xy=(3*x_lim[0],6),rotation=90,va='center',ha='center',annotation_clip=False,fontsize=9,linespacing=1.75)
    plt.annotate('Not kept out of\npoverty by SAP [mil.]',xy=(3*x_lim[0],-6),rotation=90,va='center',ha='center',annotation_clip=False,fontsize=9,linespacing=1.75)

    sns.despine(left=True,bottom=True,trim=True)
    #plt.gca().xaxis.tick_top()
    plt.grid(True,which='major',axis='y',zorder=10,alpha=0.2)
    plt.savefig('figs/sp_ESP_poverty.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')

######################################
# Plot household savings vs. household income loss
######################################
def plot_savings_vs_income_loss(hh_df,shock_code,affected_only=True):

    ######################################
    nbins = int(50)
    
    # total income <-- use to structure plot
    tot_hgt, _bins = np.histogram(hh_df.eval('pcinc_initial').clip(upper=500),bins=nbins,weights=hh_df['popwgt'])
    wid = (_bins[1]-_bins[0])

    sav_hgt = []; loss_hgt = []; transfer_hgt = []
    for n,b in enumerate(_bins): 
        
        bin_slice = '(pcinc_initial>@b)&(pcinc_initial<=@_bins[@n+1]){}'.format('&(income_loss>0)' if affected_only else '')
        #
        try: sav_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*savings').sum()/hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
        except: sav_hgt.append(0)
        #
        try: loss_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*income_loss').sum()/hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
        except: loss_hgt.append(0)
        #
        try: transfer_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*total_public').sum()/hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
        except: transfer_hgt.append(0)
        
    # plot them
    ax = plt.bar(_bins,sav_hgt,width=wid,align='edge',linewidth=0,alpha=0.4,facecolor=sns_pal[2],label='Savings',zorder=91) 
    ax = plt.bar(_bins,transfer_hgt,bottom=sav_hgt,width=wid,align='edge',linewidth=0,alpha=0.5,facecolor=sns_pal[3],label='Public transfers',zorder=91) 
    #
    ax = plt.step(_bins+wid,loss_hgt,linewidth=1,alpha=0.8,color=greys_pal[7],zorder=99,label='Income loss (1 mo.)')
    ax = plt.step(_bins+wid,[2*_h for _h in loss_hgt],linewidth=1,alpha=0.8,color=greys_pal[7],zorder=99,linestyle='--',label='Income loss (2 mos.)')
    ax = plt.step(_bins+wid,[3*_h for _h in loss_hgt],linewidth=1,alpha=0.8,color=greys_pal[7],zorder=99,linestyle=':',label='Income loss (3 mos.)')

    plt.legend(loc='upper left',labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
        
    plt.xlabel('pre-COVID income [PPP$/cap/month]',labelpad=10)
    plt.ylabel('Household value [PPP$]',labelpad=10)
    plt.xlim(0,500)
    
    plt.grid(False) 
    sns.despine(left=True,bottom=True)
    plt.plot([0,500],[0,0],color=greys_pal[4],lw=1)
    plt.savefig('figs/savings_vs_income_loss{}_{}.pdf'.format('_affected_only' if affected_only else '',shock_code),format='pdf',bbox_inches='tight') 

    # plt.yscale('log')    
    # plt.legend(loc='upper left',labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
    # plt.draw()
    # plt.savefig('figs/savings_vs_income_loss{}_{}_log.pdf'.format('_affected_only' if affected_only else '',shock_code),format='pdf',bbox_inches='tight') 

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

    plt.legend(labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
        
    plt.xlabel('Fraction of household income [%]',labelpad=10)
    plt.ylabel('Beneficiaries [millions]',labelpad=10)
    plt.yticks(0.5*np.array(range(1,4)))
    plt.xlim(0)
    
    plt.grid(False) 
    sns.despine(left=True)
    plt.savefig('figs/public_transfer_hist.pdf',format='pdf',bbox_inches='tight') 
    plt.close('all')

######################################
# this block creates 'figs/income_hist_with_transfers.pdf'
# - x ax: pc income pre-shock
# - y ax: hist with hh transfers (CCT & other public) and income loss in PPP$
######################################
def plot_nominal_receipts_vs_income_loss(hh_df,affected_only=True):
    nbins = int(50)

    # total income <-- use to structure plot
    tot_hgt, _bins = np.histogram(hh_df.eval('pcinc_initial').clip(upper=500),bins=nbins,weights=hh_df['popwgt'])
    wid = (_bins[1]-_bins[0])

    other_hgt = []; cct_hgt  = []; loss_hgt = []
    for n,b in enumerate(_bins): 

        bin_slice = '(pcinc_initial>@b)&(pcinc_initial<=@_bins[@n+1]){}'.format('&(income_loss>0)' if affected_only else '')
        try: other_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*(total_public-cct4P)/hhsize').sum()
                                    /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
        except: other_hgt.append(0)

        try: cct_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*(cct4P/hhsize)').sum()
                                  /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
        except: cct_hgt.append(0)

        try: loss_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('-1*popwgt*(income_loss/hhsize)').sum()
                                   /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
        except: loss_hgt.append(0)
        
    # plot them
    ax = plt.bar(_bins,cct_hgt,width=wid,align='edge',linewidth=0,alpha=0.8,facecolor=sns_pal[2],label='CCT (incl. 4Ps)')
    ax = plt.bar(_bins,other_hgt,width=wid,align='edge',linewidth=0,alpha=0.4,bottom=cct_hgt,facecolor=sns_pal[2],label='Other public transfers')   
    ax = plt.bar(_bins,loss_hgt,width=wid,align='edge',linewidth=0,alpha=0.4,facecolor=sns_pal[0],label='COVID shock')

    plt.legend(labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
        
    plt.xlabel('pre-COVID income [PPP$/cap/month]',labelpad=10)
    plt.ylabel('Value [PPP$/cap/month]',labelpad=10)
    plt.yticks(10*np.array(range(-8,2)))
    #plt.ylim(-70,10)
    plt.xlim(0)

    plt.grid(False) 
    sns.despine(left=True,bottom=True)
    plt.plot([0,500],[0,0],color=greys_pal[4],lw=1)
    plt.savefig('figs/income_hist_with_transfers{}.pdf'.format('_affected_only' if affected_only else ''),format='pdf',bbox_inches='tight') 
    plt.close('all')

######################################
# this block creates 'figs/income_hist_with_transfers_fractional.pdf'
# - x ax: pc income pre-shock
# - y ax: hist with hh transfers (CCT & other public) and income loss as % of total income
######################################
def plot_nominal_receipts_vs_income_loss_fractional(hh_df,affected_only=True):
    nbins = int(40)
    ul = 200

    # total income <-- use to structure plot
    tot_hgt, _bins = np.histogram(hh_df.eval('pcinc_initial').clip(upper=ul),bins=nbins,weights=hh_df['popwgt'])
    wid = (_bins[1]-_bins[0])

    other_hgt = []; cct_hgt  = []; loss_hgt = []; net_hgt = []
    for n,b in enumerate(_bins): 
        try: 
            bin_slice = '(pcinc_initial>@b)&(pcinc_initial<=@_bins[@n+1]){}'.format('&(income_loss>0)' if affected_only else '')
            other_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('1E2*popwgt*(total_public-cct4P)/hhinc').sum()
                                   /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
            cct_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('1E2*popwgt*(cct4P)/hhinc').sum()
                                 /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
            loss_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('-1*1E2*popwgt*(income_loss)/hhinc').sum()
                                  /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
            net_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('1E2*popwgt*(total_public-income_loss)/hhinc').sum()
                                 /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
        except: pass

        
    # plot them

    #ax = plt.bar(_bins[:-1],tot_hgt,width=wid,align='edge',linewidth=0,alpha=0.5,facecolor=sns_pal[2],label='Total income')
    # ^ income hist

    ax = plt.bar(_bins[:-1],cct_hgt,width=wid,align='edge',linewidth=0,alpha=0.8,facecolor=sns_pal[2],label='CCT (incl. 4Ps)')
    ax = plt.bar(_bins[:-1],other_hgt,width=wid,align='edge',linewidth=0,alpha=0.4,bottom=cct_hgt,facecolor=sns_pal[2],label='Other public transfers')   
    ax = plt.bar(_bins[:-1],loss_hgt,width=wid,align='edge',linewidth=0,alpha=0.4,facecolor=sns_pal[0],label='COVID shock')
    ax = plt.step(_bins[:-1]+wid,net_hgt,linewidth=1,alpha=0.8,color=greys_pal[7],zorder=99,label='100% scaleup, net effect')
    
    plt.legend(labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)#loc='upper right')
        
    plt.xlabel('pre-COVID income [PPP$/cap/month]',labelpad=10)
    plt.ylabel('Value [% of income]',labelpad=10)
    #plt.yticks(2*np.array(range(-3,4)))
    #plt.ylim(-6,6)
    #plt.gca().xaxis.set_ticks_position('top')
    #plt.gca().xaxis.tick_top()
    plt.xlim(0)

    plt.grid(True,axis='x',alpha=0.3) 
    sns.despine(left=True,bottom=True)
    plt.plot([0,ul],[0,0],color=greys_pal[4],lw=1)
    plt.savefig('figs/income_hist_with_transfers_fractional{}.pdf'.format('_affected_only' if affected_only else ''),format='pdf',bbox_inches='tight') 
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
            try:
                _avg.append(num/dnm)
                _avgbin.append(_bins[n])
            except: pass
    return _avg,_avgbin

def plot_pub_transfer_adequacy(hh_df,affected_only=True,enrollees_only=True,):
    nbins = int(50)
    _ul = 500

    # total income <-- use to structure plot
    _ = '(total_public!=0)'
    tot_hgt, _bins = np.histogram(hh_df.loc[hh_df.eval(_),'pcinc_initial'].clip(upper=_ul),bins=nbins,weights=hh_df.loc[hh_df.eval(_),'popwgt'])
    wid = (_bins[1]-_bins[0])
    
    for enrollees_only in [True]:#False]:
        cct_frac_hgt = []; oth_frac_hgt = []; tot_frac_hgt = [];
        for n,b in enumerate(_bins): 
            try:
                bin_slice = '(pcinc_initial>@b)&(pcinc_initial<=@_bins[@n+1]){}{}'.format('&(total_public!=0)' if enrollees_only else '',
                                                                                          '&(income_loss>0)' if affected_only else '')
                cct_frac_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*income_loss').sum()
                                          /max(1E-9,hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*cct4P').sum())))
                #
                oth_frac_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*income_loss').sum()
                                          /max(1E-9,hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*(total_public-cct4P)').sum())))
                #
                tot_frac_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*income_loss').sum()
                                          /max(1E-9,hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*total_public').sum())))
            except: pass
            
        _ls = '-' if enrollees_only else ':'
        _ra, _rabins = get_rolling_average(cct_frac_hgt,_bins,window=4)
        plt.plot(_rabins[2:],_ra[2:],ls=_ls,color=sns_pal[2],
                 label='CCT{}'.format(' (enrollees only)' if enrollees_only else ''))
        try: plt.scatter(_bins[:-1],cct_frac_hgt,s=6,alpha=0.25,color=sns_pal[2])
        except: 
            print('error in public_transfer_adequacy')
            pass

        _ra, _rabins = get_rolling_average(oth_frac_hgt,_bins,window=4)
        plt.plot(_rabins[2:],_ra[2:],ls=_ls,color=sns_pal[3],
                 label='other transfers{}'.format(' (enrollees only)' if enrollees_only else ''))
        try: plt.scatter(_bins[:-1],oth_frac_hgt,s=6,alpha=0.25,color=sns_pal[3])
        except: 
            print('error in public_transfer_adequacy')
            pass

        _ra, _rabins = get_rolling_average(tot_frac_hgt,_bins,window=4)
        plt.plot(_rabins[2:],_ra[2:],ls=_ls,color=sns_pal[4],
                 label='total transfers{}'.format(' (enrollees only)' if enrollees_only else ''))
        try: plt.scatter(_bins[:-1],tot_frac_hgt,s=30,marker="1",alpha=0.25,color=sns_pal[4])
        except: 
            print('error in public_transfer_adequacy')
            pass

        
    plt.legend(labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
        
    plt.xlabel('pre-COVID income [PPP$/cap/month]',labelpad=10)
    plt.ylabel('Ratio (monthly income loss/transfer)',labelpad=10)
    plt.xlim(0)

    plt.yscale('log')  
    plt.grid(False)#True,axis='y')
    sns.despine(left=True)
    plt.plot([0,_ul],[0,0],color=greys_pal[4],lw=1)
    plt.savefig('figs/public_transfer_adequacy{}{}.pdf'.format('_affected_only' if affected_only else '',
                                                               '_enrollees_only' if enrollees_only else ''),format='pdf',bbox_inches='tight') 
    plt.close('all')

######################################
# this block creates 'figs/savings_frac_income_loss.pdf'
# plot savings & wages lost (1-3 mos) as fraction of hh income 
######################################
def scatter_plot_savings(hh_df,affected_only=True):
    nbins = int(50)

    # total income <-- use to structure plot
    tot_hgt, _bins = np.histogram(hh_df.eval('pcinc_initial').clip(upper=500),bins=nbins,weights=hh_df['popwgt'])
    wid = (_bins[1]-_bins[0])

    sav_frac_hgt = []; loss_frac_hgt = [];
    for n,b in enumerate(_bins): 
        try: 
            bin_slice = '(pcinc_initial>@b)&(pcinc_initial<=@_bins[@n+1]){}'.format('&(income_loss>0)' if affected_only else '')
            sav_frac_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*1E2*savings/(hhinc*12)').sum()
                                      /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))
            loss_frac_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*1E2*income_loss/(hhinc*12)').sum()
                                       /hh_df.loc[hh_df.eval(bin_slice),'popwgt'].sum()))

        except: pass

    plt.scatter(_bins[:-1],sav_frac_hgt,label='Savings',marker="1")
    plt.plot(_bins[:-1],loss_frac_hgt,label='Income loss (1 mo.)')#,s=9)
    plt.plot(_bins[:-1],[2*b for b in loss_frac_hgt],label='Income loss (2 mos.)')#,s=9)
    plt.plot(_bins[:-1],[3*b for b in loss_frac_hgt],label='Income loss (3 mos.)')#,s=9)
    
    plt.legend(labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
        
    plt.xlabel('pre-COVID income [PPP$/cap/month]',labelpad=10)
    plt.ylabel('Value as % of annual income',labelpad=10)
    plt.xlim(0)

    plt.grid(False)    
    sns.despine(left=True,bottom=True)
    plt.plot([0,500],[0,0],color=greys_pal[4],lw=1)
    plt.savefig('figs/savings_frac_income_loss{}.pdf'.format('_affected_only' if affected_only else ''),format='pdf',bbox_inches='tight') 
    plt.close('all')

######################################
# this block creates 'figs/savings_adequacy.pdf'
######################################
def plot_savings_adequacy(hh_df,shock_code,affected_only=True):
    _ul = 500
    _uly = 3 if affected_only else 7
    nbins = int(50)

    # total income <-- use to structure plot
    tot_hgt, _bins = np.histogram(hh_df.eval('pcinc_initial').clip(upper=_ul),bins=nbins,weights=hh_df['popwgt'])
    wid = (_bins[1]-_bins[0])

    sav_over_loss_hgt = [];
    for n,b in enumerate(_bins): 
        try: 
            bin_slice = '(pcinc_initial>@b)&(pcinc_initial<=@_bins[@n+1]){}'.format('&(income_loss>0)' if affected_only else '')
            sav_over_loss_hgt.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*savings').sum()
                                           /hh_df.loc[hh_df.eval(bin_slice)].eval('popwgt*income_loss').sum()))
        except: pass
    ax = plt.bar(_bins[1:-1],sav_over_loss_hgt[1:],width=wid,align='edge',linewidth=0,alpha=0.8,facecolor=sns_pal[2])

    # annotate pop sizes
    for nmo in range(1,_uly+1):
        nhh = round(hh_df.loc[hh_df.eval('@nmo*income_loss>savings'),'hhwgt'].sum(),1)
        pop = round(hh_df.loc[hh_df.eval('@nmo*income_loss>savings'),'popwgt'].sum(),1)
        anno_str = '{} mil. households ({} mil. individuals)\n{}within {} month{}'.format(nhh,pop,('exhaust savings ' if nmo == 1 else ''),nmo,('' if nmo == 1 else 's'))
        plt.annotate(anno_str,xy=(0.05,nmo/_uly-0.025),xycoords='axes fraction',fontsize=9,ha='left',va='top',annotation_clip=False)

        #print(anno_str,round(hh_df.loc[hh_df.eval('@nmo*income_loss>savings')].eval('popwgt/hhsize').sum(),1))

    if affected_only: plt.yticks(np.array(range(1,4)))
    plt.xlabel('pre-COVID income [PPP$/cap/month]',labelpad=10)
    plt.ylabel('Savings adequacy [months]',labelpad=10)
    plt.xlim(0)
    plt.ylim(0,_uly)

    plt.grid(True,axis='y',alpha=1.0,lw=1)    
    sns.despine(left=True,bottom=True)
    plt.plot([0,_ul],[0,0],color=greys_pal[4],lw=1)
    plt.savefig('figs/savings_adequacy{}_{}.pdf'.format('_affected_only' if affected_only else '',shock_code),format='pdf',bbox_inches='tight') 
    plt.close('all')
