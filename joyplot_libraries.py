import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
from mc_storage_libraries import monte_carlo, load_income_impacts, load_consumption_time_series
from libraries.lib_country_dir import get_places_dict
import joypy

import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch

sns_pal = sns.color_palette('Set1', n_colors=9, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)
blues_pal = sns.color_palette('Blues_r', n_colors=4)
reds_pal = sns.color_palette('Reds_r', n_colors=4)
cool_pal = sns.color_palette('RdGy', n_colors=6)
ryg_pal = sns.color_palette('RdYlGn', n_colors=15)


######################################################
# Plotting functions (shock impacts)
def plot_sectoral_impacts(scode):
    pal = monte_carlo().ichan_cols

    ###################################################
    # wage sectors (LFS)
    mc_floss = pd.read_csv('monte_carlo/{}/frac_loss_wages.csv'.format(scode),index_col=0).T
    
    try: mc_floss = mc_floss.drop('unemployed',axis=0)
    except: pass

    try: mc_floss = mc_floss.drop('unclassified',axis=0)
    except: pass

    mc_floss['avg'] = mc_floss.mean(axis=1)
    mc_floss.sort_values(by='avg',ascending=True,inplace=True)
    mc_floss = mc_floss.drop(['avg'],axis=1).T

    #mc_loss = pd.read_csv('monte_carlo/{}/total_loss_wages.csv'.format(scode),index_col=0)
    #mc_value = pd.read_csv('monte_carlo/{}/total_value_wages.csv'.format(scode),index_col=0)

    min_floss = mc_floss.min(axis=0)
    d_floss = mc_floss.max(axis=0)-mc_floss.min(axis=0)
    mean_floss = mc_floss.mean(axis=0)

    joyplot_floss = mc_floss.T.stack().reset_index()
    joyplot_floss.columns = ['wage_sector','nsim','result']
    joyplot_grouped = joyplot_floss.groupby('wage_sector', sort=False)

    sector_labels = monte_carlo(0,'base').sector_labels
    _ylabels = [sector_labels[_] if _ in sector_labels else _.lower() for _ in mc_floss.columns]

    fig, axes = joypy.joyplot(joyplot_grouped, column='result', by='wage_sector',overlap=3,x_range=[0,100],grid='y',
                              figsize=(10,10),colormap=cm.RdYlGn_r,linewidth=0.02,alpha=0.8,ylabelsize=15,xlabelsize=15,labels=_ylabels,range_style='own')

    up_cats = ['government','information','agriculture']
    for n,y in enumerate(mc_floss.columns):
        if mean_floss[y] != 0:

            axes[n].plot([mc_floss.quantile(.25,axis=0)[y],mc_floss.quantile(.25,axis=0)[y]],[-0.01,0.01],lw=1.,color=greys_pal[7],zorder=99,clip_on=False)
            axes[n].plot([mc_floss.quantile(.75,axis=0)[y],mc_floss.quantile(.75,axis=0)[y]],[-0.01,0.01],lw=1.0,color=greys_pal[7],zorder=99,clip_on=False)
            axes[n].plot([mc_floss.quantile(.25,axis=0)[y],mc_floss.quantile(.75,axis=0)[y]],[0,0],lw=1.0,color=greys_pal[7],zorder=99,clip_on=False)
            axes[n].annotate('{}'.format(int(round(mean_floss[y]))),xy=(mean_floss[y],0.005-(-0.00 if _ylabels[n] in up_cats else 0.01)),
                             fontsize=9,ha='center',annotation_clip=False,zorder=100,
                             va=('bottom' if _ylabels[n] in up_cats else 'top'),
                             color=("white" if _ylabels[n] in up_cats else greys_pal[7]))

            # axes[n].plot([mean_floss[y],mean_floss[y]],[0,0.055],lw=2.0,color=greys_pal[7],zorder=100)
            # axes[n].annotate('{}%'.format(int(round(mean_floss[y]))),xy=(mean_floss[y]+1,0.025),fontsize=10,ha='left',va='top',weight='bold',zorder=92,color=greys_pal[8])
        else: plt.annotate('no impact',xy=(mean_floss[y]+0.95,n+0.9),fontsize=7,ha='left',va='center',weight='bold',zorder=92,style='italic')

    # plt.yticks([0.9+_ for _ in range(0,len(mc_floss.columns))],_ylabels)
    plt.xticks([0,20,40,60,80,100],['0%','20%','40%','60%','80%','100%'])

    plt.xlabel('Total income loss, by wage sector',labelpad=10,fontsize=15)
    plt.grid(False)#True,axis='x',alpha=1.0)

    sns.despine(left=True,bottom=True)
    plt.savefig('figs/sectoral_losses_wage.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')


    ###################################################
    # entrepreneurial sectors (FIES-prescribed sectors)
    ###################################################
    # wage sectors (LFS)
    mc_floss = pd.read_csv('monte_carlo/{}/frac_loss_ent.csv'.format(scode),index_col=0).T
    mc_floss['avg'] = mc_floss.mean(axis=1)
    mc_floss.sort_values(by='avg',ascending=True,inplace=True)
    mc_floss = mc_floss.drop(['avg'],axis=1).T

    #mc_loss = pd.read_csv('monte_carlo/{}/total_loss_wages.csv'.format(scode),index_col=0)
    #mc_value = pd.read_csv('monte_carlo/{}/total_value_wages.csv'.format(scode),index_col=0)

    min_floss = mc_floss.min(axis=0)
    d_floss = mc_floss.max(axis=0)-mc_floss.min(axis=0)
    mean_floss = mc_floss.mean(axis=0)

    joyplot_floss = mc_floss.T.stack().reset_index()
    joyplot_floss.columns = ['wage_sector','nsim','result']
    joyplot_grouped = joyplot_floss.groupby('wage_sector', sort=False)

    sector_labels = monte_carlo(0,'base').sector_labels
    _ylabels = [sector_labels[_] if _ in sector_labels else _.lower() for _ in mc_floss.columns]

    fig, axes = joypy.joyplot(joyplot_grouped, column='result', by='wage_sector',overlap=3,x_range=[0,100],grid='y',
                              figsize=(10,10),colormap=cm.RdYlGn_r,linewidth=0.02,alpha=0.8,ylabelsize=15,xlabelsize=15,labels=_ylabels,range_style='own')

    up_cats = ['fishing','livestock & poultry','crop farming & gardening']
    for n,y in enumerate(mc_floss.columns):

        if mean_floss[y] != 0:
            axes[n].plot([mc_floss.quantile(.25,axis=0)[y],mc_floss.quantile(.25,axis=0)[y]],[-0.01,0.01],lw=1.,color=greys_pal[7],zorder=99,clip_on=False)
            axes[n].plot([mc_floss.quantile(.75,axis=0)[y],mc_floss.quantile(.75,axis=0)[y]],[-0.01,0.01],lw=1.0,color=greys_pal[7],zorder=99,clip_on=False)
            axes[n].plot([mc_floss.quantile(.25,axis=0)[y],mc_floss.quantile(.75,axis=0)[y]],[0,0],lw=1.0,color=greys_pal[7],zorder=99,clip_on=False)
            axes[n].annotate('{}'.format(int(round(mean_floss[y]))),xy=(mean_floss[y],0.005-(-0.0 if _ylabels[n] in up_cats else 0.01)),
                             fontsize=9,ha='center',annotation_clip=False,zorder=100,
                             va=('bottom' if _ylabels[n] in up_cats else 'top'),
                             color=("white" if _ylabels[n] in up_cats else greys_pal[7]))
            # axes[n].plot([mean_floss[y],mean_floss[y]],[0,0.055],lw=2.0,color=greys_pal[7],zorder=100)
            # axes[n].annotate('{}%'.format(int(round(mean_floss[y]))),xy=(mean_floss[y]+1,0.025),fontsize=10,ha='left',va='top',weight='bold',zorder=92,color=greys_pal[8])

        else: plt.annotate('no impact',xy=(mean_floss[y]+0.95,n+0.9),fontsize=7,ha='left',va='center',weight='bold',zorder=92,style='italic')

    # plt.yticks([0.9+_ for _ in range(0,len(mc_floss.columns))],_ylabels)
    plt.xticks([0,20,40,60,80,100],['0%','20%','40%','60%','80%','100%'])

    plt.xlabel('Total income loss, by entrepreneurial sector',labelpad=10,fontsize=15)
    plt.grid(False)#True,axis='x',alpha=1.0)

    sns.despine(left=True,bottom=True)
    plt.savefig('figs/sectoral_losses_entrepreneurial.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')



def plot_losses_total_value(scode):
    plt.gcf().set_size_inches(6, 8)
    barwid = 2

    ##########################################
    # LOAD RESULTS
    # wage sector total value
    wage_val = pd.read_csv('monte_carlo/{}/total_value_wages.csv'.format(scode),index_col=0).multiply(1E-3)
    tot_wage_val = wage_val.sum(axis=1).mean()

    # wage sector losses
    wage_loss = pd.read_csv('monte_carlo/{}/total_loss_wages.csv'.format(scode),index_col=0).multiply(1E-3)
    tot_wage_loss = wage_loss.sum(axis=1).mean()
    
    #####
    # entrepreneurial total value
    ent_val = pd.read_csv('monte_carlo/{}/total_value_ent.csv'.format(scode),index_col=0).multiply(1E-3)
    tot_ent_val = ent_val.sum(axis=1).mean()

    # entrepreneurial losses
    ent_loss = pd.read_csv('monte_carlo/{}/total_loss_ent.csv'.format(scode),index_col=0).multiply(1E-3)
    tot_ent_loss = ent_loss.sum(axis=1).mean()

    #####   
    # remittances total value
    remits_value = pd.read_csv('monte_carlo/{}/total_value_remits.csv'.format(scode),index_col=0).multiply(1E-3)
    intl_remits_val = remits_value['intl'].mean()

    # remittances losses
    remits_loss = pd.read_csv('monte_carlo/{}/total_loss_remits.csv'.format(scode),index_col=0).multiply(1E-3)
    tot_remits_loss = remits_loss.sum(axis=1).mean()

    #####   
    # total economic value
    economic_value = pd.read_csv('monte_carlo/{}/total_value_economy.csv'.format(scode),index_col=0).multiply(1E-3)
    total_economic_val = economic_value['income'].mean()
    total_economic_loss = economic_value['loss'].mean()

    # concat & transpose
    #value = pd.concat([wage_val,ent_val], axis=1)
    loss = pd.concat([wage_loss,ent_loss,remits_loss], axis=1).T
    loss['avg'] = loss.mean(axis=1)
    loss = loss.loc[loss.avg>5E-2*loss['avg'].max()]
    loss = loss.sort_values(by='avg',ascending=True).drop('avg',axis=1).T


    # coloring
    # pal = monte_carlo(0,'base').ichan_cols
    # colors = [pal['nonag_wage'] if _ in wage_loss.columns else (pal['entrep'] if _ in ent_loss.columns else pal['remits']) for _ in loss.columns]

    # joyplot
    min_loss = loss.min(axis=0)
    d_loss = loss.max(axis=0)-loss.min(axis=0)
    mean_loss = loss.mean(axis=0)

    joyplot_loss = loss.T.stack().reset_index()
    joyplot_loss.columns = ['wage_sector','nsim','result']

    joyplot_grouped = joyplot_loss.groupby('wage_sector', sort=False)

    sector_labels = monte_carlo(0,'base').sector_labels

    _ylabelsA = [_+' (w)' if _ in wage_loss.columns else (_+' (e)' if _ in ent_loss.columns else _+'xxxx') for _ in loss.columns]
    _ylabels = [(sector_labels[_[:-4]]+_[-4:]).lower().replace('xxxx','') if _[:-4] in sector_labels else _.lower() for _ in _ylabelsA]

    fig, axes = joypy.joyplot(joyplot_grouped, column='result', by='wage_sector',overlap=3,x_range=[0,1.20],grid='y',
                              figsize=(10,10),linewidth=0.02,alpha=0.8,ylabelsize=15,xlabelsize=15,labels=_ylabels,range_style='own',colormap=cm.RdYlGn_r)

    for n,y in enumerate(loss.columns):
        if mean_loss[y] != 0:
            
            axes[n].plot([loss.quantile(.25,axis=0)[y],loss.quantile(.25,axis=0)[y]],[-2,2],lw=1.,color=greys_pal[7],zorder=99,clip_on=False)
            axes[n].plot([loss.quantile(.75,axis=0)[y],loss.quantile(.75,axis=0)[y]],[-2,2],lw=1.0,color=greys_pal[7],zorder=99,clip_on=False)
            axes[n].plot([loss.quantile(.25,axis=0)[y],loss.quantile(.75,axis=0)[y]],[0,0],lw=1.0,color=greys_pal[7],zorder=99,clip_on=False)
            axes[n].annotate(round(mean_loss[y],2),xy=(mean_loss[y],0.05-(-1 if _ylabels[n] == 'wholesale (w)' else 1.5)),
                             fontsize=9,ha='center',annotation_clip=False,zorder=100,
                             va=('bottom' if _ylabels[n] == 'wholesale (w)' else 'top'),
                             color=("white" if _ylabels[n] == 'wholesale (w)' else greys_pal[7]))

        else: plt.annotate('no impact',xy=(mean_loss[y]+0.95,n+0.9),fontsize=7,ha='left',va='center',weight='bold',zorder=92,style='italic')

    # loss = loss.T.stack().reset_index()
    # loss.columns = ['sector','sim','loss_value']

    # Editing /Users/brian/Software/anaconda3/lib/python3.5/site-packages/seaborn/categorical.py (draw_box_lines) to customize
    # sns.violinplot(x='loss_value', y='sector', data=loss, scale='count', inner='box',cut=0,palette=colors,saturation=0.6,linewidth=0.05)

    # sectoral labels
    sector_labels = monte_carlo(0,'base').sector_labels
    seclabels = [sector_labels[_] if _ in sector_labels else _.lower() for _ in loss.columns]
    plt.yticks([barwid/2+_*1.4 for _ in range(len(loss.columns))],seclabels,linespacing=0.80,fontsize=9)


    # totals
    plt.annotate('Monthly losses',xy=(0.655,0.868),xycoords='axes fraction',ha='left',va='top',fontsize=14,weight='bold')

    anno1 = round(1E2*economic_value['loss'].quantile(.25)/699.258,1)
    anno2 = round(1E2*economic_value['loss'].quantile(.75)/699.258,1)
    annostr = (r'PPP\${} $\endash$ {} bil.'.format(round(economic_value['loss'].quantile(.25),1),round(economic_value['loss'].quantile(.75),1))
               +'\n'+r'{}%$\endash${}% of total income'.format(round(1E2*(economic_value['loss'].quantile(.25))/total_economic_val,1),
                                                                                   round(1E2*(economic_value['loss'].quantile(.75))/total_economic_val,1),
                                                                                   round(total_economic_val,1))
               +'\n'+r'{}%$\endash${}% of GDP per month'.format(anno1,anno2))
    plt.annotate(annostr,xy=(0.985,0.838),xycoords='axes fraction',ha='right',va='top',fontsize=12,linespacing=1.25,zorder=100)
    
    # wages
    annostr = ('(w) wage sectors:\nPPP\${} $\endash$ {} bil.'.format(round(wage_loss.sum(axis=1).quantile(0.25),1),round(wage_loss.sum(axis=1).quantile(0.75),1))
               +r' ({}%$\endash${}%)'.format(int(round(1E2*wage_loss.sum(axis=1).quantile(0.25)/tot_wage_val)),
                                             int(round(1E2*wage_loss.sum(axis=1).quantile(0.75)/tot_wage_val)),
                                             round(tot_wage_val,1)))
    plt.annotate(annostr,xy=(0.655,0.758),xycoords='axes fraction',ha='left',va='top',fontsize=12,linespacing=1.25,zorder=100)

    # entrepreneurial
    annostr = ('(e) entrepreneurial sectors:\nPPP\${} $\endash$ {} bil.'.format(round(ent_loss.sum(axis=1).quantile(0.25),1),round(ent_loss.sum(axis=1).quantile(0.75),1))
               +r' ({}%$\endash${}%)'.format(int(round(1E2*ent_loss.sum(axis=1).quantile(0.25)/tot_ent_val)),
                                             int(round(1E2*ent_loss.sum(axis=1).quantile(0.75)/tot_ent_val)),
                                                 round(tot_ent_val,1)))
    plt.annotate(annostr,xy=(0.655,0.700),xycoords='axes fraction',ha='left',va='top',fontsize=12,linespacing=1.25,zorder=100)


    # remittances
    q25a = round(remits_loss['intl'].quantile(0.25),1)
    q75a = round(remits_loss['intl'].quantile(0.75),1)
    q25b = int(round(1E2*remits_loss['intl'].quantile(0.25)/intl_remits_val))
    q75b = int(round(1E2*remits_loss['intl'].quantile(0.75)/intl_remits_val))
    if q25a != q75a: annostr = ('international remittances:\nPPP\${} $\endash$ {} bil.'.format(q25a,q75a)+r' ({}%$\endash${}%)'.format(q25b,q75b))
    else: annostr = ('international remittances:\nPPP\${} bil.'.format(q25a)+r' ({}%)'.format(q25b))
    plt.annotate(annostr,xy=(0.655,0.643),xycoords='axes fraction',ha='left',va='top',fontsize=12,linespacing=1.25,zorder=100)

    # need to add bbox behind annotations

    # x-ax label
    plt.xlabel('Aggregate loss [billion PPP$/month]',labelpad=10,fontsize=15)

    plt.grid(True,axis='y',alpha=0.5,zorder=80)

    sns.despine(left=True)
    plt.savefig('figs/sectoral_losses_total_value.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')


def draw_bbox(ax, bb):
    # boxstyle=square with pad=0, i.e. bbox itself.
    p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                            abs(bb.width), abs(bb.height),
                            boxstyle="square,pad=0.",
                            ec="k", fc="none", zorder=10.,
                            )
    ax.add_patch(p_bbox)