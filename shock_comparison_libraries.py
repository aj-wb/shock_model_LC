import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os,glob
import numpy as np
from mc_storage_libraries import monte_carlo

sns_pal = sns.color_palette('Set1', n_colors=9, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)

classes = ['sub','pov','vul','sec','mc']

ax_label_dict = {'tot_loss':'Avoided wage loss [mil. PPP$/month]',
                 'pop_aff':r'$\Delta$ affected population [mil.]',
                 'di_aff':r'$\Delta$ income loss among'+'\naffected households [PPP\$/cap/day]',
                 'affpop_sub':'Extreme poverty decrease [mil.]'+'\n'+r'(i $\leq$ PPP\$1.90/day)',
                 'affpop_pov':'Poverty decrease [mil.]'+'\n'+r'(i $\leq$ PPP\$3.20\)',
                 'affpop_vul':'Economic vulnerability decrease [mil.]'+'\n'+r'(i $\leq$ PPP\$5.50)'}


def relaxation_plots(base_code,rfrac):

    plot_status_vs_valueadd(base_code,rfrac,'di_aff','affpop_sub')
    plot_status_vs_valueadd(base_code,rfrac,'di_aff','affpop_pov')
    plot_status_vs_valueadd(base_code,rfrac,'di_aff','affpop_vul')

    plot_status_vs_valueadd(base_code,rfrac,'tot_loss','affpop_sub')
    plot_status_vs_valueadd(base_code,rfrac,'tot_loss','affpop_pov')
    plot_status_vs_valueadd(base_code,rfrac,'tot_loss','affpop_vul')
    plot_status_vs_valueadd(base_code,rfrac,'tot_loss','pop_aff')
    #
    plot_sectoral_relaxation(base_code,rfrac,'tot_loss')
    plot_sectoral_relaxation(base_code,rfrac,'pop_aff')
    plot_sectoral_relaxation(base_code,rfrac,'affpop_sub')
    plot_sectoral_relaxation(base_code,rfrac,'affpop_pov')
    plot_sectoral_relaxation(base_code,rfrac,'affpop_vul')
    #plot_poverty()


def plot_status_vs_valueadd(base_code,rfrac,fom_x,fom_y):
    mc = monte_carlo(0,'base')

    dirs,labels = get_mc_directories(rfrac)
    base_x = pd.read_csv('monte_carlo/{}/{}.csv'.format(base_code,fom_x),index_col=0).sum(axis=1)
    base_y = pd.read_csv('monte_carlo/{}/{}.csv'.format(base_code,fom_y),index_col=0).sum(axis=1)

    sf_x = 1
    sf_y = 1

    for _nd,_d in enumerate(dirs):
        _fx = pd.read_csv(_d+'/{}.csv'.format(fom_x),index_col=0)
        _fy = pd.read_csv(_d+'/{}.csv'.format(fom_y),index_col=0) 

        x_val = sf_x*(base_x.mean()-_fx.sum(axis=1).mean())
        y_val = sf_y*(base_y.mean()-_fy.sum(axis=1).mean())

        x_err = 2*(base_x-_fx.sum(axis=1)).std()/np.sqrt(float(base_x.shape[0])) 
        y_err = 2*(base_y-_fy.sum(axis=1)).std()/np.sqrt(float(base_y.shape[0]))

        if  (x_val-x_err > 0 or y_val-y_err > 0):
            plt.errorbar(x_val,y_val,xerr=x_err,yerr=y_err,fmt='o',zorder=90,color=sns_pal[1],alpha=(0.7 if (x_val-x_err>0 and y_val-y_err>0) else 0.3))
            plt.annotate(labels[_nd],xy=(x_val+x_err/10,y_val+y_err/10),fontsize=8,color=greys_pal[5],rotation=0,ha='left',va='bottom',weight=500)
        
    plt.xlim(0)
    plt.ylim(0)

    plt.annotate('{}% recovery of\nsectoral activity'.format(rfrac),xy=(0.05,0.95),
                 xycoords='axes fraction',ha='left',va='top',fontsize=8,color=greys_pal[6],annotation_clip=False)

    plt.grid(False)
    sns.despine(bottom=True,left=True)
    plt.plot([0,0],plt.gca().get_ylim(),color=greys_pal[6],zorder=10)
    plt.plot(plt.gca().get_xlim(),[0,0],color=greys_pal[6],zorder=10)

    plt.xlabel(ax_label_dict[fom_x],labelpad=10,linespacing=1.75)
    plt.ylabel(ax_label_dict[fom_y],labelpad=10,linespacing=1.75)

    # save & close
    plt.savefig('figs/scatter_{}_{}.pdf'.format(fom_x,fom_y),format='pdf',bbox_inches='tight') 
    plt.close('all')




def plot_sectoral_relaxation(base_code,rfrac,fom):
    
    sf_y = 1
    
    dirs,labels = get_mc_directories(rfrac)
    base_loss = pd.read_csv('monte_carlo/{}/{}.csv'.format(base_code,fom),index_col=0).sum(axis=1)
    
    _xticks=[]
    for _nd,_d in enumerate(dirs):
        _f = pd.read_csv(_d+'/{}.csv'.format(fom),index_col=0)
        
        y_err = 2*(base_loss-_f.sum(axis=1)).std()/np.sqrt(float(base_loss.shape[0]))

        plt.bar(_nd,sf_y*(base_loss.mean()-_f.sum(axis=1).mean()),yerr=y_err,
                color=sns_pal[1],edgecolor=None,ecolor=greys_pal[5],width=0.8,alpha=0.85,zorder=99)
        
        _xticks.append(labels[_nd])
        
    plt.plot([-1,len(dirs)],[0,0],lw=0.8,color=greys_pal[7])

    # annotate
    plt.annotate('95% CI\n'+r'N$_{sims}$ = '+'{}'.format(base_loss.shape[0]),xy=(0.95,0.95),xycoords='axes fraction',
                 ha='right',va='top',fontsize=8,color=greys_pal[6])

    # axes
    plt.xticks([n for n in range(0,len(dirs))],_xticks,rotation=90)
    plt.ylabel(ax_label_dict[fom],labelpad=10,linespacing=1.75)
    sns.despine(left=True,bottom=True)

    # grid
    plt.grid(False)
    plt.grid(True,axis='y',zorder=10)

    # save & close
    plt.savefig('figs/relaxation_{}.pdf'.format(fom),format='pdf',bbox_inches='tight') 
    plt.close('all')



def get_mc_directories(rfrac):
    rootdir = os.getcwd()+'/monte_carlo/'    
    _dirs = []; _lbls = []
    #
    for subdir, dirs, files in os.walk(rootdir):
        if subdir != rootdir and rfrac in subdir: 
            _dirs.append(subdir)
            __sector = subdir.replace(rootdir+'relax_','').replace(rfrac,'')
            try: _lbls.append(monte_carlo(0,'base').sector_labels[__sector])
            except: _lbls.append(__sector)
    return _dirs,_lbls
    

