import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os,glob
import numpy as np

sns_pal = sns.color_palette('Set1', n_colors=9, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)

classes = ['sub','pov','vul','sec','mc']

def relaxation_plots(rfrac='20'):
 
    plot_sectoral_relaxation('tot_loss',rfrac)
    plot_sectoral_relaxation('pop_aff',rfrac)
    plot_sectoral_relaxation('income_sub',rfrac)
    plot_sectoral_relaxation('income_pov',rfrac)
    plot_sectoral_relaxation('income_vul',rfrac)
    #plot_poverty()



def plot_sectoral_relaxation(fom,rfrac):

    ylabel_dict = {'tot_loss':'Avoided wage loss [mil. PPP$ per month]',
                   'pop_aff':r'$\Delta$ affected population [mil.]',
                   'income_sub':r'$\Delta$ affected pop in extreme poverty'+'\n'+r'(i $\leq$ PPP\$1.90) during shock [mil.]',
                   'income_pov':r'$\Delta$ affected pop in poverty'+'\n'+r'(i $\leq$ PPP\$3.20) during shock [mil.]',
                   'income_vul':r'$\Delta$ affected pop in vulnerability'+'\n'+r'(i $\leq$ PPP\$5.50) during shock [mil.]'}
    
    sf_y = 1
    
    dirs,labels = get_mc_directories(rfrac)
    nominal_loss = pd.read_csv('monte_carlo/nominal/{}.csv'.format(fom),index_col=0).sum(axis=1)
    
    _xticks=[]
    for _nd,_d in enumerate(dirs):
        _f = pd.read_csv(_d+'/{}.csv'.format(fom),index_col=0)
        
        y_err = 2*(nominal_loss-_f.sum(axis=1)).std()/np.sqrt(float(nominal_loss.shape[0]))

        plt.bar(_nd,sf_y*(nominal_loss.mean()-_f.sum(axis=1).mean()),yerr=y_err,
                color=sns_pal[1],edgecolor=None,width=0.8,alpha=0.85,zorder=99)
        
        _xticks.append(labels[_nd])
        
    plt.plot([-1,len(dirs)],[0,0],lw=0.8,color=greys_pal[7])

    # annotate
    plt.annotate('95% CI\n'+r'N$_{sims}$ = '+'{}'.format(nominal_loss.shape[0]),xy=(0.95,0.95),xycoords='axes fraction',
                 ha='right',va='top',fontsize=8,color=greys_pal[6])

    # axes
    plt.xticks([n for n in range(0,len(dirs))],_xticks,rotation=90)
    plt.ylabel(ylabel_dict[fom],labelpad=8,linespacing=1.5)
    sns.despine(left=True,bottom=True)

    # grid
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
            _lbls.append(subdir.replace(rootdir+'relax_','').replace(rfrac,''))
    return _dirs,_lbls
    
relaxation_plots()
    
