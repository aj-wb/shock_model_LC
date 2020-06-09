import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns_pal = sns.color_palette('Set1', n_colors=8)
greys_pal = sns.color_palette('Greys', n_colors=8)

nbins=25
_fs = 9

def library_open():

	look_into_it('total_loss_wages',classwise_format=False,usecols=['construction'],altscode='relax_construction50')
	look_into_it('frac_loss_wages',classwise_format=False,usecols=['construction'],altscode='relax_construction50')
	look_into_it('pop_aff')
	# look_into_it('frac_loss_remits')
	# look_into_it('ag_wage_lossfrac')
	# look_into_it('ESP_poverty')

def look_into_it(fom,scode='base',classwise_format=True,usecols=None,altscode=None):
	
	if altscode is not None: shock_array = [scode,altscode]
	else: shock_array = [scode]

	_bins = None
	for _n,_s in enumerate(shock_array):
		df = pd.read_csv('monte_carlo/{}/{}.csv'.format(_s,fom),index_col=0)
		print(fom)

		if classwise_format: df = df.sum(axis=1)
		elif usecols is not None: df = df[usecols].sum(axis=1)

		# mean value (across simuations//no weighting)
		avg = df.mean()

		tot_hgt, _bins = np.histogram(df.squeeze(),bins=nbins)
		n = sum(np.array(tot_hgt))*1E-2
		norm_hgt = [_/n for _ in tot_hgt]
		ax = plt.bar(_bins[:-1],norm_hgt,bottom=0,width=(_bins[1]-_bins[0]),align='edge',linewidth=0,alpha=0.6,facecolor=sns_pal[_n+1],label=fom)

		plt.plot([avg,avg],[0,1.1*np.max(norm_hgt)],color=greys_pal[5],alpha=0.5)
		plt.annotate('{}\nmean = {}'.format(_s,round(avg,1)),xy=(avg*1.01,1.1*np.max(norm_hgt)),ha='left',va='top')

	plt.xlabel(fom,labelpad=10,fontsize=_fs)
	plt.ylabel('Frequency [%]',labelpad=10,fontsize=_fs)

	plt.grid(True,axis='y',alpha=0.3)
	sns.despine(left=True)
	#plt.plot([0,_ul],[0,0],color=greys_pal[4],lw=1)
	plt.savefig('error_analysis/{}{}.pdf'.format(fom,'_'+usecols[0] if usecols is not None else ''),format='pdf',bbox_inches='tight')
	plt.close('all')
