import matplotlib.pyplot as plt
from matplotlib import ticker
#import unidecode
import pandas as pd
import numpy as np
import glob

from lib_get_places_dict import get_places_dict

############################################################################# 
#######################         FROM SVG              ####################### 
#############################################################################     

from bs4 import BeautifulSoup    
from IPython.display import Image, display, HTML, SVG
#img_width = 400

import os, shutil
import math
from subprocess import Popen, PIPE, call 

def make_choropleths(hh_df,scode='base'):
    m2d = 12/365

    prov_code,reg_code = get_places_dict('PH')
    hh_df['region'].replace(reg_code,inplace=True)
    # hh_df['province'] = hh_df['prov_code'].replace(prov_code,inplace=False)

    # Make maps from single simulation
    hh_df = hh_df.set_index('region')
    reg_df = hh_df['popwgt'].sum(level='region').to_frame(name='popwgt')
    reg_df['affected_pop'] = hh_df.loc[hh_df.income_loss>0,'popwgt'].sum(level='region')
    reg_df['affected_frac'] = reg_df.eval('1E2*affected_pop/popwgt')
    reg_df['base_poverty'] = hh_df.loc[(m2d*hh_df['pcinc_initial']<=3.2),'popwgt'].sum(level='region')
    reg_df['impoverished'] = hh_df.loc[(m2d*hh_df['pcinc_initial']>3.2)&(m2d*hh_df['pcinc_final']<=3.2),'popwgt'].sum(level='region')
    reg_df['impoverished_frac_population'] = 1E2*reg_df['impoverished']/reg_df['popwgt']
    reg_df['impoverished_frac_poverty'] = 1E2*reg_df['impoverished']/reg_df['base_poverty']

    # average values from MC ensembles
    total_affected = pd.read_csv('monte_carlo/{}/pop_aff.csv'.format(scode),index_col=0).sum(axis=1)
    tot_aff_lo = round(total_affected.quantile(0.25),1)
    tot_aff_hi = round(total_affected.quantile(0.75),1)

    # total affected
    make_map_from_svg(reg_df['affected_pop'],get_svg_file('PH'),'affected_pop',
                      label='Affected population [mil.]\n'+r'(total = {}$\endash${} mil.)'.format(tot_aff_lo,tot_aff_hi),
                      do_qualitative=True,map_color='Blues',nbins=6,ticklbl_int=True)

    # fraction affected
    tot_aff_lo = int(round(1E2*total_affected.quantile(0.25)/hh_df['popwgt'].sum(),0))
    tot_aff_hi = int(round(1E2*total_affected.quantile(0.75)/hh_df['popwgt'].sum(),0))
    make_map_from_svg(reg_df['affected_frac'],get_svg_file('PH'),'affected_frac',
                      label='Fraction of population affected [%]\n'+r'overall {}$\endash${}%'.format(tot_aff_lo,tot_aff_hi),
                      do_qualitative=True,map_color='Blues',nbins=5,ticklbl_int=True,force_min=45,force_max=95)

    # individuals moved into poverty
    total_pop_pov = pd.read_csv('monte_carlo/{}/totpop_pov.csv'.format(scode),index_col=0)[['vul','sec','mc']].sum(axis=1)
    total_pop_pov_lo = round(total_pop_pov.quantile(0.25),1)
    total_pop_pov_hi = round(total_pop_pov.quantile(0.75),1)
    make_map_from_svg(reg_df['impoverished'],get_svg_file('PH'),'impoverished',
                      label='Newly-impoverished population [mil.]\n'+r'i$\leq$PPP\$3.20/day (total = {}$\endash${} mil)'.format(total_pop_pov_lo,total_pop_pov_hi),
                      do_qualitative=True,map_color='Blues',nbins=5,ticklbl_int=True)

    # individuals moved into poverty relative to total pop
    frac_increase_lo = int(round(1E2*total_pop_pov.quantile(0.25)/reg_df['popwgt'].sum()))
    frac_increase_hi = int(round(1E2*total_pop_pov.quantile(0.75)/reg_df['popwgt'].sum()))
    make_map_from_svg(reg_df['impoverished_frac_population'],get_svg_file('PH'),'impoverished_frac_population',
                      label='Fraction of regional population impoverished\n'+r'{}$\endash${}% pushed into poverty nationally'.format(frac_increase_lo,frac_increase_hi),
                      do_qualitative=True,map_color='Blues',nbins=5,ticklbl_int=True,force_min=0,force_max=50)

    # poverty increase %
    frac_increase_lo = int(round(1E2*total_pop_pov.quantile(0.25)/reg_df['base_poverty'].sum()))
    frac_increase_hi = int(round(1E2*total_pop_pov.quantile(0.75)/reg_df['base_poverty'].sum()))
    make_map_from_svg(reg_df['impoverished_frac_poverty'],get_svg_file('PH'),'impoverished_frac_poverty',
                      label='Poverty increase as % of pre-COVID incidence\n'+r'{}$\endash${}% national increase'.format(frac_increase_lo,frac_increase_hi),
                      do_qualitative=True,map_color='Blues',nbins=8,ticklbl_int=True,force_min=0,force_max=800)
    #
    cleanup()


############################################################
# functions that create choropleths

def purge(dir, pattern):
    for f in glob.glob(dir+pattern):
        os.remove(f)

def get_svg_file(myC):
    if myC == 'PH': svg_file = 'maps/BlankSimpleMapRegional.svg'
    else: assert(False)
    return svg_file

def make_map_from_svg(series_in, svg_file_path, outname, 
                      map_color='Blues',nbins=5,
                      label = "", outfolder ="maps/" ,
                      svg_handle='class',new_title=None, do_qualitative=False, res=1000, verbose=True,
                      drop_spots=None,force_min=None,force_max=None,ticklbl_int=False):
    """Makes a cloropleth map and a legend from a panda series and a blank svg map. 
    Assumes the index of the series matches the SVG classes
    Saves the map in SVG, and in PNG if Inkscape is installed.
    if provided, new_title sets the title for the new SVG map
    """

    color_maper=plt.cm.get_cmap(map_color,nbins)

    print('\nGenerating map of ',label)

    if force_min is not None: series_in.loc['xx_forcedmin'] = force_min
    if force_max is not None: series_in.loc['xx_forcedmax'] = force_max
    
    #simplifies the index to lower case without space
    series_in.index = series_in.index.str.lower().str.replace(" ","_").str.replace("-","_").str.replace(".","_").str.replace("(","_").str.replace(")","_")
    if drop_spots is not None: 
        __ds = []
        for _ds in drop_spots:
            __ds.append(_ds.lower().replace(" ","_").replace("-","_").replace(".","_").replace("(","_").replace(")","_"))
        series_in = series_in.drop([_ for _ in __ds if _ in series_in])

    #compute the colors 
    color = data_to_rgb(series_in,nbins=nbins,color_maper=color_maper,do_qual=do_qualitative)

    #Builds the CSS style for the new map  (todo: this step could get its own function)
    style_base =\
    """.{depname}
    {{  
       fill: {color};
       stroke:#000000;
       stroke-width:2;
    }}"""

    #Default style (for regions which are not in series_in)
    style =\
    """.default
    {
    fill: #bdbdbd;
    stroke:#ffffff;
    stroke-width:2;
    }
    """
    
    #builds the style line by line (using lower case identifiers)
    for c in series_in.index:
        style= style + style_base.format(depname=c,color=color[c])+ "\n"

    #output file name
    target_name = outfolder+"map_of_"+outname

    #read input 
    with open(svg_file_path, 'r',encoding='utf8') as svgfile: #MIND UTF8
        soup=BeautifulSoup(svgfile.read(),"xml")
        #print(type(soup))

    #names of regions to lower case without space   
    for p in soup.findAll("path"):

        try:
            p[svg_handle]=p[svg_handle].lower().replace(" ","_").replace("-","_").replace(".","_").replace("(","_").replace(")","_")
        except:
            pass
        #Update the title (tooltip) of each region with the numerical value (ignores missing values)
        try:
            p.title.string += "{val:.3%}".format(val=series_in.ix[p[svg_handle]])
        except:
            pass
   
    #remove the existing style attribute (unimportant)
    del soup.svg["style"]
    
    #append style
    soup.style.string = style

    #Maybe update the title
    if new_title is not None:
        soup.title.string = new_title
    else:
        new_title = ""
        
    #write output
    with open(target_name+".svg", 'w', encoding="utf-8") as svgfile:
        svgfile.write(soup.prettify())
        
    #Link to SVG
    display(HTML("<a target='_blank' href='"+target_name+".svg"+"'>SVG "+new_title+"</a>"))  #Linking to SVG instead of showing SVG directly works around a bug in the notebook where style-based colouring colors all the maps in the NB with a single color scale (due to CSS)
    
    #reports missing data
    if verbose:
        try:
            places_in_soup = [p[svg_handle] for p in soup.findAll("path")]        
            data_missing_in_svg = series_in[~series_in.index.isin(places_in_soup)].index.tolist()
            data_missing_in_series = [p for p in places_in_soup if (p not in series_in.index.tolist())]

            back_to_title = lambda x: x.replace("_"," ").title()
    
            if data_missing_in_svg:
                print("Missing in SVG: "+"; ".join(map(back_to_title,data_missing_in_svg)))
                #series_in = series_in.drop(data_missing_in_svg,axis=0)

            if data_missing_in_series:
                print("Missing in series: "+"; ".join(map(back_to_title,data_missing_in_series)))

        except:
            pass

    if shutil.which("inkscape") is None:
        print("cannot convert SVG to PNG. Install Inkscape to do so.")
        could_do_png_map = False
    else:
        #Attempts to inkscape SVG to PNG    
        process=Popen("inkscape -f {map}.svg -e {map}.png -d 150".format(map=target_name, outfolder = outfolder) , shell=True, stdout=PIPE,   stderr=PIPE)
        out, err = process.communicate()
        errcode = process.returncode
        if errcode:
            could_do_png_map = False
            print("Could not transform SVG to PNG. Error message was:\n"+err.decode())
        else:
            could_do_png_map = True

    #makes the legend with matplotlib
    l = make_legend(series_in,color_maper,nbins,label,outfolder+"legend_of_"+outname,do_qualitative,res,force_min,force_max,ticklbl_int)
    
    if shutil.which("convert") is None:
        print("Cannot merge map and legend. Install ImageMagick® to do so.")
    elif could_do_png_map:
        #Attempts to downsize to a single width and concatenate using imagemagick
        call("convert "+outfolder+"legend_of_{outname}.png -resize {w} small_legend.png".format(outname=outname,w=res), shell=True )
        call("convert "+outfolder+"map_of_{outname}.png -resize {w} small_map.png".format(outname=outname,w=res) , shell=True)
        
        merged_path = outfolder+"map_and_legend_of_{outname}.png".format(outname=outname)
        
        call("convert -append small_map.png small_legend.png "+merged_path, shell=True)
        
        #removes temp files
        if os.path.isfile("small_map.png"):
            os.remove("small_map.png")
        if os.path.isfile("small_legend.png"):
            os.remove("small_legend.png")
            
        if os.path.isfile("legend_of_{outname}.png".format(outname=outname)):
            os.remove("legend_of_{outname}.png".format(outname=outname))

        if os.path.isfile("map_of_{outname}.png".format(outname=outname)):
            os.remove("map_of_{outname}.png".format(outname=outname))

        if os.path.isfile("legend_of_{outname}.svg".format(outname=outname)):
            os.remove("legend_of_{outname}.svg".format(outname=outname))

        if os.path.isfile("map_of_{outname}.svg".format(outname=outname)):
            os.remove("map_of_{outname}.svg".format(outname=outname))
    
        if os.path.isfile(merged_path):
            return Image(merged_path)

    
import matplotlib as mpl

def make_legend(serie,cmap,nbins,label="",path=None,do_qualitative=False,res=1000,force_min=None,force_max=None,ticklbl_int=False):
    #todo: log flag

    fig = plt.figure(figsize=(8,3))
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])

    vmin = math.floor(force_min if force_min is not None else serie.min())
    vmax = math.ceil(force_max if force_max is not None else serie.max())
    spread = vmax-vmin

    # define discrete bins and normalize
    bounds =np.linspace(vmin,vmax,nbins)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    if not do_qualitative:
        #continuous legend
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='horizontal')

        font_size = 20 # Adjust as appropriate.
        cb.ax.tick_params(labelsize=font_size)

    if do_qualitative:

        delta = (vmax - vmin)/nbins

        bounds = np.array([vmin+n*delta for n in range(nbins+1)])
        norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=nbins)
        cb = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='horizontal')

        _t = bounds
        _tl = [round(_b,1) for _b in bounds]
        if ticklbl_int: _tl = [int(round(_b,0)) for _b in bounds]

        if nbins > 8: 
            _t = bounds[::2]
            _tl = _tl[::2]

        cb.set_ticks(_t)
        cb.ax.set_xticklabels(_tl)
        cb.ax.tick_params(labelsize=20)

    # if len(cb.ax.xaxis.get_ticklabels()) >= 7:
        # cb.locator = ticker.MaxNLocator(nbins=6)
        # cb.update_ticks()
    
    if False:
        if not do_qualitative and '[%]' in label or '(%)' in label: 
            label = label.replace(' [%]','').replace(' (%)','')
            cb.ax.set_xticklabels([_t.get_text()+r'%' for _t in cb.ax.get_xticklabels()])

        if not do_qualitative and '$' in label and '$\,$' not in label:
            cb.ax.set_xticklabels(['$'+_t.get_text() for _t in cb.ax.get_xticklabels()])

    if False:
        # drop final zero
        cb.ax.set_xticklabels([_t.get_text().replace('.0','') if _t.get_text()[-2:]=='.0' else _t.get_text() for _t in cb.ax.get_xticklabels()])
        # disgraceful 1-liner to keep colorbar axis uncluttered

    cb.set_label(label=label,size=20,weight=500,labelpad=14,linespacing=1.7)
    if path is not None:
        plt.savefig(path+".png",bbox_inches="tight",transparent=True,dpi=res)  
    plt.close(fig)    
    plt.close('all')

    return Image(path+".png", width=res)  

    
def n_to_one_normalizer(s,n=0):
  #affine transformation from s to [n,1]      
    y =(s-s.min())/(s.max()-s.min())
    return n+(1-n)*y
    
def bins_normalizer(x,n=7):
  #bins the data in n regular bins ( is better than pd.bin ? )     
    n=n-1
    y= n_to_one_normalizer(x,0)  #0 to 1 numbe
    return np.floor(n*y)/n

def quantile_normalizer(column, nb_quantile=5):
  #bbins in quintiles
    #print(column,nb_quantile)
    return (pd.qcut(column, nb_quantile,labels=False))/(nb_quantile-1)

def num_to_hex(x):
    h = hex(int(255*x)).split('x')[1]
    if len(h)==1:
        h="0"+h
    return h

def data_to_rgb(serie,
                color_maper=plt.cm.get_cmap("Blues_r"),
                nbins=5,
                normalizer = n_to_one_normalizer, 
                norm_param = 0, 
                na_color = "#e0e0e0",
                do_qual=False,
                force_min=None,
                force_max=None):
    """This functions transforms  data series into a series of color, using a colormap."""

    #if not do_qual:
    data_n = normalizer(serie,norm_param)
    #else: 
        # data_n = quantile_normalizer(serie)

    vmin = math.floor(force_min if force_min is not None else serie.min())
    vmax = math.ceil(force_max if force_max is not None else serie.max())
    ax1 = plt.gcf().add_axes([0.05, 0.80, 0.9, 0.15])

    delta = (vmax - vmin)/nbins
    bounds = np.array([vmin+n*delta for n in range(nbins+1)])
    norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=nbins)
    cb = mpl.colorbar.ColorbarBase(ax1, cmap=color_maper, norm=norm, orientation='horizontal')

    #here matplolib color mappers will just fill nas with the lowest color in the colormap
    colors = pd.DataFrame(color_maper(data_n),index=serie.index, columns=["r","g","b","a"]).applymap(num_to_hex)

    out = "#"+colors.r+colors.g+colors.b
    out[serie.isnull()]=na_color
    return out.str.upper()


def make_bined_legend(serie,bincolors,bins,label="",path=None, figsize=(9,9), formater=".0f", no_data_color = "#e0e0e0"):
    #todo: log flag
    #plt.rc('font', **font)
    
    patches =[]
    for i in np.arange(len(bincolors)):
        patches+=[mpatches.Patch( fc=bincolors[i],label=("{m:"+formater+"} — {M:"+formater+"}").format(m=bins[i],M=bins[i+1]))]
    
    patches+=[mpatches.Patch( fc=no_data_color,label="No data")]
    
    fig=plt.figure(figsize=figsize)
    ax=plt.gca()

def cleanup(path='maps/'):
    purge(path,'map_of_*.png')
    purge(path,'legend_of_*.png')
    purge(path,'map_of_*.svg')
    purge(path,'legend_of_*.svg')
