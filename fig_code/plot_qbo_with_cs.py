import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

test=sns.diverging_palette(255,15,sep=1,s=99,l=40,as_cmap=True)
def qbo_plot_with_cs(wind,tanom,pmin=0.9,pmax=200,tmin=0,tmax=10,cf_levels=np.linspace(-50,50,21),
             cs_levels=np.linspace(-40,40,9),figsize=(8,3),cmap=test,title='dummy',
             xlabel='Years since eruption',clblabel='Zonal wind [m/s]'):
    
    plt.figure(figsize=figsize)
    CF = wind.sel(lev=slice(pmin,pmax)).plot.contourf(x='time',levels=cf_levels,cmap=cmap,add_colorbar=False,extend='both')
    CS = tanom.sel(lev=slice(pmin,pmax)).plot.contour(x='time',levels=cs_levels,colors='black',add_colorbar=False)
#     CS.monochrome = True
    for line, lvl in zip(CS.collections, CS.levels):
        line.set_color('xkcd:grey')
        if lvl < 0:
            line.set_linestyle('--')
        elif lvl == 0:
            line.set_linestyle(':')
            line.set_color('k')
        else:
            # Optional; this is the default.
            line.set_linestyle('-')
    plt.clabel(CS,fmt='%1.0f',inline=False)
    plt.yscale('log')
    plt.gca().invert_yaxis()
    clb=plt.colorbar(CF,pad=0.01)
    clb.set_label(clblabel,fontsize=12)
    plt.xlim([tmin,tmax])
    plt.xlabel(xlabel,fontsize=14)
    plt.ylabel('Pressure level [hPa]',fontsize=14)
    plt.title(title,fontsize=14)