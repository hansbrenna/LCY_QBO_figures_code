import sys
import numpy as np
import xarray as xr
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from matplotlib.colors import Normalize
from numba import jit

def annualize_time_series(DataArray):
    number_of_years = int(np.floor(DataArray.time.shape[0]/12.))
    values = np.zeros([number_of_years])
    for i in range(number_of_years):
        ii = i*12
        values[i]=DataArray[ii:ii+12].mean()
    return values

def concatenate_individual_years(DataArray):
    number_of_years = int(np.floor(DataArray.time.shape[0]/12.))
    temp1 = DataArray.isel(time=slice(0,12)).copy(deep=True)
    temp2 = DataArray.isel(time=slice(12,24)).copy(deep=True)
    temp2.time.values=temp1.time.values
    temp3 = xr.concat([temp1,temp2],dim='record')
    #print(temp3.shape)
    for i in range(2,number_of_years):
        ii = i*12
     #   print(np.arange(ii,ii+12).shape)
        temp4 = DataArray[ii:ii+12]
        temp4.time.values=temp1.time.values
        temp3 = xr.concat([temp3,temp4],dim='record')
    #print(temp3)
    return temp3

def outside_2_sigma_tl(varstd,var):
    #varstd=varstd.std(dim='time')
    temp1=var.where(np.abs(var)>2*varstd)
    temp2=temp1.fillna(-999)
    temp3=temp2.where(temp2==-999)
    temp4=temp3.fillna(1.)
    temp5=temp4.where(temp4 != -999)
    sig95 = temp5.fillna(0.0)
    try:
        plt.contourf(sig95.lat,sig95.lev,sig95,levels=[0,0.1,1],hatches=['','..'],alpha=0.)
    except:
        print('Handle plotting of stipling manually')
    return sig95

def significant_95(r):
    #varstd=varstd.std(dim='time')
    temp1=r.where(r<0.05)
    temp2=temp1.fillna(-999)
    temp3=temp2.where(temp2==-999)
    temp4=temp3.fillna(1.)
    temp5=temp4.where(temp4 != -999)
    sig95 = temp5.fillna(0.0)
    try:
        plt.contourf(sig95.lat,sig95.lev,sig95,levels=[0,0.1,1],hatches=['','..'],alpha=0.)
    except:
        print('Handle plotting of stipling manually')
    return sig95

def outside_2_sigma(varstd,var):
    varstd=varstd.std(dim='time')
    temp1=var.where(np.abs(var)>2*varstd)
    temp2=temp1.fillna(-999)
    temp3=temp2.where(temp2==-999)
    temp4=temp3.fillna(1.)
    temp5=temp4.where(temp4 != -999)
    sig95 = temp5.fillna(0.0)
    try:
        plt.contourf(sig95.lat,sig95.lev,sig95,levels=[0,0.1,1],hatches=['','..'],alpha=0.)
    except:
        print('Handle plotting of stipling manually')
    return sig95

def pick_months(startyear=0,endyear=5,imonths=[5,6,7]):
    index = []
    for i in range(startyear,endyear):
        yearindex = list(np.array(imonths) + i*12)
        index += yearindex
    return index

def colordefs():
    colors={'deepblue':(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
    'pastelblue':(0.5725490196078431, 0.7764705882352941, 1.0),
    'darkblue':(0.0, 0.10980392156862745, 0.4980392156862745),
    'deepgreen':(0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
    'darkgreen':(0.00392156862745098, 0.4588235294117647, 0.09019607843137255),
    'pastelgreen':(0.592156862745098, 0.9411764705882353, 0.6666666666666666),
    'pastelred':(1.0, 0.6235294117647059, 0.6039215686274509),
    'deepred':(0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
    'darkred':(0.5490196078431373, 0.03529411764705882, 0.0),
    'pastelpurple':(0.8156862745098039, 0.7333333333333333, 1.0),
    'darkpurple':(0.4627450980392157, 0.0, 0.6313725490196078),
    'deeppurple':(0.5058823529411764, 0.4470588235294118, 0.6980392156862745)}
    return colors

def get_clim_with_same_dims(data, clim):
    target = data.shape
    current = clim.shape
    temp_clim = clim.copy(deep=True)
    counter = 0
    while current != target:
        temp_clim = xr.concat([temp_clim,clim],dim='time')
        current = temp_clim.shape
        counter += 1
        if counter >= 100:
            print('Could not make climatological data match shape of exp data. Tried {} times'.format(counter))
            print(temp_clim.shape)
            sys.exit()
#        print(current)
#        print(target)
    return temp_clim

def get_clim_with_same_dims_v2(data, clim):
    target = data['time'].shape
    current = clim['time'].shape
    temp_clim = clim.copy(deep=True)
    counter = 0
    while current != target:
        temp_clim = xr.concat([temp_clim,clim],dim='time')
        current = temp_clim['time'].shape
        counter += 1
        if counter >= 100:
            print('Could not make climatological data match shape of exp data. Tried {} times'.format(counter))
            print(temp_clim.shape)
            sys.exit()
#        print(current)
#        print(target)
    return temp_clim

def get_anomaly(data,clim):
    if 'record' in data.dims:
        temp = data.sel(record=0)
    else:
        temp = data.copy()
    factor = 1.0
    if data.name == 'TMO3':
        factor = 1.0/2.1415e-5
    clim_concat = get_clim_with_same_dims(temp,clim)
    #print(data.shape)
    #print(clim_concat.shape)
    replacement_time = (np.arange(clim_concat.time.shape[0])/12.)
    clim_concat.time.values=replacement_time
    anomaly = data*factor-clim_concat*factor
    return anomaly

def get_anomaly_v2(data,clim):
    if 'record' in data.dims:
        temp = data.sel(record=0)
    else:
        temp = data.copy()
    factor = 1.0
    if data.name == 'TMO3':
        factor = 1.0/2.1415e-5
    clim_concat = get_clim_with_same_dims(temp,clim)
    #print(data.shape)
    #print(clim_concat.shape)
    replacement_time = (np.arange(clim_concat.time.shape[0])/12.)
    clim_concat.time.values=replacement_time
    anomaly = data*factor-clim_concat.values*factor
    return anomaly

def get_relative_anomaly(data,clim):
    if 'record' in data.dims:
        temp = data.sel(record=0)
    else:
        temp = data.copy()
    factor = 1.0
    if data.name == 'TMO3':
        factor = 1.0/2.1415e-5
    clim_concat = get_clim_with_same_dims(temp,clim)
    #print(data.shape)
    #print(clim_concat.shape)
    replacement_time = (np.arange(clim_concat.time.shape[0])/12.)
    clim_concat.time.values=replacement_time
    anomaly = data*factor/clim_concat*factor
    return anomaly

def calculate_general_mass(da,ds,Mm,ulim,llim):
    """Function calculates the number of molecules of a gas between ulim and llim pressure levels and divides by number 
    of air molecules in same layer. This gives the overall concentration (molar mixing ratio ~ volume mixing ration) of 
    a gas in the column between ulim and llim. Parameters: da: xarray.DataArray containing gas mixing ratio; 
    ds: xarray.Dataset containing the other necessary parameters (hybrid coefficients, PS and P0); ulim: Upper pressure
    limit; llim: lower pressure limit. Call signature a = calculate_general_concentration(Cly,data,100,1)"""
    #gas_mmm = da*(MW/28.94)
    
    g=9.81
    Na = 6.022e23
    P0 = ds.P0
    PS = ds.PS
    hyai = ds.hyai
    hybi = ds.hybi
    Plevi = hyai*P0+hybi*PS
    #print(Plevi)
    
    dp = np.empty(shape=da.shape)
    
    dpa=xr.DataArray(dp,coords=da.coords,dims=da.dims)
    
    for i in range(1,Plevi.ilev.shape[0]):
        dpa[dict(lev=i-1)]=Plevi[dict(ilev=i)]-Plevi[dict(ilev=i-1)]
     
    mass_of_air_in_box = dpa/g
    #print(mass_of_air_in_box)
    #no_of_air_moles_in_box = dpa/28.94
    no_of_air_moles_in_box = mass_of_air_in_box*1000/28.94
    no_of_air_molec_in_box = no_of_air_moles_in_box*Na
    tot_no_of_air_molec = no_of_air_molec_in_box.sel(lev=slice(llim,ulim)).sum(dim='lev')
    tot_no_of_air_moles = no_of_air_moles_in_box.sel(lev=slice(llim,ulim)).sum(dim='lev')
    tot_mass_of_column_times_area = (tot_no_of_air_moles*28.94/1000.)*4*np.pi*6.3781e6**2
    #print(tot_mass_of_column_times_area)
    
    no_of_gas_moles_in_box = da*no_of_air_moles_in_box
    tot_no_of_gas_moles = no_of_gas_moles_in_box.sel(lev=slice(llim,ulim)).sum(dim='lev')
    #tot_no_of_gas_moles = tot_no_of_gas_molec/Na
    #print(tot_no_of_gas_moles)
    tot_mass_of_gas = tot_no_of_gas_moles*Mm/1000.
    #general_concentration = tot_no_of_gas_molec/tot_no_of_air_molec
    return tot_mass_of_gas

def KolmSmirnovTL(ctr, data):
    @jit
    def kolmsmirnovCalc(actr,adata,rdata,r_temp):
        for ti in range(actr.shape[1]):
            for lai in range(actr.shape[2]):
                r_temp = stats.ks_2samp(actr[:,ti,lai],adata[:,ti,lai])
                rdata[ti,lai] = r_temp[1]
        return rdata
    
    actr = ctr.values
    adata = data.values
    dummy = data.sel(record=0)
    rdata = np.zeros([ctr.shape[1],ctr.shape[2]])
    r_temp = 0.0
    rdata=kolmsmirnovCalc(actr,adata,rdata,r_temp)
    
    r = xr.DataArray(rdata,coords=dummy.coords,dims=dummy.dims)         
    return r

def return_day_array(time):
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30]
    temp = 0
    day_of_year = [temp]
    for i in days_in_month:
        temp += i
        day_of_year.append(temp)
    day_of_year = np.array(day_of_year)
    
    target = time.shape[0]
    dummy = 0
    yearp1 = day_of_year+365
    days = np.concatenate([day_of_year,yearp1])
    i = 0
    while dummy != target:
        yearp1 = yearp1+365
        days = np.concatenate([days,yearp1])
        dummy = days.shape[0]
        i+=1
        if i>100:
            break
    return days

def differentiate_DA(DataArray,time):
    dDAdt = DataArray.copy(deep=True)
    dDAdt[:] = 0.0
    dDAdt[:] = np.gradient(DataArray,time,axis=0,edge_order=2)
    return dDAdt

def differentiate_DA_lat(DataArray):
    a = 6.37e6
    phi = np.deg2rad(DataArray.lat).values
    dDAdy = DataArray.copy(deep=True)
    dDAdy[:] = 0.0
    dDAdy[:] = (1/a)*np.gradient(DataArray,(phi[1]-phi[0]),axis=-1,edge_order=2)
    return dDAdy

def differentiate_DA_z(DataArray,P0=100000):
    z = 7e3*np.log(P0/(100*DataArray.lev))
    dDAdz = DataArray.copy(deep=True)
    dDAdz[:] = 0.0
    dDAdz[:] = np.gradient(DataArray,z,axis=1,edge_order=2)
    return dDAdz