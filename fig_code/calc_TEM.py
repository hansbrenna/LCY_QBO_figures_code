import numpy as np
import pandas as pd
import xarray as xr
import Ngl

def calc_TEM_from_h6_file(data):
    #define constants
    H  = 7.0e3
    a = 6.37e6
    g  = 9.8
    om = 7.292e-5  
    R = 287.05
    Cp = 1.0035e3
    
    #load data from dataset
    P0 = data.P0.values
    time=data.time
    ntime = time.shape[0]
    lat = data.lat
    nlat = lat.shape[0]
    lon = data.lon
    nlon = lon.shape[0]
    phi = np.deg2rad(lat)
    lev = data.lev
    nlev = lev.shape[0]
    ilev = data.ilev
    nilev = ilev.shape[0]
    coslat = np.cos(phi)
    sinlat = np.sin(phi)
    zp = H*np.log(P0/(100*lev))
    zpi = H*np.log(P0/(100*ilev))
    rho = P0/(g*H) * np.exp(-zp/H)

    hyai = data.hyai
    hyam = data.hyam
    hybi = data.hybi
    hybm = data.hybm
    
    PS = data.PS
    U = data['Uzm']
    U = U.where(U<1e10)
    V = data['Vzm']
    V = V.where(V<1e10)
    W = data['Wzm']
    W = W.where(W<1e10)
    TH = data['THzm']
    TH = TH.where(TH<1e10)
    VTH = data['VTHzm']
    VTH = VTH.where(VTH<1e10)
    UV = data['UVzm']
    UV = UV.where(UV<1e10)
    UW = data['UWzm']
    UW = UW.where(UW<1e10)
    
    #create dummy DataArray object to hold interpolated variables
    dummy_arr = np.zeros(shape=[ntime,nlev,nlat])
    dummy_out = xr.DataArray(data=dummy_arr,dims=('time','lev','lat'),coords={'time':time,'lev':lev,'lat':lat})
    Up = dummy_out.copy(deep=True)
    Vp = dummy_out.copy(deep=True)
    Wp = dummy_out.copy(deep=True)
    THp = dummy_out.copy(deep=True)
    VTHp = dummy_out.copy(deep=True)
    UVp = dummy_out.copy(deep=True)
    UWp = dummy_out.copy(deep=True)
    
    #interpolate fields to midpoint pressure levels
    Up[:] = Ngl.vinth2p(np.broadcast_to(U.values[...,None],U.shape+(nlon,)),hyai.values,hybi.values,lev.values,PS.values,1,P0/100,1,True).mean(axis=-1)
    Vp[:] = Ngl.vinth2p(np.broadcast_to(V.values[...,None],V.shape+(nlon,)),hyai.values,hybi.values,lev.values,PS.values,1,P0/100,1,True).mean(axis=-1)
    Wp[:] = Ngl.vinth2p(np.broadcast_to(W.values[...,None],W.shape+(nlon,)),hyai.values,hybi.values,lev.values,PS.values,1,P0/100,1,True).mean(axis=-1)
    THp[:] = Ngl.vinth2p(np.broadcast_to(TH.values[...,None],TH.shape+(nlon,)),hyai.values,hybi.values,lev.values,PS.values,1,P0/100,1,True).mean(axis=-1)
    VTHp[:] = Ngl.vinth2p(np.broadcast_to(VTH.values[...,None],VTH.shape+(nlon,)),hyai.values,hybi.values,lev.values,PS.values,1,P0/100,1,True).mean(axis=-1)
    UVp[:] = Ngl.vinth2p(np.broadcast_to(UV.values[...,None],UV.shape+(nlon,)),hyai.values,hybi.values,lev.values,PS.values,1,P0/100,1,True).mean(axis=-1)
    UWp[:] = Ngl.vinth2p(np.broadcast_to(UW.values[...,None],UW.shape+(nlon,)),hyai.values,hybi.values,lev.values,PS.values,1,P0/100,1,True).mean(axis=-1)
    
    #calculate d\bar\theta/dz
    thbar_z = dummy_out.copy(deep=True)
    thbar_z[:] = np.gradient(THp,zp,axis=-2,edge_order=2)
    
    thbar_z.values[(thbar_z<1e-4).values]=1e-4 #put floor on thbar_z. Do not know why but from NCAR script
    
    #calculate wbar*
    derivand=(coslat*VTHp/thbar_z).transpose('time','lev','lat')
    derivative = dummy_out.copy(deep=True)
    derivative[:] = np.gradient(derivand,phi,axis=-1,edge_order=2)
    d2y = dummy_out.copy(deep=True)
    d2y[:] = np.gradient(derivative,phi,axis=-1)
    
    wstar = Wp+derivative/(a*coslat)
    wstar[:,:,0] = Wp[:,:,0]-(1./(a*sinlat[0]))*d2y[:,:,0]
    wstar[:,:,-1] = Wp[:,:,-1]-(1./(a*sinlat[-1]))*d2y[:,:,-1]
    return wstar

def calc_TEM_from_h0_file(data):
#define constants
    H  = 7.0e3
    a = 6.37e6
    g  = 9.8
    om = 7.292e-5  
    R = 287.05
    Cp = 1.0035e3
    
    print('loading data...')

    #load data from dataset
    P0 = data.P0.values
    time=data.time
    ntime = time.shape[0]
    lat = data.lat
    nlat = lat.shape[0]
    nlon = 4
    phi = np.deg2rad(lat)
    lev = data.lev
    nlev = lev.shape[0]
    ilev = data.ilev
    nilev = ilev.shape[0]
    coslat = np.cos(phi)
    sinlat = np.sin(phi)
    zp = H*np.log(P0/(100*lev))
    zpi = H*np.log(P0/(100*ilev))
    rho = P0/(g*H) * np.exp(-zp/H)

    hyai = data.hyai
    hyam = data.hyam
    hybi = data.hybi
    hybm = data.hybm

    PS = data.PS
    U = data['U']
    U = U.where(U<1e10)
    V = data['V']
    V = V.where(V<1e10)
    OMG = data['OMEGA']
    OMG = OMG.where(OMG<1e10)
    W = (-H/(lev*100)*OMG).transpose('time','lev','lat')
    TH = data['TH']
    TH = TH.where(TH<1e10)
    VTH = data['VTHzm'].squeeze(dim='zlon')
    VTH = VTH.where(VTH<1e10)
    UV = data['UVzm'].squeeze(dim='zlon')
    UV = UV.where(UV<1e10)
    UW = data['UWzm'].squeeze(dim='zlon')
    UW = UW.where(UW<1e10)
    
    print('Finished loading data')
    print('Interpolating...')

    #create dummy DataArray object to hold interpolated variables
    dummy_arr = np.zeros(shape=[ntime,nlev,nlat])
    dummy_out = xr.DataArray(data=dummy_arr,dims=('time','lev','lat'),coords={'time':time,'lev':lev,'lat':lat})
    Up = dummy_out.copy(deep=True)
    Vp = dummy_out.copy(deep=True)
    Wp = dummy_out.copy(deep=True)
    THp = dummy_out.copy(deep=True)
    VTHp = dummy_out.copy(deep=True)
    UVp = dummy_out.copy(deep=True)
    UWp = dummy_out.copy(deep=True)

    Up[:] = Ngl.vinth2p(np.broadcast_to(U.values[...,None],U.shape+(nlon,)),hyam.values,hybm.values,lev.values,np.broadcast_to(PS.values[...,None],PS.shape+(nlon,)),1,P0/100,1,True).mean(axis=-1)
    Vp[:] = Ngl.vinth2p(np.broadcast_to(V.values[...,None],V.shape+(nlon,)),hyam.values,hybm.values,lev.values,np.broadcast_to(PS.values[...,None],PS.shape+(nlon,)),1,P0/100,1,True).mean(axis=-1)
    Wp[:] = Ngl.vinth2p(np.broadcast_to(W.values[...,None],W.shape+(nlon,)),hyam.values,hybm.values,lev.values,np.broadcast_to(PS.values[...,None],PS.shape+(nlon,)),1,P0/100,1,True).mean(axis=-1)
    THp[:] = Ngl.vinth2p(np.broadcast_to(TH.values[...,None],TH.shape+(nlon,)),hyai.values,hybi.values,lev.values,np.broadcast_to(PS.values[...,None],PS.shape+(nlon,)),1,P0/100,1,True).mean(axis=-1)
    VTHp[:] = Ngl.vinth2p(np.broadcast_to(VTH.values[...,None],VTH.shape+(nlon,)),hyai.values,hybi.values,lev.values,np.broadcast_to(PS.values[...,None],PS.shape+(nlon,)),1,P0/100,1,True).mean(axis=-1)
    UVp[:] = Ngl.vinth2p(np.broadcast_to(UV.values[...,None],UV.shape+(nlon,)),hyai.values,hybi.values,lev.values,np.broadcast_to(PS.values[...,None],PS.shape+(nlon,)),1,P0/100,1,True).mean(axis=-1)
    UWp[:] = Ngl.vinth2p(np.broadcast_to(UW.values[...,None],UW.shape+(nlon,)),hyai.values,hybi.values,lev.values,np.broadcast_to(PS.values[...,None],PS.shape+(nlon,)),1,P0/100,1,True).mean(axis=-1)

    print('Finished interpolating')
    print('Calculating TEM terms')
    
     #calculate d\bar\theta/dz
    thbar_z = dummy_out.copy(deep=True)
    thbar_z[:] = np.gradient(THp,zp,axis=-2,edge_order=2)

    thbar_z.values[(thbar_z<1e-4).values]=1e-4 #put floor on thbar_z. Do not know why but from NCAR script

    print('Calculating wbar*')
    #calculate wbar*
    derivand=(coslat*VTHp/thbar_z).transpose('time','lev','lat')
    derivative = dummy_out.copy(deep=True)
    derivative[:] = np.gradient(derivand,phi,axis=-1,edge_order=2)
    d2y = dummy_out.copy(deep=True)
    d2y[:] = np.gradient(derivative,phi,axis=-1)

    wstar = Wp+derivative/(a*coslat)
    wstar[:,:,0] = Wp[:,:,0]-(1./(a*sinlat[0]))*d2y[:,:,0]
    wstar[:,:,-1] = Wp[:,:,-1]-(1./(a*sinlat[-1]))*d2y[:,:,-1]
    
    print('Calculating vbar*')
    #calculate vbar*
    derivand = (rho*VTHp/thbar_z).transpose('time','lev','lat')
    derivative = dummy_out.copy(deep=True)
    derivative[:] = np.gradient(derivand,zp,axis=-2,edge_order=2)
    vstar = Vp-derivative
    
    print('Calculating EP-flux')
    #calculate EP-flux
    fc = 2.*om*sinlat
    ubar_z = dummy_out.copy(deep=True)
    ubar_z[:] = np.gradient(Up,zp,axis=-2,edge_order=2)
    ubar_y = dummy_out.copy(deep=True)
    ucoslat = (Up*coslat).transpose('time','lev','lat')
    ubar_y[:] = (np.gradient(ucoslat.values,phi.values,axis=-1))
    ubar_y = (ubar_y/(a*coslat)).transpose('time','lev','lat')
    fy = dummy_out.copy(deep=True)
    fz = dummy_out.copy(deep=True)
    fy[:] = ((ubar_z*VTHp/thbar_z-UVp)*rho*a*coslat).transpose('time','lev','lat')
    fy[...,0] = 0.0
    fy[...,-1] = 0.0
    fz[:] = ((fc-ubar_y*VTHp/thbar_z-UWp)*rho*a*coslat).transpose('time','lev','lat')
    fz[...,0] = 0.0
    fz[...,-1] = 0.0
    
    print('Calculating divergence of EP-flux')
    
    fycoslat = (coslat*fy).transpose('time','lev','lat')
    fyy = dummy_out.copy(deep=True)
    fyy[:] = np.gradient(fycoslat,phi,axis=-1)
    fyy = (1/(a*coslat)*fyy).transpose('time','lev','lat')
    fyy = (fyy/(a*rho*coslat)).transpose('time','lev','lat')
    
    fyy[...,0] = 0.0
    fyy[...,-1] = 0.0
    
    fzz = dummy_out.copy(deep=True)
    fzz[:] = np.gradient(fz,zp,axis=-2)
    fzz = (fzz/(a*rho*coslat)).transpose('time','lev','lat')
    
    fzz[...,0] = 0.0
    fzz[...,-1] = 0.0
    
    print('Finished')
    return wstar,vstar,fy,fz,fyy,fzz