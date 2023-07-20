# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:11:03 2022

@author: fopplige

A module that contains general functions for data manipulation and plotting
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import uniform_filter1d

#%% functions for data extraction and array manipulation

def find_idx(array,values,sorting=True):
    try: iter(values)
    except TypeError:
        idx = (np.abs(array - values)).argmin()
    else:
        idx = np.zeros(len(values))
        for ii in range(len(values)):
            idx[ii] = (np.abs(array - values[ii])).argmin()
        if sorting:
            idx.sort()
    return idx.astype(int)

def find_slice(array,values):
    idx_min = (np.abs(array - values[0])).argmin()
    idx_max = (np.abs(array - values[1])).argmin() + 1
    return slice(idx_min,idx_max)

def slice_array(array,slice_length):
    slice_length = int(slice_length)
    n = array.shape
    array_out = np.zeros([int(n[0]/slice_length),slice_length,n[1]])
    for ii in range(int(n[0]/slice_length)):
        array_out[ii] = array[ii:(ii+slice_length)]
    return array_out

def array_span(array,vmin,vmax):    
    idx_min = find_idx(array,vmin)
    idx_max = find_idx(array,vmax)
    return array[idx_min:idx_max]

def line_cut(x,y,z,cut_val,axis=0):
    # axis=0 means to take a cut along y-axis where the x-axis has the value cut_val
    if len(x.shape) == 1:
        x_ax = x
    else:
        x_ax = x[0,:]
    if len(y.shape) == 1:
        y_ax = y
    else:
        y_ax = y[:,0]
    
    if axis == 0:
        idx = (np.abs(x_ax - cut_val)).argmin()
        cutx = y_ax
        cuty = z[:,idx]
    else:
        if axis == 1:
            idx = (np.abs(y_ax - cut_val)).argmin()
            cutx = x_ax
            cuty = z[idx,:]
    return cutx, cuty


#%% functions for data manipulation

def rotate_complex(number,theta):
    # rotates a complex number by angle theta in degrees 
    return number * np.exp(1j * 2*np.pi * theta/360)

def moving_average(x,N,axis=-1):
    return uniform_filter1d(x, N, axis=axis, mode='constant', origin=0)
    
def phase_unwrap(freq,cData,fit_range=0.05):
    length = len(freq)
    unwr_phase = np.unwrap(np.angle(cData))
    delay1,_ = np.polyfit(freq[0:int(fit_range*length)],unwr_phase[0:int(fit_range*length)],1)
    delay2,_ = np.polyfit(freq[int((1-fit_range)*length):-1],unwr_phase[int((1-fit_range)*length):-1],1)
    delay_avg = (delay1+delay2)/2
    cData_rot = cData*np.exp(-1j*delay_avg*freq)
    cData_rot_zero = cData_rot*np.exp(-1j*np.angle(cData_rot[0]))
    return np.angle(cData_rot_zero)


#%% plotting functions

def plot_1D(x,y,ax=None,labels=['',''],title='',linetype='-',fontsize=14,res=300):
    if ax==None:
        fig, ax = plt.subplots(1)
    else:
        fig = ax.get_figure()
    fig.dpi = res
    ax.plot(x,y,linetype)
    ax.set_xlabel(labels[0], fontsize=fontsize)
    ax.set_ylabel(labels[1], fontsize=fontsize)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig, ax

def plot_1D_multiple(x,y,ax=None,axis_labels=['',''],title='',linetype='-',fontsize=14,res=300):
    if ax==None:
        fig, ax = plt.subplots(1)
    else:
        fig = ax.get_figure()
    fig.dpi = res
    for ii in range(len(x)):
        ax.plot(x[ii],y[ii],linetype)
    ax.set_xlabel(axis_labels[0], fontsize=fontsize)
    ax.set_ylabel(axis_labels[1], fontsize=fontsize)
    ax.set_title(title)
    ax.grid()
    fig.tight_layout()
    return fig, ax

def plot_2D(x,y,z,colormap='viridis',vmin=None,vmax=None,labels=['','',''],title='',logscale=False,aspect=1,fontsize=14,res=300):
    fig, ax = plt.subplots(1)
    fig.dpi = res
    if logscale:
        z_plot = abs(z)
    else:
        z_plot = z
    if vmin is None:
        vmin = z_plot.min()
    if vmax is None:
        vmax = z_plot.max()
    # ax.set_aspect((x.max()-x.min())/(y.max()-y.min()))
    if logscale:
        plot2d = ax.pcolormesh(x,y,abs(z),cmap=colormap,norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    else:
        plot2d = ax.pcolormesh(x,y,z,cmap=colormap,vmin=vmin,vmax=vmax)
    ax.set_aspect(aspect/ax.get_data_ratio())
    ax.set_xlabel(labels[0], fontsize=fontsize)
    ax.set_ylabel(labels[1], fontsize=fontsize)
    ax.set_title(title)
    cb = fig.colorbar(plot2d,ax=ax,fraction=0.047*aspect)
    cb.set_label(labels[2], fontsize=fontsize)
    fig.tight_layout()
    return fig, ax, cb

def plot_ampl_and_phase(freq,cData,label='',filename=False,file_res=300,fit_range=0.05):
    fig, ax = plt.subplots(2)
    fig.dpi = file_res
    fig.set_figheight(5)
    fig.set_figwidth(9)
    ax[0].plot(freq,20*np.log10(np.abs(cData)),label='Amplitude (dB)')
    ax[1].plot(freq,phase_unwrap(freq,cData,fit_range=fit_range)/np.pi,label=r'Phase ($\pi$)')
    ax[0].set_ylabel('Amplitude (dB)')
    #ax[0].set_xticklabels([])
    ax[1].set_ylabel(r'Phase ($\pi$)')
    ax[1].set_xlabel('Frequency (Hz)')
    fig.suptitle(label)
    ax[0].grid()
    ax[1].grid()
    fig.tight_layout()
    if filename:
        plt.savefig(filename,dpi=file_res)

def plot_ampl(freq,cData,label='',filename=False,file_res=300):
    fig, ax = plt.subplots(1)
    fig.dpi = file_res
    fig.set_figheight(5)
    fig.set_figwidth(9)
    if len(freq.shape) == 2:
        for ii in range(freq.shape[0]):
            ax.plot(freq[ii],20*np.log10(np.abs(cData[ii])),label='Amplitude (dB)')
    else:
        ax.plot(freq,20*np.log10(np.abs(cData)),label='Amplitude (dB)')
    ax.set_ylabel('Amplitude (dB)')
    ax.set_xlabel('Frequency (Hz)')
    fig.suptitle(label)
    ax.grid()
    fig.tight_layout()
    if filename:
        plt.savefig(filename,dpi=file_res)

def plot_peak_vs_power(freq,cData,power,label=''):
    for ii in range(len(power)):
        plt.plot(freq,np.abs(cData)[ii],label=f'power = {power[ii]}')
        #plt.xlim(4.35e9,4.425e9)
        # plt.plot(FREQ, np.angle(cData))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title(f'Resonator {label}')
    plt.legend()
    plt.tight_layout()
    #plt.savefig(f'Modulo_VS_frequency_res_{label}.png',dpi=600)
    plt.show()

def plot2D_power_VS_freq(out,freq,cData,power,p_in=0,label='0'):
    po=[]
    for ii in range(len(out.power_at)):
        po.append(out.power_at[ii]+80)

    fig,ax = plt.subplots()
    im = ax.imshow(np.abs(cData),cmap=plt.cm.Reds,aspect='auto', extent=[freq[0],freq[-1],power[-1],power[0]])
    ax.plot(out.fr_res[p_in:],po[p_in:],'o-',label='peak 1')
        #im = plt.pcolormesh(mod, cmap='RdBu',shading='gouraud')
        #cset = ax.contour(mod,np.arange(0,1,0.2),linewidths=2,cmap=plt.cm.Set2)
        #ax.clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
    ax.set_title(f'Resonator {label}')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Modulo (dB)')
    ax.grid(True,alpha=0.5)
    fig.colorbar(im)
    fig.tight_layout()
    #fig.savefig(f'Plot2D_power_VS_frequency_Res_{label}.png',dpi=600)


#%% functions for manipulating plots

def format_plot(ax,
                title='',
                xlabel='',
                ylabel='',
                clabel='',
                x_lim=None,
                y_lim=None,
                res=300,
                fontsize=None,
                grid=False,
                label_kw={},
                tick_kw={},
                ):
    
    fig = ax.get_figure()
    fig.dpi = res
    if title: fig.suptitle(title,fontsize=fontsize)
    if xlabel: ax.set_xlabel(xlabel,fontsize=fontsize)
    if ylabel: ax.set_ylabel(ylabel,fontsize=fontsize)
    ax.tick_params(**tick_kw)
    ax.grid(grid)
    if x_lim: ax.set_xlim(*x_lim)
    if y_lim: ax.set_ylim(*y_lim)

def format_colorbar(cb,
                    clabel='',
                    fontsize=None,
                    labelpos='right',
                    label_kw={},
                    tick_kw={}
                    ):
    if labelpos=='right': cb.set_label(clabel,fontsize=fontsize,**label_kw)
    if labelpos=='top': cb.ax.set_title(clabel,fontsize=fontsize,**label_kw)
    cb.ax.tick_params(**tick_kw)

def set_axis_size(w,h, ax=None, tight_layout=True):
    """ w, h: width, height """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_figwidth(figw)
    ax.figure.set_figheight(figh)
    if tight_layout: ax.figure.tight_layout()
    
def force_aspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
    
def set_axis_labels(ax,title='',xlabel='',ylabel='',fontsize=14,ticksize=12):
    if title: ax.set_title(title,fontsize=fontsize)
    if xlabel: ax.set_xlabel(xlabel,fontsize=fontsize)
    if ylabel: ax.set_ylabel(ylabel,fontsize=fontsize)
    ax.tick_params(labelsize=ticksize)

def set_colorbar_labels(cb,clabel='',fontsize=14,labelsize=12):
    cb.set_label(clabel,fontsize=fontsize)
    cb.ax.tick_params(labelsize=labelsize)


#%% old functions

def plot_2D_linear(x,y,z,colormap='viridis',vmin=None,vmax=None,res=300):
    """ now included in plot_2D """
    fig, ax = plt.subplots(1)
    fig.dpi = res
    if vmin is None:
        vmin = z.min()
    if vmax is None:
        vmax = z.max()
    ax.set_aspect((x.max()-x.min())/(y.max()-y.min()))
    plot2d = ax.pcolormesh(x,y,z,cmap=colormap,vmin=vmin,vmax=vmax)
    cb = fig.colorbar(plot2d,ax=ax)
    fig.tight_layout()
    return fig, ax, cb

def plot_2D_clog(x,y,z,colormap='viridis',vmin=None,vmax=None,res=300):
    """ now included in plot_2D """
    fig, ax = plt.subplots(1)
    fig.dpi = res
    if vmin is None:
        vmin = abs(z).min()
    if vmax is None:
        vmax = abs(z).max()
    ax.set_aspect((x.max()-x.min())/(y.max()-y.min()))
    plot2d = ax.pcolormesh(x,y,abs(z),cmap=colormap,norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(plot2d,ax=ax)
    fig.tight_layout()
    return fig, ax, cb

#%% Coulomb plots

def coulomb_diamond_curr(x,y,z,vmin=None,vmax=None,log=False,title='',res=300,fontsize=14):
    if log:
        fig, ax, cb = plot_2D_clog(x,y*1e3,z*1e12,colormap='terrain',vmin=vmin,vmax=vmax)
    else:
        fig, ax, cb = plot_2D_linear(x,y*1e3,z*1e12,vmin=vmin,vmax=vmax)
    ax.set_xlabel('$V_{\mathrm{bg}}$ (V)', fontsize=fontsize)
    ax.set_ylabel('$V_{\mathrm{bias}}$ (mV)', fontsize=fontsize)
    cb.set_label(r'$I_\mathrm{d}$ (pA)',fontsize=fontsize)
    fig.suptitle(title)
    fig.tight_layout()
    
def coulomb_diamond_diff(x,y,z,vmin=None,vmax=None,log=False,title='',res=300,fontsize=14):
    # grady, gradx = np.gradient(current,y[:,0],x[0])
    diff = np.gradient(z,y[:,0],axis=0)
    if log:
        fig, ax, cb = plot_2D_clog(x,y*1e3,diff,colormap='hot',vmin=vmin,vmax=vmax)
    else:
        fig, ax, cb = plot_2D_linear(x,y*1e3,diff,colormap='hot',vmin=vmin,vmax=vmax)
    ax.set_xlabel('$V_{\mathrm{bg}}$ (V)',fontsize=fontsize)
    ax.set_ylabel('$V_{\mathrm{bias}}$ (mV)',fontsize=fontsize)
    cb.set_label('$\partial I_\mathrm{d} / \partial V_{\mathrm{bias}}$ (S)',fontsize=fontsize)
    fig.suptitle(title)
    fig.tight_layout()