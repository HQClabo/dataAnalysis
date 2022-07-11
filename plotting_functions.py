# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:11:03 2022

@author: fopplige
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

#%% general plotting functions

def find_idx(array,values):
    try: iter(values)
    except TypeError:
        idx = (np.abs(array - values)).argmin()
    else:
        idx = np.zeros(len(values))
        for ii in range(len(values)):
            idx[ii] = (np.abs(array - values[ii])).argmin()
        idx.sort()
    return idx.astype(int)

# def array_range(array,vmin,vmax,axis=0):
#     if len(array.shape) > 1:
#         for ii in range(array.shape[(axis+1)%2]):

def array_span(array,vmin,vmax):    
    idx_min = find_idx(array,vmin)
    idx_max = find_idx(array,vmax)
    return array[idx_min:idx_max]

def plot_1D(x,y,labels=['',''],title='',linetype='-',fontsize=14,res=300):
    fig, ax = plt.subplots(1)
    fig.dpi = res
    ax.plot(x,y,linetype)
    ax.set_xlabel(labels[0], fontsize=fontsize)
    ax.set_ylabel(labels[1], fontsize=fontsize)
    ax.set_title(title)
    ax.grid()
    fig.tight_layout()
    return fig, ax

def plot_1D_mulitple(x,y,axis_labels=['',''],title='',linetype='-',fontsize=14,res=300):
    fig, ax = plt.subplots(1)
    fig.dpi = res
    for ii in range(len(x)):
        ax.plot(x[ii],y[ii],linetype)
    ax.set_xlabel(axis_labels[0], fontsize=fontsize)
    ax.set_ylabel(axis_labels[1], fontsize=fontsize)
    ax.set_title(title)
    ax.grid()
    fig.tight_layout()
    return fig, ax

def line_cut(x,y,z,cut_val,axis=0):
    # axis=0 means to take a cut along y-axis where the x-axis has the value cut_val
    if axis == 0:
        idx = (np.abs(x[0,:] - cut_val)).argmin()
        cutx = y[:,0]
        cuty = z[:,idx]
    else:
        if axis == 1:
            idx = (np.abs(y[:,0] - cut_val)).argmin()
            cutx = x[0,:]
            cuty = z[idx,:]
    return cutx, cuty

def plot_2D_linear(x,y,z,colormap='viridis',vmin=None,vmax=None,res=300):
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

def set_axis_size(w,h, ax=None):
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