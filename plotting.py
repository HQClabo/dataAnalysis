# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:42:58 2022

@author: jouanny
"""

import matplotlib.pyplot as plt

def plot2D(x,y,z, axis = ["","",""], cmap = "coolwarm", size = 15, dpi = 600, vmin = None, vmax = None, title = "", figsize = None):
    fig, ax = plt.subplots(1, dpi = dpi, figsize = figsize)
    im = ax.pcolormesh(x, y, z, cmap = cmap, vmin = vmin, vmax = vmax)
    ax.set_xlabel(axis[0], size = size)
    ax.set_ylabel(axis[1], size = size)
    ax.tick_params(labelsize = size)
    ax.set_title(title)
    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize = size)
    cbar.set_label(label = axis[2], size = size)
    plt.tight_layout()

    fig.tight_layout()
    return fig, ax, cbar

def plot1D(x,y, axis = ["",""], title = "", grid = True, size = 15, dpi = 600, label = ""):
    fig, ax = plt.subplots()
    ax.set_xlabel(axis[0], size = size)
    ax.set_ylabel(axis[1], size = size)
    ax.set_title(title, size = size)
    ax.tick_params(labelsize = size)
    ax.plot(x,y, label = label)
    ax.grid(grid)
    plt.tight_layout()

    fig.tight_layout()
    return fig, ax