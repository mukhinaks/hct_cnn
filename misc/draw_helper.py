# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:13:24 2020

@author: Ksenia Mukhina
"""
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects
from matplotlib.ticker import NullFormatter
from matplotlib import colors

colorschemes = {
    'Classic':  ['#7a6f9b','#f7d08a'],
   'Spectral': ['#2d728f', '#f7d08a', '#cc3f0c'] ,   
    'Spectral_V2': ['#159299', '#f7d08a', '#cc3f0c'] , 
    'GnYlRd': ['#508c36', '#f7d08a', '#cc3f0c'] , 
    'RdOrYl': ['#cc3f0c', '#ed943b','#f7d08a'],
    'Pink':  ['#7a6f9b', '#ed6da0', '#ed943b'],
    'BuGn':  ['#2d728f', '#159299', '#508c36'],
    'Violet':  ['#7a6f9b', '#2d728f', '#159299'],
    'Vivid':  ['#7a6f9b', '#159299', '#f7d08a'],
}
    
def alex_colors(colorschemes):       
    cm = colors.LinearSegmentedColormap.from_list(
        'alex', colorschemes, N=10)
    return cm

def show_colorschemes():  
    a = np.outer(np.arange(0,1,0.01),np.ones(10))
    fig, ax = plt.figure(figsize=(10,10))
    plt.subplots_adjust(top=0.8,bottom=0.05,left=0.01,right=0.99)
    
    l=len(colors)+1
    i = 0
    for t, c in colors.items():
        plt.subplot(l,1,i+1)
        i+= 1
        plt.axis("off")
        plt.imshow(a.T,aspect='auto',cmap=alex_colors(c),origin="lower")

def text_axes(ax, data, threshold, change_color_text = False):
        size_x, size_y = data.shape
        for i in range(size_x):
            for j in range(size_y):
                v = round(data[i, j].data.item())
                if abs(v) <= 1:
                    continue
                    
                if change_color_text:
                    if v <= threshold:
                        c = 'w'
                    else:
                        c = 'black'
                else: 
                    c = 'black'
                    
                if v >= threshold:
                    fontweight="bold"
                else:
                    fontweight="normal"
                
                fontweight="bold"
                ax.text(j, i, v, fontsize = 36, 
                        fontweight=fontweight,
                       ha="center", va="center", color=c, ).set_path_effects([
                    matplotlib.patheffects.Stroke(linewidth=5, foreground='w'), 
                                         matplotlib.patheffects.Normal()])

#Parameters of aspect for different cities
#0.8 - Wienna
#0.85 - London
#1.1 - Moscow
#0.88 - Spb
def axes_settings(ax, data, title, clim, color, norm = None, text = False, threshold = 0, aspect=1):
   # data = torch.t(data)
    ax.imshow(torch.round(data), 
                      aspect=aspect, 
                      cmap=color, 
                      interpolation = None, 
                      origin="lower",
                      clim=clim,
                      norm = norm
                     )

    ax.set_title(title, fontsize = 80, )

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    
    if text:
        text_axes(ax, data, threshold)        
    
    return ax

def draw_difference(prediction, labels):
    fig, axs = plt.subplots(1, 3, figsize=(90, 30))
    
    fig.subplots_adjust( wspace = 0.05 )
    
    max_value = max(min(torch.max(labels), 100), 1)
    clim = (0.0, max_value)    
    c = colorschemes['Vivid'] 
    
    axes_settings(axs[0], prediction, 'Prediction', clim, alex_colors(c), None, True, 0)
    axes_settings(axs[1], labels, "Ground truth", clim, alex_colors(c), None, True, 0)
    
               
     #MidPointNorm(midpoint=0)
    diff = labels - torch.round(prediction)
    border = 50 
    clim=(-border, border)
    norm = colors.DivergingNorm(vmin=-border, vcenter=0., vmax=border)
    c = colorschemes['Spectral_V2'] 
    axes_settings(axs[2], diff, "Difference", clim, alex_colors(c),norm, True, 0)

    plt.show()
    
def draw_pair(prediction, labels):
    fig, axs = plt.subplots(1, 2, figsize=(60, 30))
    
    fig.subplots_adjust( wspace = 0.05 )
    
    max_value = max(min(torch.max(labels), 100), 1)
    clim = (0.0, max_value)    
    c = colorschemes['Vivid'] 
    
    axes_settings(axs[0], prediction, 'Prediction', clim, alex_colors(c), None, True, 0)
    axes_settings(axs[1], labels, "Ground truth", clim, alex_colors(c), None, True, 0)

    plt.show()
    
def draw_labels(prediction, labels):
    fig, axs = plt.subplots(1, 1, figsize=(30, 30))
    
    max_value = max(min(torch.max(labels), 100), 1)
    clim = (0.0, max_value)    
    c = colorschemes['Vivid'] 
    
    axes_settings(axs, labels, "Ground truth", clim, alex_colors(c), None, True, 0)
    plt.show()
    
def draw_boxplot(stats):
    fig, ax = plt.subplots(1, 1, figsize=(40, 5))
    boxprops = dict(
                linewidth=3, 
                facecolor='#f7d08a')
    medianprops = dict( linewidth=5, color='#7a6f9b')
    
    flierprops = dict(marker='o', markerfacecolor='#159299', markersize=12,
                  linestyle='none')
    whiskerprops = dict(linewidth=3,)
    
    capprops = dict(linewidth=3,)
    
    ax.yaxis.set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.grid(True)
    ax.tick_params(labelsize  = 24)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))

    ax.boxplot([x['SSIM'] for x in stats], vert=False, 
               boxprops = boxprops, patch_artist=True, medianprops = medianprops, widths = 0.5,
               flierprops = flierprops, whiskerprops  = whiskerprops, capprops  = capprops, showfliers = True  )