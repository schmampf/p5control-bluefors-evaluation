import numpy as np
import matplotlib.pyplot as plt
import warnings

"""
Save Figure
"""

from matplotlib.figure import Figure
from os import listdir, mkdir

def save_figure(
        fig:Figure,
        title:str,
        path:str = 'figures',
        pdf=True,
        ):
    if path is not None:
        title = f'{path}/{title}'
        if path not in listdir():
            mkdir(path)

    fig.savefig(f'{title}.png')
    if pdf:
        fig.savefig(f'{title}.pdf', dpi=600)


"""
Plot IV
"""

def IV_plotting(    
    img,
    x,
    y,
    x_lim = None,
    y_lim = None,
    xlabel=r'$V$ (mV)',
    ylabel=r'$\mu_0H$ (mT)',
    clabel=r'd$I/$d$V$ ($G_0$)',
    title=None,
    suptitle=None,
    clim = None,
    cmap = 'viridis',
    contrast = 1,
    fig_nr = 0,
    dpi=None,
    pdf=False,
    path = 'figures'
    ):
    if img.dtype == np.dtype('int32'):
        warnings.warn("img is integer. Sure?")

    stepsize_x=np.abs(x[-1]-x[-2])/2
    stepsize_y=np.abs(y[-1]-y[-2])/2
    if x_lim is None:
        x_ind = [0, -1]
    else:
        x_ind = [np.abs(x-x_lim[0]).argmin(),
                np.abs(x-x_lim[1]).argmin()]
    if y_lim is None:
        y_ind = [0, -1]
    else:
        y_ind = [np.abs(y-y_lim[0]).argmin(),
                np.abs(y-y_lim[1]).argmin()]
    ext = [x[x_ind[0]]-stepsize_x,
           x[x_ind[1]]+stepsize_x,
           y[y_ind[0]]-stepsize_y,
           y[y_ind[1]]+stepsize_y]
    img = img[y_ind[0]:y_ind[1],
              x_ind[0]:x_ind[1]]
    x = x[x_ind[0]:x_ind[1]]
    y = y[y_ind[0]:y_ind[1]]

    if clim is None:
        clim = [np.nanmean(img)-np.nanstd(img)/contrast, 
                np.nanmean(img)+np.nanstd(img)/contrast]
    if dpi is None:
        dpi=300

    plt.close(fig_nr)
    fig, (ax_img, ax_clb) = plt.subplots(
        num=fig_nr,
        ncols=2,
        figsize=(6,4),
        dpi=dpi,
        gridspec_kw={"width_ratios":[5.8,.2]},
        constrained_layout=True
        )

    im = ax_img.imshow(img, 
                       extent=ext, 
                       aspect='auto',
                       origin='lower',
                       clim=clim,
                       cmap=cmap,
                       interpolation='none')
    ax_img.set_xlabel(xlabel)
    ax_img.set_ylabel(ylabel)
    ax_img.ticklabel_format(
        axis="both", 
        style="sci", 
        scilimits=(-3,3),
        useMathText=True
    )
    ax_img.tick_params(direction='in')

    cbar=fig.colorbar(im, label=clabel, cax=ax_clb)
    ax_clb.tick_params(direction='in')
    lim = ax_img.set_xlim(ext[0],ext[1])
    lim = ax_img.set_ylim(ext[2],ext[3])

    if suptitle is None:
        plt.suptitle(title)
    else:
        plt.suptitle(suptitle)

    if title is not None:
        save_figure(fig, title, pdf=pdf, path=path)

    return fig, ax_img, ax_clb, ext


"""
Plot IVx map and x_n
"""

def IV_T_plotting(    
    img,
    x,
    y,
    x_n,
    x_lim = None,
    y_lim = None,
    xlabel=r'$V$ (mV)',
    xnlabel=r'$T$ (mK)',
    ylabel=r'$\mu_0H$ (mT)',
    clabel=r'd$I/$d$V$ ($G_0$)',
    title=None,
    suptitle=None,
    clim = None,
    cmap = 'viridis',
    contrast = 1,
    fig_nr = 0,
    dpi=None,
    ):
    if img.dtype == np.dtype('int32'):
        warnings.warn("img is integer. Sure?")

    stepsize_x=np.abs(x[-1]-x[-2])/2
    stepsize_y=np.abs(y[-1]-y[-2])/2
    if x_lim is None:
        x_ind = [0, -1]
    else:
        x_ind = [np.abs(x-x_lim[0]).argmin(),
                np.abs(x-x_lim[1]).argmin()]
    if y_lim is None:
        y_ind = [0, -1]
    else:
        y_ind = [np.abs(y-y_lim[0]).argmin(),
                np.abs(y-y_lim[1]).argmin()]
    ext = [x[x_ind[0]]-stepsize_x,
           x[x_ind[1]]+stepsize_x,
           y[y_ind[0]]-stepsize_y,
           y[y_ind[1]]+stepsize_y]
    img = img[y_ind[0]:y_ind[1],
              x_ind[0]:x_ind[1]]
    x = x[x_ind[0]:x_ind[1]]
    y = y[y_ind[0]:y_ind[1]]
    x_n = x_n[y_ind[0]:y_ind[1]]

    if clim is None:
        clim = [np.nanmean(img)-np.nanstd(img)/contrast, 
                np.nanmean(img)+np.nanstd(img)/contrast]
    if dpi is None:
        dpi=300

    plt.close(fig_nr)
    # fig=plt.figure(fig_nr)
    fig, (ax_plot, ax_img, ax_clb) = plt.subplots(
        num=fig_nr,
        ncols=3,
        figsize=(6,4),
        dpi=dpi,
        gridspec_kw={"width_ratios":[1.3,4.5,.2]},
        constrained_layout=True
        )
    ax_plot.plot(x_n, y)
    ax_plot.grid()
    ax_plot.invert_xaxis()
    ax_plot.set_ylabel(ylabel)
    ax_plot.set_xlabel(xnlabel)
    ax_plot.ticklabel_format(
        axis="both", 
        style="sci", 
        scilimits=(-3,3),
        useMathText=True
    )
    ax_plot.tick_params(direction='in')
    # ax_plot.set_xticks([.6, .4, .2])
    # ax_plot.tick_params(axis='x', labelrotation=45)

    im = ax_img.imshow(img, 
                       extent=ext, 
                       aspect='auto',
                       origin='lower',
                       clim=clim,
                       cmap=cmap,
                       interpolation='none')
    ax_img.set_xlabel(xlabel)
    ax_img.ticklabel_format(
        axis="both", 
        style="sci", 
        scilimits=(-3,3),
        useMathText=True
    )
    ax_img.tick_params(direction='in')
    ax_img.set_yticklabels([])

    cbar=fig.colorbar(im, label=clabel, cax=ax_clb)
    ax_clb.tick_params(direction='in')
    lim = ax_img.set_xlim(ext[0],ext[1])
    lim = ax_img.set_ylim(ext[2],ext[3])
    lim = ax_plot.set_ylim(ext[2],ext[3])

    if suptitle is None:
        plt.suptitle(title)
    else:
        plt.suptitle(suptitle)

    if title is not None:
        save_figure(fig, title)

    return fig, ax_img, ax_plot, ax_clb, ext
