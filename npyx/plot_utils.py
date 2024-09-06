# Matplotlib utilities

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator

from IPython.core.display import HTML, display

import os
from pathlib import Path
import pickle as pkl

import numpy as np
from math import floor, log10

from cmcrameri import cm as cmcr
cmcr=cmcr.__dict__

from npyx.utils import (
    pprint_dic,
    docstring_decorator,
    npa,
    isnumeric,
    assert_iterable,
    )

# Make matplotlib saved figures text text editable
mpl.rcParams["svg.fonttype"] = 'none'
mpl.rcParams['pdf.fonttype'] = 42 
mpl.rcParams['ps.fonttype'] = 42

# use Arial, damn it
if 'Arial' in [f.name for f in mpl.font_manager.fontManager.ttflist]:
    mpl.rcParams['font.family'] = 'Arial'
else:
    print("Oh no! Arial isn't on your system. We strongly recommend that you install Arial for your aesthetic sanity.")


#######################################################
### mplp: MatPlotLib Prettifier (Make PLots Pretty) ###
#######################################################

default_mplp_params = dict(
            # title default parameters
            title_w='regular',
            title_s=20,

            # axes labels default parameters
            axlab_w='regular',
            axlab_s=18,

            # tick labels default parameters
            ticklab_w='regular',
            ticklab_s=16,
            ticks_direction='out',
            xlabelpad=0,
            ylabelpad=0,

            # ticks default parameters
            xtickrot=0,
            ytickrot=0,
            xtickha='center',
            xtickva='top',
            ytickha='right',
            ytickva='center',

            # spines and layout default parameters
            lw=1,
            hide_top_right=True,
            hide_axis=False,
            tight_layout=False,

            # legend default parameters
            show_legend=False,
            hide_legend=False,
            legend_loc=(1,1),

            # figure saving default parameters
            saveFig=False,
            saveDir = "~/Downloads",
            figname="figure",
            _format="pdf",

            # colorbar default parameters
            colorbar=False,
            cbar_w=0.03,
            cbar_h=0.4,
            clabel=None,
            clabel_w='regular',
            clabel_s=18,
            cticks_s=16,
            cbar_pad=0.01,

            # horizontal and vertical lines default parameters
            hlines = None, # provide any iterable of values to plot horizontal lines along the y axis
            vlines = None, # provide any iterable of values to plot vertical lines along the x axis
            lines_kwargs = {'lw':1.5, 'ls':'--', 'color':'k', 'zorder':-1000}, # add any other mpl.lines.Line2D arguments

            # scalebar parameters
            xscalebar=None,
            yscalebar=None,
            xscalebar_unit="ms",
            yscalebar_unit="\u03BCV",
            scalebarkwargs={'scalepad': 0.025,
                            'fontsize': 14,
                            'lw': 3,
                            'loc': 'right',
                            'offset_x': 0,
                            'offset_y': 0},
)

@docstring_decorator(pprint_dic(default_mplp_params))
def mplp(fig=None,
         ax=None,
         figsize=None,
         axsize=None,

         xlim=None,
         ylim=None,
         xlabel=None,
         ylabel=None,
         xticks=None,
         yticks=None,
         xtickslabels=None,
         ytickslabels=None,
         reset_xticks=None,
         reset_yticks=None,

         xtickrot=None,
         ytickrot=None,
         xtickha=None,
         xtickva=None,
         ytickha=None,
         ytickva=None,
         xlabelpad=None,
         ylabelpad=None,

         axlab_w=None,
         axlab_s=None,
         ticklab_w=None,
         ticklab_s=None,
         ticks_direction=None,

         title=None,
         title_w=None,
         title_s=None,

         lw=None,
         hide_top_right=None,
         hide_axis=None,
         transparent_background=None,
         tight_layout=None,

         hspace=None,
         wspace=None,

         show_legend=None,
         hide_legend=None,
         legend_loc=None,

         saveFig=None,
         saveDir = None,
         figname=None,
         _format="pdf",

         colorbar=None,
         vmin=None, vmax=None,
         cmap=None,
         cticks=None,
         ctickslabels=None,
         clim=None,
         cbar_w=None,
         cbar_h=None,
         clabel=None,
         clabel_w=None,
         clabel_s=None,
         cticks_s=None,
         cbar_pad=None,

         hlines = None,
         vlines = None,
         lines_kwargs = None,
         prettify=True,

         xscalebar=None,
         yscalebar=None,
         xscalebar_unit=None,
         yscalebar_unit=None,
         scalebarkwargs=None,
         ):
    """
    make plots pretty
    matplotlib plotter

    Awesome utility to format matplotlib plots.
    Simply add mplp() at the end of any plotting script, feeding it with your fav parameters!

    IMPORTANT If you set prettify = False, it will only reset the parameters that you provide actively, and leave the rest as is.

    In a breeze,
        - change the weight/size/alignment/rotation of the axis labels, ticks labels, title
        - edit the x, y and colorbar axis ticks and ticks labels
        - hide the splines (edges of your figure)
        - hide all the axis, label etc in one go with hide_axis
        - save figures in any format
        - add or remove a legend
        - add a custom colorbar
        - apply tight_layout to fit your subplots properly (in a way which prevents saved plots from being clipped)

    How it works: it will grab the currently active figure and axis (plt.gcf() and plt.gca()).
    Alternatively, you can pass a matplotlib figure and specific axes as arguments.

    Default Arguments:
        {0}
    """

    global default_mplp_params

    if fig is None:
        fig = plt.gcf() if ax is None else ax.get_figure()
    if ax is None: ax=plt.gca()

    # if prettify is set to True (default),
    # mplp() will change the plot parameters in the background,
    # even if not actively passed
    if prettify:

        # limits default parameters
        if xlim is None: xlim = ax.get_xlim()
        if ylim is None: ylim = ax.get_ylim()

        # title default parameters
        if title is None: title = ax.get_title()
        if title_w is None: title_w = default_mplp_params['title_w']
        if title_s is None: title_s = default_mplp_params['title_s']

        # axes labels default parameters
        if ylabel is None: ylabel = ax.get_ylabel()
        if xlabel is None: xlabel = ax.get_xlabel()
        if axlab_w is None: axlab_w = default_mplp_params['axlab_w']
        if axlab_s is None: axlab_s = default_mplp_params['axlab_s']
        if xlabelpad is None: xlabelpad = default_mplp_params['xlabelpad']
        if ylabelpad is None: ylabelpad = default_mplp_params['ylabelpad']

        # tick labels default parameters
        if ticklab_w is None: ticklab_w = default_mplp_params['ticklab_w']
        if ticklab_s is None: ticklab_s = default_mplp_params['ticklab_s']
        if ticks_direction is None: ticks_direction = default_mplp_params['ticks_direction']

        # ticks default parameters
        if xtickrot is None: xtickrot = default_mplp_params['xtickrot']
        if ytickrot is None: ytickrot = default_mplp_params['ytickrot']
        if xtickha is None: xtickha = default_mplp_params['xtickha']
        if xtickva is None: xtickva = default_mplp_params['xtickva']
        if ytickha is None: ytickha = default_mplp_params['ytickha']
        if ytickva is None: ytickva = default_mplp_params['ytickva']

        # spines and layout default parameters
        if lw is None: lw = default_mplp_params['lw']
        if hide_top_right is None: hide_top_right = default_mplp_params['hide_top_right']
        if hide_axis is None: hide_axis = default_mplp_params['hide_axis']
        if tight_layout is None: tight_layout = default_mplp_params['tight_layout']

        # legend default parameters
        if show_legend is None: show_legend = default_mplp_params['show_legend']
        if hide_legend is None: hide_legend = default_mplp_params['hide_legend']
        if legend_loc is None: legend_loc = default_mplp_params['legend_loc']

        # figure saving default parameters
        if saveFig is None: saveFig = default_mplp_params['saveFig']
        if saveDir is None: saveDir = default_mplp_params['saveDir']
        if figname is None: figname = default_mplp_params['figname']
        if _format is None: _format = default_mplp_params['_format']

        # colorbar default parameters
        if colorbar is None: colorbar = default_mplp_params['colorbar']
        if cbar_w is None: cbar_w = default_mplp_params['cbar_w']
        if cbar_h is None: cbar_h = default_mplp_params['cbar_h']
        if clabel is None: clabel = default_mplp_params['clabel']
        if clabel_w is None: clabel_w = default_mplp_params['clabel_w']
        if clabel_s is None: clabel_s = default_mplp_params['clabel_s']
        if cticks_s is None: cticks_s = default_mplp_params['cticks_s']
        if cbar_pad is None: cbar_pad = default_mplp_params['cbar_pad']

    xscalebar_unit = default_mplp_params['xscalebar_unit']
    yscalebar_unit = default_mplp_params['yscalebar_unit']


    hfont = {'fontname':'Arial'}
    if figsize is not None:
        assert axsize is  None,\
            "You cannot set both the axes and figure size - the axes size is based on the figure size."
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
    if axsize is not None:
        assert figsize is  None,\
            "You cannot set both the axes and figure size - the axes size is based on the figure size."
        set_ax_size(ax, *axsize)

    # Opportunity to easily hide everything
    if hide_axis is not None:
        if hide_axis:
            ax.axis('off')
        else: ax.axis('on')

    # Axis labels
    if ylabel is not None:
        ax.set_ylabel(ylabel,
                      weight=axlab_w,
                      size=axlab_s,
                      labelpad=ylabelpad,
                      **hfont)
    if xlabel is not None:
        ax.set_xlabel(xlabel,
                      weight=axlab_w,
                      size=axlab_s,
                      labelpad=xlabelpad,
                      **hfont)

    # Setup x/y limits BEFORE altering the ticks
    # since the limits will alter the ticks
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    # Tick values
    if prettify and xticks is None:
        if reset_xticks:
            ax.xaxis.set_major_locator(AutoLocator())
        xticks = ax.get_xticks()
    if xticks is not None: ax.set_xticks(xticks)

    if prettify and yticks is None:
        if reset_yticks:
            ax.yaxis.set_major_locator(AutoLocator())
        yticks = ax.get_yticks()
    if yticks is not None: ax.set_yticks(yticks)

    # Tick labels
    fig.canvas.draw() # To force setting of ticklabels
    if xtickslabels is None and prettify and any(ax.get_xticklabels()):
        if isnumeric(ax.get_xticklabels()[0].get_text()):
            xtickslabels, x_nflt = get_labels_from_ticks(xticks)
        else:
            xtickslabels = ax.get_xticklabels()
    if ytickslabels is None and prettify and any(ax.get_yticklabels()):
        if isnumeric(ax.get_yticklabels()[0].get_text()):
            ytickslabels, y_nflt = get_labels_from_ticks(yticks)
        else:
            ytickslabels = ax.get_yticklabels()

    if xtickslabels is not None:
        if xticks is not None:
            assert len(xtickslabels)==len(xticks),\
                'WARNING you provided too many/few xtickslabels! Make sure that the default/provided xticks match them.'
        if xtickha is None: xtickha = ax.xaxis.get_ticklabels()[0].get_ha()
        if xtickva is None: xtickva = ax.xaxis.get_ticklabels()[0].get_va()
        ax.set_xticklabels(xtickslabels, fontsize=ticklab_s, fontweight=ticklab_w,
                            color=(0,0,0), **hfont, rotation=xtickrot, ha=xtickha, va=xtickva)
    if ytickslabels is not None:
        if yticks is not None:
            assert len(ytickslabels)==len(yticks),\
                'WARNING you provided too many/few ytickslabels! Make sure that the default/provided yticks match them.'
        if ytickha is None: ytickha = ax.yaxis.get_ticklabels()[0].get_ha()
        if ytickva is None: ytickva = ax.yaxis.get_ticklabels()[0].get_va()
        ax.set_yticklabels(ytickslabels, fontsize=ticklab_s, fontweight=ticklab_w,
                            color=(0,0,0), **hfont, rotation=ytickrot, ha=ytickha, va=ytickva)

    # Reset x/y limits a second time
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    # Title
    if title is not None: ax.set_title(title, size=title_s, weight=title_w)

    # Ticks and spines aspect
    if prettify:
        ax.tick_params(axis='both', bottom=1, left=1, top=0, right=0, width=lw, length=4, direction=ticks_direction)
    elif lw is not None or ticks_direction is not None:
        ax.tick_params(axis='both', width=lw, direction=ticks_direction)

    if hide_top_right is not None:
        spine_keys = list(ax.spines.keys())
        hide_spine_keys = ['polar'] if 'polar' in spine_keys else ['top', 'right']
        lw_spine_keys = ['polar'] if 'polar' in spine_keys else ['left', 'bottom', 'top', 'right']
        if hide_top_right and 'top' in hide_spine_keys: [ax.spines[sp].set_visible(False) for sp in hide_spine_keys]
        else: [ax.spines[sp].set_visible(True) for sp in hide_spine_keys]
        for sp in lw_spine_keys:
            ax.spines[sp].set_lw(lw)

    # remove background
    if transparent_background is not None:
        if transparent_background:
            ax.patch.set_alpha(0)
        else:
            ax.patch.set_alpha(1)

    # Optionally plot horizontal and vertical dashed lines
    if lines_kwargs is None: lines_kwargs = {}
    l_kwargs = default_mplp_params['lines_kwargs']
    l_kwargs.update(lines_kwargs) # prevalence of passed arguments

    if hlines is not None:
        assert hasattr(hlines, '__iter__'), 'hlines must be an iterable!'
        for hline in hlines:
            ax.axhline(y=hline, **l_kwargs)
    if vlines is not None:
        assert hasattr(vlines, '__iter__'), 'vlines must be an iterable!'
        for vline in vlines:
            ax.axvline(x=vline, **l_kwargs)

    # Aligning and spacing axes and labels
    if tight_layout: fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if hspace is not None: fig.subplots_adjust(hspace=hspace)
    if wspace is not None: fig.subplots_adjust(wspace=wspace)
    if prettify:
        axis_to_align = [AX for AX in fig.axes if 'AxesSubplot' in AX.__repr__()]
        fig.align_ylabels(axis_to_align)
        fig.align_xlabels(axis_to_align)

    assert not (show_legend and hide_legend),\
        "You instructed to both show and hide the legend...?"
    if legend_loc is not None:
        assert len(legend_loc) in {
            2,
            4,
        }, "legend_loc must comply to the bbox_to_anchor format ( (x,y) or (x,y,width,height))."
    if show_legend: ax.legend(bbox_to_anchor=legend_loc, loc='lower left',
                               prop={'family':'Arial'})
    elif hide_legend: ax.legend([],[], frameon=False)

    if colorbar:
        assert vmin is not None and vmax is not None and cmap is not None,\
            "You must provide vmin, vmax and cmap to show a colorbar."
        fig = add_colorbar(fig, ax, None, vmin, vmax,
                 cbar_w, cbar_h, cticks, clabel, clabel_w, clabel_s, cticks_s, ctickslabels, cmap, pad=cbar_pad,
                 clim=clim) 

    if prettify:
        fig.patch.set_facecolor('white')


    if xscalebar is not None or yscalebar is not None:
        plot_scalebar(ax, xscalebar, yscalebar,
                      xscalebar_unit, yscalebar_unit,
                      **scalebarkwargs)

    if saveFig:
        if figname is None and title is not None:
            figname = title
        save_mpl_fig(fig, figname, saveDir, _format, dpi=500)

    return fig, ax

def save_mpl_fig(fig, figname, saveDir, _format, dpi=500):

    # Fix matplotlib resolution and make text editable
    dpi_orig = mpl.rcParams['figure.dpi']
    fonttype1 = mpl.rcParams['pdf.fonttype']
    fonttype2 = mpl.rcParams['ps.fonttype']

    mpl.rcParams['figure.dpi']=dpi
    mpl.rcParams['pdf.fonttype']=42
    mpl.rcParams['ps.fonttype']=42

    if saveDir is None: saveDir = '~/Downloads'
    saveDir=Path(saveDir).expanduser()
    if not saveDir.exists():
        assert saveDir.parent.exists(), f'WARNING can only create a path of a single directory level, {saveDir.parent} must exist already!'
        saveDir.mkdir()
    p=saveDir/f"{figname}.{_format}"
    fig.savefig(p, dpi=dpi, bbox_inches='tight')

    # restaure original parameters
    mpl.rcParams['figure.dpi']=dpi_orig
    mpl.rcParams['pdf.fonttype']=fonttype1
    mpl.rcParams['ps.fonttype']=fonttype2
    # platform=sys.platform
    # if platform=='linux':
    #     bashCommand = f'sudo chmod a+rwx {p}'
    #     process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    #     output, error = process.communicate()


def plot_scalebar(ax, xscalebar=None, yscalebar=None,
                  x_unit="ms", y_unit="\u03BCV",
                  scalepad=0.025, fontsize=14, lw=3,
                  loc='right',
                  offset_x = 0, offset_y = 0):
    
    """
    Add x and/or y scalebar at the bottom of a matplotlib axis.
    
    Arguments:
        - ax: matplotlib axis
        - xscalebar: float, length of x scalebar in units of x axis.
                        If None, no x scalebar is added.
        - yscalebar: float, length of y scalebar in units of y axis
                        If None, no y scalebar is added.
        - x_unit: str, unit of x axis
        - y_unit: str, unit of y axis
        - scalepad: float, padding between scalebar and axis,
                    in fraction of axis height (arbitrary, could have been width)
        - fontsize: float, fontsize of scalebar text
        - lw: float, linewidth of scalebar
        - loc: str, location of scalebar, either 'left' or 'right'
        - offset_x: float, additional offset of scalebar in x direction,
                    in fraction of axis width (+ or -)
        - offset_y: float, additional offset of scalebar in y direction,
                    in fraction of axis height (+ or -)
    """
    
    # process arguments
    assert xscalebar is not None or yscalebar is not None,\
        "WARNING you must provide either xscalebar or yscalebar. Don't you want to plot a scalebar?"
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    axw, axh = np.diff(xlim)[0], np.diff(ylim)[0]
    bbox = ax.get_window_extent().transformed(ax.get_figure().dpi_scale_trans.inverted())
    axw_inch, axh_inch = bbox.width, bbox.height

    vpad = scalepad*axh
    hpad = scalepad*axw * (axh_inch/axw_inch)

    offset_x = float(offset_x*axw)
    offset_y = float(offset_y*axh)
    assert loc in ['left', 'right']
    if loc == 'right':
        offset_x = offset_x-0.1
    else:
        offset_x = offset_x+0.1
    offset_y = offset_y + 0.1

    # x scalebar
    if xscalebar is not None:
        xscale_y = [ylim[0], ylim[0]]
        if yscalebar is None:
            text_pos_sign = 1
            xscale_va = "bottom"
        else:
            text_pos_sign = -1
            xscale_va = "top"
        if loc == 'right':
            xscale_x = [xlim[1]-xscalebar, xlim[1]]
        else:
            xscale_x = [xlim[0], xlim[0]+xscalebar]

        # optional offset
        xscale_x = [x+offset_x for x in xscale_x]
        xscale_y = [y+offset_y for y in xscale_y]

        # plot xscalebar
        ax.plot(xscale_x, xscale_y, c='k', lw=lw)

        # plot xscalebar text
        ax.text(xscale_x[0]+np.diff(xscale_x)/2,
                xscale_y[0]+vpad*text_pos_sign,
                f"{xscalebar}{x_unit}",
                ha="center", va=xscale_va, fontsize=fontsize)

    # y scalebar
    if yscalebar is not None:
        yscale_y = [ylim[0], ylim[0]+yscalebar]
        if xscalebar is None:
            if loc == 'right':
                text_pos_sign = -1
                yscale_ha = "right"
            else:
                text_pos_sign = 1
                yscale_ha = "left"
        else:
            if loc == 'right':
                text_pos_sign = 1
                yscale_ha = "left"
            else:
                text_pos_sign = -1
                yscale_ha = "right"
        if loc == 'right':
            yscale_x = [xlim[1], xlim[1]]
        else:
            yscale_x = [xlim[0], xlim[0]]

        # optional offset
        yscale_x = [x+offset_x for x in yscale_x]
        yscale_y = [y+offset_y for y in yscale_y]

        # plot yscalebar
        ax.plot(yscale_x, yscale_y, c='k', lw=lw)
        # plot yscalebar text
        ax.text(yscale_x[0]+hpad*text_pos_sign, yscale_y[0]+np.diff(yscale_y)/2,
                f"{yscalebar}{y_unit}",
                ha=yscale_ha, va="center", fontsize=fontsize)
        

##############################
#### Plot ticks utilities ####
##############################
        

def myround(x, base=5):
    return base * np.round(x/base)

def myceil(x, base=5):
    return base * np.ceil(x/base)

def myfloor(x, base=5):
    return base * np.floor(x/base)

def ceil_power10(x):
    return 10**np.ceil(np.log10(x))

def n_decimals(x):
    return len(str(x).split('.')[1])

def get_bestticks(start, end, step=None, light=False):
    """
    Returns the best ticks for a start and end tick.
    If step is specified, it will be the space between ticks.
    If light is True, the step will be multiplied by 2.
    """
    span = end - start
    if step is None:
        upper10 = ceil_power10(span)
        if span <= upper10/5:
            step = upper10*0.01
        elif span <= upper10/2:
            step = upper10*0.05
        else:
            step = upper10*0.1
    if light: step=2*step
    assert step < span, f'Step {step} is too large for array span {span}!'
    ticks = np.arange(myceil(start, step), myfloor(end, step) + step, step)
    ticks = np.round(ticks, n_decimals(step))
    if step == int(step): ticks = ticks.astype(np.int64)

    return ticks

def get_bestticks_from_array(arr, step=None, light=False):
    """
    Returns the best ticks for a given array of values (i.e. an sparser array of equally spaced values).
    If the array if np.arange(10), the returned array will be np.arange(0,10,1).
    If np.arange(50), the returned array will be np.arange(0,50,5). And so on.
    """
    arr_sort = np.sort(arr)
    bestticks = get_bestticks(arr_sort[0], arr_sort[-1], step, light)

    if arr[0] > arr[-1]:
        bestticks = bestticks[::-1]

    return bestticks

def get_labels_from_ticks(ticks):
    ticks=npa(ticks)
    nflt=0
    for t in ticks:
        t=round(t,4)
        for roundi in range(4):
            if t == round(t, roundi):
                nflt = max(nflt, roundi)
                break
    ticks_labels=ticks.astype(np.int64) if nflt==0 else np.round(ticks.astype(float), nflt)
    jump_n=1 if nflt==0 else 2
    ticks_labels=[str(l)+'0'*(nflt+jump_n-len(str(l).replace('-',''))) for l in ticks_labels]
    return ticks_labels, nflt

def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)

###############################
#### Plot colors utilities ####
###############################

def get_all_mpl_colors():
    mpl_colors=get_mpl_css_colors(sort=True, aslist=False)
    mpl_colors={**mpl_colors, **mpl.colors.BASE_COLORS}
    mpl_colors={**mpl_colors, **mpl.colors.TABLEAU_COLORS}
    return mpl_colors

def get_mpl_css_colors(sort=True, aslist=False):
    colors=mpl.colors.CSS4_COLORS
    if sort:
        by_hsv = sorted((tuple(mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        colors={name:colors[name] for hsv, name in by_hsv}
    if aslist: colors=list(colors.values())
    return colors

def mpl_hex(color):
    'converts a matplotlib string name to its hex representation.'
    mpl_colors=get_all_mpl_colors()
    message='color should be a litteral string recognized by mpl.'
    assert isinstance(color, str), message
    basecolors={'b': 'blue', 'g': 'green', 'r': 'red', 'c': 'cyan', 'm': 'magenta', 'y': 'yellow', 'k': 'black', 'w': 'white'}
    if color in basecolors: color=basecolors[color]
    assert color in mpl_colors.keys(), message
    return mpl_colors[color]

def hex_rgb(color):
    'converts a hex color to its rgb representation.'
    message='color must be a hex string starting with #.'
    assert color[0]=='#', message
    return tuple(int(color[1:][i:i+2], 16)/255 for i in (0, 2, 4))

def to_rgb(color):
    'converts a matplotlib string name or hex string to its rgb representation.'
    message='color must either be a litteral matplotlib string name or a hex string starting with #.'
    assert isinstance(color, str), message
    mpl_colors=get_all_mpl_colors()
    if color in mpl_colors.keys(): color=mpl_hex(color)
    assert color[0]=='#', message
    return hex_rgb(color)

def to_hsv(color):
    if isinstance(color,str):
        color=to_rgb(color)
    return mpl.colors.rgb_to_hsv(color)

def to_hex(color):
    'rgb or matplotlib litteral representation to hex representation'
    return mpl_hex(color) if isinstance(color,str) else rgb_hex(color)

def rgb_hex(color):
    '''converts a (r,g,b) color (either 0-1 or 0-255) to its hex representation.
    for ambiguous pure combinations of 0s and 1s e,g, (0,0,1), (1/1/1) is assumed.'''
    message='color must be an iterable of length 3.'
    assert assert_iterable(color), message
    assert len(color)==3, message
    if all((c <= 1) & (c >= 0) for c in color): color=[int(round(c*255)) for c in color] # in case provided rgb is 0-1
    color=tuple(color)
    return '#%02x%02x%02x' % color

def html_palette(colors, maxwidth=20, as_str=False, show=True):
    'colors must be a list of (r,g,b) values ([0-255 or 0-1]) or hex strings.'
    s=55
    n=min(len(colors),maxwidth)
    col_rows=[colors[i*maxwidth:i*maxwidth+maxwidth] for i in range(len(colors)//maxwidth+1)]
    col_rows=[c for c in col_rows if any(c)]
    h=len(col_rows)
    palette = f'<svg  width="{n * s}" height="{s * h}">'
    for r,colors in enumerate(col_rows):
        for i, c in enumerate(colors):
            if not isinstance(c,str):c=rgb_hex(c)
            palette += (
                f'<rect x="{i * s}" y="{r*s}" width="{s}" height="{s}" style="fill:{c};'
                f'stroke-width:2;stroke:rgb(255,255,255)"/>'
            )
    palette += '</svg>'
    if not as_str: palette = HTML(palette)
    if show and not as_str: display(palette)
    return palette

def get_cmap(cmap_str):
    if cmap_str in list(cmcr.keys()):
        return cmcr[cmap_str]
    else:
        return mpl.cm.get_cmap(cmap_str)

def get_bounded_cmap(cmap_str, vmin, center, vmax, colorseq='linear'):
    assert vmin<=center<=vmax, 'WARNING vmin >center or center>vmax!!'
    cmap = get_cmap(cmap_str)

    vrange = max(vmax - center, center - vmin)
    if vrange==0: vrange=1
    if colorseq=='linear':
        vrange=[-vrange,vrange]
        cmin, cmax = (vmin-vrange[0])/(vrange[1]-vrange[0]), (vmax-vrange[0])/(vrange[1]-vrange[0])
        colors_reindex = np.linspace(cmin, cmax, 256)
    elif colorseq=='nonlinear':
        topratio=(vmax - center)/vrange
        bottomratio=abs(vmin - center)/vrange
        colors_reindex=np.append(np.linspace(0, 0.5, int(256*bottomratio/2)),np.linspace(0.5, 1, int(256*topratio/2)))
    cmap = mpl.colors.ListedColormap(cmap(colors_reindex))

    return cmap

def get_ncolors_cmap(n, cmap_str="tab10", plot=False):
    '''Returns homogeneously distributed n colors from specified colormap.
    Arguments:
        - cmap_str: str, matplotlib or crameri colormap
        - n_ int, n colors
        - plot: bool, whether to display colormap in HTML (works in jupyter notebooks)
    Returns:
        - colors: list of n colors homogeneously tiling cmap_str
    '''
    assert n==int(n)
    n=int(n)
    cmap = get_cmap(cmap_str)
    ids=np.linspace(0,1,n)
    colors = cmap(ids)[:,:-1].tolist() # get rid of alpha
    if plot:
        html_palette(colors, 20, 0, 1)
    return colors

def get_color_families(ncolors, nfamilies, cmapstr=None, gap_between_families=4):
    '''
    Return nfamilies of ncolors colors which are perceptually closer within than between families.

    Within each family, the colors are neighbours on a perceptually sequential colormap.

    Between each family, stand gap_between_families colors.
    Increase this value to make colors within families more similar and colors between families more distinct.

    If you decide to provide a cmap, it NEEDS to be perceptually sequential,
    as this function assumes that colors with close ranks are perceptually close.
    If no cmap is provided, the matplotlib literal colors sorted by HSV are used by default.

    '''
    if cmapstr is None:
        colors_all=get_mpl_css_colors(sort=True, aslist=True)[15:-10]
        colors=npa(colors_all)[np.linspace(0,len(colors_all)-1,(ncolors+gap_between_families)*nfamilies).astype(np.int64)].tolist()
    else:
        colors=get_ncolors_cmap((ncolors+gap_between_families//2)*nfamilies, cmapstr, plot=False)
    highsat_colors=[c for c in colors if to_hsv(c)[1]>0.4]
    seed_ids=np.linspace(0, len(highsat_colors)-ncolors, nfamilies).astype(np.int64)

    return [[highsat_colors[si+i] for i in range(ncolors)] for si in seed_ids]

def format_colors(colors):
    '''
    Turns single color or iterable of colors into an iterable of colors.
    '''
    # If string: single letter or hex, can simply use flatten
    if type(npa([colors]).flatten()[0]) in [str, np.str_]:
        colors=npa([colors]).flatten()
    elif type(colors[0]) in [float, np.float16, np.float32, np.float64]:
        colors=npa([colors,])
    else:
        colors=npa(colors)
    return colors

#############################
#### Colorbar utilities ####
#############################

def add_colorbar(fig, ax, mappable=None, vmin=None, vmax=None,
                 width=0.01, height=0.5, cticks=None,
                 clabel=None, clabel_w='regular', clabel_s=20, cticks_s=16, ctickslabels=None,
                 cmap=None, pad = 0.01, clim=None, cbar_ax=None):
    """
    Add colorbar to figure with a predefined axis.
    
    Makes sure that the size of the predefined axis does not change, but that the figure extends.
    """

    # format vmin and vmax
    if vmin is not None or vmax is not None:
        assert vmin is not None and vmax is not None, "You must provide both vmin and vmax!" 
    if vmin is not None and vmax is not None:
        assert vmin < vmax, "Make sure that vmin < vmax (cannot make a 0-range colorbar...)."
        if cticks is None:
            cticks=get_bestticks(vmin, vmax, light=True)

    # define mappable
    if mappable is None:
        assert vmin is not None or vmax is not None,\
            "If you do not provide a mappable (e.g. ax.collections[0]), you must provide vmin and vmax!"
        assert cmap is not None,\
            "If you do not provide a mappable (e.g. ax.collections[0]), you must provide a colormap (e.g. 'viridis')!"
        norm     = plt.Normalize(vmin, vmax)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])

    # create colorbar axis
    axpos   = ax.get_position()
    if cbar_ax is None:
        cbar_ax = fig.add_axes([axpos.x1+pad*axpos.width, axpos.y0,
                            width*axpos.width, height*axpos.height])

    # add colorbar
    fig.colorbar(mappable, cax=cbar_ax, ax=ax,
             orientation='vertical', label=clabel, use_gridspec=True)


    # format colorbar ticks, labels etc
    if ctickslabels is None:
        ctickslabels = cticks
    else:
        assert len(ctickslabels)==len(cticks),\
            f"ctickslabels should have the same length as cticks ({len(cticks)})!" 
    if clabel is not None:
        cbar_ax.yaxis.label.set_font_properties(mpl.font_manager.FontProperties(family='arial', weight=clabel_w, size=clabel_s))
        cbar_ax.yaxis.label.set_rotation(-90)
        cbar_ax.yaxis.label.set_va('bottom')
        cbar_ax.yaxis.label.set_ha('center')
        cbar_ax.yaxis.labelpad = 5
    cbar_ax.yaxis.set_ticks(cticks)
    cbar_ax.yaxis.set_ticklabels(ctickslabels, ha='left')
    cbar_ax.yaxis.set_tick_params(pad=5, labelsize=cticks_s)
    cbar_ax.set_ylim(clim)

    fig.canvas.draw()
    set_ax_size(ax,*fig.get_size_inches())

    return fig


#############################
#### Plot size utilities ####
#############################

def set_ax_size(ax,w,h):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


############################################
#### Plot loading and display utilities ####
############################################

def mplshow(fig):

    # create a dummy figure and use its
    # manager to display "fig"

    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

def bkshow(bkfig, title=None, save=0, savePath='~/Downloads'):
    import bokeh as bk

    if title is None: title=bkfig.__repr__()
    if save:bk.plotting.output_file(f'{title}.html')
    bk.plotting.show(bkfig)

def hvshow(hvobject, backend='matplotlib', return_mpl=False):
    '''
    Holoview utility which
    - for dynamic display, interaction and data exploration:
        in browser, pops up a holoview object as a bokeh figure
    - for static instanciation, refinement and data exploitation:
        in matplotlib current backend, pops up a holoview object as a matplotlib figure
        and eventually returns it for further tweaking.
    Arguments:
        - hvobject: a Holoviews object e.g. Element, Overlay or Layout.
        - backend: 'bokeh' or 'matplotlib', which backend to use to show figure
        - return_mpl: bool, returns a matplotlib figure

    '''
    import holoviews as hv

    assert backend in ['bokeh', 'matplotlib']
    if backend=='matplotlib' or return_mpl:
        mplfig=hv.render(hvobject, backend='matplotlib')
    if backend=='bokeh': bkshow(hv.render(hvobject, backend='bokeh'))
    elif backend=='matplotlib': mplshow(mplfig)
    if return_mpl: return mplfig


def mpl_pickledump(fig, figname, path):
    path=Path(path)
    assert path.exists(), 'WARNING provided target path does not exist!'
    figname+='.pkl'
    pkl.dump(fig, open(path/figname,'wb'))

def mpl_pickleload(fig_path):
    fig_path=Path(fig_path)
    assert fig_path.exists(), 'WARNING provided figure file path does not exist!'
    assert str(fig_path).endswith(
        '.pkl'
    ), 'WARNING provided figure file path does not end with .pkl!'
    return pkl.load(  open(fig_path,  'rb')  )

def mpl_axi_axpos(nrows, ncols, i):
    '''Converts matplotlib subplot index (as in 232 = 2x3 grid, 2nd subplot)
       into (row,col) axes grid location (in this example 0,1)'''
    ax_ids=np.arange(nrows*ncols).reshape((nrows,ncols))+1
    pos=np.argwhere(ax_ids==i)[0]
    assert any(pos), f'WARNING index {i} is too big given {nrows} rows and {ncols} columns!'
    return pos

##################################
#### Plot animation utilities ####
##################################
    
# https://towardsdatascience.com/how-to-create-animated-graphs-in-python-bb619cc2dec1
##### TO CREATE A SERIES OF PICTURES

def make_views(ax,angles,width, height, elevation=None,
                prefix='tmprot_',**kwargs):
    """
    Makes jpeg pictures of the given 3d ax, with different angles.
    Args:
        ax (3D axis): te ax
        angles (list): the list of angles (in degree) under which to
                       take the picture.
        width,height (float): size, in inches, of the output images.
        prefix (str): prefix for the files created.

    Returns: the list of files created (for later removal)
    """

    files = []
    ax.figure.set_size_inches(width,height)

    for i,angle in enumerate(angles):

        ax.view_init(elev = elevation, azim=angle)
        ax.set_xlim3d([206, 212])
        ax.set_ylim3d([208, 213])
        ax.set_zlim3d([207, 213])
        fname = '%s%03d.png'%(prefix,i)
        ax.figure.savefig(fname)
        files.append(fname)

    return files



##### TO TRANSFORM THE SERIES OF PICTURE INTO AN ANIMATION

def make_movie(files,output, fps=10,bitrate=1800,**kwargs):
    """
    Uses mencoder, produces a .mp4/.ogv/... movie from a list of
    picture files.
    """

    output_name, output_ext = os.path.splitext(output)
    command = { '.mp4' : 'mencoder "mf://%s" -mf fps=%d -o %s.mp4 -ovc lavc\
                         -lavcopts vcodec=msmpeg4v2:vbitrate=%d'
                         %(",".join(files),fps,output_name,bitrate)}

    command['.ogv'] = command['.mp4'] + '; ffmpeg -i %s.mp4 -r %d %s'%(output_name,fps,output)

    print(command[output_ext])
    output_ext = os.path.splitext(output)[1]
    os.system(command[output_ext])



def make_gif(files,output,delay=100, repeat=True,**kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """

    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s'
              %(delay,loop," ".join(files),output))




def make_strip(files,output,**kwargs):
    """
    Uses imageMagick to produce a .jpeg strip from a list of
    picture files.
    """

    os.system(f'montage -tile 1x -geometry +0+0 {" ".join(files)} {output}')



##### MAIN FUNCTION

def rotanimate(ax, width, height, angles, output, **kwargs):
    """
    Produces an animation (.mp4,.ogv,.gif,.jpeg,.png) from a 3D plot on
    a 3D ax

    Args:
        ax (3D axis): the ax containing the plot of interest
        angles (list): the list of angles (in degree) under which to
                       show the plot.
        output : name of the output file. The extension determines the
                 kind of animation used.
        **kwargs:
            - width : in inches
            - heigth: in inches
            - framerate : frames per second
            - delay : delay between frames in milliseconds
            - repeat : True or False (.gif only)
    """

    output_ext = os.path.splitext(output)[1]

    files = make_views(ax,angles, width, height, **kwargs)

    D = { '.mp4' : make_movie,
          '.ogv' : make_movie,
          '.gif': make_gif ,
          '.jpeg': make_strip,
          '.png':make_strip}

    D[output_ext](files,output,**kwargs)

    for f in files:
        os.remove(f)

def make_mpl_animation(ax, Nangles, delay, width=10, height=10, saveDir='~/Downloads', title='movie', frmt='gif', axis=True):
    '''
    ax is the figure axes that you will make rotate along its vertical axis,
    on Nangles angles (default 300),
    separated by delay time units (default 2),
    at a resolution of widthxheight pixels (default 10x10),
    saved in saveDir directory (default Downloads) with the title title (default movie) and format frmt (gif).
    '''
    assert frmt in ['gif', 'mp4', 'ogv']
    if not axis: plt.axis('off') # remove axes for visual appeal
    oldDir=os.getcwd()
    saveDir=os.path.expanduser(saveDir)
    os.chdir(saveDir)
    angles = np.linspace(0,360,Nangles)[:-1] # Take 20 angles between 0 and 360
    ttl = f'{title}.{frmt}'
    rotanimate(ax, width, height, angles,ttl, delay=delay)

    os.chdir(oldDir)