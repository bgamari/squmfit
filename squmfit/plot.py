# -*- coding: utf-8 -*-

import numpy as np
from numpy import log10
import matplotlib.pyplot as pl
from matplotlib import gridspec

def plot_fit(x, result, range=None, xscale='linear', errorbars=True, fig=None, with_residuals=True,
             legend_kwargs={}):
    """
    Plot the result of a fit.

    :param x: A key found in `user_args` for each curve in the fit
    :type result: :class:`FitResult`
    :param legend_kwargs: Keyword arguments passed to :func:`pl.legend`.
    """

    if fig is None:
        fig = pl.figure()
        
    if range is None:
        xmin = min(c.curve.user_args[x].min() for c in result.curves.values())
        xmax = max(c.curve.user_args[x].max() for c in result.curves.values())
        range = (xmin, xmax)
        
    gs = gridspec.GridSpec(2,2, width_ratios=[3,1], height_ratios=[3,1])
    if with_residuals:
        ax_fits = pl.subplot(gs[0, 0])
        ax_residuals = pl.subplot(gs[1, 0])

        ax_residuals.axhline(0, c='k')
        ax_residuals.set_xlim(range[0], range[1])
    else:
        ax_fits = pl.subplot(gs[0:1, 0])
        ax_residuals = None
        
    ax_fits.set_xlim(range[0], range[1])

    for curve in result.curves.values():
        pts_artist, fit_artist = plot_curve(x, curve, xscale=xscale, axes=ax_fits, errorbars=errorbars)
        if ax_residuals is not None:
            plot_curve_residuals(x, curve, xscale=xscale, axes=ax_residuals,
                                 c=pts_artist.lines[0].get_color( ))
            
    ax_legend = gs[0:1, 1]
    ax_fits.legend(loc='upper left',
                   bbox_to_anchor=ax_legend.get_position(fig),
                   bbox_transform=fig.transFigure,
                   mode='expand',
                   frameon=False,
                   **legend_kwargs)
            
    pl.sca(ax_fits)
    return (ax_fits, ax_residuals)
        
def plot_curve(x, result, range=None, axes=None, npts=300,
               xscale='linear', errorbars=True, label=None, **kwargs):
    """
    Plot the result of a fit against a curve.

    :param x: A key found in the `user_args` for the curve.
    :type result: :class:`CurveResult`
    :param range: The limits of the X axis.
    :param label: The label for the curve. Defaults to the curve's name.
    """
    curve = result.curve
    fit_result = result.fit_result
    xs = result.curve.user_args[x]
    if range is None:
        range = (xs[0], xs[-1])
        
    if label is None:
        label = curve.name

    yerr = 1. / curve.weights if errorbars and curve.weights is not None else None
    pts_artist = axes.errorbar(xs, curve.data, yerr=yerr, ls='', **kwargs)

    axes.set_xscale(xscale)
    if xscale in ['log', 'symlog']:
        interp_x = np.logspace(log10(range[0]), log10(range[1]), npts)
    else:
        interp_x = np.linspace(range[0], range[1], npts)

    user_args = {x: interp_x}
    fit_artist = axes.plot(interp_x, result.eval(**user_args), label=label,
                           c=pts_artist.lines[0].get_color())
    return (pts_artist, fit_artist)

def plot_curve_residuals(x, result, range=None, axes=None, npts=300, xscale='linear', **kwargs):
    xs = result.curve.user_args[x]
    if range is None:
        range = (xs[0], xs[-1])

    curve = result.curve
    fit_result = result.fit_result
    axes.set_xscale(xscale)
    axes.scatter(xs, result.residuals, marker='+', **kwargs)
