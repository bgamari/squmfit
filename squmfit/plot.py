# -*- coding: utf-8 -*-

import numpy as np
from numpy import log10
import matplotlib.pyplot as pl
from matplotlib import gridspec

def plot_fit(x, result, range=None, xscale='linear', errorbars=True, fig=None,
             with_residuals=True, abs_residuals=False, residual_range=None,
             legend_kwargs={}):
    """
    Plot the result of a fit.

    :param x: A key found in `user_args` for each curve in the fit
    :type result: :class:`FitResult`
    :type range: ``(xmin, xmax)``, optional
    :param range: Range of abscissa
    :type xscale: 'linear', 'log', or 'symlog'
    :param xscale: The scale to use for the X axis (see :func:`pl.xscale`).
    :type errorbars: bool
    :param errorbars: Plot errorbars on points.
    :type fig: :class:`pl.Figure`, optional
    :param fig: The figure on which to plot.
    :type with_residuals: bool
    :param with_residuals: Plot residuals alongside fit
    :type abs_residuals: bool
    :param abs_residuals: Whether to plot weighted (relative) or unweighted (absolute residuals).
    :type residual_range: ``(ymin, ymax)``, optional
    :param residual_range: Y range of residual plot
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
            plot_curve_residuals(x, curve, xscale=xscale, axes=ax_residuals, abs_residuals=abs_residuals,
                                 c=pts_artist.lines[0].get_color())
            
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
    :type range: tuple of two floats
    :param range: The limits of the X axis.
    :type npts: int
    :param npts: Number of interpolation points for fit.
    :type xscale: 'linear', 'log', or 'symlog'
    :param xscale: The scale to use for the X axis (see :func:`pl.xscale`).
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

def plot_curve_residuals(x, result, range=None, axes=None, xscale='linear',
                         abs_residuals=False, residual_range=None,
                         **kwargs):
    """
    Plot the residuals of a curve.

    :param x: A key found in the `user_args` for the curve.
    :type result: :class:`CurveResult`
    :type range: ``(xmin, xmax)``, optional
    :param range: The limits of the X axis
    :type xscale: 'linear', 'log', or 'symlog'
    :param xscale: The scale to use for the X axis (see :func:`pl.xscale`).
    :type axes: :class:`pl.Axes`
    :param axes: Axes to plot on.
    :type abs_residuals: bool
    :param abs_residuals: Whether to plot weighted (relative) or unweighted (absolute residuals).
    :type residual_range: ``(ymin, ymax)``, optional
    :param residual_range: Y range of residual plot
    :param kwargs: Keyword arguments to be passed to :func:`Axes.scatter`.
    """
    xs = result.curve.user_args[x]
    if range is None:
        range = (xs[0], xs[-1])

    curve = result.curve
    fit_result = result.fit_result
    axes.set_xscale(xscale)
    axes.scatter(xs, result.residuals, marker='+', **kwargs)
    if residual_range is not None:
        axes.set_yrange(residual_range[0], residual_range[1])
    # TODO: Implement abs_residuals
