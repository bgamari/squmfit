# -*- coding: utf-8 -*-

def markdown_fit_result(result, min_corr=0.5):
    """
    Pretty-print a fit result as a Markdown-formatted string

    :type result: :class:`FitResult`
    :type min_corr: float
    :param min_corr: Don't show correlations smaller than this threshold
    :rtype: str
    """
    accum = ""
    accum += '# Parameters\n\n'
    for name,value in sorted(result.params.items()):
        if result.covar is not None:
            accum += '  * %-30s    %8f ± %8f\n' % (name, value, result.stderr[name])
        else:
            accum += '  * %-30s    %f\n' % (name, value)
    accum += '\n\n'

    accum += '# Parameter Correlations\n\n'
    if result.covar is None:
        accum += 'The covariance could not be calculated.\n'
    else:
        if min_corr > 0:
            accum += 'Correlations less than %1.1f have been omitted.\n\n' % min_corr
        correls = {(param1,param2): result.correl[param1][param2]
                   for param1 in result.params.keys()
                   for param2 in result.params.keys()
                   if param1 < param2}
        for (p1,p2), c in sorted(correls.items(), key=lambda ((a,b),c): c, reverse=True):
            if abs(c) >= min_corr:
                accum += '  * %-15s / %-15s       %1.2f\n' % (p1, p2, c)

    accum += '\n\n'

    accum += '# Curves\n\n'
    for name, curve in result.curves.items():
        accum += '  * %-15s\n' % name
        accum += '    * reduced χ² = %1.3g\n' % curve.reduced_chi_sqr

    accum += '\n'
    return accum

def html_fit_result(result, min_corr=0.5):
    """
    Pretty-print a fit result as an HTML-formatted string

    :type result: :class:`FitResult`
    :type min_corr: float
    :param min_corr: Don't show correlations smaller than this threshold
    :rtype: str
    """
    accum = ""
    accum += '<h2>Parameters</h2>\n'
    accum += '<table>\n'
    accum += '  <thead><tr><th>Parameter</th><th>Value</th><th>Standard error</th></tr></thead>'
    accum += '  <tbody>'
    for name,value in sorted(result.params.items()):
        if result.covar is not None:
            accum += '    <tr><td>%s</td><td>%8f</td><td>%8f</td>\n' % (name, value, result.stderr[name])
        else:
            accum += '    <tr><td>%s</td><td>%8f</td><td>-</td>\n' % (name, value)
    accum += '  </tbody>\n</table>\n\n'

    accum += '<h2>Parameter Correlations</h2>\n'
    if result.covar is None:
        accum += '<p>The covariance could not be calculated.</p>\n'
    else:
        if min_corr > 0:
            accum += '<p>Correlations less than <em>%1.1f</em> have been omitted.</p>\n\n' % min_corr

        correls = {(param1,param2): result.correl[param1][param2]
                   for param1 in result.params.keys()
                   for param2 in result.params.keys()
                   if param1 < param2}
        accum += '<table>\n'
        accum += '  <thead><tr><th>parameter</th><th>parameter</th><th>correlation</th></thead>\n'
        accum += '  <tbody>\n'
        for (p1,p2), c in sorted(correls.items(), key=lambda ((a,b),c): c, reverse=True):
            if abs(c) >= min_corr:
                hue = 0 if c < 0 else 142
                lightness = 100 - 30 * abs(c)
                bgcolor = 'hsl(%d, 50%%, %d%%)' % (hue, lightness)
                accum += '    <tr><td>%s</td><td>%s</td><td style="background-color: %s;">%1.2f</td></tr>\n' % (p1, p2, bgcolor, c)
        accum += '  </tbody>\n'
        accum += '</table>\n'

    accum += '<h2>Curves</h2>\n'
    accum += '<ul>\n'
    for name, curve in result.curves.items():
        accum += '  <li>%-15s\n' % name
        accum += '    <ul><li>reduced χ² = %1.3g</li></ul>\n' % curve.reduced_chi_sqr
        accum += '  </li>\n'
    accum += '</ul>\n'

    return accum

def ipynb_fit_result(result, min_corr=0.5):
    """
    Produce a :class:`HTML` summary of a :class:`FitResult` suitable
    for rendering by IPython notebook.

    :type result: :class:`FitResult`
    :param result: The result to summarize.
    """
    from IPython.core.display import HTML
    return HTML(html_fit_result(result, min_corr))
