# -*- coding: utf-8 -*-

def show_fit_result(result, min_corr=0.5):
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
            accum += 'Correlations less than %f have been omitted.\n\n' % min_corr
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
