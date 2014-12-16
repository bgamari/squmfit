from __future__ import division
from model_fit import *

def exponential_model(t, amplitude, rate):
    return amplitude * np.exp(-t * rate)

import numpy as np
ExpModel = Model(exponential_model, 't rate amplitude')

# example
xs = np.arange(4000)
ys = 4 * np.exp(-xs / 400)
fit = Fit()
model = ExpModel(amplitude=fit.param('amp'), rate=fit.param('rate'))
fit.add_curve('curve1', model, ys, t=xs)
res = fit.fit({'amp': 3, 'rate': 1./100})
print res.params
print res.covar
print res.curves['curve1'].reduced_chi_sqr

from matplotlib import pyplot as pl
pl.plot(xs, ys)
pl.plot(xs, fit.eval(res.params)['curve1'])
pl.show()
