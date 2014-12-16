from __future__ import division
import numpy as np
from model_fit import *

def exponential_model(t, amplitude, rate):
    return amplitude * np.exp(-t * rate)

ExpModel = Model(exponential_model, 't rate amplitude')

xs = np.arange(4000)
ys = 4 * np.exp(-xs / 400) + np.random.normal(size=xs.shape)

fit = Fit()
model = ExpModel(amplitude=fit.param('amp'), rate=fit.param('rate'))
fit.add_curve('curve1', model, ys, t=xs)
res = fit.fit({'amp': 3, 'rate': 1./100})
print 'parameters', res.params
print 'covariance', res.covar
print 'reduced chi-squared', res.curves['curve1'].reduced_chi_sqr

from matplotlib import pyplot as pl
pl.plot(xs, ys)
pl.plot(xs, fit.eval(res.params)['curve1'])
pl.show()
