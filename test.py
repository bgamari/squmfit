from __future__ import division
import numpy as np
from squmfit import *

def exponential_model(t, amplitude, rate):
    return amplitude * np.exp(-t * rate)

ExpModel = Model(exponential_model)

# Generate an exponential decay with some Gaussian additive noise
noise_std = 0.1
xs = np.arange(4000)
ys = 4 * np.exp(-xs / 400)
ys += np.random.normal(scale=noise_std, size=xs.shape)

# Run the fit
fit = Fit()
model = ExpModel(amplitude=fit.param('amp'), rate=fit.param('rate'))
fit.add_curve('curve1', model, ys, t=xs, weights=1/noise_std)
res = fit.fit({'amp': 3, 'rate': 1./100})

print 'parameters', res.params
print 'covariance', res.covar
print 'reduced chi-squared', res.curves['curve1'].reduced_chi_sqr

from matplotlib import pyplot as pl
pl.plot(xs, ys)
pl.plot(xs, fit.eval(res.params)['curve1'])
pl.show()
