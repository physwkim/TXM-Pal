import numpy as np
from lmfit.models import GaussianModel, LinearModel, PolynomialModel

def fitPeak(xdata, ydata, algorithm='Polynomial'):
    """Do fit and return the center of the peak."""
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    if algorithm == 'Polynomial':
        pmodel = PolynomialModel(2)
        pars = pmodel.guess(ydata, x=xdata)
        result = pmodel.fit(ydata, pars, x=xdata).best_values
        cen = -1 * result['c1'] / (2 * result['c2'])

    elif algorithm == 'Gaussian':
        gmodel = GaussianModel()
        pars = gmodel.guess(ydata, x=xdata)
        lmodel = LinearModel(prefix='constant_')
        lmodel.set_param_hint('slope', vary=False)
        pars+=lmodel.make_params(intercept=np.min(ydata), slope=0)

        mod = gmodel + lmodel
        result = mod.fit(ydata, pars, x=xdata).best_values
        cen = result['center']

    return cen