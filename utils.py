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

"""
h = 6.62607015e-34  # Planck's constant in J*s
c = 299792458  # Speed of light in m/s
eV_to_J = 1.60218e-19  # Conversion factor from eV to Joules

def calcFocalLength(energy, zpDiameter, zpOutermostWidth, numZones):
    wl = h * c / (energy * eV_to_J)
    return (zpDiameter * zpOutermostWidth / wl) - (zpOutermostWidth**2 / wl) + ((2*numZones - 1) * wl / 4)
"""

def calcFocalLength(energy, zpDiameter, zpOutermostWidth, numZones):
    """Energy in eV"""
    wl = 1.239842 / energy * 1000
    return zpOutermostWidth * zpDiameter / wl / 1000

def magnification_corr_factors(energies, magnification=6, zpDiameter=300e-6, zpOutermostWidth=30e-9, numZones=2500):
    """ Return the magnification correction factor for a given magnification.
        
        Parameters
        ==========
        a, b : lens equation 1/a + 1/b = 1/f
        magnification : magnification ratio
        zpDiameter : zone plate diameter in meters
        zpOutermostWidth : outermost zone width in meters
        numZones : number of zones in zone plate
    """

    aValues = np.array([calcFocalLength(energy, zpDiameter, zpOutermostWidth, numZones) * (1+1/magnification) for energy in energies])
    bValues = np.multiply(aValues, magnification)
    bRealValues = bValues.copy()
    aDiff = np.diff(aValues)
    for idx, _ in enumerate(bRealValues):
        if idx > 0:
            bRealValues[idx] = bRealValues[idx-1] - aDiff[idx-1]

    magFactors = aValues / bRealValues

    return np.divide(magFactors[0], magFactors)