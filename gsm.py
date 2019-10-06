'''Gaussian streaming model.'''
import numpy as np
from scipy import integrate, interpolate
import collections
import sys


def peak_background_split(nu):
    a = 0.707
    p = 0.3
    dc = 1.686
    anu2 = a * nu**2
    F1 = (anu2 - 1.0 + (2. * p / (1. + np.power(anu2, p)))) / dc
    F2 = (anu2**2 - 3 * anu2) + \
        (2. * p * (2. * anu2 + 2. * p - 1)) / (1. + np.power(anu2, p))
    F2 = F2 / dc**2

    return F1, F2


class gsm:
    '''Gaussian streaming model.'''

    def __init__(self):
        # CLPT output
        self.clpt = collections.OrderedDict()
        self.clpt['xi'] = None  # real space xi
        self.clpt['v'] = None  # real space v12
        self.clpt['s'] = None  # real space velocity dispersion

        # parameters
        self.pars = collections.OrderedDict()

        self.pars['nu'] = None
        self.pars['f_v'] = None
        self.pars['sFOG'] = None

        # data
        self.data = collections.OrderedDict()
        self.data['r'] = None
        self.data['xi'] = None
        self.data['xi_L'] = None
        self.data['v'] = None
        self.data['v_L'] = None
        self.data['sigma_p'] = None
        self.data['sigma_v'] = None

        # interpolated function
        self.xir = None

    def read_clpt(self, fn_xi='', fn_v='', fn_s=''):
        '''Read CLPT output.'''
        if fn_xi == '' or fn_v == '' or fn_s == '':
            sys.exit('exit: read_clpt')

        self.clpt['xi'] = np.loadtxt(fn_xi)
        self.clpt['v'] = np.loadtxt(fn_v)
        self.clpt['s'] = np.loadtxt(fn_s)

        # not updated w/ RSD parameters
        self.data['r'] = self.clpt['xi'][:, 0]
        self.data['xi_L'] = self.clpt['xi'][:, 1]

    def set_pars(self, nu, f_v, sFOG):
        '''Set RSD parameters.'''
        F1, F2 = peak_background_split(nu)
        fb11b20 = F1
        fb10b21 = F2
        fb11b21 = F1 * F2
        fb12b20 = F1 * F1
        fb10b22 = F2 * F2

        self.pars['nu'] = nu
        self.pars['f_v'] = f_v
        self.pars['sFOG'] = sFOG

        self.data['xi'] = self.clpt['xi'][:, 2] \
            + fb11b20 * self.clpt['xi'][:, 3] \
            + fb10b21 * self.clpt['xi'][:, 4] \
            + fb12b20 * self.clpt['xi'][:, 5] \
            + fb11b21 * self.clpt['xi'][:, 6] \
            + fb10b22 * self.clpt['xi'][:, 7]

        self.xir = interpolate.interp1d(self.data['r'], self.data['xi'],
                                        bounds_error=True)

        self.data['v'] = self.clpt['v'][:, 2] \
            + fb11b20 * self.clpt['v'][:, 3] \
            + fb10b21 * self.clpt['v'][:, 4] \
            + fb12b20 * self.clpt['v'][:, 5] \
            + fb11b21 * self.clpt['v'][:, 6] \
            # + fb10b22 * self.clpt['v'][:, 7]

        self.data['v_L'] = self.clpt['v'][:, 1] * f_v * (1. + fb11b20)

        self.data['sigma_p'] = self.clpt['s'][:, 1] \
            + fb11b20 * self.clpt['s'][:, 2] \
            + fb10b21 * self.clpt['s'][:, 3] \
            + fb12b20 * self.clpt['s'][:, 4]
        self.data['sigma_p'] *= f_v * f_v / (1. + self.data['xi'])

        self.data['sigma_v'] = self.clpt['s'][:, 5] \
            + fb11b20 * self.clpt['s'][:, 6] \
            + fb10b21 * self.clpt['s'][:, 7] \
            + fb12b20 * self.clpt['s'][:, 8]
        self.data['sigma_v'] *= f_v * f_v / (1. + self.data['xi'])

    def c_xi(self, y_spanning=200, dy=0.5, sigma_p_100=27,):
        '''Compute RSD xi_0 and xi_2.'''
