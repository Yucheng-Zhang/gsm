'''Gaussian streaming model.'''
import numpy as np
from scipy import integrate, interpolate
import collections
import sys
from . import glq


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

        # some data
        self.data = collections.OrderedDict()
        self.data['r'] = None
        self.sigma_shift = None

        # mu samples and weights
        self.mu_arr = None
        self.mu_w = None

        # real-space interpolated function
        self.xi = None
        self.xi_L = None
        self.v = None
        self.v_L = None
        self.sigma_p = None
        self.sigma_v = None

    def read_clpt(self, fn_xi='', fn_v='', fn_s=''):
        '''Read CLPT output.'''
        if fn_xi == '' or fn_v == '' or fn_s == '':
            sys.exit('exit: read_clpt')

        self.clpt['xi'] = np.loadtxt(fn_xi)
        self.clpt['v'] = np.loadtxt(fn_v)
        self.clpt['s'] = np.loadtxt(fn_s)

        # not updated w/ RSD parameters
        self.data['r'] = self.clpt['xi'][:, 0]
        # self.xi_L = interpolate.interp1d(self.data['r'], self.clpt['xi'][:, 1],
        #                                  kind='linear', bounds_error=True)

    def set_mu_sample(self, kind=None):
        '''Set mu samples and weights.'''
        if kind == 'absave':
            self.mu_arr = np.array([0.01*i-0.005 for i in range(1, 101, 1)])
            self.mu_w = np.full(100, 0.01) * 2
        elif kind == 'ave':
            self.mu_arr = np.array([0.01*i-0.005 for i in range(-99, 101, 1)])
            self.mu_w = np.full(200, 0.01)
        else:
            self.mu_arr = glq.gl_x
            self.mu_w = glq.gl_w

    def set_pars(self, nu=None, f_v=None, sFOG=None, sigma_p_100=27):
        '''Set RSD parameters.'''
        if nu == None or f_v == None or sFOG == None:
            sys.exit('exit: set_pars')

        F1, F2 = peak_background_split(nu)
        fb11b20 = F1
        fb10b21 = F2
        fb11b21 = F1 * F2
        fb12b20 = F1 * F1
        fb10b22 = F2 * F2

        print(fb11b20, fb10b21, fb11b21, fb12b20, fb10b22)

        self.pars['nu'] = nu
        self.pars['f_v'] = f_v
        self.pars['sFOG'] = sFOG

        xi = self.clpt['xi'][:, 2] \
            + fb11b20 * self.clpt['xi'][:, 3] \
            + fb10b21 * self.clpt['xi'][:, 4] \
            + fb12b20 * self.clpt['xi'][:, 5] \
            + fb11b21 * self.clpt['xi'][:, 6] \
            + fb10b22 * self.clpt['xi'][:, 7]

        v = self.clpt['v'][:, 2] \
            + fb11b20 * self.clpt['v'][:, 3] \
            + fb10b21 * self.clpt['v'][:, 4] \
            + fb12b20 * self.clpt['v'][:, 5] \
            + fb11b21 * self.clpt['v'][:, 6] \
            # + fb10b22 * self.clpt['v'][:, 7]
        v = v * f_v / (1. + xi)

        # v_L = self.clpt['v'][:, 1] * f_v * (1. + fb11b20)

        sigma_p = self.clpt['s'][:, 1] \
            + fb11b20 * self.clpt['s'][:, 2] \
            + fb10b21 * self.clpt['s'][:, 3] \
            + fb12b20 * self.clpt['s'][:, 4]
        sigma_p = sigma_p * f_v * f_v / (1. + xi)

        sigma_v = self.clpt['s'][:, 5] \
            + fb11b20 * self.clpt['s'][:, 6] \
            + fb10b21 * self.clpt['s'][:, 7] \
            + fb12b20 * self.clpt['s'][:, 8]
        sigma_v = sigma_v * f_v * f_v / (1. + xi)

        # interpolated functions
        self.xi = interpolate.interp1d(self.data['r'], xi, bounds_error=False,
                                       kind='linear', fill_value=(xi[0], xi[-1]))
        self.v = interpolate.interp1d(self.data['r'], v, bounds_error=False,
                                      kind='linear', fill_value=(v[0], v[-1]))
        # self.v_L = interpolate.interp1d(self.data['r'], v_L, bounds_error=False,
        #                                 kind='linear', fill_value=(v_L[0], v_L[-1]))
        self.sigma_p = interpolate.interp1d(self.data['r'], sigma_p, bounds_error=False,
                                            kind='linear', fill_value=(sigma_p[0], sigma_p[-1]))
        self.sigma_v = interpolate.interp1d(self.data['r'], sigma_v, bounds_error=False,
                                            kind='linear', fill_value=(sigma_v[0], sigma_v[-1]))

        self.sigma_shift = self.sigma_p(100.) - sigma_p_100

    def _sigma2(self, r, mu):
        '''sigma_12^2(r,mu).'''
        mu2 = mu**2
        res = mu2 * self.sigma_p(r)
        res += self.sigma_v(r) * (1. - mu2) * 0.5
        res -= self.sigma_shift
        res += self.pars['sFOG']**2
        return res

    def _kernel(self, y, mu, r_pi, r_sigma):
        '''integrand'''
        r = np.sqrt(r_sigma**2 + y**2)  # real-space separation distance
        if r < 3.:
            return 0.
        sigma2 = self._sigma2(r, mu)
        if sigma2 < 0.:
            return 0.
        exp_idx = -0.5 * (r_pi - y - mu * self.v(r))**2 / sigma2

        return (1. + self.xi(r)) * np.exp(exp_idx) / np.sqrt(2 * np.pi * sigma2)

    def _kernel_arr(self, y, mu, r_pi, r_sigma):
        '''integrand for numpy array of y'''
        r = np.sqrt(r_sigma**2 + y**2)  # real-space separation distance
        sigma2 = self._sigma2(r, mu)
        res = np.zeros_like(y)
        idx = (r < 3.) | (sigma2 < 0.)

        exp_idx = -0.5 * (r_pi - y - mu * self.v(r))**2 / sigma2
        res = (1. + self.xi(r)) * np.exp(exp_idx) / \
            np.sqrt(2 * np.pi * sigma2)
        res[idx] = 0.

        return res

    def c_xi(self, s_arr, y_spanning=200, dy=0.5, int_m=None):
        '''Compute RSD xi_0 and xi_2.'''
        xi2d = np.zeros((s_arr.shape[0], self.mu_arr.shape[0]))

        if int_m == 'scipy':
            for i, s in enumerate(s_arr):
                for j, mu in enumerate(self.mu_arr):
                    r_sigma = s * np.sqrt(1. - mu**2)  # transverse
                    r_pi = s * mu  # los
                    xi2d[i, j] = integrate.quad(self._kernel, r_pi-y_spanning, r_pi+y_spanning,
                                                args=(mu, r_pi, r_sigma,), epsabs=0., epsrel=1e-2)[0] - 1
        else:
            for i, s in enumerate(s_arr):
                for j, mu in enumerate(self.mu_arr):
                    r_sigma = s * np.sqrt(1. - mu**2)  # transverse
                    r_pi = s * mu  # los
                    ys = np.arange(r_pi-y_spanning, r_pi+y_spanning, dy)
                    res = self._kernel_arr(ys, mu, r_pi, r_sigma)
                    xi2d[i, j] = np.sum(0.5 * dy * (res[1:]+res[:-1])) - 1

        xi2d_w = xi2d * self.mu_w
        # monopole
        fac = 1. / 2.
        xi_0 = fac * np.sum(xi2d_w, axis=1)

        # quadrupole
        L_mu = (3. * np.power(self.mu_arr, 2) - 1.) / 2.
        fac = 5. / 2.
        xi_2 = fac * np.sum(xi2d_w * L_mu, axis=1)

        return xi_0, xi_2
