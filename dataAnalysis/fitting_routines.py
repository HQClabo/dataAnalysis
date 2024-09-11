# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:25:41 2023

@author: fopplige
"""


import numpy as np
from scipy import optimize as opt
import lmfit
from resonator_tools import circuit
from matplotlib import pyplot as plt
import dataAnalysis.plotting_functions as myplt

class ResonatorFitNotch:
    """
    Class for fitting a resonator reflection to a model with a Lorentzian resonance and a linear background.
    Add the data using port.add_data, port.add_froms2p or port.add_fromtxt before fitting.
    The fitting function S21_resonator_notch can be redefine by the user.
    """

    def __init__(self):
        self.port = circuit.notch_port()
        return
    
    def S21_resonator_notch(self, fdrive, f0, kappa_int, kappa_ext, a, alpha, tau, phi0):
        delta_r = fdrive - f0
        S21 = (delta_r - 1j/2*(kappa_int + kappa_ext*(1-np.exp(1j*phi0)))) / (delta_r - 1j/2*(kappa_ext + kappa_int))
        environment = a * np.exp(1j*(alpha - tau*2*np.pi*fdrive))
        return S21 * environment

    def fit_resonator(self, fcenter=None, fspan=None, guesses: dict = {}, do_plot=False, plot_title=''):
        """
        Fit a resonator reflection data that has previously been added to the port
        to a model with a Lorentzian resonance and a linear background.

        Parameters
        ----------
        fcenter : float, optional
            Center frequency of the resonance. If None, the function will try to guess
            the center frequency from the data.
        fspan : float, optional
            Frequency span around the center frequency to fit to. If None, the entire frequency range will be used.
        guesses : dict, optional
            Initial guesses for the fit parameters. If None, the function will try to guess
            the parameters from the data.
        do_plot : bool, optional
            Whether to plot the fit results.

        Returns
        -------
        result : lmfit.model.ModelResult
            The result of the fit, including the optimal values for the fit parameters.
        """
        
        if fspan:
            self.port.cut_data(fcenter-fspan/2, fcenter+fspan/2)

        fdrive = self.port.f_data
        zdata = self.port.z_data_raw
        delay, a, alpha, fr, Ql, A2, frcal = self.port.do_calibration(fdrive, zdata)

        initial_guesses = {'f0': fr,
                           'kappa_int': fr/Ql/2,
                           'kappa_ext': fr/Ql/2,
                           'a': a,
                           'alpha': alpha,
                           'tau': delay,
                           'phi0': 0}
        for guess in guesses.keys():
            initial_guesses[guess] = guesses[guess]

        params=lmfit.Parameters()
        params.add('f0', value=initial_guesses['f0'], vary=True)
        params.add('kappa_int', value=initial_guesses['kappa_int'], vary=True)
        params.add('kappa_ext', value=initial_guesses['kappa_ext'], vary=True)
        params.add('a', value=initial_guesses['a'], vary=True)
        params.add('alpha', value=initial_guesses['alpha'], vary=True)
        params.add('tau', value=initial_guesses['tau'], vary=True)
        params.add('phi0', value=initial_guesses['phi0'], vary=True)

        model = lmfit.Model(self.S21_resonator_notch, independent_vars=['fdrive'])
        result = model.fit(zdata, params, fdrive=fdrive)

        if do_plot:
            fig, axes = plt.subplots(1,3,width_ratios=[1,1,1],gridspec_kw=dict(wspace=0.4))
            fig.set_size_inches(18/2.54, 5/2.54)
            fig.suptitle(plot_title)
            
            axes[0].plot(fdrive/1e9, 20*np.log10(abs(zdata)), marker='.', ms=2, ls='')
            axes[0].plot(fdrive/1e9, 20*np.log10(abs(result.best_fit)))
            # axes[0].plot(fdrive, 20*np.log(abs(result.eval(params))))
            myplt.format_plot(axes[0],xlabel='f (GHz)',ylabel='|S21| (dB)')
            axes[1].plot(fdrive/1e9, 180/np.pi*np.angle(zdata), marker='.', ms=2, ls='')
            axes[1].plot(fdrive/1e9, 180/np.pi*np.angle(result.best_fit))
            myplt.format_plot(axes[1],xlabel='f (GHz)',ylabel='S21 (°)')
            axes[2].plot(zdata.real, zdata.imag, marker='.', ms=2, ls='')
            axes[2].plot(result.best_fit.real, result.best_fit.imag)
            myplt.format_plot(axes[2],xlabel='Re(S21) (a.u.)',ylabel='Im(S21) (a.u.)')

        return result


class ResonatorFitReflection:
    """
    Class for fitting a resonator reflection to a model with a Lorentzian resonance and a linear background.
    Add the data using port.add_data, port.add_froms2p or port.add_fromtxt before fitting.
    The fitting function S11_resonator_reflection can be redefine by the user.
    """

    def __init__(self):
        self.port = circuit.reflection_port()
        return

    def S11_resonator_reflection(self, fdrive, f0, kappa_int, kappa_ext, a, alpha, tau):
            delta_r = fdrive - f0
            S11 = (delta_r + 1j/2*(kappa_ext - kappa_int)) / (delta_r - 1j/2*(kappa_ext + kappa_int))
            environment = a * np.exp(1j*(alpha - tau*2*np.pi*fdrive))
            return S11 * environment

    def fit_resonator(self, fcenter=None, fspan=None, guesses: dict = {}, do_plot=False, plot_title=''):
        """
        Fit a resonator reflection data that has previously been added to the port
        to a model with a Lorentzian resonance and a linear background.

        Parameters
        ----------
        fcenter : float, optional
            Center frequency of the resonance. If None, the function will try to guess
            the center frequency from the data.
        fspan : float, optional
            Frequency span around the center frequency to fit to. If None, the entire frequency range will be used.
        guesses : dict, optional
            Initial guesses for the fit parameters. If None, the function will try to guess
            the parameters from the data.
        do_plot : bool, optional
            Whether to plot the fit results.

        Returns
        -------
        result : lmfit.model.ModelResult
            The result of the fit, including the optimal values for the fit parameters.
        """
        
        if fspan:
            self.port.cut_data(fcenter-fspan/2, fcenter+fspan/2)

        fdrive = self.port.f_data
        zdata = self.port.z_data_raw
        delay, a, alpha, fr, Ql, A2, frcal = self.port.do_calibration(fdrive, zdata)

        initial_guesses = {'f0': fr,
                           'kappa_int': fr/Ql/2,
                           'kappa_ext': fr/Ql/2,
                           'a': a,
                           'alpha': alpha,
                           'tau': delay}
        for guess in guesses.keys():
            initial_guesses[guess] = guesses[guess]

        params=lmfit.Parameters()
        params.add('f0', value=initial_guesses['f0'], vary=True)
        params.add('kappa_int', value=initial_guesses['kappa_int'], vary=True)
        params.add('kappa_ext', value=initial_guesses['kappa_ext'], vary=True)
        params.add('a', value=initial_guesses['a'], vary=True)
        params.add('alpha', value=initial_guesses['alpha'], vary=True)
        params.add('tau', value=initial_guesses['tau'], vary=True)

        model = lmfit.Model(self.S11_resonator_reflection, independent_vars=['fdrive'])
        result = model.fit(zdata, params, fdrive=fdrive)

        if do_plot:
            fig, axes = plt.subplots(1,3,width_ratios=[1,1,1],gridspec_kw=dict(wspace=0.4))
            fig.set_size_inches(18/2.54, 5/2.54)
            fig.suptitle(plot_title)
            
            axes[0].plot(fdrive/1e9, 20*np.log10(abs(zdata)), marker='.', ms=2, ls='')
            axes[0].plot(fdrive/1e9, 20*np.log10(abs(result.best_fit)))
            # axes[0].plot(fdrive, 20*np.log(abs(result.eval(params))))
            myplt.format_plot(axes[0],xlabel='f (GHz)',ylabel='|S21| (dB)')
            axes[1].plot(fdrive/1e9, 180/np.pi*np.angle(zdata), marker='.', ms=2, ls='')
            axes[1].plot(fdrive/1e9, 180/np.pi*np.angle(result.best_fit))
            myplt.format_plot(axes[1],xlabel='f (GHz)',ylabel='S21 (°)')
            axes[2].plot(zdata.real, zdata.imag, marker='.', ms=2, ls='')
            axes[2].plot(result.best_fit.real, result.best_fit.imag)
            myplt.format_plot(axes[2],xlabel='Re(S21) (a.u.)',ylabel='Im(S21) (a.u.)')

        return result


# class to use 
class CombinedFitModel:
    def __init__(self,x1,y1,x2,y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def fit_func1(self):
        print('''
fit_func1 is not defined by the user yet.')
Please define your function before creating the object:
def custom_function(self):
    return 1
CombinedFitModel.fit_func1 = custom_function(self)
object = CombinedFitModel(x,y1,y2)
''')

    def fit_func2(self):
        print('''
fit_func1 is not defined by the user yet.')
Please define your function before creating the object:
def custom_function(self):
    return 1
CombinedFitModel.fit_func2 = custom_function(self)
object = CombinedFitModel(x,y1,y2)
''')

    def do_combo_fit(self, p0, **kwargs):
        combined_data = np.append(self.y1,self.y2)
        combined_x_values = np.append(self.x1, self.x2)
        coeff, coeff_cov = opt.curve_fit(self.fit_func_combo, combined_x_values, combined_data, p0=p0, **kwargs)
        return coeff, coeff_cov
    
    
    def fit_func_combo(self, combined_x_values, *args):
    # single data reference passed in, extract separate data
        extract1 = combined_x_values[:len(self.x1)] # first data
        extract2 = combined_x_values[len(self.x1):] # second data
    
        fit1 = self.fit_func1(extract1, *args)
        fit2 = self.fit_func2(extract2, *args)
    
        return np.append(fit1, fit2)
    
    


