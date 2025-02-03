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
from copy import deepcopy
import inspect
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
    
    


class NestedFit:
    """
    A class to perform multi-dataset fitting with shared and individual parameters.
    It perfors nested fits on multiple datasets with shared parameters and individual parameters.
    The inner fits are performed with the lmfit.Model class, while the outer fit is performed with the lmfit.Minimizer class.

    """

    def __init__(self, model_func, independent_var_names: list, shared_param_names: list, individual_param_names: list):
        """
        Initializes the NestedFit class with the model function and parameter names.

        Parameters:
        ----------
        model_func : callable
            The model function to be fitted to the data.
        independent_var_names : list
            List of names of the independent variables.
        shared_param_names : list
            List of names of the shared parameters.
        individual_param_names : list
            List of names of the individual parameters.
        """
        self.model_func = model_func
        self.independent_var_names = independent_var_names
        self.shared_param_names = shared_param_names
        self.individual_param_names = individual_param_names
        self.result_shared = None
        self.results_individual = None

    def _generate_params(self, param_names: list, fixed_params: list, guesses: dict, bounds: dict):
        """
        Generates lmfit.Parameters for the model function.

        Parameters:
        ----------
        param_names : list
            List of parameter names to be included.
        fixed_params : list
            List of parameter names that should be fixed.
        guesses : dict
            Dictionary of initial guesses for the parameters.
        bounds : dict
            Dictionary of bounds for the parameters.

        Returns:
        -------
        lmfit.Parameters
            The generated parameters for the model function.
        """
        signature = inspect.signature(self.model_func)
        params = lmfit.Parameters()
        for param in signature.parameters.values():
            if param.name in param_names:
                vary = param.name not in fixed_params
                guess = guesses.get(param.name, 0)
                min, max = bounds.get(param.name, (-np.inf, np.inf))
                params.add(param.name, value=guess, min=min, max=max, vary=vary)
        return params

    def individual_fit(self, params, data, independent_vars: dict, **kwargs):
        """
        Performs an individual fit on a single dataset.

        Parameters:
        ----------
        params : lmfit.Parameters
            The parameters for the fit.
        data : array-like
            The data to be fitted.
        independent_vars : dict
            Dictionary of independent variables.

        Returns:
        -------
        lmfit.ModelResult
            The result of the fit.
        """
        model = lmfit.Model(self.model_func, independent_vars=independent_vars.keys())
        result = model.fit(data, params, **independent_vars, **kwargs)
        return result

    def nested_fit(self, shared_params: lmfit.Parameters, datasets: list, independent_vars: list,
                   fixed_params: list, guesses: list, bounds: list, **kwargs):
        """
        Performs nested fits on multiple datasets with shared parameters.

        Parameters:
        ----------
        shared_params : lmfit.Parameters
            The shared parameters for the fits.
        datasets : list
            List of datasets to be fitted.
        independent_vars : list
            List of dictionaries of independent variables for each dataset. The list should have the same length as datasets,
            and each dictionary should have the same keys as the independent_var_names attribute and contain their respective values.
        fixed_params : list
            List of parameter names that should be fixed.
        guesses : list
            List of dictionaries of initial guesses for the parameters for each dataset.
        bounds : list
            List of dictionaries of bounds for the parameters for each dataset.

        Returns:
        -------
        list
            List of lmfit.ModelResult objects for each dataset.
        """
        results = []
        for i, data in enumerate(datasets):
            params = self._generate_params(self.individual_param_names, fixed_params, guesses[i], bounds[i])
            params.update(deepcopy(shared_params))
            for param in params and shared_params:
                params[param].vary = False
            result = self.individual_fit(params, data, independent_vars[i], **kwargs)
            results.append(result)
        return results

    def residuals_nested_fit(self, shared_params: lmfit.Parameters, datasets: list, independent_vars: list,
                             fixed_params: list, guesses: dict, bounds: dict, **kwargs):
        """
        Performs the nested fits on all datasets and calculates their residuals as a single 1d numpy.array.

        Parameters:
        ----------
        shared_params : lmfit.Parameters
            The shared parameters for the fits.
        datasets : list
            List of datasets to be fitted.
        independent_vars : list
            List of dictionaries of independent variables for each dataset. The list should have the same length as datasets,
            and each dictionary should have the same keys as the independent_var_names attribute and contain their respective values.
        fixed_params : list
            List of parameter names that should be fixed.
        guesses : dict
            Dictionary of initial guesses for the parameters.
        bounds : dict
            Dictionary of bounds for the parameters.

        Returns:
        -------
        np.array
            Array of residuals from the fits.
        """
        results = self.nested_fit(shared_params, datasets, independent_vars, fixed_params, guesses, bounds, **kwargs)
        residuals = np.array([])
        for result in results:
            residuals = np.append(residuals, result.residual.flatten())
        return residuals

    def fit(self, datasets: list, independent_vars: list, fixed_params: list,
                  guesses_shared: dict, bounds_shared: dict, guesses_individual: list, bounds_individual: list,
                  indiv_kws: dict=dict(), global_kws: dict=dict(), **kwargs):
        """
        Performs a global fit across multiple datasets with shared and individual parameters.

        Parameters:
        ----------
        datasets : list
            List of datasets to be fitted.
        independent_vars : list
            List of dictionaries of independent variables for each dataset. The list should have the same length as datasets,
            and each dictionary should have the same keys as the independent_var_names attribute and contain their respective values.
        fixed_params : list
            List of parameter names that should be fixed.
        guesses_shared : dict
            Dictionary of initial guesses for the shared parameters.
        bounds_shared : dict
            Dictionary of bounds for the shared parameters.
        guesses_individual : list
            List of dictionaries of initial guesses for the individual parameters for each dataset.
        bounds_individual : list
            List of dictionaries of bounds for the individual parameters for each dataset.
        indiv_kws : dict, optional
            Additional keyword arguments for individual fits.
        global_kws : dict, optional
            Additional keyword arguments for the global fit.

        Returns:
        -------
        tuple
            The result of the shared fit and a list of results for the individual fits.
        """
        shared_params = self._generate_params(self.shared_param_names, fixed_params, guesses_shared, bounds_shared)
        minimizer = lmfit.Minimizer(self.residuals_nested_fit, shared_params,
                                    fcn_args=(datasets, independent_vars, fixed_params, guesses_individual, bounds_individual),
                                    fcn_kws=indiv_kws, **kwargs)
        result_shared = minimizer.minimize(params=shared_params, **global_kws)
        results_individual = self.nested_fit(shared_params=result_shared.params,
                                             datasets=datasets, independent_vars=independent_vars, fixed_params=fixed_params,
                                             guesses=guesses_individual, bounds=bounds_individual, **indiv_kws)
        self.result_shared = result_shared
        self.results_individual = results_individual
        return result_shared, results_individual

    def print_fit_report(self, label, label_individual=None):
        if (self.result_shared is not None) and (self.results_individual is not None):
            print('')
            print('Fitted with NestedFit - ' + label)
            print('')
            print('######## Global fit ########')
            print('')
            print(lmfit.fit_report(self.result_shared))
            print('')
            print('######## Individual fits ########')
            print('')
            for i, result in enumerate(self.results_individual):
                if label_individual is not None:
                    print('### '+label_individual[i]+' ###')
                else:
                    print('### Fit '+str(i)+' ###')
                print(lmfit.fit_report(result))
                print('')
        else:
            print('No fit results available. Run the fit method first.')
