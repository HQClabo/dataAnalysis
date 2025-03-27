# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 13:43:03 2022

@author: fopplige
"""

from scipy.signal import find_peaks
from scipy.stats import linregress
import scipy.constants as const
import numpy as np
import lmfit
from resonator_tools import circuit
import matplotlib.pyplot as plt
import inspect
import dataAnalysis.plotting_functions as myplt

#%% functions for resonator fitting

def S21_resonator_notch(fdrive, fr, kappa, kappa_c, a, alpha, delay, phi0):
        delta_r = fdrive - fr
        S21 = (delta_r - 1j/2*(kappa - kappa_c*(np.exp(1j*phi0)))) / (delta_r - 1j/2*(kappa))
        environment = a * np.exp(1j*(alpha - delay*2*np.pi*fdrive))
        return S21 * environment

def S11_resonator_reflection(fdrive, fr, kappa, kappa_c, a, alpha, delay):
    delta_r = fdrive - fr
    S11 = 2*kappa_c / (kappa + 2*1j*delta_r) - 1
    environment = a * np.exp(1j * (alpha - 2*np.pi*fdrive*delay))
    return S11 * environment

def guess_resonator_params(f, data, fraction=0.1, port_type='notch'):
    # do a Lorentzian fit to the magnitude of complex to get the resonance frequency and linewidth
    model = lmfit.models.LorentzianModel() + lmfit.models.ConstantModel()
    model.set_param_hint('center', value=f.mean(), vary=True)
    sigma = f.ptp()/10
    model.set_param_hint('sigma', value=sigma, vary=True)
    model.set_param_hint('amplitude', value=-abs(data).ptp() * sigma*np.pi, max=0, vary=True)
    model.set_param_hint('c', value=np.median(abs(data)), vary=True)
    result = model.fit(data=abs(data), x=f)
    fr = result.best_values['center']
    kappa = result.best_values['sigma'] * 2
    a = result.best_values['c']

    # linear fit of first and last 10% of the unwrapped phase
    # the phase offset contains the phase of the reflection from the resonator
    if port_type == 'notch':
        phase_offset = 0
    elif port_type == 'reflection':
        phase_offset = np.pi
    # phase_offset = 0
    phase = np.unwrap(np.angle(data))
    first = int(len(f)//(1/fraction))
    last = -int(len(f)//(1/fraction))
    fit1 = linregress(f[:first], phase[:first])
    fit2 = linregress(f[last:], phase[last:])
    delay = -(fit1.slope + fit2.slope) / 2 / (2*np.pi)
    alpha = (phase[0] + 2*np.pi*delay*f[0] - phase_offset + np.pi) % (2*np.pi) - np.pi
    return fr, kappa, a, alpha, delay

def get_frequency_scaling(freq_unit='Hz'):
    if freq_unit == 'GHz':
        return 1e9
    elif freq_unit == 'MHz':
        return 1e6
    elif freq_unit == 'kHz':
        return 1e3
    elif freq_unit == 'Hz':
        return 1
    else:
        raise ValueError(f"Unsupported frequency unit: {freq_unit}. Must be one of 'GHz', 'MHz', 'kHz', 'Hz'.")

def get_single_photon_limit(fitresults, freq_unit='Hz', unit='dBm'):
    '''
    returns the amout of power in units of W necessary
    to maintain one photon on average in the cavity
    unit can be 'dbm' or 'watt'
    '''
    power_scaling = get_frequency_scaling(freq_unit)**2
    fr = fitresults['fr']
    k_c = 2*np.pi*fitresults['kappa_c']
    k_i = 2*np.pi*(fitresults['kappa']-fitresults['kappa_c'])
    power_watts = 1./(4.*k_c/(2.*np.pi*const.hbar*fr*(k_c+k_i)**2)) * power_scaling
    if unit=='dBm':
        return 10*np.log10(power_watts*1000.)
    elif unit=='watt':
        return power_watts
        
def get_photons_in_resonator(power, fitresults, freq_unit='Hz', unit='dBm'):
		'''
		returns the average number of photons
		for a given power (defaul unit is 'dbm')
		unit can be 'dBm' or 'watt'
		'''
		if fitresults!={}:
			if unit=='dBm':
				power = 10**(power/10.) /1000.
			return power / get_single_photon_limit(fitresults, freq_unit=freq_unit, unit='watt')
        
def plot_resonator_fit_lmfit(freq, data, fit_result, freq_unit='Hz', plot_initial_guesses=False):
    alpha = 0.7
    freq = freq * get_frequency_scaling(freq_unit)/1e9
    fig, axes = plt.subplots(1,3,width_ratios=[1,1,1],gridspec_kw=dict(wspace=0.4))
    axes[0].plot(freq, abs(data), marker='.', ms=2, ls='')
    axes[0].plot(freq, abs(fit_result.best_fit),alpha=alpha)
    myplt.format_plot(axes[0],xlabel='f (GHz)',ylabel='|S21|')
    axes[1].plot(freq, np.angle(data, deg=True), marker='.', ms=2, ls='')
    axes[1].plot(freq, np.angle(fit_result.best_fit, deg=True),alpha=alpha)
    myplt.format_plot(axes[1],xlabel='f (GHz)',ylabel='S21 (°)')
    axes[2].plot(data.real, data.imag, marker='.', ms=2, ls='')
    axes[2].plot(fit_result.best_fit.real, fit_result.best_fit.imag,alpha=alpha)
    myplt.format_plot(axes[2],xlabel='Re(S21) (a.u.)',ylabel='Im(S21) (a.u.)')

    if plot_initial_guesses:
        axes[0].plot(freq, abs(fit_result.init_fit),alpha=alpha,ls='--',c='k')
        axes[1].plot(freq, np.angle(fit_result.init_fit, deg=True),alpha=alpha,ls='--',c='k')
        axes[2].plot(fit_result.init_fit.real, fit_result.init_fit.imag,alpha=alpha,ls='--',c='k')
    for ax in axes:
        ax.set_aspect(1/ax.get_data_ratio())
    return fig, axes
        
def fit_frequency_sweep(
        data,
        freq,
        freq_range=None,
        power=-140,
        port_type='notch',
        method='resonator_tools',
        guesses={},
        freq_unit='Hz',
        plot=False,
        plot_initial_guesses=False,
        **kwargs
        ):
    """
    Function to fit resonances of a power sweep.

    Parameters
    ----------
    data : ndarray
        Complex data of the measured signal.
    freq : ndarray
        List of the measured frequencies.
    freq_range : list, optional
        Minimum and maximum frequency that will be considered for the fit.
    power : ndarray, optional
        Input power in dBm. Default is -140 dBm.
    port_type : str, optional
        Type of the resonator port. Choose 'notch' or 'reflection. The default is 'notch'.
    method : str, optional
        Method used for the fitting. Choose 'resonator_tools' or 'lmfit'.
    guesses : dict, optional
        Dictionary containing the initial guesses for the fit parameters for lmfit only. The default is {}.
    freq_unit : str, optional
        Unit in which frequency is provided. This will determine the proper scaling factors. Default is 'Hz'.
    plot : boolean, optional
        Enable option to plot the fit. The default is False.
    plot_initial_guesses : boolean, optional
        Enable option to plot also the initial guesses for the fit. Only active if plot=True. The default is False.
    **kwargs : dict
        Additional keyword arguments for lmfit.Model.fit.

    Returns
    -------
    fitReport : dict
        Dictionary containing the results of the fit as np.arrays for each parameter.

    """
    if method == 'resonator_tools':
        return _fit_frequency_sweep_resonator_tools(data,freq,freq_range,power,port_type,plot)
    elif method == 'lmfit':
        return _fit_frequency_sweep_lmfit(data,freq,freq_range,power,port_type,guesses,freq_unit,plot,plot_initial_guesses,**kwargs)
    else:
        raise ValueError(f"The method '{method}' is not supported. Use 'resonator_tools' or 'lmfit'")

def _fit_frequency_sweep_resonator_tools(
        data,
        freq,
        freq_range,
        power,
        port_type,
        plot
        ):
    fit_report = {}
    # define port type
    if port_type == 'notch':
        port = circuit.notch_port()
    elif port_type == 'reflection':
        port = circuit.reflection_port()
    else:
        print("This port type is not supported. Use 'notch', 'reflection' or 'transmission'")
    # cut and fit data
    port.add_data(freq,data)
    if freq_range:
        port.cut_data(*freq_range)
    # get environment parameters
    delay, a, alpha, fr, Ql, A2, frcal = port.do_calibration(port.f_data, port.z_data_raw, guessdelay=False)
    fit_report['delay'] = delay
    fit_report['a'] = a
    fit_report['alpha'] = alpha
    port.autofit()
    if plot == True:
        port.plotall()
    # add fitting results to the dictionary
    if port_type == 'notch':
        fit_report["Qi"] = port.fitresults["Qi_dia_corr"]
        fit_report["Qi_err"] = port.fitresults["Qi_dia_corr_err"]
        fit_report["Qc"] = port.fitresults["Qc_dia_corr"]
        fit_report["Qc_err"] = port.fitresults["absQc_err"]
    else:
        fit_report["Qi"] = port.fitresults["Qi"]
        fit_report["Qi_err"] = port.fitresults["Qi_err"]
        fit_report["Qc"] = port.fitresults["Qc"]
        fit_report["Qc_err"] = port.fitresults["Qc_err"]
    fit_report["Ql"] = port.fitresults["Ql"]
    fit_report["Ql_err"] = port.fitresults["Ql_err"]
    if port_type == 'notch':
        fit_report["phi0"] = port.fitresults["phi0"]
    fit_report["Nph"] = port.get_photons_in_resonator(power,unit='dBm')
    fit_report["single_photon_W"] = port.get_single_photon_limit(unit='watt')
    fit_report["single_photon_dBm"] = port.get_single_photon_limit(unit='dBm')
    fit_report["fr"] = port.fitresults["fr"]
    fit_report['fitresults'] = port.fitresults
    return fit_report

def _fit_frequency_sweep_lmfit(
        data,
        freq,
        freq_range,
        power,
        port_type,
        user_guesses,
        freq_unit,
        plot,
        plot_initial_guesses,
        **kwargs
        ):
    # Define port type
    if port_type == 'notch':
        model_func = S21_resonator_notch
    elif port_type == 'reflection':
        model_func = S11_resonator_reflection
    else:
        print("This port type is not supported. Supported types are 'notch' or 'reflection'")

    data_to_fit = data
    if freq_range:
        freq_slice = myplt.find_slice(freq, freq_range)
        freq_to_fit = freq[freq_slice]
        data_to_fit = data_to_fit[freq_slice]
    else:
        freq_to_fit = freq
    fr, kappa, a, alpha, delay = guess_resonator_params(freq_to_fit, data_to_fit, port_type=port_type)
    fixed_params = []

    # define the initial guesses for the parameters
    guesses = {}
    guesses['fr'] = fr
    guesses['kappa'] = kappa
    guesses['kappa_c'] = kappa/2
    guesses['a'] = a
    guesses['alpha']= alpha
    guesses['delay'] = delay
    if port_type == 'notch':
        guesses['phi0'] = 0

    # overwrite the initial guesses with the ones provided by the user
    for key, value in user_guesses.items():
        guesses[key] = value

    # create the lmfit.Parameters object and adjust some settings
    params=lmfit.Parameters() # object
    signature = inspect.signature(model_func)
    parameter_names = [param.name for param in signature.parameters.values() if param.name not in ['fdrive']]
    for name in parameter_names:
        params.add(name, value=guesses[name], vary=True)
    for name in fixed_params:
        params[name].vary = False
    params.add('kappa_i', expr='kappa-kappa_c')
    params.add('Ql', expr='fr/(kappa)')
    params.add('Qc', expr='fr/kappa_c')
    params.add('Qi', expr='fr/(kappa-kappa_c)')

    for param_name in ['fr', 'kappa', 'kappa_c', 'a']:
        params[param_name].min = 0
    params['alpha'].min = -np.pi
    params['alpha'].max = np.pi

    # fit the data
    model = lmfit.Model(model_func, independent_vars=['fdrive'])
    result = model.fit(data_to_fit, params, fdrive=freq_to_fit, **kwargs)
    par = result.params
    fit_report = {}
    fit_report["fr"] = par['fr'].value
    for param_name in ['kappa_i', 'kappa_c', 'kappa', 'Qi', 'Qc', 'Ql']: 
        fit_report[param_name] = par[param_name].value
        fit_report[param_name+'_err'] = par[param_name].stderr
    fit_report['a'] = par['a'].value
    fit_report['alpha'] = par['alpha'].value
    fit_report['delay'] = par['delay'].value
    fit_report["Nph"] = get_photons_in_resonator(power, result.best_values ,freq_unit=freq_unit ,unit='dBm')
    fit_report["single_photon_W"] = get_single_photon_limit(result.best_values,freq_unit=freq_unit , unit='watt')
    fit_report["single_photon_dBm"] = get_single_photon_limit(result.best_values,freq_unit=freq_unit , unit='dBm')
    fit_report['fitresults'] = result
    
    if plot:
        plot_resonator_fit_lmfit(freq_to_fit, data_to_fit, result, freq_unit=freq_unit,
                                 plot_initial_guesses=plot_initial_guesses)
    return fit_report

def fit_power_sweep(
        data,
        freq,
        power,
        freq_range=None,
        power_range=None,
        attenuation=80,
        port_type='notch',
        method='resonator_tools',
        guesses={},
        freq_unit='Hz',
        plot=False,
        plot_initial_guesses=False,
        **kwargs
        ):
    """
    Function to fit resonances of a power sweep.

    Parameters
    ----------
    data : ndarray
        Complex data of dimension (n_power,n_freq).
    freq : ndarray
        List of the measured frequencies.
    power : ndarray
        List of the different powers applied.
    freq_range : list, optional
        Minimum and maximum frequency that will be considered for the fit.
    attenuation : number, optional
        Total estimated line attenuation. The default is 80.
    port_type : str, optional
        Type of the resonator port. Choose 'notch' or 'reflection. The default is 'notch'.
    method : str, optional
        Method used for the fitting. Choose 'resonator_tools' or 'lmfit'.
    guesses : dict, optional
        Dictionary containing the initial guesses for the fit parameters for lmfit only. The default is {}.
    freq_unit : str, optional
        Unit in which frequency is provided. This will determine the proper scaling factors. Default is 'Hz'.
    plot : boolean, optional
        Enable option to plot the individual fits. The default is False.
    plot_initial_guesses : boolean, optional
        Enable option to plot also the initial guesses for the fit. Only active if plot=True. The default is False.
    **kwargs : dict
        Additional keyword arguments for lmfit.Model.fit.

    Returns
    -------
    fitReport : dict
        Dictionary containing the results of the fit as np.arrays for each parameter.

    """
    n_powers = len(power)
    fit_report = {
        "fr" : np.array([np.nan]*n_powers),
        "kappa_i" : np.array([np.nan]*n_powers),
        "kappa_i_err" : np.array([np.nan]*n_powers),
        "kappa_c" : np.array([np.nan]*n_powers),
        "kappa_c_err" : np.array([np.nan]*n_powers),
        "kappa" : np.array([np.nan]*n_powers),
        "kappa_err" : np.array([np.nan]*n_powers),
        "Qi" : np.array([np.nan]*n_powers),
        "Qi_err" : np.array([np.nan]*n_powers),
        "Qc" : np.array([np.nan]*n_powers),
        "Qc_err" : np.array([np.nan]*n_powers),
        "Ql" : np.array([np.nan]*n_powers),
        "Ql_err" : np.array([np.nan]*n_powers),
        "a" : np.array([np.nan]*n_powers),
        "alpha" : np.array([np.nan]*n_powers),
        "delay" : np.array([np.nan]*n_powers),
        "Nph" : np.array([np.nan]*n_powers),
        "single_photon_W" : np.array([np.nan]*n_powers),
        "single_photon_dBm" : np.array([np.nan]*n_powers),
        'fitresults': [None]*n_powers,
        }
    if port_type == 'notch':
        fit_report['phi0'] = np.array([np.nan]*n_powers)
    if method == 'resonator_tools':
        return _fit_power_sweep_resonator_tools(data,freq,power,freq_range,power_range,attenuation,
                                                port_type,fit_report,plot)
    elif method == 'lmfit':
        return _fit_power_sweep_lmfit(data,freq,power,freq_range,power_range,attenuation,
                                      port_type,guesses,fit_report,freq_unit,plot,plot_initial_guesses,**kwargs)
    else:
        raise ValueError(f"The method '{method}' is not supported. Use 'resonator_tools' or 'lmfit'")
    
def _fit_power_sweep_resonator_tools(
        data,
        freq,
        power,
        freq_range,
        power_range,
        attenuation,
        port_type,
        fit_report,
        plot
        ):
    for k,pwr in enumerate(power):
        if power_range:
            if (pwr < power_range[0]) or (pwr > power_range[1]):
                continue
        # define port type
        if port_type == 'notch':
            port = circuit.notch_port()
        elif port_type == 'reflection':
            port = circuit.reflection_port()
        else:
            print("This port type is not supported. Use 'notch', 'reflection' or 'transmission'")
        # cut and fit data
        port.add_data(freq,data[k])
        if freq_range:
            port.cut_data(*freq_range)
        # get environment parameters
        delay, a, alpha, fr, Ql, A2, frcal = port.do_calibration(port.f_data, port.z_data_raw, guessdelay=False)
        fit_report['delay'][k] = delay
        fit_report['a'][k] = a
        fit_report['alpha'][k] = alpha
        port.autofit()
        if plot == True:
            print(f'Power = {pwr} dBm')
            port.plotall()
        # add fitting results to the dictionary
        if port_type == 'notch':
            fit_report["Qi"][k] = port.fitresults["Qi_dia_corr"]
            fit_report["Qi_err"][k] = port.fitresults["Qi_dia_corr_err"]
            fit_report["Qc"][k] = port.fitresults["Qc_dia_corr"]
            fit_report["Qc_err"][k] = port.fitresults["absQc_err"]
        else:
            fit_report["Qi"][k] = port.fitresults["Qi"]
            fit_report["Qi_err"][k] = port.fitresults["Qi_err"]
            fit_report["Qc"][k] = port.fitresults["Qc"]
            fit_report["Qc_err"][k] = port.fitresults["Qc_err"]
        fit_report["Ql"][k] = port.fitresults["Ql"]
        fit_report["Ql_err"][k] = port.fitresults["Ql_err"]
        if port_type == 'notch':
            fit_report["phi0"] = port.fitresults["phi0"]
        fit_report["Nph"][k] = port.get_photons_in_resonator(pwr - attenuation,unit='dBm')
        fit_report["single_photon_W"][k] = port.get_single_photon_limit(unit='watt')
        fit_report["single_photon_dBm"][k] = port.get_single_photon_limit(unit='dBm')
        fit_report["fr"][k] = port.fitresults["fr"]
        fit_report['fitresults'][k] = port.fitresults
    return fit_report

def _fit_power_sweep_lmfit(
        data,
        freq,
        power,
        freq_range,
        power_range,
        attenuation,
        port_type,
        user_guesses,
        fit_report,
        freq_unit,
        plot,
        plot_initial_guesses,
        **kwargs
        ):
    for k,pwr in enumerate(power):
        if power_range:
            if (pwr < power_range[0]) or (pwr > power_range[1]):
                continue
        # Define port type
        if port_type == 'notch':
            model_func = S21_resonator_notch
        elif port_type == 'reflection':
            model_func = S11_resonator_reflection
        else:
            print("This port type is not supported. Supported types are 'notch' or 'reflection'")

        data_to_fit = data[k]
        if freq_range:
            freq_slice = myplt.find_slice(freq, freq_range)
            freq_to_fit = freq[freq_slice]
            data_to_fit = data_to_fit[freq_slice]
        else:
            freq_to_fit = freq
        fr, kappa, a, alpha, delay = guess_resonator_params(freq_to_fit, data_to_fit, port_type=port_type)
        fixed_params = []

        # define the initial guesses for the parameters
        guesses = {}
        guesses['fr'] = fr
        guesses['kappa'] = kappa
        guesses['kappa_c'] = kappa/2
        guesses['a'] = a
        guesses['alpha']= alpha
        guesses['delay'] = delay
        if port_type == 'notch':
            guesses['phi0'] = 0

        # overwrite the initial guesses with the ones provided by the user
        for key, value in user_guesses.items():
            guesses[key] = value

        # create the lmfit.Parameters object and adjust some settings
        params=lmfit.Parameters() # object
        signature = inspect.signature(model_func)
        parameter_names = [param.name for param in signature.parameters.values() if param.name not in ['fdrive']]
        for name in parameter_names:
            params.add(name, value=guesses[name], vary=True)
        for name in fixed_params:
            params[name].vary = False
        params.add('kappa_i', expr='kappa-kappa_c')
        params.add('Ql', expr='fr/(kappa)')
        params.add('Qc', expr='fr/kappa_c')
        params.add('Qi', expr='fr/(kappa-kappa_c)')

        for param_name in ['fr', 'kappa', 'kappa_c', 'a']:
            params[param_name].min = 0
        params['alpha'].min = -np.pi
        params['alpha'].max = np.pi

        # fit the data
        model = lmfit.Model(model_func, independent_vars=['fdrive'])
        result = model.fit(data_to_fit, params, fdrive=freq_to_fit, **kwargs)
        par = result.params
        fit_report["fr"][k] = par['fr'].value
        for param_name in ['kappa_i', 'kappa_c', 'kappa', 'Qi', 'Qc', 'Ql']: 
            fit_report[param_name][k] = par[param_name].value
            fit_report[param_name+'_err'][k] = par[param_name].stderr
        fit_report['a'][k] = par['a'].value
        fit_report['alpha'][k] = par['alpha'].value
        fit_report['delay'][k] = par['delay'].value
        fit_report["Nph"][k] = get_photons_in_resonator(pwr - attenuation, result.best_values, freq_unit=freq_unit ,unit='dBm')
        fit_report["single_photon_W"][k] = get_single_photon_limit(result.best_values, freq_unit=freq_unit, unit='watt')
        fit_report["single_photon_dBm"][k] = get_single_photon_limit(result.best_values, freq_unit=freq_unit, unit='dBm')
        fit_report['fitresults'][k] = result
        
        if plot:
            print(f'Power = {pwr} dBm')
            plot_resonator_fit_lmfit(freq_to_fit, data_to_fit, result, freq_unit=freq_unit,
                                     plot_initial_guesses=plot_initial_guesses)
            plt.show()
    return fit_report

def fit_field_sweep(
        data,
        freq,
        field,
        center_freqs,
        span,
        field_range=None,
        power=-140,
        port_type='notch',
        method='resonator_tools',
        guesses={},
        freq_unit='Hz',
        plot=False,
        plot_initial_guesses=False,
        **kwargs
        ):
    """
    Function to fit resonances of a magnetic field sweep.

    Parameters
    ----------
    data : ndarray
        Complex data of dimension (n_power,n_freq).
    freq : ndarray
        List of the measured frequencies.
    field : ndarray
        List of the different magnetic field values applied.
    center_freqs : list
        List of center frequencies for each magnetic field value to determine the fitting window.
    span : float
        Single value for the frequency span of the fitting window.
    power : number, optional
        Applied input power in dBm. The default is -140 dBm.
    port_type : str, optional
        Type of the resonator port. Choose 'notch' or 'reflection. The default is 'notch'.
    method : str, optional
        Method used for the fitting. Choose 'resonator_tools' or 'lmfit'.
    guesses : dict, optional
        Dictionary containing the initial guesses for the fit parameters for lmfit only. The default is {}.
    freq_unit : str, optional
        Unit in which frequency is provided. This will determine the proper scaling factors. Default is 'Hz'.
    plot : boolean, optional
        Enable option to plot the individual fits. The default is False.
    plot_initial_guesses : boolean, optional
        Enable option to plot also the initial guesses for the fit. Only active if plot=True. The default is False.
    **kwargs : dict
        Additional keyword arguments for lmfit.Model.fit.

    Returns
    -------
    fitReport : dict
        Dictionary containing the results of the fit as np.arrays for each parameter.

    """
    n_fields = len(field)
    fit_report = {
        "fr" : np.array([np.nan]*n_fields),
        "kappa_i" : np.array([np.nan]*n_fields),
        "kappa_i_err" : np.array([np.nan]*n_fields),
        "kappa_c" : np.array([np.nan]*n_fields),
        "kappa_c_err" : np.array([np.nan]*n_fields),
        "kappa" : np.array([np.nan]*n_fields),
        "kappa_err" : np.array([np.nan]*n_fields),
        "Qi" : np.array([np.nan]*n_fields),
        "Qi_err" : np.array([np.nan]*n_fields),
        "Qc" : np.array([np.nan]*n_fields),
        "Qc_err" : np.array([np.nan]*n_fields),
        "Ql" : np.array([np.nan]*n_fields),
        "Ql_err" : np.array([np.nan]*n_fields),
        "a" : np.array([np.nan]*n_fields),
        "alpha" : np.array([np.nan]*n_fields),
        "delay" : np.array([np.nan]*n_fields),
        "Nph" : np.array([np.nan]*n_fields),
        "single_photon_W" : np.array([np.nan]*n_fields),
        "single_photon_dBm" : np.array([np.nan]*n_fields),
        'fitresults': [None]*n_fields,
        }
    if port_type == 'notch':
        fit_report['phi0'] = np.array([np.nan]*n_fields)
    if method == 'resonator_tools':
        return _fit_field_sweep_resonator_tools(data,freq,field,center_freqs,span,field_range,power,
                                                port_type,fit_report,plot)
    elif method == 'lmfit':
        return _fit_field_sweep_lmfit(data,freq,field,center_freqs,span,field_range,power,
                                      port_type,guesses,fit_report,freq_unit,plot,plot_initial_guesses,**kwargs)
    else:
        raise ValueError(f"The method '{method}' is not supported. Use 'resonator_tools' or 'lmfit'")
    
def _fit_field_sweep_resonator_tools(
        data,
        freq,
        field,
        center_freqs,
        span,
        field_range,
        power,
        port_type,
        fit_report,
        plot
        ):
    for k,cfreq in enumerate(center_freqs):
        if field_range:
            if (field[k] < field_range[0]) or (field[k] > field_range[1]):
                continue
        # define port type
        if port_type == 'notch':
            port = circuit.notch_port()
        elif port_type == 'reflection':
            port = circuit.reflection_port()
        else:
            print("This port type is not supported. Use 'notch', 'reflection' or 'transmission'")
        # cut and fit data
        port.add_data(freq,data[k])
        port.cut_data(cfreq-span/2,cfreq+span/2)
        # get environment parameters
        delay, a, alpha, fr, Ql, A2, frcal = port.do_calibration(port.f_data, port.z_data_raw, guessdelay=False)
        fit_report['delay'][k] = delay
        fit_report['a'][k] = a
        fit_report['alpha'][k] = alpha
        port.autofit()
        if plot == True:
            print(f'Field = {field[k]*1e3} mT')
            port.plotall()
        # add fitting results to the dictionary
        if port_type == 'notch':
            fit_report["Qi"][k] = port.fitresults["Qi_dia_corr"]
            fit_report["Qi_err"][k] = port.fitresults["Qi_dia_corr_err"]
            fit_report["Qc"][k] = port.fitresults["Qc_dia_corr"]
            fit_report["Qc_err"][k] = port.fitresults["absQc_err"]
        else:
            fit_report["Qi"][k] = port.fitresults["Qi"]
            fit_report["Qi_err"][k] = port.fitresults["Qi_err"]
            fit_report["Qc"][k] = port.fitresults["Qc"]
            fit_report["Qc_err"][k] = port.fitresults["Qc_err"]
        fit_report["Ql"][k] = port.fitresults["Ql"]
        fit_report["Ql_err"][k] = port.fitresults["Ql_err"]
        if port_type == 'notch':
            fit_report["phi0"] = port.fitresults["phi0"]
        fit_report["Nph"][k] = port.get_photons_in_resonator(power,unit='dBm')
        fit_report["single_photon_W"][k] = port.get_single_photon_limit(unit='watt')
        fit_report["single_photon_dBm"][k] = port.get_single_photon_limit(unit='dBm')
        fit_report["fr"][k] = port.fitresults["fr"]
        fit_report['fitresults'][k] = port.fitresults
    return fit_report

def _fit_field_sweep_lmfit(
        data,
        freq,
        field,
        center_freqs,
        span,
        field_range,
        power,
        port_type,
        user_guesses,
        fit_report,
        freq_unit,
        plot,
        plot_initial_guesses,
        **kwargs
        ):
    for k,cfreq in enumerate(center_freqs):
        if field_range:
            if (field[k] < field_range[0]) or (field[k] > field_range[1]):
                continue
        # Define port type
        if port_type == 'notch':
            model_func = S21_resonator_notch
        elif port_type == 'reflection':
            model_func = S11_resonator_reflection
        else:
            print("This port type is not supported. Supported types are 'notch' or 'reflection'")
        
        data_to_fit = data[k]
        freq_slice = myplt.find_slice(freq, [cfreq-span/2,cfreq+span/2])
        freq_to_fit = freq[freq_slice]
        data_to_fit = data_to_fit[freq_slice]

        fr, kappa, a, alpha, delay = guess_resonator_params(freq_to_fit, data_to_fit, port_type=port_type)

        fixed_params = []
        # define the initial guesses for the parameters
        guesses = {}
        guesses['fr'] = fr
        guesses['kappa'] = kappa
        guesses['kappa_c'] = kappa/2
        guesses['a'] = a
        guesses['alpha']= alpha
        guesses['delay'] = delay
        if port_type == 'notch':
            guesses['phi0'] = 0
        
        # overwrite the initial guesses with the ones provided by the user
        for key, value in user_guesses.items():
            guesses[key] = value

        # create the lmfit.Parameters object and adjust some settings
        params=lmfit.Parameters() # object
        signature = inspect.signature(model_func)
        parameter_names = [param.name for param in signature.parameters.values() if param.name not in ['fdrive']]
        for name in parameter_names:
            params.add(name, value=guesses[name], vary=True)
        for name in fixed_params:
            params[name].vary = False
        params.add('kappa_i', expr='kappa-kappa_c')
        params.add('Ql', expr='fr/(kappa)')
        params.add('Qc', expr='fr/kappa_c')
        params.add('Qi', expr='fr/(kappa-kappa_c)')

        for param_name in ['fr', 'kappa', 'kappa_c', 'a']:
            params[param_name].min = 0
        params['alpha'].min = -np.pi
        params['alpha'].max = np.pi

        # fit the data
        model = lmfit.Model(model_func, independent_vars=['fdrive'])
        result = model.fit(data_to_fit, params, fdrive=freq_to_fit, **kwargs)
        par = result.params
        fit_report["fr"][k] = par['fr'].value
        for param_name in ['kappa_i', 'kappa_c', 'kappa', 'Qi', 'Qc', 'Ql']: 
            fit_report[param_name][k] = par[param_name].value
            fit_report[param_name+'_err'][k] = par[param_name].stderr
        fit_report['a'][k] = par['a'].value
        fit_report['alpha'][k] = par['alpha'].value
        fit_report['delay'][k] = par['delay'].value
        fit_report["Nph"][k] = get_photons_in_resonator(power, result.best_values, freq_unit='Hz' ,unit='dBm')
        fit_report["single_photon_W"][k] = get_single_photon_limit(result.best_values, freq_unit=freq_unit, unit='watt')
        fit_report["single_photon_dBm"][k] = get_single_photon_limit(result.best_values, freq_unit=freq_unit, unit='dBm')
        fit_report['fitresults'][k] = result
        
        if plot:
            print(f'Field = {field[k]*1e3} mT')
            plot_resonator_fit_lmfit(freq_to_fit, data_to_fit, result, freq_unit=freq_unit,
                                      plot_initial_guesses=plot_initial_guesses)
            plt.show()
    return fit_report


def fit_multi_peak(data,freq,center_freqs,span,power,attenuation=80,port_type='notch',plot=False):
    fitReport = {
        "Qi" : [],
        "Qi_err" : [],
        "Qc" : [],
        "Qc_err" : [],
        "Ql" : [],
        "Ql_err" : [],
        "Nph" : [],
        "fr" : [],
        }
    for cfreq in center_freqs:
        # define port type
        if port_type == 'notch':
            port = circuit.notch_port()
        elif port_type == 'reflection':
            port = circuit.reflection_port()
        else:
            print("This port type is not supported. Use 'notch', 'reflection' or 'transmission'")
        # cut and fit data
        port.add_data(freq,data)
        port.cut_data(cfreq-span/2,cfreq+span/2)
        # port.autofit(fr_guess=cfreq)
        port.autofit()
        if plot: port.plotall()
        # add fitting results to the dictionary
        if port_type == 'notch':
            fitReport["Qi"].append(port.fitresults["Qi_dia_corr"])
            fitReport["Qi_err"].append(port.fitresults["Qi_dia_corr_err"])
            fitReport["Qc"].append(port.fitresults["Qc_dia_corr"])
            fitReport["Qc_err"].append(port.fitresults["absQc_err"])
        else:
            fitReport["Qi"].append(port.fitresults["Qi"])
            fitReport["Qi_err"].append(port.fitresults["Qi_err"])
            fitReport["Qc"].append(port.fitresults["Qc"])
            fitReport["Qc_err"].append(port.fitresults["Qc_err"])
        fitReport["Ql"].append(port.fitresults["Ql"])
        fitReport["Ql_err"].append(port.fitresults["Ql_err"])
        fitReport["Nph"].append(port.get_photons_in_resonator(power - attenuation,unit='dBm'))
        fitReport["fr"].append(port.fitresults["fr"])
    return fitReport

def fit_correction(fitReport, threshold=1):
    """
    Filters and returns corrected fit report dictionary based on error thresholds.
    This function processes a fit report dictionary, filtering out entries
    where the relative errors of the quality factors exceed a specified threshold.
    It returns a corrected fit report dictionary and a list of indices corresponding
    to the entries that passed the filtering criteria.
    Parameters:
    -----------
    fitReport : dict
        A dictionary containing fit results and their associated errors. 
        Expected keys include:
        - "Qi": Array of internal quality factors.
        - "Qi_err": Array of errors associated with "Qi".
        - "Qc": Array of coupling quality factors.
        - "Qc_err": Array of errors associated with "Qc".
        - "Ql": Array of loaded quality factors.
        - "Ql_err": Array of errors associated with "Ql".
    threshold : float, optional
        The maximum allowable relative error for each quality factor. 
        Default is 1 (100%).
    Returns:
    --------
    fitReportCorr : dict
        A dictionary containing only the entries from the input `fitReport`
        that passed the filtering criteria. The structure of this dictionary
        matches the input `fitReport`.
    good_fit_idx : list
        A list of indices corresponding to the entries in the input `fitReport`
        that passed the filtering criteria.
    Example:
    --------
    >>> fitReport = {
    ...     "Qi": [1000, 2000],
    ...     "Qi_err": [50, 300],
    ...     "Qc": [500, 1500],
    ...     "Qc_err": [25, 200],
    ...     "Ql": [800, 1800],
    ...     "Ql_err": [40, 250],
    ... }
    >>> fit_correction(fitReport, threshold=0.1)
    (
        {
            "Qi": [1000],
            "Qi_err": [50],
            "Qc": [500],
            "Qc_err": [25],
            "Ql": [800],
            "Ql_err": [40],
        },
        [0]
    )
    """
    # create empty dictionary
    fitReportCorr = {}
    good_fit_idx = []
    for key in fitReport.keys():
        fitReportCorr[key] = np.array([])
    
    # add fit result only if the all the constrains pass
    for k in range(len(fitReport["Qi"])):
        bool_Qi = fitReport["Qi_err"][k] < threshold*fitReport["Qi"][k]
        bool_Qc = fitReport["Qc_err"][k] < threshold*fitReport["Qc"][k]
        bool_Ql = fitReport["Ql_err"][k] < threshold*fitReport["Ql"][k]
        if bool_Qi and bool_Qc and bool_Ql:
            for key in fitReportCorr.keys():
                fitReportCorr[key] = np.append(fitReportCorr[key], fitReport[key][k])
            good_fit_idx.append(k)
    return fitReportCorr, good_fit_idx

def plot_QvsP(fit_report,label='',log_y=True,threshold=None,**kwargs):
    """
    Plots the quality factors (Qi, Qc, Ql) versus photon number (Nph) with error bars.

    Parameters:
    -----------
    fit_report : dict
        Dictionary containing the results of the fit as np.arrays for each parameter.
    label : str, optional
        Title of the plot. Default is an empty string.
    log_y : bool, optional
        If True, use a logarithmic scale for the y-axis. Default is True.
    threshold : float, optional
        Threshold factor for the discarding bad fits. This factor is used to filter out the fits with large errors
        using the conditions threshold*Q < Q_err for all quality factors. If None, no fit is discarded. Default is None.
    **kwargs : dict, optional
        Additional keyword arguments passed to the errorbar function.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.
    """
    if threshold is not None:
        fit, field_idxs = fit_correction(fit_report, threshold)
    else:
        fit = fit_report
    fig, ax = plt.subplots(1)
    if log_y:
        ax.loglog()
    else:
        ax.semilogx()
    ax.errorbar(fit['Nph'],fit['Qi'],yerr=fit['Qi_err'],label='$Q_{i}$',fmt = "o",**kwargs)
    ax.errorbar(fit['Nph'],fit['Qc'],yerr=fit['Qc_err'],label='$Q_{c}$',fmt = "o",**kwargs)
    ax.errorbar(fit['Nph'],fit['Ql'],yerr=fit['Ql_err'],label='$Q_{l}$',fmt = "o",**kwargs)
    ax.legend()
    ax.set_xlabel('photon number')
    ax.set_ylabel('Q')
    ax.grid()
    fig.suptitle(label)
    fig.tight_layout()
    return fig,ax

def plot_kappavsP(fit_report,label='',log_y=True,threshold=None,freq_unit='Hz',**kwargs):
    """
    Plots the quality factors (kappa_i, kappa_c, kappa_l) versus photon number (Nph) with error bars.

    Parameters:
    -----------
    fit_report : dict
        Dictionary containing the results of the fit as np.arrays for each parameter.
    label : str, optional
        Title of the plot. Default is an empty string.
    log_y : bool, optional
        If True, use a logarithmic scale for the y-axis. Default is True.
    threshold : float, optional
        Threshold factor for the discarding bad fits. This factor is used to filter out the fits with large errors
        using the conditions threshold*Q < Q_err for all quality factors. If None, no fit is discarded. Default is None.
    freq_unit : str, optional
        Unit in which frequency is provided. This will determine the proper scaling factors. Default is 'Hz'.
    **kwargs : dict, optional
        Additional keyword arguments passed to the errorbar function.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.
    """
    if threshold is not None:
        fit, field_idxs = fit_correction(fit_report, threshold)
    else:
        fit = fit_report
    fig, ax = plt.subplots(1)
    if log_y:
        ax.loglog()
    else:
        ax.semilogx()
    kappa_scaling = get_frequency_scaling(freq_unit)/1e6
    ax.errorbar(fit['Nph'],fit['kappa_i']*kappa_scaling,yerr=fit['kappa_i_err']*kappa_scaling,label='$\kappa_{i}$',fmt = "o",**kwargs)
    ax.errorbar(fit['Nph'],fit['kappa_c']*kappa_scaling,yerr=fit['kappa_c_err']*kappa_scaling,label='$\kappa_{c}$',fmt = "o",**kwargs)
    ax.errorbar(fit['Nph'],fit['kappa']*kappa_scaling,yerr=fit['kappa_err']*kappa_scaling,label='$\kappa$',fmt = "o",**kwargs)
    ax.legend()
    ax.set_xlabel('photon number')
    ax.set_ylabel('$\kappa/2\pi$ (MHz)')
    ax.grid()
    fig.suptitle(label)
    fig.tight_layout()
    return fig,ax

def plot_QvsB(fit_report,field,label='',log_y=True,threshold=None,**kwargs):
    """
    Plots the quality factors (Qi, Qc, Ql) with error bars versus magnetic field.

    Parameters:
    -----------
    fit_report : dict
        Dictionary containing the results of the fit as np.arrays for each parameter.
    field : ndarray
        List of the magnetic field values.
    label : str, optional
        Title of the plot. Default is an empty string.
    log_y : bool, optional
        If True, use a logarithmic scale for the y-axis. Default is True.
    threshold : float, optional
        Threshold factor for the discarding bad fits. This factor is used to filter out the fits with large errors
        using the conditions threshold*Q < Q_err for all quality factors. If None, no fit is discarded. Default is None.
    **kwargs : dict, optional
        Additional keyword arguments passed to the errorbar function.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.
    """
    if threshold is not None:
        fit, field_idxs = fit_correction(fit_report, threshold)
        field = np.array([field[idx] for idx in field_idxs])
    else:
        fit = fit_report
        field = field
    fig, ax = plt.subplots(1)
    if log_y:
        ax.semilogy()
    ax.errorbar(field*1e3,fit['Qi'],yerr=fit['Qi_err'],label='$Q_{int}$',fmt = "o",**kwargs)
    ax.errorbar(field*1e3,fit['Qc'],yerr=fit['Qc_err'],label='$Q_{ext}$',fmt = "o",**kwargs)
    ax.errorbar(field*1e3,fit['Ql'],yerr=fit['Ql_err'],label='$Q_{load}$',fmt = "o",**kwargs)
    ax.legend()
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Q')
    ax.grid()
    fig.suptitle(label)
    fig.tight_layout()
    return fig,ax

def plot_kappavsB(fit_report,field,label='',log_y=True,threshold=None,freq_unit='Hz',**kwargs):
    """
    Plots the quality factors (kappa_i, kappa_c, kappa_l) with error bars versus magnetic field.

    Parameters:
    -----------
    fit_report : dict
        Dictionary containing the results of the fit as np.arrays for each parameter.
    field : ndarray
        List of the magnetic field values.
    label : str, optional
        Title of the plot. Default is an empty string.
    log_y : bool, optional
        If True, use a logarithmic scale for the y-axis. Default is True.
    threshold : float, optional
        Threshold factor for the discarding bad fits. This factor is used to filter out the fits with large errors
        using the conditions threshold*Q < Q_err for all quality factors. If None, no fit is discarded. Default is None.
    freq_unit : str, optional
        Unit in which frequency is provided. This will determine the proper scaling factors. Default is 'Hz'.
    **kwargs : dict, optional
        Additional keyword arguments passed to the errorbar function.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.
    """
    if threshold is not None:
        fit, field_idxs = fit_correction(fit_report, threshold)
        field = np.array([field[idx] for idx in field_idxs])
    else:
        fit = fit_report
        field = field
    fig, ax = plt.subplots(1)
    if log_y:
        ax.semilogy()
    kappa_scaling = get_frequency_scaling(freq_unit)/1e6
    ax.errorbar(field*1e3,fit['kappa_i']*kappa_scaling,yerr=fit['kappa_i_err']*kappa_scaling,label='$\kappa_{i}$',fmt = "o",**kwargs)
    ax.errorbar(field*1e3,fit['kappa_c']*kappa_scaling,yerr=fit['kappa_c_err']*kappa_scaling,label='$\kappa_{c}$',fmt = "o",**kwargs)
    ax.errorbar(field*1e3,fit['kappa']*kappa_scaling,yerr=fit['kappa_err']*kappa_scaling,label='$\kappa$',fmt = "o",**kwargs)
    ax.legend()
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('$\kappa/2\pi$ (MHz)')
    ax.grid()
    fig.suptitle(label)
    fig.tight_layout()
    return fig,ax

def plot_Qvsfr(fit_report,label='',log_y=True,threshold=None,freq_unit='Hz',**kwargs):
    """
    Plots the quality factors (Qi, Qc, Ql) with error bars versus resonance frequency.

    Parameters:
    -----------
    fit_report : dict
        Dictionary containing the results of the fit as np.arrays for each parameter.
    label : str, optional
        Title of the plot. Default is an empty string.
    log_y : bool, optional
        If True, use a logarithmic scale for the y-axis. Default is True.
    threshold : float, optional
        Threshold factor for the discarding bad fits. This factor is used to filter out the fits with large errors
        using the conditions threshold*Q < Q_err for all quality factors. If None, no fit is discarded. Default is None.
    freq_unit : str, optional
        Unit in which frequency is provided. This will determine the proper scaling factors. Default is 'Hz'.
    **kwargs : dict, optional
        Additional keyword arguments passed to the errorbar function.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.
    """
    if threshold is not None:
        fit, field_idxs = fit_correction(fit_report, threshold)
        frs = fit['fr'][field_idxs]
    else:
        fit = fit_report
        frs = fit['fr']
    fig, ax = plt.subplots(1)
    if log_y:
        ax.semilogy()
    fr_scaling = get_frequency_scaling(freq_unit)/1e9
    ax.errorbar(frs*fr_scaling,fit['Qi'],yerr=fit['Qi_err'],label='$Q_{int}$',fmt = "o",**kwargs)
    ax.errorbar(frs*fr_scaling,fit['Qc'],yerr=fit['Qc_err'],label='$Q_{ext}$',fmt = "o",**kwargs)
    ax.errorbar(frs*fr_scaling,fit['Ql'],yerr=fit['Ql_err'],label='$Q_{load}$',fmt = "o",**kwargs)
    ax.legend()
    ax.set_xlabel('$f_r$ (GHz)')
    ax.set_ylabel('Q')
    ax.grid()
    fig.suptitle(label)
    fig.tight_layout()
    return fig,ax

def plot_kappavsfr(fit_report,label='',log_y=True,threshold=None,freq_unit='Hz',**kwargs):
    """
    Plots the quality factors (kappa_i, kappa_c, kappa_l) with error bars versus magnetic field.

    Parameters:
    -----------
    fit_report : dict
        Dictionary containing the results of the fit as np.arrays for each parameter.
    label : str, optional
        Title of the plot. Default is an empty string.
    log_y : bool, optional
        If True, use a logarithmic scale for the y-axis. Default is True.
    threshold : float, optional
        Threshold factor for the discarding bad fits. This factor is used to filter out the fits with large errors
        using the conditions threshold*Q < Q_err for all quality factors. If None, no fit is discarded. Default is None.
    freq_unit : str, optional
        Unit in which frequency is provided. This will determine the proper scaling factors. Default is 'Hz'.
    **kwargs : dict, optional
        Additional keyword arguments passed to the errorbar function.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.
    """
    if threshold is not None:
        fit, field_idxs = fit_correction(fit_report, threshold)
        frs = fit['fr'][field_idxs]
    else:
        fit = fit_report
        frs = fit['fr']
    fig, ax = plt.subplots(1)
    if log_y:
        ax.semilogy()
    fr_scaling = get_frequency_scaling(freq_unit)/1e9
    kappa_scaling = get_frequency_scaling(freq_unit)/1e6
    ax.errorbar(frs*fr_scaling,fit['kappa_i']*kappa_scaling,yerr=fit['kappa_i_err']*kappa_scaling,label='$\kappa_{i}$',fmt = "o",**kwargs)
    ax.errorbar(frs*fr_scaling,fit['kappa_c']*kappa_scaling,yerr=fit['kappa_c_err']*kappa_scaling,label='$\kappa_{c}$',fmt = "o",**kwargs)
    ax.errorbar(frs*fr_scaling,fit['kappa']*kappa_scaling,yerr=fit['kappa_err']*kappa_scaling,label='$\kappa$',fmt = "o",**kwargs)
    ax.legend()
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('$\kappa/2\pi$ (MHz)')
    ax.grid()
    fig.suptitle(label)
    fig.tight_layout()
    return fig,ax


#%% legacy functions


def peakFindList(data, prominence = 2):
    peaks = []
    for k in data:
        peaks.append(find_peaks(k,  prominence = prominence)[0])
    return peaks

def multiPeakExtract(data, freq = None, prominence = 0.0005, height = None):
    """
    Extract multiple peaks in a single trace.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    freq : TYPE, optional
        DESCRIPTION. The default is None.
    prominence : TYPE, optional
        DESCRIPTION. The default is 0.0005.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if freq == None:
        return find_peaks(data,  prominence = prominence, height = height)[0]
    else:
        return find_peaks(data,  prominence = prominence)[0], freq[find_peaks(data,  prominence = prominence, height = height)[0][0]]
    

def multiPeakFitNotch(data, freq, power, peaks, pointSpan, attenuation = 60, plot = False):
    fitReport = {
        "Qi" : [],
        "Qi_err" : [],
        "Qc" : [],
        "Qc_err" : [],
        "Ql" : [],
        "Ql_err" : [],
        "Nph" : [],
        "fr" : []
        }
    for k in peaks:
        port = circuit.notch_port()
        port.add_data(freq[k-pointSpan:k+pointSpan], data[k-pointSpan:k+pointSpan])
        port.autofit()
        if plot == True:
            port.plotall()
        
        #adding fitting results to the dictionary
        fitReport["Qi"].append(port.fitresults["Qi_dia_corr"])
        fitReport["Qi_err"].append(port.fitresults["Qi_dia_corr_err"])
        fitReport["Qc"].append(port.fitresults["Qc_dia_corr"])
        fitReport["Qc_err"].append(port.fitresults["Qi_dia_corr"])
        fitReport["Ql"].append(port.fitresults["Ql"])
        fitReport["Ql_err"].append(port.fitresults["Ql_err"])
        fitReport["Nph"].append(port.get_photons_in_resonator(power - attenuation,unit='dBm',diacorr=True))
        fitReport["fr"].append(port.fitresults["fr"])
        
    return fitReport

def multiPeakFitNotchPowerScan(data, freq, power, peaks, pointSpan, attenuation = 60, plot = False):
    fitReportPowerScan = []
    for i,j in enumerate(power):
        print(i)
        fitReport = {
            "Qi" : [],
            "Qi_err" : [],
            "Qc" : [],
            "Qc_err" : [],
            "Ql" : [],
            "Ql_err" : [],
            "Nph" : [],
            "fr" : []
            }
        for k in peaks:
            port = circuit.notch_port()
            port.add_data(freq[k-pointSpan:k+pointSpan], data[k-pointSpan:k+pointSpan])
            port.autofit()
            if plot == True:
                port.plotall()
            
            #adding fitting results to the dictionary
            fitReport["Qi"].append(port.fitresults["Qi_dia_corr"])
            fitReport["Qi_err"].append(port.fitresults["Qi_dia_corr_err"])
            fitReport["Qc"].append(port.fitresults["Qc_dia_corr"])
            fitReport["Qc_err"].append(port.fitresults["Qi_dia_corr"])
            fitReport["Ql"].append(port.fitresults["Ql"])
            fitReport["Ql_err"].append(port.fitresults["Ql_err"])
            fitReport["Nph"].append(port.get_photons_in_resonator(power - attenuation,unit='dBm',diacorr=True))
            fitReport["fr"].append(port.fitresults["fr"])
        fitReportPowerScan.append(fitReport)
    return fitReport

def multiPeakFitReflection(data, freq, power, peaks, pointSpan, attenuation = 60, plot = False):
    fitReport = {
        "Qi" : [],
        "Qi_err" : [],
        "Qc" : [],
        "Qc_err" : [],
        "Ql" : [],
        "Ql_err" : [],
        "Nph" : [],
        "fr" : []
        }
    for k in peaks:
        port = circuit.reflection_port()
        port.add_data(freq[k-pointSpan:k+pointSpan], data[k-pointSpan:k+pointSpan])
        port.autofit()
        if plot == True:
            port.plotall()
        
        #adding fitting results to the dictionary
        fitReport["Qi"].append(port.fitresults["Qi"])
        fitReport["Qi_err"].append(port.fitresults["Qi_err"])
        fitReport["Qc"].append(port.fitresults["Qc"])
        fitReport["Qc_err"].append(port.fitresults["Qc_err"])
        fitReport["Ql"].append(port.fitresults["Ql"])
        fitReport["Ql_err"].append(port.fitresults["Ql_err"])
        fitReport["Nph"].append(port.get_photons_in_resonator(power - attenuation,unit='dBm'))
        fitReport["fr"].append(port.fitresults["fr"])
        
    return fitReport

def multiPeakFitReflectionPowerScan(data, freq, power, peaks, pointSpan, attenuation = 70, plot = False):
    fitReportPowerScan = []
    for i,j in enumerate(power):
        print(i)
        fitReport = {
            "Qi" : [],
            "Qi_err" : [],
            "Qc" : [],
            "Qc_err" : [],
            "Ql" : [],
            "Ql_err" : [],
            "Nph" : [],
            "fr" : []
            }
        for k in peaks:
            port = circuit.reflection_port()
            port.add_data(freq[k-pointSpan:k+pointSpan], data[i][k-pointSpan:k+pointSpan])
            port.autofit()
            if plot == True:
                port.plotall()
            
            #adding fitting results to the dictionary
            fitReport["Qi"].append(port.fitresults["Qi"])
            fitReport["Qi_err"].append(port.fitresults["Qi_err"])
            fitReport["Qc"].append(port.fitresults["Qc"])
            fitReport["Qc_err"].append(port.fitresults["Qc_err"])
            fitReport["Ql"].append(port.fitresults["Ql"])
            fitReport["Ql_err"].append(port.fitresults["Ql_err"])
            fitReport["Nph"].append(port.get_photons_in_resonator(j - attenuation,unit='dBm'))
            fitReport["fr"].append(port.fitresults["fr"])
        fitReportPowerScan.append(fitReport)
    return fitReportPowerScan

def multiPeakFitCorrection(fitReport, threshold = 1):
    # create empty dictionary
    fitReportCorr = {}
    for key in fitReport.keys():
        fitReportCorr[key] = []
    
    # add fit result only if the all the constrains pass
    for k in range(len(fitReport["Qi"])):
        bool_Qi = fitReport["Qi_err"][k] < threshold*fitReport["Qi"][k]
        bool_Qc = fitReport["Qc_err"][k] < threshold*fitReport["Qc"][k]
        bool_Ql = fitReport["Ql_err"][k] < threshold*fitReport["Ql"][k]
        if bool_Qi and bool_Qc and bool_Ql:
            for key in fitReportCorr.keys():
                fitReportCorr[key].append(fitReport[key][k])
    return fitReportCorr

def peakFitCorrection(fitReport,
                      params_to_check = ['Qi_dia_corr', 'absQc', 'Ql'],
                      thresholds = 1):
    """
    Parameters
    ----------
    fitReport : dict
        Dictionary that contains the parameters from a fitresult.
    params_to_check : Sequence of Strings, optional
        List of strings that will be compared with their fitting error.
        The default is ['Qi_dia_corr', 'absQc', 'Ql'].
    thresholds : float or Sequence, optional
        Relative threshold value the error must fall below in order to pass.
        If a single float is given, the same value is used for all parameters.
        Individual theshold values for each parameter can be passed as a list.
        The default is 1.

    Returns
    -------
    fitReportCorr : dict
        Dictionary of the same shape as fitReport that contains the results
        that passed the checks.

    """
    # create empty dictionary
    fitReportCorr = {}
    for key in fitReport.keys():
        fitReportCorr[key] = np.array([])
        
    # if thresholds is a single number, the same value will be used for all
    if hasattr(thresholds, '__iter__'):
        if len(thresholds) != len(params_to_check):
            print(f'The length of thresholds ({len(thresholds)}) need to match that of params_to_check ({len(params_to_check)}).')
            return 
    else:
        thresholds = [thresholds]*len(params_to_check)
    
    # add fit result only if the all the constrains pass
    
    for k in range(len(fitReport[params_to_check[0]])):
        bools = []
        for i, param in enumerate(params_to_check):            
            bools.append(fitReport[param+'_err'][k] < thresholds[i]*fitReport[param][k])
        if all(bools):
            for key in fitReportCorr.keys():
                fitReportCorr[key] = np.append(fitReportCorr[key],fitReport[key][k])
    return fitReportCorr



#%% old resonator fitting functions (summer 2021)

class Parameters:
    def __init__(param, Qi, Qc, Ql, err_Qi, err_Qc, err_Ql, fr_res, power_at, nu):
        param.Qi = Qi
        param.Qc = Qc
        param.Ql = Ql
        param.err_Qi = err_Qi
        param.err_Qc = err_Qc
        param.err_Ql = err_Ql
        param.fr_res = fr_res
        param.power_at = power_at
        param.nu = nu

    def description(param):
        return (f'Parameters \nQi = {param.Qi}\nQc = {param.Qc}\nQl = {param.Ql}\nQi = {param.Qi}\nerr_Qc = {param.err_Qc}\nerr_Ql = {param.err_Ql}\nfr_res = {param.fr_res}\npower_at = {param.power_at}\nnu = {param.nu}')

def fit_function_Pscan_notch(freq,cData,power,fr_in,fr_out,att=80,plot_var=False,err_threshold=0.2):
    port1 = circuit.notch_port()
    #z_probst = utilities.save_load()._ConvToCompl(MAG,ph, dtype='dBmagphaserad')
    Qi = []
    Qc = []
    Ql = []
    err_Qi = []
    err_Qc = []
    err_Ql = []
    fr_res = []
    power_at = []
    nu = []
    for ii in range(len(power)):
        port1.add_data(freq,cData[ii])
        port1.cut_data(fr_in,fr_out)
        port1.autofit()
        Qi_err_percent = abs(port1.fitresults['Qi_dia_corr_err'])/abs(port1.fitresults['Qi_dia_corr'])
        Qc_err_percent = abs(port1.fitresults['absQc_err'])/abs(port1.fitresults['Qc_dia_corr'])
        if err_threshold > Qi_err_percent and err_threshold > Qc_err_percent and 0 < port1.fitresults['Qi_dia_corr'] and 0 < port1.fitresults['Qc_dia_corr']:
            Qi.append(port1.fitresults['Qi_dia_corr'])
            Qc.append(port1.fitresults['Qc_dia_corr'])
            Ql.append(port1.fitresults['Ql'])
            err_Qi.append(port1.fitresults['Qi_dia_corr_err'])
            err_Qc.append(port1.fitresults['absQc_err'])
            err_Ql.append(port1.fitresults['Ql_err'])
            fr_res.append(port1.fitresults['fr'])
            power_at.append(power[ii]-att)
            nu.append(port1.get_photons_in_resonator(power[ii]-att))
            port1.z_data
        if plot_var == True:
            port1.plotall()
    out = Parameters(Qi, Qc, Ql, err_Qi, err_Qc, err_Ql, fr_res, power_at, nu)
    print('ok')
    return out

def fit_function_Pscan_refl(freq,cData,power,fr_in,fr_out,att=80,plot_var=False):
    port1 = circuit.reflection_port()
    #z_probst = utilities.save_load()._ConvToCompl(MAG,ph, dtype='dBmagphaserad')
    Qi = []
    Qc = []
    Ql = []
    err_Qi = []
    err_Qc = []
    err_Ql = []
    fr_res = []
    power_at = []
    nu = []
    for ii in range(len(power)):
        port1.add_data(freq,cData[ii])
        port1.cut_data(fr_in,fr_out)
        port1.autofit()
        Qi.append(port1.fitresults['Qi'])
        Qc.append(port1.fitresults['Qc'])
        Ql.append(port1.fitresults['Ql'])
        err_Qi.append(port1.fitresults['Qi_err'])
        err_Qc.append(port1.fitresults['Qc_err'])
        err_Ql.append(port1.fitresults['Ql_err'])
        fr_res.append(port1.fitresults['fr'])
        power_at.append(power[ii]-att)
        nu.append(port1.get_photons_in_resonator(power[ii]-att))
        port1.z_data
        if plot_var == True:
            port1.plotall()
    out = Parameters(Qi, Qc, Ql, err_Qi, err_Qc, err_Ql, fr_res, power_at, nu)
    print('ok')
    return out

def plot_Qfactors_vs_power(fit,label='',filename=False,file_res=300):
    fig, ax = plt.subplots(1)
    fig.dpi = file_res
    ax.loglog()
    ax.errorbar(fit.nu,fit.Qi,yerr=fit.err_Qi,label='Qi',fmt = "o")
    ax.errorbar(fit.nu,fit.Qc,yerr=fit.err_Qc,label='Qc',fmt = "o")
    ax.errorbar(fit.nu,fit.Ql,yerr=fit.err_Ql,label='Ql',fmt = "o")
    ax.legend()
    ax.set_xlabel('photon number')
    ax.set_ylabel('Q')
    ax.grid()
    fig.suptitle(label)
    fig.tight_layout()
    if filename:
        fig.savefig(filename,dpi=file_res)
    return fig,ax
        
def plot_Qfactors_vs_power_linear(fit,label='',filename=False,file_res=300):
    fig, ax = plt.subplots(1)
    fig.dpi = file_res
    plt.semilogx()
    ax.errorbar(fit.nu,fit.Qi,yerr=fit.err_Qi,label='Qi',fmt = "o")
    ax.errorbar(fit.nu,fit.Qc,yerr=fit.err_Qc,label='Qc',fmt = "o")
    ax.errorbar(fit.nu,fit.Ql,yerr=fit.err_Ql,label='Ql',fmt = "o")
    ax.legend()
    ax.set_xlabel('photon number')
    ax.set_ylabel('Q')
    ax.grid()
    fig.suptitle(label)
    fig.tight_layout()
    if filename:
        fig.savefig(filename,dpi=file_res)
    return fig,ax
        
