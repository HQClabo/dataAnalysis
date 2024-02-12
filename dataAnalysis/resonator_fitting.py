# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 13:43:03 2022

@author: fopplige
"""

from scipy.signal import find_peaks
import numpy as np
from resonator_tools import circuit
import matplotlib.pyplot as plt

#%% functions for resonator fitting

def fit_power_sweep(data, freq, center_freq, span, power, attenuation=80, port_type='notch', plot=False):
    """
    Function to fit resonances of a power sweep.

    Parameters
    ----------
    data : ndarray
        Complex data of dimension (n_power,n_freq).
    freq : ndarray
        List of the measured frequencies.
    center_freq : number or array-like
        Center frequency that will be used as an initial guess for the fit.
        Can be passed as a number to use the same for all powers or as a list
        to use individual values for each power.
    span : array-like
        Frequency span around center_freq that will be considered for the fit.
    power : ndarray
        List of the different powers applied.
    attenuation : number, optional
        Total estimated line attenuation. The default is 80.
    port_type : str, optional
        Type of the resonator port. Choose 'notch' or 'reflection.
        The default is 'notch'.
    plot : boolean, optional
        Enable option to plot the individual fits. The default is False.

    Returns
    -------
    fitReport : dict
        Dictionary containing the results of the fit as lists for each parameter.

    """
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
    for k,i in enumerate(power):
        # define port type
        if port_type == 'notch':
            port = circuit.notch_port()
        elif port_type == 'reflection':
            port = circuit.reflection_port()
        else:
            print("This port type is not supported. Use 'notch', 'reflection' or 'transmission'")
        # cut and fit data
        port.add_data(freq,data[k])
        port.cut_data(center_freq[k]-span/2,center_freq[k]+span/2)
        # port.autofit(fr_guess=center_freq[k])
        port.autofit()
        if plot == True:
            port.plotall()
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
        fitReport["Nph"].append(port.get_photons_in_resonator(i - attenuation,unit='dBm'))
        fitReport["fr"].append(port.fitresults["fr"])
    return fitReport

def fit_flux_sweep(data,freq,x,center_freq,span,power,attenuation=80,port_type='notch',plot=False):
    fitReport = {
        "Qi" : [],
        "Qi_err" : [],
        "Qc" : [],
        "Qc_err" : [],
        "Ql" : [],
        "Ql_err" : [],
        "Nph" : [],
        "fr" : [],
        # "guessed_freq": freq[center_freq]
        }
    try: iter(center_freq)
    except TypeError:
        center_freq = [center_freq for ii in range(len(x))]
        
    for k in range(len(x)):
        if port_type == 'notch':
            port = circuit.notch_port()
        elif port_type == 'reflection':
            port = circuit.reflection_port()
        else:
            print("This port type is not supported. Use 'notch', 'reflection' or 'transmission'")
        # port.add_data(freq[center_freq-int(spanPoints/2):center_freq+int(spanPoints/2)], data[k][center_freq-int(spanPoints/2):center_freq+int(spanPoints/2)])
        port.add_data(freq,data[k])
        port.cut_data(center_freq[k]-span/2,center_freq[k]+span/2)
        # port.autofit(fr_guess=center_freq[k])
        port.autofit()
        if plot == True:
            port.plotall()
        #adding fitting results to the dictionary
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
        fitReport["Nph"].append(port.get_photons_in_resonator(power-attenuation,unit='dBm'))
        fitReport["fr"].append(port.fitresults["fr"])
        # fitReport["guessed_freq"].append()
    return fitReport

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

def plot_QvsP(fit,label='',filename=False,file_res=300,log=True,**kwargs):
    fig, ax = plt.subplots(1)
    fig.dpi = file_res
    if log:
        ax.loglog()
    else:
        ax.semilogx()
    ax.errorbar(fit['Nph'],fit['Qi'],yerr=fit['Qi_err'],label='$Q_{int}$',fmt = "o",**kwargs)
    ax.errorbar(fit['Nph'],fit['Qc'],yerr=fit['Qc_err'],label='$Q_{ext}$',fmt = "o",**kwargs)
    ax.errorbar(fit['Nph'],fit['Ql'],yerr=fit['Ql_err'],label='$Q_{load}$',fmt = "o",**kwargs)
    ax.legend()
    ax.set_xlabel('photon number')
    ax.set_ylabel('Q')
    ax.grid()
    fig.suptitle(label)
    fig.tight_layout()
    if filename:
        fig.savefig(filename,dpi=file_res)
    return fig,ax

def fit_correction(fitReport, threshold = 1):
    # create empty dictionary
    fitReportCorr = {}
    bad_fit_idx = []
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
        else:
            bad_fit_idx.append(k)
    return fitReportCorr, bad_fit_idx

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
        
