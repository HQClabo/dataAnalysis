# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:22:53 2022

@author: jouanny
"""

from scipy.signal import find_peaks
import numpy as np
from resonator_tools import circuit
import matplotlib.pyplot as plt


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

def fitPowerSweep(data, freq, center_freq, span, power, attenuation = 80, port_type='notch', plot = False):
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
    # lowLimit = [centers[0]-int(spanPoints/2):centers[0]+int(spanPoints/2)]
    for k,i in enumerate(power):
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
        fitReport["Nph"].append(port.get_photons_in_resonator(i - attenuation,unit='dBm'))
        fitReport["fr"].append(port.fitresults["fr"])
        # fitReport["guessed_freq"].append()
        
    return fitReport

def plot_QvsP(fit,label='',filename=False,file_res=300,log=True):
    fig, ax = plt.subplots(1)
    fig.dpi = file_res
    if log:
        ax.loglog()
    else:
        ax.semilogx()
    ax.errorbar(fit['Nph'],fit['Qi'],yerr=fit['Qi_err'],label='Qi',fmt = "o")
    ax.errorbar(fit['Nph'],fit['Qc'],yerr=fit['Qc_err'],label='Qc',fmt = "o")
    ax.errorbar(fit['Nph'],fit['Ql'],yerr=fit['Ql_err'],label='Ql',fmt = "o")
    ax.legend()
    ax.set_xlabel('photon number')
    ax.set_ylabel('Q')
    ax.grid()
    fig.suptitle(label)
    fig.tight_layout()
    if filename:
        fig.savefig(filename,dpi=file_res)
    return fig,ax




