# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:22:53 2022

@author: jouanny
"""

from scipy.signal import find_peaks
import numpy as np
from resonator_tools import circuit


def peakFindList(data, prominence = 2):
    peaks = []
    for k in data:
        peaks.append(find_peaks(k,  prominence = prominence)[0])
    return peaks

def multiPeakExtract(data, freq = None, prominence = 0.0005):
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
        return find_peaks(data,  prominence = prominence)[0]
    else:
        return find_peaks(data,  prominence = prominence)[0], freq[find_peaks(data,  prominence = prominence)[0][0]]

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

def multiPeakFitCorrection(fitReport, threshold = 1):
    fitReportCorr = {
        "Qi" : [],
        "Qi_err" : [],
        "Qc" : [],
        "Qc_err" : [],
        "Ql" : [],
        "Ql_err" : [],
        "Nph_l" : [],
        "Nph_i" : [],
        "Nph_c" : [],
        "fr_i" : [],
        "fr_c" : [],
        "fr_l" : []
        }
    
    #Internal quality factor correction
    for k in range(len(fitReport["Qi"])):
        if fitReport["Qi_err"][k] > threshold*fitReport["Qi"][k]:
            fitReportCorr["Qi"].append(fitReport["Qi"][k])
            fitReportCorr["Qi_err"].append(fitReport["Qi_err"][k])
            fitReportCorr["Nph_i"].append(fitReport["Nph"][k])
            fitReportCorr["fr_i"].append(fitReport["fr"][k])
    
    #Coupling quality factor correction
    for k in range(len(fitReport["Qc"])):
        if fitReport["Qc_err"][k] > threshold*fitReport["Qc"][k]:
            fitReportCorr["Qc"].append(fitReport["Qc"][k])
            fitReportCorr["Qc_err"].append(fitReport["Qc_err"][k])
            fitReportCorr["Nph_c"].append(fitReport["Nph"][k])
            fitReportCorr["fr_c"].append(fitReport["fr"][k])
    
    #Loaded quality factor correction
    for k in range(len(fitReport["Ql"])):
        if fitReport["Ql_err"][k] > threshold*fitReport["Ql"][k]:
            fitReportCorr["Ql"].append(fitReport["Ql"][k])
            fitReportCorr["Ql_err"].append(fitReport["Ql_err"][k])
            fitReportCorr["Nph_l"].append(fitReport["Nph"][k])
            fitReportCorr["fr_l"].append(fitReport["fr"][k])
    
            
    return fitReportCorr








