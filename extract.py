# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 18:45:34 2022

@author: jouanny
"""

import numpy as np
import scipy.io as sio

def extract_data(log_file):
    data = log_file.getData()
    freq = log_file.getTraceXY()[0]
    output_power = log_file.getStepChannels()[0]["values"]
    return data, freq, output_power

def extractDataVNA(files):
    freq = []
    cData = []
    power = []
    # files = [path+file for file in filenames]
    for file in files:
        data_temp = sio.loadmat(file)
        freq_temp, cData_temp, power_temp = data_temp['Frequency'][0],data_temp['ComplexResponse'],data_temp["Power"][0]
        freq.append(freq_temp)
        cData.append(cData_temp)
        power.append(power_temp)
    return np.asarray(freq), np.asarray(cData), np.asarray(power)

def extractDataVNA_Voltage(files):
    freq = []
    cData = []
    voltage = []
    # files = [path+file for file in filenames]
    for file in files:
        data_temp = sio.loadmat(file)
        freq_temp, cData_temp, power_temp = data_temp['Frequency'][0],data_temp['ComplexResponse'],data_temp['Voltage'][0]
        freq.append(freq_temp)
        cData.append(cData_temp)
        voltage.append(power_temp)
    return np.asarray(freq), np.asarray(cData), np.asarray(voltage)