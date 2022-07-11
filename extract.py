# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 18:45:34 2022

@author: jouanny
"""

def extract_data(log_file):
    data = log_file.getData()
    freq = log_file.getTraceXY()[0]
    output_power = log_file.getStepChannels()[0]["values"]
    return data, freq, output_power