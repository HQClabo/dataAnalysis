# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:25:41 2023

@author: fopplige
"""


import numpy as np
from scipy import optimize as opt


class CombinedFitModel:
    def __init__(self,data1,data2):
        self.data1 = data1
        self.data2 = data2

    def define_fit_functions(self,fit_func1,fit_func2):
        self.fit_func1 = fit_func1
        self.fit_func2 = fit_func2

    def do_combo_fit(self,x_values, **kwargs):
        combined_data = np.append(self.data1,self.data2)
        combined_x_values = np.append(x_values, x_values)
        coeff, coeff_cov = opt.curve_fit(self.fit_func_combo, combined_x_values, combined_data, **kwargs)
        return coeff, coeff_cov
    
    def fit_func_combo(self, combined_x_values, *args):
    # single data reference passed in, extract separate data
        extract1 = combined_x_values[:len(self.fres)] # first data
        extract2 = combined_x_values[len(self.fres):] # second data
    
        fit1 = self.fit_func1(extract1, *args)
        fit2 = self.fit_func2(extract2, *args)
    
        return np.append(fit1, fit2)
    
    
# example use



