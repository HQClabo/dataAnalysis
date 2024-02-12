# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:25:41 2023

@author: fopplige
"""


import numpy as np
from scipy import optimize as opt


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
    
    


