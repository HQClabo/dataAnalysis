## Functions for charge-photon dataset analysis

import h5py
import numpy as np
import matplotlib.pyplot as plt

## Open hdf5 data set and get the VNA data
def deriv(data):
    deriv = []
    for j in range(len(data)):
        if j == 0 :
            deriv.append(0)
        else:
            deriv.append(data[j]-data[j-1])
    deriv = np.array(deriv)
    
    return deriv 

def smooth(data, n):
    
    for j in range(len(data)):
        min_ind = int(j - (n/2))
        max_ind = int(j + (n/2))
        
        if min_ind < 0 :
            min_ind = 0 
            data[j] = sum(data[0:max_ind]/(max_ind+1))
            
        elif max_ind > len(data) -1 :
            max_ind = len(data) - 1
            data[j] = sum(data[min_ind:len(data)])/(len(data)-min_ind + 1)
        else:    
            data[j] = sum(data[min_ind:max_ind])/(n+1)
    
    return data
        

def get_VNA_S21_hdf5(filename):
    
    # Open hdf5 file
    file = h5py.File(filename, 'r') 
    
    ###################################################### Get the I, Q dataset
    # Get the key corresponding to S21 data
    keys_dataset = [*file['Traces'].keys()]
    index_S21 = 'None' 
    for j in range(len(keys_dataset)):
        temp = keys_dataset[j].split('- ')
        if temp[-1] == 'S21':
            index_S21 = j
        
    VNA = file['Traces'][keys_dataset[index_S21]]
    
    VNA_I = VNA[:,0,:] # In-phase
    VNA_Q = VNA[:,1,:] # Quadrature
    
    n_y = VNA_I.shape[0]
    
    ##################################### Get the Frequency, and x-axis dataset
    
    ## x-axis return x_axis
    data = file['Data']['Data']
    x_axis = data[:, 0]
    
    ## y-axis return frequency axis
    keys_config = [*file['Step config'].keys()]
    index_center_freq = 'None'
    index_span = 'None'
    
    # Get the keys corresponding to center / span freq. 
    for j in range(len(keys_config)):
        temp = keys_config[j].split('- ')
        if temp[-1] == 'Center frequency':
            index_center_freq = j
        elif temp[-1] == 'Span':
            index_span = j 
   
    
    Center_freq = file['Step config'][keys_config[index_center_freq]]['Step items'][0][2]
    Span_freq = file['Step config'][keys_config[index_span]]['Step items'][0][2]
    
    freq = []
    for j in range(n_y):
        freq.append(Center_freq - Span_freq/2 + Span_freq*(j-1)/(n_y - 1))
    
    x_axis = np.array(x_axis)[:,0]
    freq = np.array(freq)
    return VNA_I, VNA_Q, x_axis, freq



def remove_background(data, data_bg =[]):
    
    ## INPUT : I + 1j*Q 
    ## DO NOT PUT THE dB DATA
    
    n = data.shape[0]
    m = data.shape[1]
    
    
    data_norm = np.zeros((n, m), dtype = 'complex_')
    
    if m == 0:
        for j in range(n):
            data_norm[j, m] = data[j]/data_bg[j]
        
        
    else:
        std_dev = np.std(np.abs(data[0, :]))
        for j in range(n):
            y_avg = data[j,:].mean()
            data_norm[j, :] = data[j, :]/(y_avg)
            #index_max = np.abs(data_norm[j,:]).argmax()
            data_norm[j, :] = data_norm[j, :]/(np.max(np.abs(data_norm[j, :]))) 
        
    return data_norm


def index_zero_detuning(data, x_axis, freq, f_resonator):
    n_smooth = 10
    
    detuning_probe = abs(freq - f_resonator)
    index_f_r = detuning_probe.argmin()
    index_zero_detuning = 0 
    
    
    cut_1d_f_r = data[index_f_r, :]
    # cut_1d_f_r = smooth(cut_1d_f_r, n_smooth)
    
    cut_1d_deriv = deriv(cut_1d_f_r)
    #cut_1d_deriv = smooth(cut_1d_deriv, n_smooth)[0 : -n_smooth-1]
    cut_1d_deriv = smooth(cut_1d_deriv, n_smooth)
    index_zero_detuning = index_zero_detuning + int(round( (cut_1d_deriv.argmax() + 
                                    cut_1d_deriv.argmin()) /2))   
        
    
    plt.plot(cut_1d_f_r)
    plt.plot(cut_1d_deriv)
    plt.axvline(x = index_zero_detuning)

    index_zero_detuning = cut_1d_f_r.argmax()
    
    return index_zero_detuning, index_f_r