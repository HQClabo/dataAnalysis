# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 13:43:03 2022

@author: fopplige
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from resonator_tools import circuit

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

def phase_unwrap(freq,cData,fit_range=0.05):
    length = len(freq)
    unwr_phase = np.unwrap(np.angle(cData))
    delay1,_ = np.polyfit(freq[0:int(fit_range*length)],unwr_phase[0:int(fit_range*length)],1)
    delay2,_ = np.polyfit(freq[int((1-fit_range)*length):-1],unwr_phase[int((1-fit_range)*length):-1],1)
    cData_rot = cData*np.exp(-1j*(delay1+delay2)/2*freq)
    cData_rot_zero = cData_rot*np.exp(-1j*np.angle(cData_rot[0]))
    return np.angle(cData_rot_zero)

def plot_ampl_and_phase(freq,cData,label='',filename=False,file_res=300):
    fig, ax = plt.subplots(2)
    fig.dpi = file_res
    fig.set_figheight(5)
    fig.set_figwidth(9)
    ax[0].plot(freq,20*np.log10(np.abs(cData)),label='Amplitude (dB)')
    ax[1].plot(freq,phase_unwrap(freq,cData)/np.pi,label=r'Phase ($\pi$)')
    ax[0].set_ylabel('Amplitude (dB)')
    #ax[0].set_xticklabels([])
    ax[1].set_ylabel(r'Phase ($\pi$)')
    ax[1].set_xlabel('Frequency (Hz)')
    fig.suptitle(label)
    ax[0].grid()
    ax[1].grid()
    fig.tight_layout()
    if filename:
        plt.savefig(filename,dpi=file_res)

def plot_ampl(freq,cData,label='',filename=False,file_res=300):
    fig, ax = plt.subplots(1)
    fig.dpi = file_res
    fig.set_figheight(6)
    fig.set_figwidth(15)
    if len(freq.shape) == 2:
        for ii in range(freq.shape[0]):
            ax.plot(freq[ii],20*np.log10(np.abs(cData[ii])),label='Amplitude (dB)')
    else:
        ax.plot(freq,20*np.log10(np.abs(cData)),label='Amplitude (dB)')
    ax.set_ylabel('Amplitude (dB)')
    ax.set_xlabel('Frequency (Hz)')
    fig.suptitle(label)
    ax.grid()
    fig.tight_layout()
    if filename:
        plt.savefig(filename,dpi=file_res)

def plot_peak_vs_power(freq,cData,power,label=''):
    for ii in range(len(power)):
        plt.plot(freq,np.abs(cData)[ii],label=f'power = {power[ii]}')
        #plt.xlim(4.35e9,4.425e9)
        # plt.plot(FREQ, np.angle(cData))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title(f'Resonator {label}')
    plt.legend()
    plt.tight_layout()
    #plt.savefig(f'Modulo_VS_frequency_res_{label}.png',dpi=600)
    plt.show()

def plot2D_power_VS_freq(out,freq,cData,power,p_in=0,label='0'):
    po=[]
    for ii in range(len(out.power_at)):
        po.append(out.power_at[ii]+80)

    fig,ax = plt.subplots()
    im = ax.imshow(np.abs(cData),cmap=plt.cm.Reds,aspect='auto', extent=[freq[0],freq[-1],power[-1],power[0]])
    ax.plot(out.fr_res[p_in:],po[p_in:],'o-',label='peak 1')
        #im = plt.pcolormesh(mod, cmap='RdBu',shading='gouraud')
        #cset = ax.contour(mod,np.arange(0,1,0.2),linewidths=2,cmap=plt.cm.Set2)
        #ax.clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
    ax.set_title(f'Resonator {label}')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Modulo (dB)')
    ax.grid(True,alpha=0.5)
    fig.colorbar(im)
    fig.tight_layout()
    #fig.savefig(f'Plot2D_power_VS_frequency_Res_{label}.png',dpi=600)

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
        
