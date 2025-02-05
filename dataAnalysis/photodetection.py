

import numpy as np
from matplotlib import pyplot as plt
from dataAnalysis.base import DataSet
from scipy.stats import linregress

class EfficiencyFit(DataSet):
    def __init__(self, exp, run_id):
        super().__init__(exp=exp, run_id=run_id)
        self.Id = self.get_dependent_parameter_by_name('I_d')['values']
        self.power = self.get_independent_parameter_by_name('power')['values']
        self.detuning = self.get_independent_parameter_by_name('detuning')['values']

    def fit_photocurrent_efficiency(self, freq, attenuation, power_range=None, cut_idx=None):
        self.freq = freq
        self.attenuation = attenuation
        self.power_range = power_range
        if power_range is not None:
            self.power_watts = 10**((self.power-attenuation)/10)/1000
            self.fit_idxs = (self.power_watts > power_range[0]) & (self.power_watts < power_range[1])
        else:
            self.fit_idxs = [True]*len(self.power)

        current_to_fit = self.Id[self.fit_idxs]
        if cut_idx is not None:
            current_to_fit = current_to_fit[:,cut_idx]
        
        e = 1.602*1e-19 # electron charge
        h = 6.626*1e-34 # Planck const.
        
        results_lin_low = linregress(self.power_watts[self.fit_idxs], current_to_fit)
        slope_fit_low = results_lin_low.slope
        self.efficiency = slope_fit_low*(h*freq)/e*100
        return self.efficiency

    def fit_photocurrent_efficiency_vs_detuning(self, freq, attenuation, power_range=None):
        efficiency = []
        for i in range(self.Id.shape[1]):
            self.fit_photocurrent_efficiency(freq, attenuation, power_range, cut_idx=i)
            efficiency.append(self.efficiency)
        self.efficiency = np.array(efficiency)
        return self.efficiency
    
    def plot_photocurrent_efficiency_vs_detuning(self, title_suffix='', dark_current=True):
        """
        Plot the efficiency and dark current vs. detuning
        """
        # plot efficiency and dark current
        fig, ax = plt.subplots()
        fig.set_size_inches(8/2.54, 6/2.54)
        ax = plt.gca()
        ax.plot(self.detuning*1e3, self.efficiency, 'o', ms = 2, alpha = 0.7)
        
        effi_ylim = 1.2*np.max(np.abs(self.efficiency))
        ax.set_ylim([-effi_ylim, effi_ylim])
        ax.set_xlabel('Detuning gate voltage (mV)')
        ax.set_ylabel(r'$\eta$ (%)')
        
        ax_r = ax.twinx()
        if dark_current:
            ax_r.plot(self.detuning*1e3, self.Id[0]*1e12, 'o', ms = 2, alpha = 0.7, color = 'tab:orange')
            id_ylim = 1.2*np.max(np.abs(self.Id[0]*1e12))
            ax_r.set_ylim([-id_ylim, id_ylim])
            ax_r.set_ylabel(r'Dark current (pA)')

        fig.suptitle(f'Run #{self.run_id} - Photocurrent efficiency vs. detuning' + title_suffix)
        return fig, ax, ax_r

    def get_max_efficiency_with_detuning(self):
        ind_effi_max = np.argmax(np.abs(self.efficiency))
        return np.max(np.abs(self.efficiency)), self.detuning[ind_effi_max]

    def plot_max_efficiency_fit(self, power_range=None, **kwargs):
        idx_eff_max = np.argmax(np.abs(self.efficiency))
        self.power_watts = 10**((self.power-self.attenuation)/10)/1000

        title = f'Run #{self.run_id} - $\eta$ = {abs(self.efficiency[idx_eff_max]):.2f}%, at detuning {self.detuning[idx_eff_max]*1e3:.2f} mV'
        return self.plot_photocurrent_efficiency_fit(idx_eff_max, title, power_range, **kwargs)
    
    def plot_photocurrent_efficiency_fit(self, idx=None, title='', power_range=None, **kwargs):
        if idx is not None:
            Id = self.Id[:,idx]*1e12
        else:
            Id = self.Id*1e12
        if power_range is not None:
            plot_idxs = (self.power_watts > power_range[0]) & (self.power_watts < power_range[1])
        else:
            plot_idxs = [True]*len(self.power)
        
        results_lin_low = linregress(self.power_watts[self.fit_idxs], Id[self.fit_idxs])
        slope_fit = results_lin_low.slope
        intercept_fit = results_lin_low.intercept

        fig, ax = plt.subplots()
        fig.set_size_inches(8/2.54, 6/2.54)
        ax.plot(self.power_watts[plot_idxs]*1e15, Id[plot_idxs], 'o', ms = 2, alpha = 0.6, **kwargs) 
        ax.plot(self.power_watts[self.fit_idxs]*1e15, slope_fit * self.power_watts[self.fit_idxs] + intercept_fit, 
                lw = 1, ls = '-', alpha = 0.8, **kwargs)
        ax.set_xlabel('Input Power (fW)')
        ax.set_ylabel(r'$\mathrm{I}_d$ (pA)') 
        fig.suptitle(title)
        return fig, ax