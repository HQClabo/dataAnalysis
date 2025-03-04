

import numpy as np
import lmfit
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
            self.power_watts = self.dBm2watts(self.power-attenuation)
            self.fit_idxs = (power_range[0] <= self.power_watts) & (self.power_watts <= power_range[1])
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
        for i in range(len(self.detuning)):
            self.fit_photocurrent_efficiency(freq, attenuation, power_range, cut_idx=i)
            efficiency.append(self.efficiency)
        self.efficiency = np.array(efficiency)
        return self.efficiency
    
    def fit_photocurrent_efficiency_vs_power_and_detuning(self, freq, attenuation, power_fit_range=None):
        efficiency = []
        self.power_watts = self.dBm2watts(self.power-attenuation)
        # get powers within the range
        if power_fit_range is None:
            power_fit_range = [self.power_watts[2], self.power_watts[-1]]
        self.power_fit_array = self.power_watts[(power_fit_range[0] <= self.power_watts) & (self.power_watts <= power_fit_range[1])]
        for power in self.power_fit_array:
            efficiency_temp = []
            for j in range(len(self.detuning)):
                self.fit_photocurrent_efficiency(freq, attenuation, [0,power], cut_idx=j)
                efficiency_temp.append(self.efficiency)
            efficiency.append(efficiency_temp)
        self.efficiency = np.array(efficiency)
        return self.efficiency
    
    def plot_photocurrent_efficiency_vs_detuning(self, title_suffix='', dark_current=True, **kwargs):
        """
        Plot the efficiency and dark current vs. detuning
        """
        # plot efficiency and dark current
        fig, ax = plt.subplots()
        fig.set_size_inches(8/2.54, 6/2.54)
        ax.plot(self.detuning*1e3, self.efficiency, 'o', ms = 2, alpha = 0.7)
        
        effi_ylim = 1.2*np.max(np.abs(self.efficiency))
        ax.set_ylim([-effi_ylim, effi_ylim])
        ax.set_xlabel('Detuning gate voltage (mV)')
        ax.set_ylabel(r'$\eta$ (%)')
        
        ax_r = ax.twinx()
        if dark_current:
            ax_r.plot(self.detuning*1e3, self.Id[0]*1e12, 'o', ms = 2, alpha = 0.7, color = 'tab:orange', **kwargs)
            id_ylim = 1.2*np.max(np.abs(self.Id[0]*1e12))
            ax_r.set_ylim([-id_ylim, id_ylim])
            ax_r.set_ylabel(r'Dark current (pA)')

        fig.suptitle(f'Run #{self.run_id} - Photocurrent efficiency vs. detuning' + title_suffix)
        return fig, ax, ax_r
    
    def plot_photocurrent_efficiency_vs_power(self, cut_idx=None, title_suffix='', **kwargs):
        """
        Plot the efficiency and dark current vs. detuning
        """
        if cut_idx is not None:
            efficiency = self.efficiency[:,cut_idx]
        else:
            efficiency = self.efficiency

        # plot efficiency and dark current
        fig, ax = plt.subplots()
        fig.set_size_inches(8/2.54, 5/2.54)
        title = f'Run #{self.run_id} - $\eta$ vs. Power' + title_suffix
        fig.suptitle(title)
        ax.loglog(self.power_fit_array*1e15, efficiency, marker='o', ls='', lw=0.5, ms=2, alpha=0.7, **kwargs)
        ax.set_xlabel('Power (fW)')
        ax.set_ylabel(r'$|\eta|$ (%)')
        return fig, ax

    def get_max_efficiency_with_detuning(self):
        ind_effi_max = np.argmax(np.abs(self.efficiency))
        return np.max(np.abs(self.efficiency)), self.detuning[ind_effi_max]

    def plot_max_efficiency_fit(self, power_range=None, **kwargs):
        idx_eff_max = np.argmax(np.abs(self.efficiency))

        title = f'Run #{self.run_id} - $\eta$ = {abs(self.efficiency[idx_eff_max]):.2f}%, at detuning {self.detuning[idx_eff_max]*1e3:.2f} mV'
        return self.plot_photocurrent_efficiency_fit(idx_eff_max, title, power_range, **kwargs)
    
    def plot_photocurrent_efficiency_fit(self, idx=None, title='', power_range=None, **kwargs):
        if idx is not None:
            Id = self.Id[:,idx]*1e12
        else:
            Id = self.Id*1e12
        if power_range is not None:
            plot_idxs = (power_range[0] <= self.power_watts) & (self.power_watts <= power_range[1])
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
    
    def watts2dBm(self, power_watts):
        return 10*np.log10(power_watts*1000)
    
    def dBm2watts(self, power_dBm):
        return 10**(power_dBm/10)/1000


def Ip_vs_frequency(fdrive, fr, kappa_tot, A, offset):
    delta_r = fdrive - fr
    Ip = A / (1 + (2*delta_r/kappa_tot)**2) + offset
    return Ip

class KappaDQDFit:
    """
    A class to perform Lorentzian fitting of Id vs frequency to extract Kappa DQD.
    Attributes:
    -----------
    exp : str
        The experiment identifier.
    run_id : int
        The run identifier.
    ds : DataSet
        The dataset object containing the experimental data.
    Id : np.ndarray
        The dependent parameter values (I_d).
    freq : np.ndarray
        The independent parameter values (frequency).
    detuning : np.ndarray
        The independent parameter values (detuning).
    Methods:
    --------
    fit_kappa_DQD_2D(guess_fr=None, guess_kappa_tot=100e6, guess_A=1, do_plots=False, kappa_res=None):
        Fits the Kappa DQD for each detuning value from 2D dataset.
    fit_kappa_DQD_1D(guess_fr=None, guess_kappa_tot=100e6, guess_A=1, do_plots=False, kappa_res=None):
        Fits the Kappa DQD from 1D dataset.
    plot_max_peak_fit():
        Plots the fit with the maximum peak amplitude.
    plot_single_fit(cut_idx):
        Plots a single fit for a given cut index.
    """
    def __init__(self, exp, run_id):
        ds = DataSet(exp=exp, run_id=run_id)
        self.exp = exp
        self.run_id = run_id
        self.ds = ds
        self.Id = ds.get_dependent_parameter_by_name('I_d')['values']
        self.freq = ds.get_independent_parameter_by_name('freq')['values']
        detuning = ds.get_independent_parameter_by_name('detuning')
        if detuning is None:
            self.detuning = None
        else:
            self.detuning = detuning['values']

    def fit_kappa_DQD_1D(self, guess_fr=None, guess_kappa_tot=100e6, guess_A=1, do_plots=False, kappa_res=None):
        """
        Fits the additional loss rate kappa_DQD due to dissipations in the DQD from 1D dataset (Id vs frequency).
        Parameters:
        -----------
        guess_fr : float, optional
            Initial guess for the resonance frequency. If None, the mean of self.freq is used. Default is None.
        guess_kappa_tot : float, optional
            Initial guess for the total kappa. Default is 100e6.
        guess_A : float, optional
            Initial guess for the amplitude. Default is 1.
        do_plots : bool, optional
            If True, plots the fit for each iteration. Default is False.
        kappa_res : float, optional
            Known total resonator linewidth to calculate kappa_DQD. If None, self.kappa_res and self.kappa_DQD are set to None. Default is None.
        Returns:
        --------
        results : list
            List of lmfit.ModelResult objects containing the fit results for each data cut.
        Attributes:
        -----------
        self.fit_reports : list
            List of lmfit.ModelResult objects containing the fit results for each data cut.
        self.fr : numpy.ndarray
            Array of fitted resonance frequencies.
        self.A : numpy.ndarray
            Array of fitted amplitudes.
        self.kappa_tot : numpy.ndarray
            Array of fitted total kappa values.
        self.offset : numpy.ndarray
            Array of fitted offset values.
        self.kappa_res : float or None
            Known kappa_res value or None if not provided.
        self.kappa_DQD : numpy.ndarray or None
            Array of kappa_DQD values or None if kappa_res is not provided.
        """

        if self.detuning is not None:
            raise ValueError('This is not a 1D dataset. Use the fit_kappa_DQD_2D function instead.')

        model_func = Ip_vs_frequency

        if guess_fr is None:
            guess_fr = np.mean(self.freq)
        
        # create the lmfit.Parameters object and adjust some settings
        guess_offset = self.Id[np.argmax(abs(self.Id))]
        params=lmfit.Parameters() # object
        params.add('fr', value=guess_fr, vary=True)
        params.add('kappa_tot', value=guess_kappa_tot, vary=True)
        params.add('A', value=guess_A, vary=True)
        params.add('offset', value=guess_offset, vary=True)
        params['kappa_tot'].min = 0
    
        # fit the data
        model = lmfit.Model(model_func, independent_vars=['fdrive'])
        result = model.fit(self.Id, params, fdrive=self.freq, scale_covar=True)
        if do_plots:
            self.plot_single_fit()
        self.fit_reports = result
        self.fr = result.best_values['fr']
        self.A = result.best_values['A']
        self.kappa_tot = result.best_values['kappa_tot']
        self.offset = result.best_values['offset']
        if kappa_res is None:
            self.kappa_res = None
            self.kappa_DQD = None
        else:
            self.kappa_res = kappa_res
            self.kappa_DQD = self.kappa_tot - kappa_res
        return result

    def fit_kappa_DQD_2D(self, guess_fr=None, guess_kappa_tot=100e6, guess_A=1, do_plots=False, kappa_res=None):
        """
        Fits the additional loss rate kappa_DQD due to dissipations in the DQD from 2D dataset (Id vs. detuning and frequency).
        Parameters:
        -----------
        guess_fr : float, optional
            Initial guess for the resonance frequency. If None, the mean of self.freq is used. Default is None.
        guess_kappa_tot : float, optional
            Initial guess for the total kappa. Default is 100e6.
        guess_A : float, optional
            Initial guess for the amplitude. Default is 1.
        do_plots : bool, optional
            If True, plots the fit for each iteration. Default is False.
        kappa_res : float, optional
            Known total resonator linewidth to calculate kappa_DQD. If None, self.kappa_res and self.kappa_DQD are set to None. Default is None.
        Returns:
        --------
        results : list
            List of lmfit.ModelResult objects containing the fit results for each data cut.
        Attributes:
        -----------
        self.fit_reports : list
            List of lmfit.ModelResult objects containing the fit results for each data cut.
        self.fr : numpy.ndarray
            Array of fitted resonance frequencies.
        self.A : numpy.ndarray
            Array of fitted amplitudes.
        self.kappa_tot : numpy.ndarray
            Array of fitted total kappa values.
        self.offset : numpy.ndarray
            Array of fitted offset values.
        self.kappa_res : float or None
            Known kappa_res value or None if not provided.
        self.kappa_DQD : numpy.ndarray or None
            Array of kappa_DQD values or None if kappa_res is not provided.
        """
        if self.detuning is None:
            raise ValueError('This is not a 2D dataset. Use the fit_kappa_DQD_1D function instead.')

        model_func = Ip_vs_frequency

        if guess_fr is None:
            guess_fr = np.mean(self.freq)
            
        results = []
        fr_list = []
        A_list = []
        kappa_tot_list = []
        offset_list = []
        
        for i, data_cut in enumerate(self.Id.T):
            # create the lmfit.Parameters object and adjust some settings
            guess_offset = data_cut[np.argmax(abs(data_cut))]
            params=lmfit.Parameters() # object
            params.add('fr', value=guess_fr, vary=True)
            params.add('kappa_tot', value=guess_kappa_tot, vary=True)
            params.add('A', value=guess_A, vary=True)
            params.add('offset', value=guess_offset, vary=True)
            params['kappa_tot'].min = 0
        
            # fit the data
            model = lmfit.Model(model_func, independent_vars=['fdrive'])
            result = model.fit(data_cut, params, fdrive=self.freq, scale_covar=True)
            results.append(result)
            fr_list.append(result.best_values['fr'])
            A_list.append(result.best_values['A'])
            kappa_tot_list.append(result.best_values['kappa_tot'])
            offset_list.append(result.best_values['offset'])
            if do_plots:
                self.plot_single_fit(i)
            self.fit_reports = results
            self.fr = np.array(fr_list)
            self.A = np.array(A_list)
            self.kappa_tot = np.array(kappa_tot_list)
            self.offset = np.array(offset_list)
            if kappa_res is None:
                self.kappa_res = None
                self.kappa_DQD = None
            else:
                self.kappa_res = kappa_res
                self.kappa_DQD = self.kappa_tot - kappa_res
        return results

    def plot_max_peak_fit(self):
        """
        Plots the fit of the data with the maximum peak height.

        This method finds the index of the maximum value in the array `A` and 
        calls the `plot_single_fit` method with this index to plot the fit 
        at the maximum peak.

        Returns:
            matplotlib.figure.Figure: The figure object containing the plot.
        """
        return self.plot_single_fit(np.argmax(self.A))
    
    def plot_single_fit(self, cut_idx=None):
        """
        Plots the fit of a single frequency sweep.
        Parameters:
        -----------
        cut_idx : int or None, optional
            Index to select a specific cut of the data. If None, the entire dataset is used (used for 1D datasets).
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object containing the plot.
        Notes:
        ------
        - The plot displays the detuning and the fit of the photodetection data.
        - The title of the plot includes the run ID and the detuning value.
        - The x-axis represents the drive frequency in GHz.
        - The y-axis represents the current Id in pA.
        - The plot includes a text box with the total kappa value and, if available, the DQD kappa and resonator kappa values.
        """
        if cut_idx is None:
            detuning = self.detuning
            Id = self.Id
            fit_report = self.fit_reports
        else:
            detuning = self.detuning[cut_idx]
            Id = self.Id[:, cut_idx]
            fit_report = self.fit_reports[cut_idx]

        fig, ax = plt.subplots()
        fig.suptitle(f'Run #{self.run_id}' + ' - $\kappa_{DQD}$ fit' + f' - detuning = {detuning*1e3:.2f} mV')
        ax.plot(self.freq/1e9, Id*1e12, ls='', marker='.')
        ax.plot(self.freq/1e9, fit_report.best_fit*1e12, ls='-', marker='')
        ax.set_xlabel('$f_{drive}$ (GHz)')
        ax.set_ylabel('$I_d$ (pA)')

        kappa_tot = fit_report.best_values['kappa_tot'] / 1e6
        textstr = '$\kappa_{tot}$ = '+f'{kappa_tot:.2f} MHz' + '\n'
        if self.kappa_res is not None:
            textstr += '$\kappa_{DQD}$ = '+f'{(self.kappa_DQD)*1e3:.2f} MHz' + '\n'
            textstr += '$\kappa$ = '+f'{self.kappa_res*1e3:.2f} MHz'
        
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', color='black')
        return fig, ax

        
