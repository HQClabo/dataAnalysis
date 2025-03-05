

import numpy as np
import lmfit
from matplotlib import pyplot as plt
from dataAnalysis.base import DataSet
from scipy.stats import linregress

class EfficiencyFit(DataSet):
    """
    A class used to fit and analyze the efficiency of photocurrent detection in an experimental dataset.
    Attributes
    ----------
    Id : numpy.ndarray
        The dependent parameter representing the photocurrent values.
    power : numpy.ndarray
        The independent parameter representing the power values.
    detuning : numpy.ndarray
        The independent parameter representing the detuning values.
    freq : float
        The frequency of the light used in the experiment.
    attenuation : float
        The attenuation factor applied to the power values.
    power_range : list or None
        The range of power values to consider for fitting.
    fit_idxs : numpy.ndarray
        The indices of the power values that fall within the specified power range.
    efficiency : numpy.ndarray
        The calculated efficiency of the photocurrent detection.
    power_fit_array : numpy.ndarray
        The array of power values within the specified power range for fitting.
    Methods
    -------
    fit_photocurrent_efficiency(freq, attenuation, power_range=None, cut_idx=None)
        Fits the photocurrent efficiency for a given frequency and attenuation.
    fit_photocurrent_efficiency_vs_detuning(freq, attenuation, power_range=None)
        Fits the photocurrent efficiency as a function of detuning for a given frequency and attenuation.
    fit_photocurrent_efficiency_vs_power_and_detuning(freq, attenuation, power_fit_range=None)
        Fits the photocurrent efficiency as a function of both power and detuning for a given frequency and attenuation.
    plot_photocurrent_efficiency_vs_detuning(title_suffix='', dark_current=True, **kwargs)
        Plots the photocurrent efficiency and dark current as a function of detuning.
    plot_photocurrent_efficiency_vs_power(cut_idx=None, title_suffix='', **kwargs)
        Plots the photocurrent efficiency as a function of power.
    get_max_efficiency_with_detuning()
        Returns the maximum efficiency and the corresponding detuning value.
    plot_max_efficiency_fit(power_range=None, **kwargs)
        Plots the fit of the photocurrent efficiency at the maximum efficiency point.
    plot_photocurrent_efficiency_fit(idx=None, title='', power_range=None, **kwargs)
        Plots the fit of the photocurrent efficiency for a specified detuning index.
    watts2dBm(power_watts)
        Converts power from watts to dBm.
    dBm2watts(power_dBm)
        Converts power from dBm to watts.
        """
    def __init__(self, exp, run_id):
        super().__init__(exp=exp, run_id=run_id)
        self.Id = self.get_dependent_parameter_by_name('I_d')['values']
        self.power = self.get_independent_parameter_by_name('power')['values']
        self.detuning = self.get_independent_parameter_by_name('detuning')['values']

    def fit_photocurrent_efficiency(self, freq, attenuation, power_range=None, cut_idx=None):
        """
        Fits the photocurrent efficiency based on the provided frequency, attenuation, and power range.

        Parameters:
            freq (float): The frequency of the incident light in Hz.
            attenuation (float): The attenuation factor in dB.
            power_range (tuple, optional): A tuple specifying the range of power in watts to consider for fitting. Defaults to None.
            cut_idx (int, optional): Index to cut the current data for fitting. Defaults to None.
        Returns:
            float: The calculated efficiency of the photocurrent in percentage.
        """
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
        """
        Fits the photocurrent efficiency as a function of detuning.

        This method iterates over the detuning values and fits the photocurrent efficiency
        for each detuning value. The results are stored in an array and returned.

        Parameters:
            freq (float): The frequency at which the measurement is taken.
            attenuation (float): The attenuation applied during the measurement.
            power_range (tuple, optional): The range of power values to consider for the fit. 
                                       If None, the full range is used.

        Returns:
            np.ndarray: An array containing the fitted photocurrent efficiency for each detuning value.
        """
        efficiency = []
        for i in range(len(self.detuning)):
            self.fit_photocurrent_efficiency(freq, attenuation, power_range, cut_idx=i)
            efficiency.append(self.efficiency)
        self.efficiency = np.array(efficiency)
        return self.efficiency
    
    def fit_photocurrent_efficiency_vs_power_and_detuning(self, freq, attenuation, power_fit_range=None):
        """
        Fits the photocurrent efficiency as a function of power and detuning.

        This method calculates the photocurrent efficiency for a range of power levels
        and detuning values. It first converts the power from dBm to watts, then filters
        the power levels within the specified range. For each power level, it computes
        the efficiency for each detuning value and stores the results in an array.

        Parameters:
            freq (float): The frequency at which the efficiency is measured.
            attenuation (float): The attenuation value to be subtracted from the power.
            power_fit_range (list, optional): A list containing the lower and upper bounds
                                            of the power range to fit. If None, the range
                                            is set from the third power value to the last
                                            power value in the power array.

        Returns:
            np.ndarray: A 2D array of efficiencies, where each row corresponds to a power
                        level and each column corresponds to a detuning value.
        """
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
        Plot the photocurrent efficiency and dark current vs. detuning.

        Parameters:
            title_suffix (str, optional): A suffix to add to the plot title. Default is an empty string.
            dark_current (bool, optional): If True, plot the dark current on the secondary y-axis. Default is True.
            **kwargs (dict): Additional keyword arguments to pass to the dark current plot.
        Returns:
            fig (matplotlib.figure.Figure): The figure object containing the plot.
            ax (matplotlib.axes._subplots.AxesSubplot): The primary axes object for the efficiency plot.
            ax_r (matplotlib.axes._subplots.AxesSubplot): The secondary axes object for the dark current plot (if `dark_current` is True).
        Notes:
            - The x-axis represents the detuning gate voltage in millivolts (mV).
            - The primary y-axis represents the photocurrent efficiency (η) in percentage (%).
            - The secondary y-axis represents the dark current in picoamperes (pA) if `dark_current` is True.
        """
        # plot efficiency and dark current
        fig, ax = plt.subplots()
        ax.plot(self.detuning*1e3, self.efficiency, marker='.', ls='', alpha=0.7)
        
        effi_ylim = 1.2*np.max(np.abs(self.efficiency))
        ax.set_ylim([-effi_ylim, effi_ylim])
        ax.set_xlabel('Detuning gate voltage (mV)')
        ax.set_ylabel(r'$\eta$ (%)')
        
        ax_r = ax.twinx()
        if dark_current:
            ax_r.plot(self.detuning*1e3, self.Id[0]*1e12, marker='.', ls='', alpha=0.7, color = 'tab:orange', **kwargs)
            id_ylim = 1.2*np.max(np.abs(self.Id[0]*1e12))
            ax_r.set_ylim([-id_ylim, id_ylim])
            ax_r.set_ylabel(r'Dark current (pA)')

        fig.suptitle(f'Run #{self.run_id} - Photocurrent efficiency vs. detuning' + title_suffix)
        if dark_current:
            return fig, ax, ax_r
        else:
            return fig, ax
    
    def plot_photocurrent_efficiency_vs_power(self, cut_idx=None, title_suffix='', **kwargs):
        """
        Plots the photocurrent efficiency versus power.

        Parameters:
            cut_idx (int, optional): Index to cut the efficiency array. If None, the entire efficiency array is used.
            title_suffix (str, optional): Suffix to add to the plot title.
            **kwargs: Additional keyword arguments to pass to the plot function.
        Returns:
            fig (matplotlib.figure.Figure): The figure object containing the plot.
            ax (matplotlib.axes._subplots.AxesSubplot): The axes object containing the plot.
        """
        if cut_idx is not None:
            efficiency = self.efficiency[:,cut_idx]
        else:
            efficiency = self.efficiency

        # plot efficiency and dark current
        fig, ax = plt.subplots()
        title = f'Run #{self.run_id} - $\eta$ vs. Power' + title_suffix
        fig.suptitle(title)
        ax.loglog(self.power_fit_array*1e15, efficiency, marker='.', ls='', alpha=0.7, **kwargs)
        ax.set_xlabel('Power (fW)')
        ax.set_ylabel(r'$|\eta|$ (%)')
        return fig, ax

    def get_max_efficiency_with_detuning(self):
        """
        Calculate the maximum efficiency and its corresponding detuning value.

        This method finds the maximum value of the absolute efficiency and returns
        both this maximum efficiency and the detuning value at which it occurs.

        Returns:
            tuple: A tuple containing:
                - float: The maximum efficiency value.
                - float: The detuning value corresponding to the maximum efficiency.
        """
        ind_effi_max = np.argmax(np.abs(self.efficiency))
        return np.max(np.abs(self.efficiency)), self.detuning[ind_effi_max]

    def plot_max_efficiency_fit(self, power_range=None, **kwargs):
        """
        Plots the maximum efficiency fit for the photodetection data.

        This method identifies the index of the maximum efficiency in the efficiency array,
        constructs a title for the plot including the run ID, maximum efficiency value, and
        the corresponding detuning value, and then calls the plot_photocurrent_efficiency_fit
        method to generate the plot.
        
        Parameters:
            power_range (tuple, optional): A tuple specifying the range of power values to be considered for the plot.
            **kwargs: Additional keyword arguments to be passed to the plot_photocurrent_efficiency_fit method.
        Returns:
            matplotlib.figure.Figure: The figure object containing the plot.
        """
        idx_eff_max = np.argmax(np.abs(self.efficiency))

        title = f'Run #{self.run_id} - $\eta$ = {abs(self.efficiency[idx_eff_max]):.2f}%, at detuning {self.detuning[idx_eff_max]*1e3:.2f} mV'
        return self.plot_photocurrent_efficiency_fit(idx_eff_max, title, power_range, **kwargs)
    
    def plot_photocurrent_efficiency_fit(self, idx=None, title='', power_range=None, **kwargs):
        """
        Plots the photocurrent efficiency fit.

        Parameters:
            idx (int, optional): Index to select a specific column of Id. If None, all columns are used.
            title (str, optional): Title of the plot.
            power_range (tuple, optional): Tuple specifying the range of power values to plot (min, max). If None, all power values are used.
            **kwargs: Additional keyword arguments to pass to the plot function.
        Returns:
            fig (matplotlib.figure.Figure): The figure object containing the plot.
            ax (matplotlib.axes._subplots.AxesSubplot): The axes object containing the plot.
        """
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
        ax.plot(self.power_watts[plot_idxs]*1e15, Id[plot_idxs], marker='.', ls='', alpha=0.7, **kwargs) 
        ax.plot(self.power_watts[self.fit_idxs]*1e15, slope_fit * self.power_watts[self.fit_idxs] + intercept_fit, ls = '-', alpha = 0.8, **kwargs)
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

class KappaDQDFit(DataSet):
    """
    A class to perform Lorentzian fitting of Id vs frequency to extract Kappa DQD.
    
    Attributes:
    -----------
    exp : str
        The experiment identifier.
    run_id : int
        The run identifier.
    Id : np.ndarray
        The dependent parameter values (I_d).
    power : np.ndarray
        The independent parameter values (power).
    detuning : np.ndarray
        The independent parameter values (detuning).
    fit_reports : list
        List of lmfit.ModelResult objects containing the fit results for each data cut.
    fr : numpy.ndarray
        Array of fitted resonance frequencies.
    A : numpy.ndarray
        Array of fitted amplitudes.
    kappa_tot : numpy.ndarray
        Array of fitted total kappa values.
    offset : numpy.ndarray
        Array of fitted offset values.
    kappa_res : float or None
        Known kappa_res value or None if not provided.
    kappa_DQD : numpy.ndarray or None
        Array of kappa_DQD values or None if kappa_res is not provided.

    Methods:
    --------
    fit_kappa_DQD_1D(guess_fr=None, guess_kappa_tot=100e6, guess_A=1, do_plots=False, kappa_res=None):
        Fits the Kappa DQD from 1D dataset.
    fit_kappa_DQD_2D(guess_fr=None, guess_kappa_tot=100e6, guess_A=1, do_plots=False, kappa_res=None):
        Fits the Kappa DQD for each detuning value from 2D dataset.
    plot_max_peak_fit():
        Plots the fit with the maximum peak amplitude.
    plot_single_fit(cut_idx=None, fit_report=None):
        Plots a single fit for a given cut index.
    """
    def __init__(self, exp, run_id):
        super().__init__(exp=exp, run_id=run_id)
        self.Id = self.get_dependent_parameter_by_name('I_d')['values']
        self.power = self.get_independent_parameter_by_name('power')['values']
        detuning = self.get_independent_parameter_by_name('detuning')['values']
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

            if do_plots:
                self.plot_single_fit(i, fit_report=result)
        return results

    def plot_max_peak_fit(self):
        """
        Plots the fit of the data with the maximum peak height. Ignores fits where the error of A is larger than its value.

        This method finds the index of the maximum value in the array `A` and 
        calls the `plot_single_fit` method with this index to plot the fit 
        at the maximum peak.

        Returns:
            matplotlib.figure.Figure: The figure object containing the plot.
        """
        A_errs = [report.params['A'].stderr for report in self.fit_reports]
        idx_max = np.argmax(self.A[abs(self.A)>A_errs])
        return self.plot_single_fit(idx_max)
    
    def plot_single_fit(self, cut_idx=None, fit_report=None):
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
        if fit_report is None:
            if cut_idx is None:
                fit_report = self.fit_reports
            else:
                fit_report = self.fit_reports[cut_idx]
                
        if cut_idx is None:
            detuning = self.detuning
            Id = self.Id
            kappa_DQD = self.kappa_DQD
        else:
            detuning = self.detuning[cut_idx]
            Id = self.Id[:, cut_idx]
            if self.kappa_res is not None:
                kappa_DQD = self.kappa_DQD[cut_idx]

        fig, ax = plt.subplots()
        fig.suptitle(f'Run #{self.run_id}' + ' - $\kappa_{DQD}$ fit' + f' - detuning = {detuning*1e3:.2f} mV')
        ax.plot(self.freq/1e9, Id*1e12, ls='', marker='.')
        ax.plot(self.freq/1e9, fit_report.best_fit*1e12, ls='-', marker='')
        ax.set_xlabel('$f_{drive}$ (GHz)')
        ax.set_ylabel('$I_d$ (pA)')

        kappa_tot = fit_report.best_values['kappa_tot'] / 1e6
        textstr = '$\kappa_{tot}$ = '+f'{kappa_tot:.2f} MHz' + '\n'
        if self.kappa_res is not None:
            textstr += '$\kappa_{DQD}$ = '+f'{(kappa_DQD) / 1e6:.2f} MHz' + '\n'
            textstr += '$\kappa$ = '+f'{self.kappa_res / 1e6:.2f} MHz'
        
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', color='black')
        return fig, ax

        
