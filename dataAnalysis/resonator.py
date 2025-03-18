

import numpy as np
import matplotlib.pyplot as plt
import qcodes as qc
from resonator_tools import circuit
import lmfit
from .base import DataSet
import dataAnalysis.resonator_fitting as resfit

def S21_resonator_notch(fdrive, fr, kappa_int, kappa_ext, a, alpha, delay, phi0):
        delta_r = fdrive - fr
        S21 = (delta_r - 1j/2*(kappa_int + kappa_ext*(1-np.exp(1j*phi0)))) / (delta_r - 1j/2*(kappa_ext + kappa_int))
        environment = a * np.exp(1j*(alpha - delay*2*np.pi*fdrive))
        return S21 * environment

def S11_resonator_reflection(fdrive, fr, kappa_int, kappa_ext, a, alpha, delay):
    delta_r = fdrive - fr
    S11 = (delta_r + 1j/2*(kappa_ext - kappa_int)) / (delta_r - 1j/2*(kappa_ext + kappa_int))
    environment = a * np.exp(1j*(alpha - delay*2*np.pi*fdrive))
    return S11 * environment


class DataSetVNA(DataSet):
    """
    A class representing a dataset from a Vector Network Analyzer (VNA) measurement. See DataSet class in base.py for more information.

    Args:
        exp (qcodes.dataset.experiment_container.Experiment, optional): The qcodes Experiment object.
        run_id (str, optional): The ID of the measurement run. If not provided, the last recorded run ID in exp will be used.
        station (str, optional): The name of the station. Defaults to None.
        freq_range (tuple, optional): The frequency range to extract data from. Defaults to None.

    Attributes:
        name_mag (str): The name of the magnitude parameter.
        name_phase (str): The name of the phase parameter.
        name_freq (str): The name of the frequency parameter.
        freq (ndarray): The frequency values.
        mag (ndarray): The magnitude values.
        phase (ndarray): The phase values.
        cData (ndarray): The complex data calculated from magnitude and phase.

    Methods:
        _extract_data_vna_base(): Extracts data from a VNA measurement.
        normalize_data_vna(run_id_bg, axis=0, interpolate=True): Normalizes the data using a background measurement.

    Examples:
        # Create a DataSetVNA object
        ds = DataSetVNA(exp=exp, run_id='123')

        # Normalize the data using a background measurement run (must be a single frequency sweep)
        ds.normalize_data_vna(run_id_bg='456', axis=0, interpolate=True)

        or

        # Normalize the data using a arbitrary arrays for cData and frequency
        cData = np.array([1+1j, 2+2j, 3+3j])
        freq = np.array([1, 2, 3])
        ds.normalize_data_vna(cData_bg=cData, freq_bg=freq, axis=0, interpolate=True)
    """

    def __init__(self, exp, run_id=None, station=None):
        super().__init__(exp=exp, run_id=run_id, station=station)
        self._extract_data_vna_base()

    def _extract_data_vna_base(self):
        """
        Extracts data from a VNA (Vector Network Analyzer) measurement.

        Generate attributes self.freq, self.mag, self.phase, self.cData from the paramspecs.

        Returns:
            None
        """
        # Find param names for mag, phase and freq
        mag = self.get_dependent_parameter_by_name('magnitude')
        phase = self.get_dependent_parameter_by_name('phase')
        freq = self.get_independent_parameter_by_name('frequency')

        if freq is None or mag is None or phase is None:
            raise ValueError("The dataset does not contain the required parameters for a VNA measurement.")
        
        self.name_mag = mag['name']
        mag_unit = phase['paramspec'].unit
        self.name_phase = phase['name']
        phase_unit = phase['paramspec'].unit
        self.name_freq = freq['name']

        # Convert phase to radians always
        if phase_unit == 'rad':
            pass
        elif phase_unit == '°':
            self.dependent_parameters[self.name_phase]['values'] *= np.pi/180
        else:
            raise ValueError(f'The phase \"{phase_unit}\" was not recognized.')
        # Convert mag to dB always
        if not mag_unit == 'dB':
            self.dependent_parameters[self.name_mag]['values'] = 20*np.log10(abs(self.dependent_parameters[self.name_mag]['values']))

        # Generate attributes
        self.freq = freq['values']
        self.mag = mag['values']
        self.phase = phase['values']
        self.cData = 10**(self.mag/20) * np.exp(1j*self.phase)

    def normalize_data_vna(self, run_id_bg: int=None, cData_bg: np.array=None, freq_bg: np.array=None, axis: int=0, interpolate=True):
        """
        Normalize the measurement data using a background measurement run or provided background data.
        This function normalizes the measurement data using either a 1D background measurement run identified by `run_id_bg` 
        or provided background data (`cData_bg` and `freq_bg`). It assumes that the background run ID belongs to the same 
        experiment as the data run ID.

        Args:
            run_id_bg (int, optional): ID of the background measurement run. Default is None.
            cData_bg (np.array, optional): Complex background data. Default is None.
            freq_bg (np.array, optional): Frequency data corresponding to the background data. Default is None.
            axis (int, optional): Axis along which to normalize. Default is 0.
            interpolate (bool, optional): Flag to interpolate the background trace if the number of points do not match 
                                          with the measurement data. Default is True.
        Raises:
            ValueError: If neither `run_id_bg` nor both `cData_bg` and `freq_bg` are provided.
        """
        if run_id_bg is not None:
            ds_bg = DataSetVNA(self.exp, run_id_bg)
            freq_bg = ds_bg.freq
            cData_bg = ds_bg.cData
        elif cData_bg is None or freq_bg is None:
            raise ValueError("Either run_id_bg or cData_bg and freq_bg must be provided.")
        
        self.normalize_data_mag_phase(self.name_mag, self.name_phase, cData_bg, freq_bg, axis=axis, interpolate=interpolate)
        self.mag_norm = self.dependent_parameters[self.name_mag+'_normalized']['values']
        self.phase_norm = self.dependent_parameters[self.name_phase+'_normalized']['values']
        self.cData_norm = 10**(self.mag_norm/20) * np.exp(1j*self.phase_norm)

    def plot_1D_normalized(self):
        return self.plot_1D([self.name_mag+'_normalized', self.name_phase+'_normalized'])

    def plot_2D_normalized(self):
         return self.plot_2D([self.name_mag+'_normalized', self.name_phase+'_normalized'])


class FrequencyScanVNA(DataSetVNA):
    """
    Class for 1D VNA frequency sweeps.

    Args:
        exp: Experiment.
        run_id (optional): Run ID of the measurement. If not provided, the last measurement run is used.
        station (optional): Station. 
        freq_range (optional): Tuple with the min and max frequencies of the range to use.
    """
    def __init__(self, exp, run_id=None, station=None, freq_range:tuple=None):
        # This will already extract self.freq, self.mag, self.phase
        super().__init__(exp=exp, run_id=run_id, station=station)
        # Cut data with provided frequency range
        self.extract_data_vna(freq_range)

    def extract_data_vna(self, freq_range:tuple=None):
        """
        Extracts data from a VNA measurement.

        Args:
            freq_range (tuple, optional): Frequency range to extract data from. Defaults to None.
            
        Returns:
            None
        """
        if freq_range:
            freq_slice = self.find_slice(self.freq,freq_range)
            self.freq = self.freq[freq_slice]
            self.mag = self.mag[freq_slice]
            self.phase = self.phase[freq_slice]
            self.cData = 10**(self.mag/20) * np.exp(1j*self.phase)

    def analyze(self, freq_range=None, power=-140, port_type='notch', normalized=True, method='resonator_tools', freq_unit='Hz', do_plots=True):
        """
        Perform a resonator fit of the data using the specified using resonator_tools. The results can be found in self.fit_report.

        Args:
            freq_range (tuple, optional): Frequency range to consider for analysis. Defaults to None.
            power (float, optional): Input power in dBm that arrives at the coupling port. Defaults to -140 dBm.
            port_type (str, optional): Type of port to use for analysis. Supported values are 'notch' or 'reflection'. Defaults to 'notch'.
            normalized (bool, optional): Flag indicating whether to use normalized data. Defaults to True.
            method (str, optional): Method to use for fitting. Supported values are 'resonator_tools' or 'lmfit'. Defaults to 'resonator_tools'.
            do_plots (bool, optional): Flag indicating whether to generate plots. Defaults to True.

        Returns:
            None

        Attributes:
            Dictionary containing the results of the fit.
        """
        cData = self.cData
        if normalized:
            try:
                cData = self.cData_norm
            except AttributeError:
                raise Warning("Normalized data not found. Using raw data instead.")
        freq_scaling = resfit.get_frequency_scaling(freq_unit)
        self.fit_report = resfit.fit_frequency_sweep(cData.T, self.freq/freq_scaling, freq_range, power, port_type, method, freq_unit, do_plots)


class PowerScanVNA(DataSetVNA):
    def __init__(self, exp, run_id=None, station=None, freq_range=None, power_range=None):
        super().__init__(exp=exp, run_id=run_id, station=station)
        self.extract_data_vna(freq_range, power_range)

    def extract_data_vna(self, freq_range=None, power_range=None):
        """
        Extracts data from a VNA measurement.

        Args:
            freq_range (tuple, optional): Frequency range to extract data from. Defaults to None.
            power_range (tuple, optional): Power range to extract data from. Defaults to None.
            
        Returns:
            None
        """
        for key, param in self.independent_parameters.items():
            if 'power' in param['paramspec'].name: 
                self.name_power = key
        self.power = self.independent_parameters[self.name_power]['values']

        # Select only the provided frequency and power range
        freq_slice = slice(0,None)
        power_slice = slice(0,None)
        if freq_range: 
            freq_slice = self.find_slice(self.freq, freq_range)
        if power_range: 
            power_slice = self.find_slice(self.power, power_range)
        self.freq = self.freq[freq_slice]
        self.power = self.power[power_slice]
        if self.name_freq == 'x':
            slice_2d = (power_slice, freq_slice)
        else:
            slice_2d = (freq_slice, power_slice)
        self.mag = self.mag[slice_2d]
        self.phase = self.phase[slice_2d]
        self.cData = self.cData[slice_2d]

    def analyze(self, freq_range=None, power_range=None, attenuation=0, port_type='notch', normalized=True, method='resonator_tools', freq_unit='Hz', do_plots=True):
        """
        Perform a resonator fit of the data using the specified using resonator_tools. The results can be found in self.fit_report.

        Args:
            freq_range (tuple, optional): Frequency range to consider for analysis. Defaults to None.
            power_range (tuple, optional): Power range to consider for analysis. Defaults to None.
            attenuation (float, optional): Expected attenuation in dB between the instrument output and the resonator port. Defaults to 0.
            port_type (str, optional): Type of port to use for analysis. Supported values are 'notch' or 'reflection'. Defaults to 'notch'.
            normalized (bool, optional): Flag indicating whether to use normalized data. Defaults to True.
            method (str, optional): Method to use for fitting. Supported values are 'resonator_tools' or 'lmfit'. Defaults to 'resonator_tools'.
            do_plots (bool, optional): Flag indicating whether to generate plots. Defaults to True.

        Returns:
            None
        
        Attributes:
            Dictionary containing the results of the fit as np.arrays for each parameter.
        """
        cData = self.cData
        if normalized:
            try:
                cData = self.cData_norm
            except AttributeError:
                print("Warning: Normalized data not found. Using raw data instead.")
        freq_scaling = resfit.get_frequency_scaling(freq_unit)
        self.fit_report = resfit.fit_power_sweep(cData.T, self.freq/freq_scaling, self.power, freq_range, power_range, attenuation, port_type, method, freq_unit, do_plots)
    
    def normalize_data_from_index(self, idx=-1, axis=0):
        """
        Normalize magnitude and phase with a line cut at the specified index. Default idx is -1, usually corresponding to the highest power trace.
        """
        self.normalize_data(self.name_mag, self.mag[:,idx], self.freq, axis=axis, interpolate=False)
        self.normalize_data(self.name_phase, self.phase[:,idx], self.freq, axis=axis, interpolate=False)
        self.mag_norm = self.dependent_parameters[self.name_mag+'_normalized']['values']
        self.phase_norm = self.dependent_parameters[self.name_phase+'_normalized']['values']
        self.cData_norm = 10**(self.mag_norm/20) * np.exp(1j*self.phase_norm)

    def plot_QvsP(self,label='',log_y=True,threshold=None,**kwargs):
        """
        Plots the quality factors (Qi, Qc, Ql) versus photon number (Nph) with error bars.

        Parameters:
        -----------
        label : str, optional
            Title of the plot. Default is an empty string.
        log_y : bool, optional
            If True, use a logarithmic scale for the y-axis. Default is True.
        threshold : float, optional
            Threshold factor for the discarding bad fits. This factor is used to filter out the fits with large errors
            using the conditions threshold*Q < Q_err for all quality factors. If None, no fit is discarded. Default is None.
        **kwargs : dict, optional
            Additional keyword arguments passed to the errorbar function.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object containing the plot.
        """
        return resfit.plot_QvsP(self.fit_report, label=label, log_y=log_y, threshold=threshold, **kwargs)

    def plot_kappavsP(self,label='',log_y=True,threshold=None,freq_unit='GHz',**kwargs):
        """
        Plots the quality factors (kappa_i, kappa_c, kappa_l) versus photon number (Nph) with error bars.

        Parameters:
        -----------
        label : str, optional
            Title of the plot. Default is an empty string.
        log_y : bool, optional
            If True, use a logarithmic scale for the y-axis. Default is True.
        threshold : float, optional
            Threshold factor for the discarding bad fits. This factor is used to filter out the fits with large errors
            using the conditions threshold*Q < Q_err for all quality factors. If None, no fit is discarded. Default is None.
        freq_unit : str, optional
            Unit in which frequency is provided. This will determine the proper scaling factors. Default is 'Hz'.
        **kwargs : dict, optional
            Additional keyword arguments passed to the errorbar function.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object containing the plot.
        """
        return resfit.plot_kappavsP(self.fit_report, label=label, log_y=log_y, threshold=threshold, freq_unit=freq_unit, **kwargs)


class BScanVNA(DataSetVNA):
    def __init__(self, exp, run_id=None, station=None, field_param_name='mag', freq_range=None, field_range=None):
        super().__init__(exp=exp, run_id=run_id, station=station)
        self.extract_data_vna(field_param_name, freq_range, field_range)

    def extract_data_vna(self, field_param_name, freq_range=None, field_range=None):
        """
        Extracts data from a VNA (Vector Network Analyzer) measurement.

        Args:
            field_param_name (string, optional): String contained in the magnetic field parameter name. Defaults to "mag".
            freq_range (tuple, optional): Frequency range to extract data from. Defaults to None.
            field_range (tuple, optional): Magnetic field range to extract data from. Defaults to None.
            
        Returns:
            None
        """
        self.name_field = None
        for key, param in self.independent_parameters.items():
            if field_param_name in param['paramspec'].name: self.name_field = key
        if not self.name_field: raise ValueError(f"No parameter found containing \"{field_param_name}\"")
        self.field = self.independent_parameters[self.name_field]['values']

        freq_slice = slice(0,None)
        field_slice = slice(0,None)
        if freq_range: freq_slice = self.find_slice(self.freq, freq_range)
        if field_range: field_slice = self.find_slice(self.field, field_range)
        
        self.freq = self.freq[freq_slice]
        self.field = self.field[field_slice]
        if self.name_freq == 'x':
            slice_2d = (field_slice, freq_slice)
        else:
            slice_2d = (freq_slice, field_slice)
        self.mag = self.mag[slice_2d]
        self.phase = self.phase[slice_2d]
        self.cData = self.cData[slice_2d]

    def analyze(self, freq_centers=None, freq_span=None, field_range=None, input_power=0, port_type='notch', normalized=False, freq_unit='Hz', do_plots=True):
        """
        Analyze the resonator data over a specified frequency and field range.
        Parameters:
            freq_range (tuple, optional): A tuple specifying the frequency range to analyze (min_freq, max_freq). Defaults to None.
            field_range (tuple, optional): A tuple specifying the field range to analyze (min_field, max_field). Defaults to None.
            input_power (float, optional): The input power in dBm. Defaults to 0.
            port_type (str, optional): The type of port to use for analysis. Options are 'notch' and 'reflection'. Defaults to 'notch'.
            normalized (bool, optional): Flag indicating whether to use normalized data. Defaults to False.
            do_plots (bool, optional): Whether to generate plots of the fitting results. Defaults to True.
        Returns:
            None: The results are stored in the instance variable `fit_report`.
        Raises:
            ValueError: If an unsupported port type is specified.
        Notes:
        The `fit_report` dictionary contains the following keys:
            - "Qi": Internal quality factor.
            - "Qi_err": Error in internal quality factor.
            - "Qc": Coupling quality factor.
            - "Qc_err": Error in coupling quality factor.
            - "Ql": Loaded quality factor.
            - "Ql_err": Error in loaded quality factor.
            - "Nph": Number of photons in the resonator.
            - "single_photon_W": Single photon limit in watts.
            - "single_photon_dBm": Single photon limit in dBm.
            - "fr": Resonant frequency.
            - "fitresults": List of fit results for each field.
            - "port": List of port objects for each field.
        """

        if normalized:
            cData = self.cData_norm
        else:
            cData = self.cData
        freq_scaling = resfit.get_frequency_scaling(freq_unit)
        self.fit_report = resfit.fit_field_sweep(cData.T, self.freq/freq_scaling, self.field, freq_centers, freq_span, field_range, input_power, port_type, freq_unit, do_plots)

    def get_freq_centers_JJ(self, f_max, field_flux_quantum, field_offset=0):
        """
        Calculate the expected resonant frequency of a Josephson junction qubit for each field value.

        Args:
            f_max (float): Maximum frequency of the qubit.
            field_flux_quantum (float): Field corresponding to one flux quantum.
            field_offset (float, optional): Offset field. Defaults to 0.

        Returns:
            freq_centers (np.array): Array of expected resonant frequencies.
        """
        # in numpy, the sinc function already includes the pi
        freq_centers = f_max * np.sqrt(abs(np.sinc((self.field - field_offset)/field_flux_quantum)))
        return freq_centers
    
    def get_freq_centers_SQUID(self, f_max, field_flux_quantum, field_offset=0):
        """
        Calculate the expected resonant frequency of a Josephson junction qubit for each field value.

        Args:
            f_max (float): Maximum frequency of the qubit.
            field_flux_quantum (float): Field corresponding to one flux quantum.
            field_offset (float, optional): Offset field. Defaults to 0.

        Returns:
            freq_centers (np.array): Array of expected resonant frequencies.
        """
        freq_centers = f_max * np.sqrt(abs(np.cos(np.pi*(self.field - field_offset)/field_flux_quantum)))
        return freq_centers


    def normalize_data_from_index(self, idx=-1, axis=0):
        """
        Normalize magnitude and phase with a line cut at the specified index. Default idx is -1, usually corresponding to the highest power trace.
        """
        self.normalize_data(self.name_mag, self.mag[:,idx], self.freq, axis=axis, interpolate=False)
        self.normalize_data(self.name_phase, self.phase[:,idx], self.freq, axis=axis, interpolate=False)
        self.mag_norm = self.dependent_parameters[self.name_mag+'_normalized']['values']
        self.phase_norm = self.dependent_parameters[self.name_phase+'_normalized']['values']
        self.cData_norm = 10**(self.mag_norm/20) * np.exp(1j*self.phase_norm)

    def plot_QvsB(self,label='',log_y=True,threshold=None,**kwargs):
        """
        Plots the quality factors (Qi, Qc, Ql) with error bars versus magnetic field.

        Parameters:
        -----------
        label : str, optional
            Title of the plot. Default is an empty string.
        log_y : bool, optional
            If True, use a logarithmic scale for the y-axis. Default is True.
        threshold : float, optional
            Threshold factor for the discarding bad fits. This factor is used to filter out the fits with large errors
            using the conditions threshold*Q < Q_err for all quality factors. If None, no fit is discarded. Default is None.
        **kwargs : dict, optional
            Additional keyword arguments passed to the errorbar function.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object containing the plot.
        """
        return resfit.plot_QvsB(self.fit_report, self.field, label=label, log_y=log_y, threshold=threshold, **kwargs)

    def plot_kappavsB(self,label='',log_y=True,threshold=None,freq_unit='GHz',**kwargs):
        """
        Plots the quality factors (kappa_i, kappa_c, kappa_l) with error bars versus magnetic field.

        Parameters:
        -----------
        label : str, optional
            Title of the plot. Default is an empty string.
        log_y : bool, optional
            If True, use a logarithmic scale for the y-axis. Default is True.
        threshold : float, optional
            Threshold factor for the discarding bad fits. This factor is used to filter out the fits with large errors
            using the conditions threshold*Q < Q_err for all quality factors. If None, no fit is discarded. Default is None.
        freq_unit : str, optional
            Unit in which frequency is provided. This will determine the proper scaling factors. Default is 'Hz'.
        **kwargs : dict, optional
            Additional keyword arguments passed to the errorbar function.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object containing the plot.
        """
        return resfit.plot_kappavsB(self.fit_report, self.field, label=label, log_y=log_y, threshold=threshold, freq_unit=freq_unit, **kwargs)

    def plot_Qvsfr(self,label='',log_y=True,threshold=None,freq_unit='GHz',**kwargs):
        """
        Plots the quality factors (Qi, Qc, Ql) with error bars versus resonance frequency.

        Parameters:
        -----------
        label : str, optional
            Title of the plot. Default is an empty string.
        log_y : bool, optional
            If True, use a logarithmic scale for the y-axis. Default is True.
        threshold : float, optional
            Threshold factor for the discarding bad fits. This factor is used to filter out the fits with large errors
            using the conditions threshold*Q < Q_err for all quality factors. If None, no fit is discarded. Default is None.
        freq_unit : str, optional
            Unit in which frequency is provided. This will determine the proper scaling factors. Default is 'Hz'.
        **kwargs : dict, optional
            Additional keyword arguments passed to the errorbar function.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object containing the plot.
        """
        return resfit.plot_Qvsfr(self.fit_report, label=label, log_y=log_y, threshold=threshold, freq_unit=freq_unit, **kwargs)

    def plot_kappavsfr(self,label='',log_y=True,threshold=None,freq_unit='GHz',**kwargs):
        """
        Plots the quality factors (kappa_i, kappa_c, kappa_l) with error bars versus magnetic field.

        Parameters:
        -----------
        label : str, optional
            Title of the plot. Default is an empty string.
        log_y : bool, optional
            If True, use a logarithmic scale for the y-axis. Default is True.
        threshold : float, optional
            Threshold factor for the discarding bad fits. This factor is used to filter out the fits with large errors
            using the conditions threshold*Q < Q_err for all quality factors. If None, no fit is discarded. Default is None.
        freq_unit : str, optional
            Unit in which frequency is provided. This will determine the proper scaling factors. Default is 'Hz'.
        **kwargs : dict, optional
            Additional keyword arguments passed to the errorbar function.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object containing the plot.
        """
        return resfit.plot_kappavsfr(self.fit_report, label=label, log_y=log_y, threshold=threshold, freq_unit=freq_unit, **kwargs)