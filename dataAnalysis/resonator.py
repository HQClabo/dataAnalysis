

import numpy as np
import matplotlib.pyplot as plt
import qcodes as qc
from resonator_tools import circuit
import lmfit
from .base import DataSet

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
        ds = DataSetVNA(run_id='123', exp=exp)

        # Extract data from the VNA measurement
        ds._extract_data_vna_base()

        # Normalize the data using a background measurement
        ds.normalize_data_vna(run_id_bg='456', axis=0, interpolate=True)
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
        for key, param in self.dependent_parameters.items():
            if 'magnitude' in param['paramspec'].name: 
                self.name_mag = key
            if 'phase' in param['paramspec'].name: 
                self.name_phase = key
                phase_unit = param['paramspec'].unit
        for key, param in self.independent_parameters.items():
            if 'frequency' in param['paramspec'].name: 
                self.name_freq = key
        
        # Convert phase to radians always
        if phase_unit == 'rad':
            pass
        elif phase_unit == '°':
            self.dependent_parameters[self.name_phase]['values'] *= np.pi/180
        else:
            raise ValueError(f'The phase \"{phase_unit}\" was not recognized.')

        # Generate attributes
        self.freq = self.independent_parameters[self.name_freq]['values']
        self.mag = self.dependent_parameters[self.name_mag]['values']
        self.phase = self.dependent_parameters[self.name_phase]['values']
        self.cData = 10**(self.mag/20) * np.exp(1j*self.phase)

    def normalize_data_vna(self, run_id_bg: int, axis:int=0, interpolate=True):
        """
        Normalize the measurement data using as background the data of the provided measurement run.

        Note that this function assumes that the background run id belongs to the same experiment of the data run id.

        Args: 
            run_id_bg: Id of the background measurement.
            axis (optional, default 0): Axis along which to normalize.
            interpolate (optional, default True): Boolean flag to interpolate the background trace if the number of points
                do not match with the measurement data.

        Returns:
            None
        
        """
        ds_bg = DataSetVNA(self.exp, run_id_bg)
        self.normalize_data(self.name_mag, data_bg=ds_bg.mag, x_bg=ds_bg.freq, axis=axis, interpolate=interpolate)
        self.normalize_data(self.name_phase, data_bg=ds_bg.phase, x_bg=ds_bg.freq, axis=axis, interpolate=interpolate)
        self.mag_norm = self.dependent_parameters[self.name_mag+'_normalized']['values']
        self.phase_norm = self.dependent_parameters[self.name_phase+'_normalized']['values']
        self.cData_norm = 10**(self.mag_norm/20) * np.exp(1j*self.phase_norm)
    
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

    def analyze(self, freq_range=None, input_power=0, port_type='notch', normalized=False, do_plots=True):
        """
        Perform a resonator fit of the data using the specified using resonator_tools. The results can be found in self.fit_report.

        Args:
            freq_range (tuple, optional): Frequency range to consider for analysis. Defaults to None.
            input_power (float, optional): Input power in dBm that arrives at the coupling port. Defaults to 0.
            port_type (str, optional): Type of port to use for analysis. Supported values are 'notch' or 'reflection'. Defaults to 'notch'.
            normalized (bool, optional): Flag indicating whether to use normalized data. Defaults to False.
            do_plots (bool, optional): Flag indicating whether to generate plots. Defaults to True.

        Returns:
            None
        """
        fit_report = {}

        if normalized:
            cData = self.cData_norm
        else:
            cData = self.cData
            
        # Define port type to use
        if port_type == 'notch':
            port = circuit.notch_port()
        elif port_type == 'reflection':
            port = circuit.reflection_port()
        else:
            print("This port type is not supported. Supported types are 'notch' and 'reflection'.")
        
        # Cut and fit data
        port.add_data(self.freq,cData)
        if freq_range: 
            port.cut_data(*freq_range)
        # port.autofit(fr_guess=center_freq[k])
        port.autofit()
        if do_plots == True:
            port.plotall()

        # Add fitting results to the dictionary
        if port_type == 'notch':
            fit_report["Qi"] = port.fitresults["Qi_dia_corr"]
            fit_report["Qi_err"] = port.fitresults["Qi_dia_corr_err"]
            fit_report["Qc"] = port.fitresults["Qc_dia_corr"]
            fit_report["Qc_err"] = port.fitresults["absQc_err"]
        else:
            fit_report["Qi"] = port.fitresults["Qi"]
            fit_report["Qi_err"] = port.fitresults["Qi_err"]
            fit_report["Qc"] = port.fitresults["Qc"]
            fit_report["Qc_err"] = port.fitresults["Qc_err"]
        fit_report["Ql"] = port.fitresults["Ql"]
        fit_report["Ql_err"] = port.fitresults["Ql_err"]
        fit_report["Nph"] = port.get_photons_in_resonator(input_power,unit='dBm')
        fit_report["fr"] = port.fitresults["fr"]
        fit_report['fitresults'] = port.fitresults
        self.fit_report = fit_report
        self.port = port


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


    def analyze(self, freq_range=None, power_range=None, attenuation=0, port_type='notch', normalized=False, do_plots=True):
        """
        Perform a resonator fit of the data using the specified using resonator_tools. The results can be found in self.fit_report.

        Args:
            freq_range (tuple, optional): Frequency range to consider for analysis. Defaults to None.
            power_range (tuple, optional): Power range to consider for analysis. Defaults to None.
            attenuation (float, optional): Expected attenuation in dB between the instrument output and the resonator port. Defaults to 0.
            port_type (str, optional): Type of port to use for analysis. Supported values are 'notch' or 'reflection'. Defaults to 'notch'.
            normalized (bool, optional): Flag indicating whether to use normalized data. Defaults to False.
            do_plots (bool, optional): Flag indicating whether to generate plots. Defaults to True.

        Returns:
            None
        """
        n_powers = len(self.power)
        fit_report = {
            "Qi" : np.array([np.nan]*n_powers),
            "Qi_err" : np.array([np.nan]*n_powers),
            "Qc" : np.array([np.nan]*n_powers),
            "Qc_err" : np.array([np.nan]*n_powers),
            "Ql" : np.array([np.nan]*n_powers),
            "Ql_err" : np.array([np.nan]*n_powers),
            "Nph" : np.array([np.nan]*n_powers),
            "fr" : np.array([np.nan]*n_powers),
            'fitresults': [np.nan]*n_powers,
            }

        if normalized:
            cData = self.cData_norm
        else:
            cData = self.cData

        # Cycle for all powers. k is the power index
        for k,power in enumerate(self.power):
            if power_range:
                if (power < power_range[0]) or (power > power_range[1]):
                    continue
            
            # Define port type
            if port_type == 'notch':
                port = circuit.notch_port()
            elif port_type == 'reflection':
                port = circuit.reflection_port()
            else:
                print("This port type is not supported. Supported types are 'notch' and 'reflection'.")
            
            # Cut data
            port.add_data(self.freq,cData[:,k])
            if freq_range: 
                port.cut_data(*freq_range)
            # port.autofit(fr_guess=center_freq[k])
            port.autofit()
            if do_plots == True:
                port.plotall()

            # Add fitting results to the dictionary
            if port_type == 'notch':
                fit_report["Qi"][k] = port.fitresults["Qi_dia_corr"]
                fit_report["Qi_err"][k] = port.fitresults["Qi_dia_corr_err"]
                fit_report["Qc"][k] = port.fitresults["Qc_dia_corr"]
                fit_report["Qc_err"][k] = port.fitresults["absQc_err"]
            else:
                fit_report["Qi"][k] = port.fitresults["Qi"]
                fit_report["Qi_err"][k] = port.fitresults["Qi_err"]
                fit_report["Qc"][k] = port.fitresults["Qc"]
                fit_report["Qc_err"][k] = port.fitresults["Qc_err"]
            fit_report["Ql"][k] = port.fitresults["Ql"]
            fit_report["Ql_err"][k] = port.fitresults["Ql_err"]
            fit_report["Nph"][k] = port.get_photons_in_resonator(power - attenuation,unit='dBm')
            fit_report["fr"][k] = port.fitresults["fr"]
            fit_report['fitresults'][k] = port.fitresults

        self.fit_report = fit_report
        self.port = port

    def analyze_lmfit(self, freq_range=None, power_range=None, guesses: dict = {}, attenuation=0, port_type='notch', normalized=False, do_plots=True, print_guesses=False):
        """
        Fit a resonator reflection data that has previously been added to the port
        to a model with a Lorentzian resonance and a linear background.

        Args:
            fcenter : float, optional
                Center frequency of the resonance. If None, the function will try to guess
                the center frequency from the data.
            fspan : float, optional
                Frequency span around the center frequency to fit to. If None, the entire frequency range will be used.
            guesses : dict, optional
                Initial guesses for the fit parameters. If None, the function will try to guess
                the parameters from the data.
            do_plot : bool, optional
                Whether to plot the fit results.

        Returns:
            result : lmfit.model.ModelResult
                The result of the fit, including the optimal values for the fit parameters.
        """
        if normalized:
            cData = self.cData_norm
        else:
            cData = self.cData

        results = []

        # Cycle over the powers. k is the power index
        for k,power in enumerate(self.power):
            if power_range:
                if (power < power_range[0]) or (power > power_range[1]):
                    continue

            # Define port type
            if port_type == 'notch':
                port = circuit.notch_port()
                model_func = S21_resonator_notch
            elif port_type == 'reflection':
                port = circuit.reflection_port()
                model_func = S11_resonator_reflection
            else:
                print("This port type is not supported. Supported types are 'notch' or 'reflection'")
            
            # Cut and provide data
            port.add_data(self.freq,cData[:,k])
            if freq_range:
                port.cut_data(*freq_range)
            fdrive = port.f_data
            zdata = port.z_data_raw

            # Obtain guesses with resonator_tools
            delay, a, alpha, fr, Ql, A2, frcal = port.do_calibration(fdrive, zdata)

            initial_guesses = {'fr': frcal,
                               'kappa_int': fr/Ql/2,
                               'kappa_ext': fr/Ql/2,
                               'a': a,
                               'alpha': alpha,
                               'delay': delay,
                               'phi0': 0}
            for guess in guesses.keys(): 
                initial_guesses[guess] = guesses[guess]
            if print_guesses: 
                print(initial_guesses)
    
            params=lmfit.Parameters()
            params.add('fr', value=initial_guesses['fr'], vary=True)
            params.add('kappa_int', value=initial_guesses['kappa_int'], vary=True)
            params.add('kappa_ext', value=initial_guesses['kappa_ext'], vary=True)
            params.add('a', value=initial_guesses['a'], vary=True)
            params.add('alpha', value=initial_guesses['alpha'], vary=True)
            params.add('delay', value=initial_guesses['delay'], vary=True)
            if port_type == 'notch': params.add('phi0', value=initial_guesses['phi0'], vary=True)
    
            # Perform the fit
            model = lmfit.Model(model_func, independent_vars=['fdrive'])
            result = model.fit(zdata, params, fdrive=fdrive)
    
            # Plot
            if do_plots:
                fig, axes = plt.subplots(1,3,width_ratios=[1,1,1],gridspec_kw=dict(wspace=0.4))
                fig.set_size_inches(18/2.54, 5/2.54)
                # fig.suptitle(plot_title)
                
                axes[0].plot(fdrive/1e9, 20*np.log10(abs(zdata)), marker='.', ms=2, ls='')
                axes[0].plot(fdrive/1e9, 20*np.log10(abs(result.best_fit)))
                # axes[0].plot(fdrive, 20*np.log(abs(result.eval(params))))
                # myplt.format_plot(axes[0],xlabel='f (GHz)',ylabel='|S21| (dB)')
                axes[1].plot(fdrive/1e9, 180/np.pi*np.angle(zdata), marker='.', ms=2, ls='')
                axes[1].plot(fdrive/1e9, 180/np.pi*np.angle(result.best_fit))
                # myplt.format_plot(axes[1],xlabel='f (GHz)',ylabel='S21 (°)')
                axes[2].plot(zdata.real, zdata.imag, marker='.', ms=2, ls='')
                axes[2].plot(result.best_fit.real, result.best_fit.imag)
                # myplt.format_plot(axes[2],xlabel='Re(S21) (a.u.)',ylabel='Im(S21) (a.u.)')
            plt.show()
            plt.close()
            results.append(result)

            self.fit_report_lmfit = results
            self.port = port


    
    def normalize_data_from_index(self, idx=-1, axis=0):
        """
        Normalize magnitude and phase with a line cut at the specified index. Default idx is -1, usually corresponding to the highest power trace.
        """
        self.normalize_data(self.name_mag, self.mag[:,idx], self.freq, axis=axis, interpolate=False)
        self.normalize_data(self.name_phase, self.phase[:,idx], self.freq, axis=axis, interpolate=False)
        self.mag_norm = self.dependent_parameters[self.name_mag+'_normalized']['values']
        self.phase_norm = self.dependent_parameters[self.name_phase+'_normalized']['values']
        self.cData_norm = 10**(self.mag_norm/20) * np.exp(1j*self.phase_norm)

    def plot_QvsP(self,label='',log_y=True,**kwargs):
        fit = self.fit_report
        fig, ax = plt.subplots(1)
        if log_y:
            ax.loglog()
        else:
            ax.semilogx()
        ax.errorbar(fit['Nph'],fit['Qi'],yerr=fit['Qi_err'],label='$Q_{int}$',fmt = "o",**kwargs)
        ax.errorbar(fit['Nph'],fit['Qc'],yerr=fit['Qc_err'],label='$Q_{ext}$',fmt = "o",**kwargs)
        ax.errorbar(fit['Nph'],fit['Ql'],yerr=fit['Ql_err'],label='$Q_{load}$',fmt = "o",**kwargs)
        ax.legend()
        ax.set_xlabel('photon number')
        ax.set_ylabel('Q')
        ax.grid()
        fig.suptitle(label)
        fig.tight_layout()
        return fig,ax


class BScanVNA(DataSetVNA):
    def __init__(self, exp, run_id=None, station=None, field_param_name='mag', freq_range=None, field_range=None):
        super().__init__(exp=exp, run_id=run_id, station=station)
        self.extract_data_vna(field_param_name, freq_range, field_range)

    def analyze(self, freq_range=None, power_range=None, attenuation=0, port_type='notch', do_plots=True):

        n_powers = len(self.power)
        fit_report = {
            "Qi" : np.zeros(n_powers),
            "Qi_err" : np.zeros(n_powers),
            "Qc" : np.zeros(n_powers),
            "Qc_err" : np.zeros(n_powers),
            "Ql" : np.zeros(n_powers),
            "Ql_err" : np.zeros(n_powers),
            "Nph" : np.zeros(n_powers),
            "fr" : np.zeros(n_powers),
            'fitresults': [0]*n_powers,
            }
        for k,power in enumerate(self.power):
            if power_range:
                if (power < power_range[0]) or (power > power_range[1]):
                    continue
            
            # define port type
            if port_type == 'notch':
                port = circuit.notch_port()
            elif port_type == 'reflection':
                port = circuit.reflection_port()
            else:
                print("This port type is not supported. Use 'notch', 'reflection' or 'transmission' (tbd)")
            # cut and fit data
            port.add_data(self.freq,self.cData[k])
            if freq_range: port.cut_data(*freq_range)
            # port.autofit(fr_guess=center_freq[k])
            port.autofit()
            if do_plots == True:
                port.plotall()
            # add fitting results to the dictionary
            if port_type == 'notch':
                fit_report["Qi"][k] = port.fitresults["Qi_dia_corr"]
                fit_report["Qi_err"][k] = port.fitresults["Qi_dia_corr_err"]
                fit_report["Qc"][k] = port.fitresults["Qc_dia_corr"]
                fit_report["Qc_err"][k] = port.fitresults["absQc_err"]
            else:
                fit_report["Qi"][k] = port.fitresults["Qi"]
                fit_report["Qi_err"][k] = port.fitresults["Qi_err"]
                fit_report["Qc"][k] = port.fitresults["Qc"]
                fit_report["Qc_err"][k] = port.fitresults["Qc_err"]
            fit_report["Ql"][k] = port.fitresults["Ql"]
            fit_report["Ql_err"][k] = port.fitresults["Ql_err"]
            fit_report["Nph"][k] = port.get_photons_in_resonator(power - attenuation,unit='dBm')
            fit_report["fr"][k] = port.fitresults["fr"]
            fit_report['fitresults'][k] = port.fitresults
        self.fit_report = fit_report

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

    def normalize_data_from_index(self, idx=-1, axis=0):
        """
        Normalize magnitude and phase with a line cut at the specified index. Default idx is -1, usually corresponding to the highest power trace.
        """
        self.normalize_data(self.name_mag, self.mag[:,idx], self.freq, axis=axis, interpolate=False)
        self.normalize_data(self.name_phase, self.phase[:,idx], self.freq, axis=axis, interpolate=False)
        self.mag_norm = self.dependent_parameters[self.name_mag+'_normalized']['values']
        self.phase_norm = self.dependent_parameters[self.name_phase+'_normalized']['values']
        self.cData_norm = 10**(self.mag_norm/20) * np.exp(1j*self.phase_norm)

    def plot_QvsP(self,label='',log_x=True,**kwargs):
        fit = self.fit_report
        fig, ax = plt.subplots(1)
        fig.dpi = res
        if log_x:
            ax.loglog()
        else:
            ax.semilogx()
        ax.errorbar(fit['Nph'],fit['Qi'],yerr=fit['Qi_err'],label='$Q_{int}$',fmt = "o",**kwargs)
        ax.errorbar(fit['Nph'],fit['Qc'],yerr=fit['Qc_err'],label='$Q_{ext}$',fmt = "o",**kwargs)
        ax.errorbar(fit['Nph'],fit['Ql'],yerr=fit['Ql_err'],label='$Q_{load}$',fmt = "o",**kwargs)
        ax.legend()
        ax.set_xlabel('photon number')
        ax.set_ylabel('Q')
        ax.grid()
        fig.suptitle(label)
        fig.tight_layout()
        return fig,ax


        