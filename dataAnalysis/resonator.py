

import numpy as np
import matplotlib.pyplot as plt
import qcodes as qc
from resonator_tools import circuit
import lmfit
from .base import DataSet

class DataSetVNA(DataSet):
    def __init__(self, run_id=None, exp=None, station=None, freq_range=None):
        super().__init__(run_id, exp, station)
        self._extract_data_vna_base()

    def _extract_data_vna_base(self):
        """
        Extracts data from a VNA (Vector Network Analyzer) measurement.

        Args:
            freq_range (tuple, optional): Frequency range to extract data from. Defaults to None.
            
        Returns:
            None
        """
        for key, param in self.dependent_parameters.items():
            if 'magnitude' in param['paramspec'].name: self.name_mag = key
            if 'phase' in param['paramspec'].name: self.name_phase = key
        for key, param in self.independent_parameters.items():
            if 'frequency' in param['paramspec'].name: self.name_freq = key

        self.freq = self.independent_parameters[self.name_freq]['values']
        self.mag = self.dependent_parameters[self.name_mag]['values']
        self.phase = self.dependent_parameters[self.name_phase]['values']
        self.cData = 10**(self.mag/20) * np.exp(1j*self.phase)

    def normalize_data_vna(self, run_id_bg, axis=0, interpolate=False):
        ds_bg = DataSetVNA(run_id_bg, exp=self.exp)
        bg_freq = ds_bg.independent_parameters[ds_bg.name_freq]['values']
        bg_mag = ds_bg.dependent_parameters[ds_bg.name_mag]['values']
        bg_phase = ds_bg.dependent_parameters[ds_bg.name_phase]['values']
        self.normalize_data(self.name_mag, data_bg=bg_mag, x_bg=bg_freq, axis=axis, interpolate=interpolate)
        self.normalize_data(self.name_phase, data_bg=bg_phase, x_bg=bg_freq, axis=axis, interpolate=interpolate)
        mag_norm = self.dependent_parameters[self.name_mag+'_normalized']['values']
        phase_norm = self.dependent_parameters[self.name_phase+'_normalized']['values']
        self.cData_norm = 10**(mag_norm/20) * np.exp(1j*phase_norm)
    
class FrequencyScanVNA(DataSetVNA):
    def __init__(self, run_id=None, exp=None, station=None, freq_range=None):
        super().__init__(run_id, exp, station)
        self.extract_data_vna(freq_range)

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
            
        # define port type
        if port_type == 'notch':
            port = circuit.notch_port()
        elif port_type == 'reflection':
            port = circuit.reflection_port()
        else:
            print("This port type is not supported. Use 'notch' or 'reflection'.")
        # cut and fit data
        
        port.add_data(self.freq,cData)
        if freq_range: port.cut_data(*freq_range)
        # port.autofit(fr_guess=center_freq[k])
        port.autofit()
        if do_plots == True:
            port.plotall()
        # add fitting results to the dictionary
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


    def extract_data_vna(self, freq_range=None):
        """
        Extracts data from a VNA (Vector Network Analyzer) measurement.

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

    # def normalize_data_vna(self, run_id_bg, axis=0, interpolate=False):
    #     ds_bg = DataSetVNA(run_id_bg, exp=self.exp)
    #     self.normalize_data(self.mag_name, ds_bg.dependent_parameters[ds_bg.mag_name]['values'], axis=axis, interpolate=interpolate)
    #     self.normalize_data(self.phase_name, ds_bg.dependent_parameters[ds_bg.phase_name]['values'], axis=axis, interpolate=interpolate)
    #     mag_norm = self.dependent_parameters[self.name_mag+'_normalized']['values']
    #     phase_norm = self.dependent_parameters[self.name_phase+'_normalized']['values']
    #     self.cData_norm = 10**(mag_norm/20) * np.exp(1j*phase_norm)

    def S21_resonator_notch(self, fdrive, fr, kappa_int, kappa_ext, a, alpha, delay, phi0):
        delta_r = fdrive - fr
        S21 = (delta_r - 1j/2*(kappa_int + kappa_ext*(1-np.exp(1j*phi0)))) / (delta_r - 1j/2*(kappa_ext + kappa_int))
        environment = a * np.exp(1j*(alpha - delay*2*np.pi*fdrive))
        return S21 * environment

    def S11_resonator_reflection(self, fdrive, fr, kappa_int, kappa_ext, a, alpha, delay):
        delta_r = fdrive - fr
        S11 = (delta_r + 1j/2*(kappa_ext - kappa_int)) / (delta_r - 1j/2*(kappa_ext + kappa_int))
        environment = a * np.exp(1j*(alpha - delay*2*np.pi*fdrive))
        return S11 * environment


class PowerScanVNA(DataSet):
    def __init__(self, run_id=None, exp=None, station=None, freq_range=None, power_range=None):
        super().__init__(run_id, exp, station)
        self.extract_data_vna(freq_range, power_range)

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
                print("This port type is not supported. Use 'notch' or 'reflection'.")
            # cut and fit data
            
            port.add_data(self.freq,cData[:,k])
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
        self.port = port

    def analyze_lmfit(self, freq_range=None, power_range=None, guesses: dict = {}, attenuation=0, port_type='notch', normalized=False, do_plots=True, print_guesses=False):
        """
        Fit a resonator reflection data that has previously been added to the port
        to a model with a Lorentzian resonance and a linear background.

        Parameters
        ----------
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

        Returns
        -------
        result : lmfit.model.ModelResult
            The result of the fit, including the optimal values for the fit parameters.
        """
        if normalized:
            cData = self.cData_norm
        else:
            cData = self.cData

        results = []
        for k,power in enumerate(self.power):
            if power_range:
                if (power < power_range[0]) or (power > power_range[1]):
                    continue
            # define port type
            if port_type == 'notch':
                port = circuit.notch_port()
                model_func = self.S21_resonator_notch
            elif port_type == 'reflection':
                port = circuit.reflection_port()
                model_func = self.S11_resonator_reflection
            else:
                print("This port type is not supported. Use 'notch', 'reflection' or 'transmission' (t.b.d.)")
            # cut and fit data
            
            port.add_data(self.freq,cData[:,k])
            if freq_range:
                port.cut_data(*freq_range)
            # port.autofit(fr_guess=center_freq[k])
    
            fdrive = port.f_data
            zdata = port.z_data_raw
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

            if print_guesses: print(initial_guesses)
    
            params=lmfit.Parameters()
            params.add('fr', value=initial_guesses['fr'], vary=True)
            params.add('kappa_int', value=initial_guesses['kappa_int'], vary=True)
            params.add('kappa_ext', value=initial_guesses['kappa_ext'], vary=True)
            params.add('a', value=initial_guesses['a'], vary=True)
            params.add('alpha', value=initial_guesses['alpha'], vary=True)
            params.add('delay', value=initial_guesses['delay'], vary=True)
            if port_type == 'notch': params.add('phi0', value=initial_guesses['phi0'], vary=True)
    
            model = lmfit.Model(model_func, independent_vars=['fdrive'])
            result = model.fit(zdata, params, fdrive=fdrive)
    
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

    def S21_resonator_notch(self, fdrive, fr, kappa_int, kappa_ext, a, alpha, delay, phi0):
        delta_r = fdrive - fr
        S21 = (delta_r - 1j/2*(kappa_int + kappa_ext*(1-np.exp(1j*phi0)))) / (delta_r - 1j/2*(kappa_ext + kappa_int))
        environment = a * np.exp(1j*(alpha - delay*2*np.pi*fdrive))
        return S21 * environment

    def S11_resonator_reflection(self, fdrive, fr, kappa_int, kappa_ext, a, alpha, delay):
        delta_r = fdrive - fr
        S11 = (delta_r + 1j/2*(kappa_ext - kappa_int)) / (delta_r - 1j/2*(kappa_ext + kappa_int))
        environment = a * np.exp(1j*(alpha - delay*2*np.pi*fdrive))
        return S11 * environment

    def extract_data_vna(self, freq_range=None, power_range=None):
        """
        Extracts data from a VNA (Vector Network Analyzer) measurement.

        Args:
            freq_range (tuple, optional): Frequency range to extract data from. Defaults to None.
            power_range (tuple, optional): Power range to extract data from. Defaults to None.
            
        Returns:
            None
        """
        for key, param in self.dependent_parameters.items():
            if 'magnitude' in param['paramspec'].name: 
                self.name_mag = key
            if 'phase' in param['paramspec'].name: 
                self.name_phase = key
        for key, param in self.independent_parameters.items():
            if 'frequency' in param['paramspec'].name: 
                self.name_freq = key
                
        self.freq = self.independent_parameters[self.name_freq]['values']
        self.mag = self.dependent_parameters[self.name_mag]['values']
        self.phase = self.dependent_parameters[self.name_phase]['values']
        self.cData = 10**(self.mag/20) * np.exp(1j*self.phase)
        
        
        for key, param in self.independent_parameters.items():
            if 'power' in param['paramspec'].name: 
                self.name_power = key
        self.power = self.independent_parameters[self.name_power]['values']


        if freq_range:
            freq_slice = self.find_slice(self.freq,freq_range)
        else:
            freq_slice = slice(0,len(self.freq))
        if power_range:
            power_slice = self.find_slice(self.power,power_range)
        else:
            power_slice = slice(0,len(self.power))

        self.freq = self.freq [freq_slice]
        self.power = self.power[power_slice]
        if self.name_freq == 'x':
            self.mag = self.mag[power_slice,freq_slice]
            self.phase = self.phase[power_slice,freq_slice]
        else:
            self.mag = self.mag[freq_slice,power_slice]
            self.phase = self.phase[freq_slice,power_slice]
        self.cData = 10**(self.mag/20) * np.exp(1j*self.phase)


    # def extract_data_vna(self, freq_range=None, power_range=None):
    #     """
    #     Extracts data from a VNA (Vector Network Analyzer) measurement.

    #     Args:
    #         freq_range (tuple, optional): Frequency range to extract data from. Defaults to None.
    #         power_range (tuple, optional): Power range to extract data from. Defaults to None.

    #     Returns:
    #         None
    #     """
    #     self.extract_data(self.run_id, exp=self.exp)

    #     for key, param in self.dependent_parameters.items():
    #         if 'magnitude' in param['paramspec'].name: self.name_mag = key
    #         if 'phase' in param['paramspec'].name: self.name_phase = key
    #     for key, param in self.independent_parameters.items():
    #         if 'frequency' in param['paramspec'].name: self.name_freq = key
    #         if 'power' in param['paramspec'].name: self.name_power = key

    #     freq = self.independent_parameters[self.name_freq]['values']
    #     power = self.independent_parameters[self.name_power]['values']
    #     mag = self.dependent_parameters[self.name_mag]['values']
    #     phase = self.dependent_parameters[self.name_phase]['values']
    #     if freq_range:
    #         freq_slice = self.find_slice(freq,freq_range)
    #     else:
    #         freq_slice = slice(0,len(freq))
    #     if power_range:
    #         power_slice = self.find_slice(power,power_range)
    #     else:
    #         power_slice = slice(0,len(power))
        
    #     self.freq = freq [freq_slice]
    #     self.power = power[power_slice]
    #     if self.name_freq == 'x':
    #         self.mag = mag[power_slice,freq_slice]
    #         self.phase = phase[power_slice,freq_slice]
    #     else:
    #         self.mag = mag[freq_slice,power_slice]
    #         self.phase = phase[freq_slice,power_slice]
    #     self.cData = 10**(self.mag/20) * np.exp(1j*self.phase)

    def normalize_data_from_index(self, idx=-1, axis=0):
        """
        Normalize magnitude and phase with a line cut at the specified index. Default idx is -1, usually corresponding to the highest power trace.
        """
        self.normalize_data(self.name_mag, self.mag[:,idx], self.freq, axis=0)
        self.normalize_data(self.name_phase, self.phase[:,idx], self.freq, axis=0)
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
    def __init__(self, run_id, exp=None, station=None, freq_range=None, magnet_range=None):
        super().__init__(run_id, exp, station)
        self.extract_data_vna(freq_range, power_range)

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
                print("This port type is not supported. Use 'notch', 'reflection' or 'transmission' (t.b.d.)")
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

    def extract_data_vna(self, freq_range=None, bfield_range=None, bfield_param_name='field'):
        """
        Extracts data from a VNA (Vector Network Analyzer) measurement.

        Args:
            freq_range (tuple, optional): Frequency range to extract data from. Defaults to None.
            bfield_range (tuple, optional): Magnetic field range to extract data from. Defaults to None.
            bfield_param_name (string, optional): String contained in the magnetic field parameter name. Defaults to "field".
            
        Returns:
            None
        """
        for key, param in self.independent_parameters.items():
            if bfield_param_name in param['paramspec'].name: self.name_bfield = key
        self.bfield = self.independent_parameters[self.name_bfield]['values']

        if freq_range:
            freq_slice = self.find_slice(freq,freq_range)
        else:
            freq_slice = slice(0,len(freq))
        if power_range:
            bfield_slice = self.find_slice(self.bfield,bfield_slice)
        else:
            bfield_slice = slice(0,len(power))

        self.freq = freq [freq_slice]
        self.bfield = power[bfield_slice]
        if self.name_freq == 'x':
            self.mag = mag[bfield_slice,freq_slice]
            self.phase = phase[bfield_slice,freq_slice]
        else:
            self.mag = mag[freq_slice,bfield_slice]
            self.phase = phase[freq_slice,bfield_slice]
        self.cData = 10**(self.mag/20) * np.exp(1j*self.phase)

    
    def extract_data(self, freq_range=None, b_range=None):
        conn = None
        if self.exp: conn = self.exp.conn
 
        dataset = qc.load_by_id(self.run_id, conn)
        df = dataset.to_pandas_dataframe()
        for key in df.keys():
            if 'magnitude' in key: instr_name_mag = key
            if 'phase' in key: instr_name_phase = key
        for key in df.index.names:
            if 'frequency' in key: instr_name_freq = key
            # if 'power' in key: instr_name_power = key
        data_pivot = df.pivot_table(index=instr_name_power, columns=instr_name_freq)

        freq = data_pivot[instr_name_mag].columns.values
        power = data_pivot[instr_name_mag].index.values
        if freq_range:
            freq_slice = self.find_slice(freq,freq_range)
        else:
            freq_slice = slice(0,len(freq))
        # if power_range:
        #     power_slice = self.find_slice(power,power_range)
        # else:
            # power_slice = slice(0,len(power))
        self.freq = freq[freq_slice]
        # self.power = power[power_slice]
        self.mag = data_pivot[instr_name_mag].values[power_slice,freq_slice]
        self.phase = data_pivot[instr_name_phase].values[power_slice,freq_slice]
        self.cData = 10**(self.mag/20) * (np.cos(self.phase) + 1j*np.sin(self.phase))[power_slice,freq_slice]

    def subtract_background(self, run_id, idx=-1):
        conn = None
        if self.exp: conn = self.exp.conn
        dataset = qc.load_by_id(run_id, conn)
        df = dataset.to_pandas_dataframe()
        for key in df.keys():
            if 'magnitude' in key: instr_name_mag = key
            if 'phase' in key: instr_name_phase = key
        for key in df.index.names:
            if 'frequency' in key: instr_name_freq = key
            if 'power' in key: instr_name_power = key
        data_pivot = df.pivot_table(index=instr_name_power, columns=instr_name_freq)

        if len(self.freq) != len(data_pivot[instr_name_mag].columns.values):
            print('Background trace does not have the same dimension as the data')
            return
        mag_bg = data_pivot[instr_name_mag].values[idx]
        phase_bg = data_pivot[instr_name_phase].values[idx]
        self.cData = self.cData / ( 10**(mag_bg/20) * (np.cos(phase_bg) + 1j*np.sin(phase_bg)) )

    # def plot_2D(self,freq_range=None,power_range=None,**kwargs):
    #     figs = []
    #     axes = []
    #     for _ in range(2):
    #         fig,ax = plt.subplots()
            
    #         figs.append(fig)
    #         axes.append(ax)

    #     power_plot = np.tile(self.power,(len(self.freq),1)).T
    #     freq_plot = np.tile(self.freq,(len(self.power),1))
        
    #     qc.dataset.plotting.plot_on_a_plain_grid(power_plot,freq_plot,self.mag,axes[0],**kwargs)
    #     qc.dataset.plotting.plot_on_a_plain_grid(power_plot,freq_plot,self.phase,axes[1],**kwargs)
        
    #     return figs,axes

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


        