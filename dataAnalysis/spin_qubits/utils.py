import numpy as np
from matplotlib import pyplot as plt
import lmfit

from dataAnalysis.dataset import DataSet, DataSetRFOPX

def charge_sensor_peak_model(Vg, A, V0, Gamma, offset):
    return A * (Gamma/2)**2 / ( (Vg - V0)**2 + (Gamma/2)**2) + offset

def get_init_guesses(xdata, ydata):
        V0 = xdata[np.argmax(ydata)]
        A = np.abs(np.max(ydata))
        Gamma = (xdata[-1] - xdata[0])/2
        offset = np.min(ydata)

        print(f"Guesses: A={A}, V0={V0}, Gamma={Gamma}, offset={offset}")

        return {"A": A, "V0": V0, "Gamma": Gamma, "offset": offset}

def fit_charge_sensor_peak(xdata, ydata, guesses_dict=None):
        if guesses_dict == None:
            guesses_dict = get_init_guesses(xdata, ydata)
        params = lmfit.Parameters()
        params.add("A", value=guesses_dict["A"])
        params.add("V0", value=guesses_dict["V0"])
        params.add("Gamma", value=guesses_dict["Gamma"])
        params.add("offset", value=guesses_dict["offset"])

        model = lmfit.Model(charge_sensor_peak_model, independent_vars=['Vg'])

        plt.plot(xdata, ydata, '.', color='k')
        plt.plot(xdata, model.eval(params=params, Vg=xdata), '-', color='red')
        plt.xlabel("Gate voltage (V)")
        plt.ylabel("Response")
        
        fit_result = model.fit(ydata, params, Vg=xdata)
        return model, fit_result

class ChargeSensorAnalysis(DataSet):
    def __init__(self, exp, run_id=None):
        super().__init__(exp=exp, run_id=run_id)
        self.xdata = self.independent_parameters['x']['values']

    def find_max_derivative_point(self, ydata_param_name, method='numeric', shoulder='left', **kwargs):
        if method == 'numeric':
            return self._find_max_derivative_point_numeric(ydata_param_name, shoulder=shoulder, **kwargs)
        elif method == 'lmfit':
            return self._find_max_derivative_point_lmfit(ydata_param_name, shoulder=shoulder)

    def find_peak(self, ydata_param_name):
        """
        Calibrate the charge sensor operation point by fitting the Coulomb peak with a Lorentzian and 
        looking at the point with maximum derivative of the reflected amplitude.
        
        Params:
            ydata_param_name: name of the dependent parameter that should be used.

        Returns:
            The gate voltage corresponding to the point with maximum derivative.
        """
        self.ydata = self.get_dependent_parameter_by_name(ydata_param_name)['values']
        self.V0 = self.xdata[np.argmax(self.ydata)]

        # print(f"Peak found at V0 = {self.V0:.5f} V for {ydata_param_name}.")

        # Plot
        plt.figure()
        plt.plot(self.xdata, self.ydata, '.', color='k')
        plt.xlabel("Gate voltage (V)")
        plt.ylabel(f"{ydata_param_name} ({self.get_dependent_parameter_by_name(ydata_param_name)['paramspec'].unit})")
        plt.title(f"Run #{self.run_id}")
        plt.axvline(x = self.V0, ls="-", color = 'red')

        return self.V0


    def _find_max_derivative_point_lmfit(self, ydata_param_name, shoulder='left'):
        """
        Calibrate the charge sensor operation point by fitting the Coulomb peak with a Lorentzian and 
        looking at the point with maximum derivative of the ydata.
        
        Params:
            ydata_param_name: name of the dependent parameter that should be used.
            shoulder: Either 'left' or 'right'

        Returns:
            The gate voltage corresponding to the point with maximum derivative.
        """
        self.ydata = self.get_dependent_parameter_by_name(ydata_param_name)['values']
        self.ydata_param_name = ydata_param_name

        self.model, self.fit_result = fit_charge_sensor_peak(self.xdata, self.ydata)
        self.A = self.fit_result.params['A']
        self.V0 = self.fit_result.params['V0']
        self.Gamma = self.fit_result.params['Gamma']
        self.offset = self.fit_result.params['offset']
        if shoulder == "left":
            self.V_max_deriv = self.V0 - self.Gamma/(2*np.sqrt(3))
        elif shoulder == "right":
            self.V_max_deriv = self.V0 + self.Gamma/(2*np.sqrt(3))
        else:
            raise ValueError("Invalid option for 'shoulder' parameter: must be either 'left' or 'right'.")


        plt.figure()
        plt.plot(self.xdata, self.ydata, '.', color='k')
        plt.xlabel("Gate voltage (V)")
        plt.ylabel(f"{self.ydata_param_name} ({self.get_dependent_parameter_by_name(self.ydata_param_name)['paramspec'].unit})")

        plt.plot(self.xdata, self.model.eval(params=self.fit_result.params, Vg=self.xdata), '-', color='red')
        plt.scatter(self.V_max_deriv, self.model.eval(params=self.fit_result.params, Vg=self.V_max_deriv), color='red')
        plt.text(self.V0-self.Gamma/6, self.A*2/3 + self.offset, f"V = {self.V_max_deriv:.5f} V", color='red')
        plt.title(f"Run #{self.run_id}")

        return self.V_max_deriv
    
    def _find_max_derivative_point_numeric(self, ydata_param_name, shoulder='left', filter=True, window_size=10):
        """
        Calibrate the charge sensor operation point by fitting the Coulomb peak with a Lorentzian and 
        looking at the point with maximum derivative of the reflected amplitude.
        
        Params:
            shoulder: Either 'left' or 'right'

        Returns:
            The gate voltage corresponding to the point with maximum derivative.
        """
        self.ydata = self.get_dependent_parameter_by_name(ydata_param_name)['values']
        self.ydata_param_name = ydata_param_name

        if filter:
            # print(f"Filtering CS data with a moving average of window size {window_size}.")
            # Apply a moving average filter to the ydata. Need to cut initial and final points to avoid edge effects
            self.ydata_filtered = np.convolve(self.ydata, np.ones(window_size)/window_size, mode='same')[window_size//2:-(window_size//2)]
            self.xdata_filtered = self.xdata[window_size//2:-(window_size//2)]
        else:
            self.ydata_filtered = self.ydata
            self.xdata_filtered = self.xdata
            
        # Evaluate derivative
        self.y_derivative = np.gradient(self.ydata_filtered)

        # if filter:
        #     # Apply a moving average filter to the derivative
        #     self.y_derivative = np.convolve(self.y_derivative, np.ones(window_size)/window_size, mode='same')

        if shoulder == "left":
            self.V_max_deriv = self.xdata_filtered[np.argmax(self.y_derivative)]
        elif shoulder == "right":
            self.V_max_deriv = self.xdata_filtered[np.argmin(self.y_derivative)]
        else:
            raise ValueError("Invalid option for 'shoulder' parameter: must be either 'left' or 'right'.")


        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        fig.suptitle(f"Run #{self.run_id}")

        ax1.set_title("Raw data")
        ax1.plot(self.xdata, self.ydata, '.', color='k', label='Data')
        ax1.set_xlabel("Gate voltage (V)")
        ax1.set_ylabel(f"{self.ydata_param_name} ({self.get_dependent_parameter_by_name(self.ydata_param_name)['paramspec'].unit})")
        ax1.axvline(x = self.V_max_deriv, ls="-", color = 'red')

        ax2.set_title(f"Filtered data, moving average window size = {window_size}")
        ax2.plot(self.xdata_filtered, self.ydata_filtered, '.', color='k', label='Data')
        ax2.set_xlabel("Gate voltage (V)")
        ax2.set_ylabel(f"{self.ydata_param_name} ({self.get_dependent_parameter_by_name(self.ydata_param_name)['paramspec'].unit})")
        ax2.axvline(x = self.V_max_deriv, ls="-", color = 'red')

        ax3.set_title("Derivative of filtered data")
        ax3.plot(self.xdata_filtered, self.y_derivative, '.', color='red', label='Derivative')
        ax3.set_xlabel("Gate voltage (V)")
        ax3.set_ylabel(f"{self.ydata_param_name} derivative ({self.get_dependent_parameter_by_name(self.ydata_param_name)['paramspec'].unit})")
        ax3.axvline(x = self.V_max_deriv, ls="-", color = 'red')

        return self.V_max_deriv

class STOscillationsVSDetuning(DataSet):
    def __init__(self, exp, run_id=None):
        super().__init__(exp=exp, run_id=run_id)
        self.detuning = self.independent_parameters['x']['values']
        self.evol_time = self.independent_parameters['y']['values']
        self.fft(axis=0)
        self.format = format
        try:
            self.signal = self.get_dependent_parameter_by_name('Amplitude')['values']
            self.format = 'amplitude'
            self.fft = self.dependent_parameters['param_0_fft']['values']
            self.fft_freq = self.get_independent_parameter_by_name('freq')['values']
        except:
            self.format = 'mag-phase'
            self.mag = self.get_dependent_parameter_by_name('Magnitude')['values']
            self.phase = self.get_dependent_parameter_by_name('Phase')['values']
            self.fft_mag = self.dependent_parameters['param_0_fft']['values']
            self.fft_phase = self.dependent_parameters['param_1_fft']['values']
            self.fft_freq = self.get_independent_parameter_by_name('freq')['values']
        else:
            raise ValueError(f"Invalid format '{format}'. Must be either 'amplitude' or 'mag-phase'.")

    def plot(self):
        if self.format == 'amplitude':
            fig, ax = plt.subplots(1,1)
            fig.suptitle(f'Run #{self.run_id}', y=1.01)
            plot = ax.pcolormesh(self.detuning*1e3, self.evol_time, self.signal*1e3)
            fig.colorbar(plot, label='Amplitude (mV)')
            ax.set_ylabel('Evolution time (ns)')
            ax.set_xlabel('Detuning (mV)')
            plt.show()

            # add fft in new figure
            fig_fft, ax_fft = plt.subplots(1,1)
            fig_fft.suptitle(f'Run #{self.run_id}', y=1.01)
            plot_fft = ax_fft.pcolormesh(self.detuning*1e3, self.fft_freq*1e-6, self.fft*1e3)
            fig_fft.colorbar(plot_fft, label='FFT (mV)')
            ax_fft.set_ylabel('Frequency (MHz)')
            ax_fft.set_xlabel('Detuning (mV)')

            return [fig, fig_fft], [ax, ax_fft], [plot, plot_fft]
            
        elif self.format == 'mag-phase':
            figsize = (11, 4)
            fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True)
            plt.subplots_adjust(wspace=0.3)
            fig.suptitle(f'Run #{self.run_id}')

            plot1 = ax[0].pcolormesh(self.detuning*1e3, self.evol_time, self.mag*1e3)
            fig.colorbar(plot1, ax=ax[0], label='Magnitude (mV)')
            plot2 = ax[1].pcolormesh(self.detuning*1e3, self.evol_time, self.phase)
            fig.colorbar(plot2, ax=ax[1], label='Phase (deg)')

            ax[0].set_ylabel('Evolution time (ns)')
            ax[1].set_ylabel('Evolution time (ns)')
            ax[1].set_xlabel('Detuning (mV)')
            ax[0].set_xlabel('Detuning (mV)')

            # add fft in new figure
            fig_fft, ax_fft = plt.subplots(1, 2, figsize=figsize, sharex=True)
            plt.subplots_adjust(wspace=0.3)
            fig_fft.suptitle(f'Run #{self.run_id}')
            
            plot1_fft = ax_fft[0].pcolormesh(self.detuning*1e3, self.fft_freq/1e6, self.fft_mag*1e3)
            fig_fft.colorbar(plot1_fft, ax=ax_fft[0], label='Magnitude FFT (mV)')
            plot2_fft = ax_fft[1].pcolormesh(self.detuning*1e3, self.fft_freq/1e6, self.fft_phase)
            fig_fft.colorbar(plot2_fft, ax=ax_fft[1], label='Phase FFT (deg)')
            
            ax_fft[0].set_ylabel('Frequency (MHz)')
            ax_fft[1].set_ylabel('Frequency (MHz)')
            ax_fft[1].set_xlabel('Detuning (mV)')
            ax_fft[0].set_xlabel('Detuning (mV)')
            
            return [fig, fig_fft], [ax, ax_fft], [plot1, plot2, plot1_fft, plot2_fft]

class OPXPulsedChargeStabilityDiagram(DataSet):
    def __init__(self, exp, run_id=None, attenuation=1):
        super().__init__(exp=exp, run_id=run_id)
        self.x_opx_vals = self.independent_parameters['x']['values']
        self.y_opx_vals = self.independent_parameters['y']['values']
        self.independent_parameters['x']['values'] = self.independent_parameters['x']['values']/attenuation
        self.independent_parameters['y']['values'] = self.independent_parameters['y']['values']/attenuation
        self.attenuation = attenuation

    def plot(self, format='mag-phase', transpose=True):
        if self.attenuation == 1:
            self._plot_single_axis(format=format, transpose=transpose)
        else:
            self._plot_double_axis(format=format, transpose=True)

    def _plot_single_axis(self, format='mag-phase', transpose=True):
        if format == 'mag-phase':
            fig, axs = plt.subplots(1, 2, figsize=(14, 5))
            # plt.subplots_adjust(wspace=0.4)
            # Magnitude plot
            if transpose:
                x = self.independent_parameters['y']['values']
                x_label = self.independent_parameters['y']['paramspec'].label
                y = self.independent_parameters['x']['values']
                y_label = self.independent_parameters['x']['paramspec'].label
                mag = self.dependent_parameters['param_0']['values'].T
                mag_label = self.dependent_parameters['param_0']['paramspec'].label
                phase = self.dependent_parameters['param_1']['values'].T
                phase_label = self.dependent_parameters['param_1']['paramspec'].label
            else:
                x = self.independent_parameters['x']['values']
                x_label = self.independent_parameters['x']['paramspec'].label
                y = self.independent_parameters['y']['values']
                y_label = self.independent_parameters['y']['paramspec'].label
                mag = self.dependent_parameters['param_0']['values']
                mag_label = self.dependent_parameters['param_0']['paramspec'].label
                phase = self.dependent_parameters['param_1']['values']
                phase_label = self.dependent_parameters['param_1']['paramspec'].label
        
            plot0 = axs[0].pcolormesh(x*1e3, y*1e3, mag*1e3)
            axs[0].set_xlabel("Real " + x_label + " (mV)")
            axs[0].set_ylabel("Real " + y_label + " (mV)")
            cbar = fig.colorbar(plot0, label= mag_label + " (mV)", location='right', shrink=1.0)
        
            plot0 = axs[1].pcolormesh(x*1e3, y*1e3, phase)
            axs[1].set_xlabel("Real " + x_label + " (mV)")
            axs[1].set_ylabel("Real " + y_label + " (mV)")
            fig.colorbar(plot0, label= phase_label + " (deg)", location='right', shrink=1.0)
        
        
        fig.suptitle(f"Run #{self.run_id}", y=0.98)


    def _plot_double_axis(self, format='mag-phase', transpose=True):
        if format == 'mag-phase':
            fig, axs = plt.subplots(1, 2, figsize=(12, 6.5))
            plt.subplots_adjust(wspace=0.5)
            # Magnitude plot
            if transpose:
                x = self.independent_parameters['y']['values']
                x_opx = self.y_opx_vals
                x_label = self.independent_parameters['y']['paramspec'].label
                y = self.independent_parameters['x']['values']
                y_opx = self.x_opx_vals
                y_label = self.independent_parameters['x']['paramspec'].label
                mag = self.dependent_parameters['param_0']['values'].T
                mag_label = self.dependent_parameters['param_0']['paramspec'].label
                phase = self.dependent_parameters['param_1']['values'].T
                phase_label = self.dependent_parameters['param_1']['paramspec'].label
            else:
                x = self.independent_parameters['x']['values']
                x_opx = self.x_opx_vals
                x_label = self.independent_parameters['x']['paramspec'].label
                y = self.independent_parameters['y']['values']
                y_opx = self.y_opx_vals
                y_label = self.independent_parameters['y']['paramspec'].label
                mag = self.dependent_parameters['param_0']['values']
                mag_label = self.dependent_parameters['param_0']['paramspec'].label
                phase = self.dependent_parameters['param_1']['values']
                phase_label = self.dependent_parameters['param_1']['paramspec'].label

            plot0 = axs[0].pcolormesh(x*1e3, y*1e3, mag*1e3)
            axs[0].set_xlabel("Real " + x_label + " (mV)")
            axs[0].set_ylabel("Real " + y_label + " (mV)")
            cbar = fig.colorbar(plot0, label= mag_label + " (mV)", location='top', shrink=0.8, pad=0.12)

            plot0 = axs[1].pcolormesh(x*1e3, y*1e3, phase)
            axs[1].set_xlabel("Real " + x_label + " (mV)")
            axs[1].set_ylabel("Real " + y_label + " (mV)")
            fig.colorbar(plot0, label= phase_label + " (deg)", location='top', shrink=0.8, pad=0.12)
            
            # Add other axes for raw opx values
            for ax_idx in range(2):
                ax0x = axs[ax_idx].twiny()
                ax0x.set_xlim(min(x_opx)*1e3, max(x_opx)*1e3)
                ax0x.set_xlabel("OPX " + x_label + " (mV)", color='grey')
                ax0x.tick_params(colors="grey")
                ax0y = axs[ax_idx].twinx()
                ax0y.set_ylim(min(y_opx)*1e3, max(y_opx)*1e3)
                ax0y.set_ylabel("OPX " + y_label + " (mV)", color='grey')
                ax0y.tick_params(colors="grey")

        fig.suptitle(f"Run #{self.run_id}", y=0.8)

class STOscillationsVSDetuningOld(DataSet):
    def __init__(self, exp, run_id=None, detuning=None, format='amplitude', flip_in_detuning=True):
        super().__init__(exp=exp, run_id=run_id)
        self.opx_detuning = self.independent_parameters['x']['values']
        self.detuning = detuning
        self.evol_time = self.independent_parameters['y']['values']
        self.fft(axis=0)
        self.format = format
        if format == 'amplitude':
            self.signal = self.get_dependent_parameter_by_name('Amplitude')['values']
            self.fft = self.dependent_parameters['param_0_fft']['values']
            self.fft_freq = self.get_independent_parameter_by_name('freq')['values']
            if flip_in_detuning:
                self.signal = np.flip(self.signal, axis=1)
                self.fft = np.flip(self.fft, axis=1)
        elif format == 'mag-phase':
            self.mag = self.get_dependent_parameter_by_name('Magnitude')['values']
            self.phase = self.get_dependent_parameter_by_name('Phase')['values']
            self.fft_mag = self.dependent_parameters['param_0_fft']['values']
            self.fft_phase = self.dependent_parameters['param_1_fft']['values']
            self.fft_freq = self.get_independent_parameter_by_name('freq')['values']
            if flip_in_detuning:
                self.mag = np.flip(self.mag, axis=1)
                self.phase = np.flip(self.phase, axis=1)
                self.fft_mag = np.flip(self.fft_mag, axis=1)
                self.fft_phase = np.flip(self.fft_phase, axis=1)
        else:
            raise ValueError(f"Invalid format '{format}'. Must be either 'amplitude' or 'mag-phase'.")

    def plot(self, show_opx_detuning=True):
        if self.format == 'amplitude':
            fig, ax = plt.subplots(1,1)
            fig.suptitle(f'Run #{self.run_id}', y=1.01)

            plot = ax.pcolormesh(self.detuning*1e3, self.evol_time, self.signal*1e3)
            fig.colorbar(plot, label='Amplitude (mV)')

            ax.set_ylabel('Evolution time (ns)')
            ax.set_xlabel('Detuning (mV)')
            if show_opx_detuning:
                ax2 = ax.twiny()
                ax2.set_xlim(self.opx_detuning[-1]*1e3, self.opx_detuning[0]*1e3)
                ax2.set_xlabel('OPX Detuning (mV)')
            plt.show()

            # add fft in new figure
            fig_fft, ax_fft = plt.subplots(1,1)
            fig_fft.suptitle(f'Run #{self.run_id}', y=1.01)

            plot_fft = ax_fft.pcolormesh(self.detuning*1e3, self.fft_freq*1e-6, self.fft*1e3)
            fig_fft.colorbar(plot_fft, label='FFT (mV)')

            ax_fft.set_ylabel('Frequency (MHz)')
            ax_fft.set_xlabel('Detuning (mV)')

            if show_opx_detuning:
                ax2 = ax_fft.twiny()
                ax2.set_xlim(self.opx_detuning[-1]*1e3, self.opx_detuning[0]*1e3)
                ax2.set_xlabel('OPX Detuning (mV)')
            plt.show()

            return [fig, fig_fft], [ax, ax_fft], [plot, plot_fft]
            
        elif self.format == 'mag-phase':
            
            fig, ax = plt.subplots(1, 2, figsize=(6.5, 4.0), sharex=True)
            fig.suptitle(f'Run #{self.run_id}', y=1.01)

            plot1 = ax[0].pcolormesh(self.detuning*1e3, self.evol_time, self.mag*1e3)
            fig.colorbar(plot1, ax=ax[0], label='Magnitude (mV)')
            plot2 = ax[1].pcolormesh(self.detuning*1e3, self.evol_time, self.phase)
            fig.colorbar(plot2, ax=ax[1], label='Phase (deg)')

            ax[1].set_ylabel('Evolution time (ns)')
            ax[1].set_xlabel('Detuning (mV)')
            ax02 = ax[0].twiny()
            ax02.set_xlim(self.opx_detuning[0]*1e3, self.opx_detuning[-1]*1e3)
            ax02.set_xlabel('OPX Detuning (mV)')

            ax12 = ax[1].twiny()
            ax12.set_xlim(self.opx_detuning[0]*1e3, self.opx_detuning[-1]*1e3)
            ax12.set_xlabel('OPX Detuning (mV)')


class Measurement2DVSDetuningOld(DataSet):
    def __init__(self, exp, run_id=None, detuning=None, detuning_axis=1, format='amplitude', flip_in_detuning=True):
        """
        Args:
            exp: Experiment
            run_id: Run id
            detuning: Actual detuning array at the device
            detuning_axis: 1 if the detuning is on the x axis in the plotter, 0 if on the y axis
            format: Either 'amplitude' or 'mag-phase'
            flip_in_detuning: True if the opx detuning has opposite sign as the real detuning
        """
        super().__init__(exp=exp, run_id=run_id)
        self.detuning_axis = detuning_axis
        if detuning_axis == 1:
            self.opx_detuning = self.independent_parameters['x']['values']
            self.detuning = detuning
            self.other_axis = self.independent_parameters['y']['values']
            self.other_axis_param = self.independent_parameters['y']
        elif detuning_axis == 0:
            self.opx_detuning = self.independent_parameters['y']['values']
            self.detuning = detuning
            self.other_axis = self.independent_parameters['x']['values']
            self.other_axis_param = self.independent_parameters['x']
        self.format = format
        if format == 'amplitude':
            self.signal = self.get_dependent_parameter_by_name('Amplitude')['values']
            if flip_in_detuning:
                self.signal = np.flip(self.signal, axis=detuning_axis)
        elif format == 'mag-phase':
            self.mag = self.get_dependent_parameter_by_name('Magnitude')['values']
            self.phase = self.get_dependent_parameter_by_name('Phase')['values']
            if flip_in_detuning:
                self.mag = np.flip(self.mag, axis=detuning_axis)
                self.phase = np.flip(self.phase, axis=detuning_axis)
        else:
            raise ValueError(f"Invalid format '{format}'. Must be either 'amplitude' or 'mag-phase'.")

    def plot(self, show_opx_detuning=True, detuning_name='Detuning'):
        if self.format == 'amplitude':
            fig, ax = plt.subplots(1,1)
            fig.suptitle(f'Run #{self.run_id}', y=1.01)
    
            plot = ax.pcolormesh(self.detuning*1e3, self.other_axis, self.signal*1e3)
            fig.colorbar(plot, label='Amplitude (mV)')
    
            ax.set_ylabel(f"{self.other_axis_param['paramspec'].label} ({self.other_axis_param['paramspec'].unit})")
            ax.set_xlabel(f'{detuning_name} (mV)')
            if show_opx_detuning:
                ax2 = ax.twiny()
                ax2.set_xlim(self.opx_detuning[-1]*1e3, self.opx_detuning[0]*1e3)
                ax2.set_xlabel(f'OPX {detuning_name} (mV)')
            plt.show()
    
        elif self.format == 'mag-phase':
    
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.0))
            fig.suptitle(f'Run #{self.run_id}', y=1.01)
    
            plot1 = ax[0].pcolormesh(self.detuning*1e3, self.other_axis, self.mag*1e3)
            fig.colorbar(plot1, ax=ax[0], label='Magnitude (mV)')
            plot2 = ax[1].pcolormesh(self.detuning*1e3, self.other_axis, self.phase)
            fig.colorbar(plot2, ax=ax[1], label='Phase (deg)')

            ax[0].set_xlabel(f'{detuning_name} (mV)')
            ax[0].set_ylabel(f"{self.other_axis_param['paramspec'].label} ({self.other_axis_param['paramspec'].unit})")
            ax[1].set_ylabel(f"{self.other_axis_param['paramspec'].label} ({self.other_axis_param['paramspec'].unit})")
            ax[1].set_xlabel(f'{detuning_name} (mV)')

            if show_opx_detuning:
                ax02 = ax[0].twiny()
                ax02.set_xlim(self.opx_detuning[0]*1e3, self.opx_detuning[-1]*1e3)
                ax02.set_xlabel(f'OPX {detuning_name} (mV)')
                ax12 = ax[1].twiny()
                ax12.set_xlim(self.opx_detuning[0]*1e3, self.opx_detuning[-1]*1e3)
                ax12.set_xlabel(f'OPX {detuning_name} (mV)')