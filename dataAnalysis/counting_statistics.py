"""
Module: counting_statistics.py

A class to extract tunneling times (tau) and calculate tunneling rates (Gamma)
from square pulse time-trace data using threshold detection.

This class is designed for experimental analysis in mesoscopic or quantum dot systems
where square pulse modulation is used to probe tunneling dynamics.

Author: [yryu / HQC]
Date: [2025-06-28]
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.optimize import curve_fit
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from dataAnalysis.base import DataSet
class PulseTunnelingAnalysis(DataSet):
    """
    A class for analyzing pulse train traces to extract tunneling times (tau_in, tau_out)
    and compute corresponding tunneling rates (Gamma_in, Gamma_out).
    """
      
    def __init__(self,exp,run_id = None, station = None, save_path=None,charge_state = 'low'):
        """
        Parameters:
            time_data (2D np.ndarray): Time traces (shape: [N_pulse, N_timepoints])
            voltage_data (2D np.ndarray): Voltage signal traces (same shape as time_data)
            pulse_amplitudes (1D np.ndarray): Square pulse amplitude values (N_pulse,)
            run_id (int): Identifier for the measurement run
            save_path (str): Directory to save plots and results
        """
        super().__init__(exp=exp, run_id=run_id, station=station)
        self.time = self.independent_parameters['y']['values']
        self.pulse_amplitudes = self.independent_parameters['x']['values']
        self.CS_values = self.dependent_parameters['param_0']['values'].T
        self.save_path = save_path
        self.run_id = run_id
        self.charge_state = charge_state # Charge state 'high' if voltage level is high level for hall occupation. 'low' for no hall occupation
        
        if type(save_path) == str:
            os.makedirs(save_path,exist_ok = True)
        
    def extract_taus(self, level=0.02, threshold=0.6, plot=False):
        """
        Perform tau extraction and Gamma calculation for all amplitudes.

        Parameters:
            level (float): Threshold base level
            threshold (float): Multiplier to define detection threshold
            plot (bool): Whether to save plots for each amplitude

        Returns:
            pd.DataFrame: DataFrame summarizing tau and Gamma statistics for each amplitude
        """
        results = []

        for idx in range(len(self.pulse_amplitudes)):
                        
            time = self.time
            amp = self.pulse_amplitudes[idx]
            CS_values = self.CS_values[idx]
            
            tau_in, tau_out = self._extract_single_trace(time, CS_values, level, threshold,
                                                         plot_figure=plot, amplitude=amp)
            
            tau_in_arr = np.array(tau_in)
            tau_out_arr = np.array(tau_out)
                    
            tau_in_mean = np.mean(tau_in_arr) if len(tau_in_arr) > 0 else np.nan
            tau_out_mean = np.mean(tau_out_arr) if len(tau_out_arr) > 0 else np.nan
            tau_in_std = np.std(tau_in_arr) if len(tau_in_arr) > 0 else 0
            tau_out_std = np.std(tau_out_arr) if len(tau_out_arr) > 0 else 0
            n_in = len(tau_in_arr)
            n_out = len(tau_out_arr)

            gamma_in = 1 / tau_in_mean if tau_in_mean > 0 else np.nan
            gamma_out = 1 / tau_out_mean if tau_out_mean > 0 else np.nan

            gamma_in_std = (tau_in_std / (tau_in_mean ** 2 * np.sqrt(n_in))) if tau_in_mean > 0 and n_in > 0 else np.nan # I am not sure about this. 
            gamma_out_std = (tau_out_std / (tau_out_mean ** 2 * np.sqrt(n_out))) if tau_out_mean > 0 and n_out > 0 else np.nan 

            results.append({
                "Amplitude": amp,
                "tau_in_mean": tau_in_mean,
                "tau_in_std": tau_in_std,
                "tau_in_n": n_in,
                "tau_out_mean": tau_out_mean,
                "tau_out_std": tau_out_std,
                "tau_out_n": n_out,
                "Gamma_in": gamma_in,
                "Gamma_in_std": gamma_in_std,
                "Gamma_out": gamma_out,
                "Gamma_out_std": gamma_out_std,
                "tau_in": tau_in,
                "tau_out": tau_out
                })

        self.df_results = pd.DataFrame(results)
        if type(self.save_path) == str:
            self.df_results.to_csv(os.path.join(self.save_path, f"tau_summary_Run{self.run_id}.csv"), index=False)
        return self.df_results

    def _extract_single_trace(self, time, voltage, level, threshold, plot_figure=False, amplitude=None):
        time_step = np.diff(time)[0]
        voltage = voltage - np.mean(voltage)

        up_index = voltage > 0
        down_index = voltage < 0
        up_event = voltage[up_index] - np.mean(voltage[up_index])
        down_event = voltage[down_index] - np.mean(voltage[down_index])
        up_time = time[up_index]
        down_time = time[down_index]

        high_level = level * threshold
        base_threshold = high_level / 2

        up_peaks, _ = find_peaks(up_event, height=high_level, distance=100)
        down_peaks, _ = find_peaks(np.abs(down_event), height=high_level, distance=100)

        tau_in_array = []
        tau_out_array = []

        for peak_idx in down_peaks:
            k = 0
            while peak_idx - k - 1 > 0 and down_event[peak_idx - k - 1] < -high_level:
                k += 1
            peak_idx -= k
            k = 0
            while peak_idx + k < len(down_event):
                if down_event[peak_idx + k] > -base_threshold:
                    tau_out_array.append(k * time_step)
                    break
                k += 1

        for peak_idx in up_peaks:
            k = 0
            while peak_idx - k - 1 > 0 and up_event[peak_idx - k - 1] > high_level:
                k += 1
            peak_idx -= k
            k = 0
            while peak_idx + k < len(up_event):
                if up_event[peak_idx + k] < base_threshold:
                    tau_in_array.append(k * time_step)
                    break
                k += 1

        if plot_figure and type(self.save_path) == str:
            self._plot_trace(time, voltage, up_time, up_event, up_peaks,
                             down_time, down_event, down_peaks, amplitude)
        
        if self.charge_state == 'low':
            return tau_in_array, tau_out_array
        else:
            return tau_out_array, tau_in_array

    def _plot_trace(self, time, voltage, up_time, up_event, up_peaks,
                    down_time, down_event, down_peaks, amplitude):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(down_time, down_event, label='down_event')
        axes[0].plot(down_time[down_peaks], down_event[down_peaks], 'rx', label='down_peaks')
        axes[0].set_title("Fall Events (tau_out)")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(up_time, up_event, label='up_event')
        axes[1].plot(up_time[up_peaks], up_event[up_peaks], 'gx', label='up_peaks')
        axes[1].set_title("Rise Events (tau_in)")
        axes[1].grid(True)
        axes[1].legend()

        axes[2].plot(time, voltage, label='raw_data')
        axes[2].hlines(np.mean(voltage[voltage > 0]), np.min(time), np.max(time), linestyles='--', colors='r', label='Up state')
        axes[2].hlines(np.mean(voltage[voltage < 0]), np.min(time), np.max(time), linestyles='--', colors='b', label='Down state')
        axes[2].set_title('Raw Signal')
        axes[2].grid(True)
        axes[2].legend()
        axes[2].set_xlabel('Time (s)')

        plt.tight_layout()
        if amplitude is not None:
            filename = f"Run_{self.run_id}_tau_plot_amp_{amplitude:.6f}.png"
            plt.savefig(os.path.join(self.save_path, filename), dpi=300)
            #print(f"[+] Saved plot to: {filename}")
        plt.close()

    def plot_gamma_vs_amplitude(self):
        """Plot Gamma_in and Gamma_out as a function of pulse amplitude."""
        if not hasattr(self, 'df_results'):
            raise RuntimeError("Please run extract_taus() first to compute Gamma values.")

        df = self.df_results.sort_values(by='Amplitude')
        amp = df['Amplitude'] * 1e3  # mV
        gamma_in = df['Gamma_in'] / 1e3  # kHz
        gamma_in_std = df['Gamma_in_std'] / 1e3
        gamma_out = df['Gamma_out'] / 1e3
        gamma_out_std = df['Gamma_out_std'] / 1e3

        plt.figure(figsize=(10, 6))
        plt.errorbar(amp, gamma_in, yerr=gamma_in_std, fmt='o-', capsize=4, label=r'$\Gamma_{in}$')
        plt.errorbar(amp, gamma_out, yerr=gamma_out_std, fmt='s--', capsize=4, label=r'$\Gamma_{out}$')
        plt.xlabel('Square Pulse Amplitude (mV)', fontsize=12)
        plt.ylabel(r'$\Gamma$ (kHz)', fontsize=12)
        plt.title(r'Tunneling Rate $\Gamma$ vs Pulse Amplitude', fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        

class StandardGateSweepAnalysis(DataSet):
    """
    A class to analyze voltage traces taken while sweeping gate voltage (Vg),
    using threshold crossing and histogram-based techniques to extract tau and Gamma.
    """
    def __init__(self, exp, run_id=None, station=None, save_path=None, diff_guess=None,charge_state='low',time_slice=0.5):
        super().__init__(exp=exp, run_id=run_id, station=station)
        self.time = self.independent_parameters['y']['values']
        self.gate_voltage = self.independent_parameters['x']['values']
        self.CS_values = self.dependent_parameters['param_0']['values'].T
        self.save_path = save_path
        self.diff_guess = diff_guess
        self.run_id = run_id
        self.time_slice = time_slice
        self.charge_state = charge_state # Charge state 'high' if voltage level is high level for hall occupation. 'low' for no hall occupation
        
        if type(save_path) == str:
            os.makedirs(save_path,exist_ok = True)
            
    def extract_taus(self, plot=False):
        results = []

        for idx in range(len(self.CS_values)):
            time = self.time
            gate_voltage = self.gate_voltage[idx]
            CS_values = self.CS_values[idx]
            
            threshold, means, covs, weights, use_gmm = self._estimate_threshold(CS_values)
            rise_times, fall_times = self._detect_events(time, CS_values, threshold, means)
            
            tau_in, tau_out = self._compute_taus(rise_times, fall_times)
            

            if len(tau_in) < 2 or len(tau_out) < 2:
                continue

            stats = self._summarize_taus(tau_in, tau_out, gate_voltage,fall_times)
            results.append(stats)

            if plot and type(self.save_path) == str:
                self._plot_trace(time, CS_values, threshold, rise_times, fall_times,
                                 means, covs, weights, use_gmm, gate_voltage, idx)

        self.df_results = pd.DataFrame(results)
        if type(self.save_path) == str:
            self.df_results.to_csv(os.path.join(self.save_path, f"standard_tau_summary_Run{self.run_id}.csv"), index=False)
        return self.df_results

    def plot_gamma_vs_gate_voltage(self):
        """Plot Gamma_in and Gamma_out as a function of gate voltage."""
        if not hasattr(self, 'df_results'):
            raise RuntimeError("Please run extract_taus() first to compute Gamma values.")

        df = self.df_results.sort_values(by='GateVoltage')
        Vg = df['GateVoltage'] * 1e3  # mV
        gamma_in = df['Gamma_in'] / 1e3  # kHz
        gamma_in_std = df['Gamma_in_std'] / 1e3
        gamma_out = df['Gamma_out'] / 1e3
        gamma_out_std = df['Gamma_out_std'] / 1e3

        plt.figure(figsize=(10, 6))
        plt.errorbar(Vg, gamma_in, yerr=gamma_in_std, fmt='o-', capsize=4, label=r'$\Gamma_{\mathrm{in}}$')
        plt.errorbar(Vg, gamma_out, yerr=gamma_out_std, fmt='s--', capsize=4, label=r'$\Gamma_{\mathrm{out}}$')
        plt.xlabel('Gate Voltage (mV)', fontsize=12)
        plt.ylabel(r'$\Gamma$ (kHz)', fontsize=12)
        plt.title(r'Tunneling Rate $\Gamma$ vs Gate Voltage', fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_tau_hist(self, Vg_index=0, bins=50):
        """
        Plot histograms of tau_in and tau_out for a given gate voltage index.
        
        Parameters:
        Vg_index (int): Index of the gate voltage point in df_results to plot.
        bins (int): Number of histogram bins.
        """
        if not hasattr(self, 'df_results'):
            raise RuntimeError("Please run extract_taus() first to compute tau/Gamma values.")
            
        if Vg_index >= len(self.df_results):
            raise IndexError(f"Vg_index {Vg_index} is out of bounds for result length {len(self.df_results)}.")

        row = self.df_results.iloc[Vg_index]
        tau_in = row['tau_in']
        tau_out = row['tau_out']
        Vg = row['GateVoltage'] * 1e3  # Convert to mV for labeling

        if len(tau_in) == 0 or len(tau_out) == 0:
            print(f"[!] No tau data found at index {Vg_index} (Vg = {Vg:.3f} mV)")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Tunneling Time Histograms @ Vg = {Vg:.2f} mV", fontsize=14)

        axes[0].hist(tau_in, bins=bins, color='green', alpha=0.7)
        axes[0].set_title(r"$\tau_{\mathrm{in}}$ distribution")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Count")
        axes[0].grid(True)

        axes[1].hist(tau_out, bins=bins, color='blue', alpha=0.7)
        axes[1].set_title(r"$\tau_{\mathrm{out}}$ distribution")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Count")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
        
        print(f"tau_in: {np.mean(tau_in)} and tau_out: {np.mean(tau_out)}")

    def _estimate_threshold(self, voltage):
        diff_guess = self.diff_guess
        try:
            gmm = GaussianMixture(n_components=2, random_state=0)
            gmm.fit(voltage.reshape(-1, 1))
            means = gmm.means_.flatten()
            covs = np.sqrt(gmm.covariances_.flatten())
            weights = gmm.weights_
            threshold = np.mean(means)
            mean_separation = np.abs(np.diff(means))[0]
            if mean_separation < 0.5 * diff_guess:
                raise ValueError("GMM means too close")
            return threshold, means, covs, weights, True
        except:
            hist, bins = np.histogram(voltage, bins=100, density=True)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            def single_gaussian(x, mu, sigma, amp):
                return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            guess = [np.mean(voltage), np.std(voltage), np.max(hist)]
            try:
                popt, _ = curve_fit(single_gaussian, bin_centers, hist, p0=guess)
                base_level = popt[0]
            except:
                base_level = np.median(voltage)
            if base_level < np.mean(voltage):
                threshold = base_level + 0.5 * diff_guess
                means = np.array([base_level, base_level + diff_guess])
                weights = np.array([1, 0])
            else:
                threshold = base_level - 0.5 * diff_guess
                means = np.array([base_level - diff_guess, base_level])
                weights = np.array([0, 1])
            covs = np.array([np.std(voltage)] * 2)
            return threshold, means, covs, weights, False

    def _detect_events(self, time, voltage, threshold, means):
        hysteresis_fraction = 0.2
        low_level, high_level = np.min(means), np.max(means)
        span = high_level - low_level
        thr_high = threshold + hysteresis_fraction * span
        thr_low = threshold - hysteresis_fraction * span

        rise_indices, fall_indices = [], []
        state_low = voltage[0] < threshold
        for i in range(1, len(voltage)):
            if state_low and voltage[i] >= thr_high:
                rise_indices.append(i)
                state_low = False
            elif not state_low and voltage[i] <= thr_low:
                fall_indices.append(i)
                state_low = True

        return time[rise_indices], time[fall_indices]
    
    def _cumulants(self,time, fall_time, time_slice = 0.1):
        
        bins_ = np.arange(0, np.max(time) + time_slice, time_slice)
        counts_, _ = np.histogram(fall_time, bins = bins_)
        
        c1 = np.mean(counts_)
        c2 = np.var(counts_)
        
        return c1, c2
    
    def _compute_taus(self, rise_times, fall_times):
        
        if len(rise_times) < 2 or len(fall_times) < 2:
            return [], []
        
        min_len = min(len(rise_times), len(fall_times))
        rise_times = rise_times[:min_len]
        fall_times = fall_times[:min_len]

        if rise_times[0] < fall_times[0]:
            tau_in = fall_times - rise_times
            tau_out = rise_times[1:] - fall_times[:-1]
        else:
            tau_out = rise_times - fall_times
            tau_in = fall_times[1:] - rise_times[:-1]
            
        if self.charge_state == 'low':
            return tau_in,tau_out
        else:
            return tau_out,tau_in

    def _summarize_taus(self, tau_in, tau_out, Vg,fall_time):
        
        tau_in_arr = np.array(tau_in)
        tau_out_arr = np.array(tau_out)
                
        tau_in_mean = np.mean(tau_in_arr) if len(tau_in_arr) > 0 else np.nan
        tau_out_mean = np.mean(tau_out_arr) if len(tau_out_arr) > 0 else np.nan
        tau_in_std = np.std(tau_in_arr) if len(tau_in_arr) > 0 else 0
        tau_out_std = np.std(tau_out_arr) if len(tau_out_arr) > 0 else 0
        n_in = len(tau_in_arr)
        n_out = len(tau_out_arr)

        gamma_in = 1 / tau_in_mean if tau_in_mean > 0 else np.nan
        gamma_out = 1 / tau_out_mean if tau_out_mean > 0 else np.nan

        gamma_in_std = (tau_in_std / (tau_in_mean ** 2 * np.sqrt(n_in))) if tau_in_mean > 0 and n_in > 0 else np.nan
        gamma_out_std = (tau_out_std / (tau_out_mean ** 2 * np.sqrt(n_out))) if tau_out_mean > 0 and n_out > 0 else np.nan
        
        c1,c2 = self._cumulants(self.time, fall_time,time_slice=self.time_slice) 

        return {
            "GateVoltage": Vg,
            "tau_in_mean": tau_in_mean,
            "tau_in_std": tau_in_std,
            "tau_in_n": n_in,
            "tau_out_mean": tau_out_mean,
            "tau_out_std": tau_out_std,
            "tau_out_n": n_out,
            "Gamma_in": gamma_in,
            "Gamma_in_std": gamma_in_std,
            "Gamma_out": gamma_out,
            "Gamma_out_std": gamma_out_std,
            "tau_in": tau_in,
            "tau_out": tau_out,
            "c1": c1,
            "c2": c2
            }

    def _plot_trace(self, time, voltage, threshold, rise_times, fall_times,
                     means, covs, weights, use_gmm, Vg, idx):

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(f'Run #{self.run_id} Tunneling Summary - Vg = {Vg * 1e3:.2f} mV', fontsize=14)

        hist, bins = np.histogram(voltage, bins=100, density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        axs[0].bar(bin_centers, hist, width=bins[1] - bins[0], alpha=0.6, color='gray', label='Histogram')
        x_plot = np.linspace(np.min(voltage), np.max(voltage), 1000)
        if use_gmm:
            for i in range(2):
                axs[0].plot(x_plot, weights[i] * norm.pdf(x_plot, means[i], covs[i]), '--', label=f'GMM {i + 1}')
        else:
            def single_gaussian(x, mu, sigma, amp):
                return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            amp = np.max(hist)
            mu = np.mean(voltage)
            sigma = np.std(voltage)
            axs[0].plot(x_plot, single_gaussian(x_plot, mu, sigma, amp), '--', label='Single Gaussian Fit')

        axs[0].axvline(threshold, color='blue', linestyle='--', label=f'Threshold = {threshold:.3f}')
        axs[0].set_title("Voltage Histogram")
        axs[0].legend()

        axs[1].plot(time, voltage, color='gray', label='Signal')
        axs[1].plot(rise_times, voltage[np.searchsorted(time, rise_times)], 'go', label='Rise')
        axs[1].plot(fall_times, voltage[np.searchsorted(time, fall_times)], 'ro', label='Fall')
        axs[1].axhline(threshold, color='blue', linestyle='--', label='Threshold')
        axs[1].set_title("Tunneling Events")
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'Run_{self.run_id}_tunneling_summary_index_{idx}.png'), dpi=300)
        plt.close()


class counting_analysis:
    """
    A class to compare tunneling rates (Gamma) extracted from standard gate sweeps and pulsed measurements.
    """

    def __init__(self,exp,pulse_analysis_class: PulseTunnelingAnalysis, standard_analysis_class: StandardGateSweepAnalysis,
                 pulse_center, run_id_standard, run_id_pulse,diff_guess_standard,save_path=None,charge_state='low',time_slice = 0.5):
        """
        Parameters:
            pulse_analysis_class (PulseTunnelingAnalysis): Class for pulse-based analysis
            standard_analysis_class (StandardGateSweepAnalysis): Class for standard gate sweep analysis
            database_path (str): Path to the QCoDeS database file
            pulse_center (float): Center voltage for translating pulse amplitudes to gate voltages
            run_id_standard (int): Run ID for the standard sweep measurement
            run_id_pulse (int): Run ID for the pulse measurement
            save_path (str): Directory to save plots and results
            charge_state (str): "low" if occupied hall state correspondes to lower charge sensor voltage, "high" for otherwise. 
            cutoff (float): Pulse amplitude cutoff (in same units as amplitudes) for including pulse-based data
            time_slice (float): Time slice for cumulant calculation
        """
        
        self.pulse_analysis = pulse_analysis_class(exp, save_path=save_path,charge_state=charge_state,run_id = run_id_pulse)
        self.standard_analysis = standard_analysis_class(exp, save_path=save_path,diff_guess=diff_guess_standard,charge_state=charge_state,run_id=run_id_standard,time_slice=time_slice)
        self.pulse_center = pulse_center
        self.save_path = save_path
        
        
        if not hasattr(self.standard_analysis, 'df_results'):
            result1 = self.standard_analysis.extract_taus(plot=False)
        else:
            result1 = self.standard_analysis.df_results
        if not hasattr(self.pulse_analysis, 'df_results'):
            result2 = self.pulse_analysis.extract_taus(plot=False)
        else:
            result2 = self.pulse_analysis.df_results
        
        self.result1 = result1
        self.result2 = result2
        
        if type(self.save_path) == str:
            self.result1.to_csv(os.path.join(self.save_path, f"standard_tau_summary_Run{run_id_standard}.csv"), index=False)
            self.result2.to_csv(os.path.join(self.save_path, f"pulsed_tau_summary_Run{run_id_pulse}.csv"), index=False)
        
    def plot_combined_gamma(self,cutoff=2e-3):
        """
        Compare Gamma_in and Gamma_out from standard and pulse-based measurements on a log-scaled plot.
        """
        # Standard data
        standard_voltage = self.result1['GateVoltage']
        standard_gamma_in = self.result1['Gamma_in'] / 1e3
        standard_gamma_in_std = self.result1['Gamma_in_std'] / 1e3
        standard_gamma_out = self.result1['Gamma_out'] / 1e3
        standard_gamma_out_std = self.result1['Gamma_out_std'] / 1e3
        
        standard_c1 = self.result1['c1']
        standard_c2 = self.result1['c2']

        # Pulse data
        pulse_amp = self.result2['Amplitude'].values
        pulse_gamma_in = self.result2['Gamma_in'].values / 1e3
        pulse_gamma_in_std = self.result2['Gamma_in_std'].values / 1e3
        pulse_gamma_out = self.result2['Gamma_out'].values / 1e3
        pulse_gamma_out_std = self.result2['Gamma_out_std'].values / 1e3

        valid_idx = pulse_amp > cutoff
        pulse_voltage_out = pulse_amp[valid_idx] /  2 + self.pulse_center
        pulse_voltage_in = -pulse_amp[valid_idx] /  2 + self.pulse_center
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Gamma plot
        axs[0].errorbar(standard_voltage * 1e3, standard_gamma_in, yerr=standard_gamma_in_std, fmt='o-', capsize=4,
                        label=r'$\Gamma_{in}$(Standard)', mfc='tab:green', mec='tab:green', color='tab:green', alpha=0.5)
        axs[0].errorbar(standard_voltage * 1e3, standard_gamma_out, yerr=standard_gamma_out_std, fmt='s--', capsize=4,
                        label=r'$\Gamma_{out}$(Standard)', mfc='tab:red', mec='tab:red', color='tab:red', alpha=0.5)
        axs[0].errorbar(pulse_voltage_in * 1e3, pulse_gamma_in[valid_idx], yerr=pulse_gamma_in_std[valid_idx], fmt='d-', capsize=4,
                        label=r'$\Gamma_{in}$(Pulse)', mfc='tab:green', mec='tab:green', color='tab:green', alpha=0.5)
        axs[0].errorbar(pulse_voltage_out * 1e3, pulse_gamma_out[valid_idx], yerr=pulse_gamma_out_std[valid_idx], fmt='*--', capsize=4,
                        label=r'$\Gamma_{out}$(Pulse)', mfc='tab:red', mec='tab:red', color='tab:red', alpha=0.5)
        axs[0].set_ylabel(r'$\Gamma$ (kHz)', fontsize=12)
        axs[0].set_yscale('log')
        axs[0].set_title(r'Tunneling Rate $\Gamma$ vs Gate Voltage', fontsize=14)
        axs[0].grid(True)
        axs[0].legend()

        # Cumulants plot
        ax2 = axs[1]
        ax2.plot(standard_voltage * 1e3, standard_c1, 'o-', color='tab:blue', label='$C_1$')
        ax2.plot(standard_voltage * 1e3, standard_c2, 's--', color='tab:orange', label='$C_2$')
        ax2.set_xlabel('Gate voltage (mV)', fontsize=12)
        ax2.set_ylabel('Cumulant value', fontsize=12)
        ax2.set_title('Cumulants vs Gate Voltage', fontsize=14)
        ax2.grid(True)

        # Add c2/c1 on secondary y-axis
        ax3 = ax2.twinx()
        ratio = np.array(standard_c2) / np.array(standard_c1)
        ax3.plot(standard_voltage * 1e3, ratio, '^-', color='tab:purple', label='$C_2 / C_1$')
        ax3.set_ylabel(r'$C_2 / C_1$', fontsize=12)

        # Combine legends from both y-axes
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax3.set_ylim((-0.2,2.2)); ax2.set_ylim((-0.2,2.2))
        

        
























