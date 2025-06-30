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


class PulseTunnelingAnalysis:
    """
    A class for analyzing pulse train traces to extract tunneling times (tau_in, tau_out)
    and compute corresponding tunneling rates (Gamma_in, Gamma_out).
    """

    def __init__(self, time_data, voltage_data, pulse_amplitudes, run_id, save_path):
        """
        Parameters:
            time_data (2D np.ndarray): Time traces (shape: [N_pulse, N_timepoints])
            voltage_data (2D np.ndarray): Voltage signal traces (same shape as time_data)
            pulse_amplitudes (1D np.ndarray): Square pulse amplitude values (N_pulse,)
            run_id (int): Identifier for the measurement run
            save_path (str): Directory to save plots and results
        """
        self.time_data = time_data
        self.voltage_data = voltage_data
        self.pulse_amplitudes = pulse_amplitudes
        self.run_id = run_id
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def extract_taus(self, level=0.02, threshold=0.6, plot=True):
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
        unique_amps = np.unique(self.pulse_amplitudes)

        for amp in unique_amps:
            idx = np.where(self.pulse_amplitudes == amp)[0]
            if len(idx) == 0:
                continue

            time = self.time_data[idx[0], :]
            voltage = self.voltage_data[idx[0], :]
            tau_in, tau_out = self._extract_single_trace(time, voltage, level, threshold,
                                                         plot_figure=plot, amplitude=amp)

            tau_in_arr = np.array(tau_in)
            tau_out_arr = np.array(tau_out)

            gamma_in_arr = 1 / tau_in_arr[tau_in_arr > 0] if len(tau_in_arr[tau_in_arr > 0]) > 0 else np.array([])
            gamma_out_arr = 1 / tau_out_arr[tau_out_arr > 0] if len(tau_out_arr[tau_out_arr > 0]) > 0 else np.array([])

            results.append({
                "Amplitude": amp,
                "tau_in_mean": np.mean(tau_in_arr) if len(tau_in_arr) > 0 else 0,
                "tau_in_std": np.std(tau_in_arr) if len(tau_in_arr) > 0 else 0,
                "tau_in_n": len(tau_in_arr),
                "tau_out_mean": np.mean(tau_out_arr) if len(tau_out_arr) > 0 else 0,
                "tau_out_std": np.std(tau_out_arr) if len(tau_out_arr) > 0 else 0,
                "tau_out_n": len(tau_out_arr),
                "Gamma_in": np.mean(gamma_in_arr) if len(gamma_in_arr) > 0 else np.nan,
                "Gamma_in_std": np.std(gamma_in_arr)/np.sqrt(len(gamma_in_arr)) if len(gamma_in_arr) > 0 else np.nan,
                "Gamma_out": np.mean(gamma_out_arr) if len(gamma_out_arr) > 0 else np.nan,
                "Gamma_out_std": np.std(gamma_out_arr)/np.sqrt(len(gamma_out_arr)) if len(gamma_out_arr) > 0 else np.nan,
            })

        self.df_results = pd.DataFrame(results)
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

        if plot_figure:
            self._plot_trace(time, voltage, up_time, up_event, up_peaks,
                             down_time, down_event, down_peaks, amplitude)

        return tau_in_array, tau_out_array

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



class StandardGateSweepAnalysis:
    """
    A class to analyze voltage traces taken while sweeping gate voltage (Vg),
    using threshold crossing and histogram-based techniques to extract tau and Gamma.
    """
    def __init__(self, time_data, voltage_data, gate_voltages, run_id, save_path,diff_guess = 0.012):
        self.time_data = time_data
        self.voltage_data = voltage_data
        self.gate_voltages = gate_voltages
        self.run_id = run_id
        self.save_path = save_path
        self.diff_guess = diff_guess # This is the rough estimation of voltage state difference in V.
        os.makedirs(save_path, exist_ok=True)

    def extract_taus(self, plot=True):
        results = []

        for idx in range(self.voltage_data.shape[0]):
            time, voltage = self._get_single_trace(idx)
            gate_voltage = self._get_gate_voltage(idx)

            threshold, means, covs, weights, use_gmm = self._estimate_threshold(voltage)
            rise_times, fall_times = self._detect_events(time, voltage, threshold, means)
            tau_in, tau_out = self._compute_taus(rise_times, fall_times)

            if len(tau_in) < 2 or len(tau_out) < 2:
                continue

            stats = self._summarize_taus(tau_in, tau_out, gate_voltage)
            results.append(stats)

            if plot:
                self._plot_trace(time, voltage, threshold, rise_times, fall_times,
                                 means, covs, weights, use_gmm, gate_voltage, idx)

        self.df_results = pd.DataFrame(results)
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

    def _get_single_trace(self, idx):
        return self.time_data[idx, :], self.voltage_data[idx, :]

    def _get_gate_voltage(self, idx):
        return self.gate_voltages[idx, 0] if self.gate_voltages.ndim == 2 else self.gate_voltages[idx]

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

    def _compute_taus(self, rise_times, fall_times):
        
        if len(rise_times) < 2 or len(fall_times) < 2:
            return [], []
        
        min_len = min(len(rise_times), len(fall_times))
        rise_times = rise_times[:min_len]
        fall_times = fall_times[:min_len]

        if rise_times[0] < fall_times[0]:
            tau_out = fall_times - rise_times
            tau_in = rise_times[1:] - fall_times[:-1]
        else:
            tau_in = rise_times - fall_times
            tau_out = fall_times[1:] - rise_times[:-1]

        return tau_in, tau_out

    def _summarize_taus(self, tau_in, tau_out, Vg):
        tau_in_arr = np.array(tau_in)
        tau_out_arr = np.array(tau_out)
        gamma_in_arr = 1 / tau_in_arr[tau_in_arr > 0] if len(tau_in_arr[tau_in_arr > 0]) > 0 else np.array([])
        gamma_out_arr = 1 / tau_out_arr[tau_out_arr > 0] if len(tau_out_arr[tau_out_arr > 0]) > 0 else np.array([])

        return {
            "GateVoltage": Vg,
            "tau_in_mean": np.mean(tau_in_arr) if len(tau_in_arr) > 0 else 0,
            "tau_in_std": np.std(tau_in_arr) if len(tau_in_arr) > 0 else 0,
            "tau_in_n": len(tau_in_arr),
            "tau_out_mean": np.mean(tau_out_arr) if len(tau_out_arr) > 0 else 0,
            "tau_out_std": np.std(tau_out_arr) if len(tau_out_arr) > 0 else 0,
            "tau_out_n": len(tau_out_arr),
            "Gamma_in": np.mean(gamma_in_arr) if len(gamma_in_arr) > 0 else np.nan,
            "Gamma_in_std": np.std(gamma_in_arr) / np.sqrt(len(gamma_in_arr)) if len(gamma_in_arr) > 0 else np.nan,
            "Gamma_out": np.mean(gamma_out_arr) if len(gamma_out_arr) > 0 else np.nan,
            "Gamma_out_std": np.std(gamma_out_arr) / np.sqrt(len(gamma_out_arr)) if len(gamma_out_arr) > 0 else np.nan,
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
























