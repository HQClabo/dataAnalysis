from abc import ABC, abstractmethod

import numpy as np
import ultraplot as plt
import lmfit
import qcodes as qc

from dataAnalysis.dataset import DataSet

color_g = 'blue'
color_e = 'red'

class TimeTrace(ABC, DataSet):
    """Generic format-agnostic class for storing one measurement containing time traces vs shot."""
    time: np.ndarray
    bin_time: int
    shots: np.ndarray
    num_shots: int

    @abstractmethod
    def integrate(self, num_time_bins, start_time_bin=0):
        """Method to process the integrate the time trace by the desired integration time. Every subclass must implement this."""

class ReadoutCharacterization(ABC):     # Abstract Base Class
    """Format-agnostic analysis. The idea is that subclasses need to provide methods to integrate the signal and to compute signal_g and signal_e, and this class defines the format-agnostic calculation of the histogram, visibility and fidelity."""
    dataset_g: TimeTrace
    dataset_e: TimeTrace
    signal_g: np.ndarray
    signal_e: np.ndarray
    ground_signal_is_higher: bool

    @abstractmethod
    def integrate(self, num_time_bins, start_time_bin=0):
        """Method to process the integrate the time trace by the desired integration time. Every subclass must implement this."""

    @abstractmethod
    def compute_signal(self):
        """Method to process the raw signal to save internally signal_g and signal_e that should be used for the histogram computation. Every subclass must implement this."""

    def compute_histogram(self, num_bins='auto'):
        # Combine ground and excited data to find the hist edges
        self.signal_combined = np.concatenate([self.signal_g, self.signal_e])
        self.bin_edges = np.histogram_bin_edges(self.signal_combined, bins=num_bins)
        # Compute histogram counts
        self.counts_g, _ = np.histogram(self.signal_g, bins=self.bin_edges)
        self.counts_e, _ = np.histogram(self.signal_e, bins=self.bin_edges)
        self.counts_tot = self.counts_g + self.counts_e

    def analyze_histogram(self, method='CDF'):
        """Analyze the histogram to determine the following properties:
        - visibility_vs_threshold
        - fidelity_vs_threshold
        - idx_threshold_max_visibility
        - best_threshold
        - max_visibility
        - max_fidelity
        Depending on the method used, other properties (such as fit reports) may be stored.

        Args:
            method: Analysis method, currently supports: "CDF"

        TODO: Add other analysis methods when it will be necessary
        """
        self.analysis_method = method
        if method == 'CDF':
            return self._analyze_histogram_CDF()        

    def _analyze_histogram_CDF(self):
        if not hasattr(self, "counts_g"):
            raise AttributeError("The object doesn't have histogram data. Please run the method `compute_histogram` first.")
        total_g = self.counts_g.sum()
        total_e = self.counts_e.sum()
        
        self.cumulative_prob_g = np.cumsum(self.counts_g) / total_g
        self.cumulative_prob_e = 1-np.cumsum(self.counts_e) / total_e
        
        self.visibility_vs_threshold = self.cumulative_prob_g + self.cumulative_prob_e - 1
        if self.ground_signal_is_higher: 
            self.visibility_vs_threshold *= -1
        self.fidelity_vs_threshold = (self.visibility_vs_threshold + 1)/2 
        
        self.idx_threshold_max_visibility = np.argmax(self.visibility_vs_threshold)
        self.best_threshold = self.bin_edges[self.idx_threshold_max_visibility]
        self.max_visibility = self.visibility_vs_threshold[self.idx_threshold_max_visibility]
        self.max_fidelity = self.fidelity_vs_threshold[self.idx_threshold_max_visibility]

    def analyze_fixed_integration_time(
            self, 
            num_time_bins, start_time_bin=0, 
            num_bins_hist='auto', 
            analysis_method='CDF',
            do_plots=False,
    ):
        """Full pipeline to analyze the data for one integration window. 
        This function:
        - integrates the time traces in the specified window
        - processes the integrated signal to obtain signal_g and signal_e
        - bins the data into a histogram
        - analyzes the histogram to obtain visibility, fidelity and thresholding

        Args:
            num_time_bins: number of time bins to be used to integrate the data
            start_time_bin: start time bin for the integration
            num_bins_hist: number of bins to be used for the histogram data binning
            analysis_method: method to analyze the histogram, currently supports: "CDF"
            do_plots: Boolean to do the plots: I&Q, histogram, visibility
        """
        self.integrate(num_time_bins, start_time_bin)
        self.compute_signal()
        self.compute_histogram(num_bins_hist)
        self.analyze_histogram(analysis_method)
        if do_plots:
            return self.plot_histogram()

    def analyze_sweep_integration(
            self,
            num_time_bins_range:list, start_time_bin=0,
            num_bins_hist='auto', 
            analysis_method='CDF',
            do_plots=True,
            do_subplots=False,
            do_plots_best=True,
    ):
        """Full pipeline to analyze the data, sweeping the integration time in order to find
        the best integration time giving the highest fidelity. 
        This function:
        - looop over the integration time range provided, and for each value:
            - integrates the time traces in the specified window
            - processes the integrated signal to obtain signal_g and signal_e
            - bins the data into a histogram
            - analyzes the histogram to obtain visibility, fidelity and thresholding
        
        Args:
            num_time_bins_range: formatted as [min_num_bins, max_num_bins], determined the span of integration time to be tested. The first element (min_num_bins) should be >= 1.
            start_time_bin: start time bin for the integration
            num_bins_hist: number of bins to be used for the histogram data binning
            analysis_method: method to analyze the histogram, currently supports: "CDF"
            do_plots: Boolean to do the plots: fidelity&visibility vs integration time
            do_subplots: Boolean to do the plots of the analysis for each integration time: I&Q, histogram, visibility
            do_plots_best: Boolean to plot the I&Q, histogram and visibility for the best integration time
        """
        if len(num_time_bins_range) != 2:
            raise ValueError("'num_time_bins_range' should be in the format [min_num_bins, max_num_bins]")
        # Arrays for the integration time
        min_num_time_bins, max_num_time_bins = num_time_bins_range[0], num_time_bins_range[1]
        self.num_time_bins_array = np.arange(min_num_time_bins, max_num_time_bins + 1)
        self.int_time_array = self.num_time_bins_array * self.dataset_g.bin_time
        # Prepare empty arrays for results
        self.visibility_vs_int_time = np.zeros_like(self.int_time_array)
        self.fidelity_vs_int_time = np.zeros_like(self.int_time_array)
        # Sweep over integration times
        for i, num_time_bins in enumerate(self.num_time_bins_array):
            self.analyze_fixed_integration_time(
                num_time_bins, start_time_bin,
                num_bins_hist,
                analysis_method,
                do_subplots,
            )
            self.visibility_vs_int_time[i] = self.max_visibility
            self.fidelity_vs_int_time[i] = self.max_fidelity
        # Find the best integration time
        self.idx_best_int_time = np.argmax(self.visibility_vs_int_time)
        self.best_int_time = self.int_time_array[self.idx_best_int_time]
        self.best_int_time_num_bins = self.num_time_bins_array[self.idx_best_int_time]
        self.best_int_time_fidelity = self.fidelity_vs_int_time[self.idx_best_int_time]
        self.best_int_time_visibility = self.visibility_vs_int_time[self.idx_best_int_time]

        if do_plots:
            self.plot_fidelity_vs_int_time()
        if do_plots_best:
            self.analyze_fixed_integration_time(
                self.best_int_time_num_bins, start_time_bin,
                num_bins_hist, 
                analysis_method,
                do_plots=False,
                )
            fig, axs = self.plot_histogram()
            axs[0].set_title(f"Best $t_\\mathrm{{int}} = {self.best_int_time:.0f}$ ns")

    def plot_histogram(self, plot_analysis=True):
        xvals = self.bin_edges * 1e3
        xunit = 'mV'
        if plot_analysis:
            if not hasattr(self, "visibility_vs_threshold"):
                raise AttributeError("Before plotting with plot_analysis=True, run the method 'analyze_histogram' first.")
            fig, axs = plt.subplots(refwidth="10cm", refaspect=2,
                                    ncols=1, nrows=2,
                                    sharex=True, spany=False)
            fig.suptitle(f"Run ID #{self.dataset_g.run_id}, {self.dataset_e.run_id}")
            # Histogram
            ax = axs[0]
            ax.stairs(self.counts_tot, xvals, fill=True, color='grey', alpha=0.5, label='Total')
            ax.stairs(self.counts_g, xvals, fill=False, color=color_g, label="Ground")
            ax.stairs(self.counts_e, xvals, fill=False, color=color_e, label='Excited')
            ax.format(xlabel=None, ylabel="Counts")
            ax.legend(loc='right', ncols=1, frameon=False)
            # Fidelity/visibility vs threshold
            ax=axs[1]
            bin_centers = xvals[:-1] + (xvals[1] - xvals[0])
            ax.plot(bin_centers, self.fidelity_vs_threshold*100, color='k')
            ax.format(xlabel=f'Signal ({xunit})', ylabel='Fidelity $F$ (%)')
            threshold_label = f"$V_\\mathrm{{th}} = {self.best_threshold*1e3:.3f}$ {xunit}\n$F_\\mathrm{{max}} = {self.max_fidelity*100:.1f}$%\n$V_\\mathrm{{max}} = {self.max_visibility*100:.1f}$%"
            ax.axvline(x=self.best_threshold*1e3, ls='-.', color='k', label=threshold_label)
            ax.legend(loc='right', frameon=False, ncols=1, title="Thresholding")
            ax2 = ax.dualy(lambda F: 2*F - 100, label='Visibility $V$ (%)')

            return fig, axs
        
        else:
            fig, ax = plt.subplots(refwidth="10cm", refaspect=2)
            fig.suptitle(f"Run ID #{self.dataset_g.run_id}, {self.dataset_e.run_id}")
            ax.stairs(self.counts_tot, xvals, fill=True, color='grey', alpha=0.5, label='Total')
            ax.stairs(self.counts_g, xvals, fill=False, color=color_g, label="Ground")
            ax.stairs(self.counts_e, xvals, fill=False, color=color_e, label='Excited')
            ax.format(xlabel=f'Signal ({xunit})', ylabel="Counts")
            ax.legend(loc='right', ncols=1, frameon=False)
            return fig, ax

    def plot_fidelity_vs_int_time(self):
        fig, ax = plt.subplots()
        fig.suptitle(f"Run ID #{self.dataset_g.run_id}, {self.dataset_e.run_id}")
        ax.plot(self.int_time_array, self.fidelity_vs_int_time*100, marker='.', ls='', color='k')
        ax.format(xlabel="Integration time (ns)", ylabel='Fidelity $F$ (%)')
        ax2 = ax.dualy(lambda F: 2*F - 100, label='Visibility $V$ (%)')
        return fig, ax


class RFTimeTrace(TimeTrace):
    """Class for storing one measurement containing time traces vs shot for an RF type of readout (hence containing magnitude and phase)"""
    time: np.ndarray
    bin_time: int
    shots: np.ndarray
    num_shots: int
    mag: np.ndarray
    phase: np.ndarray
    time_unit: str
    mag_unit: str
    phase_unit: str
    I: np.ndarray
    Q: np.ndarray
    I_int: np.ndarray   # I integrated in post-processing
    Q_int: np.ndarray   # Q integrated in post-processing
    def __init__(self, exp, run_id, param_mag='param_0', param_phase='param_1', compute_IQ=True):
        super().__init__(exp=exp, run_id=run_id)
        self.time = self.independent_parameters['y']['values'] 
        self.bin_time = self.time[1] - self.time[0]
        self.shots = self.independent_parameters['x']['values'] 
        self.num_shots = len(self.shots)
        self.mag = self.dependent_parameters[param_mag]['values']
        self.phase = self.dependent_parameters[param_phase]['values']
        # units
        self.time_unit = self.independent_parameters['y']['paramspec'].unit
        self.mag_unit = self.dependent_parameters[param_mag]['paramspec'].unit
        self.phase_unit = self.dependent_parameters[param_phase]['paramspec'].unit
        # make sure that the phase is in degrees, for consistency
        if self.phase_unit == 'rad':
            self.phase = np.rad2deg(self.phase)
            self.phase_unit = 'deg'
        # compute I and Q
        if compute_IQ:
            self.compute_IQ()
        
    def compute_IQ(self):
        self.I = self.mag * np.cos(np.deg2rad(self.phase))
        self.Q = self.mag * np.sin(np.deg2rad(self.phase))

    def integrate(self, num_time_bins, start_time_bin=0):
        idx_start = start_time_bin
        idx_stop = start_time_bin + num_time_bins
        self.I_int = np.sum(self.I[idx_start:idx_stop], axis=0)/num_time_bins
        self.Q_int = np.sum(self.Q[idx_start:idx_stop], axis=0)/num_time_bins

class RFReadoutMeasurement(ReadoutCharacterization):
    """Class for analyzing the characterization measurement of an RF readout."""
    dataset_g: RFTimeTrace
    dataset_e: RFTimeTrace
    ground_signal_is_higher: bool
    signal_g: np.ndarray
    signal_e: np.ndarray
    signal_combined: np.ndarray
    bin_edges: np.ndarray
    def __init__(self, exp, 
                 run_id_g=None, run_id_e=None, 
                 param_mag='param_0', param_phase='param_1',
                 compute_IQ=True,
                 ground_signal_is_higher=False):
        self.dataset_g = RFTimeTrace(exp=exp, run_id=run_id_g, 
                                    param_mag=param_mag, param_phase=param_phase, compute_IQ=compute_IQ)
        self.dataset_e = RFTimeTrace(exp=exp, run_id=run_id_e, 
                                    param_mag=param_mag, param_phase=param_phase, compute_IQ=compute_IQ)
        self.ground_signal_is_higher = ground_signal_is_higher

    def integrate(self, num_time_bins, start_time_bin=0):
        self.dataset_g.integrate(num_time_bins, start_time_bin)
        self.dataset_e.integrate(num_time_bins, start_time_bin)

    def project_onto_discrimination_axis(self):
        # Check that an integrated signal has been calculated
        if not hasattr(self.dataset_g, "I_int"):
            raise AttributeError("Error: before plotting I and Q, you need to obtain a signal by integrating the raw time trace.")
        # Calculate the centroids of G and E data
        I_g_center = np.average(self.dataset_g.I_int)
        Q_g_center = np.average(self.dataset_g.Q_int)
        I_e_center = np.average(self.dataset_e.I_int)
        Q_e_center = np.average(self.dataset_e.Q_int)
        # Build vectors corresponding to the two centroids
        g_center_vec = np.array([I_g_center, Q_g_center])
        e_center_vec = np.array([I_e_center, Q_e_center])
        # Unit vector along the discrimination axis
        u = (e_center_vec - g_center_vec) / np.linalg.norm(e_center_vec - g_center_vec)
        # Middle point in between the two centroids
        m = (g_center_vec + e_center_vec) / 2
        # Project data onto discrimination axis
        signal_g = (np.column_stack([self.dataset_g.I_int,  self.dataset_g.Q_int]) - m) @ u
        signal_e = (np.column_stack([self.dataset_e.I_int,  self.dataset_e.Q_int]) - m) @ u
        # Store signal 
        self.signal_g = signal_g
        self.signal_e = signal_e
        self.I_g_center = I_g_center
        self.Q_g_center = Q_g_center
        self.I_e_center = I_e_center
        self.Q_e_center = Q_e_center

    def compute_signal(self):
        return self.project_onto_discrimination_axis()

    def plot_IQ(self, num_shots=None, plot_analysis=False):
        if not hasattr(self.dataset_g, "I_int"):
            raise AttributeError("Before plotting I and Q, you need to obtain a signal by integrating the raw time trace. Run the method `integrate` first.")
        # Check that the number of shots requested is <= than the number of shots recorded
        if num_shots > self.dataset_g.num_shots:
            print(f"THe number of shots requested ({num_shots}) is larger than the number of shots in the measurement ({self.dataset_g.num_shots}). Using the latter number instead.")
            num_shots = self.dataset_g.num_shots

        I_g = self.dataset_g.I_int[:num_shots]
        Q_g = self.dataset_g.Q_int[:num_shots]
        I_e = self.dataset_e.I_int[:num_shots]
        Q_e = self.dataset_e.Q_int[:num_shots]

        fig, ax = plt.subplots()
        s = 5   # datapoints marker size
        fig.suptitle(f"Run ID #{self.dataset_g.run_id}, {self.dataset_e.run_id}")
        ax.scatter(I_g, Q_g, color=color_g, marker='o', s=s, alpha=0.1, label='Ground')
        ax.scatter(I_e, Q_e, color=color_e, marker='o', s=s, alpha=0.1, label='Ground')
        ax.format(xlabel="I (V)", ylabel="Q (V)")

        if plot_analysis:
            if not hasattr(self, "I_g_center"):
                raise AttributeError("In order to plot the analysis of the IQ data, run the method 'project_onto_discrimination_axis' (or the shothand version 'compute_signal' first.")
            # Centers
            ax.scatter(self.I_g_center, self.Q_g_center, color=color_g, marker='o', s=s*3)
            ax.scatter(self.I_e_center, self.Q_e_center, color=color_e, marker='o', s=s*3)
            # Line connecting centers
            ax.plot([self.I_g_center, self.I_e_center], [self.Q_g_center, self.Q_e_center], ls='dashed', color='k')
            ax.xaxis.set_inverted(False) # to make sure axis is not flipped
            ax.yaxis.set_inverted(False) # to make sure axis is not flipped
        return fig, ax

class SingleParamReadoutMeasurement(ReadoutCharacterization):
    raise NotImplementedError

