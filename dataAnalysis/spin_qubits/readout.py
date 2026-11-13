from abc import ABC, abstractmethod

import numpy as np
import ultraplot as plt
import lmfit
import qcodes as qc

from dataAnalysis.dataset import DataSet

color_g = 'blue'
color_e = 'red'

class ReadoutCharacterization(ABC):     # Abstract Base Class
    """Format-agnostic analysis. The idea is that subclasses need to provide methods to integrate the signal and to compute signal_g and signal_e, and this class defines the format-agnostic calculation of the histogram, visibility and fidelity."""
    signal_g: np.ndarray
    signal_e: np.ndarray
    ground_signal_is_higher: bool

    @abstractmethod
    def integrate(self, num_bins, start_bin=0):
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

    def compute_visibility_fidelity(self):
        if not hasattr(self, "counts_g"):
            raise AttributeError("The object doesn't have histogram data. Please run the method `compute_histogram` first.")
        total_g = self.counts_g.sum()
        total_e = self.counts_e.sum()
    
        self.cumulative_prob_g = np.cumsum(self.counts_g) / total_g
        self.cumulative_prob_e = 1-np.cumsum(self.counts_e) / total_e
    
        self.visibility_array = self.cumulative_prob_g + self.cumulative_prob_e - 1
        if self.ground_signal_is_higher: 
            self.visibility_array *= -1

        self.idx_max_visibility = np.argmax(self.visibility_array)
        self.max_visibility = np.max(self.visibility_array)
        self.threshold = self.edges[self.idx_max_visibility]
        self.fidelity = (self.max_visibility + 1)/2

class RFTimeTrace(DataSet):
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

    def integrate(self, num_bins, start_bin=0):
        idx_start = start_bin
        idx_stop = start_bin + num_bins
        self.I_int = np.sum(self.I[idx_start:idx_stop], axis=0)/num_bins
        self.Q_int = np.sum(self.Q[idx_start:idx_stop], axis=0)/num_bins

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

    def integrate(self, num_bins, start_bin=0):
        self.dataset_g.integrate(num_bins, start_bin)
        self.dataset_e.integrate(num_bins, start_bin)

    def plot_IQ(self, num_shots=None):
        if not hasattr(self.dataset_g, "I_int"):
            raise AttributeError("Error: before plotting I and Q, you need to obtain a signal by integrating the raw time trace. Run the method `integrate` first.")

        I_g = self.dataset_g.I_int[:num_shots]
        Q_g = self.dataset_g.Q_int[:num_shots]
        I_e = self.dataset_e.I_int[:num_shots]
        Q_e = self.dataset_e.Q_int[:num_shots]

        fig, ax = plt.subplots()
        fig.suptitle(f"Run ID #{self.dataset_g.run_id}, {self.dataset_e.run_id}")
        ax.scatter(I_g, Q_g, color=color_g, marker='o', s=5, alpha=0.1, label='Ground')
        ax.scatter(I_e, Q_e, color=color_e, marker='o', s=5, alpha=0.1, label='Ground')
        ax.format(xlabel="I (V)", ylabel="Q (V)")

        return fig, ax

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

        # Plot
        # Data
        fig, ax = self.plot_IQ(num_shots=1000)
        # Centers
        ax.scatter(I_g_center, Q_g_center, color=color_g, marker='o', s=10)
        ax.scatter(I_e_center, Q_e_center, color=color_e, marker='o', s=10)
        # Line connecting centers
        ax.plot([I_g_center, I_e_center], [Q_g_center, Q_e_center], ls='dashed', color='k')
        ax.xaxis.set_inverted(False) # to make sure axis is not flipped
        ax.yaxis.set_inverted(False) # to make sure axis is not flipped
    

