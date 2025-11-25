import numpy as np
import matplotlib.pyplot as plt
from dataAnalysis.base import DataSet

class SingleShotMeasurement(DataSet):
    """
    Data analysis class for single shot experiment.
    """
    def __init__(self, exp, run_id:int|list, station=None):
        """
        Parameters:
            exp: The experiment handler.
            run_id (int, list): One single or a list of two integers with the meaning [run_id_ground, run_id_excited]
        
        """
        if isinstance(run_id, int):    # one single run_id provided
            super().__init__(exp=exp, run_id=run_id, station=station)
            self.time = self.independent_parameters['y']['values'] 
            self.num_shot = self.independent_parameters['x']['values']
            self.signal_mag = self.dependent_parameters['param_0']['values'].T
            self.signal_pha = self.dependent_parameters['param_1']['values'].T
        elif isinstance(run_id, list):  
            assert len(run_id) == 2, "run_id must be either an integer or a list of two integers (one for ground state, one for excited state)"
            run_id_g, run_id_e = run_id[0], run_id[1]

            dataset_g = DataSet(exp=exp, run_id=run_id_g, station=station)
            self.time_g = dataset_g.independent_parameters['y']['values'] 
            self.num_shot_g = dataset_g.independent_parameters['x']['values'] 
            self.tot_num_shots_g = len(self.num_shot_g)
            self.signal_mag_g = dataset_g.dependent_parameters['param_0']['values'].T
            self.signal_phase_g = dataset_g.dependent_parameters['param_1']['values'].T

            dataset_e = DataSet(exp=exp, run_id=run_id_e, station=station)
            self.time_e = dataset_e.independent_parameters['y']['values'] 
            self.num_shot_e = dataset_e.independent_parameters['x']['values'] 
            self.tot_num_shots_e = len(self.num_shot_e)
            self.signal_mag_e = dataset_e.dependent_parameters['param_0']['values'].T
            self.signal_phase_e = dataset_e.dependent_parameters['param_1']['values'].T

    def plot_average_time_trace_over_shots(self):
        """
        Calculate the average time trace over all the shots for ground and excited state data and plot it.
        """
        fig = plt.figure(figsize=(6.5, 3.0))
        plt.scatter(self.time_g, np.mean(self.signal_mag_g, axis=0), s=6, alpha=0.7, label="Ground avg")
        plt.scatter(self.time_e, np.mean(self.signal_mag_e, axis=0), s=6, alpha=0.7, label="Excited avg")
        plt.xlabel("Time (ns)")
        plt.ylabel("Magnitude (a.u.)")
        plt.legend(frameon=False)
        plt.grid(alpha=0.25, linestyle="--")
        plt.tight_layout()
        plt.show()

        return fig, fig.axes[0]
        
    def build_histogram_from_time_trace(self, num_bins: int, time_index: int = 0, plot_average_time_trace=False, clip_quantiles:tuple=(0,1)):
        """
        Build 1D histograms (Ground vs Excited) from the selected time-trace column.

        Parameters:
            num_bins: number of bins
            which_column: choose which bin of the 2D traces to histogram. Default to be the first bin.
            clip_quantiles: Tuple containing lower and higher percentages for cutting outlier data.

        Returns: dict with 'histogram_ground', 'histogram_excited', 'histogram_stack', 'common_edges'
        """
        # Check data shapes match
        if self.tot_num_shots_g != self.tot_num_shots_e:
            raise ValueError("Ground num shot must equal to excited num shot")

        # Plot average time trace if asked
        if plot_average_time_trace:
            self.plot_average_time_trace_over_shots()

        # Pick the data from the provided time index and clean NaNs/infs
        ground_data = self.signal_mag_g[:, time_index]
        excited_data = self.signal_mag_e[:, time_index]
        ground_data = ground_data[np.isfinite(ground_data)]
        excited_data = excited_data[np.isfinite(excited_data)]

        both_data = np.r_[ground_data, excited_data]    # Combine ground and excited data together

        # Cut out outliers according to the given quanties
        if clip_quantiles is not None:
            lo, hi = np.quantile(both_data, clip_quantiles)     # Use combined data to calculate the quantiles
            both_data = both_data[(both_data >= lo) & (both_data <= hi)]
            ground_data = ground_data[(ground_data >= lo) & (ground_data <= hi)]
            excited_data = excited_data[(excited_data >= lo) & (excited_data <= hi)]

        # Shared edges
        edges = np.histogram_bin_edges(both_data, bins=num_bins)

        # Calculate histogram counts using same edges
        hg, _ = np.histogram(ground_data, bins=edges)
        he, _ = np.histogram(excited_data, bins=edges)
        h_total = hg + he

        # Save data in a dictionary
        hist_dict = {
            "histogram_ground": hg,
            "histogram_excited": he,
            "histogram_stack": h_total,
            "common_edges": edges,
        }
        self.histogram_dict = hist_dict

        self.histogram = Histogram(hg, he, h_total, edges)
        
        return
    

    def plot_histograms(
        self,
        type: str = "stacked",
        plot_CDF: bool = False,
        signal_label: str = "Signal",
        signal_unit: str = "mV",
        counts_label:str = "Counts",
        g_e_labels: tuple[str, str] = ("Ground", "Excited"),
        title: str | None = "Single-shot histograms",
        fig = None,
        ax = None,
    ):
        """
        Plot the histogram for the single-shot measurement. This assumes that build_histogram_time_trace has been
        previously run.

        Parameters:
            type (string, optional): Histogram type. Choose among 'stacked', 'stacked_only', 'overlay', 'separate'.
            plot_CDF (bool, optional): If true, the CDF is fitted to obtain the optimal thresholding value and the visibility.
            signal_label (string, optional): String for the x label.
            signal_unit (string, optional): Choose among 'V', 'mV', 'uV' and the signal data will be converted into the desired unit.
            counts_label (string, optional): String for the y label.
            g_e_labels (tuple, optional): Tuple of strings for the ground state and excited state labels.
            title (string, optional): String for the plot title.
            fig: Figure handler (not possible for "separate" type).
            ax: Axis handler (not possible for "separate" type).
        """
        return self.histogram.plot_histograms(
            type=type, 
            plot_CDF=plot_CDF,
            signal_label=signal_label,
            signal_unit=signal_unit,
            counts_label=counts_label,
            g_e_labels=g_e_labels,
            title=title,
            fig=fig,
            ax=ax
        )
        

class Histogram:
    def __init__(self, counts_g, counts_e, counts_tot, edges):
        self.counts_g = counts_g
        self.counts_e = counts_e
        self.counts_tot = counts_tot
        self.edges = edges
        self.left_edges = edges[:-1]
        self.bin_width = np.diff(edges)
        self.bin_centers = self.left_edges + self.bin_width/2

    def plot_histograms(
        self,
        type: str = "stacked",
        plot_CDF: bool = False,
        signal_label: str = "Signal",
        signal_unit: str = "mV",
        counts_label: str = "Counts",
        g_e_labels: tuple[str, str] = ("Ground", "Excited"),
        title: str | None = "Single-shot histograms",
        fig = None,
        ax = None,
    ):
        """
        Plot the histogram for the single-shot measurement. This assumes that build_histogram_time_trace has been
        previously run.

        Parameters:
            type (string, optional): Histogram type. Choose among 'stacked', 'stacked_only', 'overlay', 'separate'.
            plot_CDF (bool, optional): If true, the CDF is fitted to obtain the optimal thresholding value and the visibility.
            labels (tuple, optional): Tuple of strings for the ground state and excited state labels.
            x_label (string, optional): String for the x label.
            signal_unit (string, optional): Choose among 'V', 'mV', 'uV' and the signal data will be converted into the desired unit.
        """
        # For the 'separate' type, generate two axes
        if type == "separate": # return immediately the plot
            return self._separate_histogram_plot(signal_label, signal_unit, counts_label, g_e_labels, title)
        
        # For all other types, use only on axis
        else:
            if not fig and not ax:  # Generate fig and axis handlers if not provided by the user
                fig, ax = plt.subplots(figsize=(7.5, 4.6))
            self._beautify_axis(ax)
        # for the following types, edit labels and stuff later because they are the same
            if type == "stacked":
                self._stacked_histogram_plot(ax, signal_label, signal_unit, counts_label, g_e_labels, title, stacked_only=False)
            elif type == 'stacked_only':
                self._stacked_histogram_plot(ax, signal_label, signal_unit, counts_label, g_e_labels, title, stacked_only=True)
            elif type == "overlay":
                self._overlay_histogram_plot(ax, signal_label, signal_unit, counts_label, g_e_labels, title)
            else:
                raise ValueError("'type' must be 'stacked', 'stacked_only', 'overlay', or 'separate'.")
        
        # CDF
        if plot_CDF:
            ax2 = ax.twinx()
            self._beautify_axis(ax2, right=True)
            ax2.set_ylabel("Probability")
            ax2.set_ylim(0, 1.05)
    
            # compute visibility and stuff
            self.compute_visibility()
    
            # Signal unit
            if signal_unit == 'mV':
                signal_multiplier = 1e3
            elif signal_unit == 'uV':
                signal_multiplier = 1e6
            elif signal_unit == 'V':
                signal_multiplier = 1

            # Plot on right y-axis
            ax2.plot(self.bin_centers*signal_multiplier, self.cumulative_prob_g, label="Ground CDF", color="tab:blue", linestyle="--", linewidth=1.5, alpha=0.6)
            ax2.plot(self.bin_centers*signal_multiplier, self.cumulative_prob_e, label="Excited CDF", color="tab:orange", linestyle="--", linewidth=1.5, alpha=0.6)
            ax2.plot(self.bin_centers*signal_multiplier, self.visibility_array, label="Visibility", color="tab:green", linestyle="-", linewidth=1.5)

            # Annotate threshold on left axis
            ax.axvline(self.threshold*signal_multiplier, color="gray", linestyle=":", linewidth=1.5, label=f"Threshold = {self.threshold*signal_multiplier:.3f} {signal_unit}")
            fig.text(1, 0.5, f"Max visibility = {self.max_visibility:.3f}\nFidelity = {self.fidelity*100:.1f}%")
            
    
            # Combined legend
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=False, bbox_to_anchor=(1.48, 1.05))
        else:
            ax.legend(frameon=False)

        return fig, ax
    
    def compute_visibility(self):
        """
        Compute the CFD of the ground and excited state counts. Then from this calculate the visibility array, the maximum visibility, the threshold voltage for maximum visibility and the fidelity.

        #####TODO
        #WARNING
        # it is very important to define whether the singlet peak voltage is higher than triplet peak voltage or not. 
        # I might need to write a code to flip them.
        """
        total_g = self.counts_g.sum()
        total_e = self.counts_e.sum()

        self.cumulative_prob_g = np.cumsum(self.counts_g) / total_g
        self.cumulative_prob_e = 1-np.cumsum(self.counts_e) / total_e

        self.visibility_array = self.cumulative_prob_g + self.cumulative_prob_e - 1
        self.idx_max_visibility = np.argmax(self.visibility_array)
        self.max_visibility = np.max(self.visibility_array)
        self.threshold = self.edges[self.idx_max_visibility]
        self.fidelity = (self.max_visibility + 1)/2
    
    def _beautify_axis(self, ax, top=False, bottom=True, left=True, right=False):
        ax.grid(alpha=0.25, linestyle="--")
        ax.spines['top'].set_visible(top)
        ax.spines['bottom'].set_visible(bottom)
        ax.spines['right'].set_visible(right)
        ax.spines['left'].set_visible(left)

    def _stacked_histogram_plot(self, ax, signal_label, signal_unit, counts_label, g_e_labels, title, stacked_only=False):
        
        # Signal unit
        if signal_unit == 'mV':
            signal_multiplier = 1e3
            signal_label += " (mV)"
        elif signal_unit == 'uV':
            signal_multiplier = 1e6
            signal_label += " (uV)"
        elif signal_unit == 'V':
            signal_multiplier = 1
            signal_label += " (V)"

        ax.bar(self.left_edges*signal_multiplier, self.counts_tot, width=self.bin_width*signal_multiplier, align="edge", alpha=0.35, edgecolor="black", linewidth=0.6, label="Total")
        if not stacked_only:
            ax.step(self.edges*signal_multiplier, np.r_[self.counts_g, self.counts_g[-1] if self.counts_g.size else 0.0], where="post", linewidth=1.8, alpha=0.85, label=g_e_labels[0])
            ax.step(self.edges*signal_multiplier, np.r_[self.counts_e, self.counts_e[-1] if self.counts_e.size else 0.0], where="post", linewidth=1.8, alpha=0.85, label=g_e_labels[1])

        # Labels
        ax.set_xlabel(signal_label)
        ax.set_ylabel(counts_label)
        ax.set_xlim(self.edges[0]*signal_multiplier, self.edges[-1]*signal_multiplier)
        ax.set_title(title)

    def _overlay_histogram_plot(self, ax, signal_label, signal_unit, counts_label, g_e_labels, title):
        
        # Signal unit
        if signal_unit == 'mV':
            signal_multiplier = 1e3
            signal_label += " (mV)"
        elif signal_unit == 'uV':
            signal_multiplier = 1e6
            signal_label += " (uV)"
        elif signal_unit == 'V':
            signal_multiplier = 1
            signal_label += " (V)"

        ax.step(self.edges*signal_multiplier, np.r_[self.counts_g, self.counts_g[-1] if self.counts_g.size else 0.0], where="post", linewidth=1.6, label=g_e_labels[0])
        ax.step(self.edges*signal_multiplier, np.r_[self.counts_e, self.counts_e[-1] if self.counts_e.size else 0.0], where="post", linewidth=1.6, label=g_e_labels[1])

        # Labels
        ax.set_xlabel(signal_label)
        ax.set_ylabel(counts_label)
        ax.set_xlim(self.edges[0]*signal_multiplier, self.edges[-1]*signal_multiplier)
        ax.set_title(title)

    def _separate_histogram_plot(self, signal_label, signal_unit, counts_label, g_e_labels, title):

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7.5, 5.8))

        # Signal unit
        if signal_unit == 'mV':
            signal_multiplier = 1e3
            signal_label += " (mV)"
        elif signal_unit == 'uV':
            signal_multiplier = 1e6
            signal_label += " (uV)"
        elif signal_unit == 'V':
            signal_multiplier = 1
            signal_label += " (V)"

        # Generate plots
        axs[0].bar(self.left_edges*signal_multiplier, self.counts_g, width=self.bin_width*signal_multiplier, align="edge", alpha=0.9, edgecolor="black", linewidth=0.6, label=g_e_labels[0])
        axs[1].bar(self.left_edges*signal_multiplier, self.counts_e, width=self.bin_width*signal_multiplier, align="edge", alpha=0.9, edgecolor="black", linewidth=0.6, label=g_e_labels[1])

        # Beautify
        for a in axs:
            self._beautify_axis(a)
            a.set_ylabel(counts_label)
            a.legend(frameon=False)
        axs[1].set_xlabel(signal_label)
        if title:
            axs[0].set_title(title)
        plt.tight_layout()
        plt.show()

        return fig, axs
    