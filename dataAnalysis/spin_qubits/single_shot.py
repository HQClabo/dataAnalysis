import numpy as np
import matplotlib.pyplot as plt
import qcodes as qc
from dataAnalysis.dataset import DataSet
import lmfit

class TimeTraceMeasurement(DataSet):
    """
    Data analysis class for single shot experiment.
    """
    def __init__(self, exp, run_id:int, param = 'param_0', ground_signal_is_higher=False):
        """
        Parameters:
            exp: The experiment handler.
            run_id (int, list): One single or a list of two integers with the meaning [run_id_ground, run_id_excited]
        
        """
        if isinstance(run_id, int) or run_id == None:    # one single run_id provided
            super().__init__(exp=exp, run_id=run_id)
            self.num_states = 1
            self.time = self.independent_parameters['y']['values'] 
            self.num_shot = self.independent_parameters['x']['values']
            self.signal = self.dependent_parameters[param]['values'].T
            self.signal_paramspec = self.dependent_parameters[param]['paramspec']
        elif isinstance(run_id, list):  
            assert len(run_id) == 2, "run_id must be either an integer or a list of two integers (one for ground state, one for excited state)"
            run_id_g, run_id_e = run_id[0], run_id[1]
            self.num_states = 2
            if run_id_g == None and run_id_e == None:
                run_id_e = exp.last_counter
                run_id_g = exp.last_counter-1
            self.run_id = [run_id_g, run_id_e]

            dataset_g = DataSet(exp=exp, run_id=run_id_g)
            self.time_g = dataset_g.independent_parameters['y']['values'] 
            self.num_shot_g = dataset_g.independent_parameters['x']['values'] 
            self.tot_num_shots_g = len(self.num_shot_g)
            self.signal_g = dataset_g.dependent_parameters[param]['values'].T
            self.signal_g_paramspec = dataset_g.dependent_parameters[param]['paramspec']

            dataset_e = DataSet(exp=exp, run_id=run_id_e)
            self.time_e = dataset_e.independent_parameters['y']['values'] 
            self.num_shot_e = dataset_e.independent_parameters['x']['values'] 
            self.tot_num_shots_e = len(self.num_shot_e)
            self.signal_e = dataset_e.dependent_parameters[param]['values'].T
            self.signal_e_paramspec = dataset_e.dependent_parameters[param]['paramspec']
        self.ground_signal_is_higher = ground_signal_is_higher

    def plot_average_time_trace_over_shots(self, fit=True):
        """
        Calculate the average time trace over all the shots for ground and excited state data and plot it.
        """
        if self.num_states == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8.0, 3.0))
            ax.scatter(self.time, np.mean(self.signal, axis=0), s=6, alpha=0.7)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel(f"{self.signal_paramspec.label} ({self.signal_paramspec.unit})" )
            ax.set_title(f"Run ID: {self.run_id}")
            plt.tight_layout()
            return fig, ax
        
        elif self.num_states == 2:
            fig, ax = plt.subplots(1, 1, figsize=(8.0, 3.0))
            ax.scatter(self.time_g, np.mean(self.signal_g, axis=0), s=6, alpha=0.7, c='b', label='Ground state')
            ax.scatter(self.time_e, np.mean(self.signal_e, axis=0), s=6, alpha=0.7, c='orange', label='Excited state')

            if fit:
                fit_report_g, model_g = fit_T1_decay(self.time_g, np.mean(self.signal_g, axis=0))
                fit_report_e, model_e = fit_T1_decay(self.time_e, np.mean(self.signal_e, axis=0))
                self.T1_g = fit_report_g.best_values['T1']
                self.T1_err_g = fit_report_g.params['T1'].stderr
                self.T1_e = fit_report_e.best_values['T1']
                self.T1_err_e = fit_report_e.params['T1'].stderr
                ax.plot(self.time_g, model_g.eval(params=fit_report_g.params, t=self.time_g), c='b', linestyle='dashed', label=f"$T_1$ = {self.T1_g/1e3:.0f} $\mu$s ")
                ax.plot(self.time_e, model_e.eval(params=fit_report_e.params, t=self.time_e), c='orange', linestyle='dashed', label=f"$T_1$ = {self.T1_e/1e3:.0f} $\mu$s" )

            ax.set_xlabel("Time (ns)")
            ax.set_ylabel(f"{self.signal_g_paramspec.label} ({self.signal_g_paramspec.unit})" )
            ax.set_title(f"Run ID: {self.run_id}")
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 1))
            ax.grid(alpha=0.25, linestyle="--")
            ax.grid(alpha=0.25, linestyle="--")
            plt.tight_layout()
            return fig, fig.axes

    
        
    def build_histogram_from_time_trace(self, num_bins_hist: int, integrate_num_bins: int, integrate_from_bin: int = 0, clip_quantiles:tuple=(0,1), plot=True):
        """
        Build 1D histograms (Ground vs Excited) from the selected time-trace column.

        Parameters:
            num_bins_hist: number of bins
            which_column: choose which bin of the 2D traces to histogram. Default to be the first bin.
            clip_quantiles: Tuple containing lower and higher percentages for cutting outlier data.

        Returns: dict with 'histogram_ground', 'histogram_excited', 'histogram_stack', 'common_edges'
        """
        # Check data shapes match
        if self.tot_num_shots_g != self.tot_num_shots_e:
            raise ValueError("Ground num shot must equal to excited num shot")

        # Integrate to obtain the data to form histogram
        ground_data = np.zeros_like(self.num_shot_g)
        excited_data = np.zeros_like(self.num_shot_e)
        for time_index in range(integrate_from_bin, integrate_from_bin + integrate_num_bins):
            ground_data += self.signal_g[:, time_index]
            excited_data += self.signal_e[:, time_index]
        ground_data /= (integrate_num_bins)
        excited_data /= (integrate_num_bins)
        # ground_data = ground_data[np.isfinite(ground_data)]
        # excited_data = excited_data[np.isfinite(excited_data)]

        both_data = np.r_[ground_data, excited_data]    # Combine ground and excited data together

        # Cut out outliers according to the given quanties
        if clip_quantiles is not None:
            lo, hi = np.quantile(both_data, clip_quantiles)     # Use combined data to calculate the quantiles
            both_data = both_data[(both_data >= lo) & (both_data <= hi)]
            ground_data = ground_data[(ground_data >= lo) & (ground_data <= hi)]
            excited_data = excited_data[(excited_data >= lo) & (excited_data <= hi)]

        # Shared edges
        edges = np.histogram_bin_edges(both_data, bins=num_bins_hist)

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

        self.histogram = Histogram(hg, he, h_total, edges, ground_signal_is_higher=self.ground_signal_is_higher)

        if plot:
            self.plot_histograms()
        return self.histogram
    

    def plot_histograms(
        self,
        type: str = "stacked",
        plot_CDF: bool = False,
        counts_label:str = "Counts",
        g_e_labels: tuple[str, str] = ("Ground", "Excited"),
        title: str | None = None,
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
        if title == None:
            title = f"Run #{self.run_id}"
        return self.histogram.plot_histograms(
            type=type, 
            plot_CDF=plot_CDF,
            signal_label=self.signal_g_paramspec.label,
            signal_unit=self.signal_g_paramspec.unit,
            counts_label=counts_label,
            g_e_labels=g_e_labels,
            title=title,
            fig=fig,
            ax=ax
        )

    def visibility_and_fidelity_vs_integration_time(
        self, 
        num_bins_hist: int, 
        integrate_num_bins_range: list, 
        integrate_from_bin: int = 0, 
        clip_quantiles:tuple=(0,1),
        plot_best_histogram=True,
    ):
        assert len(integrate_num_bins_range) == 2
        min_integrate_num_bins = integrate_num_bins_range[0]
        max_integrate_num_bins = integrate_num_bins_range[1]

        self.time_step = self.time_g[2] - self.time_g[1] # just to avoid the first point being weird because of opx shit timing
        self.integrate_num_bins_array = np.arange(min_integrate_num_bins, max_integrate_num_bins + 1)
        self.integration_time_array = np.arange(min_integrate_num_bins, max_integrate_num_bins + 1)*self.time_step
        self.max_visibility_vs_int_time = np.zeros_like(self.integration_time_array)
        self.fidelity_vs_int_time = np.zeros_like(self.integration_time_array)
        

        for i, integrate_num_bins in enumerate(range(min_integrate_num_bins, max_integrate_num_bins + 1)):
            hist = self.build_histogram_from_time_trace(num_bins_hist=num_bins_hist, integrate_num_bins=integrate_num_bins, integrate_from_bin=integrate_from_bin, clip_quantiles=clip_quantiles)
            hist.compute_visibility()
            self.max_visibility_vs_int_time[i] = hist.max_visibility
            self.fidelity_vs_int_time[i] = hist.fidelity

        best_idx = np.argmax(self.max_visibility_vs_int_time)
        self.best_integration_time = self.integration_time_array[best_idx]
        self.best_integrate_num_bins = self.integrate_num_bins_array[best_idx]
        self.best_fidelity = self.fidelity_vs_int_time[best_idx]
        self.best_visibility = self.max_visibility_vs_int_time[best_idx]
        self.best_integrate_from_bin = integrate_from_bin
        self.best_num_bins_hist = num_bins_hist



        fig, ax = plt.subplots()
        fig.suptitle(f"Run #{self.run_id}")
        ax.plot(self.integration_time_array, self.max_visibility_vs_int_time, color='k', marker='o', markersize=5, alpha=0.5, )
        ax.set_xlabel("Integration time (ns)")
        ax.set_ylabel("Max visibility")

        ax1 = ax.twinx()
        ax1.plot(self.integration_time_array, self.fidelity_vs_int_time*100, color='k', marker='o', markersize=5, alpha=0.5)
        ax1.set_ylabel("Fidelity (%)")

        if plot_best_histogram:
            self.plot_histogram_best_integration_time()

    def plot_histogram_best_integration_time(self):
        hist = self.build_histogram_from_time_trace(self.best_num_bins_hist, self.best_integrate_num_bins, self.best_integrate_from_bin)
        hist.compute_visibility()
        fig, ax = hist.plot_histograms(plot_CDF=True)
        fig.suptitle(f"Run #{self.run_id}")
        ax.set_title(f"Histogram for the best $t_{{int}}$ = {self.best_integration_time} ns")

def fit_T1_decay(t, signal, print_guesses=False):
    """
    Currently only works if both ground and excited states are provided to the class
    """

    # Fit for ground state
    params = lmfit.Parameters()
    default_guess = calc_default_guesses(t, signal)
    # guess_to_use = combine_default_and_user_guess(default_guess, user_guess=guess_dict)
    guess_to_use = default_guess
    if print_guesses:
        print(f"Guesses = {guess_to_use}" )
    params.add('T1', value = guess_to_use['T1'])
    params.add('V0', value=guess_to_use['V0'])
    params.add('C', value=guess_to_use['C'])

    model = lmfit.Model(model_T1_decay_vs_time, independent_vars=['t'])
    fit_result = model.fit(signal, params, t=t)

    return fit_result, model

def model_T1_decay_vs_time(t, T1, V0, C):
    return V0*np.exp(-t/T1) + C

def calc_default_guesses(t, signal):
    """
        Assumes a signal like V0 * exp(-t/T1) + C
    """
    window_size = 10
    signal = np.convolve(signal, np.ones(window_size)/window_size, mode='same')[window_size//2:-(window_size//2)]
    t = t[window_size//2:-(window_size//2)]
    V0 = max(np.abs(signal)) * np.sign(signal[np.argmax(np.abs(signal))])
    C = min(np.abs(signal)) * np.sign(signal[np.argmin(np.abs(signal))])
    T1 = t[np.argmin(signal - V0/np.e + C)] # @t=T1 signal = V0/np.e + C
    guess_dict = {
        "T1": T1,
        "V0": V0,
        "C": C,
    }
    return guess_dict
        

class Histogram:
    def __init__(self, counts_g, counts_e, counts_tot, edges, ground_signal_is_higher):
        self.counts_g = counts_g
        self.counts_e = counts_e
        self.counts_tot = counts_tot
        self.edges = edges
        self.left_edges = edges[:-1]
        self.bin_width = np.diff(edges)
        self.bin_centers = self.left_edges + self.bin_width/2
        self.ground_signal_is_higher = ground_signal_is_higher

    def compute_visibility(self):
        """
        Compute the CFD of the ground and excited state counts. Then from this calculate the visibility array, the maximum visibility, the threshold voltage for maximum visibility and the fidelity.

        #WARNING
        # it is very important to define whether the ground signal is higher than excited signal or not. 
        # I might need to write a code to flip them.
        """
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
            if not hasattr(self, 'visibility_array'):
                raise AttributeError("If you want to plot the CDF, run the method 'computer_visibility' first.")
            ax2 = ax.twinx()
            self._beautify_axis(ax2, right=True)
            ax2.set_ylabel("Probability")
            ax2.set_ylim(0, 1.05)
    

            # Plot on right y-axis
            ax2.plot(self.bin_centers, self.cumulative_prob_g, label="Ground CDF", color="tab:blue", linestyle="--", linewidth=1.5, alpha=0.6)
            ax2.plot(self.bin_centers, self.cumulative_prob_e, label="Excited CDF", color="tab:orange", linestyle="--", linewidth=1.5, alpha=0.6)
            ax2.plot(self.bin_centers, self.visibility_array, label="Visibility", color="tab:green", linestyle="-", linewidth=1.5)

            # Annotate threshold on left axis
            ax.axvline(self.threshold, color="gray", linestyle=":", linewidth=1.5, label=f"Threshold = {self.threshold:.3e} {signal_unit}")
            fig.text(1, 0.5, f"Max visibility = {self.max_visibility:.3f}\nFidelity = {self.fidelity*100:.1f}%")
            
    
            # Combined legend
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=False, bbox_to_anchor=(1.48, 1.05))
        else:
            ax.legend(frameon=False)

        return fig, ax
    
    
    def _beautify_axis(self, ax, top=False, bottom=True, left=True, right=False):
        ax.grid(alpha=0.25, linestyle="--")
        ax.spines['top'].set_visible(top)
        ax.spines['bottom'].set_visible(bottom)
        ax.spines['right'].set_visible(right)
        ax.spines['left'].set_visible(left)

    def _stacked_histogram_plot(self, ax, signal_label, signal_unit, counts_label, g_e_labels, title, stacked_only=False):
        

        ax.bar(self.left_edges, self.counts_tot, width=self.bin_width, align="edge", alpha=0.35, edgecolor="black", linewidth=0.6, label="Total")
        if not stacked_only:
            ax.step(self.edges, np.r_[self.counts_g, self.counts_g[-1] if self.counts_g.size else 0.0], where="post", linewidth=1.8, alpha=0.85, label=g_e_labels[0])
            ax.step(self.edges, np.r_[self.counts_e, self.counts_e[-1] if self.counts_e.size else 0.0], where="post", linewidth=1.8, alpha=0.85, label=g_e_labels[1])

        # Labels
        ax.set_xlabel(signal_label)
        ax.set_ylabel(counts_label)
        ax.set_xlim(self.edges[0], self.edges[-1])
        ax.set_title(title)

    def _overlay_histogram_plot(self, ax, signal_label, signal_unit, counts_label, g_e_labels, title):

        ax.step(self.edges, np.r_[self.counts_g, self.counts_g[-1] if self.counts_g.size else 0.0], where="post", linewidth=1.6, label=g_e_labels[0])
        ax.step(self.edges, np.r_[self.counts_e, self.counts_e[-1] if self.counts_e.size else 0.0], where="post", linewidth=1.6, label=g_e_labels[1])

        # Labels
        ax.set_xlabel(signal_label)
        ax.set_ylabel(counts_label)
        ax.set_xlim(self.edges[0], self.edges[-1])
        ax.set_title(title)

    def _separate_histogram_plot(self, signal_label, signal_unit, counts_label, g_e_labels, title):

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7.5, 5.8))

        # Generate plots
        axs[0].bar(self.left_edges, self.counts_g, width=self.bin_width, align="edge", alpha=0.9, edgecolor="black", linewidth=0.6, label=g_e_labels[0])
        axs[1].bar(self.left_edges, self.counts_e, width=self.bin_width, align="edge", alpha=0.9, edgecolor="black", linewidth=0.6, label=g_e_labels[1])

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
    