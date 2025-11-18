"""
Version: 1.0
Module: spinqubits_analysis.py

A class to analyze the spin qubits. 
Section 
I: qubit analysis
    1D fit=====> T1, T2 .
    2D fit=====> Chevron fit, Ramsy Fit
II: Single shot analysis
    1. time trace and histogram
    2. chevron pattern
    
## update notes:
1. CHevron Fit done
2. Ramsey Fit done

##following alexei already did.
3. gTensorAnalysis
    1. Vector magnet
    2. g tensor fitting
    3. Noise spectrum analysis

Author: [Pannnnnnn / HQC]
create_Date: [2025-10-27]
V1_Date: [2025-10-29]
V1.1_Date: [2025-11-14]
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy import constants
from scipy.optimize import curve_fit, minimize
import lmfit
import scipy

from scipy.special import erf
from numpy.polynomial.legendre import leggauss

from dataAnalysis.base import DataSet, ConcatenatedDataSet
import numbers

def compute_g_factor(freq:float|np.ndarray, B:float|np.ndarray):
    """
    Compute the g factor given the Larmor frequency and the magnitude of the magnetic field.

    Arguments:
        freq: Larmor frequency
        B: Magnitude of the magnetic field.

    Returns: The g factor.
    """
    h = constants.Planck
    mu_B = constants.physical_constants['Bohr magneton'][0]
    
    return (h * freq) / (mu_B * B)


class SpinQubitAnalysis(DataSet):
    """
    Data analysis class for .
    """
    def __init__(self, exp, run_id, station=None, transpose = True):
        self.transpose = transpose
        try:
            if isinstance(run_id, list) and not run_id:
                raise ValueError('run_id must be a non-empty list or integer.')
            elif isinstance(run_id, numbers.Integral) and not isinstance(run_id, bool):
                super().__init__(exp=exp, run_id=run_id, station=station)
                if transpose == True:
                    self.time = self.independent_parameters['y']['values'] 
                    self.bin = self.independent_parameters['x']['values']
                    self.signal_mag = self.dependent_parameters['param_0']['values'].T
                    self.signal_pha = self.dependent_parameters['param_1']['values'].T
                else:
                    self.time = self.independent_parameters['x']['values']
                    self.bin = self.independent_parameters['y']['values']
                    self.signal_mag = self.dependent_parameters['param_0']['values']
                    self.signal_pha = self.dependent_parameters['param_1']['values']
            else:
                self.time = []
                self.bin = []
                self.signal_mag = []
                self.signal_pha = []
                
                for id in run_id:
                    super().__init__(exp=exp, run_id=id, station=station) 
                    self.bin.append(self.independent_parameters['x']['values'])
                    self.signal_mag.append(self.dependent_parameters['param_0']['values'].T)
                    self.signal_pha.append(self.dependent_parameters['param_1']['values'].T)
        except KeyError:
            super().__init__(exp=exp, run_id=run_id, station=station)
            self.time = self.independent_parameters['x']['values'] 
            self.signal_mag = self.dependent_parameters['param_0']['values'].T
            self.signal_pha = self.dependent_parameters['param_1']['values'].T

    def fit_T1(self):
        """
        Fit T1 from a relaxation trace using: y(t) = y_inf + A * exp(-t/T1)
    
        Parameters:
            t : 1D array (seconds)
            y : 1D array (same length)
    
        Returns:
            A dictionary with keys: T1, y_inf, A, yfit, popt, pcov, T1_stderr (1-sigma), r2
        """
        y = np.average(self.signal_mag, axis = 0)
        t = self.time

        t = np.asarray(t, float)
        y = np.asarray(y, float)
        
        assert t.ndim == 1 and y.ndim == 1 and t.size == y.size and t.size >= 3
    
        # Model
        def model(tt, y_inf, A, T1):
            return y_inf + A * np.exp(-np.maximum(tt, 0)/T1)
    
        # Initial guesses
        tspan = t[-1] - t[0] if t.size > 1 else 1.0
        dt    = max(np.median(np.diff(t)), 1e-12)
        # y_inf ~ mean of last 10% points (fallback to last point)
        n_tail = max(1, int(0.1*len(y)))
        y_inf0 = float(np.mean(y[-n_tail:]))
        A0     = float(y[0] - y_inf0)
        # rough T1: time to reach ~63% toward y_inf
        target = y_inf0 + A0*np.exp(-1.0)
        # find closest point to target
        idx = int(np.argmin(np.abs(y - target)))
        T10 = max(t[idx] - t[0], 0.1*tspan)
    

    
        p0 = [y_inf0, A0, T10]
    
        popt, pcov = curve_fit(model, t, y, p0=p0, maxfev=10000)
        yfit = model(t, *popt)
    
        # Fit quality and errors
        ss_res = float(np.sum((y - yfit)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2)) + 1e-18
        r2     = 1.0 - ss_res/ss_tot
        T1_stderr = float(np.sqrt(max(pcov[2,2], 0.0))) if pcov.shape == (3,3) else np.nan
    
        return {
            "T1": popt[2],
            "y_inf": float(popt[0]),
            "A": float(popt[1]),
            "yfit": yfit,
            "popt": popt,
            "pcov": pcov,
            "T1_stderr": T1_stderr,
            "r2": r2,
        }
    def plot_T1_fit(self, fit_res=None, ax=None, show=True, time_unit='ns'):
        """
        Plot experimental T1 relaxation data and fitted curve.
    
        Parameters
        ----------
        t : 1D array
            Time values (seconds).
        y : 1D array
            Signal values (same length as t).
        fit_res : dict, optional
            Output from `fit_T1`. If None, the function will call fit_T1.
        ax : matplotlib.axes.Axes, optional
            If provided, plot on this axis.
        show : bool
            Whether to call plt.show() at the end.
        """
        t = self.time
        y = np.average(self.signal_mag, axis = 0)
        
        if fit_res is None:
            fit_res = self.fit_T1()
    
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        if time_unit == 'ns': ## Remember the unit of OPX is in ns!
            scale = 1
        elif time_unit == 'us':
            scale = 1e-3
        elif time_unit == 'ms':
            scale = 1e-6
        elif time_unit == 's':
            scale = 1e-9
        # Plot raw data
        ax.plot(t*scale, y, 'o', label='Experimental data', markersize=5, color='blue')
    
        # Plot fitted curve
        ax.plot(t*scale, fit_res['yfit'], ls='-', label=f"Fit: T1 = {fit_res['T1']*scale:.3f} ± {fit_res['T1_stderr']*scale:.3f}{time_unit}  \nR² = {fit_res['r2']:.4f}", color='red')
    
        # Aesthetics
        ax.set_xlabel(f"Time {time_unit}")
        ax.set_ylabel("Signal")
        ax.set_title("T1 Relaxation Fit")
        ax.legend()
        ax.grid(True)
    
        if show:
            plt.tight_layout()
            plt.show()

    def fit_T2star(self, f_hint=None, alpha=2.0):
        """
        Fit y(t) ≈ y0 + A cos(2π f t + φ) * exp(-(t/T2*)**alpha)
        Returns dict with T2*, f, phase, etc.  t in seconds.
        """
        return _fit_T2star(self.signal_mag,self.time, f_hint, alpha)
        
    def plot_T2star_fit(self, fit_res = None, xlimit = 1, ax=None, alpha = 2, f_hint = None, time_unit="us", title="Ramsey (T2*) fit"):
        """
        t: time array (seconds)
        y: data array
        fit_res: dict returned by fit_T2star(...)
        """
        
        if ax is None:
            fig, ax = plt.subplots()
        if fit_res is None:
            fit_res = self.fit_T2star(f_hint=None, alpha=alpha)
        # unpack fit
        y = self.signal_mag
        t = self.time

        
        y0 = fit_res["y0"]; A = fit_res["A"]; f = fit_res["f"]
        phi = fit_res["phi"]; T2s = fit_res["T2_star"]; alpha = fit_res.get("alpha", alpha)
    
        # smooth curve
        t_fit = np.linspace(t.min(), t.max(), 1000)
        y_fit = y0 + A*np.cos(2*np.pi*f*t_fit + phi) * np.exp(-(np.maximum(t_fit,0)/T2s)**alpha)
    
        # plot
        if time_unit == 'ns': ## Remember the unit of OPX is in ns!
            scale = 1
        elif time_unit == 'us':
            scale = 1e-3
        elif time_unit == 'ms':
            scale = 1e-6
        elif time_unit == 's':
            scale = 1e-9
            
        ax.plot(t*scale, y, "o", ms=4, label=f"data, \nR² = {fit_res['r2']:.4f}")
        ax.plot(t_fit*scale, y_fit, "-", lw=2, label=f"fit (T2* = {T2s*scale:.2f} {time_unit})")
        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel("Signal")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, xlimit*t[-1]*scale)
        return ax


        
    def _ramsey2d(self, t_s, f_Hz, y0, A, f0, phi, T2s,offset_t, alpha):
        tt   = np.asarray(t_s)[:, None]                   # (Nt,1)
        ff   = np.asarray(f_Hz)[None, :]                  # (1,Nf)
    
        env  = np.exp(-(tt/T2s)**alpha)
        phase = 2*np.pi*(ff-f0)*(tt+offset_t) + phi
    
        baseline = y0
        return (y0+A*np.cos(phase)*env).ravel()
    
    def fit_T2_2D(self, alpha = 2.0, offset_t = 100, fit_methods = 'lbfgsb'):
        # ==== data (make sure of units) ====
        freq = self.bin             # Hz, shape (Nf,)
        t_ns = self.time            # ns, shape (Nt,)
        t_s  = t_ns * 1e-9                                    # seconds
        signal_mag  = self.signal_mag.T          # (Nt, Nf)
        
        # ==== model ====
        # Center *only the phase* in time to reduce f0-phi correlation.
        
        model = lmfit.Model(self._ramsey2d, independent_vars=['t_s', 'f_Hz'])
        
        # ==== initial guesses ====
        A0    = 0.5*(np.max(signal_mag)-np.min(signal_mag))
        
        p = model.make_params(
            y0=0, A=A0,
            f0=float(np.mean(freq)),                          # start near mean
            phi=0,
            T2s=12e6,                                        # 10 µs
            alpha=alpha,
            offst_t = 0
        )
        # ==== bounds & choices ====
        p['A'].set(min=A0*0.1, max=A0*2+A0*0.001)
        p['alpha'].set(vary=False)  
        p['offset_t'].set(vary=True)
        p['phi'].set(vary = True)

        p['T2s'].set(min=0, max=3*np.max(t_s))
        p['f0'].set(min=freq.min(), max=freq.max())
        p['phi'].set(min=-np.pi, max=np.pi)
        p['offset_t'].set(min=0, max=1000)
        
        # ==== fit ====
        res = model.fit(signal_mag.ravel(), params=p,
                        t_s=t_s, f_Hz=freq,
                        method=fit_methods)
        signal_mag_fit     = res.best_fit.reshape(signal_mag.shape)
        return res, signal_mag_fit

            
    # update the model after some calculation
    def _rabi2d(self, t_s, f_Hz, f_rabi, y0, A, f0, phi, T2s, alpha):
        if t_s.ndim == 1:
            tt = np.asarray(t_s)[:, None]         # (Nt,1)
            ff = np.asarray(f_Hz)[None, :]        # (1,Nf)
    
            env = np.exp(-np.power(tt / T2s, alpha))
            general_rabi = (ff - f0)**2 + (f_rabi**2)
            phase = 2*np.pi*np.sqrt(general_rabi) * tt + phi
    
            # population-type contrast: (f_rabi^2) / general_rabi
            scale_factor = (f_rabi**2) / (general_rabi)
    
            # population oscillation: 0.5 * (1 - cos)
            return (y0 + A * scale_factor * 0.5 * (1 - np.cos(phase)) * env).ravel()
    
        if t_s.ndim == 2:
            tt = np.asarray(t_s)                  # (Nt,Nf)
            ff = np.asarray(f_Hz)                 # (Nt,Nf)
    
            env = np.exp(-np.power(tt / T2s, alpha))
            general_rabi = (ff - f0)**2 + (f_rabi**2)
            phase = 2*np.pi*np.sqrt(general_rabi) * tt + phi
    
            scale_factor = (f_rabi**2) / (general_rabi)
            return y0 + A * scale_factor * 0.5 * (1 - np.cos(phase)) * env

        
    def fit_T2_rabi_2D(self,alpha = 2, index = 50, fit_methods = 'lbfgsb', flip_cut = False):
        # ==== data (make sure of units) ====
        freq = self.bin           
        t_s = self.time /10**9        
        signal_mag  = self.signal_mag.T      
    
        # ==== model ====
        # Center *only the phase* in time to reduce f0-phi correlation.
        
        model = lmfit.Model(self._rabi2d, independent_vars=['t_s', 'f_Hz'])
        if flip_cut == False:
            res_1d = _fit_T2star(signal_mag[:,index], self.time, alpha = alpha)
        else:
            res_1d = _fit_T2star(signal_mag[index], self.time, alpha = alpha)
        # ==== initial guesses ====
        y0  =  res_1d['y0']
        T2s = res_1d['T2_star']*10**-9
        A0  = res_1d['A']
        f_rabi  = res_1d['f']*10**9
        
        p = model.make_params(
            y0=y0, A=A0,
            f0=float(np.mean(freq)),                       
            phi=0,
            T2s=T2s,                                     
            alpha=alpha,
            f_rabi = f_rabi
        )
        
        # ==== bounds & choices ====
        p['alpha'].set(vary=False) 
        p['T2s'].set(min=0) 
        p['f0'].set(min=0) 
        p['f_rabi'].set(min=0) 
        # ==== fit ====
        res = model.fit(signal_mag.ravel(), params=p,
                        t_s=t_s, f_Hz=freq,
                        method=fit_methods)

        freq,time = np.meshgrid(self.bin,self.time/10**9)
        signal_mag_fit = self._rabi2d(time, freq,res.params['f_rabi'].value, 
                                 res.params['y0'].value, 
                                 res.params['A'].value, 
                                 res.params['f0'].value, 
                                 res.params['phi'].value, 
                                 res.params['T2s'].value, 
                                 res.params['alpha'].value)
        
        return res, signal_mag_fit
        
    def plot_fit_T2_2D(self, res, model = 'ramsey'):
        def track(Z): 
            peak_index = [] 
            for i in range(Z.shape[1]): 
                peak_index.append(scipy.signal.find_peaks(Z[:,i])) 
            return peak_index
        
        # --- evaluate fitted surface (or use your own Z array) ---
        freq = self.bin             # Hz, shape (Nf,)
        t_ns = self.time            # ns, shape (Nt,)
        t_s  = t_ns * 1e-9                                    # seconds
        signal_mag  = self.signal_mag.T
        
        if model == 'ramsey':
            f0   = res.params['f0'].value
            phi  = res.params['phi'].value
            y0  = res.params['y0'].value
            T2s  = res.params['T2s'].value
            A  = res.params['A'].value
            offset_t  = res.params['offset_t'].value
            alpha = res.params['alpha'].value
        

            # --- run your tracker on the FITTED surface ---

            # 2. Evaluate model: returns flat array
            signal_mag_flat = self._ramsey2d(t_s, freq, y0, A, f0, phi, T2s, offset_t, alpha)
        if model == 'rabi':
            f0   = res.params['f0'].value
            phi  = res.params['phi'].value
            y0  = res.params['y0'].value
            T2s  = res.params['T2s'].value
            A  = res.params['A'].value
            alpha = res.params['alpha'].value
            f_rabi = res.params['f_rabi'].value

            # --- run your tracker on the FITTED surface ---

            # 2. Evaluate model: returns flat array
            signal_mag_flat = self._rabi2d(t_s, freq,f_rabi, y0, A, f0, phi, T2s, alpha)
        
        # 3. Reshape to 2D (time x freq)
        signal_mag_fit = signal_mag_flat.reshape(t_s.size, freq.size)
        
        # 4. Run peak tracker (per-column)
        peaks_per_col = track(signal_mag_fit)

        # --- convert peak indices -> (freq, time) coordinates for plotting ---
        f_pts, t_pts = [], []
        for j, item in enumerate(peaks_per_col):
            # handle either tuple (idx, props) or just idx array
            idxs = item[0] if isinstance(item, tuple) or (isinstance(item, list) and len(item) == 2) else np.asarray(item)
            if idxs is None or len(idxs) == 0: 
                continue
            f_pts.extend([freq[j]] * len(idxs))
            t_pts.extend(t_ns[idxs])
        
        f_pts = np.asarray(f_pts)
        t_pts = np.asarray(t_pts)
        
        # --- plot experimental map and overlay the peak coordinates ---
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        
        # --- experimental background ---
        pcm = ax.pcolormesh(freq, t_ns, signal_mag, shading="auto", cmap="viridis")
        cbar = plt.colorbar(pcm, ax=ax, pad=0.02)
        cbar.set_label("Experimental signal")
        
        # --- peak overlay (visually improved) ---
        ax.scatter(f_pts, t_pts,
                   s=16,                      # larger dots
                   facecolors='deepskyblue', # fill color
                   edgecolors='black',       # contrast outline
                   linewidths=0.4,
                   alpha=0.9,
                   label="peaks from fit")
        
        # --- axis & labels ---
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Time (ns)")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(alpha=0.2, linestyle="--")
        fig.tight_layout()
        plt.show()
        return fig,ax
        

class BFieldInPlaneAngleSweep(ConcatenatedDataSet, DataSet):
    def __init__(self, exp, run_id:int|list, angle_values=None, angle_label="In_plane_angle", angle_unit="Deg"):
        if isinstance(run_id, int):
            DataSet.__init__(self, exp, run_id)
        elif isinstance(run_id, list):
            if angle_values is None:
                num_in_plane_angles = run_id[1] - run_id[0] + 1
                angle_values = np.linspace(0, 360, num_in_plane_angles)
            ConcatenatedDataSet.__init__(self, exp, run_id, angle_values, angle_label, angle_unit)
        self.freq = self.independent_parameters['y']['values']
        self.angle = self.independent_parameters['x']['values']
        self.mag = self.dependent_parameters['param_0']['values']

    def normalize_data_from_average(self, params_to_normalize = None, axis=0, operation="subtract"):
        super().normalize_data_from_average(params_to_normalize, axis, operation)
        self.mag_norm = self.dependent_parameters['param_0_normalized']['values']

    def find_resonances(self, min_separation_n_points=0, filter_percentile=None, sort_by='freq'):
        # Determine whether to use normalize data or not
        if hasattr(self, 'mag_norm'):
            data = self.mag_norm
        else:
            data = self.mag

        num_freqs = len(self.freq)
        num_angles = len(self.angle)
        idx1 = -np.ones(num_angles, dtype=int)
        idx2 = -np.ones(num_angles, dtype=int)
        a1   = np.full(num_angles, np.nan, dtype=float)
        a2   = np.full(num_angles, np.nan, dtype=float)

        for angle_index in range(num_angles):
            # Take the linecut for the current angle value
            col = data[:, angle_index]
            working_linecut = col.copy()

            # Optional simple background removal
            if filter_percentile is not None:
                thr = np.percentile(working_linecut, filter_percentile)
                working_linecut[working_linecut < thr] = -np.inf

            # Fin the indices of the top-2 values
            top2 = np.argpartition(working_linecut, -2)[-2:]
            top2 = top2[np.argsort(working_linecut[top2])[::-1]]  # sort by decreasing amplitude
            i1 = int(top2[0])
            i2 = int(top2[1])

            # Enforce a minimum separation if requested
            if min_separation_n_points > 0 and abs(i1 - i2) < min_separation_n_points:
                # look for the next best index not within min_sep of i1
                mask = np.ones(num_freqs, dtype=bool)
                lo_idx = max(0, i1 - min_separation_n_points)
                hi_idx = min(num_freqs, i1 + min_separation_n_points + 1)
                mask[lo_idx:hi_idx] = False
                if filter_percentile is not None:
                    mask &= (col >= thr)
                if np.any(mask):
                    j = np.argmax(working_linecut[mask])
                    i2 = int(np.flatnonzero(mask)[j])
                    # i2 = j
                

            # Save the values
            idx1[angle_index], a1[angle_index] = i1, col[i1]
            if i2 >= 0 and np.isfinite(working_linecut[i2]):
                idx2[angle_index], a2[angle_index] = i2, col[i2]

            # optional reorder by frequency (so f1 <= f2)
            if sort_by == "freq" and idx2[angle_index] >= 0 and self.freq[idx1[angle_index]] > self.freq[idx2[angle_index]]:
                idx1[angle_index], idx2[angle_index] = idx2[angle_index], idx1[angle_index]
                a1[angle_index], a2[angle_index]     = a2[angle_index], a1[angle_index]

        f1 = self.freq[idx1]
        f2 = np.where(idx2 >= 0, self.freq[idx2], np.nan)

        self.results = {"idx1": idx1, "idx2": idx2, "f1": f1, "f2": f2, "a1": a1, "a2": a2, "phi": self.angle}

        return self.results

    def extract_g_factors(self, B_in, B_out=0, **kwargs):
        if not hasattr(self, "results"):
            self.find_resonances(**kwargs)
        B_tot = np.sqrt(B_in**2 + B_out**2)
        g1 = compute_g_factor(self.results['f1'], B_tot)
        g2 = compute_g_factor(self.results['f2'], B_tot)
        self.results['g1'] = g1
        self.results['g2'] = g2

        return self.results
    
    def plot_g_factors(self, angle_labels=None, **kwargs):
        """
        Plot the calculated g factor in the plane.

        Args:
            angle_label: If None, labels are in degrees. If 'xy', 'yz' or 'xz', then the in-plane axes labels are used.

        Returns: fig, ax, (plot1, plot2)
        """
        g1 = self.results['g1']
        g2 = self.results['g2']
        angle = self.angle

        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        plot1 = ax.scatter(np.deg2rad(angle),  g1, lw=2, label="Q1", alpha=0.7)
        plot2 = ax.scatter(np.deg2rad(angle),  g2, lw=2, label="Q2", alpha=0.7)
        legend = ax.legend()
        legend.set_frame_on(False)

        if angle_labels == 'xy':
            ax.set_xticklabels(['+x', '', '+y', '', '-x', '', '-y', ''])
        elif angle_labels == 'yz':
            ax.set_xticklabels(['+y', '', '+z', '', '-y', '', '-z', ''])
        elif angle_labels == 'xz':
            ax.set_xticklabels(['+x', '', '+z', '', '-x', '', '-z', ''])

        return fig, ax, (plot1, plot2)




class SingleShotAnalysis(DataSet):
    """
    Data analysis class for single shot experiment
    """
    def __init__(self, exp, run_id:int|list, station=None):
        """
        Parameters:
            exp: The experiment handler.
            run_id (int, list): run_id or [run_id_ground, run_id_excited]
        
        """
        if isinstance(run_id, numbers.Integral):    # one single run_id provided
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
        
    def build_histogram_time_trace(self, num_bins: int, time_index: int = 0, plot_average_time_trace=False, clip_quantiles:tuple=(0,1)) -> dict:
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
        return hist_dict
    

    def plot_histograms(
        self,
        kind: str = "stacked",
        plot_CDF: bool = False,
        labels: tuple[str, str] = ("Ground", "Excited"),
        x_label: str = "Signal",
        title: str | None = "Single-shot histograms",
        signal_unit: str = "mV",
        fig = None,
        ax = None,
        hdict: dict | None = None,
    ):
        """
        Plot the histogram for the single-shot measurement. This assumes that build_histogram_time_trace has been
        previously run.

        Parameters:
            kind (string, optional): Histogram type. Choose among 'stacked', 'stacked_only', 'overlay', 'separate'.
            plot_CDF (bool, optional): If true, the CDF is fitted to obtain the optimal thresholding value and the visibility.
            labels (tuple, optional): Tuple of strings for the ground state and excited state labels.
            x_label (string, optional): String for the x label.
            signal_unit (string, optional): Choose among 'V', 'mV', 'uV' and the signal data will be converted into the desired unit.
        """
        # Get data from the histogram dictionary
        if hdict is None:
            if not hasattr(self, "histogram_dict"):
                raise ValueError("Run build_histogram_time_trace(...) first or pass hdict explicitly.")
            hdict = self.histogram_dict
        for key in ("histogram_ground", "histogram_excited", "common_edges"):
            if key not in hdict:
                raise KeyError(f"Missing key '{key}' in histogram dict.")
        hg = hdict["histogram_ground"]
        he = hdict["histogram_excited"]
        
        
        # Divide the edges values to have them in the unit specified
        if signal_unit == 'mV':
            edges = hdict["common_edges"] * 1e3
            x_label += " (mV)"
        elif signal_unit == 'uV':
            edges = hdict["common_edges"] * 1e6
            x_label += " (uV)"
        elif signal_unit == 'V':
            edges = hdict["common_edges"]
            x_label += " (V)"

        if edges.ndim != 1 or edges.size != hg.size + 1 or edges.size != he.size + 1:
            raise ValueError("'common_edges' must be length N+1 where N=len(histogram_*).")
    
        # Get left bing edges, bin width and bin centers
        left_edges = edges[:-1]
        bin_width = np.diff(edges)
        centers = left_edges + 0.5 * bin_width
    
        # For the 'separate' type, generate two axes
        if kind == "separate": # return immediately the plot
            return _separate_histogram_plot(left_edges, bin_width, hg, he, centers, labels, x_label, "Counts", title)
        # For all other types, use only on axis
        else:
            if not fig and not ax:  # Generate fig and axis handler if not provided by the user
                fig, ax = plt.subplots(figsize=(7.5, 4.6))
            _beautify_axis(ax)
        # for the following types, edit labels and stuff later because they are the same
            if kind == "stacked":
                _stacked_histogram_plot(ax, left_edges, bin_width, hg, he, edges, labels)
            elif kind == 'stacked_only':
                _stacked_histogram_plot(ax, left_edges, bin_width, hg, he, edges, labels, stacked_only=True)
            elif kind == "overlay":
                _overlay_histogram_plot(ax, centers, hg, he, edges, labels)
            else:
                raise ValueError("'kind' must be 'stacked', 'stacked_only', 'overlay', or 'separate'.")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Counts")
        ax.set_xlim(edges[0], edges[-1])
        if title:
            ax.set_title(title)
    
        # Fit the CDF and add to plot
        if plot_CDF:
            ax2 = ax.twinx()
            _beautify_axis(ax2, right=True)
            ax2.set_ylabel("Probability")
            ax2.set_ylim(0, 1.05)
    
            # CALL visibility computation
            metrics = compute_visibility_from_histogram(hdict)
            ps, pt, vis, centers, best_vis, threshold = (
                metrics["probability_singlet"],
                metrics["probability_triplet"],
                metrics["visibility"],
                metrics["centers"],
                metrics["best_visibility"],
                metrics["threshold_voltage"]
            )
    
            # Plot on right y-axis
            ax2.plot(centers, ps, label="Ground CDF", color="tab:blue", linestyle="--", linewidth=1.5)
            ax2.plot(centers, pt, label="Excited CDF", color="tab:orange", linestyle="--", linewidth=1.5)
            ax2.plot(centers, vis, label="Visibility", color="tab:green", linestyle="-", linewidth=1.5)
    
            # Annotate threshold on left axis
            ax.axvline(threshold, color="gray", linestyle=":", linewidth=1.5, label="Threshold")
            ax.annotate(
                f"Max visibility = {best_vis:.3f}\nThreshold = {threshold:.3f} {signal_unit}",
                xy=(threshold, 0.95 * ax.get_ylim()[1]),
                xytext=(10, -30), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", lw=1),
                fontsize=10, ha="left"
            )
    
            # Combined legend
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=False)
        else:
            ax.legend(frameon=False)
    
        plt.tight_layout()
        plt.show()
        return fig, ax
        
    def histogram_chevron(self, mag_bins=100, time_bins=100, num_bins=100, plot = True):
        """
        Generate histograms of rf voltage magnitudes (|S21|) from chevron measurement data.
    
        Returns:
            count_list: List of 1D histograms (counts)
            edge_list: Corresponding bin edges
        """
        self.count_list = []
        self.edge_list = []
    
        def process_single_data_set(bin, time_array, magnitude):
            num_shots = int(bin[-1])
            n = magnitude.size
            mag_2d = magnitude.reshape(num_shots, n // num_shots)
            x, y = time_mag_points(mag_2d, time_array)
    
            # Plot 2D histogram (chevron)
            H, xedges, yedges = np.histogram2d(x, y, bins=[mag_bins, time_bins])

    
            # Make 1D histogram
            A = np.asarray(mag_2d)
            t = np.asarray(time_array)
    
            if A.shape[-1] != t.size and A.shape[0] == t.size:
                A = A.T
    
            xvals = A.ravel()
            counts, edges = np.histogram(xvals, bins=num_bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            if plot:
                # --- 1D stacked histogram ---
                plt.figure(figsize=(5, 3))
                plt.plot(centers, counts, "^-", ms=5, lw=1.2, color="tab:blue")
                plt.xlabel("V_rf (V) or |S21|")
                plt.ylabel("Counts")
                plt.title("1D Histogram", fontsize=11)
                plt.grid(alpha=0.3, linestyle="--")
                plt.tight_layout()
                plt.show()
                
                # --- 2D histogram (density map) ---
                plt.figure(figsize=(5.5, 3.5))
                plt.imshow(
                    H.T,
                    origin="lower",
                    aspect="auto",
                    cmap="viridis",
                    extent=(xedges[0], xedges[-1], yedges[0], yedges[-1])
                )
                plt.colorbar(label="Counts")
                plt.xlabel("V_rf or |S21|")
                plt.ylabel("Evolution time")
                plt.title("2D Histogram (Counts vs. Time)", fontsize=11)
                plt.tight_layout()
                plt.show()

    
            return counts, edges
    
        num_shot = self.num_shot
        time_array = self.time
        magnitude = self.signal_mag
        counts, edges = process_single_data_set(num_shot, time_array, magnitude)
        self.count_list.append(counts)
        self.edge_list.append(edges)
    
    
        return self.count_list, self.edge_list

        
    def fit_histogram(
        self,
        integration_time=50000,
        T1=50000,
        VS=0.00235,
        VT=0.00422,
        num_restarts=6,
        jitter=0.3,
        show_fitting_result = False
    ):
        """
        Fit 1D histograms to Barthel model and plot.
    
        Parameters
        ----------
        integration_time : float or list of float
            Integration time(s) in ns.
        T1 : float
            Initial guess for T1 (in ns).
        VS : float
            Initial guess for singlet voltage.
        VT : float
            Initial guess for triplet voltage.
        num_restarts : int
            Number of random restarts for fitting.
        jitter : float
            Jitter level used in restarts.
    
        Returns
        -------
        histogram_dict : dict
            Dictionary containing fit results and related data.
        """
        histogram_dict = {}
    
        if not hasattr(self, "count_list") or not self.count_list:
            raise RuntimeError("Run histogram_chevron first to generate count_list and edge_list")
    
        integration_times = (
            [integration_time] if isinstance(integration_time, (int, float))
            else list(integration_time)
        )
        ## loop over all data sets.
        for i, (counts, edges) in enumerate(zip(self.count_list, self.edge_list)):
            
            tm = float(integration_times[i]) if len(integration_times) > 1 else float(integration_time) ## time steps
            Ntot = float(counts.sum())
    
            init_guess = dict(VS=VS, VT=VT) ## provide the initial guess of voltage at peak of single and peak of triplet
            
            fit = fit_barthel_one(counts, edges, tm, T1, Ntot,
                                  init=init_guess, restarts=num_restarts, jitter=jitter)## main fitting funciton
            if show_fitting_result:
                print(f"Fit result for trace {i}:", fit)
    
            centers = 0.5 * (edges[:-1] + edges[1:])
            lam = expected_counts_barthel(edges, tm, fit['VS'], fit['VT'], fit['sigma'], fit['pT'], T1, Ntot)
    
            result = plot_hist_with_cdf_bw(
                counts, edges, fit['VS'], fit['VT'], fit['sigma'], fit['pT'], T1, tm, nq=128,
                title=f'Histogram & CDFs at integration time={tm} ns'
            )
    
            histogram_dict = {
                "fit": fit,
                "edges": edges,
                "counts": counts,
                "lam": lam
            }
            histogram_dict.update(result)
    
        self.histogram_dict = histogram_dict
        return histogram_dict
            
            
class Histogram2D(SingleShotAnalysis):
        def __init__(self, exp, run_id=None, station=None, save_path=None):
            try:
                if isinstance(run_id, list) and not run_id:
                    raise ValueError('run_id must be a non-empty list or integer.')
                elif isinstance(run_id, numbers.Integral) and not isinstance(run_id, bool): ## if it is not a list
                    super().__init__(exp=exp, run_id=run_id, station=station)
                    self.time = self.independent_parameters['y']['values'] #### WARNING: the time for opx is ns!
                    self.bin = self.independent_parameters['x']['values']
                    self.signal_mag = self.dependent_parameters['param_0']['values'].T
                    self.signal_pha = self.dependent_parameters['param_1']['values'].T
                else: ## if it is a list
                    run_id = np.asarray(run_id)
                    if run_id.ndim == 1:
                        self.time = []
                        self.bin = []
                        self.signal_mag = []
                        self.signal_pha = []
                        
                        for id in run_id:
                            super().__init__(exp=exp, run_id=id, station=station)
                            self.time.append(self.independent_parameters['y']['values']) #### WARNING: the time for opx is ns!
                            self.bin.append(self.independent_parameters['x']['values'])
                            self.signal_mag.append(self.dependent_parameters['param_0']['values'].T)
                            self.signal_pha.append(self.dependent_parameters['param_1']['values'].T)
                            
                    if run_id.ndim == 2:
                        self.time_e = []
                        self.bin_e = []
                        self.signal_mag_e = []
                        self.signal_pha_e = []
    
                        self.time_g = []
                        self.bin_g = []
                        self.signal_mag_g = []
                        self.signal_pha_g = []
                        
                        for id_list in range(len(run_id)):
                            g_ind, e_ind = id_list
                            super().__init__(exp=exp, run_id=g_ind, station=station)
                            self.time_g.append(self.independent_parameters['y']['values']) #### WARNING: the time for opx is ns!
                            self.bin_g.append(self.independent_parameters['x']['values'])
                            self.signal_mag_g.append(self.dependent_parameters['param_0']['values'].T)
                            self.signal_pha_g.append(self.dependent_parameters['param_1']['values'].T)
    
                            
                            super().__init__(exp=exp, run_id=e_ind, station=station)
                            self.time_e.append(self.independent_parameters['y']['values']) #### WARNING: the time for opx is ns!
                            self.bin_e.append(self.independent_parameters['x']['values'])
                            self.signal_mag_e.append(self.dependent_parameters['param_0']['values'].T)
                            self.signal_pha_e.append(self.dependent_parameters['param_1']['values'].T)
                            
            except KeyError:
                super().__init__(exp=exp, run_id=run_id, station=station)
                self.time = self.independent_parameters['x']['values'] #### WARNING: the time for opx is ns!
                self.signal_mag = self.dependent_parameters['param_0']['values'].T
                self.signal_pha = self.dependent_parameters['param_1']['values'].T
        
        def plot_2d_histogram(self):
            histogram_time_trace



##fitting function for T2
def _model_T2star(tt, y0, A, f, phi, T2s, alpha=1):
        return y0 + A*np.cos(2*np.pi*f*tt + phi) * np.exp(-(np.maximum(tt,0)/T2s)**alpha)
        
def _fit_T2star(signal_mag, time, f_hint=None, alpha=2.0):
    """
    Fit y(t) ≈ y0 + A cos(2π f t + φ) * exp(-(t/T2*)**alpha)
    Returns dict with T2*, f, phase, etc.  t in seconds.
    """
    y = signal_mag
    t = time

    t = np.asarray(t, float)
    y = np.asarray(y, float)


    # crude guesses
    y0  = float(np.mean(y))
    A0  = 0.5*float(y.max()-y.min()) or 1e-6
    dt  = float(np.median(np.diff(t))) if t.size > 1 else 1e-9
    phi0 = 0.0
    T20  = max(0.2*(t[-1]-t[0]), 5*dt)

    
    if f_hint is None:
        Y    = np.abs(np.fft.rfft((y - y0) * np.hanning(len(y))))
        freqs= np.fft.rfftfreq(len(y), d=dt)
        idx  = np.argmax(Y[1:]) + 1 if Y.size > 1 else 0
        
        f0   = float(freqs[idx]) if idx < freqs.size else 1.0/max(t[-1]-t[0], 1e-6)
        print(f0)
    else:
        f0 = float(f_hint)
    # bounds
    p0    = [y0, A0, f0, phi0, T20]

    popt, pcov = curve_fit(_model_T2star, t, y, p0=p0, maxfev=100000)
    yfit = _model_T2star(t, *popt)

    ss_res = float(np.sum((y - yfit)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2     = 1.0 - ss_res/ss_tot
    
    return {
        "T2_star": float(popt[4]),
        "f": float(popt[2]),
        "phi": float(popt[3]),
        "A": float(popt[1]),
        "y0": float(popt[0]),
        "yfit": yfit,
        "alpha": alpha,
        "popt": popt,
        "pcov": pcov,
        "r2": r2
    }

def plot_2D_histogram(exp,run_id,station, num_bins,integration_time_array, plot_1D = True, xmax = None, xmin = None, time_unit = 'us', plot_visibility = False, plot_threshold = False):
    
    """
    run_id: list
    num_bins: int
    plot: boolen
    """
    
    histogram_stack_list = []
    histogram_ground_list = []
    histogram_excited_list = []
    common_edges_list = []
    visibility_list = []
    threshold_list = []
    
    if time_unit == 'ns': ## Remember the unit of OPX is in ns!
        scale = 1
    elif time_unit == 'us':
        scale = 1e-3
    elif time_unit == 'ms':
        scale = 1e-6
    elif time_unit == 's':
        scale = 1e-9
    for i in range(len(run_id)):
        run_id_eg = run_id[i]
        SSA = SingleShotAnalysis(exp, run_id_eg, station)
        hist_dictionary = SSA.build_histogram_time_trace(num_bins = num_bins, time_index=0, plot_average_time_trace = plot_1D)
        histogram_stack_list.append(hist_dictionary['histogram_stack'])
        histogram_ground_list.append(hist_dictionary['histogram_ground'])
        histogram_excited_list.append(hist_dictionary['histogram_excited'])
        common_edges_list.append(hist_dictionary['common_edges'])

        metrics = compute_visibility_from_histogram(hist_dictionary)
        ps, pt, vis, centers, best_vis, threshold = (
            metrics["probability_singlet"],
            metrics["probability_triplet"],
            metrics["visibility"],
            metrics["centers"],
            metrics["best_visibility"],
            metrics["threshold_voltage"]
        )
        visibility_list.append(best_vis)
        threshold_list.append(threshold)
        
    histogram_stack_array = np.asarray(histogram_stack_list)
    histogram_ground_array  = np.asarray(histogram_ground_list)
    histogram_excited_array  = np.asarray(histogram_excited_list)
    common_edges_array  = np.asarray(common_edges_list)
    
    if xmax == None:
        xmax = np.max(common_edges_array)
    if xmin == None:
        xmin = np.min(common_edges_array)
    # now we regroup into 2
    
    integration_time_array = scale * integration_time_array
    
    common_edges = make_common_edges(xmin, xmax, nbins=num_bins)  # or bin_width=BIN_WIDTH
    histogram_stack_2d = np.vstack([rebin_counts(c, e, common_edges) for c, e in zip(histogram_stack_list, common_edges_list)])

    cmap = 'Greys_r'
    
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    im = ax.imshow(
        histogram_stack_2d,
        origin='lower',
        aspect='auto',
        extent=(common_edges[0], common_edges[-1],
                float(integration_time_array.min()), float(integration_time_array.max())),
        cmap=cmap,
        interpolation='nearest',        # crisp (no blending)
        norm=PowerNorm(gamma=0.6)       # subtle contrast boost; remove for linear
    )
    
    ax.set_xlabel('V_rf (V)')
    ax.set_ylabel(r'$\tau_M$ (ns)')
    
    cb = fig.colorbar(im, ax=ax, label='Counts')
    cb.ax.tick_params(color='0.2')
    cb.outline.set_edgecolor('0.2')
    
    plt.tight_layout()
    plt.show()

    if plot_visibility:
        nice_one_d_plot(integration_time_array, visibility_list)
    if plot_threshold:
        nice_one_d_plot(integration_time_array, threshold_list)
    
    return fig, ax

def nice_one_d_plot(t, v):
    # best point
    imax = int(np.nanargmax(v))
    t_best, v_best = t[imax], v[imax]
    
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    # original points
    ax.plot(t, v, ms=4, alpha=0.5, label='visibility')
    # highlight max
    ax.axvline(t_best, ls='--', lw=1.2, color='tab:gray')
    ax.plot(t_best, v_best, marker='*', ms=12, color='tab:orange', zorder=5,
            label=f'peak: t={t_best:.3g}, vis={v_best:.3f}')
    
    # cosmetics
    ax.set_xlabel('Integration time (ns)')
    ax.set_ylabel(r'Visibility $F_S+F_T-1$')
    ax.set_ylim(0, 1.05)
    ax.grid(True, which='both', alpha=0.3)
    ax.minorticks_on()
    ax.legend(frameon=True, loc='lower right')
    fig.tight_layout()
    plt.show()
    return ax, fig
    
## 2D histogram
def make_common_edges(xmin, xmax, nbins=None, bin_width=None):
    if bin_width is not None:
        nbins = int(np.floor((xmax - xmin) / bin_width))
        xmax = xmin + nbins * bin_width          # trim to an exact multiple
        return xmin + np.arange(nbins + 1) * bin_width
    elif nbins is not None:
        return np.linspace(xmin, xmax, nbins + 1)
    else:
        raise ValueError("Provide nbins or bin_width")

def rebin_counts(counts, old_edges, new_edges):
    """Rebin histogram counts from old_edges -> new_edges.
       Counts outside [new_edges[0], new_edges[-1]] are discarded."""
    counts = np.asarray(counts, float)
    old_edges = np.asarray(old_edges, float)
    new_edges = np.asarray(new_edges, float)

    # cumulative counts at old edges
    cum = np.concatenate(([0.0], np.cumsum(counts)))
    # density per old bin
    widths = old_edges[1:] - old_edges[:-1]
    dens = counts / widths

    # for each new edge, find the old bin to its left
    j = np.searchsorted(old_edges, new_edges, side='right') - 1
    j = np.clip(j, 0, counts.size - 1)

    lefts  = old_edges[j]
    rights = old_edges[j + 1]
    dens_j = dens[j]

    # clamp new edges to the interior of the old bins (gives 0 contribution outside)
    xclamped = np.clip(new_edges, lefts, rights)

    cum_new = cum[j] + dens_j * (xclamped - lefts)
    return np.diff(cum_new)
#-----------------------------------------------# helper functions  
## plot_histogram helper function

def _beautify_axis(ax, top=False, bottom=True, left=True, right=False):
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines['top'].set_visible(top)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)


def _stacked_histogram_plot(ax, x_left, w, hg, he, edges, labels, stacked_only=False):
    h_total = hg + he
    ax.bar(x_left, h_total, width=w, align="edge", alpha=0.35, edgecolor="black", linewidth=0.6, label="Total")
    if not stacked_only:
        ax.step(edges, np.r_[hg, hg[-1] if hg.size else 0.0], where="post", linewidth=1.8, alpha=0.85, label=labels[0])
        ax.step(edges, np.r_[he, he[-1] if he.size else 0.0], where="post", linewidth=1.8, alpha=0.85, label=labels[1])


def _overlay_histogram_plot(ax, centers, hg, he, edges, labels):
    ax.step(edges, np.r_[hg, hg[-1] if hg.size else 0.0], where="post", linewidth=1.6, label=labels[0])
    ax.step(edges, np.r_[he, he[-1] if he.size else 0.0], where="post", linewidth=1.6, label=labels[1])
    # ax.fill_between(centers, 0, hg, step="mid", alpha=0.15)
    # ax.fill_between(centers, 0, he, step="mid", alpha=0.15)


def _separate_histogram_plot(x_left, w, hg, he, centers, labels, x_label, y_label, title):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7.5, 5.8))
    axs[0].bar(x_left, hg, width=w, align="edge", alpha=0.9, edgecolor="black", linewidth=0.6, label=labels[0])
    axs[1].bar(x_left, he, width=w, align="edge", alpha=0.9, edgecolor="black", linewidth=0.6, label=labels[1])
    for a in axs:
        _beautify_axis(a)
        a.set_ylabel(y_label)
        a.legend(frameon=False)
    axs[1].set_xlabel(x_label)
    if title:
        axs[0].set_title(title)
    plt.tight_layout()
    plt.show()
    return fig, axs


def compute_visibility_from_histogram(hdict):
    """
    Compute singlet/triplet CDFs and visibility, using:
      - Singlet CDF integrated from +inf → left (1 - CDF)
      - Triplet CDF integrated from -inf → right (CDF)
    
    Returns full visibility trace and threshold voltage.

    #####TODO
    #WARNING
    # it is very important to define whether the singlet peak voltage is higher than triplet peak voltage or not. 
    # I might need to write a code to flip them.
    """
    hg = np.asarray(hdict["histogram_ground"], dtype=float)
    he = np.asarray(hdict["histogram_excited"], dtype=float)
    edges = np.asarray(hdict["common_edges"], dtype=float)

    total_s = hg.sum()
    total_t = he.sum()

    prob_singlet = np.cumsum(hg) / total_s
    prob_triplet = 1-np.cumsum(he) / total_t

    visibility_arr = prob_singlet + prob_triplet - 1
    idx_best = np.argmax(visibility_arr)

    centers = edges[:-1] + 0.5 * np.diff(edges)

    return {
        "probability_singlet": prob_singlet,
        "probability_triplet": prob_triplet,
        "visibility": visibility_arr,
        "best_visibility": float(visibility_arr[idx_best]),
        "threshold_voltage": float(edges[idx_best]),
        "centers": centers
    }


## helper functions chevron histogram

def time_mag_points(mag_2d, time):
    mag_2d = np.asarray(mag_2d)
    time = np.asarray(time)

    # Make sure the last axis matches time; transpose if the other way around
    if mag_2d.shape[-1] != time.size:
        if mag_2d.shape[0] == time.size:
            mag_2d = mag_2d.T
        else:
            raise ValueError("mag_2d must be (n_shots, n_time) or (n_time, n_shots).")

    n_shots, n_time = mag_2d.shape
    x = mag_2d.reshape(-1)              # magnitudes flattened
    y = np.tile(time, n_shots)          # repeat time for each shot
    return x, y

####################################
# for detail, check https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.160503
####################################
# ---------------- Core helper functions fitting models----------------
def _edges_to_centers(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, float)
    return 0.5 * (edges[:-1] + edges[1:])

def _gauss_mass_per_bin(edges: np.ndarray, mu, sigma: float) -> np.ndarray:
    """
    ∫ N(mu, σ^2) over each bin. Works with scalar or array mu.
    edges: (nbins+1,), mu: scalar or shape (...,)
    returns: shape mu.shape + (nbins,)
    """
    edges = np.asarray(edges, float)
    mu    = np.asarray(mu, float)
    z_hi = (edges[1:]  - mu[..., None]) / (np.sqrt(2) * sigma)
    z_lo = (edges[:-1] - mu[..., None]) / (np.sqrt(2) * sigma)
    return 0.5 * (erf(z_hi) - erf(z_lo))

def _time_quadrature(tm: float, nq: int):
    """Gauss–Legendre nodes/weights mapped to [0, tm]."""
    x, w = leggauss(int(nq))          # [-1,1]
    t  = 0.5 * (x + 1.0) * tm         # [0, tm]
    wt = 0.5 * tm * w
    return t, wt
# ---------------- Barthel model ----------------
def nS_bins(edges, VS, sigma, pT):
    """Unconditional singlet mass per bin; sums to (1 - pT)."""
    return (1.0 - pT) * _gauss_mass_per_bin(edges, VS, sigma)

def nT_bins(edges, VS, VT, sigma, pT, T1, tm, nq=64):
    """
    Unconditional triplet mass per bin (no-relax + relax during readout).
    Sums to pT.
    """
    edges = np.asarray(edges, float)
    # no relaxation
    Pt_no = pT * np.exp(-tm / T1) * _gauss_mass_per_bin(edges, VT, sigma)

    # relaxation during readout: mean drifts VS -> VT
    t, wt = _time_quadrature(tm, nq)                      # (nq,)
    mu_t = VS + (VT - VS) * (t / tm)                      # (nq,)
    gb   = _gauss_mass_per_bin(edges, mu_t, sigma)        # (nq, nbins)
    weights = (np.exp(-t / T1) / T1) * wt                 # (nq,)
    Pt_relax = pT * np.sum(weights[:, None] * gb, axis=0) # (nbins,)

    return Pt_no + Pt_relax

def barthel_probs_per_bin(edges, tm, VS, VT, sigma, pT, T1, nq=64):
    """Total unconditional mass per bin: nS + nT (sums to 1)."""
    Ps = nS_bins(edges, VS, sigma, pT)
    Pt = nT_bins(edges, VS, VT, sigma, pT, T1, tm, nq)
    probs = Ps + Pt
    return np.clip(probs, 1e-15, None)

def expected_counts_barthel(edges, tm, VS, VT, sigma, pT, T1, Ntot, nq=64):
    """Expected per-bin counts from total probabilities."""
    return float(Ntot) * barthel_probs_per_bin(edges, tm, VS, VT, sigma, pT, T1, nq=nq)


# ---------------- CDF / visibility conditional Eq. 3 in paper https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.160503----------------
def fidelities_eq3_from_bins_conditional(edges, VS, VT, sigma, pT, T1, tm, nq=64):
    """
    Conditional CDFs evaluated at each edge. Returns FS, FT, visibility, Vopt.
    """
    eps = 1e-15
    Ps_u = nS_bins(edges, VS, sigma, pT)                  # sums to (1 - pT)
    Pt_u = nT_bins(edges, VS, VT, sigma, pT, T1, tm, nq)  # sums to pT

    Ps = Ps_u / max(eps, (1.0 - pT))
    Pt = Pt_u / max(eps, pT)

    Ps_left = np.concatenate([[0.0], np.cumsum(Ps)])  # length nbins+1
    Pt_left = np.concatenate([[0.0], np.cumsum(Pt)])

    FS_edges = Ps_left
    FT_edges = 1.0 - Pt_left
    vis_edges = FS_edges + FT_edges - 1.0

    iopt = int(np.argmax(vis_edges))
    return {
        "Vthr_edges": np.asarray(edges, float),
        "FS_edges": FS_edges,
        "FT_edges": FT_edges,
        "visibility_edges": vis_edges,
        "V_opt": edges[iopt],
        "F_S_opt": FS_edges[iopt],
        "F_T_opt": FT_edges[iopt],
        "Visibility": FS_edges[iopt] + FT_edges[iopt] - 1.0,
    }


# ---------------- Fitting  ----------------
def _auto_init_from_data(counts, edges):
    centers = _edges_to_centers(edges)
    top2 = np.argpartition(counts, -2)[-2:]
    i1, i2 = top2[np.argsort(counts[top2])[::-1]]
    VS0, VT0 = np.sort([centers[i1], centers[i2]])
    span = edges[-1] - edges[0]
    sigma0 = max(1e-6, 0.05 * span)
    pT0 = 0.5
    return VS0, VT0, sigma0, pT0

def fit_barthel_one(counts, edges, tm, T1, Ntot, init=None, nq=64, restarts=4, jitter=0.25, seed=12345):
    """
    Fit the histogram of single-shot readout voltages using the Barthel model
    (Poisson maximum likelihood estimation).

    Parameters
    ----------
    counts : array_like
        Histogram bin counts (observed data).
    edges : array_like
        Bin edges corresponding to the histogram.
    tm : float
        Integration time of the measurement (ns).
    T1 : float
        Relaxation time constant (ns).
    Ntot : float
        Total number of measurement shots.
    init : dict, optional
        User-supplied initial guesses for 'VS', 'VT', 'sigma', 'pT'.
    nq : int, optional
        Number of quadrature points used for numerical integration in
        expected_counts_barthel().
    restarts : int, optional
        Number of random restarts for the optimizer (for robustness).
    jitter : float, optional
        Amplitude of random perturbations for restart initializations.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing optimized parameters:
        { 'VS', 'VT', 'sigma', 'pT', 'success', 'fun', 'nit', 'message' }
    """

    # Ensure inputs are arrays
    counts = np.asarray(counts, float)
    edges  = np.asarray(edges,  float)
    assert np.all(np.diff(edges) > 0), "edges must be strictly increasing"
    span = edges[-1] - edges[0]   # total voltage range

    # --- Step 1: Auto-generate initial guesses from histogram data ---
    VS0, VT0, sigma0, pT0 = _auto_init_from_data(counts, edges)

    # --- Step 2: Apply user overrides if provided ---
    if init is not None:
        if 'VS' in init: VS0 = float(init['VS'])
        if 'VT' in init: VT0 = float(init['VT'])
        # ensure singlet peak is left of triplet peak
        if VS0 > VT0: VS0, VT0 = VT0, VS0
        if 'sigma' in init:
            sigma0 = float(init['sigma'])
        else:
            # default σ ≈ 0.25 * peak separation (clipped to reasonable range)
            gap = max(1e-9, VT0 - VS0)
            sigma0 = float(np.clip(0.25 * gap, 1e-6, 0.5 * span))
        if 'pT' in init: pT0 = float(init['pT'])

    # --- Step 3: Define helper to clip parameters within valid bounds ---
    def _clip_start(VS, VT, sigma, pT):
        # limit peak voltages within measured voltage range
        VS = float(np.clip(VS, edges[0], edges[-1]))
        VT = float(np.clip(VT, edges[0], edges[-1]))
        # ensure order (VS < VT); if not, slightly separate them
        if VS >= VT:
            mid = 0.5 * (VS + VT)
            VS = mid - 1e-3 * span
            VT = mid + 1e-3 * span
        # sigma positive but not too large
        sigma = float(np.clip(sigma, 1e-6, span))
        # triplet probability strictly between 0 and 1
        pT = float(np.clip(pT, 1e-6, 1 - 1e-6))
        return np.array([VS, VT, sigma, pT], float)

    # Initial parameter vector and optimization bounds
    base_theta0 = _clip_start(VS0, VT0, sigma0, pT0)
    bounds = [
        (edges[0], edges[-1]),  # VS bounds
        (edges[0], edges[-1]),  # VT bounds
        (1e-6, span),           # sigma bounds
        (1e-6, 1 - 1e-6)        # pT bounds
    ]

    # --- Step 4: Define negative log-likelihood function ---
    def _nll(theta):
        VS, VT, sigma, pT = theta
        # Expected Poisson rate (model prediction)
        lam = expected_counts_barthel(edges, tm, VS, VT, sigma, pT, T1, Ntot, nq=nq)
        lam = np.clip(lam, 1e-15, None)  # numerical stability
        # Poisson log-likelihood (negative sign → minimization)
        return float(np.sum(lam - counts * np.log(lam)))

    # --- Step 5: Run multiple restarts for robust optimization ---
    rng = np.random.default_rng(seed)
    best = None
    for r in range(max(1, int(restarts))):
        if r == 0:
            # first run: use deterministic base initialization
            theta0 = base_theta0
        else:
            # subsequent runs: perturb starting point randomly
            VSj = base_theta0[0] + rng.normal(scale=jitter * 0.2 * (VT0 - VS0 + 1e-12))
            VTj = base_theta0[1] + rng.normal(scale=jitter * 0.2 * (VT0 - VS0 + 1e-12))
            sigj = base_theta0[2] * np.exp(rng.normal(scale=jitter * 0.5))
            pTj = np.clip(base_theta0[3] + rng.normal(scale=jitter * 0.15), 1e-3, 1 - 1e-3)
            theta0 = _clip_start(VSj, VTj, sigj, pTj)

        # Minimize NLL using L-BFGS-B (handles bounds smoothly)
        res = minimize(_nll, theta0, method="L-BFGS-B", bounds=bounds)

        # Keep best (lowest NLL) result
        if (best is None) or (res.fun < best.fun):
            best = res

    # --- Step 6: Extract best-fit parameters ---
    VS, VT, sigma, pT = best.x

    # Return results as a dictionary
    return dict(
        VS=VS,
        VT=VT,
        sigma=sigma,
        pT=pT,
        success=bool(best.success),
        fun=float(best.fun),
        nit=int(best.nit),
        message=str(best.message)
    )



# ---------------- Plotting----------------
def plot_hist_with_cdf_bw(counts, edges, VS, VT, sigma, pT, T1, tm, nq=64, title=None, palette=None):
    """
    Plots bars (data) + total model (S+T) + CDF/visibility (twin y-axis).
    Also returns a dict with:
      - histogram_ground (lam_S)
      - histogram_excited (lam_T)
      - histogram_stack (lam_S + lam_T)
      - common_edges (edges)
    so you can reuse your class's plotter.
    """
    if palette is None:
        palette = dict(
            bar_face='#cfcfcf', bar_edge='#9a9a9a',
            model='#1f1f1f',
            FS='#395B9A', FT='#A1633B', VIS='#3B8A5D',
            Vlines='#5a5a5a'
        )

    counts = np.asarray(counts, float)
    edges  = np.asarray(edges,  float)
    centers = _edges_to_centers(edges)
    Ntot = float(np.sum(counts))

    # model components (unconditional per-bin mass) → expected counts
    Ps_u = nS_bins(edges, VS, sigma, pT)                    # (nbins,)
    Pt_u = nT_bins(edges, VS, VT, sigma, pT, T1, tm, nq)    # (nbins,)
    lam_S = Ntot * Ps_u
    lam_T = Ntot * Pt_u
    lam   = lam_S + lam_T

    # conditional CDFs / visibility
    out_cdf = fidelities_eq3_from_bins_conditional(edges, VS, VT, sigma, pT, T1, tm, nq=nq)
    vthr = out_cdf["Vthr_edges"]
    FS, FT, vis = out_cdf["FS_edges"], out_cdf["FT_edges"], out_cdf["visibility_edges"]
    vopt = out_cdf["V_opt"]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(centers, counts, width=np.diff(edges),
           color=palette['bar_face'], edgecolor=palette['bar_edge'],
           linewidth=0.6, alpha=0.9, label='data')
    ax.plot(centers, lam, color=palette['model'], linewidth=2.3, label='model (S+T)')

    ax.set_xlabel(r'$V_{\rm rf}\ \mathrm{(V)}$')
    ax.set_ylabel('Counts')
    ax.axvline(VS, color=palette['Vlines'], linestyle=(0,(1,2)), linewidth=1.2, label=f'V_S={VS:.3g} V')
    ax.axvline(VT, color=palette['Vlines'], linestyle='-.',       linewidth=1.2, label=f'V_T={VT:.3g} V')
    ax.axvline(vopt, color=palette['VIS'],    linestyle='--',     linewidth=1.4, label=f'V_opt={vopt:.3g} V')
    ax.grid(True, alpha=0.25, color='#e5e5e5')

    ax2 = ax.twinx()
    ax2.step(vthr, FS, where='post', color=palette['FS'], linewidth=2.0, label=r'$F_S$')
    ax2.step(vthr, FT, where='post', color=palette['FT'], linewidth=2.0, linestyle='-.', label=r'$F_T$')
    ax2.step(vthr, vis, where='post', color=palette['VIS'], linewidth=2.2, linestyle='--',
             label=r'$F_S + F_T - 1$')
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('Fidelity / Visibility')

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='lower right', frameon=True)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    # Return dict in the exact structure your class plotter expects
    out = dict(out_cdf)  # copy CDF info
    out.update({
        "histogram_ground": lam_S,
        "histogram_excited": lam_T,
        "histogram_stack": lam,
        "common_edges": edges,
        "lam_total": lam,
        "Ntot": Ntot,
        "VS": VS, "VT": VT, "sigma": sigma, "pT": pT, "T1": T1, "tm": tm
    })
    return out





    
    
            
    
        
            
                