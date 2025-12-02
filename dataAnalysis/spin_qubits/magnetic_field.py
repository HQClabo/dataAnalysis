import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import lmfit
from dataAnalysis.base import DataSet, ConcatenatedDataSet, val_to_index


#####################################################################################################
# CLASSES
#####################################################################################################
class BFieldInPlaneAngleSweep(ConcatenatedDataSet, DataSet):
    def __init__(self, exp, run_id:int|list, B_in_mag, B_out=0, angle_values=None, angle_label="In_plane_angle", angle_unit="Deg"):
        """
        Args:
            exp: Experiment
            run_id: [start_id, stop_id]
            B_in_mag: Magnitude of the in-plane component of the B field.
            B_out (optional, default 0): Out-of-plane component of the B field.
            angle_values(optional, default None): Swept values of the in-plane angle.
            angle_label: Label to use for the angle variable in plots.
            angle_unit: Unit of the angles.

        """
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
        self.B_in_mag = B_in_mag
        self.B_out = B_out

    def normalize_data_from_average(self, params_to_normalize = None, axis=0, operation="subtract"):
        super().normalize_data_from_average(params_to_normalize, axis, operation)
        self.mag_norm = self.dependent_parameters['param_0_normalized']['values']
    
    def find_resonances(self, follow_resonances=False, search_center=None, search_span=None, min_separation=1):
        if follow_resonances:
            assert search_center is not None and search_span is not None, "If follow_resonances is True, search_center and search_span must be provided."
            
            # Convert range into indices
            low_freq = search_center - search_span/2
            high_freq = search_center + search_span/2
            low_idx = val_to_index(self.freq, low_freq)
            high_idx = val_to_index(self.freq, high_freq)
        else:
            low_idx = 0
            high_idx = -1

        if hasattr(self, 'mag_norm'):
                data = self.mag_norm
        else:
            data = self.mag

        num_angles = len(self.angle)
        idx1_array = np.ones(num_angles, dtype=int)
        idx2_array = np.ones(num_angles, dtype=int)
        f1_array = np.array([])
        f2_array = np.array([])

        for angle_index, angle in enumerate(self.angle):
            # Take the linecut for the current angle value
            col = data[:, angle_index]
            working_linecut = col.copy()
            idx1, idx2 = self.top_two_in_range(working_linecut, start=low_idx, end=high_idx, min_sep=min_separation)
            f1 = self.freq[idx1]
            f2 = self.freq[idx2]

            # Append results
            idx1_array[angle_index] = idx1 if f1 < f2 else idx2
            idx2_array[angle_index] = idx2 if f1 < f2 else idx1
            f1_array = np.append(f1_array, min(f1, f2)) # f1 is always the lowest one
            f2_array = np.append(f2_array, max(f1, f2))

            # Adjust search range for next iteration
            if follow_resonances:
                search_center = (f1 + f2)/2
                low_freq = search_center - search_span/2
                high_freq = search_center + search_span/2
                low_idx = val_to_index(self.freq, low_freq)
                high_idx = val_to_index(self.freq, high_freq)

        
        self.results = {"idx1": idx1_array, "idx2": idx2_array, "f1": f1_array, "f2": f2_array}

    def top_two_in_range(self, arr, start=0, end=None, min_sep=1):
        """
        Find the two highest values and their indices within arr[start:end],
        with a minimum separation between the two indices.

        Parameters:
        - arr: sequence of numbers
        - start: inclusive start index (default 0)
        - end: exclusive end index (default len(arr))
        - min_sep: minimum required separation between indices (>= 1)

        Returns:
        - ((value1, idx1), (value2, idx2)) where idx1 and idx2 are the indices in arr

        Raises:
        - ValueError if the range is too small or no valid pair exists under the
        separation constraint.
        """
        if end is None or end > len(arr) or end == -1:
            end = len(arr)

        if start < 0: 
            start = 0

        if not (0 <= start < end <= len(arr)):
            raise ValueError(f"Invalid start/end range: start={start}, end={end}.")
        if min_sep < 1:
            raise ValueError("min_sep must be at least 1.")
        
        L = end - start
        # Need at least two positions with separation min_sep => L >= min_sep + 1
        if L < min_sep + 1:
            raise ValueError("Range too small for the requested separation.")

        indices = list(range(start, end))
        # Sort indices by value desc; for ties, smaller index first (deterministic)
        sorted_by_value = sorted(indices, key=lambda i: (arr[i], -i), reverse=True)

        # i will be the index of the first max, j of the second max
        # Find best index j with |j - i| >= min_sep
        i = sorted_by_value[0]
        best_j = None
        best_j_val = None
        for j in indices:
            if abs(j - i) >= min_sep:
                if best_j_val is None or arr[j] > best_j_val or (arr[j] == best_j_val and j < best_j):
                    best_j_val = arr[j]
                    best_j = j
        if best_j is not None:
            return i, best_j

        # If we get here, no valid pair exists under the constraint
        raise ValueError("No two indices satisfy the separation constraint within the given range.")

    def extract_g_factors(self, B_out=0, **kwargs):
        if not hasattr(self, "results"):
            self.find_resonances(**kwargs)
        self.B_out = B_out
        B_tot = np.sqrt(self.B_in_mag**2 + B_out**2)
        g1 = compute_g_factor(self.results['f1'], B_tot)
        g2 = compute_g_factor(self.results['f2'], B_tot)
        self.results['g1'] = g1
        self.results['g2'] = g2

        return self.results
    
    def plot_g_factors(self, angle_labels=None, fig=None, ax=None):
        """
        Plot the calculated g factor in the plane.

        Args:
            angle_label: If None, labels are in degrees. If 'xy', 'yz' or 'xz', then the in-plane axes labels are used.

        Returns: fig, ax, (plot1, plot2)
        """
        g1 = self.results['g1']
        g2 = self.results['g2']
        angle = self.angle

        # If figure handlers not provided, create them
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='polar')
        elif ax is None:   
            ax = fig.add_subplot(projection='polar')

        _, _, plot1 = plot_polar(np.deg2rad(angle), g1, style='data', fig=fig, ax=ax, label='Q1', lw=1)
        _, _, plot2 = plot_polar(np.deg2rad(angle), g2, style='data', fig=fig, ax=ax, label='Q2', lw=1)

        if angle_labels == 'xy':
            ax.set_xticklabels(['+x', '', '+y', '', '-x', '', '-y', ''])
        elif angle_labels == 'yz':
            ax.set_xticklabels(['+y', '', '+z', '', '-y', '', '-z', ''])
        elif angle_labels == 'xz':
            ax.set_xticklabels(['+x', '', '+z', '', '-x', '', '-z', ''])

        return fig, ax, (plot1, plot2)
    

class GTensorCharacterization:
    def __init__(self):
        self.xy_measurements = []
        self.yz_measurements = []
        self.xz_measurements = []
    
    def add_measurement(self, measurement: BFieldInPlaneAngleSweep, type: str):
        """
        Add an in-plane angle sweep measurements to the class, to be considered for the g-factor fitting.

        Parameters:
        - measurement: An instance of BFieldInPlaneAngleSweep
        - type: "xy", "yz" or "xz"
        """
        if type.lower() == "xy":
            self.xy_measurements.append(measurement)
        elif type.lower() == "yz":
            self.yz_measurements.append(measurement)
        elif type.lower() == "xz":
            self.xz_measurements.append(measurement)
        else:
            raise ValueError("'type' must be either 'xy', 'yz' or 'xz'.")
    
    def fit_g_tensor(self, qubit:int=1, method='leastsq'):
        """
        Fit the g tensor using the provided in-plane sweep measurements. Such measurements must be added to the class before running this method using the method 'add_measurement'.

        Parameters:
        - qubit: integer number indicating which of the two qubits to fit.
        - do_plots: Option to plot the data and the fit.
        """
        self.qubit = qubit

        # Check that there are measurements loaded
        self.xy_is_present = False
        self.yz_is_present = False
        self.xz_is_present = False
        if self.xy_measurements != []:
            self.xy_is_present = True
        if self.yz_measurements != []:
            self.yz_is_present = True
        if self.xz_measurements != []:
            self.xz_is_present = True
        
        if self.xy_is_present + self.yz_is_present + self.xz_is_present == 0:
            print("No measurement has been loaded yet. Before running the fit, add them with the method 'add_measurement'. ")
            return
        
        #----------------------------- Build data -----------------------------
        Bx_array = []
        By_array = []
        Bz_array = []
        g_factor_array = []

        # xy sweeps
        for dataset in self.xy_measurements:
            Bz = dataset.B_out
            B_in_mag = dataset.B_in_mag
            for angle, g_factor in zip(dataset.angle, dataset.results[f'g{qubit}']):
                Bx, By, _ = build_B_vector_lab_frame(B_in_mag, theta=90, phi=angle) # The order must be consistent with the measurements
                Bx_array.append(Bx)
                By_array.append(By)
                Bz_array.append(Bz)
                g_factor_array.append(g_factor)
        # xy sweeps
        for dataset in self.yz_measurements:
            Bx = dataset.B_out
            B_in_mag = dataset.B_in_mag
            for angle, g_factor in zip(dataset.angle, dataset.results[f'g{qubit}']):
                By, Bz, _ = build_B_vector_lab_frame(B_in_mag, theta=90, phi=angle)  # The order must be consistent with the measurements
                Bx_array.append(Bx)
                By_array.append(By)
                Bz_array.append(Bz)
                g_factor_array.append(g_factor)
        # yz sweeps
        for dataset in self.xz_measurements:
            By = dataset.B_out
            B_in_mag = dataset.B_in_mag
            for angle, g_factor in zip(dataset.angle, dataset.results[f'g{qubit}']):
                Bx, Bz, _ = build_B_vector_lab_frame(B_in_mag, theta=90, phi=angle) # The order must be consistent with the measurements
                Bx_array.append(Bx)
                By_array.append(By)
                Bz_array.append(Bz)
                g_factor_array.append(g_factor)

        Bx_array = np.array(Bx_array)
        By_array = np.array(By_array)
        Bz_array = np.array(Bz_array)

        #----------------------------- Fit -----------------------------
        fit_result, model = _fit_g_factors(Bx_array, By_array, Bz_array, g_factor_array, method=method)
        self.fit_result = fit_result
        self.model = model
        

        return fit_result
    
    def plot_fit_result(self, fig, axes):
        best_fit = self.fit_result.best_fit

        num_cols = self.xy_is_present + self.yz_is_present + self.xz_is_present
        num_rows = max(len(self.xy_measurements), len(self.yz_measurements), len(self.xz_measurements))

        cm = 1/2.54
        fig_width = 9*cm*num_cols
        fig_height = 7*cm*num_rows
        fig, axes = plt.subplots(num_rows, num_cols, subplot_kw = {'projection' : 'polar'})
        fig.set_size_inches(fig_width, fig_height)
        fig.set_dpi(100)
        

        # Initialize counters
        curr_col = -1   # column counter (each column for each different plane)
        data_start_index = 0   # where the current data to be plotted starts from

        if self.xy_measurements != []:
            curr_col += 1
            curr_row = -1   # row counter
            for dataset in self.xy_measurements:
                curr_row += 1

                # Get the axes (bunch of different cases depending on num_rows and num_cols)
                if num_rows > 1 and num_cols > 1:
                    ax = axes[curr_row][curr_col]
                elif num_rows == 1 and num_cols > 1:
                    ax = axes[curr_col]
                elif num_rows > 1 and num_cols == 1:
                    ax = axes[curr_row]
                elif num_rows == 1 and num_cols == 1:
                    ax = axes

                # Get the values to be plotted
                angles_rad = np.deg2rad(dataset.angle)
                g_exp = dataset.results[f'g{self.qubit}']
                num_points = len(angles_rad)
                g_fitted = best_fit[data_start_index:data_start_index + num_points]

                # Plot 
                plot_polar(angles_rad, g_exp, style='data', fig=fig, ax=ax, label='exp', lw=1, plot_legend=False)
                plot_polar(angles_rad, g_fitted, style='line', fig=fig, ax=ax, label='fit', color='k', plot_legend=False)
                ax.set_xticklabels(['+x', '', '+y', '', '-x', '', '-y', ''])

                legend = ax.legend(bbox_to_anchor=(0,0))
                legend.set_frame_on(False)

                # Adjust start index for next plot
                data_start_index += num_points

        if self.yz_measurements != []:
            curr_col += 1
            curr_row = -1   # row counter
            for dataset in self.yz_measurements:
                curr_row += 1

                # Get the axes (bunch of different cases depending on num_rows and num_cols)
                if num_rows > 1 and num_cols > 1:
                    ax = axes[curr_row][curr_col]
                elif num_rows == 1 and num_cols > 1:
                    ax = axes[curr_col]
                elif num_rows > 1 and num_cols == 1:
                    ax = axes[curr_row]
                elif num_rows == 1 and num_cols == 1:
                    ax = axes

                # Get the values to be plotted
                angles_rad = np.deg2rad(dataset.angle)
                g_exp = dataset.results[f'g{self.qubit}']
                num_points = len(angles_rad)
                g_fitted = best_fit[data_start_index:data_start_index + num_points]

                # Plot 
                plot_polar(angles_rad, g_exp, style='data', fig=fig, ax=ax, label='exp', lw=1, plot_legend=False)
                plot_polar(angles_rad, g_fitted, style='line', fig=fig, ax=ax, label='fit', color='k', plot_legend=False)
                ax.set_xticklabels(['+y', '', '+z', '', '-y', '', '-z', ''])
                legend = ax.legend(bbox_to_anchor=(0,0))
                legend.set_frame_on(False)

                # Adjust start index for next plot
                data_start_index += num_points

        if self.xz_measurements != []:
            curr_col += 1
            curr_row = -1   # row counter
            for dataset in self.xz_measurements:
                curr_row += 1

                # Get the axes (bunch of different cases depending on num_rows and num_cols)
                if num_rows > 1 and num_cols > 1:
                    ax = axes[curr_row][curr_col]
                elif num_rows == 1 and num_cols > 1:
                    ax = axes[curr_col]
                elif num_rows > 1 and num_cols == 1:
                    ax = axes[curr_row]
                elif num_rows == 1 and num_cols == 1:
                    ax = axes

                # Get the values to be plotted
                angles_rad = np.deg2rad(dataset.angle)
                g_exp = dataset.results[f'g{self.qubit}']
                num_points = len(angles_rad)
                g_fitted = best_fit[data_start_index:data_start_index + num_points]

                # Plot 
                plot_polar(angles_rad, g_exp, style='data', fig=fig, ax=ax, label='exp', lw=1, plot_legend=False)
                plot_polar(angles_rad, g_fitted, style='line', fig=fig, ax=ax, label='fit', color='k', plot_legend=False)
                ax.set_xticklabels(['+x', '', '+z', '', '-x', '', '-z', ''])
                legend = ax.legend(bbox_to_anchor=(0,0))
                legend.set_frame_on(False)

                # Adjust start index for next plot
                data_start_index += num_points
        plt.show()    
        self.fig = fig
        self.axes = axes
    


#####################################################################################################
# FUNCTIONS
#####################################################################################################
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

def build_B_vector_lab_frame(B_mag, theta, phi):
    """
    Build the 3-component vector of the magnetic field from the angles the lab frame.

    Args:
        B_mag: Magnitude of the magnetic field.
        theta: Azimuthal angle of the field (expressed in the lab frame)
        phi: In-plane angle of the field (expressed in the lab frame).

    Returns:
        The x, y and z components of the magnetic field in the lab frame. The convention used is that x and y are the in-plane axes, 
        and z is the out-of-plane axis.
    """
    # Convert degrees to radians
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    
    # Compute components
    B_x = B_mag * np.sin(theta_rad) * np.cos(phi_rad)
    B_y = B_mag * np.sin(theta_rad) * np.sin(phi_rad)
    B_z = B_mag * np.cos(theta_rad) * np.ones_like(phi_rad)

    return B_x, B_y, B_z

def g_frame_to_lab_frame_rotation(phi, theta, zeta):
    """
    Build the matrix to convert a vector from the g frame to the lab frame.

    Args:
        phi: In-plane relative rotation of the two frames.
        theta: Azimuthal relative rotation of the two frames.
        zeta: Zeta relative rotation of the two frames.

    Returns:
        The matrix to convert from the g frame to the lab frame.
    """
    
    # Convert degrees to radians
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    zeta = np.deg2rad(zeta)
    
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_zeta = np.cos(zeta)
    s_zeta = np.sin(zeta)

    R = np.array([
        [c_phi * c_theta * c_zeta - s_phi * s_zeta,
         -c_phi * c_theta * s_zeta - s_phi * c_zeta,
         c_phi * s_theta],
        
        [s_phi * c_theta * c_zeta + c_phi * s_zeta,
         -s_phi * c_theta * s_zeta + c_phi * c_zeta,
         s_phi * s_theta],
        
        [-s_theta * c_zeta,
         s_theta * s_zeta,
         c_theta]
    ])

    return R

def g_tensor_g_frame_to_lab_frame(gx, gy, gz, phi, theta, zeta):
    """
    Build the g tensor in the lab frame starting from the g vector in the g frame.

    Args:
        gx: x component of the g vector in the g frame.
        gy: y component of the g vector in the g frame.
        gz: z component of the g vector in the g frame.
        phi: In-plane angle between the g frame and the lab frame.
        theta: Azimuthal angle between the g frame and the lab frame.
        zeta: Zeta angle between the g frame and the lab frame.
    """
    g = np.diag([gx, gy, gz])
    R = g_frame_to_lab_frame_rotation(phi, theta, zeta)
    return R @ g @ np.linalg.inv(R)

def model_g_factor_lab_frame(Bx_lab, By_lab, Bz_lab, gx, gy, gz, phi, theta, zeta):
    """
    Model for the g factor in the lab frame, given a certain applied magnetic field.
    """
    g_factor_array = []
    for Bx, By, Bz in zip(Bx_lab, By_lab, Bz_lab):
        B_vector = np.vstack((Bx, By, Bz))
        g_matrix_lab_frame = g_tensor_g_frame_to_lab_frame(gx, gy, gz, phi, theta, zeta)
        gB_product = np.dot(g_matrix_lab_frame, B_vector)
        B_norm = np.linalg.norm(B_vector)
        gB_norm = np.linalg.norm(gB_product)
        g_factor_array.append(gB_norm/B_norm)
    g_factor_array = np.array(g_factor_array)

    return g_factor_array


def _fit_g_factors(Bx_lab, By_lab, Bz_lab, g_factor_lab, guesses_dict=None, limits_dict=None, method='leastsq'):
    params = lmfit.Parameters()

    default_guesses = {
        'gx': 0.6,
        'gy': 0.35,
        'gz': 11.0,
        'phi': 0,
        'theta': 0,
        'zeta': 0,
    }
    default_limits = {
        'gx': (0, 1),
        'gy': (0, 1),
        'gz': (5, 30),
        'phi': (0, 1),
        'theta': (0, 1),
        'zeta': (0, 1),
    }

    params.add('gx', 0.06, min=0, max=1, vary=True)
    params.add('gy', 0.35, min=0, max=1, vary=True)
    params.add('gz', 11, min=5, max=30, vary=True)
    params.add('phi', 0, vary=True, min=-180, max=180)
    params.add('theta', 0, vary=True, min=0, max=180)
    params.add('zeta', 0, vary=True, min=-180, max=180)

    model = lmfit.Model(model_g_factor_lab_frame, independent_vars=["Bx_lab", "By_lab", "Bz_lab"])
    fit_result = model.fit(g_factor_lab, params, Bx_lab=Bx_lab, By_lab=By_lab, Bz_lab=Bz_lab, method=method)
    return fit_result, model

def plot_polar(angles_rad, r, style='data', fig=None, ax=None, label=None, plot_legend=True, **kwargs):
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
    if style == 'data':
        plot = ax.scatter(angles_rad,  r, label=label, alpha=0.7, **kwargs)
    elif style == 'line':
        plot = ax.plot(angles_rad, r, label=label, **kwargs)
    if plot_legend:
        legend = ax.legend(bbox_to_anchor=(1, 1.1))
        legend.set_frame_on(False)

    return fig, ax, plot