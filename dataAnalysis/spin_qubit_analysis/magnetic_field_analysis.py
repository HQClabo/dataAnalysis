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
    
    def plot_g_factors(self, angle_labels=None):
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
    B_z = B_mag * np.cos(theta_rad)

    # Clean small values (e.g., below 1e-5) to zero
    B_x = np.where(np.abs(B_x) < 1e-6, 0, B_x)
    B_y = np.where(np.abs(B_y) < 1e-6, 0, B_y)
    B_z = np.where(np.abs(B_z) < 1e-6, 0, B_z)

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
    B_vector = np.stack((Bx_lab, By_lab, Bz_lab), axis = 1)
    g_matrix_lab_frame = g_tensor_g_frame_to_lab_frame(gx, gy, gz, phi, theta, zeta)
    gB_product = np.dot(g_matrix_lab_frame, B_vector.T).T
    B_norm = np.linalg.norm(B_vector)
    gB_norm = np.linalg.norm(gB_product)

    return gB_norm/B_norm


def _fit_g_factors(Bx_lab, By_lab, Bz_lab, g_factor_lab):
    params = lmfit.Parameters()
    params.add('gx', 0.06, vary=True)
    params.add('gy', 0.35, vary=True)
    params.add('gz', 11, vary=True)
    params.add('phi', 0, vary=True)
    params.add('theta', 0, vary=True)
    params.add('zeta', 0, vary=True)

    model = lmfit.Model(model_g_factor_lab_frame, independent_vars=[Bx_lab, By_lab, Bz_lab])
    fit_result = model.fit(g_factor_lab, params, Bx_lab=Bx_lab, By_lab=By_lab, Bz_lab=Bz_lab)
    return fit_result, model

def fit_g_factors_from_in_plane_sweeps(XY_sweep:BFieldInPlaneAngleSweep=None, YZ_sweep:BFieldInPlaneAngleSweep=None, XZ_sweep:BFieldInPlaneAngleSweep=None, qubit=0):
    if XY_sweep is None and YZ_sweep is None and XZ_sweep is None:
        raise ValueError("At least one of XY_sweep, YZ_sweep and XZ_sweep must be not None.")
    
    Bx_array = []
    By_array = []
    Bz_array = []
    g_factor_array = []

    # XY sweep
    if XY_sweep is not None:
        Bz = dataset.B_out
        dataset = XY_sweep
        B_in_mag = dataset.B_in_mag
        for angle, g_factor in zip(dataset.angle, dataset.results[f'g{qubit}']):
            Bx, By, _ = build_B_vector_lab_frame(B_in_mag, theta=0, phi=angle) # The order must be consistent with the measurements
            Bx_array.append(Bx)
            By_array.append(By)
            Bz_array.append(Bz)
            g_factor_array.append(g_factor)
    # YZ sweep
    if YZ_sweep is not None:
        Bx = dataset.B_out
        dataset = YZ_sweep
        B_in_mag = dataset.B_in_mag
        for angle, g_factor in zip(dataset.angle, dataset.results[f'g{qubit}']):
            By, Bz, _ = build_B_vector_lab_frame(B_in_mag, theta=0, phi=angle)  # The order must be consistent with the measurements
            Bx_array.append(Bx)
            By_array.append(By)
            Bz_array.append(Bz)
            g_factor_array.append(g_factor)
    # YZ sweep
    if XZ_sweep is not None:
        By = dataset.B_out
        dataset = XZ_sweep
        B_in_mag = dataset.B_in_mag
        for angle, g_factor in zip(dataset.angle, dataset.results[f'g{qubit}']):
            Bx, Bz, _ = build_B_vector_lab_frame(B_in_mag, theta=0, phi=angle) # The order must be consistent with the measurements
            Bx_array.append(Bx)
            By_array.append(By)
            Bz_array.append(Bz)
            g_factor_array.append(g_factor)
    
    Bx_array = np.array(Bx_array)
    By_array = np.array(By_array)
    Bz_array = np.array(Bz_array)

    fit_result, model = _fit_g_factors(Bx_array, By_array, Bz_array, g_factor_array)

    print(fit_result.params)

    return fit_result