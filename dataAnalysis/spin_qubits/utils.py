import numpy as np
from matplotlib import pyplot as plt
import lmfit

from dataAnalysis.dataset import DataSet

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
            print(f"Filtering CS data with a moving average of window size {window_size}.")
            # Apply a moving average filter to the ydata. Need to cut initial and final points to avoid edge effects
            self.ydata = np.convolve(self.ydata, np.ones(window_size)/window_size, mode='same')[window_size//2:-(window_size//2)]
            self.xdata = self.xdata[window_size//2:-(window_size//2)]

        # Evaluate derivative
        self.y_derivative = np.gradient(self.ydata)

        # if filter:
        #     # Apply a moving average filter to the derivative
        #     self.y_derivative = np.convolve(self.y_derivative, np.ones(window_size)/window_size, mode='same')

        if shoulder == "left":
            self.V_max_deriv = self.xdata[np.argmax(self.y_derivative)]
        elif shoulder == "right":
            self.V_max_deriv = self.xdata[np.argmin(self.y_derivative)]
        else:
            raise ValueError("Invalid option for 'shoulder' parameter: must be either 'left' or 'right'.")


        plt.figure()
        plt.plot(self.xdata, self.ydata, '.', color='k', label='Data')
        plt.xlabel("Gate voltage (V)")
        plt.ylabel(f"{self.ydata_param_name} ({self.get_dependent_parameter_by_name(self.ydata_param_name)['paramspec'].unit})")
        plt.axvline(x = self.V_max_deriv, ls="-", color = 'red')


        plt.figure()
        plt.plot(self.xdata, self.y_derivative, '.', color='red', label='Derivative')
        plt.xlabel("Gate voltage (V)")
        plt.ylabel(f"{self.ydata_param_name} derivative ({self.get_dependent_parameter_by_name(self.ydata_param_name)['paramspec'].unit})")
        plt.axvline(x = self.V_max_deriv, ls="-", color = 'red')

        return self.V_max_deriv

    