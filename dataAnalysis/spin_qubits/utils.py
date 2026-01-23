import numpy as np
from matplotlib import pyplot as plt
import lmfit

from dataAnalysis.base import DataSet

def charge_sensor_peak_model(Vg, A, V0, Gamma):
    return A * (Gamma/2)**2 / ( (Vg - V0)**2 + (Gamma/2)**2)

class ChargeSensorCalibration(DataSet):
    def __init__(self, exp, run_id):
        super().__init__(exp=exp, run_id=run_id)
        self.xdata = self.independent_parameters['x']['values']
        self.ydata = self.dependent_parameters['param_4']['values']

    def get_init_guesses(self):
        V0 = self.xdata[np.argmax(self.ydata)]
        A = np.max(self.ydata)
        Gamma = (self.xdata[-1] - self.xdata[0])/2

        print(f"Guesses: A={A}, V0={V0}, Gamma={Gamma}")

        return {"A": A, "V0": V0, "Gamma": Gamma}

    def fit_charge_sensor_peak(self):
        guesses_dict = self.get_init_guesses()
        params = lmfit.Parameters()
        params.add("A", value=guesses_dict["A"])
        params.add("V0", value=guesses_dict["V0"])
        params.add("Gamma", value=guesses_dict["Gamma"])

        self.model = lmfit.Model(charge_sensor_peak_model, independent_vars=['Vg'])
        
        fit_result = self.model.fit(self.ydata, params, Vg=self.xdata)
        self.fit_result = fit_result

        
    def calibrate(self, shoulder='left'):
        """
        Calibrate the charge sensor operation point by fitting the Coulomb peak with a Lorentzian and 
        looking at the point with maximum derivative of the reflected amplitude.
        
        Params:
            shoulder: Either 'left' or 'right'

        Returns:
            The gate voltage corresponding to the point with maximum derivative.
        """
        self.fit_charge_sensor_peak()
        self.A = self.fit_result.params['A']
        self.V0 = self.fit_result.params['V0']
        self.Gamma = self.fit_result.params['Gamma']
        if shoulder == "left":
            self.V_max_deriv = self.V0 - self.Gamma/(2*np.sqrt(3))
        elif shoulder == "right":
            self.V_max_deriv = self.V0 + self.Gamma/(2*np.sqrt(3))
        else:
            raise ValueError("Invalid option for 'shoulder' parameter: must be either 'left' or 'right'.")


        plt.figure()
        plt.plot(self.xdata, self.ydata, '.', color='k')
        plt.xlabel("Gate voltage (V)")
        plt.ylabel("Magnitude (V)")

        plt.plot(self.xdata, self.model.eval(params=self.fit_result.params, Vg=self.xdata), '-', color='red')
        plt.scatter(self.V_max_deriv, self.model.eval(params=self.fit_result.params, Vg=self.V_max_deriv), color='red')
        plt.text(self.V0-self.Gamma/6, self.A*2/3, f"V = {self.V_max_deriv:.5f}", color='red')

        return self.V_max_deriv

    