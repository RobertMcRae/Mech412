from cProfile import label
import control
import d2c
import unc_bound
import numpy as np
import matplotlib.pyplot as plt

from system_ID import SystemID

class ModelUncertainty():

    def __init__(self, model_collection: dict[str, SystemID]):
        self.model_collection = model_collection
        self.continuous_models = self.to_continuous_models(model_collection)
        # self.continuous_models.pop("System 2")
        # self.continuous_models.pop("System 3")
        self.nominal_plant = self.continuous_models.pop("System 1,2,3 Combined")

    def to_continuous_models(self, model_collection): 
        #Translates all discrete time models in the model collection to continuous time models
        continuous_models = {}
        for key, value in model_collection.items():
            form = value.u_y_form
            num = np.ravel(value.x_parameters[form[1]:])[::-1] #u values
            den = np.ravel(value.x_parameters[:form[1]])[::-1] #y values
            den = np.insert(den, 0, 1.0)
            
            #validate dt
            if not np.allclose(np.diff(value.T), np.diff(value.T)[0]):
                raise Exception(f"Error with time sampling of system ID {key}")
            sys_discrete = (value.y_norm/value.u_norm)*control.tf(num, den, value.T[1]-value.T[0])
            print(f"discrete tf is: {sys_discrete}")
            sys_continuous = d2c.d2c(sys_discrete)
            print(f"For {key} the system is: {sys_continuous}")
            continuous_models[key] = sys_continuous
        return continuous_models
    
    def plot_residuals(self):
        print(f"Models apart of the plant are: {list(self.continuous_models.keys())}") 
        print(f"Nominal plant is: {self.nominal_plant}")
        R = unc_bound.residuals(self.nominal_plant, list(self.continuous_models.values()))

        # Bode plot
        N_w = 500
        w_shared = np.logspace(-1, 3, N_w)

        # Compute magnitude part of R(s) in both dB and in absolute units
        mag_max_dB, mag_max_abs = unc_bound.residual_max_mag(R, w_shared)

        #Hand-tuned Uncertainty bound
        nW2 = 3
        W2 = (unc_bound.upperbound(omega=w_shared, upper_bound=mag_max_abs, degree=nW2)).minreal()
        print(f"The W2(s) found is: {W2}")
        mag_W2_abs, _, _ = control.frequency_response(W2, w_shared)
        mag_W2_dB = 20 * np.log10(mag_W2_abs)
        frequency = w_shared/(2*np.pi)  # Convert rad/s to Hz for plotting

        # Plot Bode magnitude plot in dB and in absolute units
        fig, ax = plt.subplots(2, 1)
        ax[0].set_xlabel(r'Hz (1/s)')
        ax[0].set_ylabel(r'Magnitude (dB)')
        ax[1].set_xlabel(r'Hz (1/s)')
        ax[1].set_ylabel(r'Magnitude (absolute)')
        ax[0].semilogx(frequency, mag_W2_dB, '-', color='seagreen', linewidth=1, label='Optimal W2')
        ax[1].semilogx(frequency, mag_W2_abs, '-', color='seagreen', linewidth=1, label='Optimal W2')
        for i in range(len(R)):
            mag_abs, _, _ = control.frequency_response(R[i], w_shared)
            mag_dB = 20 * np.log10(mag_abs)
            # Magnitude plot (dB)
            ax[0].semilogx(frequency, mag_dB, '--', color='C0', linewidth=1)
            # Magnitude plot (absolute).
            ax[1].semilogx(frequency, mag_abs, '--', color='C0', linewidth=1)

        # Magnitude plot (dB).
        ax[0].semilogx(frequency, mag_max_dB, '-', color='C4', label='upper bound')
        # Magnitude plot (absolute).
        ax[1].semilogx(frequency, mag_max_abs, '-', color='C4', label='upper bound')
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')
        fig.tight_layout()
        return fig
    
    def plot_nominal_plant(self):
        hz = np.logspace(-2, 2, 500)  # from 10^-2 to 10^2 Hz
        omega = 2 * np.pi * hz  # Convert to rad/s
        fig, ax = plt.subplots()
        for key, value in self.continuous_models.items():
            control.bode_plot(value, omega=omega, dB=True, Hz=True, label=key)
            fig.tight_layout()

        cptl = control.bode_plot(self.nominal_plant, omega=omega, dB=True, Hz=True, label='Nominal Plant')
        return cptl.figure