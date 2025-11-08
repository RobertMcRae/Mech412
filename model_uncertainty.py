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
        self.nominal_plant = self.continuous_models.pop("System 1,2,3 Combined")

    def to_continuous_models(self, model_collection): 
        #Translates all discrete time models in the model collection to continuous time models
        continuous_models = {}
        for key, value in model_collection.items():
            form = value.u_y_form
            num = np.ravel(value.x_parameters[form[1]:]) #u values
            den = np.ravel(value.x_parameters[:form[1]]) #y values

            #validate dt
            if not np.allclose(np.diff(value.T), np.diff(value.T)[0]):
                raise Exception(f"Error with time sampling of system ID {key}")
            sys_discrete = (value.y_norm/value.u_norm)*control.tf(num, den, value.T[1]-value.T[0])
            sys_continuous = d2c.d2c(sys_discrete)
            continuous_models[key] = sys_continuous
        return continuous_models
    
    def plot_residuals(self): 
        R = unc_bound.residuals(self.nominal_plant, list(self.continuous_models.values()))

        # Bode plot
        N_w = 500
        w_shared = np.logspace(-1, 3, N_w)

        # Compute magnitude part of R(s) in both dB and in absolute units
        mag_max_dB, mag_max_abs = unc_bound.residual_max_mag(R, w_shared)

        #Hand-tuned Uncertainty bound
        # s = control.tf("s")
        # w = 1.5
        # k = 2
        # e2 = 0.055
        # m2 = 0.43
        # W2 = ((s+w*((e2)**(1/k)))/(s/((m2)**(1/k))+w))**k
        # mag, _, _ = control.frequency_response(W2, w_shared)
        # mag_W2_dB = 20 * np.log10(mag)

        # Plot Bode magnitude plot in dB and in absolute units
        fig, ax = plt.subplots(2, 1)
        ax[0].set_xlabel(r'$\omega$ (rad/s)')
        ax[0].set_ylabel(r'Magnitude (dB)')
        ax[1].set_xlabel(r'$\omega$ (rad/s)')
        ax[1].set_ylabel(r'Magnitude (absolute)')
        # ax[0].semilogx(w_shared, mag_W2_dB, '-', color='C2', linewidth=1)
        # ax[1].semilogx(w_shared, mag, '-', color='C2', linewidth=1)
        for i in range(len(R)):
            mag_abs, _, _ = control.frequency_response(R[i], w_shared)
            mag_dB = 20 * np.log10(mag_abs)
            # Magnitude plot (dB)
            ax[0].semilogx(w_shared, mag_dB, '--', color='C0', linewidth=1)
            # Magnitude plot (absolute).
            ax[1].semilogx(w_shared, mag_abs, '--', color='C0', linewidth=1)

        # Magnitude plot (dB).
        ax[0].semilogx(w_shared, mag_max_dB, '-', color='C4', label='upper bound')
        # Magnitude plot (absolute).
        ax[1].semilogx(w_shared, mag_max_abs, '-', color='C4', label='upper bound')
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')
        return fig
        