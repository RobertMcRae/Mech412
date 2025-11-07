import control
import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from typing import Tuple, Optional

from pyparsing import line

class SystemID():
    """
    A class used for performing system identification on Input-Output data. Initialize the class by providing the inputs, outputs and time stamps
    for each data point. Optionally, the user can provide the desired form of the A matrix in the Ax=b problem.
    """

    _system_id_catalog = {} #used to store all instances of system IDs created
    _system_id_plots = {} #used to store all plot related info for each systemID
    _best_model : Optional["SystemID"] = None #Best IDed model based on NMSE test

    def __init__(self, name: str, u: NDArray[np.float64], y: NDArray[np.float64], T: NDArray[np.float64], u_y_form: Optional[Tuple[int, int]] = None, **kwargs):
        if len(u) != len(y): 
            raise ValueError(f"Input u does not have the same length as ouput y. u length: {len(u)}, y length: {len(y)}")
        self.name = name
        self.u = u
        self.y = y
        self.T = T[2:] #Adjust Time vector to match the Ax=b matrix for plotting later on?
        self.N = len(u)
        self.u_y_form = u_y_form if u_y_form is not None else (2,2)
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def plot(cls):
        return {
            "errors": cls.plot_errors(),
            "IO_best_model": cls.plot_IO_best_model()
        }
    
    @classmethod
    def plot_errors(cls): 
        #Plot of input,output and IDied output for different sizes of A matrix for system 1,2,3 can be plotted 
        #so that we can get a good feel for what m,n values we should do
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
        ax1.set_title("Error")
        ax2.set_title("Relative Error")
        for key, value in cls._system_id_plots.items():
            print(f"Preparing plots for: {key}")
            for error_type, data_set in value.items():
                if error_type == "Error":
                    ax1.plot(data_set[1], data_set[0], label=key)
                    ax1.set_ylabel("Error")
                elif error_type == "Relative Error":
                    ax2.plot(data_set[1], data_set[0], label=key)
                    ax2.set_ylabel("Relative Error (%)")

        ax1.legend()
        ax2.legend()
        return fig
    
    @classmethod
    def plot_IO_best_model(cls): #plot the input/output vs IDed output for the best model and its different num. and den. order
        if cls._best_model is None: 
            raise Exception("No best model has been identified yet. Please run at least one SystemID instance.")
        
        best_model = cls._best_model
        #For now, u_y_form is statically set inside this dictionary. A more dynamic approach can be thought of later. 
        u_y = {
            "2-2": (2,2),
            "10-10": (10,10),
        }

        for key, value in u_y.items():
            print(f"Forming Axb for U-Y form: {key}")
            _,x,_ = best_model.form_Axb(best_model.u, best_model.y, u_y_form=value)
            A_test,_,_ = best_model.form_Axb(best_model.u_test, best_model.y_test, u_y_form=value)
            y_ided = A_test @ x
            u_y[key] = y_ided
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f"Input/Output vs IDed Output for Best Model: {cls._best_model.name}")
        ax.plot(cls._best_model.t_test, cls._best_model.y_test, label="Output")
        ax.plot(cls._best_model.t_test, cls._best_model.u_test, label="Input")
        for key, y_ided in u_y.items():
            if len(y_ided) != len(cls._best_model.t_test):
                diff = len(cls._best_model.t_test) - len(y_ided)
                cls._best_model.t_test = cls._best_model.t_test[:-diff]
            ax.plot(cls._best_model.t_test, y_ided, label=f"U-Y form: {key}", linestyle='--')
        ax.legend()
        
        return fig

    def normalize_data(self):
        self.u = self.u/np.max(np.abs(self.u))
        self.y = self.y/np.max(np.abs(self.y))
    
    def run(self):
        if self.normalize: 
            self.normalize_data()
            print("Data normalized.")
        
        self.A, self.x_parameters, self.b = self.form_Axb(self.u, self.y)
        self._run_uncertainty_analysis()

    def form_Axb(self, u: NDArray[np.float64], y: NDArray[np.float64], u_y_form = (2,2)) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        #Flip data
        u = u[::-1]
        y = y[::-1]

        #Form A and b matrices
        b = y[:-np.max(u_y_form)].reshape(-1,1)
        A = np.zeros((len(u) - np.max(u_y_form), u_y_form[0] + u_y_form[1]))
        
        #Splice u/y matrix to account for different u and y values
        if u_y_form[0] != u_y_form[1]:
            diff = abs(u_y_form[0] - u_y_form[1])
            #Only way to modify the u-y arrays?
            u = u[:-diff] if u_y_form[1] > u_y_form[0] else u
            y = y[:-diff] if u_y_form[0] > u_y_form[1] else y

        #Conditionally build A matrix depending on the u_y form given
        for i in range(u_y_form[1]): #iterating the y values for the A matrix
            start = i+1
            stop = -u_y_form[1]+1+i if i != u_y_form[1]-1 else None
            print(f"Array y to index is now: {start} to {stop}")
            A[:, [i]] = -y[start : stop].reshape(-1,1) #To validate

        for i in range(u_y_form[0]): #iterating the u values for the A matrix
            start = i+1
            stop = -u_y_form[0]+1+i if i != u_y_form[0]-1 else None
            print(f"Array u to index is now: {start} to {stop}")
            A[:, [i + u_y_form[1]]] = u[start : stop].reshape(-1,1)

        rank = np.linalg.matrix_rank(A)
        print(f"Matrix A rank: {rank}, shape: {A.shape[1]}")

        #Solve the AX=b problem
        x = np.linalg.solve(A.T @ A, A.T @ b)
        print(f"The identified parameters are: {x}")
        return A,x,b

    def _run_uncertainty_analysis(self):
        """ 
        Function used to call all methods related to assesing accuracy and confidence in the identified model. Function 
        is coupled with the classes' form_Axb method to ensure all identified models are analyzed.  
        """
        self.model_report = {
            "NMSE": self.NMSE,
            "NMSE_test": self.NMSE_test(x_parameters = self.x_parameters),
            "Mean and Std Error": self.mean_and_std_error,
            "VAF": self.VAF,
            "Fit Ratio": self.fit_ratio,
        }
        SystemID._system_id_catalog[self.name] = self.model_report
        SystemID._system_id_plots[self.name] = {
            "Error" : [self.error, self.T],
            "Relative Error" : [self.relative_error, self.T]
        }
        if SystemID._best_model is None or self.NMSE_test(self.x_parameters) < SystemID._best_model.officialNMSE_test:
            SystemID._best_model = self
            SystemID._best_model.officialNMSE_test = self.NMSE_test(self.x_parameters)

    @property
    def NMSE(self) -> float:
        #Computes the normalized mean squared error of the model with the data it was identified from
        MSE = (1/self.N)*np.linalg.norm(self.b - self.A @ self.x_parameters,2)**2
        MSO = (1/self.N)*np.linalg.norm(self.b,2)**2
        return MSE/MSO
    
    def NMSE_test(self, x_parameters, u_y_form = (2,2)) -> float:
        #Computes the normalized mean squared error of the model from the test data provided
        if not hasattr(self, 'u_test') or not hasattr(self, 'y_test'):
            raise AttributeError(f"No test data was provided for the given systemID {self.name}. Please provide u_test and y_test attributes to compute NMSE_test.")
        A_test, _, b_test = self.form_Axb(self.u_test, self.y_test, u_y_form=u_y_form)
        MSE = (1/self.N)*np.linalg.norm(b_test - A_test @ x_parameters,2)**2
        MSO = (1/self.N)*np.linalg.norm(b_test,2)**2
        return MSE/MSO
    
    @property
    def error(self) -> NDArray[np.float64]:
        return self.A @ self.x_parameters - self.b
    
    @property
    def relative_error(self) -> NDArray[np.float64]:
        return self.error / np.max(np.abs(self.y)) * 100
    
    @property
    def mean_and_std_error(self) -> Tuple[float, float]:
        return np.mean(self.error), np.std(self.error)
    
    @property
    def VAF(self) -> float:
        #Returns the Variance Accounted For (VAF) of the data that the model was identified from
        return (1 - np.var(self.error)/np.var(self.y)) * 100
    
    @property
    def fit_ratio(self) -> float: 
        return (1 - np.sqrt(np.var(self.error))/np.std(self.y)) * 100

    @property
    def model_confidence(self) -> NDArray[np.float64]:
        #More computation needs to be done to draw insight from this property
        return np.linalg.norm(self.b - self.A @ self.x_parameters,2)/(self.N - (self.u_y_form[0] + self.u_y_form[1] + 1)) * np.linalg.inv(self.A.T @ self.A)
    
    def analyze_different_u_y_forms(self, u_y_forms: list[Tuple[int,int]]):
        payload = {}
        for form in u_y_forms:
            _, x, _ = self.form_Axb(self.u, self.y, u_y_form=form)
            NMSE = self.NMSE_test(x, u_y_form=form)
            payload[f"{form}"] = NMSE
        return payload
    
    def plot_u_y_forms(self, u_y_forms: dict): 
        NMSEs = [value for _, value in u_y_forms.items()]
        x = np.arange(len(u_y_forms))

        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot(x, NMSEs, marker='o')

        ax.set_xticks(x)
        ax.set_xticklabels([str(key) for key in u_y_forms.keys()])

        ax.set_xlabel("U-Y Form")
        ax.set_ylabel("NMSE")
        ax.set_title("NMSE vs U-Y Forms")
        plt.grid(True)

        return fig