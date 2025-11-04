import control
import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from typing import Tuple, Optional

class SystemID():
    """
    A class used for performing system identification on Input-Output data. Initialize the class by providing the inputs, outputs and time stamps
    for each data point. Optionally, the user can provide the desired form of the A matrix in the Ax=b problem.
    """

    _system_id_catalog = {} #used to store all instances of system IDs created
    _system_id_plots = {} #used to store all plot related info for each systemID

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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
        ax1.set_title("Error over Time")
        ax2.set_title("Relative Error over Time")
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


    def normalize_data(self):
        self.u = self.u/np.max(np.abs(self.u))
        self.y = self.y/np.max(np.abs(self.y))
    
    def run(self):
        if self.normalize: 
            self.normalize_data()
            print("Data normalized.")
        
        self.x_parameters = self.solve_Axb()
        self._run_uncertainty_analysis()

    def solve_Axb(self):
        #Flip data
        self.u = self.u[::-1]
        self.y = self.y[::-1]

        #Form A and b matrices
        b = self.y[:-self.u_y_form[1]].reshape(-1,1)
        A = np.zeros((self.N - self.u_y_form[1], self.u_y_form[0] + self.u_y_form[1]))

        #Conditionally build A matrix depending on the u_y form given
        for i in range(self.u_y_form[1]): #iterating the y values for the A matrix
            start = i+1
            stop = -self.u_y_form[1]+1+i if i != self.u_y_form[1]-1 else None
            print(f"Array y to index is now: {start} to {stop}")
            A[:, [i]] = -self.y[start : stop].reshape(-1,1) #To validate

        for i in range(self.u_y_form[0]): #iterating the u values for the A matrix
            start = i+1
            stop = -self.u_y_form[0]+1+i if i != self.u_y_form[0]-1 else None
            print(f"Array u to index is now: {start} to {stop}")
            A[:, [i + self.u_y_form[1]]] = self.u[start : stop].reshape(-1,1)

        self.rank = np.linalg.matrix_rank(A)
        print(f"Matrix A rank: {self.rank}, shape: {A.shape[1]}")
        self.A = A
        self.b = b

        #Solve the AX=b problem
        x = np.linalg.solve(A.T @ A, A.T @ b)
        print(f"The identified parameters are: {x}")
        return x

    def _run_uncertainty_analysis(self):
        """ 
        Function used to call all methods related to assesing accuracy and confidence in the identified model. Function 
        is coupled with the classes' solve_Axb method to ensure all identified models are analyzed.  
        """
        self.model_report = {
            "NMSE": self.NMSE,
            "Mean and Std Error": self.mean_and_std_error,
            "VAF": self.VAF,
            "Fit Ratio": self.fit_ratio,
        }
        SystemID._system_id_catalog[self.name] = self.model_report
        SystemID._system_id_plots[self.name] = {
            "Error" : [self.error, self.T],
            "Relative Error" : [self.relative_error, self.T]
        }

    @property
    def NMSE(self) -> float:
        #Computes the normalized mean squared error of the model with the data it was identified from
        MSE = (1/self.N)*np.linalg.norm(self.b - self.A @ self.x_parameters,2)**2
        MSO = (1/self.N)*np.linalg.norm(self.b,2)**2
        return MSE/MSO
    
    def NMSE_test(self) -> float:
        #Computes the normalized mean squared error of the model from the test data provided
        pass
    
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
    def model_confidence(self) -> float:
        #More computation needs to be done to draw insight from this property
        return np.linalg.norm(self.b - self.A @ self.x_parameters,2)/(self.N - (self.u_y_form[0] + self.u_y_form[1] + 1)) * np.linalg.inv(self.A.T @ self.A)