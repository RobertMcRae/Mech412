import control
import pathlib
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional

class SystemID():
    """
    A class used for performing system identification on Input-Output data. Initialize the class by providing the inputs, outputs and time stamps
    for each data point. Optionally, the user can provide the desired form of the A matrix in the Ax=b problem.
    """

    def __init__(self, u: NDArray[np.float64], y: NDArray[np.float64], T: NDArray[np.float64], u_y_form: Optional[Tuple[int, int]] = None, **kwargs):
        if len(u) != len(y): 
            raise ValueError(f"Input u does not have the same length as ouput y. u length: {len(u)}, y length: {len(y)}")
        self.u = u
        self.y = y
        self.T = T
        self.N = len(u)
        self.u_y_form = u_y_form if u_y_form is not None else (2,2)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def normalize_data(self):
        self.u = self.u/np.max(np.abs(self.u))
        self.y = self.y/np.max(np.abs(self.y))
    
    def run(self):
        if self.normalize: 
            self.normalize_data()
            print("Data normalized.")
        
        self.x_parameters = self.solve_Axb()
        self.run_uncertainty_analysis()

    def solve_Axb(self):
        #Flip data
        self.u = self.u[::-1]
        self.y = self.y[::-1]

        #Form A and b matrices
        b = self.y[:-self.u_y_form[1]].reshape(-1,1)
        A = np.zeros((self.N - self.u_y_form[1], self.u_y_form[0] + self.u_y_form[1]))

        #Conditionally build A matrix depending on the u_y form given
        for i in range(self.u_y_form[1]): #iterating the y values for the A matrix
            A[:, [i]] = -self.y[i+1:-self.u_y_form[1]+1+i].reshape(-1,1) #To validate

        for i in range(self.u_y_form[0]): #iterating the u values for the A matrix
            A[:, [i + self.u_y_form[1]]] = self.u[i+1:-self.u_y_form[0]+1+i].reshape(-1,1)

        self.rank = np.linalg.matrix_rank(A)
        print(f"Matrix A rank: {self.rank}, shape: {A.shape[1]}")

        #Solve the AX=b problem
        x = np.linalg.solve(A.T @ A, A.T @ b)
        print(f"The identified parameters are: {x}")
        return x

    def run_uncertainty_analysis(self):
        pass

    @property
    def NMSE(self) -> float:
        return None
    
    @property
    def error(self) -> NDArray[np.float64]:
        return None
    
    @property
    def relative_error(self) -> NDArray[np.float64]:
        return None
    
    @property
    def VAF(self) -> float:
        return None
    
    @property
    def fit_ratio(self) -> float: 
        return None

    @property
    #Need to verify what this returns
    def model_confidence(self) -> None:
        return None

#To Update
data = np.loadtxt('/Users/robertmcrae/Desktop/McGill/Fall 2025/MECH 412/Assignments/Project/PRBS_DATA/IO_data_1_0.csv', delimiter=',')
test = SystemID(u = data[:,1], y = data[:,2], T = data[:,0], normalize = True)
test.run()
data = np.loadtxt('/Users/robertmcrae/Desktop/McGill/Fall 2025/MECH 412/Assignments/Project/PRBS_DATA/IO_data_1_1.csv', delimiter=',')
test = SystemID(u = data[:,1], y = data[:,2], T = data[:,0], normalize = True)
test.run()
data = np.loadtxt('/Users/robertmcrae/Desktop/McGill/Fall 2025/MECH 412/Assignments/Project/PRBS_DATA/IO_data_1_2.csv', delimiter=',')
test = SystemID(u = data[:,1], y = data[:,2], T = data[:,0], normalize = True)
test.run()
data = np.loadtxt('/Users/robertmcrae/Desktop/McGill/Fall 2025/MECH 412/Assignments/Project/PRBS_DATA/IO_data_1_3.csv', delimiter=',')
test = SystemID(u = data[:,1], y = data[:,2], T = data[:,0], normalize = True)
test.run()