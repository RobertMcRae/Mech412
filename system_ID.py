import control
import pathlib
import numpy as np
from numpy.typing import NDArray

class SystemID():

    def __init__(self, u: NDArray[np.float64], y: NDArray[np.float64], T: NDArray[np.float64], normalize = False):
        if len(u) != len(y): 
            raise ValueError(f"Input u does not have the same length as ouput y. u length: {len(u)}, y length: {len(y)}")
        self.u = u
        self.y = y
        self.T = T
        self.N = len(u)
        self.normalize = normalize

    def normalize_data(self):
        self.u = self.u/np.max(np.abs(self.u))
        self.y = self.y/np.max(np.abs(self.y))
    
    def run(self):
        if self.normalize: 
            self.normalize_data()
            print("Data normalized.")

        #Flip data "up down"
        self.u = self.u[::-1]
        self.y = self.y[::-1]

        #Form A and b matrices
        b = self.y[:-2].reshape(-1,1)
        A = np.zeros((self.N - 2, 4))
        A[:, [0]] = -self.y[1:-1].reshape(-1,1)
        A[:, [1]] = -self.y[2:].reshape(-1,1)
        A[:, [2]] = self.u[1:-1].reshape(-1,1)
        A[:, [3]] = self.u[2:].reshape(-1,1)
        rank = np.linalg.matrix_rank(A)
        print(f"Matrix A rank: {rank}, shape: {A.shape[1]}")

        #Solve the AX=b problem
        x = np.linalg.solve(A.T @ A, A.T @ b)
        print(f"The identified parameters are: {x}")

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