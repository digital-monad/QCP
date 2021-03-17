import numpy as np
import SparseMatrix as SparseMatrix

class QuantumRegister(np.ndarray):

    def __new__(cls,n):
        # By default initialises the register to |0>
        reg = np.zeros((2**n,),dtype=complex)
        reg[0] = 1
        return reg.view(cls)

    def __matmul__(self,operator):
        return operator @ self

    def MeasureAll(self):
        cum_prob = np.cumsum(self**2)
        r = np.random.rand()
        measurement = np.searchsorted(cum_prob,r)
        print(f"Collapsed into basis state |{measurement}> (|{bin(measurement)[2:]}>)")
        return measurement