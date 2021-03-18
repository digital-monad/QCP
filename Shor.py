from QFT import QFT
import numpy as np
from QuantumRegister import QuantumRegister
import sparseGates as gates

# Test N = 15
def Shor():
    '''Runs the period finding part of Shor's algorithm to factorise numbers'''
    N = 15
    # Take a as 7
    a = 7
    # Create a register with 12 qubits
    state = QuantumRegister(12)
    # Put register 1 in the 0 state and register 2 in the 1 state
    print("Hadamarding All Counting Qubits")
    state = (gates.H**8 * gates.I**3 * gates.X) @ state
    print("Starting Controlled Unitary Period Finding")

    for i in range(8):
        U_n = gates.cU(8,4,8 - 1 - i,a,i,N)
        state = U_n @ state

    print("Applying Quantum Fourier Transform to Counting Qubits")
    state = QFT(12,state,8)


    state.MeasureAll()
