import numpy as np
from SparseMatrix import SparseMatrix
from QuantumRegister import QuantumRegister
import sparseGates as gates

# Initialise a 9 qubit quantum register
qState = QuantumRegister(9)
# Combine first 3 gates into one for efficiency
zhz = gates.Z @ gates.H @ gates.Z
# Apply tensor product to find the unitary operator
U = zhz * gates.I**8
# Apply it to the state
qState = U @ qState
qState = (gates.cX * gates.I**7) @ qState
doubleH = (gates.H * gates.H) * gates.I**7
qState = doubleH @ qState
cnot02 = gates.cXs(9,0,2)
qState = cnot02 @ qState
cnot12 = gates.cXs(9,1,2)
qState = cnot12 @ qState
H2 = gates.I**2 * gates.H * gates.I**6
qState = H2 @ qState
cnot03 = gates.cXs(9,0,3)
qState = cnot03 @ qState
cnot23 = gates.cXs(9,2,3)
qState = cnot23 @ qState
H04 = gates.H * gates.I**2 * gates.H * gates.I**5
qState = H04 @ qState
cnot04 = gates.cXs(9,0,4)
qState = cnot04 @ qState
cnot14 = gates.cXs(9,1,4)
qState = cnot14 @ qState
cnot24 = gates.cXs(9,2,4)
qState = cnot24 @ qState
Hs = gates.H**2 * gates.I**3 * gates.H**4
qState = Hs @ qState
errorGate = gates.I**3 * gates.Z * gates.I**5
qState = errorGate @ qState
cz54 = gates.cZs(9,5,4)
qState = cz54 @ qState
cnot53 = gates.cXs(9,5,3)
qState = cnot53 @ qState
cnot52 = gates.cXs(9,5,2)
qState = cnot52 @ qState
cz51 = gates.cZs(9,5,1)
qState = cz51 @ qState
cnot64 = gates.cXs(9,6,4)
qState = cnot64 @ qState
cnot63 = gates.cXs(9,6,3)
qState = cnot63 @ qState
cz62 = gates.cZs(9,6,2)
qState = cz62 @ qState
cz60 = gates.cZs(9,6,0)
qState = cz60 @ qState
cnot74 = gates.cXs(9,7,4)
qState = cnot74 @ qState
cz73 = gates.cZs(9,7,3)
qState = cz73 @ qState
cz71 = gates.cZs(9,7,1)
qState = cz71 @ qState
cnot70 = gates.cXs(9,7,0)
qState = cnot70 @ qState
cz84 = gates.cZs(9,8,4)
qState = cz84 @ qState
cz82 = gates.cZs(9,8,2)
qState = cz82 @ qState
cnot81 = gates.cXs(9,8,1)
qState = cnot81 @ qState
cnot80 = gates.cXs(9,8,0)
qState = cnot80 @ qState
finalHs = gates.I**5 * gates.H**4
qState = finalHs @ qState
qState.MeasureAll()

measurementToGate = {
    '0100' : 'X gate on qubit 0',
    '0111' : 'Y gate on qubit 0',
    '0011' : 'Z gate on qubit 0',
    '1010' : 'X gate on qubit 1',
    '1011' : 'Y gate on qubit 1',
    '0001' : 'Z gate on qubit 1',
    '0101' : 'X gate on qubit 2',
    '1101' : 'Y gate on qubit 2',
    '1000' : 'Z gate on qubit 2',
    '0010' : 'X gate on qubit 3',
    '1110' : 'Y gate on qubit 3',
    '1100' : 'Z gate on qubit 3',
    '1001' : 'X gate on qubit 4',
    '1111' : 'Y gate on qubit 4',
    '0110' : 'Z gate on qubit 4',
}




