# attempt 1 at structuring the code better
import numpy as np
from ExplicitMatrix import ExplicitMatrix
from SparseMatrix import SparseMatrix
from sparse_matrices import sparse_matrices
from Grover import Grover

# Structure: have multiple classes, each with a suite of all the operations we are going to want, then only inherit/import
# the one we actually want to use (which can be switched really easily) to a class which has many algorithms implemented

######################################################################################################################
# The class state is defined below, and the entirety of the code in main should make the general control loop
# for the program. This is to allow for the user to choose the calculation method used during runtime.


def main(rep,q,ws):
    '''The main function, takes parameters from the ui and runs the corresponding quantum program'''
    # if the gates are named the same in alternative classes, it should just be a quick swap and any written algorithms
    # should work as before (but potentially more/less efficiently)

    def qubits(q):
        zero = np.array([1,0])
        qs = []
        for i in range(q):
            qs.append(zero)
        return qs

    # user input, to decide program, calculation method, inputs.

    input = qubits(q)

    if rep == 1:
        methods = ExplicitMatrix
    if rep == 2:
        methods = sparse_matrices
    # can be user defined from a list using a case statement possibly?

    # can be user defined from a list using a case statement possibly?

    class state(methods):
        '''A quantum state, imports gate operations from the class specified by "methods", assumed to contain a
        standard set of methods'''
        def __init__(self, bits):
            '''initialise a quantum state, with the given list of quantum bits. Does the entanglement automatically'''
            self.state = np.zeros((2**bits))
            self.state[0] = 1
            self.n = len(self.state)  # = 2 ^ num of entangled qubits
            self.qubit_n = bits
            assert self.n == 2 ** self.qubit_n
            print("Initial state:")
            print(bits)
            print("entangled: ")
            print(self.state)
            print("-----------------------")

        def entangle(self, bits):
            """returns the entangled state for n Qubits"""
            # this is just a special case of the tensor product repeatedly applied

            state = [1]  # set up the start of the multiplication
            bits.reverse()  # to get the bits in reverse order, since the loop operates backwards on them
            for bit in bits:  # adds each qubit 1 by 1 to an entangled state
                for element, comparison in zip(bit, self.normalise(bit)):
                    # check that "bit" is normalised, otherwise throw error
                    assert element == comparison, "State not normalised: output will be incorrect"

                # add the next bit to the entangled state, using a tensor product:
                state = list(np.array(state).dot(bit[0])) + list(np.array(state).dot(bit[1]))

            return np.array(state)

        def measure(self,state):
            '''returns and prints a quantum measurement on the state object'''
            cum_prob = np.cumsum(state.get_state() ** 2)
            r = np.random.rand()
            measurement = np.searchsorted(cum_prob, r)
            print(f"Collapsed into basis state |{measurement}> (|{bin(measurement)[2:]}>)")
            print(f"P(|{measurement}>) = {state.get_state()[measurement] ** 2}")
            return measurement

        def print_register(self,state):
            '''prints the coefficients of the quantum state'''
            print(f"The quantum register is in the state {state.get_state()}")

        def normalise(self, vector):
            ''' returns normalised form of sent vector '''
            vector = np.array(vector)  # incase state is sent as a list
            total = 0
            for component in vector:
                total += component ** 2
            return vector / (total ** (1 / 2))

        def __matmul__(self, operator):
            '''matrix multiplication method'''
            return operator @ self

        def get_state(self):
            '''returns the coefficients in the quantum register'''
            return self.state

    np.set_printoptions(formatter={'complex_kind': '{:.5f}'.format})  # formatting, hides floating point errors from
    # making messy looking outputs

    # every time I write 'run on the state', I mean send the state object (not just tbe vector) to the method.

    # all the outputs/ calculations should go here: create a state with the given inputs,
    # run the needed programs on that state
    # to output a measurement, simply run programs.measure on the state

    # example of how to use the class:
    print("starting")
    print("...")
    print(rep)
    my_state = state(q)
    final_state = Grover(my_state, ws)
    final_state.measure(final_state)
    print("")


def UI():
    '''Handles the UI of the program'''
    print("QUANTUM COMPUTER SIMULATOR")
    print()
    print("Please provide only integer inputs")
    print()

    alg = int(input("Menu options \n (1) Grover Search \n (2) Quantum Error Correction \n (3) Quantum Fourier Transform \n (4) Shor's Algorithm \n (5) exit \n : "))
    if alg not in [1,2,3,4,5]:
        print("User input is not recognised")
        quit()


    if alg == 1:
    # Custom Grover
    # This checks answer is an option and returns error message and quits program if answer is not an availible choice

        rep = int(input("Please choose Matrix representation \n (1) Explicit \n (2) Sparse Matricies \n : "))
        print()
    # This checks answer is an option and returns error message and quits program if answer is not an availible choice
        if rep not in [1,2]:
            print("User input is not recognised")
            quit()

        q = int(input("Please choose number of Qubits \n (Provide answer as an integer): "))
        print()
        ws = list(map(int,input(f"Please provide the winner state(s) as integers < 2^{q}: ").split()))

        main(rep,q,ws)

    if alg == 2:
        # QEC
        from QEC import QEC
        QEC()

    if alg == 3:
        # QEC
        from QFT import QFT
        from QuantumRegister import QuantumRegister
        n = int(input("Number of qubits: "))
        input_vec = np.array(list(map(float,input("Input state vector (Size 2^n), e.g. 0.5 0.5 0.5 0.5 :").split())))
        QFT(n,input_vec.view(QuantumRegister),n)

    if alg == 4:
        from Shor import Shor
        print("Demo Shor Period Finding for N = 15 - you've got enough time to make yourself some tea before this is done..")
        Shor()


UI()