# General structure built up from multiple classes and UI
import numpy as np
import time

# multiple classes, each with a suite of all the operations we are going to want, then only inherit/import
# the one we actually want to use (which can be switched really easily) to a class which has many algorithms implemented



class explicit_matrices:


    def tensor_product(self, U, V):
        '''Naïve tensor product implementation'''
        (rU, cU), (rV, cV) = U.shape, V.shape
        result = np.zeros((rU * rV, cU * cV))
        for r in range(rU):
            for c in range(cU):
                result[r * rV:(r + 1) * rV, c * cV:(c + 1) * cV] = U[r, c] * V
        return result

    # gates that act upon all qubits with the same separate operation
    # hadamard is heavily annotated as an example

    def hadamard_all(self):
        '''Applies the hadamard gate to each qubit in the entangled state (by reference)'''
        n = 2  # reminder- needs generalising

        init = np.ones([2, 2], dtype=complex)  # set up array
        init[1][1] = -1  # correct the values to get a hadamard gate

        H = init * (1 / np.sqrt(2))  # normalisation
        H_single = H
        for val in range(self.qubit_n - 1):  # v slow, but works
            H = self.tensor_product(H, H_single)
        # H now is the full matrix for a hadamard gate that will be applied to every qubit in an entangled state of
        # self.n size

        # applying the matrix to the current state of the system to
        self.state = H.dot(self.state)

    def x_all(self):
        '''Applies the pauli-X gate to each qubit in the entangled state (by reference)'''
        n = 2
        X = np.ones([2, 2], dtype=complex)
        X[0][0] = 0
        X[1][1] = 0
        X_single = X

        for val in range(self.qubit_n - 1):  # v slow, but works
            X = self.tensor_product(X, X_single)

        self.state = X.dot(self.state)

    def x_all_fast(self):
        '''Applies the pauli-X gate to each qubit in the entangled state (by reference), more efficient than x_all'''
        n = self.qubit_n
        d = 2**n
        Xall = np.zeros([d,d])
        for r in range(d):
            Xall[r][d - (r+1)] = 1
        self.state = Xall.dot(self.state)

    def y_all(self):
        '''Applies the pauli-Y gate to each qubit in the entangled state (by reference)'''
        n = 2
        Y = np.zeros([2, 2], dtype=complex)
        Y[0][1] = _i
        Y[1][0] = i
        Y_single = Y
        for val in range(self.qubit_n - 1):  # v slow, but works
            Y = self.tensor_product(Y, Y_single)

        self.state = Y.dot(self.state)

    def z_all(self):
        '''Applies the pauli-Z gate to each qubit in the entangled state (by reference)'''
        Z = np.zeros([2, 2], dtype=complex)
        Z[0][0] = 1
        Z[1][1] = -1
        Z_single = Z
        for val in range(self.qubit_n - 1):  # v slow, but works
            Z = self.tensor_product(Z, Z_single)

        self.state = Z.dot(self.state)

    def phase_all(self, phi):
        '''Applies a phase change of phi to all Qubits in the register (by reference)'''
        # angle can be changed easily, set in pi-radians
        n = 2
        P = np.zeros([2, 2], dtype=complex)
        P[0][0] = 1
        P[1][1] = np.exp(i * phi * np.pi)

        for val in range(self.qubit_n - 1):  # v slow, but works
            P = self.tensor_product(P, P)

        self.state = P.dot(self.state)

    # gates that act upon a single qubit with an operation (reminder to send the qubit number)

    # empty (for now, we want every single state in this form: compose them from a 1 qubit gate + identity and
    # repeated tensor products to combine if necessary. setup depends on qubit number and which qubit is acted on)

    # gates that act on the list of qubits sent:

    def hadamard_list(self, list):
        '''applies the hadamard gate to each qubit in the list indices sent (zero based, changes made by reference)'''
        # approach: repeatedly tensor product either an identity or gate depending on if the item is in the list

        init = np.ones([2, 2], dtype=complex)  # set up array
        init[1][1] = -1  # correct the values to get a hadamard gate
        I = np.array([[1, 0], [0, 1]])
        if 0 in list:
            H = init * (1 / np.sqrt(2))  # normalisation
        else:
            H = I
        H_single = init * (1 / np.sqrt(2))

        for val in range(self.qubit_n - 1):  # v slow, but works
            if val + 1 in list:
                H = self.tensor_product(H, H_single)
            else:
                H = self.tensor_product(H, I)
        # H now is the full matrix for a hadamard gate that will be applied to every qubit in an entangled state of
        # self.n size

        # applying the matrix to the current state of the system to
        self.state = H.dot(self.state)

    def x_list(self, list):
        '''applies the Pauli-X gate to each qubit in the list indices sent (zero based, changes made by reference'''
        # approach: repeatedly tensor product either an identity or gate depending on if the item is in the list
        X = np.ones([2, 2], dtype=complex)
        X[0][0] = 0
        X[1][1] = 0
        X_single = X
        I = np.array([[1, 0], [0, 1]])
        if not 0 in list:
            X = I  # special case for if the first qubit does not have an x gate across it (zero isn't in list)
        for val in range(self.qubit_n - 1):  # v slow, but works
            if val + 1 in list:
                X = self.tensor_product(X, X_single)
            else:
                X = self.tensor_product(X, I)

        self.state = X.dot(self.state)

    def y_list(self, list):
        '''applies the Pauli-Y gate to each qubit in the list indices sent (zero based, changes made by reference'''
        # approach: repeatedly tensor product either an identity or gate depending on if the item is in the list
        Y = np.zeros([2, 2], dtype=complex)
        Y[0][1] = _i
        Y[1][0] = i
        Y_single = Y
        I = np.array([[1, 0], [0, 1]])
        if not 0 in list:
            Y = I  # special case for if the first qubit does not have an x gate across it (zero isn't in list)
        for val in range(self.qubit_n - 1):  # v slow, but works
            if val + 1 in list:
                Y = self.tensor_product(Y, Y_single)
            else:
                Y = self.tensor_product(Y, I)

        self.state = Y.dot(self.state)

    def z_list(self, list):
        '''applies the Pauli-Z gate to each qubit in the list indices sent (zero based, changes made by reference)'''
        # approach: repeatedly tensor product either an identity or gate depending on if the item is in the list
        Z = np.zeros([2, 2], dtype=complex)
        Z[0][0] = 1
        Z[1][1] = -1
        Z_single = Z
        I = np.array([[1, 0], [0, 1]])
        if not 0 in list:
            Z = I  # special case for if the first qubit does not have an x gate across it (zero isn't in list)
        for val in range(self.qubit_n - 1):  # v slow, but works
            if val + 1 in list:
                Z = self.tensor_product(Z, Z_single)
            else:
                Z = self.tensor_product(Z, I)

        self.state = Z.dot(self.state)

    # controlled gates (act on multiple qubits, in some more complicated way)

    def c_z_last(self):
        '''controlled z operating on the last qubit (changes made by reference)'''
        C = np.zeros([self.n, self.n], dtype=complex)
        for i in range(self.n):
            C[i, i] = 1

        C[self.n - 1, self.n - 1] = -1
        # print("aaa")
        # print(self.n)
        # print(C)
        self.state = C.dot(self.state)

    def cnot(self):
        '''Cnot gate'''
        n = 4
        C = np.zeros([self.n, self.n])
        C[0][0] = 1
        C[1][1] = 1
        C[2][3] = 1
        C[3][2] = 1

        self.state = C.dot(self.state)

    def oracle_general(self, ws):
        '''dummy oracle with ws as solution states (changes made by reference)'''
        # Create n qubit dummy oracle with ws as solution states

        oracle = np.eye(pow(2, int(self.qubit_n)))
        for w in ws:
            oracle[w, w] = -1
        #print(oracle)
        #print(self.state)
        self.state = np.dot(oracle, self.state)


from SparseMatrix import SparseMatrix

class sparse_matrices:


    def fromDense(self, M):
        '''defines a sparse matrix'''
        assert M.shape[0] == M.shape[1]
        sp = SparseMatrix(M.shape[0])
        for col in range(M.shape[0]):
            for row in range(M.shape[0]):
                if M[row, col] != 0:
                    sp[row, col] = M[row, col]
        return sp

    def hadamard_all(self):
        '''applies the hadamard gate to each qubit in the entangled state (by reference)'''
        n = 2  # reminder- needs generalising

        H = self.fromDense(np.array([
            [1, 1],
            [1, -1]
        ]) / np.sqrt(2))

        H = H**self.qubit_n
        self.state = H @ self.state

    def x_all(self):
        '''Applies the pauli-X gate to each qubit in the entangled state (by reference)'''
        X = self.fromDense(np.array([
            [0, 1],
            [1, 0]
        ]))

        X = X**self.qubit_n

        self.state = X @ self.state

    def x_all_fast(self):
        '''Applies the pauli-X gate to each qubit in the entangled state (by reference), more efficient than x_all'''
        n = self.qubit_n
        d = 2**n
        Xall = np.zeros([d,d])
        for r in range(d):
            Xall[r][d - (r+1)] = 1
        #self.state = Xall.dot(self.state)
        self.state = Xall @ self.state

    def y_all(self):
        '''Applies the pauli-Y gate to each qubit in the entangled state (by reference)'''
        Y = self.fromDense(np.array([
            [0, -i],
            [i, 0]
        ]))

        Y = Y ** self.qubit_n

        self.state = Y @ self.state

    def z_all(self):
        '''Applies the pauli-Z gate to each qubit in the entangled state (by reference)'''
        Z = self.fromDense(np.array([
            [1, 0],
            [0, -1]
        ]))
        Z = Z**self.qubit_n

        self.state = Z @ self.state



    # gates that act upon a single qubit with an operation (reminder to send the qubit number)

    # empty (for now, we want every single state in this form: compose them from a 1 qubit gate + identity and
    # repeated tensor products to combine if necessary. setup depends on qubit number and which qubit is acted on)

    # gates that act on the list of qubits sent:

    def hadamard_list(self, list):
        '''applies the hadamard gate to each qubit in the list indices sent (zero based, changes made by reference)'''

        H_single = self.fromDense(np.array([
            [1, 1],
            [1, -1]
        ]) / np.sqrt(2))
        I = fromDense(np.eye(2))

        init = np.ones([2, 2], dtype=complex)  # set up array
        init[1][1] = -1  # correct the values to get a hadamard gate
        if 0 in list:
            H = H_single
        else:
            H = I

        for val in range(self.qubit_n - 1):  # v slow, but works
            if val + 1 in list:
                H = H * H_single
            else:
                H = H * I
        # H now is the full matrix for a hadamard gate that will be applied to every qubit in an entangled state of
        # self.n size

        # applying the matrix to the current state of the system to
        self.state = H @ self.state

    def x_list(self, list):
        '''applies the Pauli-X gate to each qubit in the list indices sent (zero based, changes made by reference'''

        X_single = self.fromDense(np.array([
            [0, 1],
            [1, 0]
        ]))
        I = fromDense(np.eye(2))
        if not 0 in list:
            X = I  # special case for if the first qubit does not have an x gate across it (zero isn't in list)
        else:
            X = X_single
        for val in range(self.qubit_n - 1):  # v slow, but works
            if val + 1 in list:
                X = X*X
            else:
                X = X*I

        self.state = X @ self.state

    def y_list(self, list):
        '''applies the Pauli-Y gate to each qubit in the list indices sent (zero based, changes made by reference'''

        Y = np.zeros([2, 2], dtype=complex)
        Y[0][1] = _i
        Y[1][0] = i
        Y_single = Y
        I = np.array([[1, 0], [0, 1]])
        if not 0 in list:
            Y = I  # special case for if the first qubit does not have an x gate across it (zero isn't in list)
        for val in range(self.qubit_n - 1):  # v slow, but works
            if val + 1 in list:
                Y = self.tensor_product(Y, Y_single)
            else:
                Y = self.tensor_product(Y, I)

        self.state = Y.dot(self.state)

    def z_list(self, list):
        '''applies the Pauli-Z gate to each qubit in the list indices sent (zero based, changes made by reference)'''

        Z_single = self.fromDense(np.array([
            [1, 0],
            [0, -1]
        ]))
        I = fromDense(np.eye(2))
        if not 0 in list:
            Z = I  # special case for if the first qubit does not have an x gate across it (zero isn't in list)
        else:
            Z = Z_single
        for val in range(self.qubit_n - 1):  # v slow, but works
            if val + 1 in list:
                Z = Z * Z
            else:
                Z = Z * I

        self.state = Z @ self.state

    # controlled gates (act on multiple qubits, in some more complicated way)

    def c_z_last(self):
        '''controlled z operating on the last qubit (changes made by reference)'''

        C = np.zeros([self.n, self.n], dtype=complex)
        for i in range(self.n):
            C[i, i] = 1

        C[self.n - 1, self.n - 1] = -1
        # print("aaa")
        # print(self.n)
        # print(C)
        self.state = C.dot(self.state)

    def cnot(self):
        '''Cnot gate, changes made by reference'''
        n = 4
        C = np.zeros([self.n, self.n])
        C[0][0] = 1
        C[1][1] = 1
        C[2][3] = 1
        C[3][2] = 1

        self.state = C.dot(self.state)

    def oracle_general(self, ws):
        '''dummy oracle with ws as solution states (changes made by reference)'''

        oracle = np.eye(pow(2, int(self.qubit_n)))
        for w in ws:
            oracle[w, w] = -1
        # print(oracle)
        # print(self.state)
        self.state = np.dot(oracle, self.state)



class programs:
    '''Each quantum program is a method here. This class does not need to be instantiated'''

    def grovers(state, ws):
        '''Runs grovers on a state object (assuming a quantum register of zeroes), with the given ws for the oracle'''

        state.hadamard_all()
        # Perform rotation
        # Change number in for loop - getting weird results

        # Perform rotation
        # Change number in for loop - getting weird results
        i = 4* int(state.qubit_n ** (1/2))  # iteration num = sqrt N * some factor
        print(i)
        for iteration in range(40):
            # Apply the phase oracle
            state.oracle_general(ws)
            # Probability amplification (diffuser)
            state.hadamard_all()
            state.x_all_fast()
            state.c_z_last()
            state.x_all_fast()
            state.hadamard_all()
        return state

    def qft(n):
        '''Runs a quantum fourier transformation for n qubits'''

        # does not follow main program structure due to needing more gates not yet implemented
        from QuantumRegister import QuantumRegister
        import sparseGates as gates

        state = QuantumRegister(n)
        for rot in range(n - 1):
            hn = gates.H
            hn = gates.I ** rot * gates.H if rot > 0 else hn
            hn = hn * gates.I ** (n - rot - 1)
            state = hn @ state
            for qubit in range(n - 1):
                rotation = gates.CROT(n, qubit + 1, rot, qubit + 2)
                state = rotation @ state
        # hadamard the last qubit
        hn = gates.I ** (n - 1) * gates.H
        state = hn @ state
        # Swap pairs of qubits
        for qubit in range(n // 2):
            state = gates.swap(n, qubit, n - qubit - 1) @ state
        print(np.sum(state ** 2))
        state.MeasureAll()

    def error_correction(state):
        '''runs a demonstration of quantum error correction on the sent state'''
        # does not follow main program structure due to needing more gates not yet implemented
        # runs quantum error correction
        from QuantumRegister import QuantumRegister
        import sparseGates as gates

        # Initialise a 9 qubit quantum register
        qState = QuantumRegister(9)
        # Combine first 3 gates into one for efficiency
        zhz = gates.Z @ gates.H @ gates.Z
        # Apply tensor product to find the unitary operator
        U = zhz * gates.I ** 8
        # Apply it to the state



        qState = U @ qState
        qState = (gates.cX * gates.I ** 7) @ qState
        doubleH = (gates.H * gates.H) * gates.I ** 7
        qState = doubleH @ qState
        cnot02 = gates.cXs(9, 0, 2)
        qState = cnot02 @ qState
        cnot12 = gates.cXs(9, 1, 2)
        qState = cnot12 @ qState
        H2 = gates.I ** 2 * gates.H * gates.I ** 6
        qState = H2 @ qState
        cnot03 = gates.cXs(9, 0, 3)
        qState = cnot03 @ qState
        cnot23 = gates.cXs(9, 2, 3)
        qState = cnot23 @ qState
        H04 = gates.H * gates.I ** 2 * gates.H * gates.I ** 5
        qState = H04 @ qState
        cnot04 = gates.cXs(9, 0, 4)
        qState = cnot04 @ qState
        cnot14 = gates.cXs(9, 1, 4)
        qState = cnot14 @ qState
        cnot24 = gates.cXs(9, 2, 4)
        qState = cnot24 @ qState
        Hs = gates.H ** 2 * gates.I ** 3 * gates.H ** 4
        qState = Hs @ qState
        errorGate = gates.I ** 3 * gates.Z * gates.I ** 5
        qState = errorGate @ qState
        cz54 = gates.cZs(9, 5, 4)
        qState = cz54 @ qState
        cnot53 = gates.cXs(9, 5, 3)
        qState = cnot53 @ qState
        cnot52 = gates.cXs(9, 5, 2)
        qState = cnot52 @ qState
        cz51 = gates.cZs(9, 5, 1)
        qState = cz51 @ qState
        cnot64 = gates.cXs(9, 6, 4)
        qState = cnot64 @ qState
        cnot63 = gates.cXs(9, 6, 3)
        qState = cnot63 @ qState
        cz62 = gates.cZs(9, 6, 2)
        qState = cz62 @ qState
        cz60 = gates.cZs(9, 6, 0)
        qState = cz60 @ qState
        cnot74 = gates.cXs(9, 7, 4)
        qState = cnot74 @ qState
        cz73 = gates.cZs(9, 7, 3)
        qState = cz73 @ qState
        cz71 = gates.cZs(9, 7, 1)
        qState = cz71 @ qState
        cnot70 = gates.cXs(9, 7, 0)
        qState = cnot70 @ qState
        cz84 = gates.cZs(9, 8, 4)
        qState = cz84 @ qState
        cz82 = gates.cZs(9, 8, 2)
        qState = cz82 @ qState
        cnot81 = gates.cXs(9, 8, 1)
        qState = cnot81 @ qState
        cnot80 = gates.cXs(9, 8, 0)
        qState = cnot80 @ qState
        finalHs = gates.I ** 5 * gates.H ** 4
        qState = finalHs @ qState
        qState.MeasureAll()

    def measure(state):
        '''returns and prints a quantum measurement on the state object'''
        cum_prob = np.cumsum(state.get_state() ** 2)
        r = np.random.rand()
        measurement = np.searchsorted(cum_prob, r)
        print(f"Collapsed into basis state |{measurement}> (|{bin(measurement)[2:]}>)")
        print(f"P(|{measurement}>) = {state.get_state()[measurement] ** 2}")
        return measurement

    def print_register(state):
        '''prints the coefficients of the quantum state'''
        print(f"The quantum register is in the state {state.get_state()}")

# everything above here could be in a separate file if wanted, then imported
######################################################################################################################
# the class state has to be defined below, and the entirety of the code in main should make the general control loop
# for the program. This is to allow for the user to choose the calculation method used during runtime.


def main(q,ws,method):
    '''main, takes parameters from the ui and runs the corresponding quantum program'''
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

    if method == 1:
        methods = explicit_matrices
    if method == 2:
        methods = sparse_matrices
    # can be user defined from a list using a case statement possibly?

    # can be user defined from a list using a case statement possibly?

    class state(methods):
        '''A quantum state, imports gate operations from the class specified by "methods", assumed to contain a
        standard set of methods'''

        def __init__(self, bits):
            '''initialise a quantum state, with the given list of quantum bits. Does the entanglement automatically'''
            self.state = self.entangle(bits)
            self.n = len(self.state)  # = 2 ^ num of entangled qubits
            self.qubit_n = len(bits)
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

    np.set_printoptions(formatter={'complex_kind': '{:.5f}'.format})  # formatting
    # all the outputs/ calculations should go here: create a state with the given inputs,
    # run the needed programs on that state
    # to output a measurement, simply run programs.measure on the state

    # example of how to use the class:
    print("starting")
    print("...")
    my_state = state(input)
    final_state = programs.grovers(my_state, ws)
    programs.measure(final_state)
    print("")
    print("done")


def UI():
    print()
    print("Please provide only integer inputs")
    print()

    menu = int(input("Menu options \n (1) Custom Grovers \n (2) 9 Quibit Grovers with error correction \n (3) exit \n : "))
    if menu != 1:
        if menu != 2:
            if menu != 3:
                print("User input is not recognised")
                quit()

        # option for QFT was going to be added here but ran out of time, still availble to be run in its own file/format

    if menu == 1:
    # This checks answer is an option and returns error message and quits program if answer is not an availible choice

        input2 = int(input("Please choose Matrix representation \n (1) Explicit \n (2) Sparse Matricies \n : "))
        print()
    # This checks answer is an option and returns error message and quits program if answer is not an availible choice
        if input2 != 1:
            if input2 != 2:
                print("User input is not recognised")
                quit()

        q = int(input("Please choose number of Qubits \n (Provide answer as an integer): "))
        print()
    # This checks answer is an option and returns error message and quits program if answer is not an availible choice
    # it apparently doesnt work
        if q == str:
            print("Only Numbers Allowed")
            quit()

        print()
        w = np.random.randint(0,q-1)
        ws = [w,w+1]
        print(ws)
        t1 = time.time()
        main(q,ws,input2)
        t2 = time.time()
        print(t2-t1)
UI()

