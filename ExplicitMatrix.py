import numpy as np

class ExplicitMatrix:
    # this is going to be one implementation of the gates. Many will follow.
    # there may be better ways of creating this from a parent class if needed, depends how much can be reused between
    # this and sparse/lazy matrix implementations

    # weaknesses of this particular implementation (not exhaustive):
    # ~2 ^2n operations per gate application - v slow
    # gates generated each time- slow if they are used repeatedly (having the option to store may reduce time if say
    # we use hadamard again and again in one circuit, though storing adds to memory cost)


    def tensor_product(self, U, V):
        ###### should this be in the general class? should each of these classes inherit from an even further up class?
        '''Naïve tensor product implementation'''
        (rU, cU), (rV, cV) = U.shape, V.shape
        result = np.zeros((rU * rV, cU * cV),dtype=complex)
        for r in range(rU):
            for c in range(cU):
                result[r * rV:(r + 1) * rV, c * cV:(c + 1) * cV] = U[r, c] * V
        return result

    # gates that act upon all qubits with the same separate operation
    # hadamard is heavily annotated as an example

    def hadamard_all(self):
        '''applies the hadamard gate to each qubit in the entangled state (by reference)'''
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
        '''applies the hadamard gate to each qubit in the list indices sent (zero based)'''
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