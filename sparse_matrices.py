from SparseMatrix import SparseMatrix
import numpy as np

class sparse_matrices:


    def fromDense(self,M):
        '''generates a sparse matrix from a dense one'''
        assert M.shape[0] == M.shape[1]
        sp = SparseMatrix(M.shape[0])
        for col in range(M.shape[0]):
            for row in range(M.shape[0]):
                if M[row, col] != 0:
                    sp[row, col] = M[row, col]
        return sp

    def cnZ(self,n):
        # Matrix for controlled ^ n Z gate
        # There are n-1 control bits
        cnZ = np.eye(pow(2, n))
        cnZ[-1, -1] = -1
        return self.fromDense(cnZ)

    def constructOracle(n, ws):
        # Create n qubit dummy oracle with ws as solution states
        oracle = np.eye(pow(2, n))
        for w in ws:
            oracle[w, w] = -1
        return self.fromDense(oracle)

    def MeasureAll(register):
        cum_prob = np.cumsum(register ** 2)
        r = np.random.rand()
        measurement = np.searchsorted(cum_prob, r)
        print(f"Collapsed into basis state |{measurement}> (|{bin(measurement)[2:]}>)")
        return measurement

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
        '''applies the hadamard gate to each qubit in the list indices sent (zero based)'''
        # approach: repeatedly tensor product either an identity or gate depending on if the item is in the list

        H_single = self.fromDense(np.array([
            [1, 1],
            [1, -1]
        ]) / np.sqrt(2))
        I = self.fromDense(np.eye(2))

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
        # approach: repeatedly tensor product either an identity or gate depending on if the item is in the list
        X_single = self.fromDense(np.array([
            [0, 1],
            [1, 0]
        ]))
        I = self.fromDense(np.eye(2))
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
        Z_single = self.fromDense(np.array([
            [1, 0],
            [0, -1]
        ]))
        I = self.fromDense(np.eye(2))
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
        '''controlled z operating on the last qubit'''
        C = np.zeros([self.n, self.n], dtype=complex)
        for i in range(self.n):
            C[i, i] = 1

        C[self.n - 1, self.n - 1] = -1
        # print("aaa")
        # print(self.n)
        # print(C)
        self.state = C.dot(self.state)

    def cnot(self):
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
        # print(oracle)
        # print(self.state)
        self.state = np.dot(oracle, self.state)