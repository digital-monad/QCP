# random attempts, please re-write for actual version

import numpy as np


def normalise(state):
    total = 0
    for component in state:
        total += component ** 2
    return state / (total ** (1 / 2))


def tensor_prod(a, b):
    """returns the tensor product of a and b in vector form, for 2D a and b"""
    # this is a first attempt: needs generalising
    result = np.array([a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]])
    return result


def tensor(A, B):
    # for 2 d operations only
    result = np.array(
        [
            [A[0,0]*B[0,0], A[0,0]*B[0,1], A[0,1]*B[0,0], A[0,1]*B[0,1]],
            [A[0, 0] * B[1, 0], A[0, 0] * B[1, 1], A[0, 1] * B[1, 0], A[0, 1] * B[1, 1]],
            [A[1, 0] * B[0, 0], A[1, 0] * B[0, 1], A[1, 1] * B[0, 0], A[1, 1] * B[0, 1]],
            [A[1, 0] * B[1, 0], A[1, 0] * B[1, 1], A[1, 1] * B[1, 0], A[1, 1] * B[1, 1]],
        ]
    )
    return result



def C_u_double(state, u):
    # 2 state controlled operator, a is control b is operated on, operator is u
    # there are going to be quicker ways to do this, but this is more intuitive for me
    # a and b are the input states
    # u is the operation e.g. z
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, u[0, 0], u[0, 1]],
        [0, 0, u[1, 0], u[1, 1]]
    ])
    #print(matrix)
    result = matrix.dot(state)
    #print(result)
    return result


def main():

    # set up states a, b, matrix transforms
    zero = np.array([1, 0])
    one = np.array([0, 1])

    # set inputs
    a = zero
    b = zero

    # run grovers:
    print("Initial state: ")
    state = tensor_prod(a, b)
    print(state)
    hadamard = np.array([[1, 1], [1, -1]]) * (1 / 2 ** 0.5)
    z = np.array([[1, 0], [0, -1]])
    double_hadamard = tensor(hadamard, hadamard)
    state = double_hadamard.dot(state)
    # hadamard transform both states
    # oracle (sift 1):
    state = C_u_double(state, z)
    #print("test")
    #print(state)
    z_double = tensor(z, z)
    state = z_double.dot(state)
    state = C_u_double(state, z)
    state = double_hadamard.dot(state)
    print("result:")
    print(state)

    print("")
    # compare to states:
    print("states, for reference: ")
    print("zero, zero:")
    print(tensor_prod(zero, zero))
    print(" ")
    print("zero, one:")
    print(tensor_prod(zero, one))
    print(" ")
    print("one, zero:")
    print(tensor_prod(one, zero))
    print(" ")
    print("one, one:")
    print(tensor_prod(one, one))
    print(" ")






main()
