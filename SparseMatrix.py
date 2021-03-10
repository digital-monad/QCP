import numpy as np

class SparseMatrix:

    def __init__(self,dim):
        self.dim = dim
        self.cols = np.array([dict() for _ in range(dim)])
    
    def __getitem__(self,indices):
        row, col = indices
        if row in self.cols[col]:
            return self.cols[col][row]
        else:
            return 0

    def __setitem__(self,indices,value):
        row, col = indices
        self.cols[col][row] = value

    def __matmul__(self,other):
        if isinstance(other, SparseMatrix):
            result = SparseMatrix(self.dim)
            for col in range(len(other.cols)):
                for row in other.cols[col]:
                    thisColumn = self.cols[row]
                    for thisRow in thisColumn:
                        result[thisRow,col] = result[thisRow,col] + other.cols[col][row] * thisColumn[thisRow]
            return result
        elif isinstance(other, (int,float)):
            for col in range(len(self.cols)):
                self.cols[col] = dict(map(lambda x : (x[0],other*x[1]), self.cols[col].items()))
            return self

    def __rmul__(self,other):
        if isinstance(other, (int,float)):
            return self.__mul__(other)


    def __str__(self):
        output = ""
        for row in range(self.dim):
            for col in range(self.dim):
                output += str(self[row,col])
                output += " "
            output += "\n"
        return output

    def __mul__(self,other):
        result = SparseMatrix(self.dim**2)
        for row in range(self.dim**2):
            for col in range(self.dim**2):
                result[row,col] = self[row//self.dim,col//self.dim] * other[row%self.dim,col%self.dim]
        return result


def fromDense(M):
    assert M.shape[0] == M.shape[1]
    sp = SparseMatrix(M.shape[0])
    for col in range(M.shape[0]):
        for row in range(M.shape[0]):
            if M[row,col] != 0:
                sp[row,col] = M[row,col]
    return sp
