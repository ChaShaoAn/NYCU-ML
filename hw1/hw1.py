import argparse
import numpy as np
import matplotlib.pyplot as plt

def transpose(M):
    row, col = M.shape

    Mt = np.zeros((col, row))
    for r in range(row):
        for c in range(col):
            Mt[c][r] = M[r][c]

    return Mt

def product(A, B):
    rowA, colA = A.shape
    rowB, colB = B.shape

    # check if it is legal
    assert(colA == rowB)

    AB = np.zeros((rowA, colB))
    for r in range(rowA):
        for c in range(colB):
            for i in range(colA):
                AB[r][c] += A[r][i] * B[i][c]

    return AB

def inverse(M):
    # using LU decomposition
    row, col = M.shape
    # need to be n x n matrix
    assert(row == col)
    I = np.eye(row)
    # argumented matrix
    M_ext = np.hstack((M, I))

    # up to down
    for i in range(row - 1):
        for j in (range(i+1, row)):
            t = 1. * M_ext[j][i] / M_ext[i][i]
            M_ext[j][:] = M_ext[j][:] - t * M_ext[i][:]

    # standard
    for i in range(row):
        M_ext[i][:] = M_ext[i][:] * (1. / M_ext[i][i])

    # down to up
    for i in range(row -1, -1, -1):
        for j in range(i+1, row):
            M_ext[i][:] -= M_ext[j][:] * M_ext[i][j]

    # return n x n matrix
    return M_ext[:, row:]

def show_result(x, error, method='LSE'):
    if (method == 'LSE'):
        print("LSE:")
    elif (method == 'Newton'):
        print("Newton's Mwthod:")
    print("Fitting line:", end=' ')
    for i in range(x.shape[0] - 1):
        print('{}x^{}'.format(x[i][0], x.shape[0] - i - 1), end=' ')
        if(x[i]>0):
            print("+", end=' ')
        else:
            print("-", end=' ')
            x[i] *= (-1)
    print('{}'.format(x[x.shape[0]-1][0]))
    print('Total error: {}\n'.format(error[0][0]))

def LSE(A, b, lamb):
    AtA = product(transpose(A), A)
    AtA_add_lambdaI = AtA + np.eye(AtA.shape[0]) * lamb
    Atb = product(transpose(A), b)
    inv = inverse(AtA_add_lambdaI)
    x = product(inv, Atb)
    error = product(A, x) - b
    error = product(transpose(error), error)
    show_result(x, error, 'LSE')
    return x

def Newton(A, b, iter=10):
    AtA = product(transpose(A), A)
    AtA_inv = inverse(AtA)
    Atb = product(transpose(A), b)

    np.random.seed(0)
    x = np.random.rand(A.shape[1],1)
    for i in range(iter):
        gradient = product(AtA, x) - Atb
        step = product(AtA_inv, gradient)
        x = x - step
    error = product(A, x) - b
    error = product(transpose(error), error)
    show_result(x, error, 'Newton')
    return x

def get_array_from_file(input, n):
    A = []
    b = []
    dots = []
    f = open(input, "r")
    for line in f.readlines():
        split_line = line.strip().split(',')
        tmp = []
        for i in range(n):
            tmp.append(float(split_line[0])**(n-1-i))
        A.append(tmp)
        b.append(float(split_line[1]))
        dots.append(float(split_line[0]))
    A = np.asarray(A)
    b = np.asarray([b])
    # size of b would be [X,1], X = number of dots
    b = transpose(b)
    return A, b, dots

def show_plot(b, dots, formula_LSE, formula_Newton, n):
    x = np.arange(int(min(dots)-1), int(max(dots)+2))
    s_LSE=[0]
    s_Newton=[0]

    # formula to points
    for i in range(n):
        s_LSE += formula_LSE[i]*(x**(n-i-1))
        s_Newton += formula_Newton[i]*(x**(n-i-1))
    
    plt.figure('HW1')

    plt.subplot(211)
    plt.scatter(dots, b, c='r', edgecolors='k')
    plt.plot(x,s_LSE, c='k')
    # plt.xlim(min(x),max(x))
    
    plt.subplot(212)
    plt.scatter(dots, b, c='r', edgecolors='k')
    plt.plot(x, s_Newton, c='k')
    # plt.xlim(min(x),max(x))
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--lamb', type=float, default=0)
    parser.add_argument('--input', type=str, default='input.txt')
    parser.add_argument('--iter', type=int, default=10)
    args = parser.parse_args()

    # dot for visualization
    A, b, dots = get_array_from_file(args.input, args.n)
    formula_LSE = LSE(A, b, args.lamb)
    formula_Newton = Newton(A, b, args.iter)
    show_plot(b, dots, formula_LSE, formula_Newton, args.n)