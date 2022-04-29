import math
import numpy as np
import matplotlib.pyplot as plt


def univariate_gaussian_data_generator(m, s):
    # central limit theorem
    # generate 12 uniform U(0,1) deviates, add them all up, and subtract 6 – the resulting random variable will have approximately standard normal distribution.
    return m + s**0.5 * (sum(np.random.uniform(0, 1, 12)) - 6)


def gradient_decent(X, Y, lr=0.01):
    # w = np.random.rand(3, 1)
    w = np.zeros((3, 1))
    conv_count = 0
    count = 0
    while (conv_count < 5):
        w_last = w.copy()
        gradient = X.T.dot(Y - (1 / (1 + np.exp(-1*X.dot(w)))))
        w = w + lr*gradient
        # if(abs(sum(abs(w_last) - abs(w))) < 5e-2):
        if((sum(abs(w_last - w))) < 5e-3):
            conv_count += 1
        else:
            conv_count = 0
        count += 1
    # print(count)
    return w


def Newton(X, Y, N, lr=0.01):
    # w = np.random.rand(3, 1)
    w = np.zeros((3, 1))
    conv_count = 0
    count = 0
    while (conv_count < 5):
        w_last = w.copy()
        gradient = X.T.dot(Y - (1 / (1 + np.exp(-1*X.dot(w)))))
        D = np.zeros((2*N, 2*N))
        for i in range(2*N):
            XW = w[0]*1 + w[1]*X[i, 0] +w[2]*X[i, 1]
            D[i][i] = math.exp(-XW) / ((1 + math.exp(-XW))**2)
        # H = A^tDA
        H = X.T.dot(D).dot(X)
        w = w + lr*np.linalg.inv(H).dot(gradient)
        if((sum(abs(w_last - w))) < 5e-3):
            conv_count += 1
        else:
            conv_count = 0
        count += 1
    # print(count)
    return w


if __name__=='__main__':
    N = 50
    mx1 = my1 = 1
    mx2 = my2 = 3
    vx1 = vy1 = 2
    vx2 = vy2 = 4

    D1 = np.zeros((N, 2))
    D2 = np.zeros((N, 2))
    X = np.zeros((2*N, 3))
    Y = np.zeros((2*N, 1))

    for i in range(N):
        D1[i] = [univariate_gaussian_data_generator(mx1, vx1), univariate_gaussian_data_generator(my1, vy1)]
        D2[i] = [univariate_gaussian_data_generator(mx2, vx2), univariate_gaussian_data_generator(my2, vy2)]
        X[i] = [1., univariate_gaussian_data_generator(mx1, vx1), univariate_gaussian_data_generator(my1, vy1)]
        X[i + N] = [1., univariate_gaussian_data_generator(mx2, vx2), univariate_gaussian_data_generator(my2, vy2)]
        Y[i +N] = 1

    plt.subplot(131)
    plt.title('Ground Truth')
    plt.scatter(D1[:, 0], D1[:, 1], c='r')
    plt.scatter(D2[:, 0], D2[:, 1], c='b')

    w = gradient_decent(X, Y)

    plt.subplot(132)
    plt.title('Gradient Descent')
    TP = FP = TN = FN = 0
    for i in range(N):
        x = w[0]*1 + w[1]*D1[i, 0] +w[2]*D1[i, 1]
        if (1/(math.exp(-x)) > 0.5):
            plt.plot(D1[i, 0], D1[i, 1], 'o', c='b')
            FP += 1
        else:
            plt.plot(D1[i, 0], D1[i, 1], 'o', c='r')
            TN += 1
        
        x = w[0]*1 + w[1]*D2[i, 0] +w[2]*D2[i, 1]
        if (1/(math.exp(-x)) > 0.5):
            plt.plot(D2[i, 0], D2[i, 1], 'o', c='b')
            TP += 1
        else:
            plt.plot(D2[i, 0], D2[i, 1], 'o', c='r')
            FN += 1
    
    print('Gradient Descent:\n')
    print('w:')
    print(w)
    print('\nConfusion Matrix:')
    print("\t\tPredict cluster 1\tPredict cluster 2")
    print(f"In cluster 1\t\t{TP}\t\t\t{FN}")
    print(f"In cluster 2\t\t{FP}\t\t\t{TN}")
    print("\nSensitivity (Successfully predict cluster 1):", TP / (TP + FN))
    print("Specificity (Successfully predict cluster 2):", TN / (TN + FP))
    
    w = Newton(X, Y, N)

    plt.subplot(133)
    plt.title('Newton\'s Method')
    TP = FP = TN = FN = 0
    for i in range(N):
        x = w[0]*1 + w[1]*D1[i, 0] +w[2]*D1[i, 1]
        if (1/(math.exp(-x)) > 0.5):
            plt.plot(D1[i, 0], D1[i, 1], 'o', c='b')
            FP += 1
        else:
            plt.plot(D1[i, 0], D1[i, 1], 'o', c='r')
            TN += 1
        
        x = w[0]*1 + w[1]*D2[i, 0] +w[2]*D2[i, 1]
        if (1/(math.exp(-x)) > 0.5):
            plt.plot(D2[i, 0], D2[i, 1], 'o', c='b')
            TP += 1
        else:
            plt.plot(D2[i, 0], D2[i, 1], 'o', c='r')
            FN += 1
    
    print('\n---------------------------------------------------------\n')
    print('Newton\'s Method\n')
    print('w:')
    print(w)
    print('\nConfusion Matrix:')
    print("\t\tPredict cluster 1\tPredict cluster 2")
    print(f"In cluster 1\t\t{TP}\t\t\t{FN}")
    print(f"In cluster 2\t\t{FP}\t\t\t{TN}")
    print("\nSensitivity (Successfully predict cluster 1):", TP / (TP + FN))
    print("Specificity (Successfully predict cluster 2):", TN / (TN + FP))
    plt.tight_layout()
    plt.show()
