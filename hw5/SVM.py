import time
import numpy as np
import matplotlib.pyplot as plt
from libsvm.svmutil import *


def loadData(folderpath='data/'):
    train_imgs = []
    train_lbs = []
    test_imgs = []
    test_lbs = []
    with open(folderpath + 'X_train.csv') as f:
        line = f.readline()
        while (line):
            train_imgs.append([float(i) for i in line.strip().split(',')])
            line = f.readline()
    with open(folderpath + 'Y_train.csv') as f:
        line = f.readline()
        while (line):
            train_lbs.append(float(line.strip()))
            line = f.readline()
    with open(folderpath + 'X_test.csv') as f:
        line = f.readline()
        while (line):
            test_imgs.append([float(i) for i in line.strip().split(',')])
            line = f.readline()
    with open(folderpath + 'Y_test.csv') as f:
        line = f.readline()
        while (line):
            test_lbs.append(float(line.strip()))
            line = f.readline()
    return train_imgs, train_lbs, test_imgs, test_lbs


def svm(X, Y, X_test, Y_test, para):
    m =  svm_train(Y, X, para)
    p_labs, p_acc, p_vals = svm_predict(Y_test, X_test, m)
    return p_acc


def compare(X, Y, para, best_acc, best_para):
    acc = svm_train(Y, X, para)
    if (acc > best_acc):
        return acc, para
    return best_acc, best_para

def gridSearch(X, Y, kernelType):
    best_acc = 0
    best_para = f''
    costs = [0.001, 0.01, 0.1, 1, 10]
    # gammas = [0.001, 0.01, 0.1, 1]
    gammas = [1/784, 0.01, 0.1, 1]
    degrees = [2, 3, 4]
    coef0s = [0, 1, 2]
    count = 0
    start = time.time()
    if (kernelType == 0):
        for cost in costs:
            para = f'-t {kernelType} -c {cost} -q -v 3'
            count += 1
            best_acc, best_para = compare(X, Y, para, best_acc, best_para)
    elif (kernelType == 1):
        for cost in costs:
            for gamma in gammas:
                for degree in degrees:
                    for coef0 in coef0s:
                        para = f'-t {kernelType} -c {cost} -g {gamma} -d {degree} -r {coef0} -q -v 3'
                        count += 1
                        best_acc, best_para = compare(X, Y, para, best_acc, best_para)
    elif (kernelType == 2):
        for cost in costs:
            for gamma in gammas:
                para = f'-t {kernelType} -c {cost} -g {gamma} -q -v 3'
                count += 1
                best_acc, best_para = compare(X, Y, para, best_acc, best_para)
    end = time.time()
    print('\n#################################################')
    print(f'Total time: {(end - start):.2f} s')
    print(f'Total combinations: {count}')
    print(f'Optimal cross validation accuracy: {best_acc}')
    print(f'Optimal option: {best_para}')
    print('#################################################\n')
    return best_acc, best_para


def linearKernel(X1, X2):
    return np.dot(X1, X2.T)


def RBFKernel(X1, X2, gamma):
    dist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dist)

if __name__ == '__main__':
    train_imgs, train_lbs, test_imgs, test_lbs = loadData()
    
    task = 1

    if (task == 1):
        print('linear:')
        svm(train_imgs, train_lbs, test_imgs, test_lbs, f'-t 0 -d 2 -q')
        print('polynomial:')
        svm(train_imgs, train_lbs, test_imgs, test_lbs, f'-t 1 -d 2 -q')
        print('radial basis function:')
        svm(train_imgs, train_lbs, test_imgs, test_lbs, f'-t 2 -d 2 -q')
    elif (task == 2):
        best_para = f''
        print('linear:')
        l_acc, l_para = gridSearch(train_imgs, train_lbs, 0)
        print(f'linear cross-valid: acc:{l_acc}, para:{l_para}')
        best_para = l_para
        best_para = best_para.replace(best_para[-5:], '')
        svm(train_imgs, train_lbs, test_imgs, test_lbs, best_para)
        print('polynomial:')
        p_acc, p_para = gridSearch(train_imgs, train_lbs, 1)
        print(f'polynomial cross-valid: acc:{p_acc}, para:{p_para}')
        best_para = p_para
        best_para = best_para.replace(best_para[-5:], '')
        svm(train_imgs, train_lbs, test_imgs, test_lbs, best_para)
        print('radial basis function:')
        r_acc, r_para = gridSearch(train_imgs, train_lbs, 2)
        print(f'RBF cross-valid: acc:{r_acc}, para:{r_para}')
        best_para = r_para
        best_para = best_para.replace(best_para[-5:], '')
        svm(train_imgs, train_lbs, test_imgs, test_lbs, best_para)
    elif (task == 3):
        gamma = 1/len(train_imgs[0])
        imgs1 = np.array(train_imgs)
        imgs2 = np.array(test_imgs)
        print(gamma)
        train_kernel = linearKernel(imgs1, imgs1) + RBFKernel(imgs1, imgs1, gamma)
        test_kernel = linearKernel(imgs2, imgs2) + RBFKernel(imgs2, imgs2, gamma)
        train_kernel = np.hstack((np.arange(1, len(train_lbs)+1).reshape(-1, 1), train_kernel))
        test_kernel = np.hstack((np.arange(1, len(test_lbs)+1).reshape(-1, 1), test_kernel))
        m = svm_train(train_lbs, train_kernel, '-t 4')
        labs, acc, vals = svm_predict(test_lbs, test_kernel, m)