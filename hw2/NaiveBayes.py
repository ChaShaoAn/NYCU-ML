import os
import argparse
import struct
import numpy as np
from tqdm import tqdm


def load_mnist(path, tag='train'):
    label_path = os.path.join(path, '%s-labels.idx1-ubyte'%tag)
    image_path = os.path.join(path, '%s-images.idx3-ubyte'%tag)
    with open(label_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(image_path, 'rb') as imgpath:
        magic, n, row, col = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), row*col)
    return labels, images


def discreteMode(class_num=10):
    train_lbs, train_imgs =  load_mnist('./data/', tag='train')
    test_lbs, test_imgs = load_mnist('./data/', tag='t10k')

    pixels = train_imgs[0].shape[0]
    train_num = train_lbs.shape[0]
    test_num = test_lbs.shape[0]
    prior = np.zeros((class_num), dtype=float)
    likelihood = np.zeros((class_num, pixels, 32), dtype=float)

    print('start training:')
    for i in tqdm(range(train_num)):
        prior[train_lbs[i]] += 1
        for j in (range(pixels)):
            likelihood[train_lbs[i]][j][train_imgs[i][j]//8] += 1
    prior/= train_num

    likelihood_sum = np.sum(likelihood, axis=2)

    for i in range(class_num):
        for j in range(pixels):
            likelihood[i][j][:] = likelihood[i][j][:] / likelihood_sum[i][j]
            for k in range(32):
                if (likelihood[i][j][k] == 0.):
                    # np.nonzero回傳陣列中非0的index
                    likelihood[i][j][k] = np.min(likelihood[i][j][np.nonzero(likelihood[i][j])])
                    # likelihood[i][j][k] = 0.0001

    error = 0

    print('start testing:')
    f = open('discrete-output.txt', 'w')
    for i in tqdm(range(test_num)):
        prob = np.zeros(10, dtype=float)
        # P(D|θ) * P(θ), 其中P(θ)為prior
        for j in range(class_num):
            # P(θ)
            prob[j] += np.log(prior[j])
            # P(D|θ), 且假設都為independent
            for k in range(pixels):
                prob[j] += np.log(likelihood[j][k][test_imgs[i][k]//8])
        prob = prob / sum(prob)
        print('Posterior (in log scale):', file=f)
        for j in range(prob.shape[0]):
            print(f'{j}: {prob[j]}', file=f)
        pred = np.argmin(prob)
        print(f'Prediction: {pred}, Ans: {test_lbs[i]}\n', file=f)
        if (pred != test_lbs[i]):
            error += 1

    error =float(error)/test_num

    print('Imagination of numbers in Bayesian classifier:', file=f)
    img = np.zeros((class_num, pixels))
    zero = np.sum(likelihood[:,:,0:16], axis=2)
    one = np.sum(likelihood[:,:,16:32], axis=2)
    # img = (one >= zero) * 1
    img = (one > zero) * 1    
    for c in range(class_num):
        print('\n{}:'.format(c), file=f)
        for row in range (28):
            for col in range(28):
                print(img[c][row*28+col], end=' ', file=f)
            print(' ', file=f)

    print(f'\nError rate:{error}', file=f)
    f.close()


def continuousMode(class_num=10):
    train_lbs, train_imgs =  load_mnist('./data/', tag='train')
    test_lbs, test_imgs = load_mnist('./data/', tag='t10k')
    pixels = train_imgs[0].shape[0]
    train_num = train_lbs.shape[0]
    test_num = test_lbs.shape[0]
    prior = np.zeros((class_num), dtype=float)
    # E(x)
    mean = np.zeros((class_num, pixels), dtype=float)
    # E(x^2)
    mean_of_square_x = np.zeros((class_num, pixels), dtype=float)
    # var = E(x^2) - E(x)^2 <= 平方的期望值 - 期望值的平方 = σ^2
    var = np.zeros((class_num, pixels), dtype=float) 

    print('start training:')
    for i in tqdm(range(train_num)):
        prior[train_lbs[i]] += 1
        for j in (range(pixels)):
            mean[train_lbs[i]][j] += train_imgs[i][j]
            mean_of_square_x[train_lbs[i]][j] += train_imgs[i][j]**2
    
    for i in range(class_num):
        for j in range(pixels):
            mean[i][j] /= prior[i]
            mean_of_square_x[i][j] /= prior[i]
            # var = E(x^2) - E(x)^2 = σ^2 <= 平方的期望值 - 期望值的平方
            var[i][j] = mean_of_square_x[i][j] - mean[i][j]**2
    
    print('start testing:')
    # prior在這轉成機率，前面因為要用到數量，所以沒有先除train_num
    prior /= train_num
    f = open('continuous-output.txt', 'w')
    error = 0
    for i in tqdm(range(test_num)):
        prob = np.zeros((class_num), dtype=float)
        # P(D|θ) * P(θ), 其中P(θ)為prior
        for j in range(class_num):
            # P(θ)
            prob[j] += np.log(prior[j])
            # P(D|θ)
            for k in range(pixels):
                # 若為0，就只有一種可能
                if(var[j][k] == 0):
                    continue
                # P(x) = (1/2σ^2𝝅)^1/2 * e^((-1/σ^2)(x-μ)^2),
                # 只是在這已經取ln, 變成 -1/2 * (ln(2*σ^2*𝝅)) + (-1/2*σ^2) * (x-μ)^2
                prob[j] += -0.5 * np.log(2. * var[j][k] * np.pi)
                prob[j] += -((test_imgs[i][k] - mean[j][k])**2) / (2.0 * var[j][k])
        prob /= sum(prob)
        print('Posterior (in log scale):', file=f)
        for j in range(prob.shape[0]):
            print(f'{j}: {prob[j]}', file=f)
        pred = np.argmin(prob)
        print(f'Prediction: {pred}, Ans: {test_lbs[i]}\n', file=f)
        if (pred != test_lbs[i]):
            error += 1
    
    error =float(error)/test_num

    print('Imagination of numbers in Bayesian classifier:', file=f)
    img = np.zeros((class_num, pixels))
    img = (mean >= 128) * 1    
    for c in range(class_num):
        print('\n{}:'.format(c), file=f)
        for row in range (28):
            for col in range(28):
                print(img[c][row*28+col], end=' ', file=f)
            print(' ', file=f)

    print(f'\nError rate:{error}', file=f)
    f.close()
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=0)
    args = parser.parse_args()

    class_num = 10

    if args.mode == 0:
        print('-discrete mode')
        discreteMode(class_num)
    else:
        print('-continuous mode')
        continuousMode(class_num)

