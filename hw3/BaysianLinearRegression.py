import numpy as np
import matplotlib.pyplot as plt
import time


def univariate_gaussian_data_generator(m, s):
    # central limit theorem
    # generate 12 uniform U(0,1) deviates, add them all up, and subtract 6 – the resulting random variable will have approximately standard normal distribution.
    return m + s**0.5 * (sum(np.random.uniform(0, 1, 12)) - 6)


def Polynomial_basis_linear_model_generator(n, a, w):
    assert(n == len(w))
    x = np.random.uniform(-1.0, 1.0)
    y = 0.
    for i in range(n):
        y += w[i] * (x**i)
    y += univariate_gaussian_data_generator(0, a)
    return x, y


def designMatrix(x, n):
    dM = np.zeros((1, n))
    for i in range(n):
        dM[0][i] = x**i
    return dM


def paint(x, y, var):
	plt.xlim(-2.0, 2.0)
	plt.ylim(-25, 25)
	plt.plot(x, y, color = 'black')
	plt.plot(x, y+var, color = 'red')
	plt.plot(x, y-var, color = 'red')


def showResults(n, a, w, X, Y, mean, cov_inv, ten_mean, ten_cov_inv, fifty_mean, fifty_cov_inv):
	x = np.linspace(-2.0, 2.0, 100)

	#GROUND TRUTH
	plt.subplot(221)
	func = np.poly1d(np.flip(w))
	y = func(x)
	var = (1/a)
	plt.title("Ground truth")
	paint(x, y, var)
	
	#ALL
	plt.subplot(222)
	func = np.poly1d(np.flip(np.reshape(mean, n)))
	y = func(x)
	var = np.zeros((100))
	for i in range(100):
		dx = designMatrix(x[i], n)
		var[i] = 1/a +  dx.dot(cov_inv.dot(dx.T))[0][0]
	plt.title("Predict result")
	plt.scatter(X, Y, s=7)
	paint(x, y, var)

	#10 datas
	plt.subplot(223)
	func = np.poly1d(np.flip(np.reshape(ten_mean, n)))
	y = func(x)
	var = np.zeros((100))
	for i in range(100):
		dx = designMatrix(x[i], n)
		var[i] = 1/a +  dx.dot(ten_cov_inv.dot(dx.T))[0][0]
	plt.title("After 10 incomes")
	plt.scatter(X[:10], Y[:10], s=7)
	paint(x, y, var)

	#50 datas
	plt.subplot(224)
	func = np.poly1d(np.flip(np.reshape(fifty_mean, n)))
	y = func(x)
	var = np.zeros((100))
	for i in range(100):
		dx = designMatrix(x[i], n)
		var[i] = 1/a +  dx.dot(fifty_cov_inv.dot(dx.T))[0][0]
	plt.title("After 50 incomes")
	plt.scatter(X[:50], Y[:50], s=7)
	paint(x, y, var)

	plt.tight_layout()
	plt.show()


def BayesianLinearRegression(b, n, a, w):
    X = []
    Y = []
    a = 1/a
    count = 0
    con_count = 0
    prior_m = np.zeros((n, 1))
    prior_cov = np.eye(n) * b # bI
    while True:
        x, y = Polynomial_basis_linear_model_generator(n, 1/a, w)
        X.append(x)
        Y.append(y)
        # [1 x^1 x^2 x^3]
        dx = designMatrix(x, n)
        count += 1

        # P(W|D)
        post_cov = a * dx.T.dot(dx) + prior_cov # Λ = aX^TX+Λ, for first time: Λ = aX^TX+bI
        post_m = np.linalg.inv(post_cov).dot(a * dx.T.dot([[y]]) + prior_cov.dot(prior_m))  # μ = C^-1(aX^TY+Λμ), 第一次的時候為 μ = aΛ^-1X^TY，但其實也沒差，因為C^-1Λμ是0

        # P(Y|D) = N(μ^t*X^T, 1/a + X*Λ^-1*X^T)
        new_m = post_m.T.dot(dx.T)
        new_cov = 1/a + (dx.dot(np.linalg.inv(post_cov))).dot(dx.T)

        print(f'Add Data point ({x}, {y}):\n')
        print('Posterior mean:')
        print(post_m)
        print('\nPosterior variance:')
        print(np.linalg.inv(post_cov))
        print("\nPredictive distribution ~ N(%.5f, %.5f)\n" % (new_m, new_cov))

        if (count == 10):
            ten_mean = post_m.copy()
            ten_cov_inv = np.linalg.inv(post_cov).copy()
        if (count == 50):
            fifty_mean = post_m.copy()
            fifty_cov_inv = np.linalg.inv(post_cov).copy()

        flag = False
        for t in range(n):
            if abs(post_m[t] - prior_m[t]) >= 1e-3:
                flag = True
        if((count >= 1000) and (flag == False)):
            break
        prior_cov = post_cov
        prior_m = post_m
    showResults(n, a, w, X, Y, post_m, np.linalg.inv(post_cov), ten_mean, ten_cov_inv, fifty_mean, fifty_cov_inv)


if __name__=='__main__':
    b = 1
    n = 4
    a = 1
    w = [1, 2, 3, 4]
    # w = [1, 2, 3]
    BayesianLinearRegression(b, n, a, w)
