import argparse
import numpy as np


def univariate_gaussian_data_generator(m, s):
    # central limit theorem
    # X = μ + σZ
    # generate 12 uniform U(0,1) deviates, add them all up, and subtract 6 – the resulting random variable will have approximately standard normal distribution.
    return m + s**0.5 * (sum(np.random.uniform(0, 1, 12)) - 6)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=float, default=3.0)
    parser.add_argument('--s', type=float, default=5.0)
    args = parser.parse_args()
    
    m_ = args.m
    s_ = args.s

    print(f'Data point source function: N({m_}, {s_})')

    n = 0
    m = 0
    m_last = 0
    s = 0
    s_last = 0

    while True:
        # Welford's online algorithm
        n += 1
        random_data = univariate_gaussian_data_generator(m_, s_)
        print(f'Add data point: {random_data}')
        delta = random_data - m_last
        m = ((n - 1) * m_last + random_data) / n
        # m += delta / n
        delta2 = random_data - m
        s += delta * delta2
        print(f'Mean = {m}\tVariance = {s/n}')
        if ((abs(m - m_last) < 1e-3) and (abs((s/n) - (s_last/(n-1))) < 1e-3)):
            break
        m_last = m
        s_last = s
