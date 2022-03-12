import argparse
import numpy as np
import math


def C(N, m):
    # 少乘一些
    if (m > (N - m)):
        m = N - m
    c1, c2 = 1, 1
    while (m):
        c1 *= N
        N -= 1
        c2 *= m
        m -= 1
    return c1/c2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=int, default=0)
    parser.add_argument('--b', type=int, default=0)
    parser.add_argument('--input', type=str, default='input.txt')
    args = parser.parse_args()

    outcomes = []
    with open(args.input, 'r') as f:
        lines = f.readlines()
        for line in lines:
            outcomes.append(line.strip())

    case = 0
    a = args.a
    b = args.b

    for outcome in outcomes:
        case += 1
        N = len(outcome)
        m = 0
        for char in outcome:
            if char == '1':
                m += 1
        P = m/N
        likelihood = C(N, m) * (P ** m) * ((1 - P) **  (N - m))
        print(f'case {case}: {outcome}')
        print(f'Likelihood: {likelihood}')
        print(f'Beta prior:\ta = {a}\tb = {b}')
        a += m
        b += (N - m)
        print(f'Beta posterior:\ta = {a}\tb = {b}\n')