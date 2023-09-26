import math
import numpy as np
import mpmath

if __name__ == '__main__':
    n = 8 * 10**12
    m = 2 ** 3000000
    logn = mpmath.log(n)
    logm = mpmath.log(m)
    lognm = logn - logm
    antilog = mpmath.exp(lognm)
    print(antilog)
    # 8.24324187856333e-903078