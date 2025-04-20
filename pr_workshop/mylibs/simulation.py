#All libraries here
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import re
import scipy as sp
from scipy import stats
import warnings
import sys
from pathlib import Path
import os
import random
from scipy.stats import poisson

def generate_data():
    N = 10**4

    b_list = [0]*(int(N/2)) + [1]*(int(N/2))
    random.shuffle(b_list)
    bin = np.array(b_list)
    num = np.random.uniform(0, 1, N)
    X = 5.5*((num)**2) - 2.5
    mu = np.exp(X+3-1.5*(bin))
    w = np.random.negative_binomial(mu, 0.6 , N)

    alpha = -(3*X)-5+(2.5*(bin))
    zero_prob = 1/(1+np.exp(-(alpha)))
    zero = zero_prob > np.random.rand(N)

    y = w.copy()
    y[zero] = 0

    # plt.hist(w, bins=np.arange(min(w), min(w) + 31, 1), density=True, color='orange')
    # plt.hist(y, bins=np.arange(min(y), min(y) + 31, 1), density=True, color='green')
    # plt.xticks(np.arange(min(w), min(w) + 31, 1), rotation=45)
    # plt.show()

    return w, y, bin, X, zero

