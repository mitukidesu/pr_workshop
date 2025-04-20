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
from scipy.optimize import minimize
from scipy.special import gammaln

import simulation

w, y, bin, X, zero = simulation.generate_data()


def mle_poisson(beta, X, bin, y):
    pred_mu = np.exp( beta[0] +  beta[1]*X + beta[2]*bin )
    likelihood_func = np.sum(y * np.log(pred_mu) - pred_mu - gammaln(y + 1))
    return -likelihood_func

def mle_negative_binomial(beta, X, bin, y):
    pred_mu = np.exp(beta[0] + beta[1] * X + beta[2] * bin)
    theta = beta[3]

    likelihood_func = np.sum(
        gammaln(y + theta) - gammaln(theta) - gammaln(y + 1) + 
        y * np.log(pred_mu / (pred_mu + theta)) + 
        theta * np.log(theta / (pred_mu + theta))
    )
    return -likelihood_func

def calc_res():
    init_pars_poi = np.array([1,1,1])
    res_poi = minimize(mle_poisson, init_pars_poi, args=(X, bin, y), method='BFGS')

    init_pars_NB = np.array([2,2,2,1])  
    res_NB = minimize(mle_negative_binomial, init_pars_NB, args=(X, bin, y), method='BFGS')

    # print("Estimated_pars (β0, β1, β2) by Poisson model:", res_poi.x)
    # print("Estimated parameters (β0, β1, β2, θ) by Negative Binomial model:", res_NB.x)
    return res_poi, res_NB

res_poi, res_NB = calc_res()

