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
from scipy.optimize import minimize
from scipy.special import gammaln
import statistics
from scipy.stats import nbinom

import simulation, model

w, y, bin, X, zero = simulation.generate_data()
res_poi, res_NB = model.calc_res()

def predict_means_poi(N=10000, seed=42):
    np.random.seed(seed)
    u = np.random.uniform(0, 1, N)
    x = 5.5 * u**2 - 2.5

    #Criteria A
    mu_A = np.exp(res_poi.x[0] + res_poi.x[1] * x)
    w_A = np.random.poisson(mu_A)
    p_A = 1 / (1 + np.exp(-(3 * x + 5)))  # I_B = 0
    y_A = np.where(np.random.rand(N) < p_A, 0, w_A)

    #Criteria B
    mu_B = np.exp(res_poi.x[0] + res_poi.x[1] * x + res_poi.x[2])
    w_B = np.random.poisson(mu_B)
    p_B = 1 / (1 + np.exp(-(3 * x + 5 - 2.5)))  # I_B = 1
    y_B = np.where(np.random.rand(N) < p_B, 0, w_B)

    print("Average:")
    print(f"  E[w_A]: {np.mean(w_A):.4f}")
    print(f"  E[w_B]: {np.mean(w_B):.4f}")
    print(f"  E[y_A]: {np.mean(y_A):.4f}")
    print(f"  E[y_B]: {np.mean(y_B):.4f}")

    mean_w_A_poi = np.mean(w_A)
    mean_w_B_poi = np.mean(w_B)
    mean_y_A_poi = np.mean(y_A)
    mean_y_B_poi = np.mean(y_B)
    return mean_w_A_poi, mean_w_B_poi, mean_y_A_poi, mean_y_B_poi

mean_w_A_poi, mean_w_B_poi, mean_y_A_poi, mean_y_B_poi = predict_means_poi(N=10000, seed=42)

def predict_means_nb(N=10000):
    u = np.random.uniform(0, 1, N)
    x = 5.5 * (u**2) - 2.5

    mu_A = np.exp(res_NB.x[0] + res_NB.x[1] * x)
    mu_B = np.exp(res_NB.x[0] + res_NB.x[1] * x + res_NB.x[2])

    p_A = res_NB.x[-1] / (mu_A + res_NB.x[-1])
    p_B = res_NB.x[-1] / (mu_B + res_NB.x[-1])

    w_A = nbinom.rvs(n=res_NB.x[-1], p=p_A)
    w_B = nbinom.rvs(n=res_NB.x[-1], p=p_B)

    pi_A = 1 / (1 + np.exp(-(-3 * x - 5)))
    pi_B = 1 / (1 + np.exp(-(-3 * x - 5 + 2.5)))

    mask_A = np.random.rand(N) < pi_A
    mask_B = np.random.rand(N) < pi_B

    y_A = w_A.copy()
    y_B = w_B.copy()
    y_A[mask_A] = 0
    y_B[mask_B] = 0

    print("Average:")
    print(f"  E[w_A]: {np.mean(w_A):.4f}")
    print(f"  E[w_B]: {np.mean(w_B):.4f}")
    print(f"  E[y_A]: {np.mean(y_A):.4f}")
    print(f"  E[y_B]: {np.mean(y_B):.4f}")

    mean_w_A_nb = np.mean(w_A)
    mean_w_B_nb = np.mean(w_B)
    mean_y_A_nb = np.mean(y_A)
    mean_y_B_nb = np.mean(y_B)
    return mean_w_A_nb, mean_w_B_nb, mean_y_A_nb, mean_y_B_nb

mean_w_A_nb, mean_w_B_nb, mean_y_A_nb, mean_y_B_nb = predict_means_nb()

def create_graph_1():
    plt.figure(figsize=(6, 4))

    plt.scatter("A", mean_w_A_poi, color="blue", s=100, label="Poisson: $w_A$")
    plt.scatter("B", mean_w_B_poi, color="blue", s=100, label="Poisson: $w_B$")

    plt.scatter("A", mean_w_A_nb, color="green", s=100, marker="x", label="NegBin: $w_A$")
    plt.scatter("B", mean_w_B_nb, color="green", s=100, marker="x", label="NegBin: $w_B$")

    plt.title("Data: w")
    plt.ylabel("Average fitted values")
    plt.grid(True)
    plt.legend()
    plt.show()

create_graph_1()

def create_graph_2():
    plt.figure(figsize=(6, 4))

    plt.scatter("A", mean_y_A_poi, color="blue", s=100, label="Poisson: $y_A$")
    plt.scatter("B", mean_y_B_poi, color="blue", s=100, label="Poisson: $y_B$")

    plt.scatter("A", mean_y_A_nb, color="green", s=100, marker="x", label="NegBin: $y_A$")
    plt.scatter("B", mean_y_B_nb, color="green", s=100, marker="x", label="NegBin: $y_B$")

    plt.title("Data: y")
    plt.ylabel("Average fitted values")
    plt.grid(True)
    plt.legend()
    plt.show()

create_graph_2()