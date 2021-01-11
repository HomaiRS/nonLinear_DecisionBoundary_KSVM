#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:29:32 2019

@author: Homai
"""

#Importing with custom names to avoid issues with numpy / sympy matrix
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

# This function reformulates dual SVM optimization problem to makes it
# compatible with cvxopt.solvers.qp optimization solver    
def ReformulateDool(K, d): # gets kernel matrix and desired outputs
    # makesure the desired output that u parse be an numpy array
    H_Mx = d.dot(d.T)
    H_Mx = H_Mx.dot(K)
    return H_Mx

def ReformulateDeel(X, d): # gets kernel matrix and desired outputs
    # makesure the desired output that u parse be an numpy array
    H_Mx = d.dot(d.T)
    Xdot = X.T.dot(X)
    H_Mx = H_Mx.dot(Xdot)
    return H_Mx

def ReformulateDual(X,d): # gets kernel matrix and desired outputs
    # makesure the desired output that u parse be an numpy array
    d = d.reshape(-1,1) * 1.
    X_dash = d * X.T
    H_Mx = np.dot(X_dash , X_dash.T) * 1.
    return H_Mx

def fit(Kernel, y): 
    NUM = len(y)
    # we'll solve the dual
    # obtain the kernel
    P = cvxopt_matrix(np.outer(y,y) * Kernel)
    Qq = cvxopt_matrix(-np.ones((NUM, 1)))
    G = cvxopt_matrix(-np.eye(NUM))
    h = cvxopt_matrix(np.zeros(NUM))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10
    sol = cvxopt_solvers.qp(P, Qq, G, h, A, b)
    alphas = np.array(sol['x'])
    return sol, alphas

#==================Computing and printing parameters===============================#
def Report_QP_Solution(K, dd):   
    [sol2, Alpha2] = fit(K, dd)
    S = np.argwhere(Alpha2>1e-4)
    S = [elmnt[0] for elmnt in S]
    print('Alphas2 = ',Alpha2[Alpha2 > 1e-4]);  print('-'*30)
    return Alpha2, S


