#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:05:14 2019

@author: Homai
"""

import sys
import scipy.io
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from qpSVMsolver import Report_QP_Solution
from scipy.io import loadmat

def DesiredOutput(xx): # input of this function is the input pattern
    if (xx[1] < (1/5)*np.sin(10*xx[0])+0.3) | (((xx[1]-0.8)**2 + (xx[0]-0.5)**2) < (0.15)**2):
        return 1
    else:
        return -1

def GetDesiredOutput(inpuT):
    d_i = np.zeros(len(inpuT[0]))
    C_1 =[]; C_2=[]
    for i in range(len(inpuT[0])):
       d_i[i] = DesiredOutput(inpuT[:,i])
       if d_i[i] ==1:
           C_1.append(inpuT[:,i])
       elif d_i[i] ==-1:
           C_2.append(inpuT[:,i])
    return d_i, C_1, C_2
    
def GetListCol(List, iCol):
    return [x[iCol] for x in List]

def Inpute(Num):
    X = np.random.uniform(0,1,Num)
    Y = np.random.uniform(0,1,Num)
    Z = np.asanyarray(np.c_[X,Y])
    return np.transpose(Z)

# Different types of kernels
def PolyKernel(Xi, Xj): ## Polynomial kernel
    return (1 + Xi.dot(Xj))**2
  
def GaussianKernel(Xi, Xj, varX): ## Polynomial kernel
    return np.exp(-1 * ((LA.norm(Xi-Xj))**2/varX))

def KernelMatrix(InputPts, Xvar):
    N = len(InputPts[0]); K = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            K[i,j] = GaussianKernel(InputPts[:,i], InputPts[:,j], Xvar) ## K(matrix): Gaussian kernel
#            K[i,j] = PolyKernel(InputPts[:,i], InputPts[:,j]) ## K(matrix): polynomial kernel
    return K    

## this function calculates the Kernel values for the support vectors    
def KernelTest(XX_S, xi, VarianceX):
    N = len(XX_S[0])
    K = np.zeros(N)
    for i in range(N):
        K[i] = GaussianKernel(XX_S[:,i], xi, VarianceX)
    return K    
    
def CalculateBias(S, D, AlphA, iNput, XVar):
    dS = D[S[0]]
    term1 = (D * AlphA).T
    term2 = KernelTest(iNput, iNput[:,S[0]], XVar)
    return dS - np.dot(term1,term2) ## this is the bias (intercept of the decision boundary)

def DecisionBoundary(inpuT, d, alph, Idx_sv, thresh, varX): # Idx_sv: indices of support vectors
    X1 = np.arange(0,1,0.001)
    X2 = np.arange(0,1,0.001)   
    pts = np.zeros(2); g=0;
    halfSpace = []; C1Coord = []; C2Coord=[]
    gMx = np.zeros((len(X1),len(X2)))
    
    b = CalculateBias(Idx_sv, d, alph, inpuT, varX)
    for i in range(len(X1)):
        for j in range(len(X2)):
            pts[0]= X1[i]; pts[1]= X2[j]
            g = np.dot((d * alph).T, KernelTest(inpuT, pts, varX)) + b ## Calls Gaussian kernel
            gMx[i,j] = g
            if abs(0-g) < thresh:
                print('='*40); print('g for a point on the half scpae: ', g)
                halfSpace.append([pts[0], pts[1],float(g)])
            elif abs(1-g) < thresh:
                print('='*40); print('g for a point on C1: ', g)
                C1Coord.append([pts[0], pts[1],float(g)])
            elif abs(-1-g) < thresh:
                print('='*40); print('g for a point on C2: ', g)
                C2Coord.append([pts[0], pts[1],float(g)])                
    return halfSpace, C1Coord, C2Coord, gMx, X1, X2
    
def plotter(C1, C2, Input_num): ## C1 and C2 are two different classes of the data
    X_c1 = GetListCol(C1,0); Y_c1 = GetListCol(C1,1)
    X_c2 = GetListCol(C2,0); Y_c2 = GetListCol(C2,1)
    plt.scatter(X_c1, Y_c1, marker='o',c='green')
    plt.scatter(X_c2, Y_c2, marker= 'D', c='purple')
    plt.xlabel('X coordinate', fontweight='bold',fontsize=12)
    plt.ylabel('Y coordinate', fontweight='bold',fontsize=12)
#    plt.title('Kernel SVM classification (RBF)--'+str(Input_num)+' points', fontweight='bold',fontsize=14)    
    plt.title('PTA classification (20 Clusters (10 per class))', fontweight = 'bold', fontsize = 14)
    
    
def PlotSuppVectors(InputX, dd, Idx_SupVectors):
    SuppVects = InputX[:,Idx_SupVectors]; SuppVects=SuppVects.T
    SuppVects_labels = dd[Idx_SupVectors]
    suppV_info = np.c_[SuppVects, SuppVects_labels]
    SuppV_C1 = suppV_info[suppV_info[:,2]==1]
    SuppV_C2 = suppV_info[suppV_info[:,2]==-1]
    plt.scatter(SuppV_C1[:,0], SuppV_C1[:,1], marker='X', c='green')
    plt.scatter(SuppV_C1[:,0], SuppV_C1[:,1], s=140, facecolors='none', edgecolors='green')
    plt.scatter(SuppV_C2[:,0], SuppV_C2[:,1], marker= 'D', c='purple')
    plt.scatter(SuppV_C2[:,0], SuppV_C2[:,1], s=140, facecolors='none', edgecolors='purple')
    
def DecisionBoundaryPlotter(XYpoints, Zmatrix, col):
    XX = [x[0] for x in XYpoints]
    YY = [y[1] for y in XYpoints]
    plt.scatter(XX, YY, s= 20, c=col)
#    plt.contour(Gmatx, 16,colors='k')
#    plt.contour(XX, YY, Zmatrix, levels=[-1, 0, 1])
    
def PlotContourOfBoundary(X1, X2, Z):    
#    levels = [-1,0,1]
    levels = [0]
    cs = plt.contour(X1,X2,Z.T, levels)
    plt.clabel(cs,inline=2,fontsize=13)
    
def ThreeDPlot(List, markr, colur, AX):    
    X = [x[0] for x in List]
    Y = [y[1] for y in List]
    Z = [z[2] for z in List]
    AX.scatter(X, Y, Z, marker = markr, c = colur)
    AX.set_title('Decision boundaries in 3D', fontweight = 'bold', fontsize = 14)
    AX.set_xlabel('X values', fontweight = 'bold', fontsize = 13); AX.set_ylabel('Y values', fontweight = 'bold', fontsize = 13)
    AX.set_zlabel('g(x,y) values', fontweight = 'bold', fontsize = 13)
    
  
    
NumberX = 80
X = Inpute(NumberX)    
print('-'*30);print('Number of input data is = '+ str(len(X[0])));print('-'*30)

[d, C1, C2] = GetDesiredOutput(X)
d = np.reshape(d,(len(d),1))

varianceX = 1 #np.var(X)
kernel = KernelMatrix(X, varianceX)
[Alphas, Is] = Report_QP_Solution(kernel, d)

## This function solves the optimization problem and 
[HPCoord, C1Boundry, C2Boundry, matrixG, Xtest1, Xtest2] = DecisionBoundary(X, d, Alphas, Is, 1e-3, varianceX)

fig = plt.figure(figsize=(8,8))
plotter(C1, C2,NumberX)
DecisionBoundaryPlotter(HPCoord,matrixG, 'blue') # To expedite the run time: you can make the X1, X2 in this function coarser
DecisionBoundaryPlotter(C1Boundry,matrixG, 'red')
DecisionBoundaryPlotter(C2Boundry,matrixG, 'black')
PlotSuppVectors(X, d, Is)


leg=[]; leg.append('Class C1'); leg.append('Class C2')
#leg.append('{x:g(x)=0}');leg.append('{x:g(x)=1}');
#leg.append('{x:g(x)=-1}');
plt.legend(leg, loc ='upper right',fontsize=12)



fig = plt.figure(figsize=(8,8))
plotter(C1, C2,NumberX)
PlotSuppVectors(X, d, Is)
PlotContourOfBoundary(Xtest1, Xtest2, matrixG)
leg=[]; leg.append('Class C1'); leg.append('Class C2')
plt.legend(leg, loc ='upper right',fontsize=12)


