#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:46:37 2017

@author: aatina
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc.pilutil import imread, imsave

def add_gaussian_noise(sim,prop,varSigma):
   N = int(np.round(np.prod(im.shape)*prop))
   index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
   e = varSigma*np.random.randn(np.prod(im.shape)).reshape(im.shape)
   im2 = np.copy(im)
   im2[index] += e[index]
   return im2

def add_saltnpeppar_noise(im,prop):
   N = int(np.round(np.prod(im.shape)*prop))
   index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
   im2 = np.copy(im)
   im2[index] = 1-im2[index]
   return im2

def neighbours(i,j,M,N,size=8):
    if size==4:
        if (i==0 and j==0):
            n=[(0,1), (1,0)]
        elif i==0 and j==N-1:
            n=[(0,N-2), (1,N-1)]
        elif i==M-1 and j==0:
            n=[(M-1,1), (M-2,0)]
        elif i==M-1 and j==N-1:
            n=[(M-1,N-2), (M-2,N-1)]
        elif i==0:
            n=[(0,j-1), (0,j+1), (1,j)]
        elif i==M-1:
            n=[(M-1,j-1), (M-1,j+1), (M-2,j)]
        elif j==0:
            n=[(i-1,0), (i+1,0), (i,1)]
        elif j==N-1:
            n=[(i-1,N-1), (i+1,N-1), (i,N-2)]
        else:
            n=[(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
        return n
    if size==8:
        if (i==0 and j==0):
            n=[(0,1), (1,0), (1,1)]
        elif i==0 and j==N-1:
            n=[(0,N-2), (1,N-1), (1,N-2)]
        elif i==M-1 and j==0:
            n=[(M-1,1), (M-2,0),(M-2,1)]
        elif i==M-1 and j==N-1:
            n=[(M-1,N-2), (M-2,N-1),(M-2,N-2)]
        elif i==0:
            n=[(0,j-1), (0,j+1), (1,j), (1,j-1),(1,j+1)]
        elif i==M-1:
            n=[(M-1,j-1), (M-1,j+1), (M-2,j),(M-2,j-1),(M-2,j+1)]
        elif j==0:
            n=[(i-1,0), (i+1,0), (i,1),(i-1, 1), (i+1,1)]
        elif j==N-1:
            n=[(i-1,N-1), (i+1,N-1), (i,N-2), (i-1,N-2), (i+1,N-2)]
        else:
            n=[(i-1,j), (i+1,j), (i,j-1), (i,j+1), (i+1, j+1), (i+1,j-1),(i-1,j+1), (i-1,j-1)]
        return n

# proportion of pixels to alter
prop = 0.7
varSigma = 0.1
im = imread("cat_gray_32.jpg")
im = im/255
fig = plt.figure()
ax = fig.add_subplot(131)
ax.imshow(im,cmap="gray")
im2 = add_gaussian_noise(im,prop,varSigma)
ax2 = fig.add_subplot(132)
ax2.imshow(im2,cmap="gray")
im2 = add_saltnpeppar_noise(im,prop)
ax3 = fig.add_subplot(133)
ax3.imshow(im2,cmap="gray")

#x = np.array([[-1,-1,1],[1,1,-1],[1,-1,1]])
#([[0,0,1],[1,1,0],[1,0,1]])
#x = np.array([[1,1,1],[1,1,0],[1,1,1]])
#y = np.array([[0,0,1],[1,1,0],[1,0,1]])

x=im2
y=im2

w = np.array([[1.414,1,1.414],[1,1,1],[1.414,1,1.414]])

def threshold(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i,j]>0.5):
                img[i,j] = 1
            else:
                img[i,j] = -1
    return img

def inv_convert(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if(matrix[i,j] == -1):
                matrix[i,j] = 1
            else:
                matrix[i,j] = 0
    return matrix

threshold(x)
threshold(y)

def likelihood(x):
    joint_prob = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            neigh = neighbours(i,j,x.shape[0], x.shape[1])
            sum_neigh = 0
            for n in neigh:
                w_0 = n[0] - (i-1)
                w_1 = n[1] - (j-1)
                sum_neigh = (sum_neigh + x[n])*w[(w_0, w_1)]
                
            joint_prob += x[i,j]*sum_neigh + (x[i,j]+y[i,j])**2 
    return joint_prob


def prob(x):
    result = np.zeros(x.shape)
    for t in range(0,5):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
#                print("i = ",i)
#                print("j = ", j)
                
                original = x[i, j]
                
                x[i,j] = 1
#                print (x)
                x_prob = likelihood(x)
#                print("x_prob = " , x_prob)
                
                x[i,j] = -1
#                print (x)
                x_minus_prob = likelihood(x)
#                print("x_minus_prob = " , x_minus_prob)
#                print("")
                
                x[i, j] = original
                
                if(x_prob > x_minus_prob):
                    result[i,j] = 1
                else:
                    result[i,j] = -1
        x = result
    return result


result = prob(x)

inv_convert(result)
                    
imsave('output_invert.jpg', result)









