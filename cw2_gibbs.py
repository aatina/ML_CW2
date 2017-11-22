#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:43:30 2017

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
im = imread("obama_gray.jpg")
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

#w = np.array([[1.414,1,1.414],[1,1,1],[1.414,1,1.414]])

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

def convert(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if(matrix[i,j] == 1):
                matrix[i,j] = 1
            else:
                matrix[i,j] = 0
    return matrix

threshold(x)
threshold(y)

height, width = x.shape
initial_static = np.random.rand(height,width)

x0 = threshold(initial_static)

imsave('x0.jpg', x0)

def gibbs_sampling(x,y):
    for t in range(0,10):
        for i in range(height):
            for j in range(width):
                x0[i,j] = 1
                prob = joint_prob(x0,y,i,j)
                x0[i,j] = -1
                p = prob/(prob + joint_prob(x0,y,i,j))
                uniform_val = np.random.uniform(0,1)
                #print (prob, uniform_val)
                if p > uniform_val:
                    x0[i,j] = 1
                else:
                    x0[i,j] = -1
    return x0


def gibbs_sampling_rand(x,y):
    
    np.random.seed(42)
    for t in range(0,15):
        for i in range(height):
            for j in range(width):
                index_i = np.random.randint(0,height)
                index_j = np.random.randint(0,width)
                #print(index_i,index_j)
                x0[index_i,index_j] = 1
                prob = joint_prob(x0,y,index_i,index_j)
                x0[index_i,index_j] = -1
                p = prob/(prob + joint_prob(x0,y,index_i,index_j))
                uniform_val = np.random.uniform(0,1)
                #print (prob, uniform_val)
                if p > uniform_val:
                    x0[index_i,index_j] = 1
                else:
                    x0[index_i,index_j] = -1
    return x0

def get_prior(x,i,j):
    (height, width) = x.shape
    accumulation_pos = 0
    accumulation_neg = 0
    cur_pos_sum = 0
    cur_neg_sum = 0
    prior_sum = 0 
    w = 1
    neigh = neighbours(i, j, height, width)
    
    for n in neigh:
        cur_pos_sum += (w * 1 * x[n])
        cur_neg_sum += (w * -1 * x[n])
        prior_sum += (w * x[i,j] * x[n])
    
    accumulation_pos += cur_pos_sum # Add that to the previous neighbours' things
    accumulation_neg += cur_neg_sum
    
    e_pos = np.exp(accumulation_pos)
    e_neg = np.exp(accumulation_neg)
    e_prior = np.exp(prior_sum)
    
    z_val = e_pos + e_neg
    
    prior = e_prior/z_val
    
    return prior

def get_likelihood(x, y, i, j):
    w = 2.1 
    pos_val = 1 * y[i,j] * w
    neg_val = -1 * y[i,j] * w
    exp_val = np.exp(pos_val) + np.exp(neg_val)
    val = np.exp(x[i,j] * y[i,j] * w)
    
    likelihood = val/ exp_val
    
    return likelihood
    

def joint_prob(x,y,i,j):
    prior = get_prior(x,i,j)
    likelihood = get_likelihood(x,y,i,j)
    posterior = prior * likelihood
    return posterior
    
result = gibbs_sampling_rand(x,y)
inv_convert(result)
                    
imsave('obama_gibbs_random_15.jpg', result)
    
    
    
    
    