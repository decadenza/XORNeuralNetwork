#!/usr/bin/env python3
'''
Based on the code in https://iamtrask.github.io/2015/07/12/basic-python-network/
and adapted for the XOR function
'''
import numpy as np

# activation function
def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
    

X = np.array([[0,0],  # XOR function input
            [0,1],
            [1,0],
            [1,1]])

Y = np.array([[0],   # XOR function output
			[1],
			[1],
			[0]])
           

# randomly initialize our weights
np.random.seed(1)
syn0 = np.random.random((2,3)) # represents links between 3 neurons (cols) and 2 input (row)
syn1 = np.random.random((3,1)) # represents links between 3 neurons and the output

for i in range(100000): # You can increase iterations for better results
    # forward propagation
    l1 = nonlin(np.dot(X,syn0)) # (4,3) obtaining neurons final value (after activation function nonlin) one per row
    l2 = nonlin(np.dot(l1,syn1)) # (4,1) obtaining the output
    
    error2 = Y - l2 # error of the second layer size (4,1)
    
    # backward propagation
    
    delta2 = error2*nonlin(l2,deriv=True) # scalar product size (4,1)
    
    error1 = delta2.dot(syn1.T) # size (4,3)
    
    delta1 = error1*nonlin(l1,deriv=True) # (4,3) scalar product
    
    # updating synapses values
    syn1 += l1.T.dot(delta2)
    syn0 += X.T.dot(delta1)
    

print("Final error: ", np.mean(np.abs(error2)))
print("First layer synapses: ", syn0)
print("Second layer synapses: ", syn1)

# Evaluating the algorithm inserting 2 input values
while(True):
    a = input("First XOR input: ")
    b = input("Second XOR input: ")
    
    x = np.array([float(a),float(b)]) # input matrix
    
    res = nonlin(np.dot(nonlin(np.dot(x,syn0)),syn1))
    
    print(res)
    
    
    
    