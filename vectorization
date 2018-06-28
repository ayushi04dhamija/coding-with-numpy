##Implement the L1 and L2 loss functions
##Exercise: Implement the numpy vectorized version of the L1 loss. You may find the function abs(x) (absolute value of x) useful.
##L1(ŷ ,y)=∑i=0m|y(i)−ŷ (i)|

# GRADED FUNCTION: L1

def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum(abs(y-yhat),axis=0,keepdims=True)
    ### END CODE HERE ###
    
    return loss
 yhat = np.array([.9, 0.2, 0.1, .4, .9])
 y = np.array([1, 0, 0, 1, 1])
 print("L1 = " + str(L1(yhat,y)))

##Exercise: Implement the numpy vectorized version of the L2 loss. There are several way of implementing the L2 loss but you may find the function np.dot() useful. As a reminder, if  x=[x1,x2,...,xn]x=[x1,x2,...,xn] , then np.dot(x,x) =  ∑nj=0x2j∑j=0nxj2 .

#L2 loss is defined as
#L2(ŷ ,y)=∑i=0m(y(i)−ŷ (i))2
# GRADED FUNCTION: L2

def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    dot=np.dot(y-yhat,y-yhat)
    loss=np.sum(dot,axis=0,keepdims=True)
    ### END CODE HERE ###
    
    return loss
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))
