import numpy as np
import pandas as pd

df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

row, col = df.shape

df = np.array(df)

#2000-42000 rows for training
train_data = df[2000:row].T 
X_train = train_data[1:col]
y_train = train_data[0]

#first 1000 for test
test_data = df[:2000].T
X_test = test_data[1:col]
y_test = test_data[0]

#activation functions 
def ReLU(Z):
    return np.maximum(0,Z)

def derivation_ReLU(Z):
    return Z>0
    
def softmax(Z):
    return (np.exp(Z) / sum(np.exp(Z)))

#forward prop.
def forward_prop(X,W1,B1,W2,B2):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)
    return Z1,A1,Z2,A2

#one hot encoding
def one_hot(Y):
    return np.eye(Y.max() + 1)[Y]

#back prop
def back_prop(X, Y, Z1, Z2, A1, A2, W1, W2):

    m = X.shape[1]

    Y = one_hot(Y).T
    dL_dZ2 = A2 - Y
    dL_dW2 = 1/m * dL_dZ2.dot(A1.T)
    dL_dB2 = 1/m * np.sum(dL_dZ2.T)
    dL_dZ1 = (W2.T.dot(dL_dZ2)) * derivation_ReLU(Z1)
    dL_dW1 = 1/m * dL_dZ1.dot(X.T)
    dL_dB1 = 1/m * np.sum(dL_dZ1)

    return dL_dW1,dL_dB1,dL_dW2,dL_dB2


#we cant init params with jst 0
#if they are all zeroz then they learn same...
def initilize_params():
    
    np.random.seed(9)
    
    W1 = np.random.rand(64,784) * np.sqrt(1. / 784)
    B1 = np.zeros((64,1))
    
    W2 = np.random.rand(10,64) * np.sqrt(1. / 784)
    B2 = np.zeros((10,1))
    
    return W1, B1, W2, B2

#Update params
def update(W1, B1, W2, B2, dL_dW1, dL_dB1, dL_dW2, dL_dB2, alpha):
    W1 = W1 - alpha * dL_dW1 #alpha = learning rate
    B1 = B1 - alpha * dL_dB1

    W2 = W2 - alpha * dL_dW2
    B2 = B2 - alpha * dL_dB2

    return W1, B1, W2, B2 


#TRAINING...

#this will return max possibility for every digit
def preds(A2):
    return np.argmax(A2,0)

def acc(Y, preds):
    # print(Y)
    return np.sum(Y == preds)/ Y.size

def gradient_descent(X, Y, alpha, itrs):
    W1, B1, W2, B2 = initilize_params()

    for i in range(itrs):
        Z1,A1,Z2,A2 = forward_prop(X,W1,B1,W2,B2)
        dL_dW1, dL_dB1, dL_dW2, dL_dB2 = back_prop(X, Y, Z1, Z2, A1, A2, W1, W2)
        W1, B1, W2, B2 = update(W1, B1, W2, B2, dL_dW1, dL_dB1, dL_dW2, dL_dB2, alpha)

        if i %10 == 0:
            print(f'Iteration: {i}')
            pred = preds(A2)
            accuracy = acc(Y, pred)
            print(accuracy)
            
    return W1, B1,W2,B2

W1, B1, W2, B2 = gradient_descent(X_train, y_train, 0.1, 250)