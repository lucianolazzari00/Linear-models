import numpy as np
import sys
import math
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from random import seed
from random import random
from random import randint

data = load_iris()

class linear_classifier:
    def __init__(self, w=[],learning_rate=0.001,min_w=-10,max_w=10,corr_tolerance=0,model="",emergency_stop=10000):
        if(len(w)==0):
            #if user havent specified an initial vector w, generate it randomly
            arr = np.array([])
            for i in range(5):
                r = random()
                r = min_w + (r *(max_w-min_w))
                arr = np.append(arr,r)    
            self.w = arr
        else:
            #else use the vector specified by user
            self.w = np.array(w)  
        
        if model != "perceptron" and model != "logistic_regression" and model != "linear_regression":
            sys.exit("ERROR!\tUsage: model=='perceptron' or model=='logistic_regression'")

        self.model=model
        self.learning_rate = learning_rate  #learning rate
        self.emergency_stop = emergency_stop #max number of iteration for the algorithm (only for dev)
        self.min_w = min_w  #min of the range in wich the original weights are generated randomly
        self.max_w = max_w  #max of the range in wich the original weights are generated randomly
        self.corr_tolerance = corr_tolerance  # min number of corrections to continue the gradient descent
                                              # if corrrections<corr_tolerance: stop gradient descent
    
    def fit_function(self,X,Y,batch_dim=5):
        
        '''the main algorithms'''

        if self.model == "perceptron":
            num_samples = len(X)
            i=0
            corrections_counter = 9999
            while i < self.emergency_stop and corrections_counter>self.corr_tolerance:
                corrections_counter = 0
                i+=1
                for sample_index in range(num_samples):
                    curr_sample = np.copy(X[sample_index])
                    curr_sample = np.insert(curr_sample,0,1)
                    y = Y[sample_index]
                    #threshold function
                    w_dot_x = np.dot(self.w,curr_sample) 
                    y_pred = self.threshold(w_dot_x)
                    #update weights (perceptron update rule)
                    step =  self.learning_rate*(y-y_pred)*curr_sample
                    self.w= self.w + step #sgd
                    if np.sum(step) != 0:
                        corrections_counter += 1
            print("algorithm converged after",i,"iterations")

        elif self.model == "logistic_regression":
            num_samples = len(X)
            i=0
            corrections_counter = 9999
            while i < self.emergency_stop and corrections_counter>self.corr_tolerance:
                corrections_counter = 0
                i += 1
                used_index = []
                for _ in range(batch_dim):
                    random_index = 0
                    while random_index in used_index:
                        random_index = randint(0,num_samples-1)
                    used_index.append(random_index)
                    curr_sample = np.copy(X[random_index])
                    curr_sample = np.insert(curr_sample,0,1)
                    y=Y[random_index]
                    #threshold sigmoid function
                    w_dot_x = np.dot(self.w,curr_sample)
                    h = self.sigmoid_threshold(w_dot_x)
                    #binary cross entropy loss: -((y*log(h)) + (1-y) log(1-h)) 
                    #update rule(sgd): w = w + learn_rate*(h-y)*x
                    step = self.learning_rate *np.dot(curr_sample.T,(self.sigmoid_decision(h)-y))
                    self.w = self.w - step
                    if np.sum(step) != 0:
                        corrections_counter += 1
            print("algorithm converged after",i,"iterations")

        elif self.model == "linear_regression":
            num_samples = len(X)
            i=0
            corrections_counter = 9999
            while i < self.emergency_stop:
                corrections_counter = 0
                i += 1
                batch_index = []
                random_index = 0
                for _ in range(batch_dim):
                    while random_index in batch_index: #avoid using the same sample in he same batch
                        random_index = randint(0,num_samples-1)
                    batch_index.append(random_index)
                mse = 0
                for sample_index in batch_index: #epoch
                    curr_sample = np.copy(X[sample_index])
                    curr_sample = np.insert(curr_sample,0,1)
                    y=Y[sample_index]
                    #loss function: MSE
                    y_pred = np.dot(self.w,curr_sample)
                    mse += (y - y_pred)**2
                    #stochastic gradient descent
                    self.w = self.w + self.learning_rate*(y-y_pred)*curr_sample
                if mse/batch_dim < 0.01: # if mse << 0 : converged
                    break 
            print("algorithm converged after",i,"iterations")


    def threshold(self,z):
        if z>=0:
            return 1
        else:
            return 0
    
    def sigmoid_threshold(self,z):
        res=0.5
        try:
            res = 1/(1+np.exp(-z))
        except OverflowError:
            print("ov")
        return res

    def sigmoid_decision(self,z):
        if z>0.5:
            return 1
        else:
            return 0
        
    def predict(self,x):
        if self.model=="perceptron":
            x = np.insert(x,0,1)
            w_dot_x = np.dot(self.w,x) 
            return self.threshold(w_dot_x)
        elif self.model=="logistic_regression":
            x = np.insert(x,0,1)
            w_dot_x = np.dot(self.w,x) 
            h = self.sigmoid_threshold(w_dot_x)
            if h>0.5:
                return 1
            else:
                return 0
        elif self.model=="linear_regression":
            x = np.insert(x,0,1)
            w_dot_x = np.dot(self.w,x)
            res = -1
            if w_dot_x >= 2:
                res = 2
            elif w_dot_x < 0:
                res = 0
            else:
                res = round(w_dot_x)
            return res

    def make_predictions(self,X):
        Y_pred = []
        for sample in X:
            y = self.predict(sample)
            Y_pred.append(y)
        return np.asarray(Y_pred)

    

## TESTING
##


X = data["data"]
Y = data["target"]
for i in range(len(Y)):
    if Y[i]==2:
        Y[i] = 1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)

clsf = linear_classifier(model="linear_regression",learning_rate=0.01,corr_tolerance=0)
print("\n//****************First test with random initialized vector:****************//\n\n")

print(">> initialized weights vector: ",clsf.w)
w_vect=clsf.w
print("----------Linear regression with MSE-------------")
clsf.fit_function(X_train,Y_train,batch_dim=25)
print(">> weights vector after learning: ",clsf.w)
Y_pred = clsf.make_predictions(X_test)
print(">> y's predicted by the model : ",Y_pred)
print(">> y's expected from test data: ",Y_test)
print(">> Accuracy: ",accuracy_score(Y_test, Y_pred)*100,"%")

clsf.w =w_vect #use the same vector for a better comparison of the models
clsf.model = "logistic_regression"

print("\n----------Logistic regression-------------")
clsf.fit_function(X_train,Y_train,batch_dim=30)
print(">> weights vector after learning: ",clsf.w)
Y_pred = clsf.make_predictions(X_test)
print(">> y's predicted by the model : ",Y_pred)
print(">> y's expected from test data: ",Y_test)
print(">> Accuracy: ",accuracy_score(Y_test, Y_pred)*100,"%")

clsf.w =w_vect #use the same vector for a better comparison of the models
clsf.model = "perceptron"

print("\n----------Linear classification with perceptron update rule-------------")
clsf.fit_function(X_train,Y_train,batch_dim=25)
print(">> weights vector after learning: ",clsf.w)
Y_pred = clsf.make_predictions(X_test)
print(">> y's predicted by the model : ",Y_pred)
print(">> y's expected from test data: ",Y_test)
print(">> Accuracy: ",accuracy_score(Y_test, Y_pred)*100,"%")


print("\n\n//****************Second test with zeros initialized vector: ****************//\n\n")

clsf.w=np.array([0,0,0,0,0])

print(">> initialized weights vector: ",clsf.w)
w_vect=clsf.w
print("----------Linear regression with MSE-------------")
clsf.fit_function(X_train,Y_train,batch_dim=30)
print(">> weights vector after learning: ",clsf.w)
Y_pred = clsf.make_predictions(X_test)
print(">> y's predicted by the model : ",Y_pred)
print(">> y's expected from test data: ",Y_test)
print(">> Accuracy: ",accuracy_score(Y_test, Y_pred)*100,"%")

clsf.w =w_vect #use the same vector for a better comparison of the models
clsf.model = "logistic_regression"

print("\n----------Logistic regression-------------")
clsf.fit_function(X_train,Y_train,batch_dim=25)
print(">> weights vector after learning: ",clsf.w)
Y_pred = clsf.make_predictions(X_test)
print(">> y's predicted by the model : ",Y_pred)
print(">> y's expected from test data: ",Y_test)
print(">> Accuracy: ",accuracy_score(Y_test, Y_pred)*100,"%")

clsf.w =w_vect #use the same vector for a better comparison of the models
clsf.model = "perceptron"

print("\n----------Linear classification with perceptron update rule-------------")
clsf.fit_function(X_train,Y_train,batch_dim=25)
print(">> weights vector after learning: ",clsf.w)
Y_pred = clsf.make_predictions(X_test)
print(">> y's predicted by the model : ",Y_pred)
print(">> y's expected from test data: ",Y_test)
print(">> Accuracy: ",accuracy_score(Y_test, Y_pred)*100,"%")