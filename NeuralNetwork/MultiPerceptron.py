import numpy as np
from activation import *
from lossfunction import *

class MultiPerceptron():
    # Perceptron which has a weight vector and bias scalar
    # This model is trained with backpropagation

    def __init__(self, name = None):
        self.name = name

        self.weight = []
        self.bias = []
        self.weightGradient = []
        self.biasGradient = []

        self.layerNumber = 0
        # Lists are used for printing model, making model for the first train and training
        self.weight_list = []
        self.activation_function_list = []
        self.dropout_list = []
        
        return

    # print model 
    def model(self):
        if self.weight_list == []:
            print("Model is not defined")
        else :
            print("Model : ")
            for i in range(self.layerNumber):
                if i == 0:
                    print("Layer",i,": [ Weight : [ None,", self.weight_list[i],"]]",", [ Activation :",self.activation_function_list[i], "], [ dropout rate :",self.dropout_list[i],"]")
                else:
                    print("Layer",i,": [ Weight : [ ",self.weight_list[i-1],",", self.weight_list[i],"]]",", [ Activation :",self.activation_function_list[i], "], [ dropout rate :",self.dropout_list[i],"]")
        
    # Add model's Layer
    def add(self, unit, activation_function = None):
        # unit : the count of nodes
        # activation_function : the activation function for the layer

        if self.weight != []:
            print("There is model that be trained already, please initialize another one")
        else: 
            self.layerNumber += 1

            self.weight_list.append(unit)
            self.activation_function_list.append(activation_function)
            self.dropout_list.append(dropout_rate)


    def train(self, trainX, trainY, learning_rate = 0.01, loss_function = MSE):
        # The basic loss Function is the Mean Square Error
        # Use gradient descent algorithm when update the parameters
        
        # change the shape of input and output when dimension is 1
        try :
            np.array(trainX).shape[1]
        except IndexError:
            trainX = np.reshape(trainX,[np.array(trainX).shape[0],1])
        try :
            np.array(trainY).shape[1]
        except IndexError:
            trainY = np.reshape(trainY,[np.array(trainY).shape[0],1])

        if np.array(trainY).shape[0] != np.array(trainX).shape[0]:
            print("Input data and output data are not same length, please check it")
            return
            
        DimensionOfInput = np.array(trainX).shape[1]
        NumberOfDataSet = np.array(trainY).shape[0]

        # To construct the model, initialize weight matrix and bias array using lists
        if self.weight == []:
            # Hidden layer and output layer construction
            try : 
                for i in range(self.layerNumber):
                    Output_layer_input = self.weight_list[i]
                    self.bias.append(np.random.randn(self.weight_list[i]))
                    self.biasGradient.append(np.zeros([self.weight_list[i],NumberOfDataSet]))
                    if i == 0:
                        self.weight.append(np.random.randn(np.array(trainX).shape[1], self.weight_list[i]))
                        self.weightGradient.append(np.zeros([np.array(trainX).shape[1], self.weight_list[i],NumberOfDataSet]))
                    else:
                        self.weight.append(np.random.randn(self.weight_list[i-1], self.weight_list[i]))
                        self.weightGradient.append(np.zeros([self.weight_list[i-1], self.weight_list[i],NumberOfDataSet]))
            except IndexError as e:
                print("Model is not defined")
                print(e)
                return
            # Add the weight with diagonal matrix at the last to calculate the gradient iteratively
            # self.weight.append(np.diag(np.ones(self.weight_list[self.layerNumber-1])))


        # Propagation
        deltah = 1e-9   # Small value for numerical differentiation
        H_unit = []     # output value for each layers
        HD_unit = []    # differtial value for activation function of calculated z

        Loss = None

        # Forward propagation
        for i in range(self.layerNumber):
            bias = np.array(self.bias[i:i+1]) # idk why but np.array(list[index]) don't return that shape of array

            # Before activation function
            if i == 0:
                z = np.dot( trainX, self.weight[i]) + bias
            else:
                z = np.dot( h, self.weight[i]) + bias

            # After activation function and differential value of each z value
            if self.activation_function_list[i] is None:
                h = np.copy(z)
                hd = np.ones(z.shape)
            else:
                h = self.activation_function_list[i](z)
                hd = (self.activation_function_list[i](z+deltah) - self.activation_function_list[i](z-deltah)) / ( 2 * deltah )

            H_unit.append(h)    # hidden unit value after activation function
            HD_unit.append(hd)  # differential value using numerical differetiation

        # Hypothesis
        yHat = h
        Loss = np.mean(loss_function(yHat, trainY))

        # Backward propagation (Chain rule using numerical differentiation)
        initial_differential = (loss_function(yHat+deltah,trainY)-loss_function(yHat-deltah,trainY))/(2*deltah)

        for l in reversed(range(self.layerNumber)):
            # Output Layer
            if l == self.layerNumber -1:
                for i in range(self.weight_list[l-1]):
                    for j in range(self.weight_list[l]):
                        for d in range(NumberOfDataSet):
                            self.weightGradient[l][i][j][d] = initial_differential[d][j] * H_unit[l-1][d][i] * HD_unit[l][d][j]
                            self.biasGradient[l][j][d] = initial_differential[d][j] * HD_unit[l][d][j]
                        
            # Input Layer
            elif l == 0 : 
                for i in range(DimensionOfInput):
                    for j in range(self.weight_list[l]):
                        for d in range(NumberOfDataSet):
                            self.weightGradient[l][i][j][d] = 0
                            self.biasGradient[l][j][d] = 0
                            for k in range(self.weight_list[l+1]):
                                self.weightGradient[l][i][j][d] += self.weightGradient[l+1][j][k][d] / H_unit[l][d][j] * self.weight[l+1][j][k] * trainX[d][i] * HD_unit[l][d][j]
                                self.biasGradient[l][j][d] += self.weightGradient[l+1][j][k][d] * self.weight[l+1][j][k] * trainX[d][i] * HD_unit[l][d][j]
            # Hidden Layer
            else : 
                for i in range(self.weight_list[l-1]):
                    for j in range(self.weight_list[l]):
                        for d in range(NumberOfDataSet):
                            self.weightGradient[l][i][j][d] = 0
                            self.biasGradient[l][j][d] = 0
                            for k in range(self.weight_list[l+1]):
                                self.weightGradient[l][i][j][d] += self.weightGradient[l+1][j][k][d] / H_unit[l][d][j] * self.weight[l+1][j][k] * H_unit[l-1][d][i] * HD_unit[l][d][j]
                                self.biasGradient[l][j][d] += self.weightGradient[l+1][j][k][d] * self.weight[l+1][j][k] * H_unit[l-1][d][i] * HD_unit[l][d][j]


        # Update weight and bias
        for i in reversed(range(self.layerNumber)):
            self.weight[i] -= learning_rate * np.mean(self.weightGradient[i],axis = 2)
            self.bias[i] -= learning_rate * np.mean(self.biasGradient[i], axis =1)
        
        Hypothesis = yHat
        return Hypothesis, Loss

    # Predict the output with trained weight
    def predict(self, Xdata):
        try :
            np.array(Xdata).shape[1]
        except IndexError:
            trainX = np.reshape(Xdata,[np.array(trainX).shape[0],1])

        # Calculate all the layers
        for i in range(self.layerNumber):
            bias = np.array(self.bias[i:i+1]) # idk why but np.array(list[index]) don't return that shape of array
            if i == 0:
                z = np.dot( Xdata, self.weight[i]) + bias
            else:
                z = np.dot( h, self.weight[i]) + bias

            if self.activation_function_list[i] is None:
                h = np.copy(z)
            else:
                h = self.activation_function_list[i](z)
        Hypothesis = h
        return Hypothesis
    
    def printWeight(self):
        # Print Weight and Bias
        print("------------------------------------")
        for i in range(self.layerNumber):
            print("Layer ",i,":", self.weight[i].shape)
            print("weight")
            print(self.weight[i])
            print("bias")
            print(self.bias[i])
        return
        print("------------------------------------")

if __name__ == "__main__":
    # Training XOR problem
    NN = MultiPerceptron()
    NN.add(2,sigmoid)
    NN.add(1,sigmoid)
    NN.model()
    trainX = np.array([[0,0],[0,1],[1,0],[1,1]])    
    trainY = np.array([[0],[1],[1],[0]])
    
    for i in range(10000):
        Hypothesis, Loss = NN.train(trainX,trainY,learning_rate = 0.3,loss_function=cross_entropy)
        if i%1000 ==0:
            print(Hypothesis,Loss)
               
    # Training iris dataset
    import keras
    from sklearn import datasets
    from otherfunction import shuffle

    iris = datasets.load_iris()
    
    # Shuffle data set
    inputData, outputData = shuffle(iris.data,iris.target)

    # divide to input data and output data

    train_input = inputData[:-10]
    train_output = outputData[:-10]
    test_input = inputData[-10:-1]
    test_output = outputData[-10:-1]    
    
    print(train_output)     # not onehot data
    # change output to one hot data
    train_output_onehot = np.zeros([np.array(train_output).shape[0],3])
    for i in range(np.array(train_output).shape[0]):
        train_output_onehot[i][train_output[i]] = 1
        
    # also for test data
    test_output_onehot = np.zeros([np.array(test_output).shape[0],3])
    for i in range(np.array(test_output).shape[0]):
        test_output_onehot[i][test_output[i]] = 1


    NN2 = MultiPerceptron()
    NN2.add(10,tanh)
    NN2.add(10,tanh)
    NN2.add(3,sigmoid)
    for i in range(2000):
        Hypothesis, Loss = NN2.train(train_input, train_output_onehot,learning_rate = 0.1, loss_function = cross_entropy)
        if i % 100 ==0:
            print("At",i,"\n Hypothesis label : ",np.argmax(Hypothesis,axis=1), "\n Loss : ",Loss)
            print("accuracy : ",np.mean(np.argmax(Hypothesis,axis=1) == train_output))

    print("For test data : ")
    print("Label : ", test_output)
    print("Hypothesis : ", np.argmax(NN2.predict(test_input)))
    print("accuracy : ", np.mean(np.argmax(Hypothesis,axis=1) == train_output))

