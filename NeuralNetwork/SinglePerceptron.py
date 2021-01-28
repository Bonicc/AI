import numpy as np

class SinglePerceptron():
    # Perceptron which has a weight vector and bias scalar
    # This model is trained with backpropagation
    def __init__(self):
        self.weight = None
        self.bias = None
        return

    def train(self, trainX, trainY, learning_rate = 0.01, loss_function = "mean_square"):
        # The Loss Function is the square of difference of hypothesis and real Y
        # Use gradient descent algorithm when update the parameters

        # To construct the model, determine the input size of model
        if self.weight is None:
            try:
                self.weight = np.random.randn(np.array(trainX).shape[1])
            except IndexError as e:
                print(e)
                print("The input data must be a matrix")
        if self.bias is None:
            self.bias = np.random.randn(1)

        # Propagation
        try :
            # Forward propagation
            yHat = np.dot(self.weight, np.transpose(trainX)) + self.bias

            # Backward propagation
            WeightGradient = np.dot((yHat - trainY) , trainX) / np.array(trainX).shape[0]
            BiasGradient = np.mean(yHat - trainY)
            
            # Update weight and bias
            self.weight -= learning_rate * WeightGradient
            self.bias -= learning_rate * BiasGradient

        except IndexError as e:
            print(e)

        # Calculate Loss
        Loss = np.mean(np.square(yHat - trainY))

        return Loss

    def predict(self, Xdata):
        # Predict Y data with trained weight and bias
        return np.dot(self.weight, np.transpose(Xdata)) + self.bias
    
    def printWeight(self):
        # Print Weight and Bias
        print("weight :", self.weight)
        print("bias :", self.bias)

if __name__ == "__main__":
    # Model test with input size 2
    NN1 = SinglePerceptron()

    trainX = np.array([[2,2],[2,3],[2,4]])
    trainY = np.array([4,5,6])

    for _ in range(10000):
        Loss = NN1.train(trainX,trainY)

    print("Loss at", _ , ":", Loss)    
    NN1.printWeight()
    print(NN1.predict(np.array([[2,5]])))

    # Model test with input size 1
    NN2 = SinglePerceptron()
    
    trainX = np.array([[2],[3],[4]])
    trainY = np.array([4,5,6])

    for _ in range(10000):
        Loss = NN2.train(trainX, trainY)

    print("Loss at", _ , ":", Loss)    
    NN2.printWeight()
    print(NN2.predict(np.array([[4],[5]])))

    # XOR problem
    NN3 = SinglePerceptron()
    trainX = np.array([[0,0],[0,1],[1,0],[1,1]])
    trainY = np.array([0,1,1,0])

    for _ in range(10000):
        Loss = NN3.train(trainX, trainY)
    print("Loss at", _ , ":", Loss)    
    NN3.printWeight()
    print(NN3.predict(trainX))
