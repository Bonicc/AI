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

        # For the first update model determine the size of input
        if self.weight is None:
            try:
                self.weight = np.random.randn(np.array(trainX).shape[1])
            except IndexError:
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

        except:
            pass

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