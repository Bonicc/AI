import numpy as np

class SimpleNeuralNetwork():
    # Neural network which has only one scalar weight and bias parameters
    # This model is trained with backpropagation
    def __init__(self):
        self.weight = float(0)
        self.bias = float(0)
        return

    def train(self, trainX, trainY, learning_rate = 0.03):
        # The Loss Function is the square of hypothesis and real Y
        # Use gradient descent algorithm when update the parameters
        self.weight = self.weight - learning_rate * np.mean(2 * self.weight * np.transpose(trainX) * trainX + 2 * trainX * self.bias - 2 *trainX * trainY)
        self.bias = self.bias - learning_rate * np.mean(2 * self.bias + 2 * self.weight *trainX - 2 * trainY)
        Loss = np.mean(np.square(self.weight * trainX + self.bias - trainY))
        return Loss

    def predict(self, Xdata):
        # Predict Y data with trained weight and bias
        return self.weight * Xdata + self.bias
    
    def printWeight(self):
        # Print Weight and Bias
        print("weight :", self.weight)
        print("bias :", self.bias)

if __name__ == "__main__":
    # perfect data case
    NN1 = SimpleNeuralNetwork()

    trainX = np.array([2,3,4])
    trainY = np.array([4,5,6])

    for _ in range(1000):
        Loss = NN1.train(trainX,trainY)

    print("Loss at", _ , ":", Loss)    
    NN1.printWeight()
    print(NN1.predict(np.array([4,5])))

    # noise data case
    NN2 = SimpleNeuralNetwork()
    
    trainX = np.array([3,3,4])
    trainY = np.array([4,5,6])

    for _ in range(1000):
        Loss = NN2.train(trainX, trainY)

    print("Loss at", _ , ":", Loss)    
    NN2.printWeight()
    print(NN2.predict(np.array([4,5])))