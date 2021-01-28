import numpy as np

def shuffle(Xdata,Ydata,count=None):
    if count==None:
        count = np.array(Ydata).shape[0]

    for i in range(count):
        temp1 = np.random.randint(np.array(Ydata).shape[0])
        temp2 = np.random.randint(np.array(Ydata).shape[0])
        
        Xdata[temp1],Xdata[temp2] = Xdata[temp2],Xdata[temp1]
        Ydata[temp1],Ydata[temp2] = Ydata[temp2],Ydata[temp1]

    return Xdata,Ydata