import sigmoid as sig
import math
import numpy as np
import random
from sklearn import metrics
import matplotlib.pyplot as plt
# Kyle Fitzpatrick and Michael Cooper
# 3/8/24
# HW3 Logistic Regression
# CS 430-01

# Logistic Regression
class LogisticRegression:
    def __init__(self, trainingSet, theta, alpha, epsilon):
        self.trainingSet = trainingSet # 2-D array of training data
        self.xlen = len(self.trainingSet[0]) - 1
        self.xTrain = [data[0:self.xlen] for data in self.trainingSet]
        self.yTrain = [data[self.xlen] for data in self.trainingSet]
        self.theta = theta # Training parameters
        self.alpha = alpha #Learning rate alpha
        self.epsilon = epsilon #Min grad descent cost func difference
    
    def hypothesis(self,xIn):     
        return sig.sigmoid(sum(np.multiply(xIn, self.theta)))
        
    def cost(self):
        yHat = [self.hypothesis(x) for x in self.xTrain] # Calc yhat for each row in x training data
        cost = 0
        m = len(self.trainingSet)
        for i in range (len(self.trainingSet)):
            cost += -self.yTrain[i] * np.log(yHat[i]) - (1 - self.yTrain[i]) * np.log(1 - yHat[i])
        return cost / float(m)
            
        #return -1.0/len(self.trainingSet) * sum(np.multiply(self.yTrain, [np.log(y) for y in yHat]) + np.multiply(np.subtract(1,self.yTrain), [np.log(y) for y in yHat]))
    
    def train(self):
        Jold = float('inf')
        Jnew = self.cost()
        m = len(self.trainingSet)
        while abs(Jold - Jnew) > self.epsilon:
            Jold = Jnew
            newTheta = [None] * self.xlen
            yHat = [self.hypothesis(x) for x in self.xTrain] # Calc yhat for each row in x training data
            for j in range (self.xlen):
                costDer = 0.0
                for i in range (m):
                    costDer += (yHat[i] - self.yTrain[i]) * self.xTrain[i][j]
                newTheta[j] = self.theta[j] - self.alpha/m * costDer
            self.theta = newTheta
            Jnew = self.cost()
        return abs(Jold - Jnew)


# ========= Main program =========
trainingFile = open("iris.data", "r")
trainingData = []
validationData = [None]*3
for line in trainingFile:
    if line == '\n':
        break
    dataline = line.strip().split(",")
    if dataline[4] == "Iris-setosa":
        dataline[4] = 1
    else:
        dataline[4] = 0
    for i in range(4):
        dataline[i] = float(dataline[i])
    dataline.insert(0, 1.0) # Insert 1 for x0
    
    trainingData.append(dataline)
trainingFile.close()

# Randomly choose elements from the three sets
validationData[0] = []
for i in range(10):
    validationData[0].append(trainingData.pop(random.randrange(0, 49-i, 1)))

validationData[1] = []
for i in range(10):
    validationData[1].append(trainingData.pop(random.randrange(40, 89-i, 1)))

validationData[2] = []
for i in range(10):
    validationData[2].append(trainingData.pop(random.randrange(80, 109-i, 1)))

# Create and train the regression model
newRegression = LogisticRegression(trainingData, [0, 0, 0, 0, 0], .01, .001)
newRegression.train()

results = open("results.txt", "w")
print("Validation")
results.write("Validation")
print("=====")
results.write("=====")
isCorrect = 0
predicted = []
actual = []
TP = 0
FP = 0
FN = 0
TN = 0

# Compare training results to validation data
validationData = [x for xs in validationData for x in xs]
for entry in validationData:
    predict = newRegression.hypothesis(entry[:len(entry)-1])
    predicted.append(round(predict))
    act = entry[len(entry)-1]
    actual.append(act)
    if act == round(predict):
        isCorrect = 1
    else:
        isCorrect = 0
    print(entry)
    results.write(''.join(str(x) for x in entry))
    print("Actual: " + str(act))
    results.write("Actual: " + str(act))
    print("Predicted: " + str(predict))
    results.write("Predicted: " + str(predict))
    print("Correct?: " + str(isCorrect))
    results.write("Correct?: " + str(isCorrect))
    print("=====")
    results.write("=====")

print("Theta: ")
results.write("Theta: ")
print(newRegression.theta)
results.write(''.join(str(x) for x in newRegression.theta))

# Generate confusion matrix
confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

for i in range(len(predicted)):
    if predicted[i] == 1 and actual[i] == 1:
        TP += 1
    elif predicted[i] == 1 and actual[i] == 0:
        FP += 1
    elif predicted[i] == 0 and actual[i] == 1:
        FN += 1
    else:
        TN += 1

accuracy = (TP + TN) / float(TP + FP + FN + TN)
print("Accuracy: " + str(round(accuracy, 2)))
results.write("Accuracy: " + str(round(accuracy, 2)))
if (TP + FP > 0):
    precision = TP / float(TP + FP)
    print("Precision: " + str(round(precision, 2)))
    results.write("Precision: " + str(round(precision, 2)))
else:
    print("Precision: N/A")
    results.write("Precision: N/A")
        
cm_display.plot()
plt.show()
results.close()
