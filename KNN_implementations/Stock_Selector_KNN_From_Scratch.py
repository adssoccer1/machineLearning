# - the loadDataset function will need to be modified based on the format of the rows/cols of the csv file being used.
# Features of curernt CSV file: Symbol, Company Name, Security Type, Security Price, EPS Growth (TTM vs Prior TTM), EPS Growth (Last Qtr vs. Same Qtr Prior Yr), EPS Growth (3 Year History)

import csv
import math
import operator

#Load file function 
def loadDataset(filename):
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        trainingSet = []
        for x in range(1, 200):  #needs to be changed based on number of examples .
            tempTrainingExample = []
            tempTrainingExample.append(dataset[x][0])
            for y in range(4):
                currIndex = 3
                currIndex += y
                dataset[x][currIndex] = float(dataset[x][currIndex])
                tempTrainingExample.append(float(dataset[x][currIndex]))
            tempTrainingExample.append(dataset[x][2])
            trainingSet.append(tempTrainingExample)
        return trainingSet

#euclidain distance function 
def euclideanDistance(example1, example2):
    distance = 0
    for x in range(1,len(example1)-1):
        distance += pow((example1[x] - example2[x]), 2)
    return math.sqrt(distance)

#return the k nearest neighbors
def getNeighbors(traingSet, instanceToCompare, k):
    neighbors = []
    distances = []
    for x in range(len(traingSet)):
        distance = euclideanDistance(instanceToCompare, traingSet[x])
        distances.append((traingSet[x], distance))
    distances.sort(key=operator.itemgetter(1))
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def main():
    #load training set
    trainingSet = loadDataset("screener_results.csv")

    #Find stocks like TSLA
    instance_TSLA = ['TSLA', 228.0, 75.93052, 45.26066, -6.19609, 'Common Stock']

    #get the k nearest neighbors of instance_TSLA
    k = 5
    neighbors = getNeighbors(trainingSet, instance_TSLA, k)

    #print the k most similar stocks based on features/columns in the csv file
    print(neighbors)
    
main()
