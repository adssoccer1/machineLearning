
    

import csv
import math
import operator


#leaves you with a 2-d matric of stats for each player with player name as last col.
#the dimensions are 8207(18207 exs in total set) x 35 with index -1 the name of the player. 
def loadDataset(filename):
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        trainingSet = []
        for x in range(1, len(dataset)-10000):
            tempTrainingExample = []
            for y in range(34):
                currIndex = 54
                currIndex += y
                dataset[x][currIndex] = float(dataset[x][currIndex])
                tempTrainingExample.append(float(dataset[x][currIndex]))
            tempTrainingExample.append(dataset[x][2])
            trainingSet.append(tempTrainingExample)
    return trainingSet

def euclideanDistance(instance1, instance2):
    distance = 0
    for x in range(len(instance1)-1):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    neighbors = []
    distances =[]
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x])
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def findPlayerInstance(trainingSet, name):
    for i in range(len(trainingSet)):
        if name in trainingSet[i][34]:
            return i
    return 0


def main():

    #get the training set
    trainingSet = loadDataset('data.csv')

    #get exampleInstance - just gives messi if not found
    instance = findPlayerInstance(trainingSet, "Pogba")
    print(instance)

    #get the k nearest neighbors.
    neighbors = getNeighbors(trainingSet, trainingSet[instance], 40)

    #convert rows to names
    namesNeighbors = []
    for i in range(len(neighbors)):
        namesNeighbors.append(neighbors[i][34])

    #print names
    print(namesNeighbors)
    print("--------")

main()
