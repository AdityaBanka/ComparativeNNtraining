import keras
import random
import concurrent.futures


class Genetic:
    def __init__(self, X_train, Y_train, X_test, Y_test, populationSize):
        self.xTrain = X_train
        self.yTrain = Y_train
        self.xTest = X_test
        self.yTest = Y_test

        self.agents = []
        self.populationSize = populationSize
        for i in range(populationSize):
            self.agents.append(self.createModel())  

        self.fittestMom = 0
        self.fittestDad = 0      
    
    def createModel(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(4, activation="relu"))
        model.add(keras.layers.Dense(5, activation="relu"))
        model.add(keras.layers.Dense(3, activation="softmax"))
        model.compile(
            loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return(model)
    
    def getMetrics(self, model):
        loss, accuracy = model.evaluate(self.xTrain, self.yTrain, verbose=0)
        return(loss, accuracy)
    
    def crossover(self, mom, dad, childIndex):
        layerIndex = 0
        for layerMom, layerDad in zip(mom.layers, dad.layers):
            weightsMom, biasesMom = layerMom.get_weights()
            weightsDad, biasesDad = layerDad.get_weights()

            for weightMom, weightDad in zip(weightsMom, weightsDad):
                if (random.randint(1, 2) % 2 == 0):
                    weightMom = weightDad

            for biasMom, biasDad in zip(biasesMom, biasesDad):
                if (random.randint(1, 2) % 2 == 0):
                    biasMom = biasDad

            self.agents[childIndex].layers[layerIndex].set_weights([weightsMom, biasesMom])
            layerIndex += 1

    def mutate(self, rate, index):
        #rate /= 100.0 
        layerIndex = 0
        for layer in self.agents[index].layers:
            weights, biases = layer.get_weights()
            for weight in weights:
                if(random.random() < rate):
                    delta = rate/2.0 + 1
                    if (random.randint(1, 2) % 2 == 0):
                        weight += weight*delta
                    else:
                        weight -= weight*delta
            self.agents[index].layers[layerIndex].set_weights([weights, biases])
            layerIndex += 1

    def train(self, generations):

        momIndex = self.fittestMom
        minLossMom, maxAccuracyMom = self.getMetrics(self.agents[momIndex])
        
        dadIndex = self.fittestDad
        minLossDad, maxAccuracyDad = self.getMetrics(self.agents[dadIndex])
        
        
        for generation in range(generations):
            print(f"Current generation: {generation}")

            for index, agent in enumerate(self.agents):
                loss, accuracy = self.getMetrics(agent)
                
                if((loss < minLossMom)):# or (loss == minLossMom and accuracy > maxAccuracyMom)):
                    minLossMom = loss
                    maxAccuracyMom = accuracy
                    momIndex = index
                
                if((accuracy > maxAccuracyDad)):# or (accuracy == maxAccuracyDad and loss < minLossDad)):
                    minLossDad = loss
                    maxAccuracyDad = accuracy
                    dadIndex = index


            mom = self.agents[momIndex]
            dad = self.agents[dadIndex]
            for index, agent in enumerate(self.agents):
                if(index == momIndex or index == dadIndex):
                    continue

                if index > (len(self.agents)/2.0):
                    self.agents[index] = self.createModel()
                    continue

                self.crossover(mom, dad, index)
                self.mutate(1 - maxAccuracyDad, index)

            self.fittestMom = momIndex
            self.fittestDad = dadIndex

    def evaluate(self):
        lossMom, accuracyMom = self.agents[self.fittestMom].evaluate(self.xTest, self.yTest)
        lossDad, accuracyDad = self.agents[self.fittestDad].evaluate(self.xTest, self.yTest)

        return(min(lossMom, lossDad), max(accuracyMom, accuracyDad))
