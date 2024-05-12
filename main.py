import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split

import GD
import GeN

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

def main():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=100)

    # gradiantDecentModel = GD.GradiantDecent(xTrain, yTrain, xTest, yTest)
    # while True:
    #     gradiantDecentModel.train(epochs=5)
    #     loss, accuracy = gradiantDecentModel.evaluate()
    #     if(accuracy > 0.99):
    #         break

    geneticModel = GeN.Genetic(xTrain, yTrain, xTest, yTest, populationSize=100)
    while True:
        geneticModel.train(generations=1)
        loss, accuracy = geneticModel.evaluate()
        if(accuracy > 0.95):
            break

if __name__ == "__main__":
    main()