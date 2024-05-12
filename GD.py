import keras
class GradiantDecent:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.xTrain = X_train
        self.yTrain = Y_train
        self.xTest = X_test
        self.yTest = Y_test

        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(4, activation="relu"))
        self.model.add(keras.layers.Dense(5, activation="relu"))
        self.model.add(keras.layers.Dense(3, activation="softmax"))
        self.model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
    def train(self, epochs):
        self.model.fit(self.xTrain, self.yTrain, epochs=epochs, verbose=0)

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.xTest, self.yTest)
        return(loss, accuracy)
    
    