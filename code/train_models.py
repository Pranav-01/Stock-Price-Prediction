from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

class Models():
    def __init__(self, trainX, trainY, valX, valY, testX, testY, scaler, dates, plot_type) -> None:
        self.trainX = trainX
        self.trainY = trainY
        self.valX = valX
        self.valY = valY
        self.testX = testX
        self.testY = testY
        self.scaler = scaler
        self.dates = dates
        self.plot_type = plot_type

    def MAPE(self, true_val, pred_val):
        sum = 0
        for i in range(len(true_val)):
            sum += np.abs((true_val[i] - pred_val[i]) / true_val[i])

        sum = sum / len(true_val)
        return sum * 100

    def plot_predictions1(self, model, X, y, name):
        print("********PREDICTING THE FUTURE********\n")
        predictions = model.predict(X).flatten()
        r_predictions = []
        for i in range(50):
            r_predictions += [[predictions[i], predictions[i], predictions[i], predictions[i], predictions[i]]]
        final_predictions = self.scaler.inverse_transform(r_predictions)[:, 0]

        r_y = np.repeat(y, 5, axis = -1)
        final_y = self.scaler.inverse_transform(r_y)[:, 0]

        if self.plot_type:
            plt.plot(self.dates, final_predictions)
            plt.plot(self.dates, final_y)
            plt.xlabel("Day")
            plt.ylabel("Stock Price in ($)")
            plt.title(f"{name} MODEL OUTPUT\nMEAN ABSOLUTE PERCENTAGE ERROR: {self.MAPE(y, predictions)}%")
            plt.legend(["PREDICTION", "ACTUAL"])
            plt.show()

            return [], []
        else:
            return np.array(final_y), np.array(final_predictions)
    
    def LSTM(self):
        model1 = Sequential()
        model1.add(InputLayer((14, 6)))
        model1.add(LSTM(64))
        model1.add(Dense(8, 'relu'))
        model1.add(Dense(1, 'linear'))

        cp1 = ModelCheckpoint('lstm_model/', save_best_only=True)
        model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

        model1.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), epochs=100, callbacks=[cp1])
        print("********LSTM MODEL HAS SUCCESSFULLY BEEN TRAINED********\n")

        true_val, pred_val = self.plot_predictions1(model1, self.testX, self.testY, "LSTM")

        return true_val, pred_val
    
    # def LSTM(self):
    #     model1 = Sequential()
    #     model1.add(InputLayer((14, 5)))
    #     model1.add(LSTM(64, return_sequences=True))
    #     model1.add(LSTM(32))
    #     model1.add(Dense(8, 'relu'))
    #     model1.add(Dense(1, 'linear'))

    #     cp1 = ModelCheckpoint('lstm_model/', save_best_only=True)
    #     model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

    #     model1.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), epochs=10, callbacks=[cp1])
    #     print("********LSTM MODEL HAS SUCCESSFULLY BEEN TRAINED********")

    #     self.plot_predictions1(model1, self.testX, self.testY)

    #     return model1

    # def LSTM(self):
    #     model1 = Sequential()
    #     model1.add(InputLayer((14, 5)))
    #     model1.add(LSTM(32, return_sequences=True))
    #     model1.add(LSTM(64))
    #     model1.add(Dense(8, 'relu'))
    #     model1.add(Dense(1, 'linear'))

    #     cp1 = ModelCheckpoint('lstm_model/', save_best_only=True)
    #     model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

    #     model1.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), epochs=10, callbacks=[cp1])
    #     print("********LSTM MODEL HAS SUCCESSFULLY BEEN TRAINED********")

    #     self.plot_predictions1(model1, self.testX, self.testY)

    #     return model1
    
    def GRU(self):
        model2 = Sequential()
        model2.add(InputLayer((14, 6)))
        model2.add(GRU(64))
        model2.add(Dense(8, 'relu'))
        model2.add(Dense(1, 'linear'))

        cp2 = ModelCheckpoint('gru_model/', save_best_only=True)
        model2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

        model2.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), epochs=100, callbacks=[cp2])
        print("********GRU MODEL HAS SUCCESSFULLY BEEN TRAINED********\n")

        true_val, pred_val = self.plot_predictions1(model2, self.testX, self.testY, "GRU")

        return true_val, pred_val
    
    # def GRU(self):
    #     model2 = Sequential()
    #     model2.add(InputLayer((14, 5)))
    #     model2.add(GRU(64, return_sequences=True))
    #     model2.add(GRU(32))
    #     model2.add(Dense(8, 'relu'))
    #     model2.add(Dense(1, 'linear'))

    #     cp2 = ModelCheckpoint('gru_model/', save_best_only=True)
    #     model2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

    #     model2.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), epochs=10, callbacks=[cp2])
    #     print("********GRU MODEL HAS SUCCESSFULLY BEEN TRAINED********")

    #     self.plot_predictions1(model2, self.testX, self.testY)

    #     return model2

    # def GRU(self):
    #     model2 = Sequential()
    #     model2.add(InputLayer((14, 5)))
    #     model2.add(GRU(32, return_sequences=True))
    #     model2.add(GRU(64))
    #     model2.add(Dense(8, 'relu'))
    #     model2.add(Dense(1, 'linear'))

    #     cp2 = ModelCheckpoint('gru_model/', save_best_only=True)
    #     model2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

    #     model2.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), epochs=10, callbacks=[cp2])
    #     print("********GRU MODEL HAS SUCCESSFULLY BEEN TRAINED********")

    #     self.plot_predictions1(model2, self.testX, self.testY)

    #     return model2

    def CNN1D(self):
        model3 = Sequential()
        model3.add(InputLayer((14, 6)))
        model3.add(Conv1D(64, kernel_size=2))
        model3.add(Flatten())
        model3.add(Dense(8, 'relu'))
        model3.add(Dense(1, 'linear'))

        cp3 = ModelCheckpoint('cnn1d_model/', save_best_only=True)
        model3.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

        model3.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), epochs=100, callbacks=[cp3])
        print("********CNN1D MODEL HAS SUCCESSFULLY BEEN TRAINED********\n")

        true_val, pred_val = self.plot_predictions1(model3, self.testX, self.testY, "CNN1D")

        return true_val, pred_val