print("********SYSTEM BOOTING UP********")
from dataset_setup import train_data
from train_models import Models
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print("********SYSTEM BOOT UP COMPLETE********\n")

print("********INITIATING DATASET PREPARATION PROTOCOLS********\n")
path_price = 'code\dataset\TSLA.csv'
path_senti = 'code\dataset\TSLA_Senti.csv'
obj = train_data(path_price, path_senti, 14, 1)
trainX, trainY, valX, valY, testX, testY, scaler, dates = obj.csv_read()
# df = pd.read_csv(path_price)
# dates = len(df)
# dates = len(testX)
print("********DATASET PREPARATION PROTOCOLS COMPLETE********\n")
plot_type = False
lstm_model = Models(trainX, trainY, valX, valY, testX, testY, scaler, dates, plot_type)
gru_model = Models(trainX, trainY, valX, valY, testX, testY, scaler, dates, plot_type)
cnn_model = Models(trainX, trainY, valX, valY, testX, testY, scaler, dates, plot_type)

print("********INITIATING LSTM MODEL PROTOCOLS********\n")
true_val1, pred_val1 = lstm_model.LSTM()


print("********INITIATING GRU MODEL PROTOCOLS********\n")
true_val2, pred_val2 = gru_model.GRU()



print("********INITIATING CNN1D MODEL PROTOCOLS********\n")
true_val3, pred_val3 = cnn_model.CNN1D()



# print("********SYSTEM TERMINATING********\n")
print("XXXXXXXXXXXXX LSTM MAPE XXXXXXXXXXX")
print(lstm_model.MAPE(true_val1, pred_val1))
print("XXXXXXXXXXXXXX GRU MAPE XXXXXXXXXXX")
print(gru_model.MAPE(true_val2, pred_val2))
print("XXXXXXXXXXXXXX CNN MAPE XXXXXXXXXXX")
print(cnn_model.MAPE(true_val3, pred_val3))

# if plot_type == False:
np.save("LSTM_True.npy", true_val1)
np.save("GRU_True.npy", true_val2)
np.save("CNN_True.npy", true_val3)
np.save("LSTM_Pred.npy", pred_val1)
np.save("GRU_Pred.npy", pred_val2)
np.save("CNN_Pred.npy", pred_val3)
np.save("dates.npy", dates)


print("********SYSTEM TERMINATING********")