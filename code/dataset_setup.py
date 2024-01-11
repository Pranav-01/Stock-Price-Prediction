import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class train_data:
    def __init__(self, price, senti, past, future) -> None:
        self.price = price
        self.senti = senti
        self.past = past
        self.future = future
    
    def csv_read(self):
        df = pd.read_csv(self.price)
        senti_data = pd.read_csv(self.senti)

        dates = pd.to_datetime(df['Date'])
        
        cols = list(df)[1:6]

        for_training = df[cols].astype(float)
        
        scaler = StandardScaler()
        scaler  = scaler.fit(for_training)
        scaled_for_training = scaler.transform(for_training)
        
        print(senti_data)
        scaler_np = StandardScaler()
        scaler_np = scaler_np.fit(senti_data)
        senti_data_scaled = scaler_np.transform(senti_data)
        scaled_for_training = np.hstack((scaled_for_training, senti_data_scaled))

        trainX = []
        trainY = []
        valX = []
        valY = []
        testX = []
        testY = []
        
        for i in tqdm(range(self.past, len(scaled_for_training) - self.future + 1)):
            if i < len(scaled_for_training) - 200:
                trainX.append(scaled_for_training[i - self.past:i, 0:for_training.shape[1]+1])
                trainY.append(scaled_for_training[i + self.future - 1:i + self.future, 0])
            elif i < len(scaled_for_training) - 50:
                valX.append(scaled_for_training[i - self.past:i, 0:for_training.shape[1]+1])
                valY.append(scaled_for_training[i + self.future - 1:i + self.future, 0])
            else:
                testX.append(scaled_for_training[i - self.past:i, 0:for_training.shape[1]+1])
                testY.append(scaled_for_training[i + self.future - 1:i + self.future, 0])
                
        trainX, trainY, valX, valY, testX, testY = np.array(trainX), np.array(trainY), np.array(valX), np.array(valY), np.array(testX), np.array(testY)
        return trainX, trainY, valX, valY, testX, testY, scaler, dates[-50:]