import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import seaborn as sns
from matplotlib import rc

from pyESN.pyESN import ESN 

import yfinance as yf 


yf.pdr_override()

# # Create input field for our desired stock 
stock=input("Enter a stock ticker symbol: ")

# # Retrieve stock data frame (df) from yfinance API at an interval of 1m 
df = yf.download(tickers=stock,period='5d',interval='1m')
print(df.head())
df = pd.DataFrame(df)
high = df['High']
high = pd.DataFrame(high)
print(high.head())
file_path = stock +'.csv'
print (file_path)
high.to_csv(file_path, sep='\t', index=False, header=False)
# filename = 

# Read dataset
data = open(str(file_path)).read().split()
# data = open("amazon.txt").read().split()
data = np.array(data).astype('float64')
n_reservoir= 500
sparsity=0.2
rand_seed=23
spectral_radius = 1.2
noise = .0005

def MAPE(true_val, pred_val):
    sum = 0
    for i in range(len(true_val)):
        sum += np.abs((true_val[i] - pred_val[i]) / true_val[i])

    sum = sum / len(true_val)
    return sum * 100

esn = ESN(n_inputs = 1,
      n_outputs = 1, 
      n_reservoir = n_reservoir,
      sparsity=sparsity,
      random_state=rand_seed,
      spectral_radius = spectral_radius,
      noise=noise)

trainlen = 1500
future = 2
futureTotal=100
pred_tot=np.zeros(futureTotal)

for i in tqdm(range(0,futureTotal,future)):
    pred_training = esn.fit(np.ones(trainlen),data[i:trainlen+i])
    prediction = esn.predict(np.ones(future))
    pred_tot[i:i+future] = prediction[:,0]


#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)

plt.figure(figsize=(16,8))
plt.plot(range(1000,trainlen+futureTotal),data[1000:trainlen+futureTotal],'b',label="Tesla Stock", alpha=0.3)
# plt.plot(range(0,trainlen),pred_training,'.g',  alpha=0.3)
plt.plot(range(trainlen,trainlen+futureTotal),pred_tot,'k',  alpha=0.8, label='Free Running ESN')
lo,hi = plt.ylim()
plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:', linewidth=4)

plt.title(r'Tesla Ground Truth and Echo State Network Output', fontsize=25)
plt.xlabel(r'Time (2 Days)', fontsize=20,labelpad=10)
plt.ylabel(r'Price ($)', fontsize=20,labelpad=10)
plt.legend(fontsize='xx-large', loc='best')
sns.despine()
plt.show()

true = data[trainlen:trainlen+futureTotal]
pred = pred_tot
print("XXXXXXXXXXXXXX MAPE XXXXXXXXXXXXXX")
print(MAPE(true,pred))