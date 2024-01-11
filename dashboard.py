
import numpy as np
from pandas_datareader import data as pdr

# Market Data 
import yfinance as yf

#Graphing/Visualization
import datetime as dt 
import plotly.graph_objs as go 
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
# import matplotlib.pyplot as plt

def plot_graph(graph_type):
    fig = Figure(figsize=(7, 5), dpi=120)
    plot = fig.add_subplot(1, 1, 1)

    if graph_type == 'LSTM':

        data1 = np.load('LSTM_Pred.npy')
        data2 = np.load('LSTM_True.npy')
        dates  = np.load('dates.npy')
        plot.plot(dates, data2)
        plot.plot(dates, data1)
        highPred = np.array(data1)
        highTrue = np.array(data2)
        date=np.array(dates)
        highPred = str(highPred[-1])
        highTrue = str(highTrue[-1])
        date = str(date[-1])
        plot.legend(["Opening Price (true)= $"+highTrue[:6], "Opening Price (predicted)= $"+highPred[:6]])
        plot.set_title('LSTM -TSLA (predicted until '+date[:10]+')')

    elif graph_type == 'ORG':
        # data1 = np.load('GRU_Pred.npy')
        data2 = np.load('LSTM_True.npy')
        dates  = np.load('dates.npy')
        plot.plot(dates, data2)
        # highPred = np.array(data1)
        highTrue = np.array(data2)
        date=np.array(dates)
        # highPred = highPred[-1]
        highTrue = str(highTrue[-1])
        date = str(date[-1])
        plot.legend(["Opening Price (true)= $"+highTrue[:6]])
        plot.set_title('TSLA Stock Price until '+date[:10])
        # plot.plot(dates, data1)
        # plot.set_title('TSLA Stock')

    elif graph_type == 'GRU':
        data1 = np.load('GRU_Pred.npy')
        data2 = np.load('GRU_True.npy')
        dates  = np.load('dates.npy')
        plot.plot(dates, data2)
        plot.plot(dates, data1)
        highPred = np.array(data1)
        highTrue = np.array(data2)
        date=np.array(dates)
        highPred = str(highPred[-1])
        highTrue = str(highTrue[-1])
        date = str(date[-1])
        plot.legend(["Opening Price (true)= $"+highTrue[:6], "Opening Price (predicted)= $"+highPred[:6]])
        plot.set_title('GRU -TSLA (predicted until '+date[:10]+')')
        # plot.set_title('GRU - TSLA')
    
    elif graph_type == 'CNN':
        data1 = np.load('CNN_Pred.npy')
        data2 = np.load('CNN_True.npy')
        dates  = np.load('dates.npy')
        plot.plot(dates, data2)
        plot.plot(dates, data1)
        highPred = np.array(data1)
        highTrue = np.array(data2)
        date=np.array(dates)
        highPred = str(highPred[-1])
        highTrue = str(highTrue[-1])
        date = str(date[-1])
        plot.legend(["Opening Price (true)= $"+highTrue[:6], "Opening Price (predicted)= $"+highPred[:6]])
        plot.set_title('1-D CNN -TSLA (predicted until '+date[:10]+')')
        # plot.set_title('1-D CNN - TSLA')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=0, columnspan=4)
    

# Override Yahoo Finance 
yf.pdr_override()

# Create input field for our desired stock 
stock=input("Enter a stock ticker symbol: ")

# Retrieve stock data frame (df) from yfinance API at an interval of 1m 
df = yf.download(tickers=stock,period='5d',interval='1m')

print(df)

# Declare plotly figure (go)
fig=go.Figure()

fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'], name = 'market data'))

fig.update_layout(
    title= str(stock)+' Live Share Price:',
    yaxis_title='Stock Price (USD per Shares)')               

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig.show()

root = tk.Tk()

root.title("TSLA")

button_bar = tk.Button(root, text="Stock", command=lambda: plot_graph('ORG'))
button_bar.grid(row=0, column=0)

button_bar = tk.Button(root, text="LSTM", command=lambda: plot_graph('LSTM'))
button_bar.grid(row=0, column=1)

button_line = tk.Button(root, text="GRU", command=lambda: plot_graph('GRU'))
button_line.grid(row=0, column=2)

button_scatter = tk.Button(root, text="1-D CNN", command=lambda: plot_graph('CNN'))
button_scatter.grid(row=0, column=3)

root.mainloop()



