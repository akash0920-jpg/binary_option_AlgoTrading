import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ticker='BTC-USD' # 'EURUSD=X' or 'USDJPY=X'
stock =yf.Ticker(ticker)

df=stock.history(period="5d", interval="1m")
df.index = df.index.tz_convert('Asia/Kolkata')
print(df['Close'].head(10))

dic={'Diff':[],'Time1':[],'Start':[],'End':[]}

initial=df['Close'].iloc[0];initial_time=df.index[0]
trend= 1 if df['Close'].iloc[0]<df['Close'].iloc[1] else 0

for i in range(1,len(df['Close'])-1):
    new=df['Close'].iloc[i]
    new_trend=1 if new<df['Close'].iloc[i+1] else 0
    if new_trend != trend:
        dic['Diff'].append(new-initial)
        dic['End'].append(df.index[i])
        dic['Start'].append(initial_time)
        dic['Time1'].append((df.index[i]-initial_time).total_seconds())
        initial=df['Close'].iloc[i];initial_time=df.index[i]
        trend=new_trend

df=pd.DataFrame(dic)
df['Time2']=df['Time1'].shift(-1)

plt.figure(figsize=[1080,1920])
plt.scatter(df['Diff'],df['Time2'])
plt.show()
print(df.tail())
# df.to_csv(r'D:\DataSets\AlgoTrading\BITCIN\Trend9.csv')