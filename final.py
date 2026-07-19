import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.model_selection import train_test_split

ticker = 'BTC-USD'
stock = yf.Ticker(ticker)

df = stock.history(period="5d", interval="1m")
df.index = df.index.tz_convert('Asia/Kolkata')

dic = {'Diff': [], 'Time1': []}
initial = df['Close'].iloc[0]
initial_time = df.index[0]
trend = 1 if df['Close'].iloc[0] < df['Close'].iloc[1] else 0

for i in range(1, len(df['Close']) - 1):
    new = df['Close'].iloc[i]
    new_trend = 1 if new < df['Close'].iloc[i + 1] else 0
    if new_trend != trend:
        dic['Diff'].append(new - initial)
        dic['Time1'].append((df.index[i] - initial_time).total_seconds())
        initial = df['Close'].iloc[i]
        initial_time = df.index[i]
        trend = new_trend

df = pd.DataFrame(dic)
df['Time2'] = df['Time1'].shift(-1)
df['momentum'] = abs(df['Diff'] / df['Time1'])
df['accl'] = df['momentum'] / df['Time1']
df['target'] = np.where(df['Time2'] > 60, 1, 0)

# Drop the very last row because Time2 will be NaN due to the shift
df = df.dropna()

# Features and Targets
features = ['Time1', 'momentum', 'accl']
x_train = df[features].iloc[:-100]
y_train = df['target'].iloc[:-100]

x_test = df[features].iloc[-100:]
y_test = df['target'].iloc[-100:]

# x=df[features]
# y=df['target']

# x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.05)
print(len(x_train))
# Scale specifically for KNN
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Initialize Models
knn = KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, n_neighbors=2, weights='distance')
forest = RandomForestClassifier(max_depth=10, min_samples_leaf=40, min_samples_split=50, n_estimators=100, random_state=42)

# Fit Models
knn.fit(x_train_scaled, y_train)
forest.fit(x_train, y_train) # RF works great on unscaled data

# Get Probabilities for the Positive Class (1)
knn_probs = knn.predict_proba(x_test_scaled)[:, 1]
rf_probs = forest.predict_proba(x_test)[:, 1]
print(max(rf_probs),max(knn_probs))
# Ensembled Threshold Logic: Tune these to get higher precision!
knn_threshold = 0.50
rf_threshold = 0.50

# Optimized vector comparison (Way faster than a Python loop)
y_pred = np.where((knn_probs >= knn_threshold) & (rf_probs >= rf_threshold), 1, 0)

pickle.dump(knn,open('knn_model.pkl','wb'))
pickle.dump(forest,open('rf_model.pkl','wb'))
pickle.dump(scaler,open('scaler.pkl','wb'))

# Evaluate
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")

with open('Unseen_Test_result.txt', 'a') as file:
    file.write(f"{datetime.isoformat(datetime.now())} , Ensemble Precision: {precision_score(y_test, y_pred)}\n")