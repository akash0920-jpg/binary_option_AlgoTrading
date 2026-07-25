import time
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.title("Simulation of Binary Option Trading (AI Live Inference)")

ticker = "BTC-USD"
stock = yf.Ticker(ticker)

# Load trained models and scaler safely
knn = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
rf = pickle.load(open("rf_model.pkl", "rb"))

knn_threshold = 0.50
rf_threshold = 0.50

# --- 1. INITIALIZE ALL STATE MACHINE VARIABLES IN SESSION STATE ---
if "projection_history" not in st.session_state:
  st.session_state.projection_history = []
if "prev_price" not in st.session_state:
  st.session_state.prev_price = 0.0
if "prev_trend" not in st.session_state:
  st.session_state.prev_trend = -1
if "initial_price" not in st.session_state:
  st.session_state.initial_price = 0.0
if "initial_time" not in st.session_state:
  st.session_state.initial_time = None

# Fetch historical data (last 5 days of 1-minute intervals)
df = stock.history(period="5d", interval="1m")

if not df.empty:
  # Extract the Close prices and keep only the last 100 points
  prices = df["Close"].tail(100)
  current_price = prices.iloc[-1]
  last_timestamp = prices.index[-1]
  first_timestamp = prices.index[0]

  # Initialize initial_time on the very first run if it's None
  if st.session_state.initial_time is None:
    st.session_state.initial_time = last_timestamp
    st.session_state.initial_price = current_price
    st.session_state.prev_price = current_price

  # --- 2. STATE MACHINE LOGIC PERSISTING ACROSS RERUNS ---
  trend = 1 if current_price > st.session_state.prev_price else 0

  if trend != st.session_state.prev_trend:
    diff = current_price - st.session_state.initial_price
    time1 = (last_timestamp - st.session_state.initial_time).total_seconds()

    st.write(f"🔄 **Trend Flipped!** Diff: `{diff:,.2f}` | Time1: `{time1}s`")

    if time1 > 0:
      momentum = abs(diff / time1)
      accl = momentum / time1

      # Create feature dataframe
      x = pd.DataFrame(
          {"Time1": [time1], "momentum": [momentum], "accl": [accl]}
      )

      x_scaled = scaler.transform(x)

      # Predict probabilities from both models
      knn_probs = knn.predict_proba(x_scaled)[:, 1]
      rf_probs = rf.predict_proba(x)[:, 1]

      # Apply bagging threshold logic
      signal = (
          1
          if (knn_probs >= knn_threshold) and (rf_probs >= rf_threshold)
          else 0
      )

      if signal == 1 and diff > 0:
        st.success("🤖 **AI Signal:** Predicted movement DOWN (Reversal)")
      elif signal == 1 and diff < 0:
        st.success("🤖 **AI Signal:** Predicted movement UP (Reversal)")
      else:
        st.info("🤖 **AI Signal:** Market Noise / No Trade Trigger")

    # Reset state pointers for the next trend leg
    st.session_state.prev_trend = trend
    st.session_state.initial_price = current_price
    st.session_state.initial_time = last_timestamp

  st.session_state.prev_price = current_price

  # Display the current live price metric
  st.metric(label="Current BTC-USD Price", value=f"${current_price:,.2f}")

  # Dynamic buffer for Y-axis bounds
  min_price = prices.min() - 50
  max_price = prices.max() + 50

  # Calculate future timestamps
  future_1min = last_timestamp + pd.Timedelta(minutes=1)
  future_2min = last_timestamp + pd.Timedelta(minutes=2)

  # --- 3. SAVE FULL-WIDTH REFERENCE LINE TO SESSION STATE ---
  new_proj_x = [first_timestamp, future_1min]
  new_proj_y = [current_price, current_price]

  if (
      not st.session_state.projection_history
      or st.session_state.projection_history[-1]["x"][1] != future_1min
  ):
    st.session_state.projection_history.append(
        {"x": new_proj_x, "y": new_proj_y}
    )

  if len(st.session_state.projection_history) > 15:
    st.session_state.projection_history.pop(0)

  # --- 4. BUILD PLOTLY FIGURE ---
  fig = px.line(
      x=prices.index,
      y=prices.values,
      labels={"x": "Time", "y": "Price (USD)"},
      title="Live Price Action with State Machine Tracking",
  )

  # Draw projection trails
  for idx, proj in enumerate(st.session_state.projection_history):
    is_latest = idx == len(st.session_state.projection_history) - 1
    line_color = "orange" if is_latest else "gray"
    line_opacity = 1.0 if is_latest else 0.25
    line_width = 1.5 if is_latest else 1.0

    fig.add_trace(
        go.Scatter(
            x=proj["x"],
            y=proj["y"],
            mode="lines",
            line=dict(color=line_color, width=line_width, dash="dash"),
            opacity=line_opacity,
            showlegend=False,
        )
    )

  fig.update_layout(
      yaxis=dict(range=[min_price, max_price]),
      xaxis=dict(range=[first_timestamp, future_2min]),
      showlegend=False,
  )

  st.plotly_chart(fig, width="stretch")

else:
  st.warning("Waiting for data feed...")

# --- LIVE 60-SECOND COUNTDOWN TIMER ---
timer_placeholder = st.empty()
progress_bar = st.progress(0)

for remaining in range(60, 0, -1):
  timer_placeholder.markdown(
      f"⏱️ **Next candle refresh in:** `{remaining} seconds`"
  )
  progress_bar.progress((60 - remaining) / 60)
  time.sleep(1)

timer_placeholder.empty()
progress_bar.empty()

st.rerun()