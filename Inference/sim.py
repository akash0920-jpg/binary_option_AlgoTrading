import time
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Expand page layout to use the full width of the browser screen
st.set_page_config(layout="wide")

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
if "demo_balance" not in st.session_state:
    st.session_state.demo_balance = 100.0
if "trade_history" not in st.session_state:
    st.session_state.trade_history = []

# Fetch historical data (last 5 days of 1-minute intervals)
df = stock.history(period="5d", interval="1m")

# Set up a wide 3-column layout that stretches across the entire screen
left_col, mid_col, right_col = st.columns([1, 4, 1])

if not df.empty:
    prices = df["Close"].tail(100)
    prices.index = prices.index.tz_convert('Asia/Kolkata')
    current_price = prices.iloc[-1]
    last_timestamp = prices.index[-1]
    first_timestamp = prices.index[0]
    signal = 0
    
    # Initialize initial_time on the very first run if it's None
    if st.session_state.initial_time is None:
        st.session_state.initial_time = last_timestamp
        st.session_state.initial_price = current_price
        st.session_state.prev_price = current_price

    # --- 2. STATE MACHINE LOGIC & TRADE RESOLUTION ---
    trend = 1 if current_price > st.session_state.prev_price else 0
        
    if len(st.session_state.projection_history) > 0:
        if st.session_state.projection_history[-1]['color'] == 'green' and trend == 1:
            win_amount = 10.0 + 10.0 * 0.85
            st.session_state.demo_balance += win_amount
            st.session_state.trade_history.append({'time': str(last_timestamp), 'tnx': f"+${win_amount:,.2f}", 'type': 'win'})
        elif st.session_state.projection_history[-1]['color'] == 'red' and trend == 0:
            win_amount = 10.0 + 10.0 * 0.85
            st.session_state.demo_balance += win_amount
            st.session_state.trade_history.append({'time': str(last_timestamp), 'tnx': f"+${win_amount:,.2f}", 'type': 'win'})

  

    # --- MIDDLE COLUMN: LIVE CHART & AI SIGNALS ---
    with mid_col:
        # if "metric_slot_price" not in st.session_state:
        #     st.session_state.metric_slot_price = st.empty()
        # else:
        #     st.session_state.metric_slot_price = st.empty()
        # st.session_state.metric_slot_price.metric(label="Current BTC-USD Price", value=f"${current_price:,.2f}")
        st.metric(label="Current BTC-USD Price", value=f"${current_price:,.2f}")


        # Dedicated container for dynamic status outputs to prevent stacking messages
        if "status_slot" not in st.session_state:
              st.session_state.status_slot = st.empty()
        else:
            st.session_state.status_slot = st.empty()

        if trend != st.session_state.prev_trend:
            diff = current_price - st.session_state.initial_price
            time1 = (last_timestamp - st.session_state.initial_time).total_seconds()
            
            if time1 > 0:
                momentum = abs(diff / time1)
                accl = momentum / time1
    
                x = pd.DataFrame(
                    {"Time1": [time1], "momentum": [momentum], "accl": [accl]}
                )
    
                x_scaled = scaler.transform(x)
    
                knn_probs = knn.predict_proba(x_scaled)[:, 1]
                rf_probs = rf.predict_proba(x)[:, 1]
    
                signal = (
                    1
                    if (knn_probs >= knn_threshold) and (rf_probs >= rf_threshold)
                    else 0
                )
    
                if signal == 1 and diff > 0:
                    st.session_state.status_slot.success(f"🔄 **Trend Flipped!** Diff: `{diff:,.2f}` | Time1: `{time1}s`\n\n🤖 **AI Signal:** Predicted movement DOWN (Reversal)")
                    # st.success(f"🔄 **Trend Flipped!** Diff: `{diff:,.2f}` | Time1: `{time1}s`\n\n🤖 **AI Signal:** Predicted movement DOWN (Reversal)")
                    st.session_state.demo_balance -= 10.0
                    st.session_state.trade_history.append({'time': str(last_timestamp), 'tnx': "-$10.00", 'type': 'loss'})
                elif signal == 1 and diff < 0:
                    st.session_state.status_slot.success(f"🔄 **Trend Flipped!** Diff: `{diff:,.2f}` | Time1: `{time1}s`\n\n🤖 **AI Signal:** Predicted movement UP (Reversal)")
                    # st.success(f"🔄 **Trend Flipped!** Diff: `{diff:,.2f}` | Time1: `{time1}s`\n\n🤖 **AI Signal:** Predicted movement UP (Reversal)")
                    st.session_state.demo_balance -= 10.0
                    st.session_state.trade_history.append({'time': str(last_timestamp), 'tnx': "-$10.00", 'type': 'loss'})
                elif signal == 0:
                    st.session_state.status_slot.info(f"🔄 **Trend Flipped!** Diff: `{diff:,.2f}` | Time1: `{time1}s`\n\n🤖 **AI Signal:** Market Noise / No Trade Trigger")
                    # st.info(f"🔄 **Trend Flipped!** Diff: `{diff:,.2f}` | Time1: `{time1}s`\n\n🤖 **AI Signal:** Market Noise / No Trade Trigger")
        
            st.session_state.prev_trend = trend
            st.session_state.initial_price = current_price
            st.session_state.initial_time = last_timestamp
        
        st.session_state.prev_price = current_price

        min_price = prices.min() - 50
        max_price = prices.max() + 50

        future_1min = last_timestamp + pd.Timedelta(minutes=1)
        future_2min = last_timestamp + pd.Timedelta(minutes=2)

        new_proj_x = [first_timestamp, future_1min]
        new_proj_y = [current_price, current_price]

        if (
                not st.session_state.projection_history
                or st.session_state.projection_history[-1]["x"][1] != future_1min
            ):
                if signal == 1 and diff > 0 if 'diff' in locals() else False:
                    st.session_state.projection_history.append(
                        {"x": new_proj_x, "y": new_proj_y, "color": 'red'}
                    )
                elif signal == 1 and diff < 0 if 'diff' in locals() else False:
                    st.session_state.projection_history.append(
                        {"x": new_proj_x, "y": new_proj_y, "color": 'green'}
                    )
                else:
                    st.session_state.projection_history.append(
                        {"x": new_proj_x, "y": new_proj_y, "color": 'gray'}
                    )

        if len(st.session_state.projection_history) > 2:
                st.session_state.projection_history.pop(0)

        fig = px.line(
                x=prices.index,
                y=prices.values,
                labels={"x": "Time", "y": "Price (USD)"},
                title="Live Price Action with State Machine Tracking",
            )

        for idx, proj in enumerate(st.session_state.projection_history):
                is_latest = idx == len(st.session_state.projection_history) - 1
                line_color = proj["color"]
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

        if "chart_slot" not in st.session_state:
            st.session_state.chart_slot = st.empty()
            
        st.session_state.chart_slot.plotly_chart(fig, use_container_width=True, key="live_btc_chart")
     # --- LEFT COLUMN: WALLET & BALANCE ---
        with left_col:
            st.subheader("Account & Wallet")
            st.metric(label="Demo Account Balance", value=f"${st.session_state.demo_balance:,.2f}", delta="Fixed $10 Stake / Signal")
            st.markdown("---")
            st.markdown("### Active Rules")
            st.markdown("- Bet amount: **$10**")
            st.markdown("- Win Payout: **185%**")
            st.markdown("- Min Interval: **1 Min**")
    # --- RIGHT COLUMN: TRANSACTION HISTORY ---
    with right_col:
        st.subheader("History")
        if not st.session_state.trade_history:
            st.info("No transactions yet.")
        else:
            for trade in reversed(st.session_state.trade_history):
                t_time = trade['time']
                t_tnx = trade['tnx']
                t_type = trade.get('type', 'neutral')
                
                if t_type == 'win':
                    st.markdown(f"<span style='color:green; font-weight:bold;'>[{t_time}] {t_tnx}</span>", unsafe_allow_html=True)
                elif t_type == 'loss':
                    st.markdown(f"<span style='color:red; font-weight:bold;'>[{t_time}] {t_tnx}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span>[{t_time}] {t_tnx}</span>", unsafe_allow_html=True)

else:
    st.warning("Waiting for data feed...")

# --- NON-BLOCKING COUNTDOWN TIMER REPLACEMENT ---
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