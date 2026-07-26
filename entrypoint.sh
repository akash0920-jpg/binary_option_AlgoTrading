#!/bin/sh
set -e

echo "==> Training models on latest 5-day BTC-USD window (model_train.py)..."
python model_train.py

echo "==> Training complete. Launching Streamlit dashboard..."
exec streamlit run sim.py \
    --server.port=8501 \
    --server.address=0.0.0.0
