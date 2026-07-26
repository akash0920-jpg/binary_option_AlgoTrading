# ---------------------------------------------------------------------------
# Dockerfile for binary_option_AlgoTrading
# Runs the Streamlit live-inference dashboard (Inference/sim.py) which loads
# the pre-trained KNN + Random Forest ensemble and streams BTC-USD signals.
#
# NOTE: "Train and test/binomo_deploy.py" drives a real, visible browser via
# Playwright + pyautogui to click buttons on Binomo's live trading site. That
# script needs a real display/mouse and a broker login, so it is NOT wired
# into this image — it isn't something that can (or should) run headless in
# a container. This Dockerfile packages the safe, self-contained part of the
# project: the model + the simulation dashboard.
#
# On every container start, entrypoint.sh runs model_train.py first to pull
# a fresh 5-day BTC-USD window from yfinance and retrain knn/rf/scaler, then
# launches the dashboard on those fresh models. This means the container
# needs outbound internet access at RUNTIME (not just build time) to reach
# Yahoo Finance.
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# Prevent .pyc files and force stdout/stderr to be unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by pandas/numpy/scikit-learn wheels + healthcheck curl
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy what's needed: the Inference folder (model_train.py, sim.py, and the
# checked-in .pkl files as a fallback) plus the entrypoint script
COPY Inference/ ./Inference/
COPY entrypoint.sh ./Inference/entrypoint.sh

WORKDIR /app/Inference
RUN chmod +x entrypoint.sh

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# On startup: retrain knn/rf/scaler on a fresh 5-day yfinance pull, then
# launch the dashboard using those freshly trained models.
ENTRYPOINT ["./entrypoint.sh"]
