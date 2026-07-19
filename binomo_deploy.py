import asyncio
import pickle
import time
import json
import numpy as np
import os
import pyautogui
from dotenv import load_dotenv
from playwright.async_api import async_playwright

# Load credentials
load_dotenv()
email = os.getenv("BINOMO_EMAIL")
password = os.getenv("BINOMO_PASSWORD")

# Load models
with open('knn_model.pkl', 'rb') as f: knn = pickle.load(f)
with open('rf_model.pkl', 'rb') as f: rf = pickle.load(f)
with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)

state = {
    'start_price': None,
    'start_time': None,
    'current_trend': None,
    'last_price': None
}

def parse_json(payload):
    try:
        data = json.loads(payload)
        # Deep path extraction based on your logs
        if 'data' in data:
            for entry in data['data']:
                if 'assets' in entry:
                    for asset in entry['assets']:
                        if asset.get('ric') == 'Z-CRY/IDX':
                            return float(asset.get('rate', 0.0))
        return 0.0
    except:
        return 0.0

async def get_signal(diff, time_elapsed):
    momentum = abs(diff / time_elapsed) if time_elapsed > 0 else 0
    accl = momentum / time_elapsed if time_elapsed > 0 else 0
    features = np.array([[time_elapsed, momentum, accl]])
    
    knn_probs = knn.predict_proba(scaler.transform(features))[:, 1]
    rf_probs = rf.predict_proba(features)[:, 1]
    
    if (knn_probs >= 0.50) and (rf_probs >= 0.50): return 1
    return 0

async def handle_price_update(page, price):
    if price == 0.0: return
    
    if state['start_price'] is None:
        state['start_price'] = price
        state['start_time'] = time.time()
        state['last_price'] = price
        return

    # Check trend reversal
    time_elapsed = time.time() - state['start_time']
    if time_elapsed >= 60:
        diff = price - state['start_price']
        new_trend = 1 if price > state['last_price'] else -1

        if new_trend != state['current_trend']:
            signal = await get_signal(diff, time_elapsed)
            
            # Print ONLY the required data
            print(f"Prediction: {signal} | Diff: {diff:.4f} | Time1: {time_elapsed:.2f}s")
            
            if signal == 1:
                if diff < 0: await page.click("#qa_trading_dealUpButton")
                else: await page.click("#qa_trading_dealDownButton")

        state['start_price'] = price
        state['start_time'] = time.time()
        state['current_trend'] = new_trend
    
    state['last_price'] = price

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        page.on("websocket", lambda ws: ws.on("framereceived", lambda p: asyncio.create_task(handle_price_update(page, parse_json(p)))))
        
        await page.goto("https://binomo.com/trading")
        
        # Non-blocking login
        await asyncio.sleep(15)
        pyautogui.click(1683,930); await asyncio.sleep(1)
        pyautogui.typewrite(email); await asyncio.sleep(0.5)
        pyautogui.click(1683,1021); await asyncio.sleep(0.5)
        pyautogui.typewrite(password); await asyncio.sleep(0.5)
        pyautogui.click(1659,1256)
        
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())