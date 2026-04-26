import requests
import time
from datetime import datetime

# Replace this with your actual Streamlit app URL
APP_URL = "https://plantpathologydiagnosis.streamlit.app"

def ping_app():
    try:
        # We use a timeout to ensure the script doesn't hang
        response = requests.get(APP_URL, timeout=30)
        
        # Check if the request was successful
        if response.status_status == 200:
            print(f"[{datetime.now()}] Success: App is awake.")
        else:
            print(f"[{datetime.now()}] Warning: Received status code {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"[{datetime.now()}] Error: Could not reach the app. {e}")

if __name__ == "__main__":
    # If running as a continuous loop (e.g., on a VPS)
    # If running as a scheduled task, just call ping_app() once
    ping_app()
