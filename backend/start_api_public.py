"""
Start the MindSync API with an ngrok public HTTPS tunnel.

Usage:
    python start_api_public.py

This will:
    1. Start uvicorn on port 8000
    2. Open an ngrok tunnel
    3. Print the public URL to paste into app.json → extra.apiUrl
"""
import subprocess
import sys
import time
import threading
import json
import urllib.request
import os
from pathlib import Path

PORT = 8000
APP_JSON = Path(__file__).parent.parent / "app-mobile" / "app.json"


def start_api():
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", str(PORT)],
        cwd=str(Path(__file__).parent),
    )


def get_ngrok_url(retries=15, delay=1.5):
    """Poll the ngrok API until a tunnel URL appears."""
    for _ in range(retries):
        try:
            with urllib.request.urlopen("http://localhost:4040/api/tunnels") as r:
                data = json.loads(r.read())
                tunnels = data.get("tunnels", [])
                for t in tunnels:
                    if t.get("proto") == "https":
                        return t["public_url"]
        except Exception:
            pass
        time.sleep(delay)
    return None


if __name__ == "__main__":
    # 1 — Start uvicorn in a thread
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    print("⏳  Starting API server...")
    time.sleep(4)

    # 2 — Start ngrok
    ngrok_proc = subprocess.Popen(
        ["ngrok", "http", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print("⏳  Starting ngrok tunnel...")

    # 3 — Get public URL
    url = get_ngrok_url()
    if not url:
        print("❌  Could not get ngrok URL. Make sure you've authenticated:")
        print("    ngrok config add-authtoken <YOUR_TOKEN>")
        print("    (Get a free token at https://dashboard.ngrok.com/get-started/your-authtoken)")
        ngrok_proc.terminate()
        sys.exit(1)

    print(f"\n✅  Public API URL: {url}\n")
    print("=" * 60)
    print(f"  Paste this into app-mobile/app.json → extra.apiUrl:")
    print(f'  "apiUrl": "{url}"')
    print("=" * 60)
    print("\nThen rebuild the APK:")
    print("  cd app-mobile")
    print("  $env:EXPO_PUBLIC_API_URL = '{}'".format(url))
    print("  eas build --platform android --profile preview")
    print("\nPress Ctrl+C to stop.\n")

    try:
        api_thread.join()
    except KeyboardInterrupt:
        ngrok_proc.terminate()
        print("\nStopped.")
