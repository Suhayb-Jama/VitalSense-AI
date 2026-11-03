"""Start VitalSense AI with one command:  python run_all.py"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
for needed in ("models/failure_model.joblib", "models/anomaly_model.joblib"):
    if not (ROOT / needed).exists():
        sys.exit(f"Missing {needed}. Finish Phases 6 and 8 first.")


def main():
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000"], cwd=ROOT)

    print("Waiting for API...")
    for _ in range(60):
        if api.poll() is not None:
            sys.exit("API crashed on startup. Read the error above.")
        try:
            if requests.get("http://127.0.0.1:8000/health", timeout=1).ok:
                break
        except Exception:
            time.sleep(1)
    else:
        api.terminate()
        sys.exit("API did not start in time.")

    dash = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
                             "--server.port", "8501", "--server.headless", "true"], cwd=ROOT)
    time.sleep(4)
    webbrowser.open("http://localhost:8501")
    print("\nVitalSense AI is running at http://localhost:8501  (Ctrl+C to stop)\n")
    try:
        dash.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in (dash, api):
            p.terminate()


if __name__ == "__main__":
    main()