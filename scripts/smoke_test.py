"""Phase 14: smoke tests for the full pipeline. Run: python scripts\\smoke_test.py"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.engine import analyze
from src.features import FEATURE_COLUMNS
from src.fleet import Fleet

results = []


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    results.append(ok)


# 1. No leakage columns in the model features
check("Features exclude PWF / UDI / Product ID / target",
      not any(c in FEATURE_COLUMNS for c in ("PWF", "UDI", "Product ID", "Machine failure")))

# 2. Healthy fleet stays healthy
fleet = Fleet()
bad = 0
for _ in range(60):
    fleet.tick()
    bad += sum(1 for u in fleet.summary()["units"] if u["status"] != "HEALTHY")
check(f"Idle fleet stays HEALTHY for 60 ticks (non-healthy readings: {bad} of 480)", bad <= 5)

# 3. Degrading unit goes HEALTHY -> WARNING -> CRITICAL
fleet.control("SERVER-03", "start")
statuses = []
for _ in range(80):
    fleet.tick()
    statuses.append(fleet.get("SERVER-03")["status"])
check("SERVER-03 reaches WARNING", "WARNING" in statuses)
check("SERVER-03 reaches CRITICAL", "CRITICAL" in statuses)
check("WARNING comes before CRITICAL",
      "WARNING" in statuses and "CRITICAL" in statuses
      and statuses.index("WARNING") < statuses.index("CRITICAL"))
print("      sequence:", " ".join(s[0] for s in statuses))

# 4. Alerts, explanation, recommendation
levels = [a["level"] for a in fleet.alert_list() if a["equipment"] == "SERVER-03"]
check("CRITICAL alert was raised for SERVER-03", "CRITICAL" in levels)
last = fleet.get("SERVER-03")
check("Explanation has a cause and an URGENT recommendation",
      bool(last["cause"]) and last["recommendation"].startswith("URGENT"))

# 5. Other units unaffected
others = [u["status"] for u in fleet.summary()["units"] if u["id"] != "SERVER-03"]
check("Other 7 units are still HEALTHY", all(s == "HEALTHY" for s in others))

# 6. Reset works
fleet.control("SERVER-03", "reset")
for _ in range(3):
    fleet.tick()
check("SERVER-03 returns to HEALTHY after reset", fleet.get("SERVER-03")["status"] == "HEALTHY")

# 7. Edge case: extreme values do not crash
try:
    analyze({"air_temp": 320.0, "process_temp": 340.0, "rpm": 100.0, "torque": 200.0, "tool_wear": 500.0, "type": "H"})
    check("Extreme out-of-range reading handled", True)
except Exception as e:
    check(f"Extreme out-of-range reading handled ({e})", False)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)