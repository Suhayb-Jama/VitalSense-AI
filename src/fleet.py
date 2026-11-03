"""Fleet manager: simulated equipment, live analysis, history and alerts (in memory)."""

import math
import threading
import time

from collections import deque
from datetime import datetime

from src.engine import analyze
from src.simulator import EquipmentSimulator
from src.anomaly import detect
from src.predict import predict_reading

TICK_SECONDS = 2.0
HISTORY_LEN = 90
RANK = {"HEALTHY": 0, "WARNING": 1, "CRITICAL": 2}

FLEET_SPECS = [  # (id, equipment class, starting wear)
    ("SERVER-01", "L", 40), ("SERVER-02", "M", 70), ("SERVER-03", "L", 60),
    ("SERVER-04", "H", 30), ("SERVER-05", "M", 55), ("SERVER-06", "L", 75),
    ("SERVER-07", "L", 50), ("SERVER-08", "M", 90),
]


class Fleet:
    def __init__(self):
        self.sims = {eid: EquipmentSimulator(eid, t, w, seed=i)
                     for i, (eid, t, w) in enumerate(FLEET_SPECS)}
        self.latest = {}
        self.history = {eid: deque(maxlen=HISTORY_LEN) for eid in self.sims}
        self.alerts = deque(maxlen=50)
        self.last_status = {eid: "HEALTHY" for eid in self.sims}
        self.anomaly_streak = {eid: 0 for eid in self.sims}
        self.lock = threading.Lock()
        self._running = False

    # ---- one simulation step for the whole fleet ----
    def tick(self):
        new = {}
        for eid, sim in self.sims.items():
            result = analyze(sim.step())
            result["equipment_id"] = eid
            new[eid] = result

        now = datetime.now()
        with self.lock:
            for eid, result in new.items():
                result["time"] = now.isoformat(timespec="seconds")
                self._update_alerts(eid, result, now)
                self.latest[eid] = result
                r = result["reading"]
                self.history[eid].append({
                    "time": result["time"],
                    "risk_pct": result["risk_pct"],
                    "status": result["status"],
                    "process_temp": r["process_temp"],
                    "rpm": r["rpm"],
                    "torque": r["torque"],
                    "tool_wear": r["tool_wear"],
                    "power_w": round(r["torque"] * r["rpm"] * 2 * math.pi / 60, 0),
                    "anomaly": result["anomaly"]["is_anomaly"],
                })

    def _add_alert(self, now, eid, level, message):
        self.alerts.appendleft({"time": now.strftime("%H:%M:%S"), "equipment": eid,
                                "level": level, "message": message})

    def _update_alerts(self, eid, result, now):
        status, prev = result["status"], self.last_status[eid]
        if RANK[status] > RANK[prev]:
            self._add_alert(now, eid, status,
                            f"{prev} -> {status} (risk {result['risk_pct']}%). "
                            f"{result['cause']} {result['recommendation']}")
        self.last_status[eid] = status

        if result["anomaly"]["is_anomaly"]:
            self.anomaly_streak[eid] += 1
            if self.anomaly_streak[eid] == 2:   # 2 in a row, to avoid one-off noise
                self._add_alert(now, eid, "ANOMALY",
                                "Unusual sensor pattern compared with normal operating behavior.")
        else:
            self.anomaly_streak[eid] = 0

    # ---- background loop ----
    def start(self):
        self.tick()
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            time.sleep(TICK_SECONDS)
            try:
                self.tick()
            except Exception as e:
                print("tick error:", e)

    # ---- API helpers ----
    def control(self, eid, action):
        sim = self.sims[eid]
        if action == "start":
            sim.start()
        elif action == "stop":
            sim.stop()
        elif action == "reset":
            sim.reset()
            with self.lock:
                self.history[eid].clear()
                self.last_status[eid] = "HEALTHY"
                self.anomaly_streak[eid] = 0

    def summary(self):
        with self.lock:
            units = []
            for eid, sim in self.sims.items():
                r = self.latest.get(eid)
                units.append({
                    "id": eid, "type": sim.type,
                    "status": r["status"] if r else "HEALTHY",
                    "risk_pct": r["risk_pct"] if r else 0.0,
                    "anomaly": r["anomaly"]["is_anomaly"] if r else False,
                    "degrading": sim.degrading,
                })
        count = lambda s: sum(1 for u in units if u["status"] == s)
        return {"total": len(units), "healthy": count("HEALTHY"),
                "warning": count("WARNING"), "critical": count("CRITICAL"), "units": units}

    def get(self, eid):
        with self.lock:
            return self.latest.get(eid)

    def history_of(self, eid):
        with self.lock:
            return list(self.history[eid])

    def alert_list(self):
        with self.lock:
            return list(self.alerts)