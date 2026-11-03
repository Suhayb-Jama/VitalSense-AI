"""Simulated telemetry. This is NOT real data-center data."""

import numpy as np


class EquipmentSimulator:
    def __init__(self, equipment_id, equip_type="L", base_wear=60.0, seed=0):
        self.id = equipment_id
        self.type = equip_type
        self.base_wear = base_wear
        self.degradation = 0.0     # 0 = healthy, 1 = about to fail
        self.degrading = False
        self.rng = np.random.default_rng(seed)

    def start(self):
        self.degrading = True

    def stop(self):
        self.degrading = False

    def reset(self):
        self.degrading = False
        self.degradation = 0.0

    def step(self, rate=0.015):
        """Produce the next reading. ~67 steps (about 2 min at 2 s/step) to full degradation."""
        if self.degrading:
            self.degradation = min(1.0, self.degradation + rate)
        d = self.degradation
        n = self.rng.normal
        return {
            "air_temp": round(300.0 + 1.5 * d + n(0, 0.15), 2),
            "process_temp": round(310.5 + 3.0 * d + n(0, 0.2), 2),
            "rpm": round(1500 - 160 * d + n(0, 15), 1),
            "torque": round(40.0 + 28 * d + n(0, 1.5), 1),
            "tool_wear": round(self.base_wear + d * (215 - self.base_wear), 1),
            "type": self.type,
        }