"""VitalSense AI backend (final)."""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.engine import analyze
from src.fleet import Fleet
from src.predict import MODEL_NAME

fleet = Fleet()


@asynccontextmanager
async def lifespan(app: FastAPI):
    fleet.start()      # background telemetry simulation
    yield
    fleet.stop()


app = FastAPI(title="VitalSense AI API", version="1.0", lifespan=lifespan)


class Reading(BaseModel):
    air_temp: float
    process_temp: float
    rpm: float
    torque: float
    tool_wear: float
    type: str = "L"


class SimControl(BaseModel):
    action: Literal["start", "stop", "reset"]


def _check(eid: str):
    if eid not in fleet.sims:
        raise HTTPException(status_code=404, detail=f"Unknown equipment {eid}")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict")
def predict(reading: Reading):
    return analyze(reading.model_dump())


@app.get("/equipment")
def equipment():
    return fleet.summary()


@app.get("/equipment/{eid}")
def equipment_detail(eid: str):
    _check(eid)
    data = fleet.get(eid)
    if data is None:
        raise HTTPException(status_code=503, detail="No data yet")
    return data


@app.get("/equipment/{eid}/history")
def equipment_history(eid: str):
    _check(eid)
    return fleet.history_of(eid)


@app.post("/equipment/{eid}/simulation")
def equipment_simulation(eid: str, body: SimControl):
    _check(eid)
    fleet.control(eid, body.action)
    return {"equipment": eid, "action": body.action}


@app.get("/alerts")
def alerts():
    return fleet.alert_list()