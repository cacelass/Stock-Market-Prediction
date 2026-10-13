from __future__ import annotations
from dataclasses import dataclass
from typing import List
import json


@dataclass
class Alert:
    level: str
    message: str


def emit_alerts(results: dict) -> List[Alert]:
    alerts: List[Alert] = []
    if results.get("drift_detected"):
        alerts.append(Alert(level="WARN", message="Feature drift detected"))
    else:
        alerts.append(Alert(level="INFO", message="No drift detected"))
    return alerts


def alerts_to_json(alerts: List[Alert]) -> str:
    return json.dumps([{"level": a.level, "message": a.message} for a in alerts])
