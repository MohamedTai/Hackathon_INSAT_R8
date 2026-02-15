import time
import requests

class BackendClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def post_event(self, direction: str, source: str = "pc-keyboard"):
        payload = {
            "direction": direction.upper(),
            "ts_ms": int(time.time() * 1000),
            "source": source
        }
        r = requests.post(f"{self.base_url}/event", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()

    def allocate(self, L: float, W: float, H: float, est_weight: float = 0.0, item_id=None):
        payload = {
            "item_id": item_id,
            "L": float(L),
            "W": float(W),
            "H": float(H),
            "est_weight": float(est_weight)
        }
        r = requests.post(f"{self.base_url}/allocate", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
