from datetime import datetime

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def log(msg: str):
    print(f"[{now_iso()}] {msg}")
