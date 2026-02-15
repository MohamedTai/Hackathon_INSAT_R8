print("MAIN.PY LOADED", flush=True)
import sys
from config import Settings
from utils import log
from handlers import handle_in, handle_out

def main():
    cfg = Settings()

    backend = None
    if cfg.use_backend:
        try:
            import requests  # vérif lib
            from backend_client import BackendClient
            backend = BackendClient(cfg.backend_base_url)
        except Exception as e:
            log(f"Backend disabled (init error): {e}")
            backend = None

    log("Keyboard simulation ready.")
    log("Press:")
    log("  I = simulate IN (entry)")
    log("  O = simulate OUT (exit)")
    log("  Q = quit")

    while True:
        print("WAITING FOR KEY...", flush=True)
        cmd = input("> ").strip().upper()
        if cmd == "Q":
            log("Bye.")
            sys.exit(0)
        elif cmd == "I":
            handle_in(backend=backend)
        elif cmd == "O":
            handle_out(backend=backend)
        else:
            log("Unknown command. Use I / O / Q.")

if __name__ == "__main__":
    main()
