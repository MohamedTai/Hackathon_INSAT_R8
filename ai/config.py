from dataclasses import dataclass

@dataclass
class Settings:
    backend_base_url: str = "http://127.0.0.1:8000"
    use_backend: bool = True  # mets False si tu veux tester sans FastAPI
