from typing import Optional, Dict, Any
from utils import log

def recognize_stub() -> Dict[str, Any]:
    """
    Ici tu brancheras l'IA plus tard (OCR + dimensions).
    MVP: on simule des sorties.
    """
    # TODO: remplacer par OCR + estimation dimensions
    return {
        "item_id": "MAT-00001",
        "dims": {"L": 120.0, "W": 80.0, "H": 150.0},
        "est_weight": 200.0,
        "est_qty": 1.0
    }

def handle_in(backend=None):
    """
    Fonction exécutée quand tu simules une entrée.
    """
    log("IN triggered (keyboard). Running handle_in() ...")

    rec = recognize_stub()
    item_id = rec["item_id"]
    dims = rec["dims"]
    est_weight = rec["est_weight"]

    log(f"Recognized (stub): item_id={item_id}, dims={dims}, est_weight={est_weight}kg")

    # Exemple logique IN : demander un slot recommandé
    if backend:
        backend.post_event("IN")
        alloc = backend.allocate(
            L=dims["L"], W=dims["W"], H=dims["H"],
            est_weight=est_weight,
            item_id=item_id
        )
        log(f"Backend allocate -> {alloc}")
    else:
        log("Backend disabled. (No /event, no /allocate)")

    log("handle_in() done.")

def handle_out(backend=None):
    """
    Fonction exécutée quand tu simules une sortie.
    """
    log("OUT triggered (keyboard). Running handle_out() ...")

    rec = recognize_stub()
    item_id = rec["item_id"]

    # Exemple logique OUT : pour l’instant on log seulement
    # TODO: plus tard tu feras : mesure dimensions -> règle de trois -> quantité sortie
    log(f"Recognized (stub): item_id={item_id}")

    if backend:
        backend.post_event("OUT")
        log("Backend event OUT posted.")
    else:
        log("Backend disabled. (No /event)")

    log("handle_out() done.")
