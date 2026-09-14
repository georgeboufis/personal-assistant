"""
logging_config.py
------------------
Κεντρική ρύθμιση της καταγραφής (logging) για όλη την εφαρμογή.

ΓΙΑΤΙ ΤΟ ΧΡΕΙΑΖΟΜΑΣΤΕ:
Ο βοηθός κάνει πράγματα όταν δεν κοιτάς - κυρίως η πρωινή ενημέρωση στις
08:30. Αν αποτύχει εκεί, χωρίς logging δεν το μαθαίνεις ποτέ: απλά μια
μέρα αναρωτιέσαι γιατί δεν ήρθε.

Επίσης, το execute_tool_calls πιάνει κάθε εξαίρεση και τη μετατρέπει σε
ευγενική απάντηση προς τον χρήστη. Καλό για την εμπειρία χρήσης, αλλά
σημαίνει ότι τα σφάλματα ΕΞΑΦΑΝΙΖΟΝΤΑΙ. Μπορεί ένα εργαλείο να αποτυγχάνει
συνεχώς και να μην το υποψιάζεσαι. Το logging τα κάνει ορατά χωρίς να
χαλάσει την εμπειρία.

ΙΔΙΩΤΙΚΟΤΗΤΑ - σημαντική σχεδιαστική απόφαση:
Ο βοηθός διαβάζει τα emails σου. Αν καταγράφαμε περιεχόμενο, θα φτιάχναμε
ένα αρχείο στον δίσκο με αντίγραφα της αλληλογραφίας σου - χωρίς
κρυπτογράφηση, που μεγαλώνει συνεχώς, και που εύκολα ξεχνάς ότι υπάρχει.

Γι' αυτό καταγράφουμε ΜΕΤΑΔΕΔΟΜΕΝΑ, όχι περιεχόμενο:
  - ΝΑΙ: ποιο εργαλείο κλήθηκε, πόσο έκανε, πέτυχε, πόσους χαρακτήρες
    επέστρεψε, ποιο session
  - ΟΧΙ: κείμενο email, θέματα, ονόματα επαφών, τίτλοι event

Αν χρειαστείς αναλυτική καταγραφή για debugging, υπάρχει η ρύθμιση
LOG_CONTENT=true - αλλά είναι απενεργοποιημένη από προεπιλογή και καλό
είναι να μένει έτσι.
"""

import logging
import logging.handlers
from pathlib import Path

from app.config import get_settings

# Πού γράφονται τα logs. Ο φάκελος είναι ήδη στο .gitignore (*.log).
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "assistant.log"

# Μέγιστο μέγεθος αρχείου πριν την περιστροφή, και πόσα παλιά κρατάμε.
# Χωρίς αυτό, ένα log αρχείο μεγαλώνει επ' άπειρον μέχρι να γεμίσει τον
# δίσκο - κλασικό πρόβλημα σε συστήματα που τρέχουν για μήνες.
MAX_BYTES = 2 * 1024 * 1024   # 2 MB ανά αρχείο
BACKUP_COUNT = 3              # κρατάμε 3 παλιότερα -> το πολύ ~8 MB συνολικά

# Το πρόθεμα όλων των logger μας. Χρησιμοποιώντας ιεραρχία ονομάτων
# ("assistant.tools", "assistant.briefing", κλπ) μπορούμε να ρυθμίσουμε
# όλα μαζί, και να τα ξεχωρίζουμε εύκολα από τα logs του uvicorn.
ROOT_LOGGER_NAME = "assistant"

_configured = False


def setup_logging() -> None:
    """
    Ρυθμίζει το logging. Καλείται μία φορά, στην εκκίνηση του server.

    Δύο προορισμοί:
      - Κονσόλα: για όταν δουλεύεις και κοιτάς το terminal.
      - Αρχείο:  για όλα τα υπόλοιπα - κυρίως ό,τι συμβαίνει στις 08:30.
    """
    global _configured
    if _configured:
        return

    settings = get_settings()
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(settings.log_level.upper())

    # Εμποδίζουμε τη διάδοση προς τον root logger, ώστε τα μηνύματά μας να
    # μην εμφανίζονται δύο φορές (μία από εμάς, μία από τη ρύθμιση του
    # uvicorn που πιάνει τον root).
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-5s %(name)-20s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if settings.log_to_file:
        LOG_DIR.mkdir(exist_ok=True)
        # Ο RotatingFileHandler κόβει το αρχείο όταν φτάσει το MAX_BYTES και
        # κρατάει περιορισμένο αριθμό παλιότερων - έτσι το logging δεν
        # μπορεί ποτέ να γεμίσει τον δίσκο.
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _configured = True
    logger.info("logging ενεργό (level=%s, αρχείο=%s)",
                settings.log_level.upper(), settings.log_to_file)


def get_logger(name: str) -> logging.Logger:
    """
    Επιστρέφει logger για ένα υποσύστημα.

    Χρήση: get_logger("tools"), get_logger("briefing"), κλπ.
    Το πρόθεμα "assistant." μπαίνει αυτόματα.
    """
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def safe(text: str, limit: int = 80) -> str:
    """
    Ετοιμάζει κείμενο για καταγραφή, σεβόμενο την ιδιωτικότητα.

    Αν το LOG_CONTENT είναι απενεργοποιημένο (προεπιλογή), επιστρέφει μόνο
    το μήκος - αρκετό για να καταλάβεις "ήρθε άδειο;" ή "ήταν τεράστιο;"
    χωρίς να γράψεις στον δίσκο το περιεχόμενο των email σου.
    """
    if text is None:
        return "<κενό>"

    if not get_settings().log_content:
        return f"<{len(text)} χαρακτήρες>"

    cleaned = text.replace("\n", " ")
    if len(cleaned) > limit:
        return f"{cleaned[:limit]}..."
    return cleaned
