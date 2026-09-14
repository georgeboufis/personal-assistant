"""
scheduler.py
-------------
Προγραμματισμένες εργασίες - προς το παρόν μόνο η πρωινή ενημέρωση.

Γιατί BackgroundScheduler και όχι AsyncIOScheduler:
Οι κλήσεις μας προς τα Google APIs και το Gemini είναι σύγχρονες
(blocking). Αν τις τρέχαμε στο event loop του FastAPI, θα "πάγωναν" τον
server για όσο διαρκεί η ενημέρωση (μερικά δευτερόλεπτα). Ο
BackgroundScheduler τρέχει σε δικό του thread, οπότε ο server συνεχίζει
να απαντάει κανονικά όσο φτιάχνεται η ενημέρωση.

ΠΡΑΚΤΙΚΟΣ ΠΕΡΙΟΡΙΣΜΟΣ που πρέπει να ξέρεις:
Ο scheduler ζει μέσα στη διεργασία του uvicorn. Αν ο server δεν τρέχει
στις 8:30 (κλειστό Mac, σε ύπνο, ή απλά σταματημένος server), η εργασία
ΔΕΝ θα εκτελεστεί - δεν υπάρχει τρόπος να "τρέξει αργότερα μόνη της".
Γι' αυτό το UI δημιουργεί την ενημέρωση on-demand όταν την ανοίξεις:
έτσι τη βλέπεις ούτως ή άλλως, απλά λίγο αργότερα.
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import get_settings
from app.core.briefing import USER_TIMEZONE, run_scheduled_briefing
from app.core.logging_config import get_logger

log = get_logger("scheduler")

_scheduler: BackgroundScheduler | None = None


def start_scheduler() -> None:
    """
    Ξεκινάει τον scheduler και καταχωρεί την πρωινή ενημέρωση.

    Καλείται μία φορά, κατά την εκκίνηση του FastAPI.
    """
    global _scheduler

    settings = get_settings()

    if not settings.briefing_enabled:
        log.info("η πρωινή ενημέρωση είναι απενεργοποιημένη")
        return

    if _scheduler is not None:
        return  # ήδη τρέχει

    _scheduler = BackgroundScheduler(timezone=USER_TIMEZONE)

    _scheduler.add_job(
        run_scheduled_briefing,
        trigger=CronTrigger(
            hour=settings.briefing_hour,
            minute=settings.briefing_minute,
            timezone=USER_TIMEZONE,
        ),
        kwargs={"send_email": settings.briefing_send_email},
        id="morning_briefing",
        name="Πρωινή ενημέρωση",
        # Αν ο server ήταν κλειστός την ώρα εκτέλεσης και ανοίξει λίγο
        # αργότερα, ο APScheduler θα προσπαθούσε να "προλάβει" τη χαμένη
        # εκτέλεση. Το misfire_grace_time λέει πόσο αργά είναι ακόμα
        # χρήσιμο (1 ώρα) - μετά από αυτό, το παραλείπει (μια ενημέρωση
        # στις 14:00 για το πρωί δεν έχει νόημα).
        misfire_grace_time=3600,
        # Αν για κάποιο λόγο συσσωρευτούν πολλές εκτελέσεις, τρέξε μόνο μία.
        coalesce=True,
        replace_existing=True,
    )

    _scheduler.start()
    log.info(
        "πρωινή ενημέρωση προγραμματίστηκε για %02d:%02d (ώρα Ελλάδας)",
        settings.briefing_hour,
        settings.briefing_minute,
    )


def stop_scheduler() -> None:
    """Σταματάει τον scheduler καθαρά, κατά τον τερματισμό του server."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        log.info("ο scheduler σταμάτησε")
