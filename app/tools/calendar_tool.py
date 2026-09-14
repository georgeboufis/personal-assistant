"""
calendar_tool.py
-----------------
Εδώ ορίζουμε τα "εργαλεία" (tools) που δίνουμε στον agent για να
αλληλεπιδρά με το Google Calendar.

Τι είναι ένα "LangChain tool";
Είναι μια κανονική Python function με ΔΥΟ ειδικά χαρακτηριστικά:
  1. Το decorator @tool τη "σημαδεύει" ως κάτι που το LLM μπορεί να
     επιλέξει να καλέσει.
  2. Το docstring ΚΑΙ τα type hints της function ΕΙΝΑΙ αυτά που διαβάζει
     το Gemini για να καταλάβει "τι κάνει αυτό το εργαλείο" και "τι
     παραμέτρους χρειάζεται" - ΔΕΝ είναι απλά σχόλια για εμάς τους
     προγραμματιστές, είναι μέρος του "API" που βλέπει το μοντέλο!
     Γι' αυτό τα docstrings εδώ είναι πιο περιγραφικά απ' ό,τι σε άλλα
     αρχεία - το κοινό τους δεν είναι μόνο άνθρωπος, είναι και το LLM.

Γιατί χτίζουμε το service object σε κάθε κλήση (όχι μόνιμο singleton):
Το get_google_credentials() είναι ήδη γρήγορο όταν το token είναι
έγκυρο (απλά διαβάζει ένα μικρό αρχείο), και μας εξασφαλίζει ότι ΠΑΝΤΑ
έχουμε ενεργό/ανανεωμένο token πριν από κάθε κλήση στο πραγματικό Google
API - προτιμάμε αυτή τη μικρή, αμελητέα επανάληψη αντί να κινδυνέψουμε
να χρησιμοποιήσουμε ληγμένο token από ένα παλιό cached service object.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from googleapiclient.discovery import build
from langchain_core.tools import tool

from app.core import google_auth

# Ζώνη ώρας χρήστη - χρησιμοποιείται ώστε το Google Calendar να
# επιστρέφει/δέχεται ώρες στη σωστή τοπική ώρα. Ονομασία IANA (όχι
# σταθερό offset) ώστε να χειρίζεται αυτόματα θερινή/χειμερινή ώρα.
USER_TIMEZONE_NAME = "Europe/Athens"
USER_TIMEZONE = ZoneInfo(USER_TIMEZONE_NAME)


def _get_calendar_service():
    """Χτίζει ένα Google Calendar API service object, έτοιμο για χρήση."""
    creds = google_auth.get_google_credentials()
    return build("calendar", "v3", credentials=creds)


def _format_event_start(event: dict) -> str:
    """
    Μετατρέπει την ώρα έναρξης ενός event σε ευανάγνωστη μορφή, στη ζώνη
    ώρας του χρήστη.

    Γιατί χρειάζεται: το Google API επιστρέφει ISO timestamps που μπορεί
    να είναι σε οποιαδήποτε ζώνη (συχνά UTC). Αν τα δείξουμε ως έχουν,
    ο χρήστης (και το LLM) μπορεί να μπερδευτεί - π.χ. ένα event στις
    18:00 ώρα Ελλάδας φαίνεται ως 15:00Z.
    """
    start = event["start"]

    # All-day events έχουν 'date' (μόνο ημερομηνία) αντί για 'dateTime'.
    if "date" in start:
        return f"{start['date']} (ολοήμερο)"

    dt = datetime.fromisoformat(start["dateTime"]).astimezone(USER_TIMEZONE)
    # Παράδειγμα εξόδου: "Τρίτη 01/09/2026 18:00"
    return dt.strftime("%A %d/%m/%Y %H:%M")


def get_events_for_day(day: datetime) -> list[str]:
    """
    Επιστρέφει τα events μιας ΣΥΓΚΕΚΡΙΜΕΝΗΣ ημέρας, μορφοποιημένα.

    ΔΕΝ είναι @tool - δεν το βλέπει το LLM. Το καλούμε εμείς κατευθείαν
    από την πρωινή ενημέρωση, όπου θέλουμε ντετερμινιστικό αποτέλεσμα και
    όχι απόφαση του μοντέλου για το τι να ζητήσει.

    Args:
        day: οποιαδήποτε ώρα μέσα στη ζητούμενη ημέρα.
    """
    service = _get_calendar_service()

    # Από 00:00 έως 23:59:59 της ημέρας, σε τοπική ώρα.
    day_local = day.astimezone(USER_TIMEZONE)
    start = day_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=start.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    lines = []
    for event in result.get("items", []):
        when = _format_event_start(event)
        title = event.get("summary", "(χωρίς τίτλο)")
        location = event.get("location", "")
        line = f"{when} - {title}"
        if location:
            line += f" (τοποθεσία: {location})"
        lines.append(line)

    return lines


@tool
def list_upcoming_events(max_results: int = 10) -> str:
    """
    Επιστρέφει τα επόμενα events από το Google Calendar του χρήστη,
    ταξινομημένα χρονολογικά (πιο κοντινό πρώτο).

    Χρησιμοποίησε αυτό το εργαλείο όταν ο χρήστης ρωτάει για το πρόγραμμά
    του, τι έχει σήμερα/αύριο/αυτή την εβδομάδα, ή γενικά για τυχόν
    προγραμματισμένα ραντεβού/events.

    Args:
        max_results: πόσα events να επιστραφούν το πολύ (default 10).
    """
    service = _get_calendar_service()

    # Το Google Calendar API θέλει το "now" σε RFC3339 format (ISO 8601
    # με timezone) - το .isoformat() από ένα timezone-aware datetime
    # παράγει ακριβώς αυτό.
    now = datetime.now(timezone.utc).isoformat()

    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])

    if not events:
        return "Δεν υπάρχουν προσεχή events στο calendar."

    lines = []
    for event in events:
        start = _format_event_start(event)
        title = event.get("summary", "(χωρίς τίτλο)")
        lines.append(f"- {start}: {title}")

    return "\n".join(lines)


@tool
def create_calendar_event(
    summary: str,
    start_time: str,
    end_time: str,
    description: str = "",
) -> str:
    """
    Δημιουργεί ένα νέο event στο Google Calendar του χρήστη.

    Χρησιμοποίησε αυτό το εργαλείο όταν ο χρήστης ζητάει να προγραμματίσεις
    ένα ραντεβού, meeting, υπενθύμιση, ή οποιοδήποτε event με συγκεκριμένη
    ώρα έναρξης/λήξης.

    Args:
        summary: ο τίτλος του event (π.χ. "Ραντεβού με τον δικηγόρο").
        start_time: ώρα έναρξης σε ISO 8601 format ΜΕ timezone,
            π.χ. "2026-09-01T15:00:00+03:00" (Ελλάδα = +03:00 το καλοκαίρι,
            +02:00 τον χειμώνα).
        end_time: ώρα λήξης, ίδιο format με το start_time.
        description: προαιρετική περιγραφή/σημειώσεις για το event.
    """
    service = _get_calendar_service()

    event_body = {
        "summary": summary,
        "description": description,
        # Δηλώνουμε ΡΗΤΑ το timeZone μαζί με το dateTime. Αν το LLM στείλει
        # timestamp χωρίς offset (π.χ. "2026-09-01T18:00:00"), το Google το
        # ερμηνεύει σε ΑΥΤΗ τη ζώνη αντί να υποθέσει UTC - ένα λιγότερο
        # σημείο όπου μπορεί να προκύψει λάθος ώρα.
        "start": {"dateTime": start_time, "timeZone": USER_TIMEZONE_NAME},
        "end": {"dateTime": end_time, "timeZone": USER_TIMEZONE_NAME},
    }

    created_event = (
        service.events().insert(calendarId="primary", body=event_body).execute()
    )

    return (
        f"Δημιουργήθηκε το event '{summary}' "
        f"({start_time} - {end_time}). "
        f"Σύνδεσμος: {created_event.get('htmlLink', '(χωρίς link)')}"
    )
