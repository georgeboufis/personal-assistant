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


def get_events_for_day(day: datetime, include_past: bool = False) -> list[str]:
    """
    Επιστρέφει τα events μιας ΣΥΓΚΕΚΡΙΜΕΝΗΣ ημέρας, μορφοποιημένα.

    ΔΕΝ είναι @tool - δεν το βλέπει το LLM. Το καλούμε εμείς κατευθείαν
    από το dashboard και την πρωινή ενημέρωση.

    Args:
        day: οποιαδήποτε ώρα μέσα στη ζητούμενη ημέρα.
        include_past: αν False (προεπιλογή), αποκλείει events που έχουν
            ήδη τελειώσει μέχρι την ώρα του "day" - π.χ. αν ρωτήσεις στις
            13:47, δεν θα δεις ό,τι έγινε το πρωί. Αν True, δείχνει ΟΛΗ
            την ημέρα από τα μεσάνυχτα.
    """
    service = _get_calendar_service()

    # Από 00:00 έως 23:59:59 της ημέρας, σε τοπική ώρα.
    day_local = day.astimezone(USER_TIMEZONE)
    day_start = day_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end = day_start + timedelta(days=1)
    
    
    start = day_start if include_past else day_local

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

RECURRENCE_RULES = {
    "daily": "RRULE:FREQ=DAILY",
    "weekly": "RRULE:FREQ=WEEKLY",
    "monthly": "RRULE:FREQ=MONTHLY",
    "yearly": "RRULE:FREQ=YEARLY"
}


def _get_busy_intervals(day: datetime) -> list[tuple[datetime, datetime]]:
    """
    Επιστρέφει τα χρονικά διαστήματα (έναρξη, λήξη) που είναι ΗΔΗ
    κατειλημμένα από events μιας ημέρας, σε τοπική ώρα.

    ΔΕΝ είναι @tool - το χρησιμοποιεί εσωτερικά το find_free_time.
    Αγνοεί ολοήμερα events, αφού δεν έχουν συγκεκριμένη ώρα και δεν
    μπλοκάρουν κάποιο συγκεκριμένο χρονικό παράθυρο.
    """
    service = _get_calendar_service()

    day_start = day.astimezone(USER_TIMEZONE).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    day_end = day_start + timedelta(days=1)

    result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=day_start.isoformat(),
            timeMax=day_end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    intervals = []
    for event in result.get("items", []):
        start_raw = event["start"].get("dateTime")
        end_raw = event["end"].get("dateTime")
        if not start_raw or not end_raw:
            continue
        intervals.append(
            (
                datetime.fromisoformat(start_raw).astimezone(USER_TIMEZONE),
                datetime.fromisoformat(end_raw).astimezone(USER_TIMEZONE),
            )
        )
    return intervals 

@tool
def find_free_time(
    date: str,
    duration_minutes: int = 60,
    earliest_hour: int = 9,
    latest_hour: int = 21,
) -> str:
    """
    Βρίσκει ελεύθερα χρονικά διαστήματα σε μια συγκεκριμένη ημέρα.

    Χρησιμοποίησε αυτό όταν ο χρήστης ρωτάει πότε είναι ελεύθερος, ή πριν
    προτείνεις ώρα για νέο ραντεβού - ώστε να μην προτείνεις κάτι που
    συγκρούεται με υπάρχον event.

    Args:
        date: η ημερομηνία σε μορφή YYYY-MM-DD.
        duration_minutes: πόσα λεπτά χρειάζεται το ελεύθερο διάστημα
            (προεπιλογή 60).
        earliest_hour: από ποια ώρα να ψάξει, 24ωρη μορφή (προεπιλογή 9).
        latest_hour: μέχρι ποια ώρα να ψάξει, 24ωρη μορφή (προεπιλογή 21).
    """
    try:
        day = datetime.strptime(date.strip()[:10], "%Y-%m-%d").replace(
            tzinfo=USER_TIMEZONE
        )
    except ValueError:
        return f"Η ημερομηνία '{date}' δεν είναι έγκυρη. Χρησιμοποίησε μορφή YYYY-MM-DD."
    
    busy = _get_busy_intervals(day)
    duration = timedelta(minutes=duration_minutes)
    
    window_start = day.replace(hour=earliest_hour, minute=0)
    window_end = day.replace(hour=latest_hour, minute=0)

    free_slots = []
    cursor = window_start
    for busy_start, busy_end in busy:
        if busy_start > cursor and busy_start - cursor >= duration:
            free_slots.append((cursor, busy_start))
        cursor = max(cursor, busy_end)
    if window_end - cursor >= duration:
        free_slots.append((cursor, window_end))
    
    if not free_slots:
        return (
            f"Δεν βρέθηκε ελεύθερο διάστημα {duration_minutes} λεπτών στις "
            f"{date} μεταξύ {earliest_hour}:00-{latest_hour}:00."
        )
    
    lines = [f"Ελεύθερα διαστήματα στις {date} (τουλάχιστον {duration_minutes} λεπτά):"]
    for slot_start, slot_end in free_slots:
        lines.append(f"- {slot_start.strftime('%H:%M')} έως {slot_end.strftime('%H:%M')}")
    
    return "\n".join(lines)




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
        lines.append(f"- ID: {event.get('id', '')}\n  {start}: {title}")

    return "\n".join(lines)


@tool
def create_calendar_event(
    summary: str,
    start_time: str,
    end_time: str,
    description: str = "",
    repeat: str = "",
) -> str:
    """
    Δημιουργεί ένα νέο event στο Google Calendar του χρήστη.

    Χρησιμοποίησε αυτό το εργαλείο όταν ο χρήστης ζητάει να προγραμματίσεις
    ένα ραντεβού, meeting, μάθημα, ή οποιοδήποτε event με συγκεκριμένη
    ώρα έναρξης/λήξης.

    Args:
        summary: ο τίτλος του event (π.χ. "Ραντεβού με τον δικηγόρο").
        start_time: ώρα έναρξης σε ISO 8601 format ΜΕ timezone,
            π.χ. "2026-09-01T15:00:00+03:00".
        end_time: ώρα λήξης, ίδιο format με το start_time.
        description: προαιρετική περιγραφή/σημειώσεις για το event.
        repeat: αν το event ΕΠΑΝΑΛΑΜΒΑΝΕΤΑΙ, δώσε "daily", "weekly",
            "monthly" ή "yearly". Άφησέ το κενό για μονό event. Η
            επανάληψη γίνεται στην ίδια ημέρα/ώρα με το πρώτο event
            (π.χ. "weekly" σε event Τετάρτης 18:30 -> κάθε Τετάρτη στις
            18:30), απεριόριστα στο μέλλον εκτός αν ο χρήστης πει τέλος.
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

    rule = RECURRENCE_RULES.get(repeat.lower()) if repeat else None
    if rule:
        event_body["recurrence"] = [rule]

    
    created_event = (
        service.events().insert(calendarId="primary", body=event_body).execute()
    )

    message = (
        f"Δημιουργήθηκε το event '{summary}' "
        f"({start_time} - {end_time}). "
    )

    if rule:
        message += f", επαναλαμβανόμενο ({repeat})"
    message += f". Σύνδεσμος¨{created_event.get('htmlLink', '(χωρίς link)')}"
    return message

@tool
def delete_calendar_event(event_id: str) -> str:
    """
    Διαγράφει ΟΡΙΣΤΙΚΑ ένα event από το ημερολόγιο του χρήστη. Η ενέργεια
    ΔΕΝ αναιρείται.

    Χρησιμοποίησε αυτό ΜΟΝΟ όταν ο χρήστης ζητήσει ρητά να διαγράψεις ή
    να ακυρώσεις ένα συγκεκριμένο event. Το event_id το παίρνεις ΠΑΝΤΑ
    από τα αποτελέσματα του list_upcoming_events (εμφανίζεται ως "ID:")
    - ποτέ μην το μαντεύεις.

    Args:
        event_id: το αναγνωριστικό του event.
    """
    service = _get_calendar_service()
    service.events().delete(calendarId="primary", eventId=event_id).execute()
    return f"Το event {event_id} διαγράφηκε οριστικά από το ημερολόγιο."


@tool
def update_calendar_event(
    event_id: str,
    start_time: str = "",
    end_time: str = "",
    summary: str = "",
) -> str:
    """
    Τροποποιεί ΥΠΑΡΧΟΝ event - αλλάζει ώρα (μετακίνηση) ή/και τίτλο. Άφησε
    κενό ό,τι ΔΕΝ θέλεις να αλλάξεις.

    Χρησιμοποίησε αυτό όταν ο χρήστης θέλει να μετακινήσει ή να
    μετονομάσει ένα event που ήδη υπάρχει. Το event_id το παίρνεις ΠΑΝΤΑ
    από το list_upcoming_events - ποτέ μην το μαντεύεις.

    Args:
        event_id: το αναγνωριστικό του event προς τροποποίηση.
        start_time: νέα ώρα έναρξης σε ISO 8601 με timezone, ή κενό.
        end_time: νέα ώρα λήξης, ίδιο format, ή κενό.
        summary: νέος τίτλος, ή κενό.
    """
    service = _get_calendar_service()

    body = {}
    if summary:
        body["summary"] = summary
    if start_time:
        body["start"] = {"dateTime": start_time, "timeZone": USER_TIMEZONE_NAME}
    if end_time:
        body["end"] = {"dateTime": end_time, "timeZone": USER_TIMEZONE_NAME}

    if not body:
        return "Δεν δόθηκε καμία αλλαγή - πες μου τι θέλεις να τροποποιήσω."

    updated = (
        service.events()
        .patch(calendarId="primary", eventId=event_id, body=body)
        .execute()
    )
    return f"Το event '{updated.get('summary', event_id)}' ενημερώθηκε επιτυχώς."