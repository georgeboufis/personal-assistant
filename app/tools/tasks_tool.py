"""
tasks_tool.py
--------------
Τα εργαλεία για τη λίστα εργασιών (Google Tasks).

Γιατί χρειάζεται ξεχωριστά από το ημερολόγιο:
Το ημερολόγιο απαντάει στο "ΠΟΤΕ συμβαίνει κάτι" - έχει αρχή και τέλος,
και σε δεσμεύει σε συγκεκριμένη ώρα. Η λίστα εργασιών απαντάει στο "ΤΙ
πρέπει να κάνω" - μπορεί να έχει προθεσμία, αλλά δεν καταλαμβάνει χρόνο.

"Συνέντευξη Τετάρτη 12:00" -> ημερολόγιο.
"Στείλε το CV μέχρι την Παρασκευή" -> εργασία.

ΣΗΜΑΝΤΙΚΟΣ ΠΕΡΙΟΡΙΣΜΟΣ ΤΟΥ GOOGLE TASKS API:
Το πεδίο "due" δέχεται μεν πλήρες RFC3339 timestamp, αλλά η Google
ΑΓΝΟΕΙ ΤΗΝ ΩΡΑ και κρατάει μόνο την ημερομηνία. Αν στείλεις "Παρασκευή
17:00", θα αποθηκευτεί ως "Παρασκευή" σκέτο.

Αυτό δεν είναι δικό μας bug - είναι σχεδιαστική επιλογή της Google. Το
χειριζόμαστε ρητά: κανονικοποιούμε την είσοδο σε ημερομηνία και το
δηλώνουμε στο docstring, ώστε ούτε το μοντέλο ούτε ο χρήστης να περιμένει
κάτι που δεν θα συμβεί. Αν χρειάζεται συγκεκριμένη ώρα, ανήκει στο
ημερολόγιο, όχι εδώ.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from googleapiclient.discovery import build
from langchain_core.tools import tool

from app.core import google_auth

USER_TIMEZONE = ZoneInfo("Europe/Athens")

# Το "@default" είναι ειδικό αναγνωριστικό της Google για την προεπιλεγμένη
# λίστα εργασιών του χρήστη. Έτσι δεν χρειάζεται να ρωτήσουμε πρώτα ποιες
# λίστες υπάρχουν - λειτουργεί για κάθε λογαριασμό.
DEFAULT_TASKLIST = "@default"


def _get_tasks_service():
    """Χτίζει ένα Google Tasks API service object."""
    creds = google_auth.get_google_credentials()
    return build("tasks", "v1", credentials=creds)


def _normalize_due_date(due_date: str) -> str:
    """
    Μετατρέπει μια ημερομηνία προθεσμίας στη μορφή που θέλει το API.

    Δεχόμαστε είτε "2026-09-05" είτε πλήρες timestamp, και επιστρέφουμε
    πάντα μεσάνυχτα UTC εκείνης της ημέρας - γιατί, όπως εξηγείται στο
    docstring του module, η Google θα πετάξει την ώρα ούτως ή άλλως.
    Κρατώντας τη σταθερή, αποφεύγουμε το φαινόμενο όπου μια προθεσμία
    "Παρασκευή 23:00 ώρα Ελλάδας" καταλήγει Πέμπτη σε UTC.
    """
    date_part = due_date.strip()[:10]  # κρατάμε μόνο το YYYY-MM-DD
    # Επικυρώνουμε ότι είναι όντως ημερομηνία - αν όχι, σκάει εδώ με
    # καθαρό μήνυμα αντί να το απορρίψει το Google API με κρυπτικό error.
    datetime.strptime(date_part, "%Y-%m-%d")
    return f"{date_part}T00:00:00.000Z"


def _format_task(task: dict) -> str:
    """Μετατρέπει μια εργασία σε ευανάγνωστη γραμμή."""
    title = task.get("title") or "(χωρίς τίτλο)"
    parts = [f"- ID: {task.get('id', '')}", f"  Τίτλος: {title}"]

    due = task.get("due")
    if due:
        # Το API επιστρέφει RFC3339· κρατάμε μόνο την ημερομηνία, αφού η
        # ώρα δεν έχει νόημα σε αυτό το API.
        parts.append(f"  Προθεσμία: {due[:10]}")

    notes = task.get("notes")
    if notes:
        parts.append(f"  Σημειώσεις: {notes[:200]}")

    return "\n".join(parts)


def get_pending_tasks(max_results: int = 20) -> list[str]:
    """
    Επιστρέφει τις εκκρεμείς εργασίες, μορφοποιημένες.

    ΔΕΝ είναι @tool - το καλεί κατευθείαν η πρωινή ενημέρωση, όπου θέλουμε
    ντετερμινιστικό αποτέλεσμα και όχι απόφαση του μοντέλου.
    """
    service = _get_tasks_service()

    result = (
        service.tasks()
        .list(
            tasklist=DEFAULT_TASKLIST,
            showCompleted=False,
            showHidden=False,
            maxResults=max_results,
        )
        .execute()
    )

    lines = []
    for task in result.get("items", []):
        title = task.get("title") or "(χωρίς τίτλο)"
        due = task.get("due")
        lines.append(f"{title} (προθεσμία: {due[:10]})" if due else title)

    return lines


@tool
def list_tasks(include_completed: bool = False, max_results: int = 20) -> str:
    """
    Επιστρέφει τις εργασίες (to-do) του χρήστη από το Google Tasks.

    Χρησιμοποίησε αυτό όταν ο χρήστης ρωτάει τι έχει να κάνει, ποιες
    εκκρεμότητες έχει, ή τι προθεσμίες τον περιμένουν. Για ραντεβού και
    συναντήσεις με συγκεκριμένη ώρα, χρησιμοποίησε το list_upcoming_events.

    Args:
        include_completed: αν True, δείχνει και τις ολοκληρωμένες.
        max_results: πόσες το πολύ (default 20).
    """
    service = _get_tasks_service()

    result = (
        service.tasks()
        .list(
            tasklist=DEFAULT_TASKLIST,
            showCompleted=include_completed,
            showHidden=include_completed,
            maxResults=max_results,
        )
        .execute()
    )
    items = result.get("items", [])

    if not items:
        return "Δεν υπάρχουν εργασίες στη λίστα."

    formatted = []
    for task in items:
        line = _format_task(task)
        if task.get("status") == "completed":
            line += "\n  Κατάσταση: ολοκληρωμένη"
        formatted.append(line)

    return "\n\n".join(formatted)


@tool
def create_task(title: str, due_date: str = "", notes: str = "") -> str:
    """
    Δημιουργεί νέα εργασία (to-do) στη λίστα του χρήστη.

    Χρησιμοποίησε αυτό όταν ο χρήστης θέλει να θυμάται να κάνει κάτι, ή
    όταν ένα email αναφέρει ενέργεια που πρέπει να γίνει μέχρι κάποια
    προθεσμία (π.χ. "στείλτε μας τα δικαιολογητικά μέχρι τη Δευτέρα").

    ΠΡΟΣΟΧΗ - το Google Tasks αποθηκεύει ΜΟΝΟ ΗΜΕΡΟΜΗΝΙΑ στην προθεσμία,
    όχι ώρα. Αν ο χρήστης χρειάζεται κάτι σε συγκεκριμένη ώρα (π.χ.
    "τηλεφώνησε στις 3"), αυτό ανήκει στο ημερολόγιο - χρησιμοποίησε το
    create_calendar_event.

    Args:
        title: τι πρέπει να γίνει (π.χ. "Αποστολή CV στην Accenture").
        due_date: προθεσμία σε μορφή YYYY-MM-DD (π.χ. "2026-09-05").
            Άφησέ το κενό αν δεν υπάρχει προθεσμία.
        notes: προαιρετικές λεπτομέρειες ή συμφραζόμενα.
    """
    service = _get_tasks_service()

    body = {"title": title}
    if notes:
        body["notes"] = notes
    if due_date:
        body["due"] = _normalize_due_date(due_date)

    created = (
        service.tasks().insert(tasklist=DEFAULT_TASKLIST, body=body).execute()
    )

    message = f"Δημιουργήθηκε η εργασία '{title}'"
    if due_date:
        message += f" με προθεσμία {due_date[:10]}"
    return f"{message}. (ID: {created.get('id', 'άγνωστο')})"


@tool
def complete_task(task_id: str) -> str:
    """
    Σημειώνει μια εργασία ως ολοκληρωμένη.

    Χρησιμοποίησε αυτό όταν ο χρήστης λέει ότι έκανε κάτι από τη λίστα του.
    Το task_id το παίρνεις από τα αποτελέσματα του list_tasks - εμφανίζεται
    ως "ID:" σε κάθε εργασία.

    Args:
        task_id: το αναγνωριστικό της εργασίας.
    """
    service = _get_tasks_service()

    # Χρησιμοποιούμε patch (μερική ενημέρωση) αντί για update (πλήρης
    # αντικατάσταση), ώστε να μη χρειάζεται να στείλουμε ξανά όλα τα πεδία
    # - και να μην κινδυνεύουμε να σβήσουμε κατά λάθος τις σημειώσεις.
    updated = (
        service.tasks()
        .patch(
            tasklist=DEFAULT_TASKLIST,
            task=task_id,
            body={"status": "completed"},
        )
        .execute()
    )

    return f"Η εργασία '{updated.get('title', task_id)}' σημειώθηκε ως ολοκληρωμένη."


@tool
def delete_task(task_id: str) -> str:
    """
    Διαγράφει οριστικά μια εργασία από τη λίστα.

    Χρησιμοποίησε αυτό ΜΟΝΟ όταν ο χρήστης θέλει να εξαφανίσει εντελώς μια
    εργασία (π.χ. την πρόσθεσε κατά λάθος). Αν απλώς την ολοκλήρωσε,
    χρησιμοποίησε το complete_task - διατηρεί το ιστορικό.

    Args:
        task_id: το αναγνωριστικό της εργασίας.
    """
    service = _get_tasks_service()
    service.tasks().delete(tasklist=DEFAULT_TASKLIST, task=task_id).execute()
    return f"Η εργασία {task_id} διαγράφηκε οριστικά."
