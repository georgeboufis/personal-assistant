"""
main.py
-------
Entry point του FastAPI εφαρμογής.

Γιατί ξεκινάμε μόνο με health-check:
Πριν φτιάξουμε τον πραγματικό agent (LangGraph, tools, κλπ), θέλουμε να
επιβεβαιώσουμε ότι τα "θεμέλια" δουλεύουν: ότι το config φορτώνεται σωστά,
ότι το Redis είναι προσβάσιμο, και ότι το Gemini API key είναι έγκυρο.
Έτσι, αν κάτι δεν δουλεύει αργότερα, ξέρουμε ότι δεν είναι πρόβλημα
υποδομής (θα το έχουμε ήδη αποκλείσει).

Εκτέλεση (τοπικά στο Mac σου, ΟΧΙ στο sandbox):
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.core.redis_client import check_redis_connection
from app.core.memory import load_history, save_history
from app.core.briefing import generate_briefing, get_or_create_todays_briefing
from app.core.logging_config import get_logger, safe, setup_logging
from app.core.scheduler import start_scheduler, stop_scheduler
from app.core.confirmation import (
    build_decline_messages,
    get_pending_tool_calls,
    interpret_answer,
)
from app.agents.graph import describe_tool_call, execute_tool_calls, get_agent
from app.tools import calendar_tool, tasks_tool 
from app.core.briefing import USER_TIMEZONE

from datetime import datetime 

# Πού βρίσκεται ο φάκελος με τα στατικά αρχεία του UI.
# Το χτίζουμε ΣΧΕΤΙΚΑ με τη θέση αυτού του αρχείου (__file__) αντί να
# γράψουμε σκέτο "app/static" - έτσι δουλεύει σωστά ανεξάρτητα από το
# ποιος είναι ο τρέχων φάκελος όταν ξεκινάει ο server.
STATIC_DIR = Path(__file__).parent / "static"

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Τρέχει κώδικα στην εκκίνηση και στον τερματισμό του server.

    Εδώ ξεκινάμε τον scheduler (πρωινή ενημέρωση) και τον σταματάμε καθαρά
    όταν κλείνει ο server - αλλιώς θα έμενε "ζόμπι" thread στη μνήμη.
    """
    # Το logging πρώτο, ώστε ό,τι κάνει ο scheduler στην εκκίνηση να
    # καταγραφεί κανονικά.
    setup_logging()
    log.info("ο βοηθός ξεκίνησε")
    start_scheduler()
    yield
    stop_scheduler()
    log.info("ο βοηθός σταμάτησε")

app = FastAPI(
    title="AI Personal Assistant",
    description="Agent που σχεδιάζει ταξίδια, διαχειρίζεται calendar/email.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict:
    """
    Endpoint ελέγχου υγείας του συστήματος.

    Επιστρέφει την κατάσταση κάθε εξάρτησης ξεχωριστά (και όχι απλά
    "ok"/"error") ώστε αν κάτι χαλάσει να ξέρουμε ΑΜΕΣΩΣ ΤΙ χάλασε,
    χωρίς να χρειάζεται να ψάχνουμε logs.
    """
    settings = get_settings()

    return {
        "status": "ok",
        "app_env": settings.app_env,
        "checks": {
            "config_loaded": True,
            "gemini_api_key_present": bool(settings.google_api_key)
            and settings.google_api_key != "your_gemini_api_key_here",
            "redis_connected": check_redis_connection(),
        },
    }


@app.get("/briefing")
def get_briefing(refresh: bool = False) -> dict:
    """
    Επιστρέφει τη σημερινή ενημέρωση.

    Αν δεν έχει δημιουργηθεί ακόμα (π.χ. το Mac κοιμόταν στις 8:30), τη
    φτιάχνει εκείνη τη στιγμή. Έτσι δεν χάνεις ποτέ την ενημέρωση, απλά
    μπορεί να τη δεις λίγο αργότερα.

    Args:
        refresh: αν True, ξαναφτιάχνει την ενημέρωση από την αρχή ακόμα κι
            αν υπάρχει αποθηκευμένη - χρήσιμο αν άλλαξε κάτι μέσα στη μέρα.
    """
    text = generate_briefing() if refresh else get_or_create_todays_briefing()
    return {"briefing": text}

@app.get("/dashboard")
def get_dashboard() -> dict:
    """
    Γρήγορη σύνοψη της σημερινής ημέρας, για το sidebar του UI:
    τα ραντεβού και οι εκκρεμείς εργασίες.

    Δεν περνάει από το Gemini - χρησιμοποιεί τις ΙΔΙΕΣ helper functions
    με την πρωινή ενημέρωση (calendar_tool.get_events_for_day,
    tasks_tool.get_pending_tasks), που κάνουν απευθείας ανάγνωση από τα
    Google APIs. Γι' αυτό είναι ασφαλές να το καλεί το UI ελεύθερα,
    ακόμα και σε κάθε refresh - μηδενικό κόστος σε quota, καμία ενέργεια.
    """
    now = datetime.now(USER_TIMEZONE)
    
    try: 
        events = calendar_tool.get_events_for_day(now)
    except Exception as exc:
        log.error("Αποτυχία ανάγνωσης ημερολογίου για dashboard: %s", exc, exc_info=True)
        events = []
    
    try:
        tasks = tasks_tool.get_pending_tasks(max_results=10)
    except Exception as exc:
        log.error("Αποτυχία ανάγνωσης εκκρεμών εργασιών για dashboard: %s", exc, exc_info=True)
        tasks = []
    
    return {
        "date": now.strftime("%Y-%m-%d"),
        "events": events,
        "tasks": tasks,
    }

@app.get("/")
def serve_ui() -> FileResponse:
    """
    Σερβίρει το web UI (chat interface).

    Χρησιμοποιούμε FileResponse αντί να "mount"-άρουμε ολόκληρο φάκελο
    στατικών αρχείων, γιατί έχουμε ΕΝΑ μόνο αρχείο - πιο απλό, και δεν
    εκθέτουμε κατά λάθος άλλα αρχεία του φακέλου.

    Η τεκμηρίωση του API παραμένει διαθέσιμη στο /docs.
    """
    return FileResponse(STATIC_DIR / "index.html")


def _extract_text(content) -> str:
    """
    Εξάγει καθαρό, απλό κείμενο από το .content ενός μηνύματος.

    Γιατί χρειάζεται αυτό:
    Με τα μοντέλα Gemini 3.x, το .content ενός AIMessage δεν είναι πάντα
    απλό string - μπορεί να είναι μια ΛΙΣΤΑ από "content blocks", π.χ.
    [{"type": "text", "text": "...", "extras": {"signature": "..."}}].
    Το "signature" είναι εσωτερικό metadata (χρησιμοποιείται ώστε το
    Gemini να "θυμάται" το σκεπτικό του σε πολυβηματικές συνομιλίες με
    tools) - δεν έχει καμία αξία για τον χρήστη που βλέπει το response,
    γι' αυτό το αγνοούμε και κρατάμε μόνο το πραγματικό κείμενο.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(text_parts)

    # Απρόσμενη μορφή - μετέτρεψέ το σε string ως ασφαλές fallback,
    # αντί να σκάσει το endpoint.
    return str(content)


class ChatRequest(BaseModel):
    """
    Το body που περιμένουμε στο POST /chat.

    session_id: ταυτοποιεί ΠΟΙΑ συνομιλία είναι αυτή, ώστε να φορτώνουμε
        το σωστό ιστορικό από το Redis.

    confirm: συμπληρώνεται ΜΟΝΟ όταν ο χρήστης απαντάει σε αίτημα έγκρισης
        πατώντας τα κουμπιά του UI (true = έγκριση, false = άρνηση). Όταν
        γράφει ελεύθερο κείμενο μένει None, και προσπαθούμε να καταλάβουμε
        την πρόθεσή του από τις λέξεις.
    """

    message: str
    session_id: str = "default"
    confirm: bool | None = None


def _run_agent_and_respond(history: list, session_id: str) -> dict:
    """
    Καλεί τον agent με το δοσμένο ιστορικό, αποθηκεύει το αποτέλεσμα, και
    χτίζει το response.

    Είναι ξεχωριστή function γιατί την καλούμε από ΤΡΙΑ σημεία της ροής
    (νέο μήνυμα / μετά από έγκριση / μετά από άρνηση) και δεν θέλουμε να
    επαναλάβουμε τη λογική τρεις φορές.
    """
    result = get_agent().invoke({"messages": history})
    messages = result["messages"]

    save_history(session_id, messages)

    response = {
        "response": _extract_text(messages[-1].content),
        "session_id": session_id,
    }

    # Σταμάτησε ο agent ζητώντας έγκριση; Αν ναι, στείλε στο UI τι ακριβώς
    # θέλει να κάνει, ώστε να εμφανίσει τα κουμπιά Ναι/Όχι.
    pending = get_pending_tool_calls(messages)
    if pending:
        response["pending_confirmation"] = {
            "actions": [describe_tool_call(tc) for tc in pending]
        }
        # Όταν το Gemini ζητάει εργαλείο, το content του μηνύματος είναι
        # συχνά κενό. Βάζουμε δικό μας κείμενο ώστε ο χρήστης να μη δει
        # κενή φυσαλίδα.
        if not response["response"].strip():
            response["response"] = "Θέλω να κάνω την παρακάτω ενέργεια:"

    return response


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """
    Το βασικό endpoint της συνομιλίας.

    Η ροή έχει τρεις πιθανές διαδρομές:

    Α) ΚΑΝΟΝΙΚΟ ΜΗΝΥΜΑ (δεν εκκρεμεί τίποτα)
       -> πρόσθεσε το μήνυμα στο ιστορικό, τρέξε τον agent.

    Β) ΕΚΚΡΕΜΕΙ ΕΓΚΡΙΣΗ και ο χρήστης απάντησε ναι/όχι
       -> ναι:  εκτέλεσε τις ενέργειες, δώσε τα αποτελέσματα στον agent.
       -> όχι:  κατέγραψε την άρνηση, ενημέρωσε τον agent.

    Γ) ΕΚΚΡΕΜΕΙ ΕΓΚΡΙΣΗ αλλά ο χρήστης έγραψε κάτι άσχετο
       -> ακύρωσε την εκκρεμή ενέργεια (στην αμφιβολία δεν εκτελούμε) και
          χειρίσου το κείμενο ως νέο μήνυμα.
    """
    history = load_history(request.session_id)
    pending = get_pending_tool_calls(history)

    # --- Α) Καμία εκκρεμότητα: κανονική ροή ---
    if not pending:
        log.info(
            "μήνυμα | session=%s | %s",
            request.session_id,
            safe(request.message),
        )
        history.append(HumanMessage(content=request.message))
        return _run_agent_and_respond(history, request.session_id)

    answer = interpret_answer(request.message, request.confirm)
    names = ", ".join(tc["name"] for tc in pending)

    # --- Β1) Ο χρήστης ενέκρινε ---
    if answer is True:
        log.warning("ΕΓΚΡΙΘΗΚΕ | session=%s | %s", request.session_id, names)
        history.extend(execute_tool_calls(pending))
        return _run_agent_and_respond(history, request.session_id)

    # --- Β2) Ο χρήστης αρνήθηκε ---
    if answer is False:
        log.info("απορρίφθηκε | session=%s | %s", request.session_id, names)
        history.extend(
            build_decline_messages(
                pending,
                "Ο χρήστης ΔΕΝ ενέκρινε αυτή την ενέργεια, οπότε δεν "
                "εκτελέστηκε. Ρώτησέ τον τι θα ήθελε να κάνεις αντ' αυτού.",
            )
        )
        return _run_agent_and_respond(history, request.session_id)

    # --- Γ) Ασαφής απάντηση: ακύρωσε και συνέχισε με το νέο μήνυμα ---
    log.info(
        "ακυρώθηκε (αλλαγή θέματος) | session=%s | %s",
        request.session_id,
        names,
    )
    history.extend(
        build_decline_messages(
            pending,
            "Η ενέργεια ακυρώθηκε επειδή ο χρήστης άλλαξε θέμα χωρίς να την "
            "εγκρίνει. Μην την εκτελέσεις. Απάντησε στο νέο του μήνυμα.",
        )
    )
    history.append(HumanMessage(content=request.message))
    return _run_agent_and_respond(history, request.session_id)
