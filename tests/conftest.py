"""
conftest.py
------------
Κοινές "fixtures" (προπαρασκευασμένα εξαρτήματα) που μοιράζονται όλα τα
tests. Το pytest τις βρίσκει αυτόματα - δεν χρειάζεται import.

Η φιλοσοφία των tests αυτού του project:
Δεν χτυπάμε ΠΟΤΕ πραγματικά APIs (Google, Gemini) ούτε πραγματικό Redis.
Λόγοι:
  - Ταχύτητα: όλο το suite τρέχει σε δευτερόλεπτα, όχι λεπτά.
  - Αξιοπιστία: τα tests δεν αποτυγχάνουν επειδή έπεσε το δίκτυο ή
    εξαντλήθηκε το δωρεάν quota του Gemini.
  - Ασφάλεια: δεν υπάρχει περίπτωση ένα test να στείλει πραγματικό email
    ή να γράψει στο πραγματικό σου ημερολόγιο.

Αντ' αυτού, αντικαθιστούμε αυτές τις εξαρτήσεις με "διπλά" (fakes/mocks)
που συμπεριφέρονται όπως τα πραγματικά, αλλά τα ελέγχουμε εμείς.
"""

import base64

import pytest
from langchain_core.messages import AIMessage
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Περιβάλλον
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    """
    Ορίζει ψεύτικες μεταβλητές περιβάλλοντος πριν από ΚΑΘΕ test.

    Το autouse=True σημαίνει ότι εφαρμόζεται αυτόματα παντού - χωρίς αυτό,
    τα tests θα προσπαθούσαν να διαβάσουν το πραγματικό σου .env (και θα
    απέτυχαν σε μηχάνημα που δεν το έχει, π.χ. σε CI).
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")


@pytest.fixture(autouse=True)
def clear_caches():
    """
    Καθαρίζει όλα τα lru_cache πριν και μετά από κάθε test.

    ΓΙΑΤΙ ΕΙΝΑΙ ΚΡΙΣΙΜΟ: χρησιμοποιούμε @lru_cache σε πολλά σημεία
    (get_settings, get_llm, get_agent, _get_llm_with_tools). Χωρίς αυτό
    το καθάρισμα, το πρώτο test θα "κλείδωνε" ένα mock μέσα στην cache και
    όλα τα επόμενα tests θα έπαιρναν το ΙΔΙΟ mock - με αποτέλεσμα άλλα
    tests να περνάνε ψευδώς και άλλα να αποτυγχάνουν ανεξήγητα.
    """
    def _clear():
        from app.config import get_settings
        from app.core.llm_client import get_llm
        from app.agents.graph import get_agent, _get_llm_with_tools

        get_settings.cache_clear()
        get_llm.cache_clear()
        get_agent.cache_clear()
        _get_llm_with_tools.cache_clear()

    _clear()
    yield
    _clear()


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------

class FakeRedis:
    """
    Ελάχιστη υλοποίηση Redis πάνω σε ένα απλό dict.

    Υλοποιεί μόνο ό,τι χρησιμοποιεί ο κώδικάς μας (get/set/delete/ping).
    Το TTL (παράμετρος ex) το δεχόμαστε αλλά το αγνοούμε - στα tests δεν
    μας ενδιαφέρει η λήξη.
    """

    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value

    def delete(self, key):
        self.store.pop(key, None)

    def ping(self):
        return True


@pytest.fixture
def fake_redis(monkeypatch):
    """Αντικαθιστά το πραγματικό Redis με το FakeRedis παντού."""
    fake = FakeRedis()
    monkeypatch.setattr(
        "app.core.redis_client.get_redis_client", lambda: fake
    )
    return fake


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

@pytest.fixture
def make_llm(monkeypatch):
    """
    Εργοστάσιο για ψεύτικα LLM.

    Χρήση μέσα σε test:
        make_llm([AIMessage(content="γεια")])            # μία απάντηση
        make_llm([msg_with_tool_call, final_msg])        # διαδοχικές

    Επιστρέφει το mock, ώστε το test να μπορεί να ελέγξει π.χ. πόσες φορές
    κλήθηκε ή με τι ορίσματα.
    """

    def _make(responses):
        iterator = iter(responses)

        # Το "με εργαλεία" μοντέλο (αυτό που επιστρέφει το .bind_tools()).
        bound = MagicMock()
        bound.invoke.side_effect = lambda messages: next(iterator)

        # Το "γυμνό" μοντέλο που επιστρέφει το get_llm().
        raw = MagicMock()
        raw.bind_tools.return_value = bound
        # Αν κάποιος το καλέσει ΧΩΡΙΣ bind_tools (π.χ. το briefing), να
        # δουλεύει κι έτσι, μοιραζόμενο τον ίδιο iterator.
        raw.invoke.side_effect = lambda messages: next(iterator)

        monkeypatch.setattr("app.core.llm_client.get_llm", lambda *a, **k: raw)
        return raw

    return _make


# ---------------------------------------------------------------------------
# Google APIs
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_calendar(monkeypatch):
    """Αντικαθιστά το Google Calendar service με mock."""
    service = MagicMock()
    # Λογικά defaults, ώστε ένα test που δεν νοιάζεται για το calendar να
    # μη χρειάζεται να τα ορίσει.
    service.events.return_value.list.return_value.execute.return_value = {
        "items": []
    }
    service.events.return_value.insert.return_value.execute.return_value = {
        "htmlLink": "https://calendar.google.com/event?eid=test"
    }
    monkeypatch.setattr(
        "app.tools.calendar_tool._get_calendar_service", lambda: service
    )
    return service


@pytest.fixture
def fake_gmail(monkeypatch):
    """Αντικαθιστά το Gmail service με mock."""
    service = MagicMock()
    messages = service.users.return_value.messages.return_value
    messages.list.return_value.execute.return_value = {"messages": []}
    messages.send.return_value.execute.return_value = {"id": "sent-test"}
    service.users.return_value.drafts.return_value.create.return_value.execute.return_value = {
        "id": "draft-test"
    }
    service.users.return_value.getProfile.return_value.execute.return_value = {
        "emailAddress": "test@example.com"
    }
    monkeypatch.setattr(
        "app.tools.gmail_tool._get_gmail_service", lambda: service
    )
    return service


@pytest.fixture
def no_scheduler(monkeypatch):
    """
    Απενεργοποιεί τον scheduler στα tests του API.

    Χωρίς αυτό, κάθε TestClient θα ξεκινούσε πραγματικό background thread -
    περιττό, και θα άφηνε threads να τρέχουν μετά το τέλος των tests.
    """
    monkeypatch.setattr("app.core.scheduler.start_scheduler", lambda: None)
    monkeypatch.setattr("app.core.scheduler.stop_scheduler", lambda: None)


# ---------------------------------------------------------------------------
# Βοηθητικά
# ---------------------------------------------------------------------------

def b64(text: str) -> str:
    """Κωδικοποιεί κείμενο όπως το επιστρέφει το Gmail API."""
    return base64.urlsafe_b64encode(text.encode()).decode()


def decode_raw(raw: str) -> str:
    """Αποκωδικοποιεί το raw MIME που στέλνουμε στο Gmail API."""
    return base64.urlsafe_b64decode(raw).decode("utf-8", errors="replace")


def tool_call(name: str, args: dict, call_id: str = "call-1") -> dict:
    """Φτιάχνει ένα tool_call dict, όπως το παράγει το Gemini."""
    return {"name": name, "args": args, "id": call_id}


def ai_with_tools(*calls) -> AIMessage:
    """AIMessage που ζητάει ένα ή περισσότερα εργαλεία."""
    return AIMessage(content="", tool_calls=list(calls))
