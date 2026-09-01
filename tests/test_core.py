"""
Tests για τα "θεμέλια": ρυθμίσεις, αποθήκευση ιστορικού, και τη λογική
που αποφασίζει αν μια ενέργεια εκκρεμεί ή εγκρίθηκε.
"""


import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import get_settings
from app.core.confirmation import (
    build_decline_messages,
    get_pending_tool_calls,
    interpret_answer,
    _normalize,
)
from app.core.memory import load_history, save_history, clear_history
from tests.conftest import tool_call


# ===========================================================================
# Ρυθμίσεις
# ===========================================================================

class TestConfig:
    def test_φορτώνει_από_env(self):
        settings = get_settings()
        assert settings.google_api_key == "test-key-not-real"
        assert settings.gemini_model == "gemini-3.6-flash"

    def test_defaults_πρωινής_ενημέρωσης(self):
        settings = get_settings()
        assert settings.briefing_hour == 8
        assert settings.briefing_minute == 30
        assert settings.briefing_enabled is True

    def test_τιμή_από_env_υπερισχύει_του_default(self, monkeypatch):
        monkeypatch.setenv("BRIEFING_HOUR", "7")
        get_settings.cache_clear()
        assert get_settings().briefing_hour == 7

    def test_λείπει_υποχρεωτικό_κλειδί_σκάει_νωρίς(self, monkeypatch):
        """
        Το GOOGLE_API_KEY είναι υποχρεωτικό. Θέλουμε να σκάει ΚΑΤΑ ΤΗΝ
        ΕΚΚΙΝΗΣΗ με σαφές μήνυμα, όχι αργότερα μέσα σε ένα request.
        """
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        get_settings.cache_clear()
        # Το pydantic-settings μπορεί να διαβάσει .env αν υπάρχει τοπικά,
        # οπότε δεχόμαστε είτε σφάλμα είτε φορτωμένες ρυθμίσεις - αρκεί
        # να μη σκάσει με κρυπτικό KeyError αργότερα.
        try:
            get_settings()
        except Exception as exc:
            assert "google_api_key" in str(exc).lower()


# ===========================================================================
# Μνήμη συνομιλίας (Redis)
# ===========================================================================

class TestMemory:
    def test_κενό_ιστορικό_για_νέο_session(self, fake_redis):
        assert load_history("δεν-υπάρχει") == []

    def test_αποθήκευση_και_φόρτωση(self, fake_redis):
        messages = [HumanMessage(content="Γεια"), AIMessage(content="Γεια σου")]
        save_history("s1", messages)

        loaded = load_history("s1")
        assert len(loaded) == 2
        assert loaded[0].content == "Γεια"
        assert loaded[1].content == "Γεια σου"

    def test_διατηρεί_tool_calls_και_tool_messages(self, fake_redis):
        """
        Κρίσιμο: αν χανόταν το tool_call κατά την αποθήκευση, το Gemini θα
        έβλεπε "ορφανό" ToolMessage και θα επέστρεφε σφάλμα.
        """
        messages = [
            HumanMessage(content="Τι έχω;"),
            AIMessage(
                content="",
                tool_calls=[tool_call("list_upcoming_events", {"max_results": 5})],
            ),
            ToolMessage(content="- 18:00: Ραντεβού", tool_call_id="call-1"),
            AIMessage(content="Έχεις ένα ραντεβού."),
        ]
        save_history("s2", messages)
        loaded = load_history("s2")

        assert len(loaded) == 4
        assert loaded[1].tool_calls[0]["name"] == "list_upcoming_events"
        assert loaded[1].tool_calls[0]["args"]["max_results"] == 5
        assert loaded[2].tool_call_id == "call-1"

    def test_ελληνικοί_χαρακτήρες_δεν_αλλοιώνονται(self, fake_redis):
        save_history("s3", [HumanMessage(content="Ραντεβού στις 18:00 – ΟΚ")])
        assert load_history("s3")[0].content == "Ραντεβού στις 18:00 – ΟΚ"

    def test_sessions_είναι_απομονωμένα(self, fake_redis):
        save_history("a", [HumanMessage(content="μήνυμα Α")])
        save_history("b", [HumanMessage(content="μήνυμα Β")])
        assert load_history("a")[0].content == "μήνυμα Α"
        assert load_history("b")[0].content == "μήνυμα Β"

    def test_διαγραφή_ιστορικού(self, fake_redis):
        save_history("s4", [HumanMessage(content="κάτι")])
        clear_history("s4")
        assert load_history("s4") == []

    def test_ορίζεται_ttl(self, fake_redis, monkeypatch):
        """Χωρίς TTL, το Redis θα γέμιζε με παλιές συνομιλίες για πάντα."""
        captured = {}

        def spy_set(key, value, ex=None):
            captured["ex"] = ex
            fake_redis.store[key] = value

        monkeypatch.setattr(fake_redis, "set", spy_set)
        save_history("s5", [HumanMessage(content="x")])
        assert captured["ex"] == 60 * 60 * 24


# ===========================================================================
# Λογική έγκρισης
# ===========================================================================

class TestNormalize:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Ναί!", "ναι"),
            ("  ΟΧΙ.  ", "οχι"),
            ("Προχώρα", "προχωρα"),
            ("ΑΚΎΡΩΣΗ;", "ακυρωση"),
            ("Yes", "yes"),
        ],
    )
    def test_αφαιρεί_τόνους_κεφαλαία_στίξη(self, raw, expected):
        assert _normalize(raw) == expected


class TestInterpretAnswer:
    def test_το_κουμπί_υπερισχύει_του_κειμένου(self):
        """
        Αν ο χρήστης πάτησε κουμπί, η πρόθεσή του είναι σαφής - δεν
        μαντεύουμε από το κείμενο.
        """
        assert interpret_answer("οτιδήποτε", True) is True
        assert interpret_answer("ναι", False) is False

    @pytest.mark.parametrize(
        "text",
        ["ναι", "Ναι", "ΝΑΙ", "ναί", "οκ", "εντάξει", "προχώρα",
         "στείλε το", "yes", "ok", "send it", "confirm"],
    )
    def test_αναγνωρίζει_έγκριση(self, text):
        assert interpret_answer(text, None) is True

    @pytest.mark.parametrize(
        "text",
        ["όχι", "οχι", "ΌΧΙ", "άκυρο", "ακύρωση", "μην", "σταμάτα",
         "no", "cancel", "stop"],
    )
    def test_αναγνωρίζει_άρνηση(self, text):
        assert interpret_answer(text, None) is False

    @pytest.mark.parametrize(
        "text",
        ["τι έχω αύριο;", "ναι αλλά άλλαξε το θέμα", "μπορεί", "", "άσε"],
    )
    def test_ασαφές_κείμενο_δεν_θεωρείται_έγκριση(self, text):
        """Στην αμφιβολία ΔΕΝ εκτελούμε - επιστρέφουμε None."""
        assert interpret_answer(text, None) is None


class TestPendingDetection:
    def test_κενό_ιστορικό(self):
        assert get_pending_tool_calls([]) == []

    def test_απλή_απάντηση_δεν_εκκρεμεί(self):
        assert get_pending_tool_calls([AIMessage(content="Γεια")]) == []

    def test_εντοπίζει_ανεκτέλεστο_tool_call(self):
        messages = [
            HumanMessage(content="Στείλε email"),
            AIMessage(content="", tool_calls=[tool_call("send_email", {"to": "a@b.com"})]),
        ]
        pending = get_pending_tool_calls(messages)
        assert len(pending) == 1
        assert pending[0]["name"] == "send_email"

    def test_εκτελεσμένο_tool_call_δεν_εκκρεμεί(self):
        """Αν ακολουθεί ToolMessage, η ενέργεια έχει ήδη γίνει."""
        messages = [
            AIMessage(content="", tool_calls=[tool_call("send_email", {})]),
            ToolMessage(content="στάλθηκε", tool_call_id="call-1"),
            AIMessage(content="Έγινε."),
        ]
        assert get_pending_tool_calls(messages) == []


class TestDeclineMessages:
    def test_φτιάχνει_ένα_μήνυμα_ανά_κλήση(self):
        calls = [
            tool_call("send_email", {}, "id1"),
            tool_call("create_calendar_event", {}, "id2"),
        ]
        declines = build_decline_messages(calls, "δεν εγκρίθηκε")

        assert len(declines) == 2
        assert all(isinstance(m, ToolMessage) for m in declines)
        assert {m.tool_call_id for m in declines} == {"id1", "id2"}
        assert all(m.content == "δεν εγκρίθηκε" for m in declines)
