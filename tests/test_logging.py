"""
Tests για την καταγραφή (logging).

Δύο πράγματα έχουν πραγματική σημασία εδώ:

  1. ΙΔΙΩΤΙΚΟΤΗΤΑ: το log δεν πρέπει να περιέχει περιεχόμενο από τα emails
     ή τα μηνύματα του χρήστη, εκτός αν το ζητήσει ρητά. Διαφορετικά
     φτιάχνουμε αρχείο με αντίγραφα της αλληλογραφίας του.

  2. ΟΡΑΤΟΤΗΤΑ ΣΦΑΛΜΑΤΩΝ: ο κώδικας πιάνει εξαιρέσεις και τις μετατρέπει
     σε ευγενικές απαντήσεις. Αν δεν καταγράφονταν, τα προβλήματα θα ήταν
     εντελώς αόρατα - ειδικά στην πρωινή ενημέρωση που τρέχει χωρίς
     επίβλεψη.
"""

import logging

import pytest
from langchain_core.messages import AIMessage

from app.config import get_settings
from app.core.logging_config import get_logger, safe
from tests.conftest import ai_with_tools, tool_call


class TestSafe:
    def test_κρύβει_περιεχόμενο_από_προεπιλογή(self):
        """
        Χωρίς αυτό, το log θα γέμιζε με αποσπάσματα από τα email του
        χρήστη - σε αρχείο χωρίς καμία προστασία.
        """
        result = safe("Αγαπητέ Γιώργο, η συνέντευξή σας είναι...")
        assert "Γιώργο" not in result
        assert "χαρακτήρες" in result

    def test_δείχνει_το_μήκος(self):
        """Το μήκος αρκεί για διάγνωση: ήρθε άδειο; ήταν τεράστιο;"""
        assert "5 χαρακτήρες" in safe("12345")

    def test_με_log_content_δείχνει_περιεχόμενο(self, monkeypatch):
        monkeypatch.setenv("LOG_CONTENT", "true")
        get_settings.cache_clear()
        assert "Γεια σου" in safe("Γεια σου Γιώργο")

    def test_κόβει_μεγάλο_κείμενο(self, monkeypatch):
        monkeypatch.setenv("LOG_CONTENT", "true")
        get_settings.cache_clear()
        result = safe("α" * 500, limit=50)
        assert len(result) < 100
        assert result.endswith("...")

    def test_αντικαθιστά_αλλαγές_γραμμής(self, monkeypatch):
        """Μια πολύγραμμη καταχώρηση σπάει το format του log."""
        monkeypatch.setenv("LOG_CONTENT", "true")
        get_settings.cache_clear()
        assert "\n" not in safe("πρώτη\nδεύτερη")

    def test_κενή_τιμή(self):
        assert safe(None) == "<κενό>"


class TestLoggerNaming:
    def test_ιεραρχία_ονομάτων(self):
        """
        Το κοινό πρόθεμα επιτρέπει να ρυθμίσουμε όλους τους loggers μας
        μαζί, και να τους ξεχωρίζουμε από αυτούς του uvicorn.
        """
        assert get_logger("tools").name == "assistant.tools"
        assert get_logger("briefing").name == "assistant.briefing"


class TestToolLogging:
    def test_επιτυχία_καταγράφεται_με_χρόνο(self, fake_calendar, caplog):
        from app.agents.graph import execute_tool_calls

        with caplog.at_level(logging.INFO, logger="assistant.agent"):
            execute_tool_calls(
                [tool_call("list_upcoming_events", {"max_results": 5})]
            )

        assert "list_upcoming_events ok" in caplog.text

    def test_σφάλμα_καταγράφεται_αν_και_κρύβεται_από_τον_χρήστη(
        self, monkeypatch, caplog
    ):
        """
        Το ΠΙΟ ΣΗΜΑΝΤΙΚΟ test αυτού του αρχείου.

        Ο χρήστης παίρνει ευγενική απάντηση - καλό για την εμπειρία. Αλλά
        αν δεν καταγραφόταν το σφάλμα, ένα εργαλείο θα μπορούσε να
        αποτυγχάνει συστηματικά χωρίς να το μάθεις ποτέ.
        """
        from app.agents.graph import execute_tool_calls

        def boom():
            raise RuntimeError("Gmail API quota exceeded")

        monkeypatch.setattr("app.tools.gmail_tool._get_gmail_service", boom)

        with caplog.at_level(logging.ERROR, logger="assistant.agent"):
            results = execute_tool_calls(
                [tool_call("list_recent_emails", {"max_results": 5})]
            )

        # Ο χρήστης βλέπει ήπιο μήνυμα...
        assert "ΣΦΑΛΜΑ" in results[0].content
        # ...αλλά η αιτία καταγράφηκε
        assert "quota exceeded" in caplog.text
        assert "ERROR" in caplog.text

    def test_ανύπαρκτο_εργαλείο_καταγράφεται(self, caplog):
        from app.agents.graph import execute_tool_calls

        with caplog.at_level(logging.ERROR, logger="assistant.agent"):
            execute_tool_calls([tool_call("δεν_υπάρχει", {})])

        assert "ανύπαρκτο εργαλείο" in caplog.text


class TestConfirmationLogging:
    def test_αναμονή_έγκρισης_καταγράφεται_ως_warning(self, make_llm, caplog):
        """
        Σημείο όπου η ροή σταματάει και περιμένει άνθρωπο - πρέπει να
        ξεχωρίζει οπτικά όταν διαβάζεις τα logs.
        """
        from app.agents.graph import get_agent
        from langchain_core.messages import HumanMessage

        make_llm(
            [
                ai_with_tools(
                    tool_call("send_email", {"to": "a@b.com", "subject": "x", "body": "y"})
                )
            ]
        )

        with caplog.at_level(logging.WARNING, logger="assistant.agent"):
            get_agent().invoke({"messages": [HumanMessage(content="στείλε")]})

        assert "σε αναμονή έγκρισης" in caplog.text
        assert "send_email" in caplog.text

    def test_έγκριση_καταγράφεται(self, client, make_llm, fake_gmail, caplog):
        """
        Η έγκριση είναι το σημείο όπου ο χρήστης ανέλαβε ευθύνη για μια μη
        αναστρέψιμη ενέργεια. Πρέπει να υπάρχει ίχνος.
        """
        make_llm(
            [
                ai_with_tools(
                    tool_call("send_email", {"to": "a@b.com", "subject": "x", "body": "y"})
                ),
                AIMessage(content="Στάλθηκε."),
            ]
        )

        client.post("/chat", json={"message": "στείλε", "session_id": "log1"})

        with caplog.at_level(logging.INFO, logger="assistant.api"):
            client.post(
                "/chat",
                json={"message": "Ναι", "session_id": "log1", "confirm": True},
            )

        assert "ΕΓΚΡΙΘΗΚΕ" in caplog.text

    def test_άρνηση_καταγράφεται(self, client, make_llm, fake_gmail, caplog):
        make_llm(
            [
                ai_with_tools(
                    tool_call("send_email", {"to": "a@b.com", "subject": "x", "body": "y"})
                ),
                AIMessage(content="Εντάξει."),
            ]
        )

        client.post("/chat", json={"message": "στείλε", "session_id": "log2"})

        with caplog.at_level(logging.INFO, logger="assistant.api"):
            client.post(
                "/chat",
                json={"message": "Όχι", "session_id": "log2", "confirm": False},
            )

        assert "απορρίφθηκε" in caplog.text

    def test_το_μήνυμα_του_χρήστη_δεν_καταγράφεται_αυτούσιο(
        self, client, make_llm, caplog
    ):
        make_llm([AIMessage(content="ok")])

        with caplog.at_level(logging.INFO, logger="assistant.api"):
            client.post(
                "/chat",
                json={"message": "ο κωδικός μου είναι μυστικός123", "session_id": "p"},
            )

        assert "μυστικός123" not in caplog.text
        assert "χαρακτήρες" in caplog.text


class TestBriefingLogging:
    """
    Η πρωινή ενημέρωση τρέχει στις 08:30 χωρίς επίβλεψη. Χωρίς logging, μια
    αποτυχία εκεί είναι εντελώς αόρατη.
    """

    @pytest.fixture
    def briefing_data(self, monkeypatch):
        monkeypatch.setattr(
            "app.tools.calendar_tool.get_events_for_day", lambda day: ["10:00 - X"]
        )
        monkeypatch.setattr(
            "app.tools.gmail_tool.get_unread_emails",
            lambda max_results=10, newer_than_days=2: ["email 1", "email 2"],
        )
        monkeypatch.setattr(
            "app.tools.tasks_tool.get_pending_tasks", lambda max_results=20: ["task 1"]
        )

    def test_καταγράφει_πλήθη_δεδομένων(
        self, fake_redis, make_llm, briefing_data, caplog
    ):
        """Μεταδεδομένα, όχι περιεχόμενο: πόσα, όχι τι."""
        from app.core.briefing import generate_briefing

        make_llm([AIMessage(content="σύνοψη")])

        with caplog.at_level(logging.INFO, logger="assistant.briefing"):
            generate_briefing()

        assert "1 ραντεβού" in caplog.text
        assert "1 εργασίες" in caplog.text
        assert "2 αδιάβαστα" in caplog.text

    def test_αποτυχία_αποστολής_καταγράφεται(
        self, fake_redis, make_llm, briefing_data, monkeypatch, caplog
    ):
        """
        Χωρίς αυτό το log, μια αποτυχία στις 08:30 θα ήταν εντελώς αόρατη:
        απλά μια μέρα δεν θα ερχόταν η ενημέρωση και δεν θα ήξερες γιατί.
        """
        from app.core.briefing import run_scheduled_briefing

        def boom(subject, body):
            raise RuntimeError("SMTP timeout")

        monkeypatch.setattr("app.tools.gmail_tool.send_email_to_self", boom)
        make_llm([AIMessage(content="η ενημέρωσή σου")])

        with caplog.at_level(logging.ERROR, logger="assistant.briefing"):
            run_scheduled_briefing(send_email=True)

        assert "αποτυχία αποστολής" in caplog.text
        assert "SMTP timeout" in caplog.text

    def test_επιτυχής_αποστολή_καταγράφεται(
        self, fake_redis, make_llm, briefing_data, monkeypatch, caplog
    ):
        from app.core.briefing import run_scheduled_briefing

        monkeypatch.setattr(
            "app.tools.gmail_tool.send_email_to_self", lambda s, b: "id"
        )
        make_llm([AIMessage(content="ενημέρωση")])

        with caplog.at_level(logging.INFO, logger="assistant.briefing"):
            run_scheduled_briefing(send_email=True)

        assert "στάλθηκε με email" in caplog.text
