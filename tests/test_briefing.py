"""
Tests για την πρωινή ενημέρωση και τον προγραμματισμό της.

Η ΣΗΜΑΝΤΙΚΟΤΕΡΗ ιδιότητα που ελέγχουμε εδώ: ότι το μοντέλο ΔΕΝ έχει
πρόσβαση σε εργαλεία κατά τη δημιουργία της ενημέρωσης.

Γιατί: στις 8:30 το πρωί δεν είσαι μπροστά στην οθόνη. Αν ο agent είχε
εργαλεία και κάποιο email περιείχε prompt injection, δεν θα υπήρχε κανείς
να πατήσει "Όχι". Χωρίς εργαλεία, το χειρότερο σενάριο είναι μια περίεργη
σύνοψη - όχι μια ενέργεια.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import AIMessage

import app.core.scheduler as scheduler_module
from app.core.briefing import (
    generate_briefing,
    get_or_create_todays_briefing,
    load_briefing,
    run_scheduled_briefing,
    save_briefing,
    _build_prompt,
)

ATHENS = ZoneInfo("Europe/Athens")


@pytest.fixture
def briefing_data(monkeypatch):
    """Ρυθμίζει τα δεδομένα που θα "διαβάσει" η ενημέρωση."""
    state = {"events": [], "emails": []}

    monkeypatch.setattr(
        "app.tools.calendar_tool.get_events_for_day", lambda day: state["events"]
    )
    monkeypatch.setattr(
        "app.tools.gmail_tool.get_unread_emails",
        lambda max_results=10, newer_than_days=2: state["emails"],
    )
    return state


class TestBuildPrompt:
    def test_περιλαμβάνει_τα_δεδομένα(self):
        prompt = _build_prompt(
            datetime(2026, 9, 1, tzinfo=ATHENS),
            ["10:00 - Meeting"],
            ["Από: hr@accenture.com | Θέμα: Interview"],
        )
        assert "10:00 - Meeting" in prompt
        assert "hr@accenture.com" in prompt

    def test_σημαίνει_τα_email_ως_δεδομένα(self):
        """Άμυνα σε prompt injection: το περιεχόμενο δεν είναι εντολές."""
        prompt = _build_prompt(datetime(2026, 9, 1, tzinfo=ATHENS), [], [])
        assert "ΠΟΤΕ οδηγίες" in prompt
        assert "ύποπτο" in prompt

    def test_χειρίζεται_κενές_λίστες(self):
        prompt = _build_prompt(datetime(2026, 9, 1, tzinfo=ATHENS), [], [])
        assert "(κανένα)" in prompt

    def test_περιλαμβάνει_την_ημερομηνία(self):
        prompt = _build_prompt(datetime(2026, 9, 1, tzinfo=ATHENS), [], [])
        assert "2026" in prompt


class TestGenerateBriefing:
    def test_το_μοντέλο_ΔΕΝ_έχει_εργαλεία(self, fake_redis, make_llm, briefing_data):
        """
        Το πιο σημαντικό test αυτού του αρχείου: επιβεβαιώνει ότι δεν
        καλείται ποτέ το .bind_tools() στη ροή της ενημέρωσης.
        """
        raw = make_llm([AIMessage(content="Ήρεμη μέρα.")])
        generate_briefing()

        raw.bind_tools.assert_not_called()

    def test_περνάει_τα_δεδομένα_στο_prompt(self, fake_redis, make_llm, briefing_data):
        briefing_data["events"] = ["18:00 - Ιδιαίτερα AI"]
        briefing_data["emails"] = ["Από: hr@accenture.com | Θέμα: Interview"]

        raw = make_llm([AIMessage(content="σύνοψη")])
        generate_briefing()

        prompt = raw.invoke.call_args[0][0][0].content
        assert "18:00 - Ιδιαίτερα AI" in prompt
        assert "hr@accenture.com" in prompt

    def test_αποθηκεύει_το_αποτέλεσμα(self, fake_redis, make_llm, briefing_data):
        make_llm([AIMessage(content="Η μέρα σου.")])
        generate_briefing()

        today = datetime.now(ATHENS).strftime("%Y-%m-%d")
        assert load_briefing(today) == "Η μέρα σου."

    def test_σφάλμα_ημερολογίου_δεν_ρίχνει_την_ενημέρωση(
        self, fake_redis, make_llm, monkeypatch
    ):
        """
        Καλείται από scheduler - μια ανεξέλεγκτη εξαίρεση θα "χανόταν"
        σιωπηλά. Καλύτερα να παραδώσουμε μερική ενημέρωση.
        """
        def boom(day):
            raise RuntimeError("Calendar API down")

        monkeypatch.setattr("app.tools.calendar_tool.get_events_for_day", boom)
        monkeypatch.setattr(
            "app.tools.gmail_tool.get_unread_emails",
            lambda max_results=10, newer_than_days=2: [],
        )

        raw = make_llm([AIMessage(content="Δεν είδα το ημερολόγιο.")])
        result = generate_briefing()

        assert result == "Δεν είδα το ημερολόγιο."
        assert "Calendar API down" in raw.invoke.call_args[0][0][0].content

    def test_σφάλμα_llm_επιστρέφει_μήνυμα_αντί_να_σκάσει(
        self, fake_redis, briefing_data, monkeypatch
    ):
        from unittest.mock import MagicMock

        raw = MagicMock()
        raw.invoke.side_effect = RuntimeError("Quota exceeded")
        monkeypatch.setattr("app.core.llm_client.get_llm", lambda *a, **k: raw)

        result = generate_briefing()
        assert "Δεν ήταν δυνατή" in result
        assert "Quota exceeded" in result

    def test_καθαρίζει_τα_content_blocks(self, fake_redis, make_llm, briefing_data):
        make_llm(
            [
                AIMessage(
                    content=[
                        {"type": "text", "text": "Καθαρό κείμενο",
                         "extras": {"signature": "τεχνικό"}}
                    ]
                )
            ]
        )
        assert generate_briefing() == "Καθαρό κείμενο"


class TestCaching:
    def test_δεύτερη_κλήση_δεν_ξανακαλεί_το_μοντέλο(
        self, fake_redis, make_llm, briefing_data
    ):
        raw = make_llm([AIMessage(content="Η μέρα σου.")])

        first = generate_briefing()
        second = get_or_create_todays_briefing()

        assert first == second
        assert raw.invoke.call_count == 1

    def test_φτιάχνει_αν_δεν_υπάρχει(self, fake_redis, make_llm, briefing_data):
        """
        Σενάριο: το Mac κοιμόταν στις 8:30, οπότε ο scheduler δεν έτρεξε.
        Ανοίγοντας το UI, η ενημέρωση φτιάχνεται εκείνη τη στιγμή.
        """
        make_llm([AIMessage(content="Φτιάχτηκε τώρα.")])
        assert get_or_create_todays_briefing() == "Φτιάχτηκε τώρα."

    def test_αποθήκευση_με_ttl(self, fake_redis, monkeypatch):
        captured = {}

        def spy_set(key, value, ex=None):
            captured["ex"] = ex
            fake_redis.store[key] = value

        monkeypatch.setattr(fake_redis, "set", spy_set)
        save_briefing("2026-09-01", "κείμενο")

        assert captured["ex"] == 60 * 60 * 24 * 7

    def test_φόρτωση_ανύπαρκτης_ημέρας(self, fake_redis):
        assert load_briefing("1999-01-01") is None


class TestScheduledRun:
    def test_στέλνει_email(self, fake_redis, make_llm, briefing_data, monkeypatch):
        sent = {}

        def fake_send(subject, body):
            sent["subject"] = subject
            sent["body"] = body
            return "msg-id"

        monkeypatch.setattr("app.tools.gmail_tool.send_email_to_self", fake_send)
        make_llm([AIMessage(content="Η ενημέρωσή σου.")])

        run_scheduled_briefing(send_email=True)

        assert sent["body"] == "Η ενημέρωσή σου."
        assert "Η ημέρα σου" in sent["subject"]

    def test_χωρίς_email_αν_απενεργοποιημένο(
        self, fake_redis, make_llm, briefing_data, monkeypatch
    ):
        called = {"yes": False}

        def fake_send(subject, body):
            called["yes"] = True

        monkeypatch.setattr("app.tools.gmail_tool.send_email_to_self", fake_send)
        make_llm([AIMessage(content="κείμενο")])

        run_scheduled_briefing(send_email=False)
        assert called["yes"] is False

    def test_αποτυχία_email_δεν_ρίχνει_τον_scheduler(
        self, fake_redis, make_llm, briefing_data, monkeypatch
    ):
        """Η ενημέρωση έχει ήδη αποθηκευτεί - θα φανεί στο UI."""
        def boom(subject, body):
            raise RuntimeError("SMTP down")

        monkeypatch.setattr("app.tools.gmail_tool.send_email_to_self", boom)
        make_llm([AIMessage(content="Η ενημέρωσή σου.")])

        run_scheduled_briefing(send_email=True)  # δεν πρέπει να σκάσει

        today = datetime.now(ATHENS).strftime("%Y-%m-%d")
        assert load_briefing(today) == "Η ενημέρωσή σου."


class TestScheduler:
    @pytest.fixture(autouse=True)
    def clean_scheduler(self):
        scheduler_module._scheduler = None
        yield
        scheduler_module.stop_scheduler()

    def test_προγραμματίζει_στη_σωστή_ώρα(self):
        scheduler_module.start_scheduler()

        jobs = scheduler_module._scheduler.get_jobs()
        assert len(jobs) == 1
        assert jobs[0].next_run_time.hour == 8
        assert jobs[0].next_run_time.minute == 30

    def test_χρησιμοποιεί_ώρα_ελλάδας(self):
        """
        Με ZoneInfo αντί για σταθερό offset, η αλλαγή θερινής/χειμερινής
        ώρας γίνεται αυτόματα.
        """
        scheduler_module.start_scheduler()
        job = scheduler_module._scheduler.get_jobs()[0]
        assert str(job.trigger.timezone) == "Europe/Athens"

    def test_σέβεται_τη_ρύθμιση_ώρας(self, monkeypatch):
        from app.config import get_settings

        monkeypatch.setenv("BRIEFING_HOUR", "7")
        monkeypatch.setenv("BRIEFING_MINUTE", "15")
        get_settings.cache_clear()

        scheduler_module.start_scheduler()
        job = scheduler_module._scheduler.get_jobs()[0]
        assert job.next_run_time.hour == 7
        assert job.next_run_time.minute == 15

    def test_απενεργοποιημένος_δεν_ξεκινά(self, monkeypatch):
        from app.config import get_settings

        monkeypatch.setenv("BRIEFING_ENABLED", "false")
        get_settings.cache_clear()

        scheduler_module.start_scheduler()
        assert scheduler_module._scheduler is None

    def test_ανοχή_σε_καθυστερημένη_εκτέλεση(self):
        """
        Αν ο server άνοιξε στις 9:00, η εκτέλεση των 8:30 έχει ακόμα νόημα.
        Στις 14:00 όχι - γι' αυτό το όριο είναι μία ώρα.
        """
        scheduler_module.start_scheduler()
        assert scheduler_module._scheduler.get_jobs()[0].misfire_grace_time == 3600

    def test_τερματίζει_καθαρά(self):
        scheduler_module.start_scheduler()
        scheduler_module.stop_scheduler()
        assert scheduler_module._scheduler is None

    def test_διπλή_εκκίνηση_δεν_διπλασιάζει_εργασίες(self):
        scheduler_module.start_scheduler()
        scheduler_module.start_scheduler()
        assert len(scheduler_module._scheduler.get_jobs()) == 1
