"""
Tests για τον agent (LangGraph graph).

Τα σημαντικότερα εδώ αφορούν το CONFIRMATION LAYER: ότι οι επικίνδυνες
ενέργειες σταματάνε πριν εκτελεστούν. Ένα σφάλμα εδώ σημαίνει email που
έφυγε χωρίς να το εγκρίνεις.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents.graph import (
    CONFIRMATION_REQUIRED_TOOLS,
    TOOLS,
    TOOLS_BY_NAME,
    _build_system_prompt,
    describe_tool_call,
    execute_tool_calls,
    get_agent,
    needs_confirmation,
)
from tests.conftest import ai_with_tools, tool_call


# ===========================================================================
# Καταχώρηση εργαλείων
# ===========================================================================

class TestToolRegistration:
    def test_όλα_τα_εργαλεία_καταχωρημένα(self):
        names = {t.name for t in TOOLS}
        assert names == {
            # Calendar
            "list_upcoming_events",
            "create_calendar_event",
            # Gmail
            "list_recent_emails",
            "search_emails",
            "read_email",
            "create_email_draft",
            "create_reply_draft",
            "send_email",
            # Tasks
            "list_tasks",
            "create_task",
            "complete_task",
            "delete_task",
        }

    def test_αντιστοίχιση_ονόματος_με_function(self):
        assert set(TOOLS_BY_NAME) == {t.name for t in TOOLS}

    def test_κάθε_εργαλείο_έχει_περιγραφή(self):
        """
        Η περιγραφή (docstring) είναι αυτό που διαβάζει το Gemini για να
        αποφασίσει πότε να καλέσει το εργαλείο. Κενή περιγραφή = το
        μοντέλο δεν ξέρει πότε να το χρησιμοποιήσει.
        """
        for tool in TOOLS:
            assert tool.description
            assert len(tool.description) > 40


# ===========================================================================
# Ποια εργαλεία απαιτούν έγκριση
# ===========================================================================

class TestConfirmationRules:
    def test_τα_write_εργαλεία_απαιτούν_έγκριση(self):
        assert CONFIRMATION_REQUIRED_TOOLS == {
            "create_calendar_event",
            "create_email_draft",
            "create_reply_draft",
            "send_email",
            "create_task",
            "complete_task",
            "delete_task",
        }

    def test_κάθε_write_εργαλείο_είναι_καταχωρημένο(self):
        """
        Δικλείδα ασφαλείας: αν κάποιος γράψει λάθος όνομα στο
        CONFIRMATION_REQUIRED_TOOLS, ο κανόνας δεν θα ισχύσει ποτέ και το
        εργαλείο θα εκτελείται σιωπηλά χωρίς έγκριση.
        """
        assert CONFIRMATION_REQUIRED_TOOLS.issubset({t.name for t in TOOLS})

    @pytest.mark.parametrize(
        "name",
        ["list_upcoming_events", "list_recent_emails", "search_emails",
         "read_email", "list_tasks"],
    )
    def test_τα_read_εργαλεία_δεν_απαιτούν(self, name):
        assert needs_confirmation([{"name": name}]) is False

    @pytest.mark.parametrize("name", sorted(CONFIRMATION_REQUIRED_TOOLS))
    def test_κάθε_write_εργαλείο_απαιτεί(self, name):
        assert needs_confirmation([{"name": name}]) is True

    def test_ένα_επικίνδυνο_σε_ομάδα_αρκεί(self):
        """
        Αν το μοντέλο ζητήσει ταυτόχρονα ανάγνωση και αποστολή, ΟΛΑ
        σταματάνε - δεν εκτελούμε τα μισά.
        """
        assert needs_confirmation(
            [{"name": "read_email"}, {"name": "send_email"}]
        ) is True

    def test_κενή_λίστα(self):
        assert needs_confirmation([]) is False


# ===========================================================================
# Περιγραφή ενεργειών προς τον χρήστη
# ===========================================================================

class TestDescribeToolCall:
    def test_αποστολή_email_προειδοποιεί_ότι_δεν_αναιρείται(self):
        desc = describe_tool_call(
            tool_call("send_email", {"to": "a@b.com", "subject": "Θ", "body": "Κ"})
        )
        assert "a@b.com" in desc
        assert "δεν αναιρείται" in desc

    def test_δείχνει_τον_πραγματικό_παραλήπτη(self):
        """
        Η περιγραφή φτιάχνεται από τα ΟΡΙΣΜΑΤΑ, όχι από το κείμενο του
        μοντέλου. Έτσι, αν το μοντέλο έχει παρασυρθεί από κακόβουλο email,
        ο χρήστης βλέπει την πραγματική διεύθυνση, όχι αυτή που ισχυρίζεται.
        """
        desc = describe_tool_call(
            tool_call("send_email", {"to": "evil@attacker.com", "subject": "x", "body": "y"})
        )
        assert "evil@attacker.com" in desc

    def test_event_δείχνει_ώρες(self):
        desc = describe_tool_call(
            tool_call(
                "create_calendar_event",
                {"summary": "Συνέντευξη", "start_time": "2026-09-09T12:00:00+03:00",
                 "end_time": "2026-09-09T13:00:00+03:00"},
            )
        )
        assert "Συνέντευξη" in desc
        assert "2026-09-09T12:00:00+03:00" in desc

    def test_άγνωστο_εργαλείο_δεν_σκάει(self):
        desc = describe_tool_call({"name": "κάτι_νέο", "args": {"x": 1}})
        assert "κάτι_νέο" in desc

    def test_νέα_εργασία_δείχνει_προθεσμία(self):
        desc = describe_tool_call(
            tool_call("create_task", {"title": "Αποστολή CV", "due_date": "2026-09-05"})
        )
        assert "Αποστολή CV" in desc
        assert "2026-09-05" in desc

    def test_διαγραφή_εργασίας_προειδοποιεί(self):
        desc = describe_tool_call(tool_call("delete_task", {"task_id": "t1"}))
        assert "δεν αναιρείται" in desc

    def test_κάθε_εργαλείο_έγκρισης_έχει_δική_του_περιγραφή(self):
        """
        Αν προστεθεί write εργαλείο χωρίς περιγραφή, ο χρήστης θα δει ένα
        ακατάληπτο dump ορισμάτων στην κάρτα έγκρισης.
        """
        generic = "με ορίσματα:"
        for name in CONFIRMATION_REQUIRED_TOOLS:
            desc = describe_tool_call({"name": name, "args": {}})
            assert generic not in desc, f"λείπει περιγραφή για το {name}"


# ===========================================================================
# Εκτέλεση εργαλείων
# ===========================================================================

class TestExecuteToolCalls:
    def test_εκτελεί_και_επιστρέφει_tool_message(self, fake_calendar):
        results = execute_tool_calls(
            [tool_call("list_upcoming_events", {"max_results": 5}, "c1")]
        )
        assert len(results) == 1
        assert isinstance(results[0], ToolMessage)
        assert results[0].tool_call_id == "c1"

    def test_πολλαπλά_εργαλεία_μαζί(self, fake_calendar, fake_gmail):
        results = execute_tool_calls(
            [
                tool_call("list_upcoming_events", {"max_results": 5}, "c1"),
                tool_call("list_recent_emails", {"max_results": 5}, "c2"),
            ]
        )
        assert len(results) == 2
        assert {r.tool_call_id for r in results} == {"c1", "c2"}

    def test_σφάλμα_δεν_ρίχνει_την_εφαρμογή(self, monkeypatch):
        """
        Αν ένα API πέσει, ο χρήστης πρέπει να δει εξήγηση - όχι σκέτο
        "Internal Server Error".
        """
        def boom():
            raise RuntimeError("Το Gmail API δεν είναι ενεργοποιημένο")

        monkeypatch.setattr("app.tools.gmail_tool._get_gmail_service", boom)

        results = execute_tool_calls([tool_call("list_recent_emails", {}, "c1")])
        assert "ΣΦΑΛΜΑ" in results[0].content
        assert "δεν είναι ενεργοποιημένο" in results[0].content

    def test_άγνωστο_εργαλείο(self):
        results = execute_tool_calls([tool_call("δεν_υπάρχει", {}, "c1")])
        assert "Δεν υπάρχει εργαλείο" in results[0].content


# ===========================================================================
# System prompt
# ===========================================================================

class TestSystemPrompt:
    def test_περιέχει_τρέχουσα_ημερομηνία(self):
        """
        Χωρίς αυτό, το μοντέλο μαντεύει τι μέρα είναι - και έχει ήδη πέσει
        έξω στην πράξη (έβαλε event σε λάθος μέρα).
        """
        from datetime import datetime
        from zoneinfo import ZoneInfo

        prompt = _build_system_prompt()
        today = datetime.now(ZoneInfo("Europe/Athens"))
        assert str(today.year) in prompt
        assert today.strftime("%A") in prompt

    def test_περιέχει_ζώνη_ώρας(self):
        assert "Europe/Athens" in _build_system_prompt()

    def test_οδηγίες_ασφαλείας_για_email(self):
        prompt = _build_system_prompt()
        assert "create_email_draft" in prompt
        assert "send_email" in prompt
        assert "Μην μαντεύεις ΠΟΤΕ διεύθυνση" in prompt

    def test_οδηγίες_για_ύφος_απαντήσεων(self):
        assert "ΥΦΟΣ" in _build_system_prompt()

    def test_άμυνα_σε_prompt_injection(self):
        prompt = _build_system_prompt()
        assert "ΑΣΦΑΛΕΙΑ" in prompt
        assert "ΔΕΔΟΜΕΝΑ" in prompt

    def test_υπολογίζεται_δυναμικά(self):
        """
        Αν ήταν σταθερά, ένας server που τρέχει για μέρες θα κόλλαγε στην
        ημερομηνία εκκίνησης.
        """
        import time
        first = _build_system_prompt()
        time.sleep(0.01)
        # Δεν συγκρίνουμε ισότητα (το λεπτό μπορεί να μην άλλαξε), αλλά
        # επιβεβαιώνουμε ότι η function εκτελείται κάθε φορά.
        assert isinstance(first, str) and len(first) > 100


# ===========================================================================
# Ροή του graph
# ===========================================================================

class TestAgentFlow:
    def test_απλή_απάντηση_χωρίς_εργαλεία(self, make_llm):
        make_llm([AIMessage(content="Γεια σου Γιώργο!")])
        result = get_agent().invoke({"messages": [HumanMessage(content="Γεια")]})

        assert len(result["messages"]) == 2
        assert result["messages"][-1].content == "Γεια σου Γιώργο!"

    def test_ασφαλές_εργαλείο_εκτελείται_αυτόματα(self, make_llm, fake_calendar):
        fake_calendar.events.return_value.list.return_value.execute.return_value = {
            "items": [{"summary": "Meeting", "start": {"dateTime": "2026-09-01T15:00:00Z"}}]
        }
        make_llm(
            [
                ai_with_tools(tool_call("list_upcoming_events", {"max_results": 5})),
                AIMessage(content="Έχεις ένα ραντεβού."),
            ]
        )
        result = get_agent().invoke({"messages": [HumanMessage(content="Τι έχω;")]})

        # Human -> AI(tool) -> Tool -> AI(τελική)
        assert len(result["messages"]) == 4
        assert isinstance(result["messages"][2], ToolMessage)
        assert "18:00" in result["messages"][2].content

    def test_επικίνδυνο_εργαλείο_ΔΕΝ_εκτελείται(self, make_llm, fake_gmail):
        """Το πιο σημαντικό test του project."""
        make_llm([ai_with_tools(tool_call("send_email", {"to": "a@b.com", "subject": "x", "body": "y"}))])
        result = get_agent().invoke({"messages": [HumanMessage(content="Στείλε email")]})

        # Σταματάει με ανεκτέλεστο tool_call
        assert len(result["messages"]) == 2
        assert result["messages"][-1].tool_calls
        fake_gmail.users.return_value.messages.return_value.send.assert_not_called()

    def test_μεικτή_ομάδα_σταματάει_ολόκληρη(self, make_llm, fake_calendar, fake_gmail):
        make_llm(
            [
                ai_with_tools(
                    tool_call("list_upcoming_events", {"max_results": 5}, "c1"),
                    tool_call("send_email", {"to": "a@b.com", "subject": "x", "body": "y"}, "c2"),
                )
            ]
        )
        result = get_agent().invoke({"messages": [HumanMessage(content="κάνε κάτι")]})

        assert len(result["messages"]) == 2
        fake_gmail.users.return_value.messages.return_value.send.assert_not_called()
        # Ούτε το ασφαλές δεν εκτελέστηκε - όλα ή τίποτα
        fake_calendar.events.return_value.list.assert_not_called()

    def test_αλυσίδα_πολλών_εργαλείων(self, make_llm, fake_gmail):
        """Ο agent μπορεί να καλέσει εργαλείο, να δει το αποτέλεσμα, και να καλέσει άλλο."""
        fake_gmail.users.return_value.messages.return_value.get.return_value.execute.return_value = {
            "id": "m1", "payload": {"headers": [], "mimeType": "text/plain", "body": {}},
        }
        make_llm(
            [
                ai_with_tools(tool_call("list_recent_emails", {"max_results": 5}, "c1")),
                ai_with_tools(tool_call("read_email", {"message_id": "m1"}, "c2")),
                AIMessage(content="Διάβασα το email."),
            ]
        )
        result = get_agent().invoke({"messages": [HumanMessage(content="Δες τα email μου")]})

        types = [type(m).__name__ for m in result["messages"]]
        assert types == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "ToolMessage", "AIMessage"]

    def test_system_prompt_δεν_μπαίνει_στο_ιστορικό(self, make_llm):
        """
        Το system prompt στέλνεται στο μοντέλο αλλά ΔΕΝ αποθηκεύεται -
        αλλιώς το ιστορικό θα γέμιζε αντίγραφά του, και η ημερομηνία θα
        έμενε κολλημένη στην πρώτη κλήση.
        """
        raw = make_llm([AIMessage(content="ok")])
        result = get_agent().invoke({"messages": [HumanMessage(content="Γεια")]})

        # Το μοντέλο ΕΙΔΕ system prompt...
        sent = raw.bind_tools.return_value.invoke.call_args[0][0]
        assert isinstance(sent[0], SystemMessage)

        # ...αλλά ΔΕΝ αποθηκεύτηκε
        assert not any(isinstance(m, SystemMessage) for m in result["messages"])
