"""
Tests για τα HTTP endpoints - το επίπεδο που "βλέπει" ο χρήστης.

Εδώ δοκιμάζονται οι ΠΛΗΡΕΙΣ διαδρομές: από το HTTP request, μέσα από τον
agent, μέχρι την απάντηση και την αποθήκευση στο Redis.
"""

from langchain_core.messages import AIMessage, SystemMessage

from tests.conftest import ai_with_tools, tool_call


SEND_EMAIL_CALL = tool_call(
    "send_email",
    {"to": "hr@accenture.com", "subject": "Ευχαριστώ", "body": "Κείμενο"},
    "c1",
)


# ===========================================================================
# Βασικά endpoints
# ===========================================================================

class TestBasicEndpoints:
    def test_root_σερβίρει_το_ui(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Προσωπικός Βοηθός" in response.text

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["checks"]["config_loaded"] is True
        assert data["checks"]["redis_connected"] is True

    def test_health_εντοπίζει_placeholder_κλειδί(self, client, monkeypatch):
        """
        Ένας συνηθισμένος τρόπος να "χαθεί" κάποιος: αντέγραψε το
        .env.example και ξέχασε να βάλει πραγματικό κλειδί.
        """
        from app.config import get_settings

        monkeypatch.setenv("GOOGLE_API_KEY", "your_gemini_api_key_here")
        get_settings.cache_clear()

        data = client.get("/health").json()
        assert data["checks"]["gemini_api_key_present"] is False

    def test_docs_διαθέσιμα(self, client):
        assert client.get("/docs").status_code == 200


# ===========================================================================
# Απλή συνομιλία
# ===========================================================================

class TestChat:
    def test_απλό_μήνυμα(self, client, make_llm):
        make_llm([AIMessage(content="Γεια σου Γιώργο!")])
        response = client.post("/chat", json={"message": "Γεια", "session_id": "s1"})

        assert response.status_code == 200
        assert response.json()["response"] == "Γεια σου Γιώργο!"

    def test_καθαρίζει_τα_content_blocks_του_gemini(self, client, make_llm):
        """
        Τα μοντέλα Gemini 3.x επιστρέφουν λίστα από blocks με τεχνικά
        metadata (signature). Ο χρήστης πρέπει να δει μόνο το κείμενο.
        """
        make_llm(
            [
                AIMessage(
                    content=[
                        {"type": "text", "text": "Καθαρή απάντηση",
                         "extras": {"signature": "μακρύ-τεχνικό-string"}}
                    ]
                )
            ]
        )
        data = client.post("/chat", json={"message": "x", "session_id": "s"}).json()

        assert data["response"] == "Καθαρή απάντηση"
        assert "signature" not in str(data)

    def test_θυμάται_μέσα_στο_ίδιο_session(self, client, make_llm, fake_redis):
        seen = []

        def capture(messages):
            seen.append(len(messages))
            return AIMessage(content="ok")

        raw = make_llm([])
        raw.bind_tools.return_value.invoke.side_effect = capture

        client.post("/chat", json={"message": "Με λένε Γιώργο", "session_id": "mem"})
        client.post("/chat", json={"message": "Πώς με λένε;", "session_id": "mem"})

        # 1η κλήση: system + human = 2
        # 2η κλήση: system + human + ai + human = 4
        assert seen[0] == 2
        assert seen[1] == 4

    def test_sessions_δεν_διαρρέουν(self, client, make_llm):
        seen = []

        def capture(messages):
            seen.append(len(messages))
            return AIMessage(content="ok")

        raw = make_llm([])
        raw.bind_tools.return_value.invoke.side_effect = capture

        client.post("/chat", json={"message": "α", "session_id": "one"})
        client.post("/chat", json={"message": "β", "session_id": "two"})

        assert seen[0] == seen[1] == 2

    def test_δεν_αποθηκεύει_system_prompt(self, client, make_llm, fake_redis):
        import json
        from langchain_core.messages import messages_from_dict

        make_llm([AIMessage(content="ok")])
        client.post("/chat", json={"message": "Γεια", "session_id": "sp"})

        stored = messages_from_dict(json.loads(fake_redis.get("conversation:sp")))
        assert not any(isinstance(m, SystemMessage) for m in stored)


# ===========================================================================
# Confirmation flow - οι τέσσερις διαδρομές
# ===========================================================================

class TestConfirmationFlow:
    def test_1_ζητά_έγκριση_χωρίς_να_εκτελέσει(self, client, make_llm, fake_gmail):
        make_llm([ai_with_tools(SEND_EMAIL_CALL)])
        data = client.post("/chat", json={"message": "Στείλε email", "session_id": "c1"}).json()

        assert "pending_confirmation" in data
        assert "hr@accenture.com" in data["pending_confirmation"]["actions"][0]
        fake_gmail.users.return_value.messages.return_value.send.assert_not_called()

    def test_2_έγκριση_εκτελεί_ακριβώς_μία_φορά(self, client, make_llm, fake_gmail):
        make_llm([ai_with_tools(SEND_EMAIL_CALL), AIMessage(content="Στάλθηκε.")])

        client.post("/chat", json={"message": "Στείλε email", "session_id": "c2"})
        data = client.post(
            "/chat", json={"message": "Ναι", "session_id": "c2", "confirm": True}
        ).json()

        assert "pending_confirmation" not in data
        fake_gmail.users.return_value.messages.return_value.send.assert_called_once()

    def test_3_άρνηση_δεν_εκτελεί(self, client, make_llm, fake_gmail):
        make_llm([ai_with_tools(SEND_EMAIL_CALL), AIMessage(content="Εντάξει, δεν το έστειλα.")])

        client.post("/chat", json={"message": "Στείλε email", "session_id": "c3"})
        data = client.post(
            "/chat", json={"message": "Όχι", "session_id": "c3", "confirm": False}
        ).json()

        assert "pending_confirmation" not in data
        fake_gmail.users.return_value.messages.return_value.send.assert_not_called()

    def test_4_αλλαγή_θέματος_ακυρώνει(self, client, make_llm, fake_gmail):
        make_llm([ai_with_tools(SEND_EMAIL_CALL), AIMessage(content="Δεν έστειλα τίποτα.")])

        client.post("/chat", json={"message": "Στείλε email", "session_id": "c4"})
        client.post("/chat", json={"message": "Άσε, τι έχω αύριο;", "session_id": "c4"})

        fake_gmail.users.return_value.messages.return_value.send.assert_not_called()

    def test_αποδοχή_με_κείμενο_αντί_κουμπιού(self, client, make_llm, fake_gmail):
        """Ο χρήστης μπορεί να γράψει "ναι" αντί να πατήσει το κουμπί."""
        make_llm([ai_with_tools(SEND_EMAIL_CALL), AIMessage(content="Στάλθηκε.")])

        client.post("/chat", json={"message": "Στείλε email", "session_id": "c5"})
        client.post("/chat", json={"message": "ναι", "session_id": "c5"})

        fake_gmail.users.return_value.messages.return_value.send.assert_called_once()

    def test_ιστορικό_μένει_έγκυρο_μετά_από_άρνηση(self, client, make_llm, fake_gmail, fake_redis):
        """
        Κάθε tool_call πρέπει να έχει αντίστοιχο αποτέλεσμα, αλλιώς η
        επόμενη κλήση στο Gemini σκάει με "ορφανό tool call".
        """
        import json
        from langchain_core.messages import messages_from_dict

        make_llm([ai_with_tools(SEND_EMAIL_CALL), AIMessage(content="ok")])

        client.post("/chat", json={"message": "Στείλε email", "session_id": "c6"})
        client.post("/chat", json={"message": "-", "session_id": "c6", "confirm": False})

        stored = messages_from_dict(json.loads(fake_redis.get("conversation:c6")))
        ai_calls = sum(
            len(m.tool_calls) for m in stored
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        )
        tool_results = sum(1 for m in stored if type(m).__name__ == "ToolMessage")
        assert tool_results >= ai_calls

    def test_ασφαλή_εργαλεία_δεν_διακόπτονται(self, client, make_llm, fake_calendar):
        make_llm(
            [
                ai_with_tools(tool_call("list_upcoming_events", {"max_results": 5})),
                AIMessage(content="Δεν έχεις τίποτα."),
            ]
        )
        data = client.post("/chat", json={"message": "Τι έχω;", "session_id": "safe"}).json()

        assert "pending_confirmation" not in data
        fake_calendar.events.return_value.list.assert_called_once()

    def test_email_με_προθεσμία_γίνεται_εργασία(
        self, client, make_llm, fake_gmail, fake_tasks
    ):
        """
        Η ροή που κάνει τον βοηθό χρήσιμο: διαβάζει email που ζητάει κάτι
        μέχρι κάποια ημερομηνία, και προτείνει εκκρεμότητα - με το
        συμφραζόμενο στις σημειώσεις.
        """
        from tests.conftest import b64

        fake_gmail.users.return_value.messages.return_value.get.return_value.execute.return_value = {
            "id": "m1",
            "payload": {
                "headers": [
                    {"name": "From", "value": "hr@accenture.com"},
                    {"name": "Subject", "value": "Documents needed"},
                ],
                "mimeType": "text/plain",
                "body": {"data": b64("Please send your documents by Monday 7 September.")},
            },
        }
        make_llm(
            [
                ai_with_tools(tool_call("read_email", {"message_id": "m1"}, "c1")),
                ai_with_tools(
                    tool_call(
                        "create_task",
                        {"title": "Αποστολή δικαιολογητικών στην Accenture",
                         "due_date": "2026-09-07",
                         "notes": "από hr@accenture.com"},
                        "c2",
                    )
                ),
                AIMessage(content="Πρόσθεσα την εκκρεμότητα."),
            ]
        )

        # Βήμα 1: διαβάζει το email (ασφαλές, εκτελείται)
        data = client.post(
            "/chat", json={"message": "Διάβασε το m1 και κανόνισέ το", "session_id": "flow"}
        ).json()

        # Βήμα 2: ζητάει έγκριση για την εργασία, ΔΕΝ τη δημιουργεί
        assert "pending_confirmation" in data
        action = data["pending_confirmation"]["actions"][0]
        assert "Αποστολή δικαιολογητικών" in action
        assert "2026-09-07" in action
        fake_tasks.tasks.return_value.insert.assert_not_called()

        # Βήμα 3: μετά την έγκριση, δημιουργείται
        client.post("/chat", json={"message": "Ναι", "session_id": "flow", "confirm": True})
        fake_tasks.tasks.return_value.insert.assert_called_once()


# ===========================================================================
# Πρωινή ενημέρωση
# ===========================================================================

class TestBriefing:
    def test_επιστρέφει_ενημέρωση(self, client, make_llm, fake_calendar, fake_gmail, fake_tasks):
        make_llm([AIMessage(content="Ήρεμη μέρα, ένα ραντεβού.")])
        response = client.get("/briefing")

        assert response.status_code == 200
        assert response.json()["briefing"] == "Ήρεμη μέρα, ένα ραντεβού."

    def test_δεύτερη_κλήση_χρησιμοποιεί_cache(
        self, client, make_llm, fake_calendar, fake_gmail, fake_tasks
    ):
        """Χωρίς cache, κάθε άνοιγμα του UI θα κατανάλωνε quota του Gemini."""
        raw = make_llm([AIMessage(content="Η μέρα σου.")])

        client.get("/briefing")
        client.get("/briefing")

        assert raw.invoke.call_count == 1

    def test_refresh_ξαναφτιάχνει(self, client, make_llm, fake_calendar, fake_gmail, fake_tasks):
        raw = make_llm([AIMessage(content="πρώτη"), AIMessage(content="δεύτερη")])

        client.get("/briefing")
        data = client.get("/briefing?refresh=true").json()

        assert raw.invoke.call_count == 2
        assert data["briefing"] == "δεύτερη"


# ===========================================================================
# Ανθεκτικότητα
# ===========================================================================

class TestResilience:
    def test_σφάλμα_εργαλείου_δίνει_κατανοητή_απάντηση(self, client, make_llm, monkeypatch):
        """Ο χρήστης πρέπει να δει εξήγηση, όχι HTTP 500."""
        def boom():
            raise RuntimeError("Το Gmail API δεν είναι ενεργοποιημένο")

        monkeypatch.setattr("app.tools.gmail_tool._get_gmail_service", boom)
        make_llm(
            [
                ai_with_tools(tool_call("list_recent_emails", {"max_results": 5})),
                AIMessage(content="Δεν μπόρεσα να δω τα email σου."),
            ]
        )
        response = client.post("/chat", json={"message": "Τι email έχω;", "session_id": "err"})

        assert response.status_code == 200
        assert "Δεν μπόρεσα" in response.json()["response"]

    def test_κενό_μήνυμα_δεν_σκάει(self, client, make_llm):
        make_llm([AIMessage(content="Πες μου τι χρειάζεσαι.")])
        assert client.post("/chat", json={"message": "", "session_id": "e"}).status_code == 200

    def test_λείπει_πεδίο_δίνει_422(self, client):
        """Το FastAPI πρέπει να απορρίπτει άκυρα requests πριν τον agent."""
        assert client.post("/chat", json={"session_id": "x"}).status_code == 422

    def test_default_session_id(self, client, make_llm):
        make_llm([AIMessage(content="ok")])
        data = client.post("/chat", json={"message": "Γεια"}).json()
        assert data["session_id"] == "default"
